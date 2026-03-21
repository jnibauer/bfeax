"""
MultipoleExpansion: high-level object that builds and evaluates Phi(x,y,z)
and the reconstructed density rho(x,y,z).

Inner cusp subtraction
----------------------
For density profiles with a steep inner slope (rho_lm ~ A r^alpha near r=0),
fitting splines directly to rho_lm can be poorly conditioned near r_min and
produces large residuals when extrapolating inward.  We instead:

  1. Estimate (alpha_lm, A_lm) from the first few radial grid points.
  2. Subtract the power-law background  bg(r) = A_lm * r^alpha_lm  before
     fitting the spline — the residual rho_lm(r) - bg(r) is much smoother.
  3. At evaluation time, add bg(r) back analytically so the full rho_lm is
     recovered exactly on the grid and extrapolated smoothly inside r_min.

Usage
-----
    exp = MultipoleExpansion.from_density(rho, r_min, r_max, n_r, l_max)
    phi        = exp(x, y, z)
    rho_rec    = exp.density(x, y, z)
    ax, ay, az = exp.acceleration(x, y, z)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .grid import make_radial_grid
from .density_coeffs import density_to_sph_coeffs
from .poisson import solve_poisson_lm
from .spline import (
    natural_cubic_spline_coeffs,
    natural_cubic_spline_coeffs_batch,
    spline_eval,
)
from .sph_harm import ylm_real, ylm_grid


# ---------------------------------------------------------------------------
# Inner power-law helper
# ---------------------------------------------------------------------------

def _fit_inner_power_law(
    r: Float[Array, "nr"],
    rho_vals: Float[Array, "nr"],
    global_scale: float,
) -> tuple:
    """
    Estimate the inner power-law background  bg(r) = A * r^alpha
    from the first 3 radial grid points.

    Parameters
    ----------
    r            : radial grid, shape (nr,)
    rho_vals     : rho_lm values on the grid, shape (nr,)
    global_scale : max |rho_lm| over ALL modes and ALL radii — used to
                   decide whether this mode is non-negligible.  Comparing
                   against the per-mode max would flag numerical-noise modes
                   (all values ~1e-15) as valid, producing huge A values.

    Returns (alpha, A, valid).
    Preserves the sign of rho_vals[0] so bg(r[0]) == rho_vals[0].
    """
    log_rho = jnp.log(jnp.abs(rho_vals[:3]) + 1e-300)
    log_r   = jnp.log(r[:3])
    alpha   = jnp.mean(jnp.diff(log_rho) / jnp.diff(log_r))

    sign0 = jnp.sign(rho_vals[0])
    # Use exp(alpha * log r) to avoid r**alpha overflow with dynamic alpha
    A     = sign0 * jnp.abs(rho_vals[0]) * jnp.exp(-alpha * jnp.log(r[0]))

    # Key: threshold against the *global* dominant amplitude, not per-mode.
    # Symmetry-zero modes (e.g. odd-l in an octant-symmetric density) have
    # all values at machine-epsilon level; their per-mode max is also ~eps,
    # so a relative threshold of 1e-8*per_mode_max would incorrectly pass them.
    valid = jnp.abs(rho_vals[0]) > 1e-6 * global_scale
    return alpha, A, valid


# ---------------------------------------------------------------------------
# Spheroid fast path — JIT-compiled core
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(10, 11, 12))
def _spheroid_core(r_grid, rho0, sph_alpha, beta, gamma, a, p, q, r_cut, xi,
                   l_max, n_theta, n_phi):
    """
    Full pipeline: spheroid params → spline coefficient arrays.

    Static args (control shapes / loop unrolling): l_max, n_theta, n_phi.
    All others are traced — changing spheroid params does NOT recompile.

    Returns (phi_coeffs, rho_res_coeffs, rho_alphas, rho_As) where
    coeffs are (a, b, c, d) tuples of shape (n_modes, n_r-1) arrays.
    """
    n_r   = r_grid.shape[0]
    log_r = jnp.log(r_grid)
    dr    = jnp.diff(r_grid)

    # ── Angular quadrature grid (computed at trace time) ─────────────────
    _nodes, _weights = np.polynomial.legendre.leggauss(n_theta)
    cos_th     = jnp.array(_nodes)
    gl_weights = jnp.array(_weights)
    theta_1d   = jnp.arccos(cos_th)
    phi_1d     = jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = jnp.meshgrid(theta_1d, phi_1d, indexing="ij")

    sin_theta = jnp.sin(theta_grid)
    cos_theta = jnp.cos(theta_grid)
    cos_phi   = jnp.cos(phi_grid)
    sin_phi   = jnp.sin(phi_grid)

    dphi = 2.0 * jnp.pi / n_phi
    w2d  = gl_weights[:, None] * dphi                    # (n_theta, n_phi)

    Ylm     = ylm_real(l_max, theta_grid, phi_grid)
    lm_keys = [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]
    Y_stack = jnp.stack([Ylm[k] for k in lm_keys])      # (n_modes, n_theta, n_phi)
    n_modes = len(lm_keys)

    # ── Spheroidal radius on the angular grid ────────────────────────────
    f_ij = jnp.sqrt(
        sin_theta**2 * cos_phi**2
        + sin_theta**2 * sin_phi**2 / p**2
        + cos_theta**2 / q**2
    )  # (n_theta, n_phi)

    # r̃ for every (radius, angle) combination
    r_tilde = r_grid[:, None, None] * f_ij[None, :, :]  # (n_r, n_theta, n_phi)

    # ── Density evaluation (vectorised, no vmap) ─────────────────────────
    log_s     = jnp.log(r_tilde / a)
    s_alpha   = jnp.exp(sph_alpha * log_s)               # s^alpha
    term1     = jnp.exp(-gamma * log_s)                   # s^(-gamma)
    term2     = jnp.exp(
        (gamma - beta) / sph_alpha * jnp.log1p(s_alpha)
    )  # (1 + s^alpha)^((gamma-beta)/alpha)
    rho_vals  = rho0 * term1 * term2

    # Outer cutoff  (guard: when xi==0, cutoff = 1)
    cutoff   = jnp.exp(-(r_tilde / r_cut) ** xi)
    rho_vals = rho_vals * jnp.where(xi == 0.0, 1.0, cutoff)

    # ── Angular projection via einsum ────────────────────────────────────
    rho_lm_all = jnp.einsum('rij,lij,ij->rl', rho_vals, Y_stack, w2d)
    # shape: (n_r, n_modes)

    # ── Poisson solve (batched over m for each l) ────────────────────────
    _4PI = 4.0 * jnp.pi
    phi_lm_all = jnp.zeros((n_r, n_modes))

    for l in range(l_max + 1):
        i0  = l * l
        i1  = (l + 1) ** 2
        n_m = 2 * l + 1
        rho_m = rho_lm_all[:, i0:i1]                    # (n_r, n_m)

        f_in  = rho_m * r_grid[:, None] ** (l + 2)
        f_out = rho_m * r_grid[:, None] ** (1 - l)

        # Inner boundary correction
        log_rho3 = jnp.log(jnp.abs(rho_m[:3, :]) + 1e-300)
        log_r3   = jnp.log(r_grid[:3])
        a_in = jnp.mean(
            jnp.diff(log_rho3, axis=0) / jnp.diff(log_r3)[:, None], axis=0
        )  # (n_m,)
        sign_in = jnp.sign(rho_m[0, :])
        A_in    = sign_in * jnp.abs(rho_m[0, :]) / (r_grid[0] ** a_in)

        rho_sc = jnp.max(jnp.abs(rho_m), axis=0) + 1e-300
        active_in = jnp.abs(rho_m[0, :]) > 1e-8 * rho_sc
        exp_in    = a_in + (l + 3)
        safe_in   = jnp.where(jnp.abs(exp_in) > 1e-6, exp_in, 1e-6)
        dI_in     = A_in * r_grid[0] ** (a_in + (l + 3)) / safe_in
        dI_in     = jnp.where(active_in & (exp_in > 0.0), dI_in, 0.0)

        trap_in = 0.5 * (f_in[:-1] + f_in[1:]) * dr[:, None]
        I_in    = jnp.concatenate(
            [jnp.zeros((1, n_m)), jnp.cumsum(trap_in, axis=0)], axis=0
        ) + dI_in[None, :]

        # Outer boundary correction
        log_rho3o = jnp.log(jnp.abs(rho_m[-3:, :]) + 1e-300)
        log_r3o   = jnp.log(r_grid[-3:])
        a_out = jnp.mean(
            jnp.diff(log_rho3o, axis=0) / jnp.diff(log_r3o)[:, None], axis=0
        )
        active_out = rho_m[-1, :] > 0.0
        tail_conv  = a_out < (l - 2)
        safe_out   = jnp.where(
            jnp.abs(l - a_out - 2) > 1e-6, l - a_out - 2, 1e-6
        )
        dI_out = rho_m[-1, :] * r_grid[-1] ** (2 - l) / safe_out
        dI_out = jnp.where(active_out & tail_conv, dI_out, 0.0)

        trap_out  = 0.5 * (f_out[:-1] + f_out[1:]) * dr[:, None]
        I_out_rev = jnp.cumsum(trap_out[::-1], axis=0)[::-1]
        I_out     = jnp.concatenate(
            [I_out_rev, jnp.zeros((1, n_m))], axis=0
        ) + dI_out[None, :]

        phi_m = -_4PI / (2 * l + 1) * (
            r_grid[:, None] ** (-(l + 1)) * I_in
            + r_grid[:, None] ** l * I_out
        )
        phi_lm_all = phi_lm_all.at[:, i0:i1].set(phi_m)

    # ── Cusp subtraction (vectorised over all modes) ─────────────────────
    global_scale = jnp.max(jnp.abs(rho_lm_all))

    v3      = rho_lm_all[:3, :]
    log_v3  = jnp.log(jnp.abs(v3) + 1e-300)
    log_r3c = jnp.log(r_grid[:3])
    rho_alphas = jnp.mean(
        jnp.diff(log_v3, axis=0) / jnp.diff(log_r3c)[:, None], axis=0
    )  # (n_modes,)

    signs  = jnp.sign(rho_lm_all[0, :])
    rho_As = signs * jnp.abs(rho_lm_all[0, :]) * jnp.exp(
        -rho_alphas * jnp.log(r_grid[0])
    )
    valid      = jnp.abs(rho_lm_all[0, :]) > 1e-6 * global_scale
    rho_As     = jnp.where(valid, rho_As, 0.0)
    rho_alphas = jnp.where(valid, rho_alphas, 0.0)

    bg           = rho_As[None, :] * jnp.exp(rho_alphas[None, :] * log_r[:, None])
    residual_all = rho_lm_all - bg                       # (n_r, n_modes)

    # ── Batch spline fitting ─────────────────────────────────────────────
    phi_coeffs     = natural_cubic_spline_coeffs_batch(log_r, phi_lm_all.T)
    rho_res_coeffs = natural_cubic_spline_coeffs_batch(log_r, residual_all.T)

    return phi_coeffs, rho_res_coeffs, rho_alphas, rho_As


# ---------------------------------------------------------------------------
# General fast path — JIT-compiled build from density values on a grid
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(4,))
def _build_expansion_from_grid(r_grid, rho_on_grid, Y_stack, w2d, l_max):
    """
    JIT-compiled pipeline: density values on the angular grid → spline coefficients.

    Parameters
    ----------
    r_grid      : (n_r,)                  — radial grid
    rho_on_grid : (n_r, n_theta, n_phi)   — density evaluated at every grid point
    Y_stack     : (n_modes, n_theta, n_phi) — Y_lm values on the angular grid
    w2d         : (n_theta, n_phi)         — quadrature weights
    l_max       : int (static)             — maximum harmonic degree

    This function never recompiles when the density values change — only when
    the grid dimensions or l_max change.
    """
    n_r     = r_grid.shape[0]
    n_modes = Y_stack.shape[0]
    log_r   = jnp.log(r_grid)
    dr      = jnp.diff(r_grid)

    # ── Angular projection ───────────────────────────────────────────────
    rho_lm_all = jnp.einsum('rij,lij,ij->rl', rho_on_grid, Y_stack, w2d)

    # ── Poisson solve (batched over m for each l) ────────────────────────
    _4PI = 4.0 * jnp.pi
    phi_lm_all = jnp.zeros((n_r, n_modes))

    for l in range(l_max + 1):
        i0  = l * l
        i1  = (l + 1) ** 2
        n_m = 2 * l + 1
        rho_m = rho_lm_all[:, i0:i1]

        f_in  = rho_m * r_grid[:, None] ** (l + 2)
        f_out = rho_m * r_grid[:, None] ** (1 - l)

        # Inner boundary correction
        log_rho3 = jnp.log(jnp.abs(rho_m[:3, :]) + 1e-300)
        log_r3   = jnp.log(r_grid[:3])
        a_in = jnp.mean(
            jnp.diff(log_rho3, axis=0) / jnp.diff(log_r3)[:, None], axis=0
        )
        sign_in = jnp.sign(rho_m[0, :])
        A_in    = sign_in * jnp.abs(rho_m[0, :]) / (r_grid[0] ** a_in)
        rho_sc    = jnp.max(jnp.abs(rho_m), axis=0) + 1e-300
        active_in = jnp.abs(rho_m[0, :]) > 1e-8 * rho_sc
        exp_in    = a_in + (l + 3)
        safe_in   = jnp.where(jnp.abs(exp_in) > 1e-6, exp_in, 1e-6)
        dI_in     = A_in * r_grid[0] ** (a_in + (l + 3)) / safe_in
        dI_in     = jnp.where(active_in & (exp_in > 0.0), dI_in, 0.0)

        trap_in = 0.5 * (f_in[:-1] + f_in[1:]) * dr[:, None]
        I_in    = jnp.concatenate(
            [jnp.zeros((1, n_m)), jnp.cumsum(trap_in, axis=0)], axis=0
        ) + dI_in[None, :]

        # Outer boundary correction
        log_rho3o = jnp.log(jnp.abs(rho_m[-3:, :]) + 1e-300)
        log_r3o   = jnp.log(r_grid[-3:])
        a_out = jnp.mean(
            jnp.diff(log_rho3o, axis=0) / jnp.diff(log_r3o)[:, None], axis=0
        )
        active_out = rho_m[-1, :] > 0.0
        tail_conv  = a_out < (l - 2)
        safe_out   = jnp.where(
            jnp.abs(l - a_out - 2) > 1e-6, l - a_out - 2, 1e-6
        )
        dI_out = rho_m[-1, :] * r_grid[-1] ** (2 - l) / safe_out
        dI_out = jnp.where(active_out & tail_conv, dI_out, 0.0)

        trap_out  = 0.5 * (f_out[:-1] + f_out[1:]) * dr[:, None]
        I_out_rev = jnp.cumsum(trap_out[::-1], axis=0)[::-1]
        I_out     = jnp.concatenate(
            [I_out_rev, jnp.zeros((1, n_m))], axis=0
        ) + dI_out[None, :]

        phi_m = -_4PI / (2 * l + 1) * (
            r_grid[:, None] ** (-(l + 1)) * I_in
            + r_grid[:, None] ** l * I_out
        )
        phi_lm_all = phi_lm_all.at[:, i0:i1].set(phi_m)

    # ── Cusp subtraction (vectorised) ────────────────────────────────────
    global_scale = jnp.max(jnp.abs(rho_lm_all))

    v3      = rho_lm_all[:3, :]
    log_v3  = jnp.log(jnp.abs(v3) + 1e-300)
    log_r3c = jnp.log(r_grid[:3])
    rho_alphas = jnp.mean(
        jnp.diff(log_v3, axis=0) / jnp.diff(log_r3c)[:, None], axis=0
    )
    signs  = jnp.sign(rho_lm_all[0, :])
    rho_As = signs * jnp.abs(rho_lm_all[0, :]) * jnp.exp(
        -rho_alphas * jnp.log(r_grid[0])
    )
    valid      = jnp.abs(rho_lm_all[0, :]) > 1e-6 * global_scale
    rho_As     = jnp.where(valid, rho_As, 0.0)
    rho_alphas = jnp.where(valid, rho_alphas, 0.0)

    bg           = rho_As[None, :] * jnp.exp(rho_alphas[None, :] * log_r[:, None])
    residual_all = rho_lm_all - bg

    # ── Batch spline fitting ─────────────────────────────────────────────
    phi_coeffs     = natural_cubic_spline_coeffs_batch(log_r, phi_lm_all.T)
    rho_res_coeffs = natural_cubic_spline_coeffs_batch(log_r, residual_all.T)

    return phi_coeffs, rho_res_coeffs, rho_alphas, rho_As


# ---------------------------------------------------------------------------
# ExpansionGrid — precomputed quadrature grid for fast repeated builds
# ---------------------------------------------------------------------------

class ExpansionGrid:
    """
    Precomputed angular quadrature grid for fast MultipoleExpansion builds.

    Separates density evaluation (user's code, may retrace) from the
    build pipeline (angular projection + Poisson + splines — JIT'd once,
    never recompiles when the density changes).

    Usage
    -----
        grid = ExpansionGrid(r_min=1e-2, r_max=300, n_r=128, l_max=8)

        # Option A: pass a density function (convenient)
        exp = grid(my_density)

        # Option B: pass precomputed density values (fastest)
        rho_vals = my_density(grid.x, grid.y, grid.z)   # vectorised
        exp = grid.from_values(rho_vals)
    """

    def __init__(
        self,
        r_min: float,
        r_max: float,
        n_r: int,
        l_max: int,
        n_theta: int | None = None,
        n_phi: int | None = None,
    ):
        self.l_max = l_max
        if n_theta is None:
            n_theta = 3 * (l_max + 2)
        if n_phi is None:
            n_phi = 4 * l_max + 7

        self._r_grid = make_radial_grid(n_r, r_min, r_max)

        # Angular grid
        theta_grid, phi_grid, Ylm, gl_weights = ylm_grid(l_max, n_theta, n_phi)
        lm_keys = [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]
        self._Y_stack = jnp.stack([Ylm[k] for k in lm_keys])

        dphi = 2.0 * math.pi / n_phi
        self._w2d = gl_weights[:, None] * dphi

        # Cartesian coordinates of every grid point (for user convenience)
        sin_theta = jnp.sin(theta_grid)
        ux = sin_theta * jnp.cos(phi_grid)
        uy = sin_theta * jnp.sin(phi_grid)
        uz = jnp.cos(theta_grid)

        r = self._r_grid
        self.x = r[:, None, None] * ux[None, :, :]   # (n_r, n_theta, n_phi)
        self.y = r[:, None, None] * uy[None, :, :]
        self.z = r[:, None, None] * uz[None, :, :]

    def __call__(
        self,
        rho,
        sigma_taper: bool = False,
    ) -> "MultipoleExpansion":
        """Build a MultipoleExpansion from a density function rho(x, y, z)."""
        rho_on_grid = jax.vmap(jax.vmap(jax.vmap(rho)))(self.x, self.y, self.z)
        return self.from_values(rho_on_grid, sigma_taper=sigma_taper)

    def from_values(
        self,
        rho_on_grid: Float[Array, "n_r n_theta n_phi"],
        sigma_taper: bool = False,
    ) -> "MultipoleExpansion":
        """
        Build from precomputed density values on the grid.

        This is the fastest path: no density tracing, no JIT recompilation.
        Shape must be (n_r, n_theta, n_phi) matching this grid.
        """
        phi_coeffs, rho_res_coeffs, rho_alphas, rho_As = \
            _build_expansion_from_grid(
                self._r_grid, rho_on_grid,
                self._Y_stack, self._w2d, self.l_max,
            )
        log_r   = jnp.log(self._r_grid)
        stacked = (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)
        return MultipoleExpansion(
            self.l_max, {}, {}, sigma_taper=sigma_taper, _stacked=stacked,
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultipoleExpansion:
    """
    Gravitational potential and reconstructed density via multipole expansion.

    Stores spline coefficients for both Phi_lm(r) and rho_lm(r), where the
    rho splines represent the *residual* after subtracting the inner power-law
    background (which is stored separately and added back at evaluation).

    Parameters
    ----------
    l_max       : int
    phi_splines : dict (l,m) -> (log_r_knots, spline_coeffs)
    rho_splines : dict (l,m) -> (alpha, A, log_r_knots, residual_coeffs)
    """

    def __init__(
        self,
        l_max: int,
        phi_splines: dict[tuple[int, int], tuple],
        rho_splines: dict[tuple[int, int], tuple],
        sigma_taper: bool = False,
        *,
        _stacked: tuple | None = None,
    ):
        self.l_max = l_max
        self._phi_splines = phi_splines
        self._rho_splines = rho_splines
        self.sigma_taper = sigma_taper

        # Optional stacked representation for fast-path eval (from_spheroid).
        # Format: (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)
        # where phi_coeffs/rho_res_coeffs are (a,b,c,d) tuples of (n_modes, n_r-1).
        self._stacked = _stacked

        # Precompute Lanczos sigma factors  σ_l = sinc(l / (L+1))
        L = l_max
        ls = jnp.arange(L + 1, dtype=float)
        x  = jnp.pi * ls / (L + 1)
        self._sigma_arr = jnp.where(ls == 0, 1.0, jnp.sin(x) / x)  # shape (L+1,)

        # Per-mode sigma weight vector for stacked path
        if _stacked is not None:
            sigma_per_mode = []
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    sigma_per_mode.append(float(self._sigma_arr[l]))
            self._sigma_vec = jnp.array(sigma_per_mode)  # (n_modes,)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_density(
        cls,
        rho: Callable,
        r_min: float,
        r_max: float,
        n_r: int,
        l_max: int,
        n_theta: int | None = None,
        n_phi: int | None = None,
        sigma_taper: bool = False,
    ) -> "MultipoleExpansion":
        """
        Build a MultipoleExpansion from a density function rho(x,y,z).

        Parameters
        ----------
        rho     : callable rho(x,y,z) -> scalar
        r_min   : inner radius of the radial grid
        r_max   : outer radius of the radial grid
        n_r     : number of radial grid points
        l_max   : maximum spherical harmonic degree
        n_theta      : GL quadrature nodes  (default: 3*(l_max+2))
        n_phi        : uniform phi points   (default: 4*l_max+7)
        sigma_taper  : if True, apply Lanczos sigma factors to the density
                       reconstruction to suppress Gibbs ringing.  Has no
                       effect on the potential evaluation.
        """
        r_grid = make_radial_grid(n_r, r_min, r_max)
        log_r  = jnp.log(r_grid)

        # Angular projection -> rho_lm(r) at grid points
        rho_lm = density_to_sph_coeffs(rho, r_grid, l_max, n_theta, n_phi)

        # ── rho splines with inner cusp subtraction ──────────────────
        # Global scale: max |rho_lm| over all modes and all radii.
        # Used to identify negligible modes (symmetry-zeros, numerical noise).
        global_scale = float(max(
            float(jnp.max(jnp.abs(v))) for v in rho_lm.values()
        ))

        rho_splines: dict[tuple[int, int], tuple] = {}
        for key, vals in rho_lm.items():
            alpha, A, valid = _fit_inner_power_law(r_grid, vals, global_scale)

            # Zero background for negligible/noise modes
            A_store     = jnp.where(valid, A,     jnp.zeros(()))
            alpha_store = jnp.where(valid, alpha, jnp.zeros(()))

            bg       = A_store * jnp.exp(alpha_store * log_r)
            residual = vals - bg
            coeffs   = natural_cubic_spline_coeffs(log_r, residual)
            rho_splines[key] = (alpha_store, A_store, log_r, coeffs)

        # ── Poisson solve -> Phi_lm splines ──────────────────────────
        phi_splines = solve_poisson_lm(r_grid, rho_lm)

        return cls(l_max, phi_splines, rho_splines, sigma_taper=sigma_taper)

    @classmethod
    def from_spheroid(
        cls,
        rho0: float,
        alpha: float,
        beta: float,
        gamma: float,
        a: float,
        *,
        p: float = 1.0,
        q: float = 1.0,
        r_cut: float | None = None,
        xi: float = 0.0,
        r_min: float,
        r_max: float,
        n_r: int = 128,
        l_max: int = 8,
        n_theta: int | None = None,
        n_phi: int | None = None,
        sigma_taper: bool = False,
    ) -> "MultipoleExpansion":
        """
        Build a MultipoleExpansion from Agama-style spheroid parameters.

        Much faster than from_density(SpheroidDensity(...), ...) because:
          - density is evaluated via vectorised array ops (no vmap)
          - spline fitting is batched (one matrix factorisation for all modes)
          - the JIT-compiled core only recompiles when grid shapes change,
            NOT when spheroid parameters change
        """
        if n_theta is None:
            n_theta = 3 * (l_max + 2)
        if n_phi is None:
            n_phi = 4 * l_max + 7

        r_cut_eff = float(r_cut) if r_cut is not None else 1e30
        xi_eff    = float(xi)    if r_cut is not None else 0.0

        r_grid = make_radial_grid(n_r, r_min, r_max)
        log_r  = jnp.log(r_grid)

        phi_coeffs, rho_res_coeffs, rho_alphas, rho_As = _spheroid_core(
            r_grid,
            float(rho0), float(alpha), float(beta), float(gamma), float(a),
            float(p), float(q), r_cut_eff, xi_eff,
            l_max, n_theta, n_phi,
        )

        # Store stacked arrays directly — skip dict unpacking
        stacked = (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)

        return cls(l_max, {}, {}, sigma_taper=sigma_taper, _stacked=stacked)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_stacked(
        self,
        log_r_query: Float[Array, "..."],
        theta: Float[Array, "..."],
        phi: Float[Array, "..."],
        kind: str = "phi",
    ) -> Float[Array, "..."]:
        """Vectorised eval using stacked arrays (from_spheroid fast path)."""
        log_r_knots, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As = self._stacked

        # Ensure at least 1-D for indexing, then squeeze at the end
        query_shape = log_r_query.shape
        log_r_q = jnp.atleast_1d(log_r_query)

        # Batched spline lookup: find interval, compute polynomial
        idx = jnp.searchsorted(log_r_knots, log_r_q, side="right") - 1
        idx = jnp.clip(idx, 0, log_r_knots.shape[0] - 2)
        dt  = log_r_q - log_r_knots[idx]

        if kind == "phi":
            a, b, c, d = phi_coeffs
        else:
            a, b, c, d = rho_res_coeffs

        # a shape: (n_modes, n_r-1), idx shape: (N,)
        vals = a[:, idx] + b[:, idx] * dt + c[:, idx] * dt**2 + d[:, idx] * dt**3
        # vals shape: (n_modes, N)

        if kind == "rho":
            # Add back power-law background: A * exp(alpha * log_r)
            bg = rho_As[:, None] * jnp.exp(rho_alphas[:, None] * log_r_q[None, :])
            vals = vals + bg
            if self.sigma_taper:
                vals = vals * self._sigma_vec[:, None]

        # Y_lm at query points
        theta_1d = jnp.atleast_1d(theta)
        phi_1d   = jnp.atleast_1d(phi)
        Ylm      = ylm_real(self.l_max, theta_1d, phi_1d)
        lm_keys  = [(l, m) for l in range(self.l_max + 1) for m in range(-l, l + 1)]
        Y_arr    = jnp.stack([Ylm[k] for k in lm_keys])  # (n_modes, N)

        result = jnp.sum(vals * Y_arr, axis=0)
        return result.reshape(query_shape)

    def _eval_phi_lm(
        self,
        log_r_query: Float[Array, "..."],
    ) -> dict[tuple[int, int], Float[Array, "..."]]:
        out = {}
        for key, (log_r_knots, coeffs) in self._phi_splines.items():
            out[key] = spline_eval(log_r_knots, coeffs, log_r_query)
        return out

    def _eval_rho_lm(
        self,
        r_query: Float[Array, "..."],
        log_r_query: Float[Array, "..."],
    ) -> dict[tuple[int, int], Float[Array, "..."]]:
        """Evaluate rho_lm = residual_spline + A * r^alpha."""
        out = {}
        for key, (alpha, A, log_r_knots, coeffs) in self._rho_splines.items():
            residual = spline_eval(log_r_knots, coeffs, log_r_query)
            bg       = A * jnp.exp(alpha * log_r_query)  # A * r^alpha via log
            out[key] = residual + bg
        return out

    def _sum_over_lm(
        self,
        lm_vals: dict[tuple[int, int], Float[Array, "..."]],
        theta: Float[Array, "..."],
        phi: Float[Array, "..."],
        shape,
        apply_sigma: bool = False,
    ) -> Float[Array, "..."]:
        Ylm    = ylm_real(self.l_max, theta, phi)
        result = jnp.zeros(shape)
        for (l, m), val in lm_vals.items():
            w = self._sigma_arr[l] if apply_sigma else 1.0
            result = result + w * val * Ylm[(l, m)]
        return result

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: Float[Array, "..."],
        y: Float[Array, "..."],
        z: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Evaluate Phi(x, y, z).  Sigma tapering is never applied here."""
        r, theta, phi = _cartesian_to_spherical(x, y, z)
        log_r = jnp.log(jnp.clip(r, 1e-30))
        if self._stacked is not None:
            return self._eval_stacked(log_r, theta, phi, kind="phi")
        phi_lm = self._eval_phi_lm(log_r)
        return self._sum_over_lm(phi_lm, theta, phi, r.shape)

    def density(
        self,
        x: Float[Array, "..."],
        y: Float[Array, "..."],
        z: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """
        Evaluate the reconstructed density rho_rec(x, y, z).

        rho_rec = sum_{l,m} [sigma_l] * rho_lm(r) * Y_lm(theta, phi)

        The optional Lanczos sigma factor sigma_l = sin(pi*l/(L+1))/(pi*l/(L+1))
        is applied when sigma_taper=True was passed to from_density().
        It suppresses Gibbs ringing at the cost of mild smoothing near the
        sharpest angular gradients.
        """
        r, theta, phi = _cartesian_to_spherical(x, y, z)
        log_r = jnp.log(jnp.clip(r, 1e-30))
        if self._stacked is not None:
            return self._eval_stacked(log_r, theta, phi, kind="rho")
        rho_lm = self._eval_rho_lm(r, log_r)
        return self._sum_over_lm(
            rho_lm, theta, phi, r.shape,
            apply_sigma=self.sigma_taper,
        )

    def acceleration(
        self,
        x: Float[Array, "..."],
        y: Float[Array, "..."],
        z: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Return gravitational acceleration (ax, ay, az) = -grad Phi via autodiff.

        Uses jax.vmap(jax.grad(...)) so each point gets its own scalar gradient,
        avoiding the batch-indexing bug that arises from stacking into (N,3) and
        indexing with xyz[0], xyz[1], xyz[2] (which selects rows, not columns).
        """
        x, y, z = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)
        shape = x.shape

        grad_fn = jax.vmap(jax.grad(self.__call__, argnums=(0, 1, 2)))
        ax, ay, az = grad_fn(x.ravel(), y.ravel(), z.ravel())
        return -ax.reshape(shape), -ay.reshape(shape), -az.reshape(shape)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def rho_lm_amplitudes(self, r_val: float) -> dict[tuple[int, int], float]:
        """Return |rho_lm(r)| at a single radius."""
        r_arr   = jnp.array([r_val])
        log_arr = jnp.log(r_arr)
        lm_vals = self._eval_rho_lm(r_arr, log_arr)
        return {k: float(jnp.abs(v[0])) for k, v in lm_vals.items()}

    def inner_slopes(self) -> dict[tuple[int, int], tuple[float, float]]:
        """Return (alpha, A) estimated at r_min for each mode — useful for diagnostics."""
        return {
            k: (float(alpha), float(A))
            for k, (alpha, A, _, _) in self._rho_splines.items()
        }


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _cartesian_to_spherical(x, y, z):
    r     = jnp.sqrt(x**2 + y**2 + z**2)
    r_safe = jnp.where(r == 0.0, 1e-30, r)
    theta = jnp.arccos(jnp.clip(z / r_safe, -1.0, 1.0))
    phi   = jnp.arctan2(y, x) % (2 * math.pi)
    return r, theta, phi
