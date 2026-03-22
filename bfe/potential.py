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
from .spline import (
    natural_cubic_spline_coeffs_batch,
    spline_eval,
)
from .sph_harm import ylm_real, ylm_grid, _alp_recurrence, _normalization


# ---------------------------------------------------------------------------
# Symmetry-aware mode selection
# ---------------------------------------------------------------------------

def _lm_keys(l_max: int, symmetry: str | None = None) -> list[tuple[int, int]]:
    """
    Return the list of (l, m) modes for a given symmetry.

    Parameters
    ----------
    l_max    : maximum harmonic degree
    symmetry : one of "spherical", "axisymmetric", "triaxial", or None.
               None means all modes (general case).

    Returns
    -------
    List of (l, m) tuples in canonical order.
    """
    if symmetry is None:
        return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]
    elif symmetry == "spherical":
        return [(0, 0)]
    elif symmetry == "axisymmetric":
        return [(l, 0) for l in range(l_max + 1)]
    elif symmetry == "triaxial":
        # Octant symmetry: even l, even m >= 0 (cosine terms only)
        # x→-x kills odd-m cosine terms; y→-y kills all sine terms (m<0);
        # z→-z requires l+m even; combined with m even → l even.
        return [(l, m) for l in range(0, l_max + 1, 2)
                for m in range(0, l + 1, 2)]
    else:
        raise ValueError(
            f"Unknown symmetry {symmetry!r}. "
            f"Use 'spherical', 'axisymmetric', 'triaxial', or None."
        )


# ---------------------------------------------------------------------------
# Poisson solve — scan over all (l, m) modes
# ---------------------------------------------------------------------------

def _poisson_scan(
    rho_lm_all: "Float[Array, 'nr n_modes']",
    r_grid:     "Float[Array, 'nr']",
    l_per_mode: "Float[Array, 'n_modes']",
) -> "Float[Array, 'nr n_modes']":
    """
    Solve the radial Poisson equation for every (l,m) mode via jax.lax.scan.

    Replaces the Python `for l in range(l_max+1)` loop that previously
    unrolled l_max+1 separate XLA subgraphs.  Scan compiles a single body
    and steps through all n_modes = (l_max+1)^2 modes sequentially, which
    substantially reduces XLA compilation time for large l_max.

    Parameters
    ----------
    rho_lm_all : (n_r, n_modes)  — angular-projected density coefficients
    r_grid     : (n_r,)          — radial grid
    l_per_mode : (n_modes,)      — l value (as float) for each mode, in the
                                   same ordering as rho_lm_all columns.
    Returns
    -------
    phi_lm_all : (n_r, n_modes)
    """
    log_r = jnp.log(r_grid)
    dr    = jnp.diff(r_grid)
    _4PI  = 4.0 * jnp.pi

    def _solve_one(_, xs):
        rho_col, l_f = xs   # (n_r,), scalar float

        f_in  = rho_col * jnp.exp((l_f + 2.0) * log_r)
        f_out = rho_col * jnp.exp((1.0 - l_f)  * log_r)

        # Inner power-law correction (0 → r_min)
        log_rho3 = jnp.log(jnp.abs(rho_col[:3]) + 1e-300)
        alpha_in = jnp.mean(jnp.diff(log_rho3) / jnp.diff(log_r[:3]))
        A_in     = jnp.sign(rho_col[0]) * jnp.abs(rho_col[0]) * jnp.exp(-alpha_in * log_r[0])
        rho_sc   = jnp.max(jnp.abs(rho_col)) + 1e-300
        active_in = jnp.abs(rho_col[0]) > 1e-8 * rho_sc
        exp_in   = alpha_in + l_f + 3.0
        safe_in  = jnp.where(jnp.abs(exp_in) > 1e-6, exp_in, 1e-6)
        dI_in    = A_in * jnp.exp((alpha_in + l_f + 3.0) * log_r[0]) / safe_in
        dI_in    = jnp.where(active_in & (exp_in > 0.0), dI_in, 0.0)

        trap_in = 0.5 * (f_in[:-1] + f_in[1:]) * dr
        I_in    = jnp.concatenate([jnp.zeros(1), jnp.cumsum(trap_in)]) + dI_in

        # Outer power-law correction (r_max → ∞)
        log_rho3o = jnp.log(jnp.abs(rho_col[-3:]) + 1e-300)
        alpha_out = jnp.mean(jnp.diff(log_rho3o) / jnp.diff(log_r[-3:]))
        active_out = rho_col[-1] > 0.0
        tail_conv  = alpha_out < (l_f - 2.0)
        safe_out   = jnp.where(jnp.abs(l_f - alpha_out - 2.0) > 1e-6,
                               l_f - alpha_out - 2.0, 1e-6)
        dI_out     = rho_col[-1] * jnp.exp((2.0 - l_f) * log_r[-1]) / safe_out
        dI_out     = jnp.where(active_out & tail_conv, dI_out, 0.0)

        trap_out = 0.5 * (f_out[:-1] + f_out[1:]) * dr
        I_out    = jnp.concatenate(
            [jnp.cumsum(trap_out[::-1])[::-1], jnp.zeros(1)]
        ) + dI_out

        phi_col = -_4PI / (2.0 * l_f + 1.0) * (
            jnp.exp(-(l_f + 1.0) * log_r) * I_in
            + jnp.exp(l_f * log_r) * I_out
        )
        return None, phi_col

    _, phi_T = jax.lax.scan(_solve_one, None, (rho_lm_all.T, l_per_mode))
    return phi_T.T   # (n_r, n_modes)




# ---------------------------------------------------------------------------
# Spheroid fast path — JIT-compiled core
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(10, 11, 12, 13))
def _spheroid_core(r_grid, rho0, sph_alpha, beta, gamma, a, p, q, r_cut, xi,
                   l_max, n_theta, n_phi, lm_keys_tuple):
    """
    Full pipeline: spheroid params → spline coefficient arrays.

    Static args (control shapes / loop unrolling): l_max, n_theta, n_phi, lm_keys_tuple.
    All others are traced — changing spheroid params does NOT recompile.

    Returns (phi_coeffs, rho_res_coeffs, rho_alphas, rho_As) where
    coeffs are (a, b, c, d) tuples of shape (n_modes, n_r-1) arrays.
    """
    lm_keys = list(lm_keys_tuple)
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

    # ── Poisson solve (scan over all modes) ──────────────────────────────
    l_per_mode = jnp.array([float(l) for l, m in lm_keys])
    phi_lm_all = _poisson_scan(rho_lm_all, r_grid, l_per_mode)

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

@partial(jax.jit, static_argnums=(4, 5))
def _build_expansion_from_grid(r_grid, rho_on_grid, Y_stack, w2d, l_max,
                               lm_keys_tuple):
    """
    JIT-compiled pipeline: density values on the angular grid → spline coefficients.

    Parameters
    ----------
    r_grid      : (n_r,)                  — radial grid
    rho_on_grid : (n_r, n_theta, n_phi)   — density evaluated at every grid point
    Y_stack     : (n_modes, n_theta, n_phi) — Y_lm values on the angular grid
    w2d         : (n_theta, n_phi)         — quadrature weights
    l_max       : int (static)             — maximum harmonic degree
    lm_keys_tuple : tuple of (l,m) pairs (static) — modes to compute

    This function never recompiles when the density values change — only when
    the grid dimensions or l_max change.
    """
    lm_keys = list(lm_keys_tuple)
    n_r     = r_grid.shape[0]
    n_modes = Y_stack.shape[0]
    log_r   = jnp.log(r_grid)
    dr      = jnp.diff(r_grid)

    # ── Angular projection ───────────────────────────────────────────────
    rho_lm_all = jnp.einsum('rij,lij,ij->rl', rho_on_grid, Y_stack, w2d)

    # ── Poisson solve (scan over all modes) ──────────────────────────────
    l_per_mode = jnp.array([float(l) for l, m in lm_keys])
    phi_lm_all = _poisson_scan(rho_lm_all, r_grid, l_per_mode)

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
        symmetry: str | None = None,
    ):
        self.l_max = l_max
        self.symmetry = symmetry
        if n_theta is None:
            n_theta = 3 * (l_max + 2)
        if n_phi is None:
            n_phi = 4 * l_max + 7

        self._r_grid = make_radial_grid(n_r, r_min, r_max)
        self._lm_keys_list = _lm_keys(l_max, symmetry)

        # Angular grid
        theta_grid, phi_grid, Ylm, gl_weights = ylm_grid(l_max, n_theta, n_phi)
        self._Y_stack = jnp.stack([Ylm[k] for k in self._lm_keys_list])

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
    ) -> "MultipoleExpansion":
        """Build a MultipoleExpansion from a density function rho(x, y, z)."""
        rho_on_grid = jax.vmap(jax.vmap(jax.vmap(rho)))(self.x, self.y, self.z)
        return self.from_values(rho_on_grid)

    def from_values(
        self,
        rho_on_grid: Float[Array, "n_r n_theta n_phi"],
    ) -> "MultipoleExpansion":
        """
        Build from precomputed density values on the grid.

        This is the fastest path: no density tracing, no JIT recompilation.
        Shape must be (n_r, n_theta, n_phi) matching this grid.
        """
        lm_tuple = tuple(self._lm_keys_list)
        phi_coeffs, rho_res_coeffs, rho_alphas, rho_As = \
            _build_expansion_from_grid(
                self._r_grid, rho_on_grid,
                self._Y_stack, self._w2d, self.l_max, lm_tuple,
            )
        log_r   = jnp.log(self._r_grid)
        stacked = (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)
        return MultipoleExpansion(
            self.l_max, {}, {}, _stacked=stacked, symmetry=self.symmetry,
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
        *,
        _stacked: tuple | None = None,
        symmetry: str | None = None,
    ):
        self.l_max = l_max
        self.symmetry = symmetry
        self._phi_splines = phi_splines
        self._rho_splines = rho_splines
        self._lm_keys = _lm_keys(l_max, symmetry)

        # Optional stacked representation for fast-path eval (from_spheroid).
        # Format: (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)
        # where phi_coeffs/rho_res_coeffs are (a,b,c,d) tuples of (n_modes, n_r-1).
        self._stacked = _stacked

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
        symmetry: str | None = None,
    ) -> "MultipoleExpansion":
        """
        Build a MultipoleExpansion from a density function rho(x,y,z).

        Parameters
        ----------
        rho      : callable rho(x,y,z) -> scalar
        r_min    : inner radius of the radial grid
        r_max    : outer radius of the radial grid
        n_r      : number of radial grid points
        l_max    : maximum spherical harmonic degree
        n_theta  : GL quadrature nodes  (default: 3*(l_max+2))
        n_phi    : uniform phi points   (default: 4*l_max+7)
        symmetry : "spherical", "axisymmetric", "triaxial", or None
        """
        if n_theta is None:
            n_theta = 3 * (l_max + 2)
        if n_phi is None:
            n_phi = 4 * l_max + 7

        r_grid = make_radial_grid(n_r, r_min, r_max)
        log_r  = jnp.log(r_grid)
        lm     = _lm_keys(l_max, symmetry)

        # Build angular quadrature grid
        theta_grid, phi_grid, Ylm, gl_weights = ylm_grid(l_max, n_theta, n_phi)
        Y_stack = jnp.stack([Ylm[k] for k in lm])
        w2d     = gl_weights[:, None] * (2.0 * math.pi / n_phi)

        # Cartesian coordinates of every grid point: (n_r, n_theta, n_phi)
        sin_theta = jnp.sin(theta_grid)
        ux = sin_theta * jnp.cos(phi_grid)
        uy = sin_theta * jnp.sin(phi_grid)
        uz = jnp.cos(theta_grid)
        x  = r_grid[:, None, None] * ux[None, :, :]
        y  = r_grid[:, None, None] * uy[None, :, :]
        z  = r_grid[:, None, None] * uz[None, :, :]

        # Evaluate density on the full grid, then run the batched pipeline
        rho_on_grid = jax.vmap(jax.vmap(jax.vmap(rho)))(x, y, z)
        phi_coeffs, rho_res_coeffs, rho_alphas, rho_As = _build_expansion_from_grid(
            r_grid, rho_on_grid, Y_stack, w2d, l_max, tuple(lm),
        )

        stacked = (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)
        return cls(l_max, {}, {}, _stacked=stacked, symmetry=symmetry)

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
        symmetry: str | None = None,
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

        # Auto-detect symmetry from axis ratios when not specified
        if symmetry is None:
            if p == 1.0 and q == 1.0:
                symmetry = "spherical"
            elif p == 1.0 or q == 1.0:
                symmetry = "axisymmetric"
            else:
                symmetry = "triaxial"

        r_cut_eff = float(r_cut) if r_cut is not None else 1e30
        xi_eff    = float(xi)    if r_cut is not None else 0.0

        r_grid = make_radial_grid(n_r, r_min, r_max)
        log_r  = jnp.log(r_grid)

        lm = _lm_keys(l_max, symmetry)
        lm_tuple = tuple(lm)

        phi_coeffs, rho_res_coeffs, rho_alphas, rho_As = _spheroid_core(
            r_grid,
            float(rho0), float(alpha), float(beta), float(gamma), float(a),
            float(p), float(q), r_cut_eff, xi_eff,
            l_max, n_theta, n_phi, lm_tuple,
        )

        # Store stacked arrays directly — skip dict unpacking
        stacked = (log_r, phi_coeffs, rho_res_coeffs, rho_alphas, rho_As)

        return cls(l_max, {}, {}, _stacked=stacked, symmetry=symmetry)

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

        # Y_lm at query points
        theta_1d = jnp.atleast_1d(theta)
        phi_1d   = jnp.atleast_1d(phi)
        Ylm      = ylm_real(self.l_max, theta_1d, phi_1d)
        Y_arr    = jnp.stack([Ylm[k] for k in self._lm_keys])  # (n_modes, N)

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
    ) -> Float[Array, "..."]:
        Ylm     = ylm_real(self.l_max, theta, phi)
        keys    = list(lm_vals.keys())
        val_arr = jnp.stack([lm_vals[k] for k in keys])  # (n_modes, *shape)
        Y_arr   = jnp.stack([Ylm[k]     for k in keys])  # (n_modes, *shape)
        return jnp.sum(val_arr * Y_arr, axis=0)

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: Float[Array, "..."],
        y: Float[Array, "..."],
        z: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Evaluate Phi(x, y, z)."""
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
        """Evaluate the reconstructed density rho_rec(x, y, z)."""
        r, theta, phi = _cartesian_to_spherical(x, y, z)
        log_r = jnp.log(jnp.clip(r, 1e-30))
        if self._stacked is not None:
            return self._eval_stacked(log_r, theta, phi, kind="rho")
        rho_lm = self._eval_rho_lm(r, log_r)
        return self._sum_over_lm(rho_lm, theta, phi, r.shape)

    def force(
        self,
        x: Float[Array, "..."],
        y: Float[Array, "..."],
        z: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """
        Compute gravitational force (Fx, Fy, Fz) = -∇Φ analytically.

        Much faster than acceleration() because it avoids reverse-mode
        autodiff — forces are computed directly from spline derivatives
        and analytical spherical-harmonic derivatives.  Modes with
        negligible coefficients are pruned automatically.
        """
        x, y, z = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)
        shape = x.shape
        if self._stacked is not None:
            if not hasattr(self, '_force_jit'):
                self._force_jit = self._build_force_fn()
            Fx, Fy, Fz = self._force_jit(x.ravel(), y.ravel(), z.ravel())
            return Fx.reshape(shape), Fy.reshape(shape), Fz.reshape(shape)
        # Fallback to autodiff for dict-based representation
        return self.acceleration(x, y, z)

    def _build_force_fn(self):
        """
        Build a JIT-compiled force function specialised to this expansion.

        Prunes modes with negligible coefficients — for a spherical density
        this reduces from 81 modes to 1, giving ~80× speedup.
        """
        log_r_knots, phi_coeffs, _, _, _ = self._stacked
        a_c, b_c, c_c, d_c = phi_coeffs
        l_max = self.l_max

        # ── Identify active modes (within symmetry-allowed set) ──────────
        max_per_mode = np.max(
            np.abs(np.asarray(a_c)) + np.abs(np.asarray(b_c))
            + np.abs(np.asarray(c_c)) + np.abs(np.asarray(d_c)),
            axis=1,
        )
        global_max = float(np.max(max_per_mode))
        threshold = 1e-10 * global_max

        lm_keys = self._lm_keys
        active_idx = [i for i, v in enumerate(max_per_mode) if v > threshold]
        active_lm = [lm_keys[i] for i in active_idx]
        n_active = len(active_lm)

        if n_active == 0:
            @jax.jit
            def _zero_force(x, y, z):
                return jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)
            return _zero_force

        # Pruned coefficients: (n_active, n_r-1)
        idx_arr = np.array(active_idx)
        a_p = jnp.asarray(np.asarray(a_c)[idx_arr])
        b_p = jnp.asarray(np.asarray(b_c)[idx_arr])
        c_p = jnp.asarray(np.asarray(c_c)[idx_arr])
        d_p = jnp.asarray(np.asarray(d_c)[idx_arr])

        # Which l and m values do we actually need?
        l_needed = max(l for l, _ in active_lm)
        m_needed = max(abs(m) for _, m in active_lm)

        # Pre-compute normalisation constants
        norms = []
        for l, ms in active_lm:
            m = abs(ms)
            N_lm = _normalization(l, m)
            norms.append(math.sqrt(2) * N_lm if m > 0 else N_lm)

        # Build the specialised JIT function as a closure.
        # Captured arrays become XLA constants; Python ints/lists control
        # loop unrolling at trace time.
        log_r_k = log_r_knots   # captured constant

        @jax.jit
        def _force(x, y, z):
            # ── Coordinates ──────────────────────────────────────────
            r_xy_sq = x * x + y * y
            r_sq = r_xy_sq + z * z
            r = jnp.sqrt(r_sq)
            r_safe = jnp.maximum(r, 1e-30)
            r_xy = jnp.sqrt(r_xy_sq)
            r_xy_safe = jnp.maximum(r_xy, 1e-30)

            cos_theta = z / r_safe
            sin_theta = r_xy / r_safe
            sin_theta_safe = jnp.maximum(sin_theta, 1e-30)
            cos_phi = jnp.where(r_xy > 1e-30, x / r_xy_safe, 1.0)
            sin_phi = jnp.where(r_xy > 1e-30, y / r_xy_safe, 0.0)

            log_r = jnp.log(r_safe)
            inv_r = 1.0 / r_safe

            # ── Spline lookup ────────────────────────────────────────
            idx = jnp.searchsorted(log_r_k, log_r, side="right") - 1
            idx = jnp.clip(idx, 0, log_r_k.shape[0] - 2)
            dt = log_r - log_r_k[idx]
            dt2 = dt * dt

            # ── Spline values & derivatives for active modes ─────────
            phi_vals  = (a_p[:, idx] + b_p[:, idx] * dt
                         + c_p[:, idx] * dt2 + d_p[:, idx] * (dt2 * dt))
            phi_dlogr = (b_p[:, idx] + 2.0 * c_p[:, idx] * dt
                         + 3.0 * d_p[:, idx] * dt2)

            # ── Legendre recurrence (only up to l_needed) ────────────
            P = _alp_recurrence(l_needed, cos_theta)

            # ── Chebyshev trig recurrence (only up to m_needed) ──────
            cos_m = [jnp.ones_like(cos_phi)]
            sin_m = [jnp.zeros_like(cos_phi)]
            for mm in range(1, m_needed + 1):
                cos_m.append(cos_m[mm - 1] * cos_phi - sin_m[mm - 1] * sin_phi)
                sin_m.append(sin_m[mm - 1] * cos_phi + cos_m[mm - 1] * sin_phi)

            # ── Y_lm and derivatives for active modes only ───────────
            Y_list = []
            dYdth_list = []
            dYdphi_st_list = []

            for i_mode, (l, ms) in enumerate(active_lm):
                m = abs(ms)
                fac = norms[i_mode]

                if ms > 0:
                    trig = cos_m[m]
                    trig_other = sin_m[m]
                elif ms < 0:
                    trig = sin_m[m]
                    trig_other = cos_m[m]
                else:
                    trig = 1.0
                    trig_other = None

                Plm = P[l][m]
                Y_list.append(fac * Plm * trig)

                # dY/dθ
                if l == 0:
                    dYdth_list.append(jnp.zeros_like(cos_theta))
                else:
                    P_prev = P[l - 1][m] if m <= l - 1 else 0.0
                    dPdth = (l * cos_theta * Plm - (l + m) * P_prev) / sin_theta_safe
                    dYdth_list.append(fac * dPdth * trig)

                # (1/sinθ) dY/dφ
                if ms == 0:
                    dYdphi_st_list.append(jnp.zeros_like(cos_phi))
                else:
                    dYdphi_st_list.append(
                        -ms * fac * Plm / sin_theta_safe * trig_other
                    )

            Y_arr = jnp.stack(Y_list)
            dYdth_arr = jnp.stack(dYdth_list)
            dYdphi_st_arr = jnp.stack(dYdphi_st_list)

            # ── Assemble forces ──────────────────────────────────────
            dPhi_dr = jnp.einsum('mn,mn->n', phi_dlogr, Y_arr) * inv_r
            dPhi_dth_r = jnp.einsum('mn,mn->n', phi_vals, dYdth_arr) * inv_r
            dPhi_dphi_rs = jnp.einsum('mn,mn->n', phi_vals, dYdphi_st_arr) * inv_r

            Fx = -(dPhi_dr * sin_theta * cos_phi
                   + dPhi_dth_r * cos_theta * cos_phi
                   - dPhi_dphi_rs * sin_phi)
            Fy = -(dPhi_dr * sin_theta * sin_phi
                   + dPhi_dth_r * cos_theta * sin_phi
                   + dPhi_dphi_rs * cos_phi)
            Fz = -(dPhi_dr * cos_theta
                   - dPhi_dth_r * sin_theta)

            return Fx, Fy, Fz

        return _force

    def acceleration(
        self,
        x: Float[Array, "..."],
        y: Float[Array, "..."],
        z: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Return gravitational acceleration (ax, ay, az) = -grad Phi via autodiff."""
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
