"""
Fast spheroidal density profile using a 1-D log-log spline lookup table.

The Agama Spheroid density:

    ρ(r̃) = ρ₀ (r̃/a)^{-γ} [1 + (r̃/a)^α]^{(γ-β)/α}  ×  exp[-(r̃/r_cut)^ξ]

where the spheroidal radius is

    r̃ = √(x² + (y/p)² + (z/q)²)

Parameters
----------
rho0 : float   — density normalisation ρ₀
alpha : float  — transition sharpness α (> 0)
beta  : float  — outer power-law slope β
gamma : float  — inner power-law slope γ  (< 3 for finite mass)
a     : float  — scale radius
p     : float  — y-axis ratio  (default 1 → spherical)
q     : float  — z-axis ratio  (default 1 → spherical)
r_cut : float or None — outer cutoff radius (None → no cutoff)
xi    : float  — cutoff strength ξ  (default 0 → no cutoff)
n_r   : int    — number of LUT knots  (default 512; 256 is usually fine)
r_min : float  — inner edge of the LUT grid
r_max : float  — outer edge of the LUT grid

Implementation
--------------
At construction, we:
  1. Evaluate the formula analytically on n_r log-spaced r̃ values.
  2. Fit a natural cubic spline to  log(ρ) vs log(r̃).

At evaluation, we:
  1. Compute r̃ from (x, y, z)                    — O(1) per point
  2. Look up log(ρ) via the spline                 — O(log n_r) per point
  3. Return exp(log ρ)                             — O(1) per point

The class is callable as  rho(x, y, z)  and is compatible with
MultipoleExpansion.from_density.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .spline import natural_cubic_spline_coeffs, spline_eval


# ---------------------------------------------------------------------------
# Internal: analytical formula on a scalar spheroidal radius
# ---------------------------------------------------------------------------

def _rho_analytical(r_tilde, rho0, alpha, beta, gamma, a, r_cut_eff, xi):
    """
    Evaluate the spheroid density at scalar (or batched) spheroidal radius.
    r_cut_eff : pass jnp.inf when there is no cutoff.
    """
    s = r_tilde / a
    # Power-law × transition
    rho = rho0 * s ** (-gamma) * (1.0 + s ** alpha) ** ((gamma - beta) / alpha)
    # Outer cutoff (exp term = 1 when r_cut_eff = inf)
    cutoff = jnp.exp(-(r_tilde / r_cut_eff) ** xi)
    rho = rho * jnp.where(xi == 0.0, 1.0, cutoff)
    return rho


# ---------------------------------------------------------------------------
# SpheroidDensity
# ---------------------------------------------------------------------------

class SpheroidDensity:
    """
    Fast spheroidal density  ρ(x, y, z)  via a 1-D log-log spline LUT.

    The density is a function of the spheroidal radius only, so a single
    1-D interpolation table suffices — evaluation is O(log n_r) after a
    single sqrt.

    Parameters
    ----------
    rho0, alpha, beta, gamma, a : floats — profile shape (see module docstring)
    p, q  : axis ratios (y and z);  default 1.0 → spherical
    r_cut : outer cutoff radius (None = no cutoff)
    xi    : cutoff strength (0 = no cutoff)
    n_r   : number of spline knots (default 512)
    r_min, r_max : radial range for the LUT
    """

    def __init__(
        self,
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
        n_r: int = 512,
        r_min: float | None = None,
        r_max: float | None = None,
    ):
        self.rho0  = float(rho0)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.a     = float(a)
        self.p     = float(p)
        self.q     = float(q)
        self.xi    = float(xi)
        self.r_cut_eff = float(r_cut) if r_cut is not None else float("inf")

        # --- default grid bounds -----------------------------------------------
        # Inner: well inside the cusp ( ~ a/1000 )
        # Outer: well outside any cutoff or transition
        if r_min is None:
            r_min = a * 1e-4
        if r_max is None:
            if r_cut is not None:
                r_max = max(a * 1e3, r_cut * 100.0)
            else:
                r_max = a * 1e4

        self.r_min = float(r_min)
        self.r_max = float(r_max)

        # --- build the LUT -------------------------------------------------------
        r_grid_full = jnp.exp(
            jnp.linspace(jnp.log(self.r_min), jnp.log(self.r_max), n_r)
        )

        rho_grid_full = _rho_analytical(
            r_grid_full,
            self.rho0, self.alpha, self.beta, self.gamma, self.a,
            self.r_cut_eff, self.xi,
        )

        # Trim to the range where ρ is numerically significant.
        # This prevents log(ρ) → -∞ when an exponential cutoff drives ρ → 0,
        # which would corrupt the spline coefficients.
        rho_np  = np.asarray(rho_grid_full)
        r_np    = np.asarray(r_grid_full)
        sig_mask = rho_np > 1e-100 * rho_np.max()
        last_sig = int(np.where(sig_mask)[0][-1]) + 1
        r_np    = r_np[:last_sig]
        rho_np  = rho_np[:last_sig]

        # Work in log-log space for smooth interpolation
        log_r   = jnp.array(np.log(r_np))
        log_rho = jnp.array(np.log(rho_np))

        self._log_r_knots    = log_r
        self._log_rho_coeffs = natural_cubic_spline_coeffs(log_r, log_rho)
        # Beyond the last knot the spline is clamped; density is negligible there.
        self._log_r_max_lut  = float(log_r[-1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def r_tilde(self, x, y, z):
        """Spheroidal radius r̃ = √(x² + (y/p)² + (z/q)²)."""
        return jnp.sqrt(x**2 + (y / self.p)**2 + (z / self.q)**2)

    def __call__(self, x, y, z):
        """
        Evaluate ρ(x, y, z) via the spline LUT.
        Fully JAX-traceable and JIT-compatible.
        """
        rt = self.r_tilde(x, y, z)
        log_rt = jnp.log(jnp.where(rt > 0.0, rt, self.r_min))
        # Clamp to the fitted range; beyond the upper edge density is negligible
        log_rt_clamped = jnp.clip(log_rt, self._log_r_knots[0], self._log_r_max_lut)
        log_rho = spline_eval(self._log_r_knots, self._log_rho_coeffs, log_rt_clamped)
        rho = jnp.exp(log_rho)
        # Zero out beyond the significant range (exponential cutoff region)
        return jnp.where(log_rt <= self._log_r_max_lut, rho, 0.0)

    def analytical(self, x, y, z):
        """Direct analytical evaluation (no spline) — for accuracy checks."""
        rt = self.r_tilde(x, y, z)
        return _rho_analytical(
            rt,
            self.rho0, self.alpha, self.beta, self.gamma, self.a,
            self.r_cut_eff, self.xi,
        )
