"""
Solve the radial Poisson equation for each (l,m) mode:

    Phi_lm(r) = -4*pi*G / (2l+1) * [
        r^{-(l+1)} * integral_0^r  rho_lm(r') r'^{l+2} dr'
      + r^l       * integral_r^inf rho_lm(r') r'^{1-l} dr'
    ]

Boundary corrections
--------------------
Both the inner (0 → r_min) and outer (r_max → ∞) truncation of the grid
introduce bias.  We estimate the power-law slope of rho_lm at each boundary
and append analytical tails:

  Inner tail  (r < r_min):  rho_lm ~ A_in  * r^alpha_in
    ΔI_in  = A_in  * r_min^{alpha_in + l + 3} / (alpha_in + l + 3)
    [added as a constant to I_in(r) for all r ≥ r_min]

  Outer tail  (r > r_max):  rho_lm ~ A_out * r^alpha_out
    ΔI_out = A_out * r_max^{2 - l}   / (l - alpha_out - 2)
    [added as a constant to I_out(r) for all r ≤ r_max]

Both require that the respective power-law integral converges and that the
boundary value is non-negligible; jnp.where guards handle the degenerate cases.

Units: G = 1 convention.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float

from .spline import natural_cubic_spline_coeffs, spline_eval


_4PI_G = 4.0 * 3.141592653589793  # G = 1


def solve_poisson_lm(
    r: Float[Array, "nr"],
    rho_lm: dict[tuple[int, int], Float[Array, "nr"]],
) -> dict[tuple[int, int], tuple]:
    """
    Given rho_lm(r) on a radial grid, return spline coefficients for
    Phi_lm(r) for every (l,m) pair.

    Returns
    -------
    dict (l,m) -> (log_r_knots, spline_coeffs)
    """
    log_r = jnp.log(r)
    phi_splines: dict[tuple[int, int], tuple] = {}

    for (l, m), rho_vals in rho_lm.items():
        phi_vals = _green_function_integral(r, rho_vals, l)
        coeffs = natural_cubic_spline_coeffs(log_r, phi_vals)
        phi_splines[(l, m)] = (log_r, coeffs)

    return phi_splines


@partial(jax.jit, static_argnums=(2,))
def _green_function_integral(
    r: Float[Array, "nr"],
    rho_lm: Float[Array, "nr"],
    l: int,
) -> Float[Array, "nr"]:
    """
    Compute Phi_lm(r) via the two-pass Green's function formula with
    inner and outer power-law boundary corrections.
    """
    dr = jnp.diff(r)                       # (nr-1,)

    f_in  = rho_lm * r ** (l + 2)          # integrand for I_in
    f_out = rho_lm * r ** (1 - l)          # integrand for I_out

    # ── Inner power-law correction (0 → r_min) ────────────────────────
    # Estimate slope from first 3 grid points via log-log finite difference.
    log_rho_first = jnp.log(jnp.abs(rho_lm[:3]) + 1e-300)
    log_r_first   = jnp.log(r[:3])
    alpha_in      = jnp.mean(jnp.diff(log_rho_first) / jnp.diff(log_r_first))

    sign_in  = jnp.sign(rho_lm[0])
    A_in     = sign_in * jnp.abs(rho_lm[0]) / (r[0] ** alpha_in)

    # Guard: only apply when boundary value is non-negligible
    rho_scale    = jnp.max(jnp.abs(rho_lm)) + 1e-300
    inner_active = jnp.abs(rho_lm[0]) > 1e-8 * rho_scale

    # Convergence: need alpha_in + l + 3 > 0  (l is a static Python int)
    exp_in   = alpha_in + (l + 3)          # l+3 is a compile-time constant
    safe_in  = jnp.where(jnp.abs(exp_in) > 1e-6, exp_in, 1e-6)

    delta_I_in = A_in * r[0] ** (alpha_in + (l + 3)) / safe_in
    delta_I_in = jnp.where(inner_active & (exp_in > 0.0), delta_I_in, 0.0)

    # ── Interior cumulative sum (inside-out) ──────────────────────────
    trap_in = 0.5 * (f_in[:-1] + f_in[1:]) * dr
    I_in    = jnp.concatenate([jnp.zeros(1), jnp.cumsum(trap_in)]) + delta_I_in

    # ── Outer power-law correction (r_max → ∞) ───────────────────────
    log_rho_last = jnp.log(jnp.abs(rho_lm[-3:]) + 1e-300)
    log_r_last   = jnp.log(r[-3:])
    alpha_out    = jnp.mean(jnp.diff(log_rho_last) / jnp.diff(log_r_last))

    outer_active = rho_lm[-1] > 0.0
    tail_conv    = alpha_out < (l - 2)

    safe_out    = jnp.where(jnp.abs(l - alpha_out - 2) > 1e-6,
                            l - alpha_out - 2, 1e-6)
    delta_I_out = rho_lm[-1] * r[-1] ** (2 - l) / safe_out
    delta_I_out = jnp.where(outer_active & tail_conv, delta_I_out, 0.0)

    # ── Exterior cumulative sum (outside-in) ──────────────────────────
    trap_out    = 0.5 * (f_out[:-1] + f_out[1:]) * dr
    I_out_rev   = jnp.cumsum(trap_out[::-1])[::-1]
    I_out       = jnp.concatenate([I_out_rev, jnp.zeros(1)]) + delta_I_out

    phi = -_4PI_G / (2 * l + 1) * (
        r ** (-(l + 1)) * I_in + r ** l * I_out
    )
    return phi
