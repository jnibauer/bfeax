"""
Project rho(x,y,z) onto spherical harmonic coefficients rho_lm(r).

For each radial grid point r_i we integrate over angles:
    rho_lm(r_i) = integral rho(r_i, theta, phi) * Y_lm(theta, phi) dOmega

using a Gauss-Legendre x uniform-phi quadrature grid.

The result is a dict (l,m) -> 1-D array of length n_r.
"""

import math
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .grid import make_radial_grid
from .sph_harm import ylm_grid


def density_to_sph_coeffs(
    rho: Callable,              # rho(x, y, z) -> scalar or array
    r_grid: Float[Array, "nr"],
    l_max: int,
    n_theta: int | None = None,
    n_phi: int | None = None,
    lm_keys: list[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], Float[Array, "nr"]]:
    """
    Compute rho_lm(r) for all (l,m) with 0<=l<=l_max, -l<=m<=l.

    Parameters
    ----------
    rho      : callable (x,y,z) -> scalar.  Will be vmapped internally.
    r_grid   : 1-D array of radii (log-spaced recommended)
    l_max    : maximum spherical harmonic degree
    n_theta  : number of Gauss-Legendre nodes (default: 2*l_max + 2)
    n_phi    : number of uniform phi points   (default: 2*l_max + 3)

    Returns
    -------
    dict (l, m) -> array shape (n_r,)
    """
    if n_theta is None:
        # GL rule with n points integrates polynomials of degree 2n-1 exactly.
        # NFW is non-polynomial, so use 3x the minimum to stay well-converged.
        n_theta = 3 * (l_max + 2)
    if n_phi is None:
        # Need >= 2*l_max+1 to resolve cos(l_max * phi).  Add generous margin
        # and keep odd so phi=0 and phi=pi are never simultaneously sampled.
        n_phi = 4 * l_max + 7

    # Build angular grid and Y_lm values — static shapes
    theta_grid, phi_grid, Ylm, gl_weights = ylm_grid(l_max, n_theta, n_phi)
    # theta_grid, phi_grid : (n_theta, n_phi)
    # gl_weights           : (n_theta,) in cos(theta) space

    # Uniform phi weights
    dphi = 2.0 * math.pi / n_phi

    # Pre-compute quadrature weights: w[i,j] = gl_weights[i] * dphi
    # (Gauss-Legendre is already in d(cos theta), so we don't need sin theta)
    w2d = gl_weights[:, None] * dphi  # (n_theta, n_phi)

    # Convert angular grid to Cartesian unit vectors
    sin_theta = jnp.sin(theta_grid)
    ux = sin_theta * jnp.cos(phi_grid)  # (n_theta, n_phi)
    uy = sin_theta * jnp.sin(phi_grid)
    uz = jnp.cos(theta_grid)

    # Use provided lm_keys or default to all modes
    if lm_keys is None:
        lm_keys = sorted(Ylm.keys())

    # Pack Y values and quadrature into arrays indexed by a flat lm index
    Y_stack = jnp.stack([Ylm[k] for k in lm_keys])  # (n_lm, n_theta, n_phi)

    @jax.jit
    def _project(r_grid_inner):
        def _coeffs_at_r_vec(r):
            x = r * ux
            y = r * uy
            z = r * uz
            rho_vals = jax.vmap(jax.vmap(rho))(x, y, z)  # (n_theta, n_phi)
            weighted = rho_vals * w2d
            return jnp.einsum("lij,ij->l", Y_stack, weighted)  # (n_lm,)
        return jax.vmap(_coeffs_at_r_vec)(r_grid_inner)  # (n_r, n_lm)

    all_coeffs = _project(r_grid)

    out: dict[tuple[int, int], Float[Array, "nr"]] = {}
    for i, key in enumerate(lm_keys):
        out[key] = all_coeffs[:, i]

    return out
