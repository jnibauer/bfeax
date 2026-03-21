"""Real (orthonormal) spherical harmonics Y_lm(theta, phi).

Convention:
  m > 0  ->  Y_lm = sqrt(2) * Re[Y_l^m]  (cosine)
  m = 0  ->  Y_l0
  m < 0  ->  Y_lm = sqrt(2) * Im[Y_l^|m|]  (sine)

Normalization: integral Y_lm * Y_l'm' dOmega = delta_{ll'} delta_{mm'}

JAX sharp-bit notes:
  - lmax is a *Python int* (static) so loops unroll at trace time.
  - All array shapes are static; no dynamic indexing via traced ints.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Associated Legendre polynomials via recurrence (un-normalised)
# ---------------------------------------------------------------------------

def _alp_recurrence(l_max: int, x: Float[Array, "..."]):
    """
    Compute P_l^m(x) for all 0 <= m <= l <= l_max via the standard
    recurrence.  Returns a list-of-lists P[l][m] where each entry is an
    array with the same shape as x.

    We use the Schmidt semi-normalized form internally, then fold the
    full normalization into ylm_real.
    """
    # P[l][m] stored as Python list for static structure
    P = [[None] * (l_max + 1) for _ in range(l_max + 1)]

    # Seed: P_0^0 = 1
    P[0][0] = jnp.ones_like(x)

    sin_theta = jnp.sqrt(jnp.clip(1.0 - x * x, 0.0))

    for l in range(1, l_max + 1):
        # Diagonal: P_l^l = -(2l-1) * sin * P_{l-1}^{l-1}
        P[l][l] = -(2 * l - 1) * sin_theta * P[l - 1][l - 1]
        # Sub-diagonal: P_l^{l-1} = (2l-1) * x * P_{l-1}^{l-1}
        P[l][l - 1] = (2 * l - 1) * x * P[l - 1][l - 1]
        # Upward recurrence for m <= l-2
        for m in range(l - 2, -1, -1):
            P[l][m] = ((2 * l - 1) * x * P[l - 1][m] - (l + m - 1) * P[l - 2][m]) / (l - m)

    return P


def _normalization(l: int, m: int) -> float:
    """Full orthonormal normalization factor for Y_l^m."""
    # sqrt( (2l+1)/(4pi) * (l-|m|)! / (l+|m|)! )
    am = abs(m)
    factor = (2 * l + 1) / (4 * math.pi) * math.factorial(l - am) / math.factorial(l + am)
    return math.sqrt(factor)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ylm_real(
    l_max: int,
    theta: Float[Array, "..."],
    phi: Float[Array, "..."],
) -> dict[tuple[int, int], Float[Array, "..."]]:
    """
    Compute all real spherical harmonics Y_lm for 0 <= l <= l_max,
    -l <= m <= l.

    Parameters
    ----------
    l_max : int  (static Python int — determines output dict structure)
    theta : colatitude in [0, pi]
    phi   : azimuth in [0, 2*pi)

    Returns
    -------
    dict mapping (l, m) -> array of same shape as theta/phi.
    """
    cos_theta = jnp.cos(theta)
    P = _alp_recurrence(l_max, cos_theta)

    out: dict[tuple[int, int], Float[Array, "..."]] = {}
    for l in range(l_max + 1):
        # m = 0
        N0 = _normalization(l, 0)
        out[(l, 0)] = N0 * P[l][0]
        for m in range(1, l + 1):
            N = _normalization(l, m)
            cos_m = jnp.cos(m * phi)
            sin_m = jnp.sin(m * phi)
            out[(l,  m)] = math.sqrt(2) * N * P[l][m] * cos_m
            out[(l, -m)] = math.sqrt(2) * N * P[l][m] * sin_m

    return out


def ylm_grid(
    l_max: int,
    n_theta: int,
    n_phi: int,
) -> tuple[
    Float[Array, "n_theta n_phi"],
    Float[Array, "n_theta n_phi"],
    dict[tuple[int, int], Float[Array, "n_theta n_phi"]],
    Float[Array, "n_theta"],
]:
    """
    Build a Gauss-Legendre × uniform-phi integration grid and evaluate
    all Y_lm on it.

    Returns
    -------
    theta_grid, phi_grid : 2-D meshes
    Ylm                  : dict (l,m) -> (n_theta, n_phi) array
    gl_weights           : 1-D Gauss-Legendre weights in cos(theta)
    """
    # Gauss-Legendre nodes/weights in cos(theta) on [-1, 1]
    import numpy as np
    _nw = np.polynomial.legendre.leggauss(n_theta)
    nodes   = jnp.array(_nw[0])   # (n_theta,)
    weights = jnp.array(_nw[1])   # (n_theta,)
    cos_th = nodes
    theta_1d = jnp.arccos(cos_th)

    phi_1d = jnp.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)

    theta_grid, phi_grid = jnp.meshgrid(theta_1d, phi_1d, indexing="ij")
    Ylm = ylm_real(l_max, theta_grid, phi_grid)

    return theta_grid, phi_grid, Ylm, weights


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature (pure Python, static)
# ---------------------------------------------------------------------------

def _gauss_legendre(n: int):
    """Return (nodes, weights) for Gauss-Legendre quadrature on [-1,1]."""
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return list(zip(nodes.tolist(), weights.tolist()))
