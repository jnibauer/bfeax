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


def ylm_force_components(
    l_max: int,
    cos_theta: Float[Array, "..."],
    sin_theta: Float[Array, "..."],
    cos_phi: Float[Array, "..."],
    sin_phi: Float[Array, "..."],
) -> tuple[Float[Array, "n ..."], Float[Array, "n ..."], Float[Array, "n ..."]]:
    """
    Compute Y_lm, dY_lm/dθ, and (1/sinθ) dY_lm/dφ for analytical force evaluation.

    Avoids arccos/arctan2 by working directly with cos/sin values.
    Uses Chebyshev recurrence for cos(mφ), sin(mφ) instead of per-m trig calls.

    Parameters
    ----------
    l_max     : max harmonic degree (static Python int)
    cos_theta : cos(θ), shape (N,)
    sin_theta : sin(θ), shape (N,) — must be non-negative
    cos_phi   : cos(φ), shape (N,)
    sin_phi   : sin(φ), shape (N,)

    Returns
    -------
    Y_arr                    : (n_modes, N)
    dYdtheta_arr             : (n_modes, N)
    dYdphi_over_sintheta_arr : (n_modes, N)
    """
    P = _alp_recurrence(l_max, cos_theta)
    sin_theta_safe = jnp.maximum(jnp.abs(sin_theta), 1e-30)

    # Chebyshev recurrence: cos(mφ), sin(mφ) from cos(φ), sin(φ)
    cos_m = [jnp.ones_like(cos_phi)]    # cos(0·φ) = 1
    sin_m = [jnp.zeros_like(cos_phi)]   # sin(0·φ) = 0
    for m in range(1, l_max + 1):
        cos_m.append(cos_m[m - 1] * cos_phi - sin_m[m - 1] * sin_phi)
        sin_m.append(sin_m[m - 1] * cos_phi + cos_m[m - 1] * sin_phi)

    Y_list = []
    dYdth_list = []
    dYdphi_st_list = []

    for l in range(l_max + 1):
        for ms in range(-l, l + 1):     # m_signed, matches lm_keys ordering
            m = abs(ms)
            N = _normalization(l, m)
            fac = math.sqrt(2) if m > 0 else 1.0

            # Trig factor for this (l, ms) and the "other" trig for dY/dφ
            if ms > 0:
                trig = cos_m[m]
                trig_other = sin_m[m]
            elif ms < 0:
                trig = sin_m[m]
                trig_other = cos_m[m]
            else:
                trig = jnp.ones_like(cos_phi)
                trig_other = None

            # Y_lm
            Y_list.append(fac * N * P[l][m] * trig)

            # dY_lm/dθ = fac · N · dP_l^m/dθ · trig
            # dP_l^m/dθ = [l·cosθ·P_l^m − (l+m)·P_{l−1}^m] / sinθ
            if l == 0:
                dYdth_list.append(jnp.zeros_like(cos_theta))
            else:
                P_prev = P[l - 1][m] if m <= l - 1 else jnp.zeros_like(cos_theta)
                dPdth = (l * cos_theta * P[l][m] - (l + m) * P_prev) / sin_theta_safe
                dYdth_list.append(fac * N * dPdth * trig)

            # (1/sinθ) · dY_lm/dφ
            # dY_{l,m}/dφ = −m_signed · Y_{l,−m}
            # (1/sinθ) · dY/dφ = −ms · fac · N · P_l^m / sinθ · trig_other
            if ms == 0:
                dYdphi_st_list.append(jnp.zeros_like(cos_phi))
            else:
                dYdphi_st_list.append(
                    -ms * fac * N * P[l][m] / sin_theta_safe * trig_other
                )

    return jnp.stack(Y_list), jnp.stack(dYdth_list), jnp.stack(dYdphi_st_list)


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
