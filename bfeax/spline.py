"""
Natural cubic splines in JAX on a fixed (possibly log-spaced) 1-D grid.

JAX sharp-bit notes:
  - Spline *fitting* (solving the tridiagonal system) uses jnp.linalg.solve
    on a static-shape matrix — safe to JIT.
  - Evaluation uses jnp.searchsorted which is supported under JIT.
  - No Python-level branching on traced values.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def natural_cubic_spline_coeffs(
    x: Float[Array, "n"],
    y: Float[Array, "n"],
) -> tuple[
    Float[Array, "n-1"],
    Float[Array, "n-1"],
    Float[Array, "n-1"],
    Float[Array, "n-1"],
]:
    """
    Fit a natural cubic spline to data (x, y) with x strictly increasing.

    Returns (a, b, c, d) such that on interval [x_i, x_{i+1}]:
        S(t) = a[i] + b[i]*dt + c[i]*dt^2 + d[i]*dt^3
    where dt = t - x[i].
    """
    n = x.shape[0]
    h = jnp.diff(x)          # (n-1,)

    # Build RHS for second derivatives M
    rhs = 3.0 * (jnp.diff(y[1:]) / h[1:] - jnp.diff(y[:-1]) / h[:-1])
    # rhs has shape (n-2,)

    # Tridiagonal system (natural BC: M_0 = M_{n-1} = 0)
    diag_main = 2.0 * (h[:-1] + h[1:])   # (n-2,)
    diag_sub  = h[1:-1]                    # (n-3,)

    # Assemble as dense matrix (n-2) x (n-2) — fine for moderate n
    A = (
        jnp.diag(diag_main)
        + jnp.diag(diag_sub, k=1)
        + jnp.diag(diag_sub, k=-1)
    )
    M_inner = jnp.linalg.solve(A, rhs)     # (n-2,)

    # Pad with natural BC
    M = jnp.concatenate([jnp.zeros(1), M_inner, jnp.zeros(1)])  # (n,)

    # Compute polynomial coefficients on each interval
    a = y[:-1]
    b = jnp.diff(y) / h - h * (2.0 * M[:-1] + M[1:]) / 3.0
    c = M[:-1]
    d = jnp.diff(M) / (3.0 * h)

    return a, b, c, d


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def spline_eval(
    x_knots: Float[Array, "n"],
    coeffs: tuple,
    x_query: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Evaluate a spline (from natural_cubic_spline_coeffs) at x_query.
    Clamps to boundary for out-of-range queries.
    """
    a, b, c, d = coeffs
    # Clamp index to valid range [0, n-2]
    idx = jnp.searchsorted(x_knots, x_query, side="right") - 1
    idx = jnp.clip(idx, 0, x_knots.shape[0] - 2)

    dt = x_query - x_knots[idx]
    return a[idx] + b[idx] * dt + c[idx] * dt**2 + d[idx] * dt**3


def natural_cubic_spline_coeffs_batch(
    x: Float[Array, "n"],
    Y: Float[Array, "k n"],
) -> tuple[
    Float[Array, "k n-1"],
    Float[Array, "k n-1"],
    Float[Array, "k n-1"],
    Float[Array, "k n-1"],
]:
    """
    Fit natural cubic splines to k curves sharing the same x knots.

    Single matrix factorisation — much faster than k separate calls
    to natural_cubic_spline_coeffs when k is large.
    """
    h = jnp.diff(x)                                          # (n-1,)

    RHS = 3.0 * (
        jnp.diff(Y[:, 1:], axis=1) / h[1:]
        - jnp.diff(Y[:, :-1], axis=1) / h[:-1]
    )  # (k, n-2)

    diag_main = 2.0 * (h[:-1] + h[1:])
    diag_sub  = h[1:-1]
    A = (
        jnp.diag(diag_main)
        + jnp.diag(diag_sub, k=1)
        + jnp.diag(diag_sub, k=-1)
    )
    # Solve A @ M_inner_j = RHS_j for all j simultaneously
    M_inner = jnp.linalg.solve(A, RHS.T).T                   # (k, n-2)

    k = Y.shape[0]
    M = jnp.concatenate(
        [jnp.zeros((k, 1)), M_inner, jnp.zeros((k, 1))],
        axis=1,
    )  # (k, n)

    a = Y[:, :-1]
    b = jnp.diff(Y, axis=1) / h - h * (2.0 * M[:, :-1] + M[:, 1:]) / 3.0
    c = M[:, :-1]
    d = jnp.diff(M, axis=1) / (3.0 * h)

    return a, b, c, d


def spline_deriv(
    x_knots: Float[Array, "n"],
    coeffs: tuple,
    x_query: Float[Array, "..."],
) -> Float[Array, "..."]:
    """First derivative of the spline."""
    a, b, c, d = coeffs
    idx = jnp.searchsorted(x_knots, x_query, side="right") - 1
    idx = jnp.clip(idx, 0, x_knots.shape[0] - 2)

    dt = x_query - x_knots[idx]
    return b[idx] + 2.0 * c[idx] * dt + 3.0 * d[idx] * dt**2
