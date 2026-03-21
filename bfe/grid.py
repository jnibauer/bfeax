"""Log-spaced radial grids and associated utilities."""
import jax.numpy as jnp
from jaxtyping import Array, Float


def make_radial_grid(n: int, r_min: float, r_max: float) -> Float[Array, "n"]:
    """Return n log-uniformly spaced radii in [r_min, r_max]."""
    return jnp.exp(jnp.linspace(jnp.log(r_min), jnp.log(r_max), n))
