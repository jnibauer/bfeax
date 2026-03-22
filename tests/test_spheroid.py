"""
Smoke test + benchmark for SpheroidDensity.

Checks:
  1. LUT vs analytical agrees to < 0.01% everywhere on the grid.
  2. JIT-compiled throughput on 1M random points.
"""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from bfeax import SpheroidDensity

# ---------------------------------------------------------------------------
# NFW-like parameters  (alpha=1, beta=3, gamma=1)
# ---------------------------------------------------------------------------
nfw = SpheroidDensity(
    rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=10.0,
    p=0.8, q=0.5,
    n_r=512, r_min=1e-3, r_max=1e4,
)

# --- accuracy on a dense test grid -----------------------------------------
r_test = jnp.exp(jnp.linspace(jnp.log(nfw.r_min * 1.01), jnp.log(nfw.r_max * 0.99), 2000))
x_test, y_test, z_test = r_test, jnp.zeros_like(r_test), jnp.zeros_like(r_test)

lut  = nfw(x_test, y_test, z_test)
ref  = nfw.analytical(x_test, y_test, z_test)
relerr = jnp.abs(lut - ref) / (jnp.abs(ref) + 1e-300)

print(f"LUT vs analytical — max relative error: {float(relerr.max()):.2e}")
print(f"LUT vs analytical — median rel error:   {float(jnp.median(relerr)):.2e}")
assert float(relerr.max()) < 1e-4, "LUT error too large!"

# --- with outer cutoff -------------------------------------------------------
cut = SpheroidDensity(
    rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=10.0,
    r_cut=100.0, xi=2.0,
    n_r=512, r_min=1e-3, r_max=1e4,
)
lut2  = cut(x_test, y_test, z_test)
ref2  = cut.analytical(x_test, y_test, z_test)
scale = jnp.max(jnp.abs(ref2))  # normalise against peak density
relerr2 = jnp.abs(lut2 - ref2) / (jnp.abs(ref2) + 1e-6 * scale)
print(f"With cutoff — max relative error:       {float(relerr2.max()):.2e}")
assert float(relerr2.max()) < 1e-4, "Cutoff LUT error too large!"

# --- JIT throughput ----------------------------------------------------------
rho_jit = jax.jit(nfw)

key = jax.random.PRNGKey(0)
N = 1_000_000
pts = jax.random.uniform(key, (3, N), minval=-50.0, maxval=50.0)
x, y, z = pts[0], pts[1], pts[2]

# Warm up
_ = rho_jit(x, y, z).block_until_ready()

t0 = time.perf_counter()
_ = rho_jit(x, y, z).block_until_ready()
t1 = time.perf_counter()
print(f"JIT throughput: {N/(t1-t0)/1e6:.1f}M points/s  ({(t1-t0)*1e3:.1f} ms for {N//1000}k pts)")

print("\nAll checks passed.")
