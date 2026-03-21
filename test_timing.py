"""Timing and convergence test."""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from bfe import MultipoleExpansion

b = 1.0; M = 1.0

def rho_plummer(x, y, z):
    r2 = x**2 + y**2 + z**2
    return 3 * M / (4 * jnp.pi * b**3) * (1 + r2 / b**2) ** (-2.5)

def phi_exact(x, y, z):
    return -M / jnp.sqrt(x**2 + y**2 + z**2 + b**2)

# Convergence with n_r
print("=== Convergence with n_r (l_max=4) ===")
r_test = jnp.array([0.5, 1.0, 2.0, 5.0])
for n_r in [16, 32, 64, 128]:
    t0 = time.perf_counter()
    exp = MultipoleExpansion.from_density(rho_plummer, 1e-2, 1e2, n_r, l_max=4)
    dt_build = time.perf_counter() - t0
    phi_bfe = exp(r_test, jnp.zeros_like(r_test), jnp.zeros_like(r_test))
    phi_ref = phi_exact(r_test, jnp.zeros_like(r_test), jnp.zeros_like(r_test))
    err = float(jnp.max(jnp.abs((phi_bfe - phi_ref) / phi_ref)))
    print(f"  n_r={n_r:3d}  build={dt_build:.3f}s  max_rel_err={err:.2e}")

# Convergence with l_max (for a non-spherical density: off-axis evaluation)
print("\n=== Convergence with l_max (n_r=64, off-axis) ===")
x_test = jnp.array([0.5, 1.0, 2.0])
y_test = jnp.array([0.3, 0.7, 1.5])
z_test = jnp.array([0.2, 0.5, 1.0])
for l_max in [0, 2, 4, 6]:
    exp = MultipoleExpansion.from_density(rho_plummer, 1e-2, 1e2, 64, l_max=l_max)
    phi_bfe = exp(x_test, y_test, z_test)
    phi_ref = phi_exact(x_test, y_test, z_test)
    err = float(jnp.max(jnp.abs((phi_bfe - phi_ref) / phi_ref)))
    print(f"  l_max={l_max}  max_rel_err={err:.2e}")

# JIT evaluation timing
print("\n=== JIT evaluation throughput ===")
exp = MultipoleExpansion.from_density(rho_plummer, 1e-2, 1e2, 64, l_max=4)
phi_jit = jax.jit(exp.__call__)

N = 10_000
key = jax.random.PRNGKey(0)
pts = jax.random.normal(key, (3, N))
_ = phi_jit(pts[0], pts[1], pts[2]).block_until_ready()  # warm up

t0 = time.perf_counter()
for _ in range(10):
    phi_jit(pts[0], pts[1], pts[2]).block_until_ready()
dt = (time.perf_counter() - t0) / 10
print(f"  {N} points, l_max=4: {dt*1000:.2f} ms/call  ({N/dt/1e6:.2f} M pts/s)")
