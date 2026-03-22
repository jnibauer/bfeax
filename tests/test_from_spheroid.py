"""Test from_spheroid against from_density and benchmark."""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bfeax import MultipoleExpansion, SpheroidDensity

# --- Parameters ---
params = dict(rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=10.0)
grid   = dict(r_min=1e-2, r_max=300.0, n_r=128, l_max=8)

# --- Build via both paths ---
rho = SpheroidDensity(**params, p=0.8, q=0.5, n_r=512, r_min=1e-3, r_max=1e4)

print("Building via from_density (reference)...")
exp_ref = MultipoleExpansion.from_density(rho, **grid)

print("Building via from_spheroid (warmup)...")
exp_fast = MultipoleExpansion.from_spheroid(**params, p=0.8, q=0.5, **grid)

# --- Accuracy comparison ---
key = jax.random.PRNGKey(42)
N = 10000
pts = jax.random.uniform(key, (3, N), minval=-50.0, maxval=50.0)
x, y, z = pts[0], pts[1], pts[2]

phi_ref  = exp_ref(x, y, z)
phi_fast = exp_fast(x, y, z)
relerr   = jnp.abs(phi_ref - phi_fast) / (jnp.abs(phi_ref) + 1e-300)
print(f"\nPotential: max rel error = {float(relerr.max()):.2e}, "
      f"median = {float(jnp.median(relerr)):.2e}")

rho_ref  = exp_ref.density(x, y, z)
rho_fast = exp_fast.density(x, y, z)
relerr_r = jnp.abs(rho_ref - rho_fast) / (jnp.abs(rho_ref) + 1e-300)
print(f"Density:   max rel error = {float(relerr_r.max()):.2e}, "
      f"median = {float(jnp.median(relerr_r)):.2e}")

# --- Benchmark from_spheroid (post-warmup) ---
N_iter = 20
t0 = time.perf_counter()
for i in range(N_iter):
    # Vary a parameter each iteration to confirm no recompilation
    exp_i = MultipoleExpansion.from_spheroid(
        rho0=1.0 + 0.01*i, alpha=1.0, beta=3.0, gamma=1.0, a=10.0,
        p=0.8, q=0.5, **grid,
    )
t1 = time.perf_counter()
print(f"\nfrom_spheroid: {(t1-t0)*1e3/N_iter:.1f} ms avg ({N_iter} iters)")

# Compare against from_density
N_iter2 = 5
t2 = time.perf_counter()
for i in range(N_iter2):
    rho_i = SpheroidDensity(rho0=1.0+0.01*i, alpha=1.0, beta=3.0, gamma=1.0, a=10.0,
                            p=0.8, q=0.5, n_r=512, r_min=1e-3, r_max=1e4)
    exp_i = MultipoleExpansion.from_density(rho_i, **grid)
t3 = time.perf_counter()
print(f"from_density:  {(t3-t2)*1e3/N_iter2:.1f} ms avg ({N_iter2} iters)")

print(f"\nSpeedup: {(t3-t2)/N_iter2 / ((t1-t0)/N_iter):.1f}x")
