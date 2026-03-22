"""Test ExpansionGrid correctness and benchmark all build paths."""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bfeax import MultipoleExpansion, SpheroidDensity, ExpansionGrid

# --- Parameters ---
axis = dict(p=0.8, q=0.5)
grid = dict(r_min=1e-2, r_max=300.0, n_r=128, l_max=8)

rho = SpheroidDensity(rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=10.0,
                      **axis, n_r=512, r_min=1e-3, r_max=1e4)

# --- Reference: from_density ---
print("Building reference (from_density)...")
exp_ref = MultipoleExpansion.from_density(rho, **grid)

# --- ExpansionGrid ---
eg = ExpansionGrid(**grid)

# --- Accuracy check ---
exp_eg   = eg(rho)
rho_vals = jax.vmap(jax.vmap(jax.vmap(rho)))(eg.x, eg.y, eg.z)
exp_fv   = eg.from_values(rho_vals)

key = jax.random.PRNGKey(42)
pts = jax.random.uniform(key, (3, 10000), minval=-50.0, maxval=50.0)
x, y, z = pts[0], pts[1], pts[2]

phi_ref = exp_ref(x, y, z)
for label, exp_test in [("grid(rho)", exp_eg), ("from_values", exp_fv)]:
    err = jnp.abs(exp_test(x, y, z) - phi_ref) / (jnp.abs(phi_ref) + 1e-300)
    print(f"  {label:15s}  phi err: max={float(err.max()):.1e}  med={float(jnp.median(err)):.1e}")

# --- Warmup all paths ---
print("\nWarming up JIT caches...")
_ = MultipoleExpansion.from_density(rho, **grid)
_ = MultipoleExpansion.from_spheroid(rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0,
                                      a=10.0, **axis, **grid)
_ = eg(rho)
_ = eg.from_values(rho_vals)

def plummer(x, y, z):
    r2 = x**2 + y**2 + z**2
    return 3.0 / (4.0 * jnp.pi * 5.0**3) * (1.0 + r2 / 25.0) ** (-2.5)
_ = eg(plummer)

# Pre-build all SpheroidDensity variants (so this doesn't count in timing)
N = 30
rho_variants = [
    SpheroidDensity(rho0=1.0+0.01*i, alpha=1.0, beta=3.0, gamma=1.0, a=10.0,
                    **axis, n_r=512, r_min=1e-3, r_max=1e4)
    for i in range(N)
]

# Also pre-evaluate on grid
rho_on_grid_variants = [
    jax.vmap(jax.vmap(jax.vmap(rv)))(eg.x, eg.y, eg.z)
    for rv in rho_variants
]
for v in rho_on_grid_variants:
    v.block_until_ready()

print("\n--- Benchmark (all warmup done, SpheroidDensity pre-built) ---\n")

# 1. from_density
t0 = time.perf_counter()
for rv in rho_variants[:5]:
    e = MultipoleExpansion.from_density(rv, **grid)
t1 = time.perf_counter()
t_from_density = (t1 - t0) / 5
print(f"  from_density:                {t_from_density*1e3:7.1f} ms")

# 2. from_spheroid (specialised, fastest for spheroids)
t0 = time.perf_counter()
for i in range(N):
    e = MultipoleExpansion.from_spheroid(rho0=1.0+0.01*i, alpha=1.0, beta=3.0,
                                         gamma=1.0, a=10.0, **axis, **grid)
t1 = time.perf_counter()
t_spheroid = (t1 - t0) / N
print(f"  from_spheroid:               {t_spheroid*1e3:7.1f} ms")

# 3. grid(rho) — pass density function
t0 = time.perf_counter()
for rv in rho_variants:
    e = eg(rv)
t1 = time.perf_counter()
t_grid_rho = (t1 - t0) / N
print(f"  grid(rho):                   {t_grid_rho*1e3:7.1f} ms  (eval + build)")

# 4. grid.from_values — pass pre-evaluated density
t0 = time.perf_counter()
for rv in rho_on_grid_variants:
    e = eg.from_values(rv)
t1 = time.perf_counter()
t_from_values = (t1 - t0) / N
print(f"  grid.from_values:            {t_from_values*1e3:7.1f} ms  (build only)")

# 5. Density evaluation cost alone (for reference)
t0 = time.perf_counter()
for rv in rho_variants:
    vals = jax.vmap(jax.vmap(jax.vmap(rv)))(eg.x, eg.y, eg.z)
    vals.block_until_ready()
t1 = time.perf_counter()
t_eval = (t1 - t0) / N
print(f"  density eval only:           {t_eval*1e3:7.1f} ms  (reference)")

# 6. grid(plummer) — same function every time (best case for JIT cache)
t0 = time.perf_counter()
for _ in range(N):
    e = eg(plummer)
t1 = time.perf_counter()
t_plummer = (t1 - t0) / N
print(f"  grid(plummer):               {t_plummer*1e3:7.1f} ms  (cached func)")

print(f"\n  Speedup vs from_density:")
print(f"    from_spheroid:     {t_from_density/t_spheroid:5.1f}x")
print(f"    grid(rho):         {t_from_density/t_grid_rho:5.1f}x")
print(f"    grid.from_values:  {t_from_density/t_from_values:5.1f}x")
print(f"    grid(plummer):     {t_from_density/t_plummer:5.1f}x")
