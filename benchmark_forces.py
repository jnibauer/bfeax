"""
Wall-clock benchmark: bfeax vs Agama force evaluations.

Both codes expand an NFW profile with identical parameters:
    l_max=8, n_r=128, r_min=0.01, r_max=300

Three competitors:
  - bfeax          : jax.jit(jax.vmap(grad(Phi)))  — JIT-compiled, vmapped
  - Agama Multipole: same spherical-harmonic expansion, compiled C++
  - Agama exact NFW: analytic C++ NFW potential (best-case Agama baseline)

Protocol per N:
  1. Generate N random points uniformly in angle, log-uniform in r ∈ [0.1, 20].
  2. TWO warmup calls per N:
       - Call 1: triggers XLA compilation for that shape (slow).
       - Call 2: first execution on compiled code — primes hardware pipelines.
     Neither warmup call is included in the timing.
  3. N_REPS timed calls; report median wall time.
  4. jax.block_until_ready() ensures XLA dispatch is fully complete before
     stopping the clock (JAX uses async dispatch by default).

Note on JIT coverage: accel_fn = jax.jit(jax.vmap(grad(Phi))).
ylm_real() contains Python for-loops but they unroll at JAX *trace* time
and are compiled into the XLA graph — they do NOT re-execute on hot calls.
After warmup every timed call is pure XLA execution with no Python overhead.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import agama
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
agama.setUnits(mass=1, length=1, velocity=1)

from bfe import MultipoleExpansion

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L_MAX  = 8
N_R    = 128
R_MIN  = 0.01
R_MAX  = 300.0
N_WARMUP = 2    # warmup calls per N: call 1 = compile, call 2 = prime hardware
N_REPS   = 10   # timed repetitions after warmup (median reported)

N_vals = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000]

# ---------------------------------------------------------------------------
# Build expansions
# ---------------------------------------------------------------------------

print("Building bfeax expansion ...")
t0 = time.perf_counter()
exp = MultipoleExpansion.from_spheroid(
    rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=1.0,
    r_min=R_MIN, r_max=R_MAX, n_r=N_R, l_max=L_MAX,
)
print(f"  done in {time.perf_counter()-t0:.2f}s")

# JIT + vmap the gradient of the potential
# jit is on the outside so the entire vmapped computation is one XLA program.
_grad_phi = jax.grad(exp.__call__, argnums=(0, 1, 2))
accel_fn  = jax.jit(jax.vmap(_grad_phi))

print("Building Agama Multipole ...")
t0 = time.perf_counter()
pot_agama_multi = agama.Potential(
    type='Multipole', density='NFW',
    mass=1, scaleRadius=1,
    lmax=L_MAX, gridSizeR=N_R,
    rmin=R_MIN, rmax=R_MAX,
)
print(f"  done in {time.perf_counter()-t0:.2f}s")

print("Building Agama exact NFW ...")
t0 = time.perf_counter()
pot_agama_nfw = agama.Potential(type='NFW', mass=1, scaleRadius=1)
print(f"  done in {time.perf_counter()-t0:.2f}s\n")

# ---------------------------------------------------------------------------
# Timing loop
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)

times_bfe   = []
times_multi = []
times_nfw   = []

print(f"{'N':>8}  {'compile (ms)':>13}  {'warmup2 (ms)':>13}  {'benchmark (ms)':>15}  "
      f"{'Agama Multi (ms)':>17}  {'Agama NFW (ms)':>15}")
print("-" * 90)

for N in N_vals:
    # Random points: log-uniform r, uniform on sphere
    r     = np.exp(rng.uniform(np.log(0.1), np.log(20.0), N))
    cos_t = rng.uniform(-1.0, 1.0, N)
    sin_t = np.sqrt(1.0 - cos_t**2)
    phi_a = rng.uniform(0.0, 2.0 * np.pi, N)
    x_np  = r * sin_t * np.cos(phi_a)
    y_np  = r * sin_t * np.sin(phi_a)
    z_np  = r * cos_t
    xyz   = np.column_stack([x_np, y_np, z_np])

    x_j = jnp.array(x_np)
    y_j = jnp.array(y_np)
    z_j = jnp.array(z_np)

    # ── bfeax warmup: time each call explicitly to verify JIT behaviour ───
    # Call 1: includes XLA compilation — should be much slower than call 2.
    t0 = time.perf_counter()
    out = accel_fn(x_j, y_j, z_j)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - t0

    # Call 2: compiled code, hardware primed — this is our "floor".
    t0 = time.perf_counter()
    out = accel_fn(x_j, y_j, z_j)
    jax.block_until_ready(out)
    t_warmup2 = time.perf_counter() - t0

    # Timed reps: should match t_warmup2 closely (no compilation overhead).
    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        out = accel_fn(x_j, y_j, z_j)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    times_bfe.append(np.median(ts))

    # ── Agama Multipole ───────────────────────────────────────────────────
    for _ in range(N_WARMUP):
        pot_agama_multi.force(xyz)
    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        pot_agama_multi.force(xyz)
        ts.append(time.perf_counter() - t0)
    times_multi.append(np.median(ts))

    # ── Agama exact NFW ───────────────────────────────────────────────────
    for _ in range(N_WARMUP):
        pot_agama_nfw.force(xyz)
    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        pot_agama_nfw.force(xyz)
        ts.append(time.perf_counter() - t0)
    times_nfw.append(np.median(ts))

    print(f"{N:>8d}  "
          f"{t_compile*1e3:>13.2f}  "
          f"{t_warmup2*1e3:>13.3f}  "
          f"{times_bfe[-1]*1e3:>15.3f}  "
          f"{times_multi[-1]*1e3:>17.3f}  "
          f"{times_nfw[-1]*1e3:>15.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

N_arr    = np.array(N_vals)
t_bfe    = np.array(times_bfe)
t_multi  = np.array(times_multi)
t_nfw    = np.array(times_nfw)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Force evaluation benchmark  —  NFW, l_max={L_MAX}, n_r={N_R}",
    fontsize=13, fontweight="bold",
)

# ── Left: wall clock time vs N ───────────────────────────────────────────
ax = axes[0]
ax.loglog(N_arr, t_bfe   * 1e3, "o-",  color="C0", lw=2,   ms=7, label="bfeax  (JAX jit+vmap)")
ax.loglog(N_arr, t_multi * 1e3, "s--", color="C1", lw=2,   ms=7, label="Agama Multipole (C++)")
ax.loglog(N_arr, t_nfw   * 1e3, "^:",  color="C2", lw=1.5, ms=7, label="Agama exact NFW (C++)")

# Reference lines: O(N) slopes
x_ref = np.array([N_arr[2], N_arr[-1]])
for ref_N, ref_t, col in [(N_arr[2], t_bfe[2],   "C0"),
                           (N_arr[2], t_multi[2], "C1"),
                           (N_arr[2], t_nfw[2],   "C2")]:
    ax.loglog(x_ref, ref_t * (x_ref / ref_N), color=col, lw=0.6, ls="-", alpha=0.35)

ax.set_xlabel("Number of force evaluations $N$", fontsize=11)
ax.set_ylabel("Wall clock time (ms)", fontsize=11)
ax.set_title("Wall clock time vs $N$")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

# Annotate the bfeax JIT overhead (flat region at small N)
ax.annotate(
    "JIT overhead\n(fixed cost)",
    xy=(N_arr[0], t_bfe[0] * 1e3),
    xytext=(N_arr[1] * 1.5, t_bfe[0] * 1e3 * 3),
    fontsize=8, color="C0",
    arrowprops=dict(arrowstyle="->", color="C0", lw=0.8),
)

# ── Right: throughput (evaluations per second) ───────────────────────────
ax = axes[1]
ax.loglog(N_arr, N_arr / t_bfe,   "o-",  color="C0", lw=2,   ms=7, label="bfeax  (JAX jit+vmap)")
ax.loglog(N_arr, N_arr / t_multi, "s--", color="C1", lw=2,   ms=7, label="Agama Multipole (C++)")
ax.loglog(N_arr, N_arr / t_nfw,   "^:",  color="C2", lw=1.5, ms=7, label="Agama exact NFW (C++)")

ax.set_xlabel("Number of force evaluations $N$", fontsize=11)
ax.set_ylabel("Throughput  (evaluations / second)", fontsize=11)
ax.set_title("Throughput vs $N$")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

# Shade the "bfeax wins" region if it exists
bfe_faster = N_arr[t_bfe < t_multi]
if len(bfe_faster) > 0:
    crossover = bfe_faster[0]
    ax.axvline(crossover, color="gray", ls="--", lw=1)
    ax.annotate(f"bfeax faster\n→ N ≥ {crossover:,}",
                xy=(crossover, ax.get_ylim()[0]),
                xytext=(crossover * 1.5, (N_arr / t_bfe).max() * 0.1),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

fig.tight_layout()
fig.savefig("benchmark_forces.png", dpi=150)
print("\nSaved: benchmark_forces.png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(f"\n{'N':>8}  {'bfeax':>10}  {'Agama×':>10}  {'speedup':>10}")
print("-" * 45)
for i, N in enumerate(N_vals):
    speedup = t_multi[i] / t_bfe[i]
    print(f"{N:>8d}  {t_bfe[i]*1e3:>9.3f}ms  {t_multi[i]*1e3:>9.3f}ms  {speedup:>9.2f}×")
