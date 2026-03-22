"""
Wall-clock benchmark: bfeax vs Agama force evaluations.

Both codes expand a **triaxial** NFW-like profile (axis ratios p=0.8, q=0.6)
with identical grid parameters:
    l_max=8, n_r=128, r_min=0.01, r_max=300

This is the realistic use-case: all spherical-harmonic modes are active,
so neither code can short-circuit on symmetry-zero modes.

Two competitors:
  - bfeax          : analytical force (spline + Y_lm derivatives, JIT-compiled)
  - Agama Multipole: same spherical-harmonic expansion, compiled C++

Protocol:
  1. Generate all point sets and pre-warm BOTH codes at every N (JIT
     compilation for each array shape happens here, entirely before timing).
  2. For each N, run N_REPS timed calls and report the median.
  3. jax.block_until_ready() ensures async XLA dispatch is fully complete
     before stopping the clock.
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

from bfeax import MultipoleExpansion

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L_MAX  = 8
N_R    = 128
R_MIN  = 0.01
R_MAX  = 300.0
N_WARMUP = 3    # warmup calls per N (all before any timing)
N_REPS   = 10   # timed repetitions (median reported)

# Triaxial axis ratios — ensures all (l,m) modes are populated
P_AXIS = 0.8
Q_AXIS = 0.6

N_vals = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000]

# ---------------------------------------------------------------------------
# Build expansions
# ---------------------------------------------------------------------------

print("Building bfeax expansion (triaxial NFW) ...")
t0 = time.perf_counter()
exp = MultipoleExpansion.from_spheroid(
    rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=1.0,
    p=P_AXIS, q=Q_AXIS,
    r_min=R_MIN, r_max=R_MAX, n_r=N_R, l_max=L_MAX,
)
print(f"  done in {time.perf_counter()-t0:.2f}s")

accel_fn = lambda x, y, z: exp.force(x, y, z)

print("Building Agama Multipole (triaxial NFW) ...")
t0 = time.perf_counter()
pot_agama_multi = agama.Potential(
    type='Multipole',
    density=agama.Density(
        type='Spheroid', densityNorm=1, scaleRadius=1,
        axisRatioY=P_AXIS, axisRatioZ=Q_AXIS,
        alpha=1, beta=3, gamma=1,
    ),
    lmax=L_MAX, gridSizeR=N_R,
    rmin=R_MIN, rmax=R_MAX,
)
print(f"  done in {time.perf_counter()-t0:.2f}s")

# ---------------------------------------------------------------------------
# Pre-generate all point sets
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
point_sets = {}
for N in N_vals:
    r     = np.exp(rng.uniform(np.log(0.1), np.log(20.0), N))
    cos_t = rng.uniform(-1.0, 1.0, N)
    sin_t = np.sqrt(1.0 - cos_t**2)
    phi_a = rng.uniform(0.0, 2.0 * np.pi, N)
    x_np  = r * sin_t * np.cos(phi_a)
    y_np  = r * sin_t * np.sin(phi_a)
    z_np  = r * cos_t
    point_sets[N] = {
        'xyz': np.column_stack([x_np, y_np, z_np]),
        'x_j': jnp.array(x_np),
        'y_j': jnp.array(y_np),
        'z_j': jnp.array(z_np),
    }

# ---------------------------------------------------------------------------
# Warmup: compile XLA for every array shape BEFORE any timing
# ---------------------------------------------------------------------------

print("\nWarming up (JIT compilation for all array shapes) ...")
t0 = time.perf_counter()
for N in N_vals:
    pts = point_sets[N]
    for _ in range(N_WARMUP):
        out = accel_fn(pts['x_j'], pts['y_j'], pts['z_j'])
        jax.block_until_ready(out)
        pot_agama_multi.force(pts['xyz'])
print(f"  done in {time.perf_counter()-t0:.1f}s\n")

# ---------------------------------------------------------------------------
# Timing loop (no compilation should happen here)
# ---------------------------------------------------------------------------

times_bfe   = []
times_multi = []

print(f"{'N':>8}  {'bfeax (ms)':>12}  {'Agama (ms)':>12}  {'bfeax/Agama':>12}")
print("-" * 52)

for N in N_vals:
    pts = point_sets[N]

    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        out = accel_fn(pts['x_j'], pts['y_j'], pts['z_j'])
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    times_bfe.append(np.median(ts))

    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        pot_agama_multi.force(pts['xyz'])
        ts.append(time.perf_counter() - t0)
    times_multi.append(np.median(ts))

    ratio = times_bfe[-1] / times_multi[-1]
    marker = " <-- bfeax faster" if ratio < 1.0 else ""
    print(f"{N:>8d}  "
          f"{times_bfe[-1]*1e3:>12.3f}  "
          f"{times_multi[-1]*1e3:>12.3f}  "
          f"{ratio:>12.2f}x{marker}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

N_arr   = np.array(N_vals)
t_bfe   = np.array(times_bfe)
t_multi = np.array(times_multi)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Force evaluation benchmark  —  triaxial NFW  (p={P_AXIS}, q={Q_AXIS}),  "
    f"l_max={L_MAX},  n_r={N_R}",
    fontsize=12, fontweight="bold",
)

# ── Left: wall clock time vs N ────────────────────────────────────────────
ax = axes[0]
ax.loglog(N_arr, t_bfe   * 1e3, "o-",  color="C0", lw=2, ms=7,
          label="bfeax  (JAX analytical)")
ax.loglog(N_arr, t_multi * 1e3, "s--", color="C1", lw=2, ms=7,
          label="Agama Multipole (C++)")

ax.set_xlabel("Number of force evaluations $N$", fontsize=11)
ax.set_ylabel("Wall clock time (ms)", fontsize=11)
ax.set_title("Wall clock time vs $N$")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

# ── Right: speedup ratio ─────────────────────────────────────────────────
ax = axes[1]
ratio = t_multi / t_bfe   # > 1 means bfeax is faster
ax.semilogx(N_arr, ratio, "o-", color="C0", lw=2, ms=7)
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.fill_between(N_arr, ratio, 1.0,
                where=(ratio >= 1.0), alpha=0.15, color="C0",
                label="bfeax faster")
ax.fill_between(N_arr, ratio, 1.0,
                where=(ratio < 1.0), alpha=0.15, color="C1",
                label="Agama faster")

ax.set_xlabel("Number of force evaluations $N$", fontsize=11)
ax.set_ylabel("Speedup  (Agama time / bfeax time)", fontsize=11)
ax.set_title("bfeax speedup over Agama")
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

# Annotate peak speedup
peak_idx = np.argmax(ratio)
if ratio[peak_idx] > 1.0:
    ax.annotate(
        f"{ratio[peak_idx]:.1f}x faster",
        xy=(N_arr[peak_idx], ratio[peak_idx]),
        xytext=(N_arr[peak_idx] * 3, ratio[peak_idx] * 1.15),
        fontsize=10, fontweight="bold", color="C0",
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.2),
    )

fig.tight_layout()
fig.savefig("benchmark_forces.png", dpi=150)
print("\nSaved: benchmark_forces.png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(f"\n{'N':>8}  {'bfeax':>10}  {'Agama':>10}  {'speedup':>10}")
print("-" * 45)
for i, N in enumerate(N_vals):
    speedup = t_multi[i] / t_bfe[i]
    print(f"{N:>8d}  {t_bfe[i]*1e3:>9.3f}ms  {t_multi[i]*1e3:>9.3f}ms  {speedup:>9.2f}x")
