"""
benchmark_build.py — bfeax fused build+force vs Agama build+force.

Varies the flattening q at fixed other parameters.  Measures the wall-clock
time to build the expansion AND evaluate the force at a single point.

bfeax strategy
--------------
  A single @jax.jit function captures the full pipeline:
    q  ->  _spheroid_core()  ->  phi_coeffs  ->  force at (x,y,z)

  After warmup (one XLA compilation, not timed), each subsequent call with a
  new q just re-executes the compiled graph — no Python overhead, no rebuild.

Agama strategy
--------------
  agama.Potential(...) rebuilds the C++ internal structure from scratch each
  call.  There is no way to fuse or pre-compile against varying parameters.

Protocol
--------
  1. Warm up bfeax JIT for N_WARMUP calls (compilation happens here).
  2. For each q: run N_REPS timed bfeax calls; report median.
  3. For each q: run N_REPS timed Agama (build + force) calls; report median.
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import agama
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
agama.setUnits(mass=1, length=1, velocity=1)

from bfe.potential import _spheroid_core, _lm_keys
from bfe.grid import make_radial_grid
from bfe.sph_harm import _alp_recurrence, _normalization


# ── Fixed parameters ──────────────────────────────────────────────────────────

RHO0  = 1.0
ALPHA = 1.0   # inner slope
BETA  = 3.0   # outer slope
GAMMA = 1.0   # transition
A     = 1.0   # scale radius
P     = 0.8   # y/x axis ratio (fixed)

L_MAX = 8
N_R   = 128
R_MIN = 0.01
R_MAX = 300.0

N_WARMUP = 3
N_REPS   = 20

# Single query point
XQ, YQ, ZQ = 2.0, 1.5, 1.0

# Flattening values to sweep
Q_VALS = np.round(np.linspace(0.3, 1.0, 15), 4)


# ── Precomputed constants (captured as XLA constants in the JIT closure) ──────

_r_grid     = make_radial_grid(N_R, R_MIN, R_MAX)
_log_r_grid = jnp.log(_r_grid)

_n_theta = L_MAX + 2
_n_phi   = 2 * L_MAX + 1

# Use triaxial symmetry throughout: even l, even m >= 0.
# For q close to 1 the odd modes have near-zero coefficients anyway.
_lm_keys_list  = _lm_keys(L_MAX, "triaxial")
_lm_keys_tuple = tuple(_lm_keys_list)

# Precompute normalization factors (Python-level, static wrt JIT)
_l_needed = max(l for l, _ in _lm_keys_list)
_m_needed = max(abs(m) for _, m in _lm_keys_list)
_norms = [
    math.sqrt(2) * _normalization(l, abs(m)) if abs(m) > 0 else _normalization(l, 0)
    for l, m in _lm_keys_list
]


# ── Force evaluation from traced phi_coeffs ───────────────────────────────────

def _force_from_coeffs(phi_coeffs, log_r_k, x, y, z):
    """
    Evaluate (Fx, Fy, Fz) at a single point from traced spline coefficients.

    phi_coeffs : (a, b, c, d) each shape (n_modes, n_r-1) — traced JAX arrays
    log_r_k    : (n_r,) — log-spaced knot positions, treated as constant
    x, y, z    : scalars

    Uses all modes in _lm_keys_list (no amplitude-based pruning, since
    coefficient values are traced and can't be inspected at compile time).
    """
    a_c, b_c, c_c, d_c = phi_coeffs

    # Coordinates
    r_xy_sq       = x * x + y * y
    r_safe        = jnp.maximum(jnp.sqrt(r_xy_sq + z * z), 1e-30)
    r_xy          = jnp.sqrt(r_xy_sq)
    r_xy_safe     = jnp.maximum(r_xy, 1e-30)
    cos_theta     = z / r_safe
    sin_theta     = r_xy / r_safe
    sin_theta_safe = jnp.maximum(sin_theta, 1e-30)
    cos_phi = jnp.where(r_xy > 1e-30, x / r_xy_safe, 1.0)
    sin_phi = jnp.where(r_xy > 1e-30, y / r_xy_safe, 0.0)
    log_r   = jnp.log(r_safe)
    inv_r   = 1.0 / r_safe

    # Spline lookup
    idx = jnp.clip(
        jnp.searchsorted(log_r_k, log_r, side="right") - 1,
        0, log_r_k.shape[0] - 2,
    )
    dt  = log_r - log_r_k[idx]
    dt2 = dt * dt

    phi_vals  = a_c[:, idx] + b_c[:, idx] * dt + c_c[:, idx] * dt2 + d_c[:, idx] * (dt2 * dt)
    phi_dlogr = b_c[:, idx] + 2.0 * c_c[:, idx] * dt + 3.0 * d_c[:, idx] * dt2

    # Legendre and trig recurrences — unrolled at trace time (l/m are static)
    P = _alp_recurrence(_l_needed, cos_theta)
    cos_m = [jnp.ones_like(cos_phi)]
    sin_m = [jnp.zeros_like(cos_phi)]
    for mm in range(1, _m_needed + 1):
        cos_m.append(cos_m[mm - 1] * cos_phi - sin_m[mm - 1] * sin_phi)
        sin_m.append(sin_m[mm - 1] * cos_phi + cos_m[mm - 1] * sin_phi)

    Y_list, dYdth_list, dYdphi_st_list = [], [], []
    for i_mode, (l, ms) in enumerate(_lm_keys_list):
        m   = abs(ms)
        fac = _norms[i_mode]

        trig       = cos_m[m] if ms > 0 else (sin_m[m] if ms < 0 else 1.0)
        trig_other = sin_m[m] if ms > 0 else (cos_m[m] if ms < 0 else None)

        Plm = P[l][m]
        Y_list.append(fac * Plm * trig)

        if l == 0:
            dYdth_list.append(jnp.zeros_like(cos_theta))
        else:
            P_prev = P[l - 1][m] if m <= l - 1 else 0.0
            dYdth_list.append(
                fac * (l * cos_theta * Plm - (l + m) * P_prev) / sin_theta_safe * trig
            )

        if ms == 0:
            dYdphi_st_list.append(jnp.zeros_like(cos_phi))
        else:
            dYdphi_st_list.append(-ms * fac * Plm / sin_theta_safe * trig_other)

    Y_arr         = jnp.stack(Y_list)
    dYdth_arr     = jnp.stack(dYdth_list)
    dYdphi_st_arr = jnp.stack(dYdphi_st_list)

    dPhi_dr      = jnp.dot(phi_dlogr, Y_arr)         * inv_r
    dPhi_dth_r   = jnp.dot(phi_vals,  dYdth_arr)     * inv_r
    dPhi_dphi_rs = jnp.dot(phi_vals,  dYdphi_st_arr) * inv_r

    Fx = -(dPhi_dr * sin_theta * cos_phi + dPhi_dth_r * cos_theta * cos_phi - dPhi_dphi_rs * sin_phi)
    Fy = -(dPhi_dr * sin_theta * sin_phi + dPhi_dth_r * cos_theta * sin_phi + dPhi_dphi_rs * cos_phi)
    Fz = -(dPhi_dr * cos_theta           - dPhi_dth_r * sin_theta)

    return Fx, Fy, Fz


# ── Fused JIT function ────────────────────────────────────────────────────────

@jax.jit
def build_and_force(q, x, y, z):
    """
    Full pipeline under one JIT: spheroid with flattening q → force at (x,y,z).

    All other params (rho0, alpha, beta, gamma, a, p, grid, modes) are
    captured as closure constants and folded into the XLA graph at compile time.
    Rerunning with a new q value requires no Python overhead or recompilation.
    """
    phi_coeffs, _, _, _ = _spheroid_core(
        _r_grid,
        RHO0, ALPHA, BETA, GAMMA, A, P, q,
        1e30, 0.0,
        L_MAX, _n_theta, _n_phi, _lm_keys_tuple,
    )
    return _force_from_coeffs(phi_coeffs, _log_r_grid, x, y, z)


# ── Warmup ────────────────────────────────────────────────────────────────────

xj = jnp.array(XQ)
yj = jnp.array(YQ)
zj = jnp.array(ZQ)

print(f"Warming up bfeax fused JIT  (l_max={L_MAX}, n_r={N_R}) ...")
t0 = time.perf_counter()
for _ in range(N_WARMUP):
    out = build_and_force(jnp.array(Q_VALS[0]), xj, yj, zj)
    jax.block_until_ready(out)
print(f"  done in {time.perf_counter() - t0:.2f}s\n")


# ── Benchmark ─────────────────────────────────────────────────────────────────

print(f"{'q':>6}  {'bfeax (ms)':>12}  {'Agama (ms)':>12}  {'speedup':>10}")
print("-" * 48)

times_bfe   = []
times_agama = []

for q_val in Q_VALS:
    q_j = jnp.array(q_val)

    # bfeax: fused build + force
    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        out = build_and_force(q_j, xj, yj, zj)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    times_bfe.append(np.median(ts))

    # Agama: build Potential + evaluate force
    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        pot = agama.Potential(
            type="Multipole",
            density=agama.Density(
                type="Spheroid", densityNorm=RHO0, scaleRadius=A,
                axisRatioY=P, axisRatioZ=float(q_val),
                alpha=ALPHA, beta=BETA, gamma=GAMMA,
            ),
            lmax=L_MAX, gridSizeR=N_R, rmin=R_MIN, rmax=R_MAX,
        )
        pot.force([[XQ, YQ, ZQ]])
        ts.append(time.perf_counter() - t0)
    times_agama.append(np.median(ts))

    speedup = times_agama[-1] / times_bfe[-1]
    print(f"{q_val:>6.3f}  {times_bfe[-1]*1e3:>12.2f}  {times_agama[-1]*1e3:>12.2f}  {speedup:>9.1f}x")


# ── Plot ──────────────────────────────────────────────────────────────────────

q_arr   = np.array(Q_VALS)
t_bfe   = np.array(times_bfe)
t_agama = np.array(times_agama)
speedup = t_agama / t_bfe

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Build + force benchmark  —  spheroid NFW (p={P}),  "
    f"l_max={L_MAX},  n_r={N_R}\n"
    f"Query point: ({XQ}, {YQ}, {ZQ})",
    fontsize=11, fontweight="bold",
)

ax = axes[0]
ax.plot(q_arr, t_bfe   * 1e3, "o-",  color="C0", lw=2, ms=7, label="bfeax  (fused JIT)")
ax.plot(q_arr, t_agama * 1e3, "s--", color="C1", lw=2, ms=7, label="Agama  (build + force)")
ax.set_xlabel("Flattening $q$", fontsize=11)
ax.set_ylabel("Wall clock time (ms)", fontsize=11)
ax.set_title("Build + force time vs $q$")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(q_arr, speedup, "o-", color="C0", lw=2, ms=7)
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.fill_between(q_arr, speedup, 1.0, where=(speedup >= 1), alpha=0.15, color="C0", label="bfeax faster")
ax.fill_between(q_arr, speedup, 1.0, where=(speedup  < 1), alpha=0.15, color="C1", label="Agama faster")
ax.set_xlabel("Flattening $q$", fontsize=11)
ax.set_ylabel("Speedup  (Agama time / bfeax time)", fontsize=11)
ax.set_title("bfeax speedup over Agama")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

peak_idx = int(np.argmax(speedup))
if speedup[peak_idx] > 1.0:
    ax.annotate(
        f"{speedup[peak_idx]:.0f}x faster",
        xy=(q_arr[peak_idx], speedup[peak_idx]),
        xytext=(q_arr[peak_idx] + 0.05, speedup[peak_idx] * 1.1),
        fontsize=10, fontweight="bold", color="C0",
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.2),
    )

fig.tight_layout()
fig.savefig("benchmark_build.png", dpi=150)
print("\nSaved: benchmark_build.png")
