"""
Convergence sweep for n_theta and n_phi quadrature defaults.

Uses triaxial NFW (q1=0.8, q2=0.5) as the reference — the hardest
standard case in the test suite.

Reports density reconstruction error at l_max=8 for a grid of
(n_theta, n_phi) values.  Current defaults are marked with *.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from bfe import MultipoleExpansion

# ── Triaxial NFW ──────────────────────────────────────────────────────────────
rho_s, r_s = 1.0, 1.0
q1, q2 = 0.8, 0.5

def rho_triaxial(x, y, z):
    m = jnp.clip(jnp.sqrt(x**2 + (y / q1)**2 + (z / q2)**2), 1e-30)
    return rho_s / (m / r_s * (1.0 + m / r_s) ** 2)

# Reference evaluation points
key = jax.random.PRNGKey(42)
pts = jax.random.normal(key, (3, 2000)) * 2.0
xp, yp, zp = pts
rp = np.sqrt(np.array(xp)**2 + np.array(yp)**2 + np.array(zp)**2)
mask = (rp > 0.05) & (rp < 50.0)
xp, yp, zp = xp[mask], yp[mask], zp[mask]
rho_true = np.array(rho_triaxial(xp, yp, zp))

R_MIN, R_MAX, N_R = 1e-2, 3e2, 128
L_MAX = 8

# Current defaults for reference
default_n_theta = 3 * (L_MAX + 2)   # 30
default_n_phi   = 4 * L_MAX + 7     # 39

# Theoretical minimums:
#   GL n_theta points integrates polynomials of degree 2*n_theta-1 exactly.
#   To resolve Y_lm (degree l_max in cos θ), strict minimum is ceil((l_max+1)/2) = 5.
#   For phi: Nyquist for cos(l_max * phi) requires n_phi > 2*l_max = 16.

n_theta_vals = [
    (L_MAX + 2) // 2 + 1,    # ~strict polynomial min = 6
    L_MAX + 2,                 # 1x margin = 10
    2 * (L_MAX + 2),           # 2x margin = 20
    3 * (L_MAX + 2),           # current default = 30  (*)
    4 * (L_MAX + 2),           # 4x over-sampled = 40
]

n_phi_vals = [
    2 * L_MAX + 1,   # Nyquist minimum = 17
    2 * L_MAX + 3,   # just above Nyquist = 19
    3 * L_MAX + 4,   # ~1.5x margin = 28
    4 * L_MAX + 7,   # current default = 39  (*)
    6 * L_MAX + 7,   # 3x margin = 55
]

print(f"l_max={L_MAX}  current defaults: n_theta={default_n_theta}, n_phi={default_n_phi}\n")

# ── Sweep n_theta (fix n_phi at current default) ──────────────────────────────
print("=== n_theta sweep  (n_phi fixed at default) ===")
print(f"{'n_theta':>8}  {'median_err':>11}  {'90pct_err':>10}  {'build_s':>8}  {'pts_eval':>8}")
print("-" * 54)

for n_theta in n_theta_vals:
    t0 = time.perf_counter()
    exp = MultipoleExpansion.from_density(
        rho_triaxial, R_MIN, R_MAX, N_R, l_max=L_MAX,
        n_theta=n_theta, n_phi=default_n_phi,
    )
    dt = time.perf_counter() - t0
    rho_rec = np.array(exp.density(xp, yp, zp))
    rel_err = np.abs((rho_rec - rho_true) / rho_true)
    marker = " *" if n_theta == default_n_theta else "  "
    print(f"{n_theta:>7}{marker}  {np.median(rel_err):11.3e}  {np.percentile(rel_err, 90):10.3e}  {dt:8.2f}  {n_theta * default_n_phi:8d}")

# ── Sweep n_phi (fix n_theta at current default) ──────────────────────────────
print(f"\n=== n_phi sweep  (n_theta fixed at default) ===")
print(f"{'n_phi':>7}  {'median_err':>11}  {'90pct_err':>10}  {'build_s':>8}  {'pts_eval':>8}")
print("-" * 54)

for n_phi in n_phi_vals:
    t0 = time.perf_counter()
    exp = MultipoleExpansion.from_density(
        rho_triaxial, R_MIN, R_MAX, N_R, l_max=L_MAX,
        n_theta=default_n_theta, n_phi=n_phi,
    )
    dt = time.perf_counter() - t0
    rho_rec = np.array(exp.density(xp, yp, zp))
    rel_err = np.abs((rho_rec - rho_true) / rho_true)
    marker = " *" if n_phi == default_n_phi else "  "
    print(f"{n_phi:>6}{marker}  {np.median(rel_err):11.3e}  {np.percentile(rel_err, 90):10.3e}  {dt:8.2f}  {default_n_theta * n_phi:8d}")

# ── 2D grid: a few (n_theta, n_phi) combinations ─────────────────────────────
print(f"\n=== 2D sweep: (n_theta, n_phi) combinations ===")
print(f"{'n_theta':>8}  {'n_phi':>6}  {'median_err':>11}  {'90pct_err':>10}  {'build_s':>8}  {'total_pts':>10}")
print("-" * 64)

combos = [
    (2 * (L_MAX + 2), 2 * L_MAX + 3),   # aggressive reduction
    (2 * (L_MAX + 2), 3 * L_MAX + 4),   # moderate reduction
    (3 * (L_MAX + 2), 2 * L_MAX + 3),   # reduce phi only
    (3 * (L_MAX + 2), 3 * L_MAX + 4),   # reduce phi mildly
    (default_n_theta, default_n_phi),    # current default  (*)
]

for n_theta, n_phi in combos:
    t0 = time.perf_counter()
    exp = MultipoleExpansion.from_density(
        rho_triaxial, R_MIN, R_MAX, N_R, l_max=L_MAX,
        n_theta=n_theta, n_phi=n_phi,
    )
    dt = time.perf_counter() - t0
    rho_rec = np.array(exp.density(xp, yp, zp))
    rel_err = np.abs((rho_rec - rho_true) / rho_true)
    marker = " *" if (n_theta, n_phi) == (default_n_theta, default_n_phi) else "  "
    print(f"{n_theta:>7}{marker}  {n_phi:>6}  {np.median(rel_err):11.3e}  {np.percentile(rel_err, 90):10.3e}  {dt:8.2f}  {n_theta * n_phi:10d}")
