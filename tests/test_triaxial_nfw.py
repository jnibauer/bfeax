"""
Test: triaxial NFW halo.

    rho(m) = rho_s / ( m/r_s * (1 + m/r_s)^2 )
    m^2    = x^2 + (y/q1)^2 + (z/q2)^2      (triaxial, q1>q2)

Axes: x (major) > y (intermediate, q1=0.8) > z (minor, q2=0.5)

Tests:
  1. Density reconstruction  rho_rec(x,y,z) vs rho_input(x,y,z)
  2. Which (l,m) modes carry the power
  3. Potential along each principal axis
  4. 2D slices of both density and potential
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from bfeax import MultipoleExpansion

# ── Triaxial NFW ──────────────────────────────────────────────────────────────
rho_s = 1.0
r_s   = 1.0
q1    = 0.8   # y-axis ratio  (major=x, intermediate=y, minor=z)
q2    = 0.5   # z-axis ratio

def m_ellip(x, y, z):
    return jnp.sqrt(x**2 + (y / q1)**2 + (z / q2)**2)

def rho_triaxial(x, y, z):
    m = jnp.clip(m_ellip(x, y, z), 1e-30)
    return rho_s / (m / r_s * (1.0 + m / r_s) ** 2)

def rho_triaxial_np(x, y, z):
    m = np.sqrt(x**2 + (y / q1)**2 + (z / q2)**2)
    m = np.maximum(m, 1e-30)
    return rho_s / (m / r_s * (1.0 + m / r_s) ** 2)

# ── Build expansions at several l_max ────────────────────────────────────────
R_MIN, R_MAX, N_R = 1e-2, 3e2, 128

l_max_vals = [2, 4, 6, 8, 10]
exps = {}
print("=== Building expansions ===")
for lm in l_max_vals:
    t0 = time.perf_counter()
    exps[lm] = MultipoleExpansion.from_density(
        rho_triaxial, R_MIN, R_MAX, N_R, l_max=lm
    )
    print(f"  l_max={lm}  build={time.perf_counter()-t0:.2f}s")

# ── 1. Density reconstruction error ──────────────────────────────────────────
print("\n=== Density reconstruction error ===")
key = jax.random.PRNGKey(42)
pts = jax.random.normal(key, (3, 2000)) * 2.0   # cluster around r~2
xp, yp, zp = pts[0], pts[1], pts[2]

rp = np.sqrt(np.array(xp)**2 + np.array(yp)**2 + np.array(zp)**2)
# exclude points very close to origin or beyond grid
mask = (rp > 0.05) & (rp < 50.0)
xp, yp, zp = xp[mask], yp[mask], zp[mask]
rp = rp[mask]

rho_true = np.array(rho_triaxial(xp, yp, zp))

print(f"\n{'l_max':>6}  {'median_relerr':>14}  {'90pct_relerr':>13}  {'max_relerr':>11}")
print("-" * 52)
for lm in l_max_vals:
    rho_rec = np.array(exps[lm].density(xp, yp, zp))
    rel_err = np.abs((rho_rec - rho_true) / rho_true)
    print(f"  {lm:4d}  {np.median(rel_err):14.3e}  {np.percentile(rel_err,90):13.3e}  {np.max(rel_err):11.3e}")

# ── 2. Mode amplitudes at r=1 ─────────────────────────────────────────────────
print("\n=== rho_lm amplitudes at r=1 (l_max=8) ===")
amps = exps[8].rho_lm_amplitudes(r_val=1.0)
# Sort by amplitude descending, print top 20
sorted_amps = sorted(amps.items(), key=lambda kv: kv[1], reverse=True)
total_power = sum(v**2 for _, v in amps.items())
print(f"{'(l,m)':>8}  {'|rho_lm|':>12}  {'frac power':>11}")
print("-" * 36)
cum = 0.0
for (l, m), a in sorted_amps[:20]:
    cum += a**2 / total_power
    print(f"  ({l:2d},{m:3d})  {a:12.4e}  {cum:11.4f}")

# ── 3. Potential along principal axes ────────────────────────────────────────
print("\n=== Potential along principal axes (l_max=8) ===")
r_ax = np.exp(np.linspace(np.log(0.1), np.log(20), 60))
zr   = np.zeros_like(r_ax)
rj   = jnp.array(r_ax)
zj   = jnp.zeros_like(rj)

phi_x = np.array(exps[10](rj, zj, zj))   # along x
phi_y = np.array(exps[10](zj, rj, zj))   # along y
phi_z = np.array(exps[10](zj, zj, rj))   # along z

print(f"{'r':>6}  {'Phi(x)':>10}  {'Phi(y)':>10}  {'Phi(z)':>10}  {'aniso_xz':>10}")
for i in [5, 15, 25, 35, 45, 55]:
    aniso = abs(phi_x[i] - phi_z[i]) / abs(phi_x[i])
    print(f"{r_ax[i]:6.2f}  {phi_x[i]:10.4f}  {phi_y[i]:10.4f}  {phi_z[i]:10.4f}  {aniso:10.4f}")

# ── 4. Slices ─────────────────────────────────────────────────────────────────
print("\n=== Generating slice plots ===")
exp8 = exps[10]

ng  = 100
lim = 4.0
gv  = np.linspace(-lim, lim, ng)

# x-y slice (z=0)
XX_xy, YY_xy = np.meshgrid(gv, gv)
ZZ_xy = np.zeros_like(XX_xy)

# x-z slice (y=0)
XX_xz, ZZ_xz = np.meshgrid(gv, gv)
YY_xz = np.zeros_like(XX_xz)

def eval_grid(exp, Xg, Yg, Zg):
    Xj = jnp.array(Xg.ravel())
    Yj = jnp.array(Yg.ravel())
    Zj = jnp.array(Zg.ravel())
    phi_g = np.array(exp(Xj, Yj, Zj)).reshape(ng, ng)
    rho_g = np.array(exp.density(Xj, Yj, Zj)).reshape(ng, ng)
    return phi_g, rho_g

phi_xy, rho_xy = eval_grid(exp8, XX_xy, YY_xy, ZZ_xy)
phi_xz, rho_xz = eval_grid(exp8, XX_xz, YY_xz, ZZ_xz)

# True density for comparison
rho_true_xy = rho_triaxial_np(XX_xy, YY_xy, ZZ_xy)
rho_true_xz = rho_triaxial_np(XX_xz, YY_xz, ZZ_xz)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    f"Triaxial NFW  (q1={q1}, q2={q2}, l_max=10, n_r={N_R})",
    fontsize=13, fontweight="bold"
)

def _log_density(rho):
    return np.log10(np.clip(rho, 1e-4 * np.nanmax(rho), None))

kw_rho = dict(cmap="inferno", origin="lower",
              extent=[-lim, lim, -lim, lim], aspect="equal")
kw_phi = dict(cmap="viridis", origin="lower",
              extent=[-lim, lim, -lim, lim], aspect="equal")

# Row 0: x-y plane (z=0)
ax = axes[0, 0]
im = ax.imshow(_log_density(rho_true_xy), **kw_rho)
ax.contour(_log_density(rho_true_xy), levels=12,
           extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
plt.colorbar(im, ax=ax); ax.set_title("log10 rho_true  (x-y)"); ax.set_xlabel("x"); ax.set_ylabel("y")

ax = axes[0, 1]
im = ax.imshow(_log_density(rho_xy), **kw_rho)
ax.contour(_log_density(rho_xy), levels=12,
           extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
plt.colorbar(im, ax=ax); ax.set_title("log10 rho_rec  (x-y)"); ax.set_xlabel("x")

ax = axes[0, 2]
rel_err_xy = np.abs((rho_xy - rho_true_xy) / (rho_true_xy + 1e-30))
im = ax.imshow(np.log10(rel_err_xy + 1e-10), vmin=-4, vmax=0,
               cmap="RdYlGn_r", **{k: v for k, v in kw_rho.items()
                                    if k not in ("cmap",)})
plt.colorbar(im, ax=ax); ax.set_title("log10 |rho_err|  (x-y)"); ax.set_xlabel("x")

ax = axes[0, 3]
im = ax.imshow(phi_xy, **kw_phi)
ax.contour(phi_xy, levels=15,
           extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
plt.colorbar(im, ax=ax); ax.set_title("Phi  (x-y)"); ax.set_xlabel("x")

# Row 1: x-z plane (y=0)
ax = axes[1, 0]
im = ax.imshow(_log_density(rho_true_xz), **kw_rho)
ax.contour(_log_density(rho_true_xz), levels=12,
           extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
plt.colorbar(im, ax=ax); ax.set_title("log10 rho_true  (x-z)"); ax.set_xlabel("x"); ax.set_ylabel("z")

ax = axes[1, 1]
im = ax.imshow(_log_density(rho_xz), **kw_rho)
ax.contour(_log_density(rho_xz), levels=12,
           extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
plt.colorbar(im, ax=ax); ax.set_title("log10 rho_rec  (x-z)"); ax.set_xlabel("x")

ax = axes[1, 2]
rel_err_xz = np.abs((rho_xz - rho_true_xz) / (rho_true_xz + 1e-30))
im = ax.imshow(np.log10(rel_err_xz + 1e-10), vmin=-4, vmax=0,
               cmap="RdYlGn_r", **{k: v for k, v in kw_rho.items()
                                    if k not in ("cmap",)})
plt.colorbar(im, ax=ax); ax.set_title("log10 |rho_err|  (x-z)"); ax.set_xlabel("x")

ax = axes[1, 3]
im = ax.imshow(phi_xz, **kw_phi)
ax.contour(phi_xz, levels=15,
           extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
plt.colorbar(im, ax=ax); ax.set_title("Phi  (x-z)"); ax.set_xlabel("x")

plt.tight_layout()
plt.savefig("triaxial_nfw.png", dpi=150)
print("Saved: triaxial_nfw.png")


# ── 4b. x-z error map side-by-side for each l_max ────────────────────────────
print("Generating x-z convergence figure ...")
lm_show = [4, 6, 8, 10]
fig2, axes2 = plt.subplots(2, len(lm_show), figsize=(5 * len(lm_show), 9))
fig2.suptitle(
    f"Triaxial NFW x-z slice: density error convergence  (q1={q1}, q2={q2})",
    fontsize=12, fontweight="bold"
)

for col, lm in enumerate(lm_show):
    rho_g = np.array(exps[lm].density(
        jnp.array(XX_xz.ravel()),
        jnp.array(YY_xz.ravel()),
        jnp.array(ZZ_xz.ravel()),
    )).reshape(ng, ng)
    rel_err_g = np.abs((rho_g - rho_true_xz) / (rho_true_xz + 1e-30))
    med = np.median(rel_err_g)

    # Top row: reconstructed density
    ax = axes2[0, col]
    im = ax.imshow(_log_density(rho_g), **kw_rho)
    ax.contour(_log_density(rho_g), levels=12,
               extent=[-lim, lim, -lim, lim], colors="w", linewidths=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"rho_rec  l_max={lm}")
    ax.set_xlabel("x"); ax.set_ylabel("z" if col == 0 else "")

    # Bottom row: error map
    ax = axes2[1, col]
    im = ax.imshow(np.log10(rel_err_g + 1e-10), vmin=-4, vmax=0,
                   cmap="RdYlGn_r", **{k: v for k, v in kw_rho.items()
                                        if k not in ("cmap",)})
    plt.colorbar(im, ax=ax)
    ax.set_title(f"log10|err|  median={med:.1e}")
    ax.set_xlabel("x"); ax.set_ylabel("z" if col == 0 else "")

plt.tight_layout()
plt.savefig("triaxial_xz_convergence.png", dpi=150)
print("Saved: triaxial_xz_convergence.png")

# ── 5. Density convergence with l_max ────────────────────────────────────────
print("\n=== Density convergence with l_max at fixed point (2, 1.5, 1) ===")
xpt = jnp.array([2.0]);  ypt = jnp.array([1.5]);  zpt = jnp.array([1.0])
rho_pt_true = float(rho_triaxial(xpt, ypt, zpt)[0])
print(f"  rho_true = {rho_pt_true:.6f}")
for lm in l_max_vals:
    rho_pt_rec = float(exps[lm].density(xpt, ypt, zpt)[0])
    err = abs(rho_pt_rec - rho_pt_true) / rho_pt_true
    print(f"  l_max={lm:2d}  rho_rec={rho_pt_rec:.6f}  relerr={err:.3e}")
