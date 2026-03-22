"""
Diagnostic test: Miyamoto-Nagai disk (highly flattened system).

    rho(R,z) = (b^2 M / 4pi) * [ a R^2 + (a + 3 w)(a + w)^2 ]
               / [ (R^2 + (a + w)^2)^(5/2)  *  w^3 ]
    Phi(R,z) = -G M / sqrt(R^2 + (a + w)^2)
    w        = sqrt(z^2 + b^2)

With G=1, M=1, a=1, b=0.1  (axis ratio b/a = 0.1 — strongly flattened).

Key questions tested:
  1. How does accuracy depend on l_max?  A disk needs many even-l modes.
  2. Is accuracy uniform across angles, or does the midplane vs polar axis differ?
  3. Are the default angular quadrature settings enough, or does the sharp
     equatorial peak require more GL nodes?
  4. Which (l,m) modes carry the most power?
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

from bfeax import MultipoleExpansion

# ---------------------------------------------------------------------------
# Miyamoto-Nagai
# ---------------------------------------------------------------------------

a = 1.0
b = 0.1   # b/a = 0.1 -> strongly flattened disk
M = 1.0   # G = 1


def _w(z):
    return jnp.sqrt(z**2 + b**2)


def rho_mn(x, y, z):
    R2 = x**2 + y**2
    w  = _w(z)
    num = b**2 * M / (4.0 * jnp.pi) * (
        a * R2 + (a + 3.0 * w) * (a + w)**2
    )
    den = (R2 + (a + w)**2) ** 2.5 * w**3
    return num / den


def phi_mn(x, y, z):
    R2 = x**2 + y**2
    return -M / jnp.sqrt(R2 + (a + _w(z))**2)


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------

R_MIN = 1e-2
R_MAX = 50.0
N_R   = 128
L_MAX = 10   # main expansion; convergence sweep goes higher

# ---------------------------------------------------------------------------
# Figure 1: l_max convergence — median error vs l_max
#           separately for equatorial (z≈0) and polar (R≈0) points
# ---------------------------------------------------------------------------

print("=== l_max convergence sweep ===")

# Sample points in two regimes
rng = np.random.default_rng(42)
n_pts = 300
r_samp = np.exp(rng.uniform(np.log(0.1), np.log(20.0), n_pts))

# Equatorial: small |cos θ|
phi_ang = rng.uniform(0, 2 * np.pi, n_pts)
x_eq = r_samp * np.cos(phi_ang)
y_eq = r_samp * np.sin(phi_ang)
z_eq = r_samp * rng.uniform(-0.05, 0.05, n_pts)   # near midplane

# Polar: large |cos θ|
x_pol = r_samp * rng.uniform(-0.1, 0.1, n_pts)
y_pol = r_samp * rng.uniform(-0.1, 0.1, n_pts)
z_pol = np.sign(rng.uniform(-1, 1, n_pts)) * np.sqrt(
    np.clip(r_samp**2 - x_pol**2 - y_pol**2, 0, None)
)

def rel_err(pred, true):
    return float(jnp.median(jnp.abs((pred - true) / (jnp.abs(true) + 1e-30))))

x_eq_j = jnp.array(x_eq); y_eq_j = jnp.array(y_eq); z_eq_j = jnp.array(z_eq)
x_pol_j = jnp.array(x_pol); y_pol_j = jnp.array(y_pol); z_pol_j = jnp.array(z_pol)

rho_eq_true  = jax.vmap(rho_mn)(x_eq_j, y_eq_j, z_eq_j)
phi_eq_true  = jax.vmap(phi_mn)(x_eq_j, y_eq_j, z_eq_j)
rho_pol_true = jax.vmap(rho_mn)(x_pol_j, y_pol_j, z_pol_j)
phi_pol_true = jax.vmap(phi_mn)(x_pol_j, y_pol_j, z_pol_j)

l_max_vals = [2, 4, 6, 8, 10, 14, 18]
results = []

for lm in l_max_vals:
    t0 = time.time()
    exp = MultipoleExpansion.from_density(
        rho_mn, r_min=R_MIN, r_max=R_MAX, n_r=N_R, l_max=lm,
    )
    dt = time.time() - t0

    re_rho_eq  = rel_err(jax.vmap(exp.density)(x_eq_j, y_eq_j, z_eq_j),  rho_eq_true)
    re_phi_eq  = rel_err(jax.vmap(exp)(x_eq_j, y_eq_j, z_eq_j),           phi_eq_true)
    re_rho_pol = rel_err(jax.vmap(exp.density)(x_pol_j, y_pol_j, z_pol_j), rho_pol_true)
    re_phi_pol = rel_err(jax.vmap(exp)(x_pol_j, y_pol_j, z_pol_j),         phi_pol_true)

    results.append((lm, re_rho_eq, re_phi_eq, re_rho_pol, re_phi_pol, dt))
    print(f"  l_max={lm:2d}  rho_eq={re_rho_eq:.2e}  phi_eq={re_phi_eq:.2e}"
          f"  rho_pol={re_rho_pol:.2e}  phi_pol={re_phi_pol:.2e}  t={dt:.1f}s")

results = np.array(results)

fig1, axes = plt.subplots(1, 2, figsize=(11, 4))
fig1.suptitle(
    f"Miyamoto-Nagai disk  (a={a}, b={b})  —  l_max convergence",
    fontsize=12, fontweight="bold"
)

for ax, col_eq, col_pol, ylabel, title in [
    (axes[0], 1, 3, r"median $|\Delta\rho/\rho|$", "Density error"),
    (axes[1], 2, 4, r"median $|\Delta\Phi/\Phi|$", "Potential error"),
]:
    ax.semilogy(results[:, 0], results[:, col_eq],  "o-", label="equatorial (|cos θ| < 0.05)")
    ax.semilogy(results[:, 0], results[:, col_pol], "s--", label="polar (|cos θ| ≈ 1)")
    ax.axhline(1e-2, color="gray", ls="--", lw=0.8, label="1%")
    ax.axhline(1e-3, color="gray", ls=":",  lw=0.8, label="0.1%")
    ax.set_xlabel(r"$\ell_{\max}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)

fig1.tight_layout()
fig1.savefig("disk_convergence.png", dpi=150)
print("Saved: disk_convergence.png")

# ---------------------------------------------------------------------------
# Build the main expansion at L_MAX for remaining figures
# ---------------------------------------------------------------------------

print(f"\nBuilding main expansion l_max={L_MAX} ...")
t0 = time.time()
exp = MultipoleExpansion.from_density(
    rho_mn, r_min=R_MIN, r_max=R_MAX, n_r=N_R, l_max=L_MAX,
)
print(f"  done in {time.time()-t0:.2f}s")

# ---------------------------------------------------------------------------
# Figure 2: Profiles along equatorial (z=0) and polar (x=y=0) axes
# ---------------------------------------------------------------------------

r_prof = jnp.exp(jnp.linspace(jnp.log(R_MIN * 1.5), jnp.log(R_MAX * 0.8), 300))

# Equatorial: x=r, y=z=0
rho_eq_prof_true = jax.vmap(rho_mn)(r_prof, jnp.zeros_like(r_prof), jnp.zeros_like(r_prof))
phi_eq_prof_true = jax.vmap(phi_mn)(r_prof, jnp.zeros_like(r_prof), jnp.zeros_like(r_prof))
rho_eq_prof_bfe  = jax.vmap(exp.density)(r_prof, jnp.zeros_like(r_prof), jnp.zeros_like(r_prof))
phi_eq_prof_bfe  = jax.vmap(exp)(r_prof, jnp.zeros_like(r_prof), jnp.zeros_like(r_prof))

# Polar: x=y=0, z=r
rho_pol_prof_true = jax.vmap(rho_mn)(jnp.zeros_like(r_prof), jnp.zeros_like(r_prof), r_prof)
phi_pol_prof_true = jax.vmap(phi_mn)(jnp.zeros_like(r_prof), jnp.zeros_like(r_prof), r_prof)
rho_pol_prof_bfe  = jax.vmap(exp.density)(jnp.zeros_like(r_prof), jnp.zeros_like(r_prof), r_prof)
phi_pol_prof_bfe  = jax.vmap(exp)(jnp.zeros_like(r_prof), jnp.zeros_like(r_prof), r_prof)

r_np = np.array(r_prof)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle(
    f"Miyamoto-Nagai profiles  (a={a}, b={b}, l_max={L_MAX})",
    fontsize=12, fontweight="bold"
)

# Density profiles
ax = axes2[0, 0]
ax.loglog(r_np, np.array(rho_eq_prof_true),  "k-",   lw=2,   label=r"true  (equatorial)")
ax.loglog(r_np, np.array(rho_eq_prof_bfe),   "C0--", lw=1.5, label=r"BFE   (equatorial)")
ax.loglog(r_np, np.array(rho_pol_prof_true), "k:",   lw=2,   label=r"true  (polar)")
ax.loglog(r_np, np.array(rho_pol_prof_bfe),  "C1--", lw=1.5, label=r"BFE   (polar)")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\rho(r)$")
ax.set_title("Density profiles"); ax.legend(fontsize=9)

# Potential profiles
ax = axes2[0, 1]
ax.semilogx(r_np, np.array(phi_eq_prof_true),  "k-",   lw=2,   label=r"true  (equatorial)")
ax.semilogx(r_np, np.array(phi_eq_prof_bfe),   "C0--", lw=1.5, label=r"BFE   (equatorial)")
ax.semilogx(r_np, np.array(phi_pol_prof_true), "k:",   lw=2,   label=r"true  (polar)")
ax.semilogx(r_np, np.array(phi_pol_prof_bfe),  "C1--", lw=1.5, label=r"BFE   (polar)")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\Phi(r)$")
ax.set_title("Potential profiles"); ax.legend(fontsize=9)

# Density relative error
ax = axes2[1, 0]
ax.loglog(r_np, np.abs(np.array(rho_eq_prof_bfe  - rho_eq_prof_true)  / np.array(rho_eq_prof_true)),
          "C0-", lw=1.5, label="equatorial")
ax.loglog(r_np, np.abs(np.array(rho_pol_prof_bfe - rho_pol_prof_true) / np.array(rho_pol_prof_true)),
          "C1-", lw=1.5, label="polar")
ax.axhline(1e-2, color="gray", ls="--", lw=0.8)
ax.axhline(1e-3, color="gray", ls=":",  lw=0.8)
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$|\Delta\rho/\rho|$")
ax.set_title("Density relative error"); ax.legend(fontsize=9)
ax.set_ylim(1e-6, 2)

# Potential relative error
ax = axes2[1, 1]
ax.loglog(r_np, np.abs(np.array(phi_eq_prof_bfe  - phi_eq_prof_true)  / np.abs(np.array(phi_eq_prof_true))),
          "C0-", lw=1.5, label="equatorial")
ax.loglog(r_np, np.abs(np.array(phi_pol_prof_bfe - phi_pol_prof_true) / np.abs(np.array(phi_pol_prof_true))),
          "C1-", lw=1.5, label="polar")
ax.axhline(1e-2, color="gray", ls="--", lw=0.8)
ax.axhline(1e-3, color="gray", ls=":",  lw=0.8)
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$|\Delta\Phi/\Phi|$")
ax.set_title("Potential relative error"); ax.legend(fontsize=9)
ax.set_ylim(1e-6, 2)

fig2.tight_layout()
fig2.savefig("disk_profiles.png", dpi=150)
print("Saved: disk_profiles.png")

# ---------------------------------------------------------------------------
# Figure 3: 2-D density residual in x-z plane
# ---------------------------------------------------------------------------

n_grid = 150
lim_xz = 6.0
xi = np.linspace(-lim_xz, lim_xz, n_grid)
zi = np.linspace(-lim_xz, lim_xz, n_grid)
XX, ZZ = np.meshgrid(xi, zi, indexing="ij")
YY = np.zeros_like(XX)

XX_j = jnp.array(XX); YY_j = jnp.array(YY); ZZ_j = jnp.array(ZZ)

rho_2d_true = np.array(jax.vmap(jax.vmap(rho_mn))(XX_j, YY_j, ZZ_j))
phi_2d_true = np.array(jax.vmap(jax.vmap(phi_mn))(XX_j, YY_j, ZZ_j))
rho_2d_bfe  = np.array(jax.vmap(jax.vmap(exp.density))(XX_j, YY_j, ZZ_j))
phi_2d_bfe  = np.array(jax.vmap(jax.vmap(exp))(XX_j, YY_j, ZZ_j))

rho_res_2d = (rho_2d_bfe - rho_2d_true) / (rho_2d_true + 1e-30)
phi_res_2d = (phi_2d_bfe - phi_2d_true) / np.abs(phi_2d_true + 1e-30)

# Mask inside r_min and outside r_max
r_2d = np.sqrt(XX**2 + ZZ**2)
mask = (r_2d < R_MIN) | (r_2d > R_MAX * 0.9)
rho_res_2d[mask] = np.nan
phi_res_2d[mask] = np.nan

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    f"2-D residuals in x-z plane  (a={a}, b={b}, l_max={L_MAX})",
    fontsize=12, fontweight="bold"
)

for ax, data, title, label in [
    (axes3[0], rho_res_2d,
     r"Density: $(\rho_{\rm BFE}-\rho_{\rm true})/\rho_{\rm true}$",
     "relative error"),
    (axes3[1], phi_res_2d,
     r"Potential: $(\Phi_{\rm BFE}-\Phi_{\rm true})/|\Phi_{\rm true}|$",
     "relative error"),
]:
    vmax = np.nanpercentile(np.abs(data), 97)
    vmax = max(vmax, 1e-4)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(xi, zi, data.T, cmap="RdBu_r", norm=norm, rasterized=True)
    cb = fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=9)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$z$")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    # Mark the disk scale radius and scale height
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(a * np.cos(theta_c), a * np.sin(theta_c), "k--", lw=0.8, label=r"$r=a$")
    ax.axhline( b, color="gray", ls=":", lw=0.8, label=r"$z=\pm b$")
    ax.axhline(-b, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=8, loc="upper right")

fig3.tight_layout()
fig3.savefig("disk_2d_residual.png", dpi=150)
print("Saved: disk_2d_residual.png")

# ---------------------------------------------------------------------------
# Figure 4: Mode amplitude spectrum at r = 1 (near the disk scale radius)
# ---------------------------------------------------------------------------

r_probe = 1.0
log_r_probe = jnp.log(jnp.array([r_probe]))
lm_vals = exp._eval_rho_lm(jnp.array([r_probe]), log_r_probe)

# Collect amplitudes; group by l
l_vals_all = sorted(set(l for l, m in lm_vals.keys()))
amp_by_l = {}
for l in l_vals_all:
    modes = {m: float(jnp.abs(lm_vals[(l, m)][0])) for m in range(-l, l+1) if (l, m) in lm_vals}
    amp_by_l[l] = max(modes.values()) if modes else 0.0

total_power = sum(v**2 for v in amp_by_l.values())

fig4, (ax_amp, ax_frac) = plt.subplots(1, 2, figsize=(11, 4))
fig4.suptitle(
    f"Mode amplitudes at r={r_probe}  (a={a}, b={b}, l_max={L_MAX})",
    fontsize=12, fontweight="bold"
)

ls = list(amp_by_l.keys())
amps = [amp_by_l[l] for l in ls]
colors = ["C0" if l % 2 == 0 else "C3" for l in ls]

ax_amp.bar(ls, amps, color=colors)
ax_amp.set_xlabel(r"$\ell$"); ax_amp.set_ylabel(r"$\max_m |\rho_{\ell m}(r=1)|$")
ax_amp.set_title("Mode amplitudes (blue=even, red=odd)")
ax_amp.set_yscale("log")

# Cumulative fractional power (even modes only, since disk is z-symmetric)
even_ls = [l for l in ls if l % 2 == 0]
even_amps = np.array([amp_by_l[l] for l in even_ls])
cum_power = np.cumsum(even_amps**2) / sum(even_amps**2 + 1e-300)

ax_frac.plot(even_ls, cum_power, "o-", color="C0")
ax_frac.axhline(0.99,  color="gray", ls="--", lw=0.8, label="99%")
ax_frac.axhline(0.999, color="gray", ls=":",  lw=0.8, label="99.9%")
ax_frac.set_xlabel(r"$\ell$ (even modes only)")
ax_frac.set_ylabel("Cumulative fractional power")
ax_frac.set_title("How many modes needed?")
ax_frac.legend(fontsize=9)
ax_frac.set_ylim(0, 1.05)

fig4.tight_layout()
fig4.savefig("disk_modes.png", dpi=150)
print("Saved: disk_modes.png")

# ---------------------------------------------------------------------------
# Figure 5: Quadrature sensitivity — does n_theta matter?
# ---------------------------------------------------------------------------

print("\n=== Quadrature sensitivity at l_max=10 ===")

n_theta_vals = [
    3 * (L_MAX + 2),          # default
    5 * (L_MAX + 2),          # 5/3x default
    8 * (L_MAX + 2),          # 8/3x default
]
labels = ["default (3×)", "5×", "8×"]

fig5, axes5 = plt.subplots(1, 2, figsize=(11, 4))
fig5.suptitle(
    f"Quadrature sensitivity: n_theta at l_max={L_MAX}",
    fontsize=12, fontweight="bold"
)

r_check = jnp.exp(jnp.linspace(jnp.log(0.15), jnp.log(15.0), 200))
x_check = r_check; y_check = jnp.zeros_like(r_check); z_check = jnp.zeros_like(r_check)
rho_check_true = jax.vmap(rho_mn)(x_check, y_check, z_check)
phi_check_true = jax.vmap(phi_mn)(x_check, y_check, z_check)

for n_th, lbl in zip(n_theta_vals, labels):
    print(f"  n_theta={n_th} ({lbl}) ...", end=" ", flush=True)
    t0 = time.time()
    e = MultipoleExpansion.from_density(
        rho_mn, r_min=R_MIN, r_max=R_MAX, n_r=N_R, l_max=L_MAX,
        n_theta=n_th,
    )
    print(f"{time.time()-t0:.1f}s")
    rho_bfe_q = jax.vmap(e.density)(x_check, y_check, z_check)
    phi_bfe_q = jax.vmap(e)(x_check, y_check, z_check)
    axes5[0].loglog(np.array(r_check),
                    np.abs(np.array((rho_bfe_q - rho_check_true) / rho_check_true)),
                    lw=1.5, label=lbl)
    axes5[1].loglog(np.array(r_check),
                    np.abs(np.array((phi_bfe_q - phi_check_true) / jnp.abs(phi_check_true))),
                    lw=1.5, label=lbl)

for ax, ylabel, title in [
    (axes5[0], r"$|\Delta\rho/\rho|$",   "Density error vs quadrature (equatorial)"),
    (axes5[1], r"$|\Delta\Phi/\Phi|$", "Potential error vs quadrature (equatorial)"),
]:
    ax.axhline(1e-2, color="gray", ls="--", lw=0.8)
    ax.axhline(1e-3, color="gray", ls=":",  lw=0.8)
    ax.set_xlabel(r"$r$"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(1e-7, 2)

fig5.tight_layout()
fig5.savefig("disk_quadrature.png", dpi=150)
print("Saved: disk_quadrature.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"""
=== Summary ===
  Profile : Miyamoto-Nagai  a={a}, b={b}  (b/a = {b/a})
  Grid    : n_r={N_R}, r=[{R_MIN}, {R_MAX}], l_max={L_MAX}

  l_max convergence (equatorial / polar):""")
for row in results:
    lm, re_rho_eq, re_phi_eq, re_rho_pol, re_phi_pol, dt = row
    print(f"    l_max={int(lm):2d}  rho: {re_rho_eq:.2e} / {re_rho_pol:.2e}"
          f"   phi: {re_phi_eq:.2e} / {re_phi_pol:.2e}")
