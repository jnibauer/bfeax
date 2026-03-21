"""
Compare density reconstruction with and without sigma-factor tapering
on the triaxial NFW halo.

Side-by-side x-z slices:
  col 0: no taper  (l_max 6, 8, 10)
  col 1: sigma taper (same l_max)
  + error vs true density
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from bfe import MultipoleExpansion

# ── density ───────────────────────────────────────────────────────────────────
rho_s, r_s, q1, q2 = 1.0, 1.0, 0.8, 0.5

def rho_triaxial(x, y, z):
    m = jnp.clip(jnp.sqrt(x**2 + (y/q1)**2 + (z/q2)**2), 1e-30)
    return rho_s / (m/r_s * (1.0 + m/r_s)**2)

def rho_triaxial_np(x, y, z):
    m = np.maximum(np.sqrt(x**2 + (y/q1)**2 + (z/q2)**2), 1e-30)
    return rho_s / (m/r_s * (1.0 + m/r_s)**2)

R_MIN, R_MAX, N_R = 1e-2, 3e2, 128
L_MAX_VALS = [6, 8, 10]

# ── build both variants for each l_max ───────────────────────────────────────
print("Building expansions ...")
exps_plain  = {}
exps_sigma  = {}
for lm in L_MAX_VALS:
    exps_plain[lm] = MultipoleExpansion.from_density(
        rho_triaxial, R_MIN, R_MAX, N_R, l_max=lm, sigma_taper=False)
    exps_sigma[lm] = MultipoleExpansion.from_density(
        rho_triaxial, R_MIN, R_MAX, N_R, l_max=lm, sigma_taper=True)
    print(f"  l_max={lm} done")

# ── x-z evaluation grid ───────────────────────────────────────────────────────
ng  = 120
lim = 4.0
gv  = np.linspace(-lim, lim, ng)
XX, ZZ = np.meshgrid(gv, gv)
YY = np.zeros_like(XX)

XJ = jnp.array(XX.ravel())
ZJ = jnp.array(ZZ.ravel())
YJ = jnp.array(YY.ravel())

rho_true = rho_triaxial_np(XX, YY, ZZ)

def get_rho(exp):
    return np.array(exp.density(XJ, YJ, ZJ)).reshape(ng, ng)

def log_err(rho_rec):
    rel = np.abs((rho_rec - rho_true) / (rho_true + 1e-30))
    return np.log10(rel + 1e-10)

def log_rho(rho):
    return np.log10(np.clip(rho, 1e-4 * np.nanmax(rho), None))

# ── print median errors ───────────────────────────────────────────────────────
print(f"\n{'l_max':>6}  {'plain median':>14}  {'sigma median':>14}  {'reduction':>10}")
print("-" * 52)
for lm in L_MAX_VALS:
    rp = get_rho(exps_plain[lm])
    rs = get_rho(exps_sigma[lm])
    ep = np.median(np.abs((rp - rho_true) / rho_true))
    es = np.median(np.abs((rs - rho_true) / rho_true))
    print(f"  {lm:4d}  {ep:14.3e}  {es:14.3e}  {ep/es:10.2f}x")

# ── figure: 3 l_max × (plain | sigma | err_plain | err_sigma) ────────────────
ncols = 4
nrows = len(L_MAX_VALS)
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
fig.suptitle(
    f"Triaxial NFW x-z: no taper vs sigma taper  (q1={q1}, q2={q2})",
    fontsize=13, fontweight="bold",
)

kw = dict(origin="lower", extent=[-lim, lim, -lim, lim], aspect="equal")

for row, lm in enumerate(L_MAX_VALS):
    rp = get_rho(exps_plain[lm])
    rs = get_rho(exps_sigma[lm])

    vmin_rho = log_rho(rho_true).min()
    vmax_rho = log_rho(rho_true).max()

    # col 0: plain density
    ax = axes[row, 0]
    im = ax.imshow(log_rho(rp), vmin=vmin_rho, vmax=vmax_rho,
                   cmap="inferno", **kw)
    ax.contour(log_rho(rp), 10, extent=[-lim,lim,-lim,lim],
               colors="w", linewidths=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"plain  l_max={lm}"); ax.set_xlabel("x"); ax.set_ylabel("z")

    # col 1: sigma density
    ax = axes[row, 1]
    im = ax.imshow(log_rho(rs), vmin=vmin_rho, vmax=vmax_rho,
                   cmap="inferno", **kw)
    ax.contour(log_rho(rs), 10, extent=[-lim,lim,-lim,lim],
               colors="w", linewidths=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"sigma  l_max={lm}"); ax.set_xlabel("x")

    # col 2: plain error
    ax = axes[row, 2]
    im = ax.imshow(log_err(rp), vmin=-4, vmax=0,
                   cmap="RdYlGn_r", **kw)
    plt.colorbar(im, ax=ax)
    med = np.median(np.abs((rp-rho_true)/rho_true))
    ax.set_title(f"plain err  med={med:.1e}"); ax.set_xlabel("x")

    # col 3: sigma error
    ax = axes[row, 3]
    im = ax.imshow(log_err(rs), vmin=-4, vmax=0,
                   cmap="RdYlGn_r", **kw)
    plt.colorbar(im, ax=ax)
    med = np.median(np.abs((rs-rho_true)/rho_true))
    ax.set_title(f"sigma err  med={med:.1e}"); ax.set_xlabel("x")

plt.tight_layout()
plt.savefig("sigma_comparison.png", dpi=150)
print("\nSaved: sigma_comparison.png")
