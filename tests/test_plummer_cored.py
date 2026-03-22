"""
Diagnostic test: Plummer sphere (cored density profile).

The Plummer profile is the canonical *cored* density — it is finite and
flat as r -> 0, in contrast to cusped profiles like NFW.  This script
checks whether the pipeline handles a cored inner boundary correctly and
produces four diagnostic figures:

  1. Radial profiles: rho_true vs rho_rec, and Phi_true vs Phi_bfe.
  2. Relative error vs radius for both density and potential.
  3. l_max convergence: how quickly does the BFE improve?
  4. 2-D slice through the density residual (rho_rec - rho_true) / rho_true
     in the x-z plane — should be noise-flat for a spherical profile.

Analytical formulae (G = 1, M = 1, b = 1):
    rho(r) = (3 / 4*pi) * (1 + r^2)^{-5/2}
    Phi(r) = -1 / sqrt(r^2 + 1)
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
# Plummer definitions
# ---------------------------------------------------------------------------

b = 1.0   # scale radius
M = 1.0   # total mass (G = 1)


def rho_plummer(x, y, z):
    r2 = x**2 + y**2 + z**2
    return 3.0 * M / (4.0 * jnp.pi * b**3) * (1.0 + r2 / b**2) ** (-2.5)


def phi_plummer(x, y, z):
    r2 = x**2 + y**2 + z**2
    return -M / jnp.sqrt(r2 + b**2)


# Vectorised wrappers for plotting on 2-D grids
def rho_vec(x, y, z):
    return jax.vmap(jax.vmap(rho_plummer))(x, y, z)

def phi_vec(x, y, z):
    return jax.vmap(jax.vmap(phi_plummer))(x, y, z)


# ---------------------------------------------------------------------------
# Build expansion
# ---------------------------------------------------------------------------

N_R   = 128
L_MAX = 6    # Plummer is spherical -> l=0 dominates; l_max=6 is generous
R_MIN = 1e-2
R_MAX = 100.0

print(f"Building MultipoleExpansion  l_max={L_MAX}  n_r={N_R} ...")
t0 = time.time()
exp = MultipoleExpansion.from_density(
    rho_plummer,
    r_min=R_MIN, r_max=R_MAX,
    n_r=N_R, l_max=L_MAX,
)
print(f"  done in {time.time()-t0:.2f}s")

# ---------------------------------------------------------------------------
# Figure 1: Radial profiles and errors
# ---------------------------------------------------------------------------

r_plot = jnp.exp(jnp.linspace(jnp.log(R_MIN), jnp.log(R_MAX * 0.9), 300))
x_plot = r_plot
y_plot = jnp.zeros_like(r_plot)
z_plot = jnp.zeros_like(r_plot)

rho_true = jax.vmap(rho_plummer)(x_plot, y_plot, z_plot)
phi_true = jax.vmap(phi_plummer)(x_plot, y_plot, z_plot)
rho_bfe  = jax.vmap(exp.density)(x_plot, y_plot, z_plot)
phi_bfe  = jax.vmap(exp)(x_plot, y_plot, z_plot)

rho_relerr = jnp.abs((rho_bfe - rho_true) / rho_true)
phi_relerr = jnp.abs((phi_bfe - phi_true) / jnp.abs(phi_true))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Plummer sphere (cored profile) — BFE diagnostics", fontsize=13, fontweight="bold")

r_np = np.array(r_plot)

# Top-left: density profiles
ax = axes[0, 0]
ax.loglog(r_np, np.array(rho_true), "k-",  lw=2,   label=r"$\rho_{\rm true}$")
ax.loglog(r_np, np.array(rho_bfe),  "C0--", lw=1.5, label=r"$\rho_{\rm BFE}$")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\rho(r)$")
ax.set_title("Density profile")
ax.legend()
ax.set_xlim(R_MIN, R_MAX * 0.9)

# Top-right: potential profiles
ax = axes[0, 1]
ax.semilogx(r_np, np.array(phi_true), "k-",  lw=2,   label=r"$\Phi_{\rm true}$")
ax.semilogx(r_np, np.array(phi_bfe),  "C1--", lw=1.5, label=r"$\Phi_{\rm BFE}$")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\Phi(r)$")
ax.set_title("Potential profile")
ax.legend()
ax.set_xlim(R_MIN, R_MAX * 0.9)

# Bottom-left: density relative error
ax = axes[1, 0]
ax.loglog(r_np, np.array(rho_relerr), "C0-", lw=1.5)
ax.axhline(1e-2, color="gray", ls="--", lw=1, label="1%")
ax.axhline(1e-3, color="gray", ls=":",  lw=1, label="0.1%")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$|\Delta\rho\,/\,\rho_{\rm true}|$")
ax.set_title(f"Density relative error  (median {float(jnp.median(rho_relerr)):.2e})")
ax.legend(fontsize=9)
ax.set_xlim(R_MIN, R_MAX * 0.9)
ax.set_ylim(1e-8, 1e-0)

# Bottom-right: potential relative error
ax = axes[1, 1]
ax.loglog(r_np, np.array(phi_relerr), "C1-", lw=1.5)
ax.axhline(1e-2, color="gray", ls="--", lw=1, label="1%")
ax.axhline(1e-3, color="gray", ls=":",  lw=1, label="0.1%")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$|\Delta\Phi\,/\,|\Phi_{\rm true}||$")
ax.set_title(f"Potential relative error  (median {float(jnp.median(phi_relerr)):.2e})")
ax.legend(fontsize=9)
ax.set_xlim(R_MIN, R_MAX * 0.9)
ax.set_ylim(1e-8, 1e-0)

fig.tight_layout()
fig.savefig("plummer_profiles.png", dpi=150)
print("Saved: plummer_profiles.png")

# ---------------------------------------------------------------------------
# Figure 2: l_max convergence
# ---------------------------------------------------------------------------

l_max_vals = [0, 2, 4, 6]
r_test = jnp.exp(jnp.linspace(jnp.log(R_MIN), jnp.log(50.0), 200))
x_t = r_test; y_t = jnp.zeros_like(r_test); z_t = jnp.zeros_like(r_test)
rho_t = jax.vmap(rho_plummer)(x_t, y_t, z_t)
phi_t = jax.vmap(phi_plummer)(x_t, y_t, z_t)

fig2, (ax_rho, ax_phi) = plt.subplots(1, 2, figsize=(11, 4))
fig2.suptitle("l_max convergence — Plummer sphere", fontsize=12, fontweight="bold")

rho_stats = []
phi_stats = []

for l in l_max_vals:
    e = MultipoleExpansion.from_density(
        rho_plummer,
        r_min=R_MIN, r_max=R_MAX,
        n_r=N_R, l_max=l,
    )
    rho_rec = jax.vmap(e.density)(x_t, y_t, z_t)
    phi_rec = jax.vmap(e)(x_t, y_t, z_t)
    re_rho  = jnp.abs((rho_rec - rho_t) / rho_t)
    re_phi  = jnp.abs((phi_rec - phi_t) / jnp.abs(phi_t))
    rho_stats.append(float(jnp.median(re_rho)))
    phi_stats.append(float(jnp.median(re_phi)))
    lbl = rf"$\ell_{{\max}}={l}$"
    ax_rho.loglog(np.array(r_test), np.array(re_rho), lw=1.4, label=lbl)
    ax_phi.loglog(np.array(r_test), np.array(re_phi), lw=1.4, label=lbl)

for ax, title, ylabel in [
    (ax_rho, "Density rel. error vs $r$", r"$|\Delta\rho/\rho|$"),
    (ax_phi, "Potential rel. error vs $r$", r"$|\Delta\Phi/|\Phi||$"),
]:
    ax.axhline(1e-2, color="gray", ls="--", lw=0.8)
    ax.axhline(1e-3, color="gray", ls=":",  lw=0.8)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(R_MIN, 50.0)
    ax.set_ylim(1e-10, 1.0)

fig2.tight_layout()
fig2.savefig("plummer_convergence.png", dpi=150)
print("Saved: plummer_convergence.png")

# ---------------------------------------------------------------------------
# Figure 3: 2-D density residual slice in x-z plane
# ---------------------------------------------------------------------------

n_grid = 120
lim = 8.0
xi = np.linspace(-lim, lim, n_grid)
zi = np.linspace(-lim, lim, n_grid)
XX, ZZ = np.meshgrid(xi, zi, indexing="ij")
YY = np.zeros_like(XX)

XX_j = jnp.array(XX); YY_j = jnp.array(YY); ZZ_j = jnp.array(ZZ)

rho_2d_true = np.array(rho_vec(XX_j, YY_j, ZZ_j))
rho_2d_bfe  = np.array(
    jax.vmap(jax.vmap(exp.density))(XX_j, YY_j, ZZ_j)
)
phi_2d_true = np.array(phi_vec(XX_j, YY_j, ZZ_j))
phi_2d_bfe  = np.array(
    jax.vmap(jax.vmap(exp))(XX_j, YY_j, ZZ_j)
)

rho_res = (rho_2d_bfe - rho_2d_true) / rho_2d_true
phi_res = (phi_2d_bfe - phi_2d_true) / np.abs(phi_2d_true)

# Mask points outside the grid domain
mask = (np.sqrt(XX**2 + ZZ**2) < R_MIN) | (np.sqrt(XX**2 + ZZ**2) > R_MAX * 0.9)
rho_res[mask] = np.nan
phi_res[mask] = np.nan

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("2-D residuals in x-z plane — Plummer sphere", fontsize=12, fontweight="bold")

vmax_rho = max(1e-4, np.nanpercentile(np.abs(rho_res), 99))
vmax_phi = max(1e-6, np.nanpercentile(np.abs(phi_res), 99))

for ax, data, vmax, title, label in [
    (axes3[0], rho_res, vmax_rho,
     r"Density residual $(\rho_{\rm BFE}-\rho_{\rm true})/\rho_{\rm true}$",
     r"relative error"),
    (axes3[1], phi_res, vmax_phi,
     r"Potential residual $(\Phi_{\rm BFE}-\Phi_{\rm true})/|\Phi_{\rm true}|$",
     r"relative error"),
]:
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(xi, zi, data.T, cmap="RdBu_r", norm=norm, rasterized=True)
    cb = fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=9)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$z$")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    # Overplot a circle at r = b (scale radius)
    theta_circ = np.linspace(0, 2*np.pi, 300)
    ax.plot(b*np.cos(theta_circ), b*np.sin(theta_circ), "k--", lw=0.8, label=r"$r=b$")
    ax.legend(fontsize=8)

fig3.tight_layout()
fig3.savefig("plummer_2d_residual.png", dpi=150)
print("Saved: plummer_2d_residual.png")

# ---------------------------------------------------------------------------
# Figure 4: inner slope diagnostic — does cusp subtraction handle a core?
# ---------------------------------------------------------------------------

# Show rho_lm(r) for the (0,0) monopole, and the fitted power-law background
key = (0, 0)
log_r_knots, _, _, rho_alphas, rho_As = exp._stacked
mode_idx = exp._lm_keys.index(key)
alpha_lm = rho_alphas[mode_idx]
A_lm     = rho_As[mode_idx]

r_knots = np.exp(np.array(log_r_knots))
rho_00_true = np.array(
    jax.vmap(rho_plummer)(jnp.array(r_knots),
                          jnp.zeros(len(r_knots)),
                          jnp.zeros(len(r_knots)))
) / np.sqrt(4 * np.pi)   # Y_00 = 1/sqrt(4pi)

alpha_val = float(alpha_lm)
A_val     = float(A_lm)
bg_vals   = A_val * r_knots ** alpha_val  # fitted background

fig4, (ax_sl, ax_bg) = plt.subplots(1, 2, figsize=(11, 4))
fig4.suptitle(r"Inner slope diagnostic — $\rho_{00}(r)$ for Plummer", fontsize=12, fontweight="bold")

ax_sl.loglog(r_knots, rho_00_true,  "ko", ms=3, label=r"$\rho_{00}$ on grid")
ax_sl.loglog(r_knots, np.abs(bg_vals), "C2--", lw=1.5,
             label=rf"fitted bg: $A\,r^{{\alpha}}$,  $\alpha={alpha_val:.3f}$")
ax_sl.axvline(b, color="gray", ls=":", lw=1, label=r"$r=b$ (scale radius)")
ax_sl.set_xlabel(r"$r$"); ax_sl.set_ylabel(r"$|\rho_{00}(r)|$")
ax_sl.set_title("Monopole coefficient and fitted background")
ax_sl.legend(fontsize=9)
ax_sl.set_xlim(R_MIN, R_MAX)

residual_vals = rho_00_true - bg_vals
ax_bg.semilogx(r_knots, residual_vals, "C0o-", ms=3, lw=1, label="residual (spline input)")
ax_bg.axhline(0, color="k", lw=0.8)
ax_bg.axvline(b, color="gray", ls=":", lw=1, label=r"$r=b$")
ax_bg.set_xlabel(r"$r$"); ax_bg.set_ylabel(r"$\rho_{00} - A\,r^\alpha$")
ax_bg.set_title("Spline residual after background subtraction")
ax_bg.legend(fontsize=9)
ax_bg.set_xlim(R_MIN, R_MAX)

fig4.tight_layout()
fig4.savefig("plummer_inner_slope.png", dpi=150)
print("Saved: plummer_inner_slope.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n=== Summary ===")
print(f"  Profile         : Plummer  (b={b}, M={M}, G=1)")
print(f"  Grid            : n_r={N_R}, r=[{R_MIN}, {R_MAX}], l_max={L_MAX}")
print(f"  Inner slope α   : {alpha_val:.4f}  (expected ≈ 0 for a core)")
print(f"  Density  median |Δρ/ρ|   : {float(jnp.median(rho_relerr)):.2e}")
print(f"  Potential median |ΔΦ/Φ|  : {float(jnp.median(phi_relerr)):.2e}")

assert float(jnp.median(rho_relerr)) < 1e-3, "Density error too large"
assert float(jnp.median(phi_relerr)) < 1e-3, "Potential error too large"
print("\nPASSED")
