"""
Test: flattened (oblate) NFW halo.

    rho(m) = rho_s / ( m/r_s * (1 + m/r_s)^2 )
    m^2    = x^2 + y^2 + (z/q)^2       (oblate flattening q < 1)

Because q != 1, the density has no l=0-only representation; higher even-l
terms (l=2, 4, ...) are needed.  We compare the JAX BFE against a
scipy.integrate.quad reference computed to high precision.
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

jax.config.update("jax_enable_x64", True)

from bfe import MultipoleExpansion

# ── Parameters ────────────────────────────────────────────────────────────────
rho_s = 1.0
r_s   = 1.0
q     = 0.6          # oblate flattening; 1 = sphere
G     = 1.0

R_MIN, R_MAX, N_R = 1e-2, 3e2, 128

# ── Density ───────────────────────────────────────────────────────────────────
def rho_nfw(x, y, z):
    m = jnp.sqrt(x**2 + y**2 + (z / q)**2)
    m = jnp.clip(m, 1e-30)
    return rho_s / (m / r_s * (1.0 + m / r_s) ** 2)

def rho_nfw_np(x, y, z):
    m = np.sqrt(x**2 + y**2 + (z / q)**2)
    m = np.maximum(m, 1e-30)
    return rho_s / (m / r_s * (1.0 + m / r_s) ** 2)

# ── Scipy reference ───────────────────────────────────────────────────────────
# For an axisymmetric density only m=0 modes survive, so:
#   rho_l0(r) = 2*pi * int_0^pi rho(r,theta) * Y_l0(theta) * sin(theta) dtheta
#   Phi_l0(r) = -4*pi*G/(2l+1) * [r^{-(l+1)} * I_in(r) + r^l * I_out(r)]

from scipy.special import lpmv  # associated Legendre

def _Y_l0_np(l, cos_theta):
    """Un-normalised P_l(cos_theta) with full Y_l0 normalisation."""
    norm = math.sqrt((2 * l + 1) / (4 * math.pi))
    return norm * lpmv(0, l, cos_theta)

def rho_l0_scipy(r_val, l):
    """Angular projection integral at one radius."""
    def integrand(cos_th):
        sin_th = math.sqrt(max(1.0 - cos_th**2, 0.0))
        z = r_val * cos_th
        rxy = r_val * sin_th
        return rho_nfw_np(rxy, 0.0, z) * _Y_l0_np(l, cos_th) * 2 * math.pi
    val, _ = integrate.quad(integrand, -1.0, 1.0, limit=200, epsabs=1e-10, epsrel=1e-10)
    return val

def phi_l0_scipy_at_r(r_arr, l):
    """High-accuracy Phi_l0(r) via scipy quad."""
    # Build rho_l0 on a fine internal grid, then integrate
    r_fine = np.exp(np.linspace(np.log(R_MIN * 0.5), np.log(3e3), 2048))
    rho_fine = np.array([rho_l0_scipy(r, l) for r in r_fine])

    phi_out = np.zeros(len(r_arr))
    for i, r in enumerate(r_arr):
        # Interior: int_0^r rho_l0(r') r'^{l+2} dr'
        mask_in = r_fine <= r
        if mask_in.sum() > 1:
            I_in = np.trapz(rho_fine[mask_in] * r_fine[mask_in]**(l+2),
                            r_fine[mask_in])
        else:
            I_in = 0.0
        # Exterior: int_r^inf rho_l0(r') r'^{1-l} dr'
        mask_out = r_fine >= r
        if mask_out.sum() > 1:
            I_out = np.trapz(rho_fine[mask_out] * r_fine[mask_out]**(1-l),
                             r_fine[mask_out])
        else:
            I_out = 0.0
        phi_out[i] = -4 * math.pi * G / (2*l+1) * (
            r**(-(l+1)) * I_in + r**l * I_out
        )
    return phi_out

def phi_scipy_reference(r_arr, l_max, theta_arr):
    """
    Full potential at (r, theta) summed over even l.
    theta in [0, pi].
    """
    phi = np.zeros((len(r_arr), len(theta_arr)))
    for l in range(0, l_max + 1, 2):     # only even l for m=0 axisym
        print(f"  scipy reference l={l}...", end=" ", flush=True)
        phi_lm = phi_l0_scipy_at_r(r_arr, l)
        for j, th in enumerate(theta_arr):
            phi[:, j] += phi_lm * _Y_l0_np(l, math.cos(th))
        print("done")
    return phi

# ── Build JAX expansions ──────────────────────────────────────────────────────
print("=== Building JAX MultipoleExpansions ===")
l_max_vals = [2, 4, 8]
exps = {}
for lm in l_max_vals:
    t0 = time.perf_counter()
    exps[lm] = MultipoleExpansion.from_density(
        rho_nfw, R_MIN, R_MAX, N_R, l_max=lm
    )
    print(f"  l_max={lm:2d}  build={time.perf_counter()-t0:.2f}s")

# ── Test points ───────────────────────────────────────────────────────────────
# Radii and two polar angles: equatorial (theta=pi/2) and polar (theta=0)
r_test   = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
theta_eq = math.pi / 2   # equatorial
theta_po = 0.0            # polar

x_eq = r_test;  y_eq = np.zeros_like(r_test);  z_eq = np.zeros_like(r_test)
x_po = np.zeros_like(r_test); y_po = np.zeros_like(r_test); z_po = r_test

# ── Scipy reference (l_max=12 should be more than enough) ─────────────────────
print("\n=== Computing scipy reference (l_max_ref=12) ===")
L_REF = 12
phi_ref = phi_scipy_reference(r_test, L_REF, [theta_eq, theta_po])
# phi_ref shape: (n_r, 2)  -> col 0 = equatorial, col 1 = polar
phi_ref_eq = phi_ref[:, 0]
phi_ref_po = phi_ref[:, 1]

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n=== Equatorial plane (theta=pi/2) ===")
print(f"{'r':>6}  {'ref':>12}", end="")
for lm in l_max_vals:
    print(f"  {'bfe_l'+str(lm):>12}  {'relerr':>9}", end="")
print()
print("-" * (6 + 14 + len(l_max_vals) * 24))

jnp_r = jnp.array(r_test)
jnp_z = jnp.zeros_like(jnp_r)

for i, r in enumerate(r_test):
    print(f"{r:6.2f}  {phi_ref_eq[i]:12.6f}", end="")
    for lm in l_max_vals:
        phi_bfe = float(exps[lm](jnp_r[i:i+1], jnp_z[i:i+1], jnp_z[i:i+1])[0])
        err = abs((phi_bfe - phi_ref_eq[i]) / phi_ref_eq[i])
        print(f"  {phi_bfe:12.6f}  {err:9.2e}", end="")
    print()

print("\n=== Polar axis (theta=0) ===")
print(f"{'r':>6}  {'ref':>12}", end="")
for lm in l_max_vals:
    print(f"  {'bfe_l'+str(lm):>12}  {'relerr':>9}", end="")
print()
print("-" * (6 + 14 + len(l_max_vals) * 24))

for i, r in enumerate(r_test):
    print(f"{r:6.2f}  {phi_ref_po[i]:12.6f}", end="")
    for lm in l_max_vals:
        phi_bfe = float(exps[lm](jnp_z[i:i+1], jnp_z[i:i+1], jnp_r[i:i+1])[0])
        err = abs((phi_bfe - phi_ref_po[i]) / phi_ref_po[i])
        print(f"  {phi_bfe:12.6f}  {err:9.2e}", end="")
    print()

# Anisotropy: Phi(eq) != Phi(po) proves l>0 terms are active
print("\n=== Anisotropy: |Phi_eq - Phi_po| / |Phi_eq| ===")
print("(should be > 0 for q=0.6, = 0 for sphere)")
for i, r in enumerate(r_test):
    aniso = abs(phi_ref_eq[i] - phi_ref_po[i]) / abs(phi_ref_eq[i])
    print(f"  r={r:.1f}  anisotropy={aniso:.4f}")

# ── Visualisation: potential on x-z slice ────────────────────────────────────
print("\n=== Generating potential slice plot ===")
exp_ref = exps[max(l_max_vals)]   # best BFE

ng = 80
lim = 5.0
xv = np.linspace(-lim, lim, ng)
zv = np.linspace(-lim, lim, ng)
XX, ZZ = np.meshgrid(xv, zv)
YY = np.zeros_like(XX)

XJ = jnp.array(XX.ravel())
ZJ = jnp.array(ZZ.ravel())
YJ = jnp.array(YY.ravel())

phi_grid = np.array(exps[8](XJ, YJ, ZJ)).reshape(ng, ng)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: potential map
ax = axes[0]
levels = np.linspace(phi_grid.min(), phi_grid.max() * 0.5, 30)
cf = ax.contourf(XX, ZZ, phi_grid, levels=50, cmap="viridis")
ax.contour(XX, ZZ, phi_grid, levels=15, colors="w", linewidths=0.5, alpha=0.5)
plt.colorbar(cf, ax=ax, label=r"$\Phi(x, 0, z)$")
ax.set_xlabel("x"); ax.set_ylabel("z")
ax.set_title(f"Flattened NFW (q={q}, l_max=8)")
ax.set_aspect("equal")

# Right: convergence vs l_max at r=1
r_curve = np.exp(np.linspace(np.log(0.05), np.log(20), 100))
zc = np.zeros_like(r_curve)
rj = jnp.array(r_curve)
zj = jnp.zeros_like(rj)

ax = axes[1]
# Reference curve (scipy, equatorial)
phi_ref_curve = phi_scipy_reference(r_curve, L_REF, [theta_eq])[:, 0]
ax.plot(r_curve, phi_ref_curve, "k-", lw=2, label=f"scipy ref (l_max={L_REF})")

colors = ["C0", "C1", "C2"]
for col, lm in zip(colors, l_max_vals):
    phi_bfe_curve = np.array(exps[lm](rj, zj, zj))
    ax.plot(r_curve, phi_bfe_curve, "--", color=col, label=f"BFE l_max={lm}")

ax.set_xscale("log")
ax.set_xlabel("r"); ax.set_ylabel(r"$\Phi(r, 0, 0)$")
ax.set_title("Convergence with l_max (equatorial)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("flattened_nfw.png", dpi=150)
print("Saved: flattened_nfw.png")
