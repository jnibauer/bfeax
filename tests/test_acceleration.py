"""
test_acceleration.py — verify exp.acceleration() correctness.

Two independent checks:

  1. Finite-difference check  (model-agnostic)
     ∇Φ estimated with central differences at 50 random points.
     Both the Plummer sphere and the triaxial NFW are tested.

  2. Analytical check  (Plummer sphere only)
     The Plummer potential is  Φ(r) = -GM / sqrt(r² + a²),
     giving  a_r = -GM r / (r² + a²)^{3/2}  radially.
     We compare the BFE acceleration magnitude to this formula.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from bfeax import MultipoleExpansion

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ── helpers ────────────────────────────────────────────────────────────────────

def finite_diff_grad(exp, xs, ys, zs, h=1e-5):
    """Central-difference gradient of Phi at each point."""
    def phi(x, y, z):
        return np.array(exp(jnp.array([x]), jnp.array([y]), jnp.array([z])))[0]

    n = len(xs)
    gx = np.empty(n); gy = np.empty(n); gz = np.empty(n)
    for i in range(n):
        x, y, z = xs[i], ys[i], zs[i]
        gx[i] = (phi(x+h, y, z) - phi(x-h, y, z)) / (2*h)
        gy[i] = (phi(x, y+h, z) - phi(x, y-h, z)) / (2*h)
        gz[i] = (phi(x, y, z+h) - phi(x, y, z-h)) / (2*h)
    return -gx, -gy, -gz   # acceleration = -grad Phi


def check_fd(name, exp, xs, ys, zs, tol_median=0.01, tol_max=0.05):
    """Compare BFE acceleration against finite differences."""
    ax_fd, ay_fd, az_fd = finite_diff_grad(exp, xs, ys, zs)

    ax_b = np.array(exp.acceleration(jnp.array(xs), jnp.array(ys), jnp.array(zs))[0])
    ay_b = np.array(exp.acceleration(jnp.array(xs), jnp.array(ys), jnp.array(zs))[1])
    az_b = np.array(exp.acceleration(jnp.array(xs), jnp.array(ys), jnp.array(zs))[2])

    # Call once to get all three components together
    ax_b, ay_b, az_b = [np.array(a) for a in
                        exp.acceleration(jnp.array(xs), jnp.array(ys), jnp.array(zs))]

    mag_fd  = np.sqrt(ax_fd**2 + ay_fd**2 + az_fd**2)
    mag_bfe = np.sqrt(ax_b**2  + ay_b**2  + az_b**2)

    # Vector relative error (norm of difference / norm of truth)
    vec_err = np.sqrt((ax_b-ax_fd)**2 + (ay_b-ay_fd)**2 + (az_b-az_fd)**2) / (mag_fd + 1e-30)

    median_err = np.median(vec_err)
    max_err    = np.max(vec_err)

    ok = (median_err < tol_median) and (max_err < tol_max)
    tag = PASS if ok else FAIL

    print(f"  [{tag}]  {name:30s}  median={median_err:.3e}  max={max_err:.3e}")
    return ok


# ── Plummer sphere ─────────────────────────────────────────────────────────────

print("\n=== Plummer sphere ===")

G_code, M, a = 1.0, 1.0, 1.0

def rho_plummer(x, y, z):
    r2 = x**2 + y**2 + z**2
    return (3*M / (4*jnp.pi)) * a**2 / (r2 + a**2)**2.5

exp_pl = MultipoleExpansion.from_density(
    rho_plummer, r_min=1e-2, r_max=100.0, n_r=128, l_max=0
)

rng = np.random.default_rng(0)
r_s   = np.exp(rng.uniform(np.log(0.1), np.log(10), 50))
theta = np.arccos(rng.uniform(-1, 1, 50))
phi_s = rng.uniform(0, 2*np.pi, 50)
xs_pl = r_s * np.sin(theta) * np.cos(phi_s)
ys_pl = r_s * np.sin(theta) * np.sin(phi_s)
zs_pl = r_s * np.cos(theta)

# 1a. Finite-difference check
check_fd("Plummer FD check", exp_pl, xs_pl, ys_pl, zs_pl, tol_median=1e-3, tol_max=5e-3)

# 1b. Analytical check
ax_b, ay_b, az_b = [np.array(a) for a in
    exp_pl.acceleration(jnp.array(xs_pl), jnp.array(ys_pl), jnp.array(zs_pl))]

r_pts = np.sqrt(xs_pl**2 + ys_pl**2 + zs_pl**2)
a_mag_analytic = G_code * M * r_pts / (r_pts**2 + a**2)**1.5   # |a_r|

# BFE radial component (should equal a_mag_analytic for l_max=0 Plummer)
a_mag_bfe = np.sqrt(ax_b**2 + ay_b**2 + az_b**2)
rel_analytic = np.abs(a_mag_bfe - a_mag_analytic) / a_mag_analytic

med_an = np.median(rel_analytic)
max_an = np.max(rel_analytic)
ok_an  = (med_an < 5e-3) and (max_an < 2e-2)
tag    = PASS if ok_an else FAIL
print(f"  [{tag}]  {'Plummer analytical check':30s}  median={med_an:.3e}  max={max_an:.3e}")


# ── Triaxial NFW ───────────────────────────────────────────────────────────────

print("\n=== Triaxial NFW (q1=0.8, q2=0.5, l_max=6) ===")

rho_s, r_s_nfw, q1, q2 = 1.0, 1.0, 0.8, 0.5

def rho_nfw(x, y, z):
    m = jnp.clip(jnp.sqrt(x**2 + (y/q1)**2 + (z/q2)**2), 1e-30)
    return rho_s / (m/r_s_nfw * (1.0 + m/r_s_nfw)**2)

exp_nfw = MultipoleExpansion.from_density(
    rho_nfw, r_min=1e-2, r_max=300.0, n_r=128, l_max=6
)

r_pts   = np.exp(rng.uniform(np.log(0.1), np.log(20), 50))
theta   = np.arccos(rng.uniform(-1, 1, 50))
phi_pts = rng.uniform(0, 2*np.pi, 50)
xs_nfw  = r_pts * np.sin(theta) * np.cos(phi_pts)
ys_nfw  = r_pts * np.sin(theta) * np.sin(phi_pts)
zs_nfw  = r_pts * np.cos(theta)

check_fd("NFW FD check", exp_nfw, xs_nfw, ys_nfw, zs_nfw, tol_median=0.02, tol_max=0.08)


# ── Direction check (acceleration points inward) ──────────────────────────────

print("\n=== Direction sanity check (a · r̂ < 0 everywhere) ===")
for name, exp, xs, ys, zs in [
    ("Plummer", exp_pl, xs_pl, ys_pl, zs_pl),
    ("NFW",     exp_nfw, xs_nfw, ys_nfw, zs_nfw),
]:
    ax_b, ay_b, az_b = [np.array(a) for a in
        exp.acceleration(jnp.array(xs), jnp.array(ys), jnp.array(zs))]
    r_vec = np.stack([xs, ys, zs], axis=1)
    a_vec = np.stack([ax_b, ay_b, az_b], axis=1)
    r_hat = r_vec / (np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-30)
    a_dot_rhat = np.einsum("ni,ni->n", a_vec, r_hat)
    frac_inward = np.mean(a_dot_rhat < 0)
    ok = frac_inward > 0.95
    tag = PASS if ok else FAIL
    print(f"  [{tag}]  {name:10s}  fraction pointing inward: {frac_inward:.2%}")


print()
