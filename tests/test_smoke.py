"""
Smoke test: Plummer sphere.

    rho(r) = 3M / (4*pi*b^3) * (1 + r^2/b^2)^{-5/2}
    Phi(r) = -G*M / sqrt(r^2 + b^2)

With G=1, M=1, b=1.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from bfeax import MultipoleExpansion

# --- Plummer density -------------------------------------------------------
b = 1.0
M = 1.0

def rho_plummer(x, y, z):
    r2 = x**2 + y**2 + z**2
    return 3 * M / (4 * jnp.pi * b**3) * (1 + r2 / b**2) ** (-2.5)

def phi_plummer_exact(x, y, z):
    r2 = x**2 + y**2 + z**2
    return -M / jnp.sqrt(r2 + b**2)

# --- Build expansion -------------------------------------------------------
print("Building MultipoleExpansion (l_max=4, n_r=64)...")
exp = MultipoleExpansion.from_density(
    rho_plummer,
    r_min=1e-2,
    r_max=1e2,
    n_r=64,
    l_max=4,          # Plummer is spherical -> only l=0 matters
)
print("Done.")

# --- Test at a grid of radii -----------------------------------------------
r_test = jnp.array([0.1, 0.3, 1.0, 3.0, 10.0])
x_test = r_test
y_test = jnp.zeros_like(r_test)
z_test = jnp.zeros_like(r_test)

phi_bfe  = exp(x_test, y_test, z_test)
phi_true = phi_plummer_exact(x_test, y_test, z_test)
rel_err  = jnp.abs((phi_bfe - phi_true) / phi_true)

print("\nr       phi_bfe       phi_exact     rel_err")
print("-" * 55)
for i in range(len(r_test)):
    print(f"{r_test[i]:.2f}    {phi_bfe[i]:.6f}    {phi_true[i]:.6f}    {rel_err[i]:.2e}")

max_err = float(jnp.max(rel_err))
print(f"\nMax relative error: {max_err:.2e}")
assert max_err < 5e-3, f"Too large: {max_err:.2e}"
print("PASSED")
