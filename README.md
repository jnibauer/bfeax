# bfeax

**JAX-based basis function expansion for gravitational potentials.**

Convert a density function ρ(x,y,z) into a smooth, differentiable potential
Φ(x,y,z) via spherical harmonic + radial spline decomposition. Fully JIT-compiled,
GPU-ready, and autodiff-compatible through `jax.grad`.

---

## Installation

```bash
pip install git+https://github.com/jnibauer/bfeax.git
```

For development (editable install):

```bash
git clone https://github.com/jnibauer/bfeax.git
cd bfeax
pip install -e .
```

> **Note:** `bfeax` requires JAX with 64-bit precision enabled. Install JAX for your
> platform following the [official instructions](https://jax.readthedocs.io/en/latest/installation.html).
> Always call `jax.config.update("jax_enable_x64", True)` before using `bfeax`.

---

## Quickstart

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from bfeax import MultipoleExpansion

# Define any JAX-traceable density
def rho(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    return 1.0 / (r * (1 + r)**2)   # NFW

# Build the expansion
exp = MultipoleExpansion.from_density(rho, r_min=1e-2, r_max=300.0, n_r=128, l_max=8)

# Evaluate
phi        = exp(1.0, 0.5, 0.3)           # potential  Φ(x,y,z)
rho_rec    = exp.density(1.0, 0.5, 0.3)   # reconstructed density
ax, ay, az = exp.acceleration(1.0, 0.5, 0.3)  # -grad[phi] via autodiff
```

---

## Example: triaxial NFW halo

`bfeax` can handle **non-spherical** density profiles.
Here we build a triaxial NFW halo with axis ratios p=0.8, q=0.5 and exploit
octant symmetry to make the build ~3× faster.

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from bfeax import MultipoleExpansion, SpheroidDensity

# --- Define a triaxial NFW density -------------------------------------------
# SpheroidDensity wraps the Agama spheroid formula with a fast spline lookup table.
# Profile: rho ~ r^{-gamma} [1 + (r/a)^alpha]^{(gamma-beta)/alpha}
#   gamma=1, beta=3, alpha=1 => NFW   with axis ratios p=0.8, q=0.5

rho = SpheroidDensity(
    rho0=1.0, alpha=1.0, beta=3.0, gamma=1.0, a=1.0,
    p=0.8, q=0.5,           # triaxial: y squashed to 80%, z to 50%
    r_min=1e-3, r_max=500.0,
)

# --- Build the multipole expansion -------------------------------------------
# symmetry="triaxial" skips modes that vanish by symmetry (~3x fewer modes)
exp = MultipoleExpansion.from_density(
    rho,
    r_min=1e-2, r_max=300.0,
    n_r=128, l_max=8,
    symmetry="triaxial",
)

# --- Evaluate anywhere -------------------------------------------------------
x, y, z = 2.0, 1.5, 0.8

phi        = exp(x, y, z)
rho_rec    = exp.density(x, y, z)
ax, ay, az = exp.acceleration(x, y, z)


# --- Batch evaluation (JIT-compiled, GPU-friendly) ---------------------------
import jax
N = 1_000_000
key = jax.random.PRNGKey(0)
pts = jax.random.normal(key, (N, 3))

phi_batch = jax.jit(jax.vmap(exp))(pts[:, 0], pts[:, 1], pts[:, 2])
```

---

## Built-in density profiles

`SpheroidDensity` provides a fast spline lookup table implementation of the Agama spheroid:

$$
\rho(\tilde{r}) = \rho_0 \left(\frac{\tilde{r}}{a}\right)^{-\gamma} \left[ 1 + \left(\frac{\tilde{r}}{a}\right)^\alpha \right]^{\frac{\gamma - \beta}{\alpha}} \exp\left[ -\left(\frac{\tilde{r}}{r_{\text{cut}}}\right)^\xi \right]
$$

where r̃ = √(x² + (y/p)² + (z/q)²) is the spheroidal radius.

| Profile | alpha | beta | gamma |
|---------|-------|------|-------|
| NFW | 1 | 3 | 1 |
| Hernquist | 1 | 4 | 1 |
| Plummer | 2 | 5 | 0 |
| Jaffe | 1 | 4 | 2 |

---

## Symmetry options

Specifying `symmetry` skips modes that vanish by symmetry, reducing build time
and memory without sacrificing accuracy.

| `symmetry=` | Modes kept | Speedup |
|-------------|-----------|---------|
| `None` | All (l, m) | 1× |
| `"axisymmetric"` | Even l, m = 0 | ~(2l+1)× |
| `"triaxial"` | Even l, even m ≥ 0 | ~3× |
| `"spherical"` | (0, 0) only | maximum |

---

## Key parameters

| Parameter | Typical value | Notes |
|-----------|--------------|-------|
| `l_max` | 6–10 | 8 is a good default for triaxial halos |
| `n_r` | 64–128 | 128 converges to ~0.1% for NFW |
| `r_min` | ~1e-2 × r_s | Should sit well inside the inner cusp |
| `r_max` | ≥ 30 × r_s | Must be large for NFW; outer tail integral is critical |


---

## Package layout

```
bfeax/
  grid.py           — log-spaced radial grids
  sph_harm.py       — real orthonormal Y_lm via Legendre recurrence
  density_coeffs.py — project ρ(x,y,z) → ρ_lm(r) via vmap+jit
  spline.py         — natural cubic splines (JIT-safe)
  poisson.py        — Green's function radial Poisson solver
  potential.py      — MultipoleExpansion (main API)
  spheroid.py       — SpheroidDensity (Agama-style profile)
```
