# CLAUDE.md — bfe_claude

JAX-based basis function expansion for gravitational potentials.
Converts an arbitrary density function ρ(x,y,z) to a smooth potential Φ(x,y,z)
via spherical harmonic + radial spline decomposition, inspired by Agama.

---

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)   # always required
import jax.numpy as jnp
from bfe import MultipoleExpansion

def rho(x, y, z): ...   # any JAX-traceable scalar function

exp = MultipoleExpansion.from_density(rho, r_min=1e-2, r_max=3e2, n_r=128, l_max=8)
phi        = exp(x, y, z)              # potential
rho_rec    = exp.density(x, y, z)     # reconstructed density
ax, ay, az = exp.acceleration(x, y, z)  # -grad Phi via autodiff
```

---

## Package layout

```
bfe/
  grid.py          — log-spaced radial grids
  sph_harm.py      — real orthonormal Y_lm via Legendre recurrence;
                     Gauss-Legendre × uniform-phi angular quadrature grid
  density_coeffs.py — project rho(x,y,z) -> rho_lm(r) via vmap+jit
  spline.py        — natural cubic splines in JAX (JIT-safe)
  poisson.py       — Green's function radial Poisson solver with
                     inner + outer power-law boundary corrections
  potential.py     — MultipoleExpansion class (main user-facing API)
```

---

## Algorithm

### 1. Angular projection (density_coeffs.py)

For each radius r_i on the log grid:

    rho_lm(r_i) = ∫ rho(r_i, θ, φ) Y_lm(θ, φ) dΩ

Quadrature: Gauss-Legendre in cos θ (n_theta points) × uniform in φ (n_phi points).
All radii are processed in a single `jax.vmap` call over the grid.

Defaults (empirically converged for triaxial NFW):

    n_theta = l_max + 2
    n_phi   = 2 * l_max + 1   (always odd, so φ=0 and φ=π not hit together)

Convergence benchmarks show the error plateaus at these values and does not
improve with more points.  The old defaults (3*(l_max+2) and 4*l_max+7) were
~3x and ~2x over-sampled, resulting in ~7x more density evaluations per shell.

### 2. Poisson solve (poisson.py)

For each (l, m) mode, solve the radial Poisson equation via the Green's function:

    Φ_lm(r) = -4πG/(2l+1) * [ r^{-(l+1)} I_in(r)  +  r^l I_out(r) ]

where I_in and I_out are cumulative trapezoidal integrals inside-out and outside-in.

**Boundary corrections** (key for profiles with slow outer falloff like NFW):

- *Inner tail*: estimates power-law slope α from the first 3 grid points,
  adds the analytical contribution ∫_0^{r_min} analytically to I_in.
- *Outer tail*: estimates slope from the last 3 grid points,
  adds the analytical contribution ∫_{r_max}^∞ analytically to I_out.

The outer correction is critical for NFW (ρ ~ r^{-3}, r_max must be large
or the exterior integral is significantly truncated). Use r_max ≥ 30 × r_s.

### 3. Inner cusp subtraction (potential.py)

For density profiles with steep inner slopes (NFW: ρ_lm ~ A r^{-1} near r=0),
fitting splines directly to rho_lm is poorly conditioned near r_min.

Instead, for each (l,m) mode:
1. Estimate inner slope α and amplitude A from the first 3 grid points.
2. Subtract bg(r) = A r^α before spline fitting → smooth residual.
3. At evaluation: rho_lm(r) = residual_spline(r) + A r^α.

Guard: only applied when |rho_lm(r_min)| > 1e-6 × global_scale,
where global_scale is the max amplitude over ALL modes. This prevents
spurious large backgrounds on symmetry-zero modes (e.g. odd-l modes
for an octant-symmetric density) that sit at machine-epsilon level.

### 4. Evaluation

    Φ(x,y,z) = Σ_{l,m} Φ_lm(r) Y_lm(θ,φ)
    ρ_rec(x,y,z) = Σ_{l,m} rho_lm(r) Y_lm(θ,φ)

Accelerations are computed by `jax.grad` through `__call__`.

---

## Key parameters

| Parameter | Typical value | Notes |
|-----------|--------------|-------|
| `l_max` | 6–10 | 8 is a good default for mildly triaxial halos |
| `n_r` | 64–128 | 128 converges to ~0.1% for NFW |
| `r_min` | ~1e-2 × r_s | Should be well inside the inner cusp |
| `r_max` | ≥ 30 × r_s | **Must be large for NFW** — outer tail diverges logarithmically |

---

## Convergence behaviour

For a triaxial NFW (q1=0.8, q2=0.5), n_r=128, r_max=300:

**Density reconstruction (median relative error over random points r ∈ [0.05, 50])**

| l_max | median error |
|-------|-------------|
| 4 | ~3% |
| 6 | ~0.8% |
| 8 | ~0.2% |
| 10 | ~0.06% |

Error drops ~4× per Δl=2 — clean geometric convergence.

**Potential (vs scipy reference, equatorial plane)**

| l_max | median error |
|-------|-------------|
| 2 | ~0.14% |
| 4 | ~0.17% |
| 8 | ~0.43% |

The potential converges faster than the density; l_max=2–4 is often sufficient.

---

## JAX sharp bits

- **Always enable float64**: `jax.config.update("jax_enable_x64", True)`.
  Without it, the cubic spline tridiagonal solve loses precision and the
  Poisson integrals drift.

- **l_max is a static Python int** throughout. It controls loop unrolling
  in `sph_harm.py`, array shapes in `density_coeffs.py`, and `static_argnums`
  in `poisson.py`. Never pass l_max as a traced value.

- **Dynamic alpha in power-law terms**: `A * r^alpha` is computed as
  `A * exp(alpha * log(r))` everywhere to avoid overflow when alpha is a
  traced JAX scalar.

- **jnp.where evaluates both branches**: all conditional corrections
  (inner/outer tail, cusp subtraction validity) guard the *value* with
  `jnp.where` before using it, not after, to avoid NaN propagation from
  the unselected branch.

- **Build-time JIT**: `density_to_sph_coeffs` is JIT-compiled internally.
  The first call for a given (l_max, n_theta, n_phi) shape triggers XLA
  compilation (~0.5–1s). Subsequent builds with the same shapes reuse
  the compiled kernel.

---

## Tests

| File | What it tests |
|------|--------------|
| `tests/test_smoke.py` | Plummer sphere — checks potential accuracy vs analytical result |
| `tests/test_timing.py` | Plummer convergence with n_r and l_max; JIT eval throughput |
| `tests/test_flattened_nfw.py` | Oblate NFW (q=0.6) — compares against scipy reference |
| `tests/test_triaxial_nfw.py` | Triaxial NFW (q1=0.8, q2=0.5) — density reconstruction, mode amplitudes, x-z convergence |

Run all: `python tests/test_smoke.py && python tests/test_triaxial_nfw.py`

---

## Known limitations / future work

- **Particle input**: no `from_particles(pos, mass)` yet. Would require
  binning particles onto the angular grid at each radius (or KDE).

- **Automatic grid selection**: r_min and r_max must be set manually.
  Agama auto-selects them from the density profile; could be added with
  a density survey step.

- **Non-triaxial densities**: the angular quadrature defaults are sized for
  triaxial halos. For highly non-smooth densities (e.g. disk + halo composites)
  n_theta and n_phi may need to be increased manually.

- **Acceleration memory**: `acceleration()` uses `jax.grad` through `__call__`,
  which materialises the full Jacobian. For batched point evaluation,
  `jax.vmap(jax.grad(...))` is more efficient.
