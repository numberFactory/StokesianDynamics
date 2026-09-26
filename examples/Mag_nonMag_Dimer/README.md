# `bonded_dimer_EM_force.py`

Simulates a **bonded dimer**: two spheres held together by a stiff spring,
an orientation ("torque") spring, and a bond-direction spring, mimicking a
physically/chemically bonded pair of colloids, with only one particle
magnetic, in a uniaxial field that flips sign periodically. Derived from
`ladder_to_ring_EM_force.py`.

> **Bug fix included:** the permanent moment was previously observed to
> drift relative to the vector connecting the two spheres, despite the
> orientation spring. Root cause and fix are in §4b below.

**Units** (as elsewhere in this codebase): lengths in µm, energy/torque in
aJ, forces in aJ/µm (= pN), magnetic field in mT, magnetic moment in aJ/mT.

---

## Parameters

### Particles / fluid
| Name | Value | Meaning |
|---|---|---|
| `a` | 1.35 µm | Particle radius (`2a` = 2.7 µm) |
| `eta` | 8.9e-4 | Fluid viscosity |
| `kT` | 0.0041419464 aJ | Thermal energy |
| `g` | ≈0.0606 | Net gravity/wall force scale (buoyancy-corrected weight). Rescaled *volumetrically*, `g = g_base·(a/a_base)³`, from the base script's value at `a_base = 2.25 µm` |
| `Stoch` | `True` | If `True`, Brownian forces/torques are added on top of everything below (handled internally by the solver, consistent with `kT`) |
| `MAG_IDX` / `NONMAG_IDX` | 0 / 1 | Which particle is magnetic |

### Magnetic field & moment
| Name | Value | Meaning |
|---|---|---|
| `mu_dipole` | 0.66 aJ/mT | Permanent moment of the magnetic particle. Converted from 6.6×10⁻¹⁶ A·m² via `1 aJ/mT = 1e-15 A·m²` |
| `chi_exp` | 1.27 | Magnetic susceptibility |
| `RB_0` | ≈10.42 | Induced-moment prefactor: `m_induced = RB_0 · B(t)`. Closed form `RB_0 = (10/3)·a³·χ` |
| `B_amp` | 2.0 mT | Field amplitude |
| `B_dwell` | 5.0 s | Field points `+ŷ` for this long, then `-ŷ`, then `+ŷ`, ... |
| `MOMENT_MODE` | `'random'` | Selects how `m_body_local` (below) is chosen: `'random'` or `'axis_aligned'` |
| `m_body_local` | depends on `MOMENT_MODE` | The permanent moment's direction **in the magnetic particle's own body frame**. `'random'`: a random, fixed unit vector (a random combination of its triad vectors), seed 42. `'axis_aligned'`: locked to the particle's own first triad axis, `[1, 0, 0]` in body coordinates. Either way it's rotated into the world frame every step via the particle's current orientation, so it's always locked to the particle, not the lab |

### Steric repulsion
| Name | Value | Meaning |
|---|---|---|
| `firm_delta` | 1e-2 | Dimensionless overlap/softness parameter |
| `debye_firm` | ≈0.01173 µm | Steric repulsion decay length, `2a·firm_delta/ln(10)` |
| `repulsion_firm` | 0.0331 | Steric repulsion strength |
| `repulsion_soft`, `debye_soft` | 0.0, 0.225 | Optional longer-range soft (Yukawa) repulsion — currently off |

### Bond (translational spring + orientation spring + bond-direction spring)
| Name | Value | Meaning |
|---|---|---|
| `r0_spring` | ≈2.7235 µm | Spring rest length, `2a + 2·debye_firm` |
| `k_spring` | ≈75.3 aJ/µm² | Translational spring stiffness, `KAPPA_SPRING_COEFF·kT/(2·debye_firm)²` |
| `k_theta` | ≈4.142 aJ | Orientation-spring stiffness, `KAPPA_THETA_COEFF·kT` (see below) |
| `k_dir` | ≈4.142 aJ | Bond-direction-spring stiffness, set equal to `k_theta`. Ties each particle's fixed body-frame reference axis (`bond_dir_body`, set once at `t=0`) to the *current lab-frame bond direction* `r̂`. See §4b — this is the bug-fix term. |
| `bond_dir_body` | per particle, set once at `t=0` | Each particle's reference axis, expressed in its own body frame, that should stay aligned with `r̂`. Computed once at start-up as `R_kᵀ(0) · r̂(0)`, so at `t=0` it exactly equals the initial bond direction rotated into that body's frame — i.e. it costs nothing initially, by construction |

> **`MOMENT_MODE = 'axis_aligned'`** is a convenient debugging setting: since
> the dimer starts with both particles at identity orientation, placed
> along the lab x-axis, setting `m_body_local = [1,0,0]` makes the initial
> permanent moment point exactly along the bond too. That makes it easy to
> visually check the bond-direction spring (§4b below) — the gold moment
> arrow should then track the bond line directly, with no offset to account
> for a random initial angle.

The two pre-existing energy-scale coefficients are set **asymmetrically**:
`KAPPA_SPRING_COEFF = 10·kT` (translational spring), `KAPPA_THETA_COEFF =
1000·kT` (orientation spring) — i.e. the orientation spring is ~100x
stiffer, in energy-scale terms, than the translational one. The new
bond-direction spring reuses `KAPPA_THETA_COEFF` (`k_dir = k_theta`), on the
reasoning that it should be roughly as stiff as the spring it's plugging a
gap next to.

### Simulation / output
| Name | Value | Meaning |
|---|---|---|
| `dt` | 1e-3 s | Timestep |
| `t_end` | 20.0 s | Total simulated time (covers 4 field flips) |
| `plot_dt` | 0.1 s | Time between saved frames (→ `n_plot` steps, ~200 frames total) |
| `L` | (0,0,0) | Open geometry, no periodicity |
| Output | `bonded_dimer_frames[_stochastic]/frame_*.png` | One PNG per saved frame (PyVista 3-D render) |

---

## Forces and torques

Computed each step in `force_torque_calculator`, combining a numba kernel
(gravity/wall + steric) with a small amount of plain-NumPy code for the two
physics pieces that only ever apply to the single explicit pair (0, 1): the
bond and the magnetic field.

1. **Gravity + firm wall repulsion** (per particle, z only)
   Gravity pulls down with force `g`; an exponential "firm" repulsion from
   the wall balances it, giving each particle a soft equilibrium height:
   ```
   F_z = -g + (rep_firm/debye_firm)·exp(-(h-contact)/debye_firm)   if h > contact
   F_z = -g + rep_firm/debye_firm                                   otherwise
   ```
   where `contact = a·(1-firm_delta)`.

2. **Pairwise steric repulsion** (between the two particles)
   Same functional form as above, but along the line connecting the two
   centres — a short-range "hard" contact repulsion (plus an optional
   longer-range soft Yukawa term, currently off) that keeps the spheres
   from interpenetrating, independent of the bond.

3. **Bond spring** (harmonic, explicit pair)
   ```
   F_i = k_spring·(r - r0_spring)·r̂,   F_j = -F_i
   ```
   Pulls/pushes the two centres back to separation `r0_spring`. This is
   the force that "mimics" the bond holding the dimer together.

4. **Orientation ("torque") spring** (explicit pair)
   Built from the energy `U = -k_theta·Tr(Rᵢᵀ Rⱼ) = -k_theta·Σₖ (eᵢ,ₖ·eⱼ,ₖ)`,
   where `eᵢ,ₖ` is the k-th body axis (column of the rotation matrix) of
   particle *i* in the world frame. `U` is minimised when `Rᵢ = Rⱼ`. Its
   gradient gives a clean, singularity-free torque:
   ```
   τᵢ = k_theta · Σₖ (eᵢ,ₖ × eⱼ,ₖ),   τⱼ = -τᵢ
   ```
   This is zero when the two particles share the same orientation, and
   otherwise rotates each one toward the other's — i.e. it makes them
   rotate together like a rigid-ish bonded pair, not just stay a fixed
   distance apart. (The plot's per-axis triad colouring — pink/turquoise/
   purple for axis 0/1/2, shared across both particles — is there specifically
   so you can visually check how well this spring is doing its job: e.g. how
   parallel are the two particles' pink arrows?)

   **4b. Bug: the moment drifts relative to the bond, even with this spring.**
   Springs 3 and 4 above constrain `|r|` (separation) and `Rᵢ` vs. `Rⱼ`
   (the two particles' orientations *relative to each other*) — but nothing
   constrains either one relative to the actual lab-frame direction of the
   bond, `r̂ = (rⱼ−rᵢ)/|rⱼ−rᵢ|`. So the pair is free to rotate together, as
   a rigid unit, in any direction, while `r̂` is independently free to swing
   around — nothing couples the two. Under Brownian rotational diffusion
   (`Stoch=True`) each undergoes its own random walk, and they slowly
   separate: exactly the reported symptom, since the permanent moment is
   locked only to the magnetic particle's own orientation, not to `r̂`.

   The fix is a third spring, built from the energy
   ```
   U = -k_dir · (eᵢ + eⱼ) · r̂,        eₖ = Rₖ · bond_dir_body[k]
   ```
   where `bond_dir_body[k]` is particle *k*'s reference axis, fixed in its
   own body frame and set once at `t=0` (so `U=0` initially, by
   construction — see the table above). Note the **sum** `eᵢ + eⱼ`, not a
   difference: both reference axes are defined to point the *same* way as
   the initial bond direction, consistent with spring 4 above already
   forcing `Rᵢ≈Rⱼ` (hence `eᵢ≈eⱼ`, not anti-parallel). `U` is minimised
   when both `eᵢ` and `eⱼ` are aligned with `r̂`. Its gradient gives:
   ```
   τᵢ = k_dir·(eᵢ × r̂),   τⱼ = k_dir·(eⱼ × r̂)
   ```
   plus a matching pair of forces (from `∂r̂/∂rⱼ = (I − r̂⊗r̂)/|r|`, needed
   for the torque to be consistent with a translation-invariant energy):
   ```
   perpₖ = eₖ − (eₖ·r̂)·r̂
   F_dir = (k_dir/|r|)·(perpᵢ + perpⱼ)     # on j; −F_dir on i
   ```
   This is exactly zero at true equilibrium (`eᵢ=eⱼ=r̂`), and otherwise
   pulls the shared orientation and the bond direction back toward each
   other — closing the gap that springs 3–4 alone left open.

5. **External magnetic field torque** (magnetic particle only)
   ```
   m_perm     = mu_dipole · (R_mag @ m_body_local)      # permanent, body-locked
   m_induced  = RB_0 · B(t)                              # linear response
   τ_mag      = (m_perm + m_induced) × B(t)
   ```
   `B(t)` is the ±ŷ square wave described above. Since the field is
   spatially uniform, it exerts **torque only, no net force**. The
   non-magnetic particle has no moment, so it gets no contribution here at
   all — this is the only place in the code where the two particles are
   treated asymmetrically.

6. **Brownian (stochastic) forces/torques** — only when `Stoch = True`.
   Not computed in `force_torque_calculator`; added by the solver itself
   (`Update_Bodies_Trap(..., stochastic=True)`), consistent with `kT`, on
   top of everything above.

---

## Visualization notes

- Sphere colours: cornflower blue = magnetic particle, white = non-magnetic.
- Triad arrows (thin): pink/turquoise/purple = axis 0/1/2, coloured the
  same way on **both** particles so corresponding axes can be compared.
- Thick gold/orange arrow: the magnetic particle's current permanent moment.
- Floating green arrow (offset above the dimer, so it never overlaps the
  spheres): the applied field's current direction.
- A faint grey plane at `z=0` marks the wall.

## Other notes

- `OMP_NUM_THREADS=1` is forced at the very top of the file (before
  numpy/numba are imported), to avoid thread-oversubscription.
- `k_spring` (~75.3 aJ/µm², from `KAPPA_SPRING_COEFF=10·kT`) is at a
  relatively gentle level, but `k_theta` and the new `k_dir` (~4.14 aJ
  each, from `KAPPA_THETA_COEFF=1000·kT`) are now the stiffest terms in the
  system, and there are two of them acting on orientation instead of one —
  so if `dt = 1e-3` turns out to be numerically unstable, these rotational
  springs are the more likely culprit. This hasn't been runtime-verified (no
  numba/pyvista in the sandbox this was written in); shrink `dt` first if it
  blows up or oscillates.
