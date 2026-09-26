"""
bonded_dimer_EM_force.py
-------------------------
Simulation of a single 'bonded' dimer of two spheres, mimicking a chemically
or physically bonded pair of colloids. Derived from ladder_to_ring_EM_force.py,
with the following changes:

  1. Two bodies only, held together as a rigid-ish 'bond' via
       - a stiff harmonic centre-centre spring (rest length 2a + 2*debye_firm)
       - a harmonic orientation ('torque') spring that penalises the two
         bodies' orientation matrices differing from one another
       - a harmonic bond-DIRECTION spring that penalises each body's
         (body-frame-fixed) reference direction differing from the actual
         bond vector r_hat -- see the "BUG FIX" note below; without this,
         the pair's shared orientation can drift relative to the line
         connecting the two centres even though the orientation spring above
         keeps the two bodies aligned WITH EACH OTHER.
     rather than the many-body dipole-mediated ladder/ring physics of the
     base script.
  2. Only ONE of the two particles (index MAG_IDX) is magnetic (permanent +
     induced moment); the other feels zero magnetic force/torque.
  3. The magnetic particle's permanent-moment direction is set in its BODY
     frame, fixed once at t=0, and transformed into the world frame every
     step via the current rotation matrix -- i.e. it is rigidly locked to
     the particle's orientation, not to the lab frame. `MOMENT_MODE`
     selects how that body-frame direction is chosen:
       - 'random'       : a random unit vector (a random, fixed combination
                          of the body's triad vectors) -- the original
                          behaviour.
       - 'axis_aligned' : locked to the particle's own first triad axis
                          (local x), i.e. m_body_local = [1, 0, 0].
  4. New physical scale: 2a = 2.7 um, applied field 2 mT, and a permanent
     moment converted from 6.6e-16 A*m^2 into this code's aJ/mT convention.
  5. The field is a simple uniaxial square wave along +/- y (no rotating
     in-plane field, no separate static z-field/electric-dipole term).
  6. Visualisation is a 3-D PyVista render (transparent spheres; thin triad
     arrows on both particles coloured by axis -- pink/turquoise/purple for
     axis 0/1/2, shared across both particles so corresponding axes can be
     compared; one thick gold/orange arrow for the magnetic particle's
     permanent moment; a floating green arrow showing the applied field's
     direction) instead of the base script's 2-D matplotlib discs.

BUG FIX (permanent moment drifting relative to the bond vector)
----------------------------------------------------------------
The orientation spring above only ties R_i to R_j -- it has NO reference to
the bond vector r_hat = (r_j - r_i)/|r_j - r_i| at all, and the
translational spring only constrains |r|, which is completely direction-
agnostic. So the two bodies can (and, under Brownian rotational diffusion
with Stoch=True, WILL) stay mutually aligned with each other while BOTH
drift together relative to r_hat, since nothing was penalising that. Since
the permanent moment is locked to R_mag (not to r_hat), it visibly wanders
relative to the line connecting the spheres -- this was the reported bug.

Fix: a third bond term, the bond-DIRECTION spring, ties a body-fixed
reference direction of EACH particle to the actual bond vector r_hat. The
reference direction is computed ONCE at t=0 (exactly analogous to
`m_body_local` for the permanent moment): `bond_dir_body[k] = R_k(0)^T @
r_hat(0)` for k in {0,1} -- i.e. "whatever direction the bond happens to
point in, in each particle's own initial body frame". With the initial
conditions used here (both bodies start at identity orientation, placed
along x), this works out to x-hat for both, but computing it this way
avoids hardcoding that assumption. The energy is

    U = -k_dir * (e_i + e_j) . r_hat,   e_k = R_k @ bond_dir_body[k]

(NOT e_i - e_j: since the orientation spring above already requires
R_i = R_j, and hence e_i = e_j, at equilibrium; a "-" here would demand
e_i and e_j be ANTI-parallel, which directly contradicts the orientation
spring and can never be simultaneously satisfied). Its gradient gives

    tau_i = k_dir * (e_i x r_hat),   tau_j = k_dir * (e_j x r_hat)
    F_j   = (k_dir/|r|) * (perp(e_i) + perp(e_j)) ,   F_i = -F_j

where perp(e) = e - (e.r_hat) r_hat is the component of e transverse to the
bond. The force term (needed for Newton's-third-law consistency, since this
is an internal two-body potential, not an external one-body torque like the
magnetic one) was derived and included, not just the torque. k_dir defaults
to k_theta (same energy scale as the orientation spring) -- change it
separately if you want the two decoupled.

Design notes / judgment calls (flagged here since the task left them
under-specified -- please sanity check / adjust):

  - "firm_debeye" in the spring rest-length/stiffness spec is taken to mean
    `debye_firm`, the steric-repulsion decay length already used elsewhere
    in this code (computed from `firm_delta` and `a`), not the dimensionless
    `firm_delta` itself.
  - The orientation ('torque') spring is built from the energy
        U = -k_theta * Tr(R_i^T R_j) = -k_theta * sum_k (e_i_k . e_j_k)
    (e_i_k/e_j_k = the k-th body-frame axis of particle i/j in world frame),
    which is minimised when R_i = R_j. Its gradient gives a clean,
    singularity-free torque
        tau_i = k_theta * sum_k (e_i_k x e_j_k),   tau_j = -tau_i
    matching the cross-product style already used for magnetic torques
    elsewhere in this codebase. The two spring energy-scale coefficients are
    now set asymmetrically, per explicit instruction: KAPPA_SPRING_COEFF
    (translational spring) = 10*kT, KAPPA_THETA_COEFF (this one) = 1000*kT
    -- i.e. the orientation spring is now ~100x stiffer, in energy-scale
    terms, than the translational one. k_theta is applied directly to the
    dimensionless rotation rather than rescaled by a squared length, since
    angles are already dimensionless.
  - `g` (gravity/wall force scale) is now volumetrically rescaled from the
    base script's a=2.25 value (0.28041) by (a/a_base)**3, since it
    represents a buoyancy-corrected weight (~ particle volume). `repulsion_
    firm` is still kept as-is (not part of what was requested to rescale).
    RB_0 (susceptibility prefactor) is likewise recomputed for the new `a`,
    since its formula and `chi_exp` were given explicitly in the base script.
  - `firm_delta = 1e-2` (10x the base script's 1e-3) and `dt = 1e-3` (~16x
    the previous 6.25e-5) are both set per explicit instruction. Note the
    larger `firm_delta` also makes `debye_firm` (and hence the steric
    repulsion's decay length) 10x longer/softer, which offsets some of the
    stiffness-vs-timestep risk from the larger `dt`. `k_spring` (~75 aJ/um^2
    at the current a/firm_delta, from KAPPA_SPRING_COEFF=10*kT) is back down
    to a relatively gentle level, but `k_theta` (~4.14 aJ, from
    KAPPA_THETA_COEFF=1000*kT) is now substantially stiffer than either
    spring has been so far -- so if this combination is numerically unstable
    at dt=1e-3, the orientation spring is now the more likely culprit, not
    the translational one. This could not be verified by actually running
    the simulation in this environment (numba/pyvista are not installed
    here), so please confirm stability once you run it; if it blows up or
    oscillates, `dt` is the first thing to shrink back down.
  - `Stoch = True`: Brownian motion is now included (per explicit
    instruction), on top of the deterministic bond/field/steric forces.
  - `OMP_NUM_THREADS` is forced to "1" at the very top of the file, before
    numpy/numba are imported (per explicit instruction).
  - Initial condition: both particles start with identical (identity)
    orientation, separated by exactly the spring's rest length along x, at
    the analytically-estimated single-particle wall-equilibrium height
    (so there's no initial transient in z). The magnetic particle's random
    moment direction uses a fixed seed (42) for reproducibility.
"""

import sys
import os

# Force single-threaded OpenMP -- must be set before numpy/numba are
# imported, since numba's threading layer and numpy's BLAS backend both
# read this at import time (avoids thread-oversubscription/contention).
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import numpy as np
import time
from functools import partial
from numba import njit, prange
from scipy.spatial.transform import Rotation

from body import Body
from pyStokesianDynamics import pyStokesianDynamics

import pyvista as pv


# =============================================================================
def main():
# =============================================================================

    # ── Physical parameters ───────────────────────────────────────────────────
    a   = 1.35           # particle radius (um)   [2a = 2.7 um, as requested]
    eta = 8.9e-4         # fluid viscosity                    (kept from base)
    kT  = 0.0041419464   # thermal energy                     (kept from base)

    # g (gravity/wall force scale) represents a buoyancy-corrected weight,
    # i.e. ~ particle volume, so rescale it volumetrically from the base
    # script's a=2.25 value rather than reusing it as-is.
    a_base = 2.25          # base script's particle radius (um)
    g_base = 0.28041       # base script's g at a_base
    g = g_base * (a / a_base) ** 3

    Stoch = True          # include stochastic forces/torques (Brownian motion)

    # ── Magnetic field parameters ─────────────────────────────────────────────
    # Permanent moment given as 6.6e-16 A*m^2; convert to this code's aJ/mT
    # convention: U = -m.B, with m in A*m^2 and B in Tesla gives U in Joules
    # (A*m^2 * T = J).  1 aJ/mT = 1e-18 J / 1e-3 T = 1e-15 J/T = 1e-15 A*m^2,
    # so   m [aJ/mT] = m [A*m^2] / 1e-15 = m [A*m^2] * 1e15.
    mu_dipole_SI_Am2  = 6.6e-16                 # A*m^2
    AM2_TO_AJ_PER_MT  = 1.0e15                  # 1 aJ/mT = 1e-15 A*m^2
    mu_dipole         = mu_dipole_SI_Am2 * AM2_TO_AJ_PER_MT   # -> 0.66 aJ/mT

    chi_exp = 1.27        # magnetic susceptibility            (kept from base)
    # RB_0 = (4/3 pi a^3) * chi_exp / mu_0, converted into aJ/mT per mT of
    # applied field when `a` is given in um; this reduces to the closed form
    # RB_0 = (10/3) * a**3 * chi_exp (verified: reproduces the base script's
    # RB_0 = 48.22 for a=2.25, chi_exp=1.27).  Recomputed here for the new a.
    RB_0 = (10.0 / 3.0) * (a ** 3) * chi_exp

    B_amp   = 2.0          # applied field amplitude (mT)
    B_dwell = 5.0           # seconds the field spends pointing each way
                            # before flipping: +y for B_dwell s, -y for
                            # B_dwell s, +y again, ...

    def B_field_fn(t):
        """Uniaxial square-wave field along +/- y."""
        half_cycles = int(t // B_dwell)
        sign = 1.0 if (half_cycles % 2 == 0) else -1.0
        return np.array([0.0, sign * B_amp, 0.0])

    # ── Which particle is magnetic ────────────────────────────────────────────
    MAG_IDX    = 0    # magnetic particle   (permanent + induced moment)
    NONMAG_IDX = 1    # non-magnetic particle (feels no magnetic force/torque)

    # Permanent-moment direction, expressed in the magnetic particle's OWN
    # body frame (i.e. some fixed combination of its triad vectors). Either
    # way it is set once here and re-expressed in the world frame every
    # step via the particle's current rotation matrix (see
    # force_torque_calculator) -- it is always body-frame-locked, never a
    # fixed lab-frame direction.
    #
    #   'random'       : a random unit vector in the body frame (the
    #                    original behaviour).
    #   'axis_aligned' : locked to the particle's own first body/triad axis
    #                    (local x), i.e. m_body_local = [1, 0, 0]. Since the
    #                    dimer starts with both particles at identity
    #                    orientation and placed along the lab x-axis, this
    #                    also makes the initial moment point exactly along
    #                    the bond -- a convenient, easy-to-interpret case
    #                    for checking the bond-direction spring (§4b in the
    #                    README) visually, since the gold moment arrow
    #                    should then track the bond line directly.
    MOMENT_MODE = 'axis_aligned'   # 'random' | 'axis_aligned'

    if MOMENT_MODE == 'axis_aligned':
        m_body_local = np.array([1.0, 0.0, 0.0])
    elif MOMENT_MODE == 'random':
        rng          = np.random.default_rng(42)
        m_body_local = rng.normal(size=3)
        m_body_local /= np.linalg.norm(m_body_local)
    else:
        raise ValueError(
            f"Unknown MOMENT_MODE {MOMENT_MODE!r}; expected 'random' or "
            f"'axis_aligned'."
        )

    # ── Interaction (steric) parameters ───────────────────────────────────────
    firm_delta          = 1e-2
    debye_firm          = 2.0 * a * firm_delta / np.log(10.0)
    repulsion_firm      = 0.0331          # kept from base (see docstring note)
    repulsion_soft      = 0.0             # soft Yukawa off
    debye_soft          = 0.225

    # ── Bond parameters (spring + orientation/'torque' spring) ───────────────
    r0_spring = 2.0 * a + 2.0 * debye_firm
    KAPPA_SPRING_COEFF = 10.0              # translational-spring energy scale
    k_spring  = KAPPA_SPRING_COEFF * kT / (2.0 * debye_firm) ** 2

    KAPPA_THETA_COEFF = 1000.0              # orientation-spring energy scale
    k_theta = KAPPA_THETA_COEFF * kT

    k_dir = k_theta   # bond-DIRECTION spring (bug-fix term, see module
                       # docstring): same energy scale as the orientation
                       # spring by default; set independently if desired.

    # ── Box geometry (open -- no periodicity) ─────────────────────────────────
    L            = np.array([0.0, 0.0, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    # ── Simulation parameters ─────────────────────────────────────────────────
    dt      = 1e-3          # timestep (s)   (per explicit instruction --
                             # see docstring re: stiffness/dt risk)
    t_end   = 20.0          # stop time (s)  (covers 4 field flips at B_dwell=5s)
    n_steps = int(t_end / dt)
    plot_dt = 0.1            # ~seconds of simulated time between saved frames
    n_plot  = max(1, int(round(plot_dt / dt)))   # steps between frames
    solver_tolerance = 5e-3

    # ── Output ────────────────────────────────────────────────────────────────
    if Stoch:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'bonded_dimer_frames_stochastic')
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'bonded_dimer_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f"N = 2 particles (bonded dimer), dt = {dt}, n_steps = {n_steps}, "
          f"t_end = {t_end} s")
    print(f"  MOMENT_MODE = {MOMENT_MODE!r}, m_body_local = {m_body_local}")
    print(f"  mu_dipole = {mu_dipole:.4f} aJ/mT, RB_0 = {RB_0:.4f}, "
          f"B_amp = {B_amp} mT, B_dwell = {B_dwell} s")
    print(f"  k_spring = {k_spring:.2f} aJ/um^2, r0_spring = {r0_spring:.5f} um, "
          f"k_theta = {k_theta:.5f} aJ, k_dir = {k_dir:.5f} aJ")

    # ── Initial positions and orientations ────────────────────────────────────
    # Single-particle wall-equilibrium height (gravity balanced by firm wall
    # repulsion), used so there's no initial transient settling in z:
    #   g = (repulsion_firm/debye_firm) * exp(-(h-contact)/debye_firm)
    contact = a * (1.0 - firm_delta)
    z_eq = contact + debye_firm * np.log(repulsion_firm / (debye_firm * g))

    # Place the pair along x, separated by exactly the spring's rest length,
    # both starting with identical (identity) orientation so the orientation
    # spring begins completely unstressed.
    bodies = [
        Body(location=np.array([-0.5 * r0_spring, 0.0, z_eq]),
             orientation=Rotation.identity()),
        Body(location=np.array([ 0.5 * r0_spring, 0.0, z_eq]),
             orientation=Rotation.identity()),
    ]
    N = len(bodies)

    # Reference bond direction, expressed in EACH particle's own initial
    # body frame (fixed for the whole run, exactly analogous to
    # m_body_local for the permanent moment) -- see the "BUG FIX" note
    # above. Computed from the actual initial configuration rather than
    # assumed, so this stays correct even if the initial placement/
    # orientation above is changed later.
    r0_vec = bodies[1].location - bodies[0].location
    d_hat0 = r0_vec / np.linalg.norm(r0_vec)
    bond_dir_body = np.array([
        bodies[0].orientation.as_matrix().T @ d_hat0,
        bodies[1].orientation.as_matrix().T @ d_hat0,
    ])

    # ── Initialise solver ─────────────────────────────────────────────────────
    solver = pyStokesianDynamics(
        bodies=bodies, a=a, eta=eta,
        periodic_length=L, z_max=z_max_solver,
        debye_length=firm_delta,
    )
    solver.kT                  = kT
    solver.dt                  = dt
    solver.tolerance           = solver_tolerance
    solver.num_rejections_wall = 0
    solver.num_rejections_jump = 0
    solver.Set_R_Mats()

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    t_wall = time.perf_counter()

    for step in range(n_steps):
        t_sim = step * dt
        B_now = B_field_fn(t_sim)

        FT_calc = partial(
            force_torque_calculator,
            a=a, g=g, L=L,
            mu_dipole=mu_dipole, RB_0=RB_0,
            mag_idx=MAG_IDX, m_body_local=m_body_local,
            B_field_fn=B_field_fn, t_sim=t_sim,
            rep_firm=repulsion_firm, deb_firm=debye_firm,
            firm_delta=firm_delta,
            rep_soft=repulsion_soft, deb_soft=debye_soft,
            k_spring=k_spring, r0_spring=r0_spring, k_theta=k_theta,
            k_dir=k_dir, bond_dir_body=bond_dir_body,
        )

        solver.Update_Bodies_Trap(FT_calc, stochastic=Stoch, print_residual=False)
        solver.print_timings()

        if step % n_plot == 0:
            elapsed = time.perf_counter() - t_wall
            print(f"  step {step:7d}/{n_steps}  t={t_sim:.4f} s  "
                  f"B=({B_now[0]:.2f},{B_now[1]:.2f},{B_now[2]:.2f}) mT  "
                  f"elapsed={elapsed:.1f}s")
            plot_frame_3d(bodies, a, m_body_local, MAG_IDX,
                          frame_idx, t_sim, B_now, out_dir)
            frame_idx += 1

    print(f"\nSimulation complete in {time.perf_counter()-t_wall:.1f}s")


# =============================================================================
# Visualisation (3-D, PyVista)
# =============================================================================
def plot_frame_3d(bodies, a, m_body_local, mag_idx, step, t_sim, B_now, out_dir):
    """
    Render the dimer as two transparent spheres:
      - magnetic particle (mag_idx)      : cornflower blue
      - non-magnetic particle            : white
    Each sphere gets three thin arrows (its body-frame triad, i.e. its full
    orientation) coloured by AXIS -- pink/turquoise/purple for axis 0/1/2 --
    the same colour on both particles, so corresponding axes can be compared
    between them. The magnetic particle additionally gets one thick
    gold/orange arrow along its current (world-frame) permanent moment. A
    floating green arrow above the dimer shows the applied field's current
    direction. A thin grey plane at z=0 indicates the wall.
    """
    CORNFLOWER = (0.392, 0.584, 0.929)
    WHITE      = (1.0, 1.0, 1.0)
    GOLD       = (255 / 255.0, 165 / 255.0, 0.0)   # same gold as base script

    # Triad axis colours: same axis index -> same colour on BOTH particles,
    # so corresponding axes can be visually compared/identified between the
    # two bonded particles (e.g. how aligned are their two "pink" axes?).
    PINK      = (1.0, 0.0, 0.45)
    TURQUOISE = (0.251, 0.878, 0.816)
    PURPLE    = (0.5, 0.0, 0.5)
    TRIAD_COLORS = (PINK, TURQUOISE, PURPLE)   # indexed by axis 0, 1, 2

    FIELD_COLOR = (0.13, 0.55, 0.13)   # floating applied-field arrow (green)

    thin_len  = 0.85 * a
    thick_len = 1.15 * a

    plotter = pv.Plotter(off_screen=True, window_size=[900, 900])
    plotter.set_background('white')

    locs   = np.array([b.location for b in bodies])
    centre = locs.mean(axis=0)

    # wall marker
    floor = pv.Plane(center=(centre[0], centre[1], 0.0),
                      direction=(0, 0, 1),
                      i_size=12 * a, j_size=12 * a)
    plotter.add_mesh(floor, color=(0.85, 0.85, 0.85), opacity=0.4)

    for k, b in enumerate(bodies):
        center = b.location
        color  = CORNFLOWER if k == mag_idx else WHITE

        sphere = pv.Sphere(radius=a, center=center,
                            theta_resolution=48, phi_resolution=48)
        plotter.add_mesh(sphere, color=color, opacity=0.35,
                          specular=0.4, smooth_shading=True)

        R = b.orientation.as_matrix()
        for axis in range(3):
            d = R[:, axis]
            arrow = pv.Arrow(start=center - 0.5 * thin_len * d, direction=d,
                              scale=thin_len,
                              tip_length=0.25, tip_radius=0.05,
                              shaft_radius=0.02)
            plotter.add_mesh(arrow, color=TRIAD_COLORS[axis])

        if k == mag_idx:
            m_dir = R @ m_body_local
            norm  = np.linalg.norm(m_dir)
            if norm > 1e-12:
                m_dir = m_dir / norm
                arrow = pv.Arrow(start=center - 0.5 * thick_len * m_dir,
                                  direction=m_dir, scale=thick_len,
                                  tip_length=0.3, tip_radius=0.15,
                                  shaft_radius=0.07)
                plotter.add_mesh(arrow, color=GOLD)

    # floating arrow showing the applied field's direction, offset above the
    # dimer (along +z) so it never overlaps the spheres/bond
    B_norm = np.linalg.norm(B_now)
    if B_norm > 1e-12:
        b_hat        = B_now / B_norm
        field_len    = 2.0 * a
        field_offset = 3.5 * a
        field_base   = centre + np.array([0.0, 0.0, field_offset])
        arrow = pv.Arrow(start=field_base - 0.5 * field_len * b_hat,
                          direction=b_hat, scale=field_len,
                          tip_length=0.3, tip_radius=0.12,
                          shaft_radius=0.05)
        plotter.add_mesh(arrow, color=FIELD_COLOR)

    plotter.add_text(
        f"t = {t_sim:.3f} s   B = ({B_now[0]:.2f}, {B_now[1]:.2f}, "
        f"{B_now[2]:.2f}) mT",
        position='upper_left', font_size=14, color='black')

    cam_dist = 6.0 * a
    plotter.camera_position = [
        (centre[0] + cam_dist, centre[1] - cam_dist, centre[2] + 0.6 * cam_dist),
        tuple(centre),
        (0, 0, 1),
    ]
    plotter.enable_anti_aliasing()

    fname = os.path.join(out_dir, f'frame_{step:07d}.png')
    plotter.screenshot(fname)
    plotter.close()


# =============================================================================
# Force / torque calculator
# =============================================================================
def force_torque_calculator(bodies, r_vecs, **kwargs):
    """
    Returns (2N, 3) array of [force_i, torque_i] for each body (N=2).

    Forces:
      - Gravity + firm wall repulsion  (single-body, z direction)
      - Pair steric repulsion (firm + soft Yukawa)
      - Bond spring between the two particles (harmonic, rest length
        r0_spring)
      - Bond-direction spring's force term (see module docstring "BUG FIX")

    Torques:
      - External magnetic torque tau = m_total x B(t), applied ONLY to the
        particle at index mag_idx (the other has zero moment => zero torque)
      - Orientation ('torque') spring between the two particles, penalising
        R_i != R_j (see module docstring for the derivation)
      - Bond-direction spring, penalising each particle's body-fixed
        reference direction differing from the actual bond vector r_hat
        (see module docstring "BUG FIX" -- this is what keeps the permanent
        moment from drifting relative to the line connecting the spheres)
    """
    a          = kwargs['a']
    g          = kwargs['g']
    L          = kwargs['L']
    mu_dipole  = kwargs['mu_dipole']
    RB_0       = kwargs['RB_0']
    mag_idx    = kwargs['mag_idx']
    m_body_local = kwargs['m_body_local']
    B_field_fn = kwargs['B_field_fn']
    t_sim      = kwargs['t_sim']
    rep_firm   = kwargs['rep_firm']
    deb_firm   = kwargs['deb_firm']
    firm_delta = kwargs['firm_delta']
    rep_soft   = kwargs['rep_soft']
    deb_soft   = kwargs['deb_soft']
    k_spring   = kwargs['k_spring']
    r0_spring  = kwargs['r0_spring']
    k_theta    = kwargs['k_theta']
    k_dir      = kwargs['k_dir']
    bond_dir_body = kwargs['bond_dir_body']

    N = len(bodies)
    r = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3)
    R_mats = np.array([b.orientation.as_matrix() for b in bodies])   # (N,3,3)

    # ── gravity + wall + steric repulsion (numba) ─────────────────────────────
    force, torque = _wall_and_steric_forces(
        r, L, a, g, rep_firm, deb_firm, firm_delta, rep_soft, deb_soft)

    # ── bond spring (explicit pair 0,1) ───────────────────────────────────────
    i, j = 0, 1
    dr = r[j] - r[i]
    r_norm = np.linalg.norm(dr)
    if r_norm > 1e-12:
        r_hat = dr / r_norm
        F_bond = k_spring * (r_norm - r0_spring) * r_hat   # on i, toward j if stretched
        force[i] += F_bond
        force[j] -= F_bond

    # ── orientation ('torque') spring (explicit pair 0,1) ─────────────────────
    R_i, R_j = R_mats[i], R_mats[j]
    tau = np.zeros(3)
    for axis in range(3):
        tau += k_theta * np.cross(R_i[:, axis], R_j[:, axis])
    torque[i] += tau
    torque[j] -= tau

    # ── bond-DIRECTION spring (explicit pair 0,1) -- BUG FIX ──────────────────
    # Ties each particle's body-fixed reference direction to the actual bond
    # vector r_hat, not just to each other's orientation. Without this, R_i
    # and R_j can stay mutually aligned (satisfying the spring above) while
    # BOTH drift together relative to r_hat itself, since nothing else
    # constrains r_hat's direction (the translational spring only fixes
    # |r|) -- this was the reported bug (the permanent moment, locked to
    # R_mag, visibly wandering relative to the line connecting the spheres).
    # See module docstring for the derivation.
    if r_norm > 1e-12:
        e_i = R_i @ bond_dir_body[0]
        e_j = R_j @ bond_dir_body[1]

        torque[i] += k_dir * np.cross(e_i, r_hat)
        torque[j] += k_dir * np.cross(e_j, r_hat)

        perp_i = e_i - np.dot(e_i, r_hat) * r_hat
        perp_j = e_j - np.dot(e_j, r_hat) * r_hat
        F_dir = (k_dir / r_norm) * (perp_i + perp_j)   # force on j; -F_dir on i
        force[j] += F_dir
        force[i] -= F_dir

    # ── external magnetic field: force-free (uniform field), torque only,
    #    and only on the magnetic particle ────────────────────────────────────
    B_now = np.asarray(B_field_fn(t_sim), dtype=np.float64)
    R_mag = R_mats[mag_idx]
    m_perm_world = mu_dipole * (R_mag @ m_body_local)
    m_induced    = RB_0 * B_now
    m_total      = m_perm_world + m_induced
    torque[mag_idx] += np.cross(m_total, B_now)

    FT = np.zeros((2 * N, 3))
    FT[0::2] = force
    FT[1::2] = torque
    return FT


@njit(parallel=True, fastmath=True)
def _wall_and_steric_forces(r, L, a, g,
                             rep_firm, deb_firm, firm_delta,
                             rep_soft, deb_soft):
    """
    Numba kernel: gravity + firm wall repulsion (single-body) and pairwise
    steric repulsion (firm contact + optional soft Yukawa). No magnetic or
    bond physics here -- those are added in force_torque_calculator, since
    with only N=2 particles (and only one ever magnetic) they reduce to a
    single explicit pair rather than a general pairwise sum.
    """
    N      = r.shape[0]
    force  = np.zeros((N, 3))
    torque = np.zeros((N, 3))

    # ── single-body: gravity + wall repulsion ────────────────────────────────
    for i in prange(N):
        force[i, 2] -= g
        h       = r[i, 2]
        contact = a * (1.0 - firm_delta)
        if h > contact:
            force[i, 2] += (rep_firm / deb_firm) * np.exp(-(h - contact) / deb_firm)
        else:
            force[i, 2] += rep_firm / deb_firm

    # ── pairwise steric repulsion ──────────────────────────────────────────────
    for i in prange(N):
        for j in range(N):
            if i == j:
                continue

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r[j, k] - r[i, k]
                if L[k] > 0:
                    dr[k] -= int(dr[k] / L[k] + 0.5 * (
                        int(dr[k] > 0) - int(dr[k] < 0))) * L[k]

            r_norm = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            if r_norm < 1e-12:
                continue

            offset_firm = 2.0 * a * (1.0 - firm_delta)
            for k in range(3):
                if r_norm > offset_firm:
                    force[i, k] += -(
                        (rep_firm / deb_firm) *
                        np.exp(-(r_norm - offset_firm) / deb_firm) / r_norm
                    ) * dr[k]
                else:
                    force[i, k] += -(rep_firm / deb_firm / r_norm) * dr[k]
                if rep_soft > 0.0:
                    offset_soft = 2.0 * a
                    if r_norm > offset_soft:
                        force[i, k] += -(
                            (rep_soft / deb_soft) *
                            np.exp(-(r_norm - offset_soft) / deb_soft) / r_norm
                        ) * dr[k]
                    else:
                        force[i, k] += -(rep_soft / deb_soft / r_norm) * dr[k]

    return force, torque


# =============================================================================
if __name__ == '__main__':
    main()
