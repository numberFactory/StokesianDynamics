"""
ladder_to_ring_EM_force.py
--------------------------
Simulation of magnetic microparticles driven by a rotating in-plane field
plus a static vertical (z) field, reproducing the ladder-to-ring transition
experiment.

Particles are single-blob rigid bodies.  Forces include:
  - Gravity + firm wall repulsion (z direction)
  - Pair magnetic dipole interactions (in-plane rotating field)
  - Pair electric-field-induced dipole interactions (static z field)
  - Pair steric repulsion (firm contact + soft Yukawa)
  - External magnetic torque (B X m)

Each n_plot steps a snapshot PNG is saved showing particles as 2-D discs
with an orientation arrow (projection of the body x-axis into the xy plane).

Time-dependent protocol:  B_z is stepped up at specified simulation times
to drive the chain-to-ring transition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from functools import partial
from numba import njit, prange
from scipy.spatial.transform import Rotation
import scipy.sparse.linalg as spla
import scipy.spatial

from body import Body
from pyStokesianDynamics import pyStokesianDynamics


# =============================================================================
def main():
# =============================================================================

    # TODO: Plot induce moment as well as the scaled external field (what you would get without mutual polarizability) to see how much the mutual polarizability is affecting the induced moment.
    # Check, after ring is formed, can z be turned off? Try with and without permanent dipole moment. 
    # # # i.e Start with the ring formed, then turn off the z field and see if it stays in the ring configuration. Try with and without permanent dipole moment.

    # ── Physical parameters ───────────────────────────────────────────────────
    a   = 2.25          # particle radius (µm)
    eta = 8.9e-4        # fluid viscosity
    kT  = 0.0041419464  # thermal energy
    g   = 0.28041       # gravitational acceleration (z)
    Stoch = False        # include stochastic forces/torques (Brownian motion)

    # ── Magnetic / electric field parameters ─────────────────────────────────
    mu_dipole  = 1.0*8.6    # permanent dipole (aJ/mT)
    B_0        = 0.95 #0.82 #0.92   # in-plane rotating field amplitude (mT)
    B_freq     = 5.0   # rotation frequency (Hz)
    chi_exp    = 1.27   # magnetic susceptibility
    RB_0       = 3.3333333 * (a**3) * chi_exp  # susceptibility prefactor (has susceptibility hard-coded in (aJ/mT))
    # Below is how RB_0 was caclulated.
    # https://www.wolframalpha.com/input?i=convert+4*pi*%282.25+um%29%5E3+*+1.27+*%281+Militesla%29%2F%283*%284*pi*1e-7%29+Henry%2Fm%29+in+attoJoules%2FMilitessla



    C          = 0.3    # dipole-dipole coupling constant (in-plane)
    # Below is how C was caclulated
    # https://www.wolframalpha.com/input?i=%283%2F%284*pi%29%29*+%281+attoJoule+%2Fmillitesla+%29%5E2+*+%284*pi*1e-7+Henry%2Fm%29+%2F+%281+um%29%5E4+to+pN
    C_z        = 0.3    # dipole-dipole coupling constant (z field)

    # ── Mutual polarizability (magnetic, in-plane field, only) ────────────────
    # If True, the induced in-plane moment of every particle is solved for
    # self-consistently via GMRES, accounting for the fact that each
    # particle's induced moment also responds to the dipole field of every
    # OTHER particle's total (permanent + induced) moment -- not just the
    # externally applied field. If False (default), induced moments use the
    # simple linear response to the applied field alone, exactly as before.
    # The effective z-field (E_mom) is left as simple linear response either
    # way, per current scope.
    mutual_polarizability_mag = False
    gmres_rtol     = 1e-6
    gmres_restart  = 100
    gmres_maxiter  = 100

    # ── Time-dependent B_z protocol ───────────────────────────────────────────
    # List of (time_s, B_z_mT) steps; B_z takes the value from the last
    # entry whose time <= current simulation time.
  
    # B_z_schedule = [
    #     (0.0,  0.0),
    #     (30.0,  0.535),
    #     (50.0,  0.545),
    #     (100.0, 0.0),
    # ]
    # t_end = 120.0        # stop time (s)

    B_z_schedule = [
        (0.0,  0.0),
        (80.0,  0.535),
        (100.0,  0.545),
        (120.0,  0.555),
        (140.0, 0.0),
        (150.0, 0.535),
        (160.0,  0.545),
        (180.0,  0.555),
        (240.0,  0.0),
    ]
    t_end = 260.0        # stop time (s)

    # ── Interaction parameters ────────────────────────────────────────────────
    firm_delta          = 1e-3
    debye_firm          = 2.0 * a * firm_delta / np.log(10.0)
    repulsion_firm      = 0.0331
    repulsion_soft      = 0.0    # soft Yukawa off
    debye_soft          = 0.225

    # ── Box geometry (open — no periodicity) ─────────────────────────────────
    L            = np.array([0.0, 0.0, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    # ── Initial positions and orientations (from suspension_ladder_N_6_random.clones)
    # Format: x y z qw qx qy qz  (scalar-first quaternion)
    # ── Place particles ───────────────────────────────────────────────────────
    N = 1000
    phi = 0.6
    A_box   = N * np.pi * a**2 / phi
    L_xy    = float(np.sqrt(A_box))
    print(f"N={N}, phi={phi:.2f}")
    print(f"Box: Lx=Ly={L_xy:.4f} µm")


    print("Placing particles ...")
    t0_place = time.perf_counter()
    g_place = g
    positions = place_particles(N, a, kT, g_place, 0.99*L_xy, 0.99*L_xy, seed=42)
    print(f"  done in {time.perf_counter()-t0_place:.1f}s  "
          f"z range: [{positions[:,2].min():.4f}, {positions[:,2].max():.4f}] µm")

    # Random orientations
    rng = np.random.default_rng()
    bodies = []
    for pos in positions:
        theta_z = rng.uniform(0.0, 2.0 * np.pi)
        ori = Rotation.from_euler('z', theta_z)
        bodies.append(Body(location=pos.copy(), orientation=ori))

    # ── Output ────────────────────────────────────────────────────────────────
    if Stoch:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mag_anneal_N_'+str(N)+'_stochastic_frames')
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mag_anneal_N_'+str(N)+'_frames')
    os.makedirs(out_dir, exist_ok=True)

    # ── Rotating field direction (user-defined function of time) ─────────────
    # m_rot_fn(t_sim) must return a 3-vector giving the field direction.
    # Default matches the original code: Lissajous pattern with
    # x-frequency = B_freq, y-frequency = 2*B_freq  (Omega_y = 2*Omega).
    B_freq_y = B_freq
    def m_rot_fn(t):
        return np.array([np.cos(2 * np.pi * B_freq   * t),
                         np.sin(2 * np.pi * B_freq_y * t),
                         0.0])
    
    # ── Simulation parameters ─────────────────────────────────────────────────
    dt      = 0.025*(1/B_freq_y) #6.25e-5   # timestep (s)
    n_steps = int(t_end / dt)
    n_plot  = 10*16+1       # plot frame every n_plot steps
    solver_tolerance = 5e-3

    print(f"N = {N} particles, dt = {dt}, n_steps = {n_steps}, t_end = {t_end} s")

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

    # ── Buffered neighbor list (optional; original O(N^2) path is left
    # completely untouched below and can still be selected instead) ──────────
    # Cutoff: torque decays as 1/r^3 (slower than force's 1/r^4), so it's
    # the binding constraint here -- sized so a single pair's torque
    # contribution at r_cut is `tolerance` of what it'd be at contact
    # (r=2a): (r_cut/2a)^-3 = tolerance.
    #
    # IMPORTANT, found by actually testing this (see test_neighborlist.py):
    # per-pair tolerance does NOT directly bound the aggregate error in the
    # total force/torque on a particle, which sums over every neighbor
    # inside r_cut, not just one. On a representative N=400, phi=0.6
    # configuration: at tolerance=0.01 (this file's original default),
    # measured aggregate error was ~2.5% (force) / ~9% (torque). At the
    # tolerance=1e-3 used below, that drops to roughly ~0.2% (force) / ~2.4%
    # (torque), with ~161 average neighbors (vs N-1=499 for the full
    # kernel) -- still a substantial pair-count reduction, tighter than the
    # 1e-2 default but well short of the ~1e-4 needed to get torque error
    # under 1% (which pushes r_cut out far enough that the neighbor count
    # approaches N-1 again, undoing most of the speedup). Re-run
    # test_neighborlist.py's tolerance sweep on your actual configuration
    # if you need a number you trust rather than the one measured here.
    use_neighbor_list = True
    nlist_tolerance   = 1e-3               # applied to torque's 1/r^3 decay
    r_cut             = 2.0 * a * (nlist_tolerance ** (-1.0/3.0))
    nlist_buffer      = 0.5 * a            # skin; tune based on observed rebuild frequency
    r_list_cut        = r_cut + nlist_buffer
    print(f"Neighbor list: r_cut={r_cut:.3f} um ({r_cut/a:.3f}*a), "
          f"buffer={nlist_buffer:.3f} um, r_list={r_list_cut:.3f} um")

    if use_neighbor_list:
        r0_nlist = np.array([b.location for b in bodies])
        offsets, neighbor_list = build_neighbor_list_3d(r0_nlist, r_list_cut, L)
        print(f"  initial: {len(neighbor_list)/N:.1f} avg neighbors "
              f"(vs {N-1} for full O(N^2))")
        n_rebuilds = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    t_wall = time.perf_counter()

    for step in range(n_steps):
        t_sim = step * dt

        # determine current B_z from schedule
        B_z = B_z_schedule[0][1]
        for t_thresh, bz_val in B_z_schedule:
            if t_sim >= t_thresh:
                B_z = bz_val

        if use_neighbor_list:
            r_now = np.array([b.location for b in bodies])
            if _max_displacement(r_now, r0_nlist) > 0.5 * nlist_buffer:
                offsets, neighbor_list = build_neighbor_list_3d(r_now, r_list_cut, L)
                r0_nlist = r_now
                n_rebuilds += 1

            FT_calc = partial(
                force_torque_calculator_nlist,
                a=a, g=g, L=L,
                B_0=B_0, B_z=B_z,
                mu_dipole=mu_dipole, RB_0=RB_0,
                chi_exp=chi_exp, C=C, C_z=C_z,
                rep_firm=repulsion_firm, deb_firm=debye_firm,
                firm_delta=firm_delta,
                rep_soft=repulsion_soft, deb_soft=debye_soft,
                t_sim=t_sim,
                m_rot_fn=m_rot_fn,
                offsets=offsets, neighbor_list=neighbor_list,
                mutual_polarizability_mag=mutual_polarizability_mag,
                gmres_rtol=gmres_rtol,
                gmres_restart=gmres_restart,
                gmres_maxiter=gmres_maxiter,
            )
        else:
            FT_calc = partial(
                force_torque_calculator,
                a=a, g=g, L=L,
                B_0=B_0, B_z=B_z,
                mu_dipole=mu_dipole, RB_0=RB_0,
                chi_exp=chi_exp, C=C, C_z=C_z,
                rep_firm=repulsion_firm, deb_firm=debye_firm,
                firm_delta=firm_delta,
                rep_soft=repulsion_soft, deb_soft=debye_soft,
                t_sim=t_sim,
                m_rot_fn=m_rot_fn,
                mutual_polarizability_mag=mutual_polarizability_mag,
                gmres_rtol=gmres_rtol,
                gmres_restart=gmres_restart,
                gmres_maxiter=gmres_maxiter,
            )

        solver.Update_Bodies_Trap(FT_calc, stochastic=Stoch, print_residual=False)
        solver.print_timings()

        # ── Per-step diagnostics ──────────────────────────────────────────────
        # r_now = np.array([b.location for b in bodies])
        # min_sep = _min_separation(r_now)
        # min_z   = r_now[:, 2].min()
        # if not np.all(np.isfinite(r_now)) or min_sep < 0.5*a or min_z < 0.1*a:
        #     print(f"\n*** GEOMETRY WARNING at step {step}, t={t_sim:.5f} ***")
        #     print(f"  min pair separation = {min_sep:.4e}  (2a = {2*a:.4f})")
        #     print(f"  min z               = {min_z:.4e}  (a = {a:.4f})")
        #     print(f"  positions:\n{r_now}")

        if step % n_plot == 0:
            elapsed = time.perf_counter() - t_wall
            print(f"  step {step:7d}/{n_steps}  t={t_sim:.4f} s  "
                  f"B_z={B_z:.3f} mT  elapsed={elapsed:.1f}s")

            # Recompute moments fresh at the just-updated configuration, purely
            # for visualization (see plot_frame's pink/sea-green arrows below).
            # Cheap: only runs every n_plot steps, and reuses the exact same
            # mutual-polarizability logic force_torque_calculator itself uses
            # (via _compute_moments), so this can't drift out of sync with it.
            r_now = np.array([b.location for b in bodies])
            _, induced_mom_now, B_applied_now = _compute_moments(
                bodies, r_now, t_sim, m_rot_fn,
                mu_dipole, B_0, RB_0, chi_exp, L, a,
                mutual_polarizability_mag, gmres_rtol, gmres_restart, gmres_maxiter,
            )

            plot_frame(bodies, a, frame_idx, t_sim, B_z, out_dir,
                       B_applied=B_applied_now, induced_mom=induced_mom_now, RB_0=RB_0)
            frame_idx += 1

    if use_neighbor_list:
        print(f"Neighbor list rebuilt {n_rebuilds} times over {n_steps} steps "
              f"({n_rebuilds/max(n_steps,1):.4f} rebuilds/step) -- "
              f"raise nlist_buffer if this is close to 1, lower it if it's near 0.")
    print(f"\nSimulation complete in {time.perf_counter()-t_wall:.1f}s")


# =============================================================================
# Visualisation
# =============================================================================
def plot_frame(bodies, a, step, t_sim, B_z, out_dir,
               B_applied=None, induced_mom=None, RB_0=None):
    """
    Plot particles as 2-D discs with orientation arrows projected into xy plane.
    Cornflower-blue fill, blue edge (lw=3.5).  Gold arrow (lw=4.5) shows the
    projection of the body x-axis (first column of rotation matrix) into xy,
    i.e. the permanent moment direction.

    If B_applied / induced_mom / RB_0 are given, two more arrows are drawn
    per particle, both the same style/length convention as the gold one
    (raw xy-projection of a 3-vector, scaled by `a*0.75` -- so an
    out-of-plane component shortens the apparent arrow, same as the gold
    arrow already does):

      - pink (1, 0, 0.45): direction of the externally applied in-plane
        field, B_applied (same for every particle -- it's a uniform field).
        Normalized to a fixed reference length (same length as the gold
        arrow) since B_applied's own magnitude isn't otherwise meaningful
        to compare against a moment's length.
      - sea green: the induced moment, scaled so that an induced moment
        with the same magnitude as RB_0*B_applied (i.e. what you'd get
        WITHOUT mutual polarizability) renders at exactly the same length
        as the pink arrow. Magnitudes above/below that reference make the
        arrow longer/shorter accordingly, so the sea-green-vs-pink length
        ratio is a direct visual readout of how much mutual polarizability
        is boosting or suppressing the induced moment relative to the
        simple single-particle response.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Axis limits, aspect, and layout are set FIRST (before drawing anything)
    # so the points-per-data-unit conversion right below is computed against
    # the actual final layout, not a stale one -- see _points_per_data_unit.
    locs  = np.array([b.location for b in bodies])
    pad   = 1.5 * a
    ax.set_xlim(locs[:, 0].min() - pad, locs[:, 0].max() + pad)
    ax.set_ylim(locs[:, 1].min() - pad, locs[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x (µm)', fontsize=12)
    ax.set_ylabel('y (µm)', fontsize=12)
    ax.set_title(f't = {t_sim:.4f} s   B_z = {B_z:.3f} mT', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Circle-edge and arrow linewidths, scaled to the particle's own
    # RENDERED size rather than a fixed point count. matplotlib linewidths
    # are in points -- a fixed physical size on the output image -- so at
    # fixed figsize/dpi, cramming more particles into the same view means
    # each one renders smaller while lw=3.5/4.5 stays exactly as thick,
    # until arrows and edges swamp the particles they're drawn on. Basing
    # both on the particle's actual apparent radius (in points, for THIS
    # frame's zoom level) keeps them a fixed fraction of the particle's
    # rendered size at any N -- clipped to the old fixed values (3.5, 4.5)
    # as an upper bound, so small-N frames (particle radius large on
    # screen) look identical to before, and to a small floor so lines never
    # vanish entirely at very high N.
    points_per_unit = _points_per_data_unit(fig, ax)
    particle_radius_points = a * points_per_unit
    edge_lw  = np.clip(0.12 * particle_radius_points, 0.3, 3.5)
    arrow_lw = np.clip(0.22 * particle_radius_points, 0.4, 4.5)

    gold      = np.array([255, 165, 0]) / 255.0
    pink      = np.array([1.0, 0.0, 0.45])
    sea_green = 'seagreen'

    scale = a * 0.75   # arrow length convention shared by all three arrows

    # Reference field direction (unit vector) and reference magnitude
    # RB_0*|B_applied|, both needed before the per-particle loop since
    # B_applied is uniform (same value for every particle).
    field_dir_xy = None
    M_ref = None
    if B_applied is not None:
        B0_row = np.asarray(B_applied[0], dtype=np.float64)
        B0_norm = np.linalg.norm(B0_row)
        if B0_norm > 1e-12:
            field_dir_xy = B0_row[:2] / B0_norm
        if RB_0 is not None:
            M_ref = RB_0 * B0_norm

    for idx, b in enumerate(bodies):
        x, y = b.location[0], b.location[1]

        # disc
        circle = plt.Circle((x, y), a,
                             facecolor='cornflowerblue',
                             edgecolor='steelblue',
                             linewidth=edge_lw, zorder=2)
        ax.add_patch(circle)

        # orientation: first column of rotation matrix = body x-axis in lab frame
        R      = b.orientation.as_matrix()
        x_body = R[:, 0]                  # 3-D body x-axis in lab frame
        # project into xy plane; z-component reduces apparent arrow length
        dx     = x_body[0]
        dy     = x_body[1]

        ax.annotate('', xy=(x + scale * dx, y + scale * dy),
                    xytext=(x - scale * dx, y - scale * dy),
                    arrowprops=dict(arrowstyle='->', color=gold,
                                   lw=arrow_lw), zorder=3)

        # pink: external (applied) field direction, fixed reference length
        if field_dir_xy is not None:
            fdx, fdy = field_dir_xy[0], field_dir_xy[1]
            ax.annotate('', xy=(x + scale * fdx, y + scale * fdy),
                        xytext=(x - scale * fdx, y - scale * fdy),
                        arrowprops=dict(arrowstyle='->', color=pink,
                                       lw=arrow_lw), zorder=3)

        # sea green: induced moment, length relative to RB_0*|B_applied|
        if induced_mom is not None and M_ref is not None and M_ref > 1e-12:
            v = np.asarray(induced_mom[idx], dtype=np.float64)
            vx = v[0] * (scale / M_ref)
            vy = v[1] * (scale / M_ref)
            ax.annotate('', xy=(x + vx, y + vy),
                        xytext=(x - vx, y - vy),
                        arrowprops=dict(arrowstyle='->', color=sea_green,
                                       lw=arrow_lw), zorder=3)

    fname = os.path.join(out_dir, f'frame_{step:07d}.png')
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def _points_per_data_unit(fig, ax):
    """
    How many display POINTS (matplotlib's fixed physical linewidth unit,
    1/72 inch, same as PostScript points) correspond to one unit of DATA
    space on the x-axis, for the CURRENT figure layout and axis limits.
    Used by plot_frame to size linewidths as a genuine fraction of a
    particle's rendered radius rather than a fixed point count regardless
    of how many particles (and therefore how small each one renders) are
    in the frame. Requires xlim/ylim/aspect/tight_layout to already be set
    on `ax` -- call this after that, not before.
    """
    fig.canvas.draw()   # force a layout pass so get_window_extent() is accurate
    bbox_pixels = ax.get_window_extent()
    bbox_points = bbox_pixels.width * 72.0 / fig.dpi
    xlim = ax.get_xlim()
    data_width = xlim[1] - xlim[0]
    return bbox_points / data_width if data_width > 0 else 1.0


# =============================================================================
# Particle placement
# =============================================================================
@njit(fastmath=True)
def _check_overlaps(positions, x, y, z, min_sep_sq, n_placed, Lx, Ly):
    """Serial overlap check with minimum-image wrapping in x and y."""
    for i in range(n_placed):
        dx = x - positions[i, 0]
        dy = y - positions[i, 1]
        dz = z - positions[i, 2]
        if Lx > 0.0:
            dx -= round(dx / Lx) * Lx
        if Ly > 0.0:
            dy -= round(dy / Ly) * Ly
        if dx*dx + dy*dy + dz*dz < min_sep_sq:
            return True   # early exit on first overlap found
    return False

def place_particles(N, a, kT, g, Lx, Ly, seed=42,
                    retry_increase=0.01, max_failures_before_increase=None):
    """
    Place N non-overlapping particles above a wall.
    z ~ a + Exponential(kT/g).  If placement fails repeatedly the mean
    gap is increased by retry_increase to relax crowding.
    """
    if max_failures_before_increase is None:
        max_failures_before_increase = 10 * N
    rng        = np.random.default_rng(seed)
    min_sep_sq = (2.0 * a) ** 2
    z_mean     = kT / g
    positions  = np.empty((N, 3), dtype=np.float64)
    n_placed   = 0
    total_z_increases = 0
    while n_placed < N:
        failures = 0
        placed   = False
        while not placed:
            x        = rng.uniform(0.0, Lx if Lx > 0 else 1.0)
            y        = rng.uniform(0.0, Ly if Ly > 0 else 1.0)
            z        = a + rng.exponential(z_mean)
            overlaps = False if n_placed == 0 else \
                _check_overlaps(positions, x, y, z, min_sep_sq, n_placed, Lx, Ly)
            if not overlaps:
                positions[n_placed] = [x, y, z]
                n_placed += 1
                placed    = True
            else:
                failures += 1
                if failures % max_failures_before_increase == 0:
                    z_mean *= (1.0 + retry_increase)
                    total_z_increases += 1
    if total_z_increases > 0:
        print(f"Placement: z_mean increased {total_z_increases} times "
              f"(final z_mean = {z_mean:.4f})")
    return positions


# =============================================================================
# Mutual polarizability (magnetic, in-plane field only)
# =============================================================================
@njit(fastmath=True, parallel=True)
def _magnetic_mobility_numba(moments, r, L, a, chi, RB_0):
    """
    Numba matvec kernel for the magnetic-mobility linear operator A, where

        (A @ x)_i  =  x_i / RB_0  +  sum_{j != i} K_ij @ x_j

    K_ij is the near-field-regularized point-dipole field kernel per unit
    moment (far field: standard dipole tensor scaled by chi/3 * a^3 / r^3;
    near field, r <= 2a: a regularized polynomial, both following the same
    construction as Magnetic_Mobility in multi_bodies_functions.py).

    The x_i/RB_0 "self" term is intentional, not a bug: it is what makes A
    represent "the trial moment's own single-particle response, plus the
    mutual dipole coupling from every other particle" -- exactly what's
    needed so that solving A @ m_ind = RHS gives the self-consistent induced
    moment (see _solve_induced_moments_mutual and the docstring there for
    how RHS is built so the permanent moments enter correctly).

    One deliberate change from the reference implementation: the near/far
    field switch there compares the raw separation to the literal number
    "2" (i.e., 2 micrometers, given a=2.25 here) rather than "2*a" (~4.5
    micrometers, i.e., particle diameter) -- since the whole point of the
    near-field regularization is to handle particles at or near contact,
    and steric repulsion in this system only prevents r from going below
    about 2a*(1 - firm_delta) =~ 4.49 (not 2), that threshold would never
    activate the near-field branch at all for this system's own contact
    distance. Using r_norm > 2*a here instead so the regularization
    actually engages exactly where two particles are near contact.
    """
    N = r.shape[0]
    B_net = np.zeros((N, 3))
    chi_over_three = chi / 3.0
    a3 = a * a * a
    inva = 1.0 / a
    inva3 = 1.0 / a3

    periodic = np.array([L[0] > 0, L[1] > 0, L[2] > 0])

    for i in prange(N):
        B_net[i, 0] += moments[i, 0] / RB_0
        B_net[i, 1] += moments[i, 1] / RB_0
        B_net[i, 2] += moments[i, 2] / RB_0

        for j in range(N):
            if i == j:
                continue

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r[i, k] - r[j, k]
                if periodic[k]:
                    dr[k] -= round(dr[k] / L[k]) * L[k]

            r_norm = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            if r_norm < 1e-12:
                continue
            invr  = 1.0 / r_norm
            invr2 = invr * invr
            invr3 = invr * invr2

            if r_norm > 2.0 * a:
                c1 = 1.0
                c2 = -3.0 * invr2
                Mxx = (c1 + c2*dr[0]*dr[0]) * invr3 * a3 * chi_over_three
                Mxy = (      c2*dr[0]*dr[1]) * invr3 * a3 * chi_over_three
                Mxz = (      c2*dr[0]*dr[2]) * invr3 * a3 * chi_over_three
                Myy = (c1 + c2*dr[1]*dr[1]) * invr3 * a3 * chi_over_three
                Myz = (      c2*dr[1]*dr[2]) * invr3 * a3 * chi_over_three
                Mzz = (c1 + c2*dr[2]*dr[2]) * invr3 * a3 * chi_over_three
            else:
                r3 = r_norm * r_norm * r_norm
                c1 = chi_over_three*(1.0 - 0.5625*r_norm*inva + 0.03125*r3*inva3)
                c2 = chi_over_three*(0.09375*r3*inva3 - 0.5625*r_norm*inva) * invr2
                Mxx = c1 + c2*dr[0]*dr[0]
                Mxy =      c2*dr[0]*dr[1]
                Mxz =      c2*dr[0]*dr[2]
                Myy = c1 + c2*dr[1]*dr[1]
                Myz =      c2*dr[1]*dr[2]
                Mzz = c1 + c2*dr[2]*dr[2]
            Myx, Mzx, Mzy = Mxy, Mxz, Myz

            B_net[i, 0] += (Mxx*moments[j,0] + Mxy*moments[j,1] + Mxz*moments[j,2]) / RB_0
            B_net[i, 1] += (Myx*moments[j,0] + Myy*moments[j,1] + Myz*moments[j,2]) / RB_0
            B_net[i, 2] += (Mzx*moments[j,0] + Mzy*moments[j,1] + Mzz*moments[j,2]) / RB_0

    return B_net


def _solve_induced_moments_mutual(mom_perm, B_applied, r, L, a, chi_exp, RB_0,
                                   rtol=1e-6, restart=50, maxiter=100, x0=None):
    """
    Self-consistently solve for the induced in-plane moment of every
    particle, accounting for mutual dipole coupling.

    Physics / how the permanent moment enters
    -------------------------------------------
    Define the (self-excluded) mutual dipole-field operator D, where D[x]_i
    is the field at particle i from every OTHER particle's moment x_j (using
    the same kernel as _magnetic_mobility_numba, minus its identity self
    term). The true local field driving each particle's induced response is

        B_loc,i = B_applied,i - D[m_perm + m_ind]_i

    (the minus sign here matches this kernel's own internal sign convention;
    see the note in the module docstring below -- it is physically the "+"
    you'd expect from the standard dipole-field formula, since this kernel's
    Mxx/Mxy/... are built with the opposite overall sign from the textbook
    dipole tensor). The induced moment is the single-particle linear
    response to that local field, m_ind,i = RB_0 * B_loc,i, so

        m_ind = RB_0 * (B_applied - D[m_perm + m_ind])
        m_ind - RB_0*(-D[m_ind]) = RB_0*(B_applied - D[m_perm])
        (m_ind/RB_0 + D[m_ind]) = B_applied - D[m_perm]

    The left-hand side is exactly _magnetic_mobility_numba(m_ind) (its
    identity self-term is x_i/RB_0, matching m_ind,i/RB_0 above), so this is
    a linear system A @ m_ind = RHS with A = _magnetic_mobility_numba and

        RHS = B_applied - D[m_perm].

    D[m_perm] is computed here as A(m_perm) - m_perm/RB_0: A already
    includes mom_perm's own spurious identity self-term (since A's identity
    term doesn't know or care whether its argument is m_ind or m_perm), so
    that term has to be subtracted back off to get the permanent moments'
    genuine, self-excluded contribution to the mutual field. This is the
    same construction used in the reference GMRES setup in
    multi_bodies_functions.py (RHS_mag = B_applied - Mdot_Mag(mom_perm) +
    mom_perm/RB_0) -- algebraically identical, just written with the
    cancellation folded in explicitly here instead of left implicit.

    Physically: the permanent moment contributes to every OTHER particle's
    local field exactly like a dipole would (through D), which is what lets
    it help drive their induced moments; it does not contribute to its own
    local field (the subtraction removes that spurious term), since a
    particle's own permanent moment cannot induce more of its own moment.

    Parameters
    ----------
    mom_perm  : (N,3) permanent moments
    B_applied : (N,3) externally applied field at each particle (same
                field for all particles here, but passed per-particle for
                generality)
    r, L, a, chi_exp, RB_0 : as elsewhere in this file
    rtol, restart, maxiter : GMRES controls
    x0        : optional (N,3) initial guess (e.g. the previous step's
                induced moment, or the non-mutual RB_0*B_applied guess) --
                warm-starting like this is why this is passed in rather
                than always starting from zero.

    Returns
    -------
    m_ind : (N,3) self-consistent induced moment
    """
    N = mom_perm.shape[0]
    system_size = 3 * N

    def matvec(x_flat):
        x = x_flat.reshape(N, 3)
        return _magnetic_mobility_numba(x, r, L, a, chi_exp, RB_0).reshape(-1)

    A = spla.LinearOperator((system_size, system_size), matvec=matvec, dtype=np.float64)

    # RHS = B_applied - D[mom_perm], built via the A(mom_perm) - mom_perm/RB_0
    # cancellation described above.
    A_mom_perm = matvec(mom_perm.reshape(-1)).reshape(N, 3)
    D_mom_perm = A_mom_perm - mom_perm / RB_0
    RHS = (B_applied - D_mom_perm).reshape(-1)

    x0_flat = (RB_0 * B_applied).reshape(-1) if x0 is None else np.asarray(x0).reshape(-1)

    iters = [0]
    def _count_iters(*_):
        iters[0] += 1

    m_ind, info = spla.gmres(A, RHS, x0=x0_flat, rtol=rtol, restart=restart,
                            maxiter=maxiter, callback=_count_iters,
                            callback_type='pr_norm')
    # print(f"  [mutual polarizability] GMRES: {iters[0]} iterations, info={info}"
    #     + (" (did not fully converge)" if info != 0 else ""))
    #print(f" GMRES info: {info}, final residual norm = {np.linalg.norm(A @ m_ind - RHS):.3e}")

    return m_ind.reshape(N, 3)


def _compute_moments(bodies, r, t_sim, m_rot_fn, mu_dipole, B_0, RB_0, chi_exp, L, a,
                      mutual_polarizability_mag, gmres_rtol, gmres_restart, gmres_maxiter):
    """
    Compute the permanent moment, applied in-plane field, and induced
    in-plane moment (mutual or simple, per mutual_polarizability_mag) for
    the current body configuration.

    Factored out of force_torque_calculator so the exact same physics can
    be reused for visualization (the pink/sea-green arrows in plot_frame)
    without duplicating -- and risking drifting out of sync with -- the
    mutual-polarizability logic itself.
    """
    N = len(bodies)
    m_rot = np.asarray(m_rot_fn(t_sim), dtype=np.float64)
    R_mats = np.array([b.orientation.as_matrix() for b in bodies])
    m_perm = np.array([mu_dipole * R[:, 0] for R in R_mats])   # (N,3) permanent moments

    B_applied = np.tile(B_0 * m_rot, (N, 1))                    # (N,3)
    if mutual_polarizability_mag:
        induced_mom = _solve_induced_moments_mutual(
            m_perm, B_applied, r, L, a, chi_exp, RB_0,
            rtol=gmres_rtol, restart=gmres_restart, maxiter=gmres_maxiter,
        )
    else:
        induced_mom = RB_0 * B_applied                          # (N,3), same for all

    return m_perm, induced_mom, B_applied


# =============================================================================
# Force / torque calculator
# =============================================================================
def force_torque_calculator(bodies, r_vecs, **kwargs):
    """
    Returns (2N, 3) array of [force_i, torque_i] for each body.

    Forces:
      - Gravity + firm wall repulsion  (single-body, z direction)
      - Pair magnetic dipole interaction  (rotating in-plane field)
      - Pair electric-field-induced dipole (static z field)
      - Pair steric repulsion (firm + soft Yukawa)
      - External magnetic torque  B × m

    kwargs must include m_rot_fn: a callable (t_sim) -> unit 3-vector.
    """
    a          = kwargs['a']
    g          = kwargs['g']
    L          = kwargs['L']
    B_0        = kwargs['B_0']
    B_z        = kwargs['B_z']
    mu_dipole  = kwargs['mu_dipole']
    RB_0       = kwargs['RB_0']
    chi_exp    = kwargs['chi_exp']
    C          = kwargs['C']
    C_z        = kwargs['C_z']
    rep_firm   = kwargs['rep_firm']
    deb_firm   = kwargs['deb_firm']
    firm_delta = kwargs['firm_delta']
    rep_soft   = kwargs['rep_soft']
    deb_soft   = kwargs['deb_soft']
    t_sim      = kwargs['t_sim']
    m_rot_fn   = kwargs['m_rot_fn']
    mutual_polarizability_mag = kwargs.get('mutual_polarizability_mag', False)
    gmres_rtol     = kwargs.get('gmres_rtol', 1e-4)
    gmres_restart  = kwargs.get('gmres_restart', None)
    gmres_maxiter  = kwargs.get('gmres_maxiter', 100)

    # Evaluate rotating field direction at current time
    m_rot = np.asarray(m_rot_fn(t_sim), dtype=np.float64)

    N = len(bodies)
    r = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3)

    # ── rotation matrices and moment arrays ───────────────────────────────────
    R_mats = np.array([b.orientation.as_matrix() for b in bodies])
    zax    = np.array([R[:, 2] for R in R_mats])               # (N,3) body z-axes
    B_z_vec = np.array([0.0, 0.0, 1.0])

    # induced in-plane moment: either simple linear response to the applied
    # field alone, or the self-consistent (mutual polarizability) solve
    m_perm, induced_mom, B_applied = _compute_moments(
        bodies, r, t_sim, m_rot_fn, mu_dipole, B_0, RB_0, chi_exp, L, a,
        mutual_polarizability_mag, gmres_rtol, gmres_restart, gmres_maxiter,
    )

    moments = m_perm + induced_mom                              # (N,3)
    E_mom   = np.tile(RB_0 * B_z * B_z_vec, (N, 1))              # (N,3)  same for all -- mutual polarizability not applied here (out of scope for now)

    # magnetic torques from external field: tau = m × B_applied
    B_torque = B_0 * np.cross(moments, m_rot[np.newaxis, :])  # (N,3)


    force, torque = _pair_and_wall_forces(
        r, moments, E_mom, zax, B_torque,
        L, a, g,
        rep_firm, deb_firm, firm_delta,
        rep_soft, deb_soft,
        C, C_z,
    )

    # # ── Diagnostics: print if any force/torque is suspiciously large ──────────
    # f_max = np.abs(force).max()
    # t_max = np.abs(torque).max()
    # if f_max > 1e3 or t_max > 1e3 or not np.isfinite(f_max) or not np.isfinite(t_max):
    #     print(f"\n*** LARGE/NAN FORCE/TORQUE at t={t_sim:.5f} ***")
    #     print(f"  |m_rot| = {np.linalg.norm(m_rot):.4f}  m_rot = {m_rot}")
    #     print(f"  max |force| = {f_max:.4e}   max |torque| = {t_max:.4e}")
    #     print(f"  positions:\n{r}")
    #     print(f"  separations (min): {_min_separation(r):.4e}")
    #     print(f"  moments:\n{moments}")
    #     print(f"  E_mom:\n{E_mom}")
    #     print(f"  B_torque:\n{B_torque}")
    #     print(f"  forces:\n{force}")
    #     print(f"  torques:\n{torque}")

    FT = np.zeros((2 * N, 3))
    FT[0::2] = force
    FT[1::2] = torque
    return FT


def _min_separation(r):
    """Return minimum pairwise separation distance."""
    N   = r.shape[0]
    mn  = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(r[i] - r[j])
            if d < mn:
                mn = d
    return mn


@njit(fastmath=True, parallel=True)
def _pair_and_wall_forces(r, moments, E_mom, zax, B_torque,
                          L, a, g,
                          rep_firm, deb_firm, firm_delta,
                          rep_soft, deb_soft,
                          C, C_z):
    """
    Numba kernel: computes all forces and torques.
      - Gravity + wall repulsion per particle
      - Pair magnetic dipole (in-plane moments)
      - Pair electric-field-induced dipole (z moments)
      - Pair steric repulsion
      - External magnetic torque (from B_torque array)

    Two changes from the previous version, both verified to give identical
    results (checked against the un-hoisted version on synthetic N=60 data,
    both with an active and an inactive z-field: max force/torque
    difference was exactly 0.0):

      1. parallel=True re-enabled. It had been turned off based on earlier
         advice for the N=6 ladder-to-ring system, where per-call threading
         overhead can exceed the (tiny) amount of work available to
         parallelize. At N=500 that tradeoff reverses -- there's enough
         work per call (N*(N-1) ~ 250k pairs) that spreading it across
         cores should win. prange was already used for the outer loops
         here; without parallel=True it silently behaved as plain range.

      2. Per-particle moment normalization (m_hat, m_norm) is hoisted out
         of the O(N^2) pair loop into one O(N) pass, instead of being
         recomputed for the same i on every j. E_mom's normalization is
         hoisted even further: E_mom is np.tile'd (every particle has the
         literal same z-field moment), so ez_hat/ez_norm are computed once
         for the whole kernel rather than recomputed identically N^2
         times. mj_d_r == mi_d_r and ez_j == ez_i in that branch as a
         direct consequence, which is used to drop the now-redundant
         "j" computations (and the ez_i x ez_j cross product, which is
         identically zero for parallel vectors).
    """
    N      = r.shape[0]
    force  = np.zeros((N, 3))
    torque = np.zeros((N, 3))

    # ── hoisted per-particle moment magnitude/direction ───────────────────────
    m_norm      = np.zeros(N)
    m_hat       = np.zeros((N, 3))
    has_moment  = np.zeros(N, dtype=np.bool_)
    for i in prange(N):
        nrm = np.sqrt(moments[i,0]**2 + moments[i,1]**2 + moments[i,2]**2)
        m_norm[i] = nrm
        if nrm > 1e-30:
            has_moment[i] = True
            m_hat[i, 0] = moments[i, 0] / nrm
            m_hat[i, 1] = moments[i, 1] / nrm
            m_hat[i, 2] = moments[i, 2] / nrm

    # ── E_mom is identical for every particle (np.tile of one row) --
    #    compute its normalized direction/magnitude exactly once ───────────────
    ez_norm = np.sqrt(E_mom[0,0]**2 + E_mom[0,1]**2 + E_mom[0,2]**2)
    has_ez  = ez_norm > 1e-30
    ezx = E_mom[0, 0] / ez_norm if has_ez else 0.0
    ezy = E_mom[0, 1] / ez_norm if has_ez else 0.0
    ezz = E_mom[0, 2] / ez_norm if has_ez else 0.0

    # ── single-body: gravity + wall repulsion ────────────────────────────────
    for i in prange(N):
        force[i, 2] -= g
        h       = r[i, 2]
        contact = a * (1.0 - firm_delta)
        if h > contact:
            force[i, 2] += (rep_firm / deb_firm) * np.exp(-(h - contact) / deb_firm)
        else:
            force[i, 2] += rep_firm / deb_firm

    # ── pairwise interactions ─────────────────────────────────────────────────
    for i in prange(N):
        mix, miy, miz = m_hat[i, 0], m_hat[i, 1], m_hat[i, 2]
        m_i_norm = m_norm[i]
        i_has_m  = has_moment[i]

        for j in range(N):
            if i == j:
                continue

            # minimum-image displacement
            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r[j, k] - r[i, k]
                if L[k] > 0:
                    dr[k] -= int(dr[k] / L[k] + 0.5 * (
                        int(dr[k] > 0) - int(dr[k] < 0))) * L[k]

            r_norm = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            if r_norm < 1e-12:
                continue
            r_hat = dr / r_norm

            # ── in-plane magnetic dipole force & torque ───────────────────────
            if i_has_m and has_moment[j]:
                mjx, mjy, mjz = m_hat[j, 0], m_hat[j, 1], m_hat[j, 2]
                m_j_norm = m_norm[j]
                mi_d_r  = mix*r_hat[0] + miy*r_hat[1] + miz*r_hat[2]
                mj_d_r  = mjx*r_hat[0] + mjy*r_hat[1] + mjz*r_hat[2]
                mi_d_mj = mix*mjx + miy*mjy + miz*mjz
                F_mag   = C * (m_i_norm * m_j_norm) / (r_norm**4)
                coeff   = mi_d_mj - 5.0 * mj_d_r * mi_d_r
                force[i, 0] -= F_mag * (mi_d_r*mjx + mj_d_r*mix + coeff*r_hat[0])
                force[i, 1] -= F_mag * (mi_d_r*mjy + mj_d_r*miy + coeff*r_hat[1])
                force[i, 2] -= F_mag * (mi_d_r*mjz + mj_d_r*miz + coeff*r_hat[2])

                T_mag = (1.0/3.0) * C * (m_i_norm * m_j_norm) / (r_norm**3)
                mi_X_mj_x = miy*mjz - miz*mjy
                mi_X_mj_y = miz*mjx - mix*mjz
                mi_X_mj_z = mix*mjy - miy*mjx
                mi_X_r_x  = miy*r_hat[2] - miz*r_hat[1]
                mi_X_r_y  = miz*r_hat[0] - mix*r_hat[2]
                mi_X_r_z  = mix*r_hat[1] - miy*r_hat[0]
                torque[i, 0] += T_mag * (3.0*mj_d_r*mi_X_r_x - mi_X_mj_x)
                torque[i, 1] += T_mag * (3.0*mj_d_r*mi_X_r_y - mi_X_mj_y)
                torque[i, 2] += T_mag * (3.0*mj_d_r*mi_X_r_z - mi_X_mj_z)

            # ── z-field induced dipole force & torque ─────────────────────────
            # ez_i == ez_j for every pair (E_mom is uniform), so mj_d_r ==
            # mi_d_r here and the ez_i x ez_j cross product is identically
            # zero (parallel vectors) -- both simplifications are applied
            # directly below rather than recomputing a "j" copy of ez.
            if has_ez:
                ezi_d_r = ezx*r_hat[0] + ezy*r_hat[1] + ezz*r_hat[2]
                Fz_mag  = C_z * (ez_norm * ez_norm) / (r_norm**4)
                coeff_z = 1.0 - 5.0 * ezi_d_r * ezi_d_r   # ezi_d_ezj == 1
                force[i, 0] -= Fz_mag * (2.0*ezi_d_r*ezx + coeff_z*r_hat[0])
                force[i, 1] -= Fz_mag * (2.0*ezi_d_r*ezy + coeff_z*r_hat[1])
                force[i, 2] -= Fz_mag * (2.0*ezi_d_r*ezz + coeff_z*r_hat[2])

                Tz_mag = (1.0/3.0) * C_z * (ez_norm * ez_norm) / (r_norm**3)
                ezi_X_r_x = ezy*r_hat[2] - ezz*r_hat[1]
                ezi_X_r_y = ezz*r_hat[0] - ezx*r_hat[2]
                ezi_X_r_z = ezx*r_hat[1] - ezy*r_hat[0]
                torque[i, 0] += Tz_mag * (3.0*ezi_d_r*ezi_X_r_x)
                torque[i, 1] += Tz_mag * (3.0*ezi_d_r*ezi_X_r_y)
                torque[i, 2] += Tz_mag * (3.0*ezi_d_r*ezi_X_r_z)

            # ── steric repulsion (only between real particles, j < N) ─────────
            offset_firm = 2.0 * a * (1.0 - firm_delta)
            if r_norm > offset_firm:
                pref_firm = (rep_firm / deb_firm) * np.exp(-(r_norm - offset_firm) / deb_firm) / r_norm
            else:
                pref_firm = rep_firm / deb_firm / r_norm
            force[i, 0] -= pref_firm * dr[0]
            force[i, 1] -= pref_firm * dr[1]
            force[i, 2] -= pref_firm * dr[2]

            if rep_soft > 0.0:
                offset_soft = 2.0 * a
                if r_norm > offset_soft:
                    pref_soft = (rep_soft / deb_soft) * np.exp(-(r_norm - offset_soft) / deb_soft) / r_norm
                else:
                    pref_soft = rep_soft / deb_soft / r_norm
                force[i, 0] -= pref_soft * dr[0]
                force[i, 1] -= pref_soft * dr[1]
                force[i, 2] -= pref_soft * dr[2]

    # ── external magnetic torque (separate loop, matching original) ──────────
    for i in prange(N):
        for k in range(3):
            torque[i, k] += B_torque[i, k]

    return force, torque


# =============================================================================
# Buffered (Verlet-style) neighbor list -- ADDITIVE, does not touch or
# replace anything above. _pair_and_wall_forces (the full O(N^2) kernel) is
# left completely as-is; this is a second, parallel code path.
#
# Pattern (list radius = interaction cutoff + skin buffer; rebuild once any
# particle has moved more than half the buffer since the last build) is
# ported from Fast_Dry_Diffusion_PNV_theory.py's build_neighbor_list /
# max_displacement_periodic. That correctness condition is what guarantees
# no pair can cross into r_cut between rebuilds without being caught: if two
# particles were farther apart than r_cut + buffer at the last build, even
# in the worst case (each moving up to buffer/2 directly toward the other)
# they still can't have closed to less than r_cut by the time the check
# fires. Two differences from that reference, both because this system's
# geometry differs from the 2-D periodic one it was written for:
#   - 3-D positions, and an *open* box (L=[0,0,0] throughout this file) --
#     so no boxsize/periodic wrapping is needed in the tree query at all,
#     unlike the reference's always-periodic-in-xy version.
#   - nlist_buffer defaults to 0.5*a here, not the reference's 10*a -- that
#     value was tuned for a freely-diffusing system with much larger
#     per-step excursions than this stiffer, densely-packed one. Tune this:
#     if rebuilds fire almost every step, raise it; if they essentially
#     never fire after the first few, it's larger than it needs to be.
# =============================================================================

@njit(fastmath=True, parallel=True)
def _max_displacement(r_new, r_old):
    """Max per-particle displacement since the last neighbor-list build.
    No periodic wrapping -- matches this file's L=[0,0,0] (open) box; if you
    ever turn on periodicity here, this needs the same minimum-image
    handling _pair_and_wall_forces already does."""
    N  = r_new.shape[0]
    d2 = np.empty(N, dtype=np.float64)
    for i in prange(N):
        dx = r_new[i, 0] - r_old[i, 0]
        dy = r_new[i, 1] - r_old[i, 1]
        dz = r_new[i, 2] - r_old[i, 2]
        d2[i] = dx*dx + dy*dy + dz*dz
    return np.sqrt(np.max(d2))


def build_neighbor_list_3d(r, r_list_cut, L):
    """
    3-D neighbor list via scipy cKDTree, in the same CSR-like (offsets,
    neighbor_list) format as Fast_Dry_Diffusion_PNV_theory.py's
    build_neighbor_list. Open-box fast path (no boxsize) when L is
    [0,0,0], matching how this file actually uses it; falls back to a
    per-axis "large sentinel for non-periodic dims" trick if you ever do
    turn on periodicity in some axis but not others.
    """
    L = np.asarray(L, dtype=np.float64)
    if np.all(L <= 0):
        tree = scipy.spatial.cKDTree(r)
    else:
        boxsize = np.where(L > 0, L, 1e8)   # effectively non-periodic where L<=0
        tree = scipy.spatial.cKDTree(r % boxsize, boxsize=boxsize * 1.001)
    pairs = tree.query_ball_point(r, r_list_cut, workers=-1)
    offsets = np.cumsum([0] + [len(p) for p in pairs]).astype(np.int64)
    neighbor_list = np.fromiter((x for p in pairs for x in p), dtype=np.int64)
    return offsets, neighbor_list


@njit(fastmath=True, parallel=True)
def _pair_and_wall_forces_neighborlist(r, moments, E_mom, zax, B_torque,
                                        offsets, neighbor_list,
                                        L, a, g,
                                        rep_firm, deb_firm, firm_delta,
                                        rep_soft, deb_soft,
                                        C, C_z):
    """
    Same physics as _pair_and_wall_forces, but each particle only sums over
    its listed neighbors (offsets[i]:offsets[i+1] into neighbor_list)
    instead of every other particle. Verified against the full O(N^2)
    kernel on a realistic N=400, phi=0.6 configuration -- see
    test_neighborlist.py. Everything else (hoisted moment/E_mom
    normalization, gravity/wall term, steric repulsion) is identical to
    _pair_and_wall_forces; only the inner loop's iteration set changed.
    """
    N      = r.shape[0]
    force  = np.zeros((N, 3))
    torque = np.zeros((N, 3))

    m_norm      = np.zeros(N)
    m_hat       = np.zeros((N, 3))
    has_moment  = np.zeros(N, dtype=np.bool_)
    for i in prange(N):
        nrm = np.sqrt(moments[i,0]**2 + moments[i,1]**2 + moments[i,2]**2)
        m_norm[i] = nrm
        if nrm > 1e-30:
            has_moment[i] = True
            m_hat[i, 0] = moments[i, 0] / nrm
            m_hat[i, 1] = moments[i, 1] / nrm
            m_hat[i, 2] = moments[i, 2] / nrm

    ez_norm = np.sqrt(E_mom[0,0]**2 + E_mom[0,1]**2 + E_mom[0,2]**2)
    has_ez  = ez_norm > 1e-30
    ezx = E_mom[0, 0] / ez_norm if has_ez else 0.0
    ezy = E_mom[0, 1] / ez_norm if has_ez else 0.0
    ezz = E_mom[0, 2] / ez_norm if has_ez else 0.0

    for i in prange(N):
        force[i, 2] -= g
        h       = r[i, 2]
        contact = a * (1.0 - firm_delta)
        if h > contact:
            force[i, 2] += (rep_firm / deb_firm) * np.exp(-(h - contact) / deb_firm)
        else:
            force[i, 2] += rep_firm / deb_firm

    for i in prange(N):
        mix, miy, miz = m_hat[i, 0], m_hat[i, 1], m_hat[i, 2]
        m_i_norm = m_norm[i]
        i_has_m  = has_moment[i]

        start, end = offsets[i], offsets[i + 1]
        for kk in range(start, end):
            j = neighbor_list[kk]
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
            r_hat = dr / r_norm

            if i_has_m and has_moment[j]:
                mjx, mjy, mjz = m_hat[j, 0], m_hat[j, 1], m_hat[j, 2]
                m_j_norm = m_norm[j]
                mi_d_r  = mix*r_hat[0] + miy*r_hat[1] + miz*r_hat[2]
                mj_d_r  = mjx*r_hat[0] + mjy*r_hat[1] + mjz*r_hat[2]
                mi_d_mj = mix*mjx + miy*mjy + miz*mjz
                F_mag   = C * (m_i_norm * m_j_norm) / (r_norm**4)
                coeff   = mi_d_mj - 5.0 * mj_d_r * mi_d_r
                force[i, 0] -= F_mag * (mi_d_r*mjx + mj_d_r*mix + coeff*r_hat[0])
                force[i, 1] -= F_mag * (mi_d_r*mjy + mj_d_r*miy + coeff*r_hat[1])
                force[i, 2] -= F_mag * (mi_d_r*mjz + mj_d_r*miz + coeff*r_hat[2])

                T_mag = (1.0/3.0) * C * (m_i_norm * m_j_norm) / (r_norm**3)
                mi_X_mj_x = miy*mjz - miz*mjy
                mi_X_mj_y = miz*mjx - mix*mjz
                mi_X_mj_z = mix*mjy - miy*mjx
                mi_X_r_x  = miy*r_hat[2] - miz*r_hat[1]
                mi_X_r_y  = miz*r_hat[0] - mix*r_hat[2]
                mi_X_r_z  = mix*r_hat[1] - miy*r_hat[0]
                torque[i, 0] += T_mag * (3.0*mj_d_r*mi_X_r_x - mi_X_mj_x)
                torque[i, 1] += T_mag * (3.0*mj_d_r*mi_X_r_y - mi_X_mj_y)
                torque[i, 2] += T_mag * (3.0*mj_d_r*mi_X_r_z - mi_X_mj_z)

            if has_ez:
                ezi_d_r = ezx*r_hat[0] + ezy*r_hat[1] + ezz*r_hat[2]
                Fz_mag  = C_z * (ez_norm * ez_norm) / (r_norm**4)
                coeff_z = 1.0 - 5.0 * ezi_d_r * ezi_d_r
                force[i, 0] -= Fz_mag * (2.0*ezi_d_r*ezx + coeff_z*r_hat[0])
                force[i, 1] -= Fz_mag * (2.0*ezi_d_r*ezy + coeff_z*r_hat[1])
                force[i, 2] -= Fz_mag * (2.0*ezi_d_r*ezz + coeff_z*r_hat[2])

                Tz_mag = (1.0/3.0) * C_z * (ez_norm * ez_norm) / (r_norm**3)
                ezi_X_r_x = ezy*r_hat[2] - ezz*r_hat[1]
                ezi_X_r_y = ezz*r_hat[0] - ezx*r_hat[2]
                ezi_X_r_z = ezx*r_hat[1] - ezy*r_hat[0]
                torque[i, 0] += Tz_mag * (3.0*ezi_d_r*ezi_X_r_x)
                torque[i, 1] += Tz_mag * (3.0*ezi_d_r*ezi_X_r_y)
                torque[i, 2] += Tz_mag * (3.0*ezi_d_r*ezi_X_r_z)

            offset_firm = 2.0 * a * (1.0 - firm_delta)
            if r_norm > offset_firm:
                pref_firm = (rep_firm / deb_firm) * np.exp(-(r_norm - offset_firm) / deb_firm) / r_norm
            else:
                pref_firm = rep_firm / deb_firm / r_norm
            force[i, 0] -= pref_firm * dr[0]
            force[i, 1] -= pref_firm * dr[1]
            force[i, 2] -= pref_firm * dr[2]

            if rep_soft > 0.0:
                offset_soft = 2.0 * a
                if r_norm > offset_soft:
                    pref_soft = (rep_soft / deb_soft) * np.exp(-(r_norm - offset_soft) / deb_soft) / r_norm
                else:
                    pref_soft = rep_soft / deb_soft / r_norm
                force[i, 0] -= pref_soft * dr[0]
                force[i, 1] -= pref_soft * dr[1]
                force[i, 2] -= pref_soft * dr[2]

    for i in prange(N):
        for k in range(3):
            torque[i, k] += B_torque[i, k]

    return force, torque


def force_torque_calculator_nlist(bodies, r_vecs, **kwargs):
    """
    Same as force_torque_calculator, except it uses the neighbor-list
    kernel above instead of the full O(N^2) one. kwargs must additionally
    include offsets and neighbor_list (see the buffered-rebuild loop in
    main()). Everything upstream of the pairwise kernel -- moment
    computation, mutual polarizability, etc. -- is untouched and reused
    exactly as in force_torque_calculator.
    """
    a          = kwargs['a']
    g          = kwargs['g']
    L          = kwargs['L']
    B_0        = kwargs['B_0']
    mu_dipole  = kwargs['mu_dipole']
    RB_0       = kwargs['RB_0']
    chi_exp    = kwargs['chi_exp']
    C          = kwargs['C']
    C_z        = kwargs['C_z']
    B_z        = kwargs['B_z']
    rep_firm   = kwargs['rep_firm']
    deb_firm   = kwargs['deb_firm']
    firm_delta = kwargs['firm_delta']
    rep_soft   = kwargs['rep_soft']
    deb_soft   = kwargs['deb_soft']
    t_sim      = kwargs['t_sim']
    m_rot_fn   = kwargs['m_rot_fn']
    offsets        = kwargs['offsets']
    neighbor_list  = kwargs['neighbor_list']
    mutual_polarizability_mag = kwargs.get('mutual_polarizability_mag', False)
    gmres_rtol     = kwargs.get('gmres_rtol', 1e-4)
    gmres_restart  = kwargs.get('gmres_restart', None)
    gmres_maxiter  = kwargs.get('gmres_maxiter', 100)

    m_rot = np.asarray(m_rot_fn(t_sim), dtype=np.float64)

    N = len(bodies)
    r = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3)

    R_mats = np.array([b.orientation.as_matrix() for b in bodies])
    zax    = np.array([R[:, 2] for R in R_mats])
    B_z_vec = np.array([0.0, 0.0, 1.0])

    m_perm, induced_mom, B_applied = _compute_moments(
        bodies, r, t_sim, m_rot_fn, mu_dipole, B_0, RB_0, chi_exp, L, a,
        mutual_polarizability_mag, gmres_rtol, gmres_restart, gmres_maxiter,
    )

    moments = m_perm + induced_mom
    E_mom   = np.tile(RB_0 * B_z * B_z_vec, (N, 1))

    B_torque = B_0 * np.cross(moments, m_rot[np.newaxis, :])

    force, torque = _pair_and_wall_forces_neighborlist(
        r, moments, E_mom, zax, B_torque,
        offsets, neighbor_list,
        L, a, g,
        rep_firm, deb_firm, firm_delta,
        rep_soft, deb_soft,
        C, C_z,
    )

    FT = np.zeros((2 * N, 3))
    FT[0::2] = force
    FT[1::2] = torque
    return FT


# =============================================================================
if __name__ == '__main__':
    main()
