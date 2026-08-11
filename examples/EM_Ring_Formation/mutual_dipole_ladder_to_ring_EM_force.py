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
    B_0        = 0.65 #0.82 #0.92   # in-plane rotating field amplitude (mT)
    B_freq     = 20.0   # rotation frequency (Hz)
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
    mutual_polarizability_mag = True
    gmres_rtol     = 1e-6
    gmres_restart  = 100
    gmres_maxiter  = 100

    # ── Time-dependent B_z protocol ───────────────────────────────────────────
    # List of (time_s, B_z_mT) steps; B_z takes the value from the last
    # entry whose time <= current simulation time.
    # B_z_schedule = [
    #     (0.0,  0.635),
    #     (4.0,  0.735),
    #     (12.0, 0.740),
    #     (20.0, 0.745),
    #     (28.0, 0.755),
    #     (36.0, 0.760),
    #     (44.0, 0.765),
    #     (52.0, 0.100),
    # ]
    # t_end = 60.0        # stop time (s)

    # B_z_schedule = [ # for B_0 = 0.82
    #     (0.0,  0.535),
    #     (7.0,  0.635),
    #     (19.5, 0.645),
    #     (30.0, 0.655),
    #     (38.0, 0.665),
    #     (50.75, 0.670),
    #     (70.75, 0.100),
    # ]
    # t_end = 100.0        # stop time (s)

    B_z_schedule = [
        (0.0,  0.435),
        (15.12,  0.475),
        (22.53, 0.485),
        # (30.0, 0.495),
        # (38.0, 0.511),
        # (50.75, 0.521),
        # (58.75, 0.53),
        # (64.75, 0.54),
        (60.75, 0.100),
    ]
    t_end = 100.0        # stop time (s)

    # ── Interaction parameters ────────────────────────────────────────────────
    firm_delta          = 1e-2 #1e-3
    debye_firm          = 2.0 * a * firm_delta / np.log(10.0)
    repulsion_firm      = 0.0331
    repulsion_soft      = 0.0    # soft Yukawa off
    debye_soft          = 0.225

    # ── Box geometry (open — no periodicity) ─────────────────────────────────
    L            = np.array([0.0, 0.0, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    # ── Output ────────────────────────────────────────────────────────────────
    if Stoch:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_MDI_ladder_to_ring_frames_stochastic')
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_MDI_ladder_to_ring_frames')
    os.makedirs(out_dir, exist_ok=True)

    # ── Initial positions and orientations (from suspension_ladder_N_6_random.clones)
    # Format: x y z qw qx qy qz  (scalar-first quaternion)
    clones = [
        [-2.2725,  0.0,           2.295,  0.169897885187,  0.0, 0.0,  0.985461672826 ],
        [ 0.0,     3.9360854602,  2.295,  0.882290504023,  0.0, 0.0, -0.470705286258 ],
        [ 2.2725,  0.0,           2.295,  0.815967256919,  0.0, 0.0, -0.578098119384 ],
        [ 0.0,    -3.9360854602,  2.295,  0.577953169684,  0.0, 0.0,  0.81606993184  ],
        [ 4.545,  -3.9360854602,  2.295,  0.0524575746726, 0.0, 0.0,  0.998623153577 ],
        [-4.545,   3.9360854602,  2.295,  0.487765686157,  0.0, 0.0,  0.872974590356 ],
    ]
    bodies = []
    for row in clones:
        loc = np.array(row[:3])
        qw, qx, qy, qz = row[3], row[4], row[5], row[6]
        orientation = Rotation.from_quat([qx, qy, qz, qw])   # scipy: (x,y,z,w)
        bodies.append(Body(location=loc, orientation=orientation))

    N = len(bodies)

    # ── Rotating field direction (user-defined function of time) ─────────────
    # m_rot_fn(t_sim) must return a 3-vector giving the field direction.
    # Default matches the original code: Lissajous pattern with
    # x-frequency = B_freq, y-frequency = 2*B_freq  (Omega_y = 2*Omega).
    B_freq_y = 2.0 * B_freq
    def m_rot_fn(t):
        return np.array([np.cos(2 * np.pi * B_freq   * t),
                         np.sin(2 * np.pi * B_freq_y * t),
                         0.0])
    
    # ── Simulation parameters ─────────────────────────────────────────────────
    dt      = 0.05*(1/B_freq_y) #6.25e-5   # timestep (s)
    n_steps = int(t_end / dt)
    n_plot  = 10*16+1       # plot frame every n_plot steps
    solver_tolerance = 5e-3

    print(f"N = 6 particles, dt = {dt}, n_steps = {n_steps}, t_end = {t_end} s")

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

        # determine current B_z from schedule
        B_z = B_z_schedule[0][1]
        for t_thresh, bz_val in B_z_schedule:
            if t_sim >= t_thresh:
                B_z = bz_val

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
                             linewidth=3.5, zorder=2)
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
                                   lw=4.5), zorder=3)

        # pink: external (applied) field direction, fixed reference length
        if field_dir_xy is not None:
            fdx, fdy = field_dir_xy[0], field_dir_xy[1]
            ax.annotate('', xy=(x + scale * fdx, y + scale * fdy),
                        xytext=(x - scale * fdx, y - scale * fdy),
                        arrowprops=dict(arrowstyle='->', color=pink,
                                       lw=4.5), zorder=3)

        # sea green: induced moment, length relative to RB_0*|B_applied|
        if induced_mom is not None and M_ref is not None and M_ref > 1e-12:
            v = np.asarray(induced_mom[idx], dtype=np.float64)
            vx = v[0] * (scale / M_ref)
            vy = v[1] * (scale / M_ref)
            ax.annotate('', xy=(x + vx, y + vy),
                        xytext=(x - vx, y - vy),
                        arrowprops=dict(arrowstyle='->', color=sea_green,
                                       lw=4.5), zorder=3)

    # fit axes tightly to particle extents
    locs  = np.array([b.location for b in bodies])
    pad   = 1.5 * a
    ax.set_xlim(locs[:, 0].min() - pad, locs[:, 0].max() + pad)
    ax.set_ylim(locs[:, 1].min() - pad, locs[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x (µm)', fontsize=12)
    ax.set_ylabel('y (µm)', fontsize=12)
    ax.set_title(f't = {t_sim:.4f} s   B_z = {B_z:.3f} mT', fontsize=12)
    ax.grid(True, alpha=0.3)

    fname = os.path.join(out_dir, f'frame_{step:07d}.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


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


@njit(fastmath=True) #parallel=True)
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

    # ── pairwise interactions ─────────────────────────────────────────────────
    for i in prange(N):
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
            m_i_norm = np.sqrt(moments[i,0]**2 + moments[i,1]**2 + moments[i,2]**2)
            m_j_norm = np.sqrt(moments[j,0]**2 + moments[j,1]**2 + moments[j,2]**2)
            if m_i_norm > 1e-30 and m_j_norm > 1e-30:
                m_i = moments[i, :] / m_i_norm
                m_j = moments[j, :] / m_j_norm
                mi_d_r = m_i[0]*r_hat[0] + m_i[1]*r_hat[1] + m_i[2]*r_hat[2]
                mj_d_r = m_j[0]*r_hat[0] + m_j[1]*r_hat[1] + m_j[2]*r_hat[2]
                mi_d_mj = m_i[0]*m_j[0] + m_i[1]*m_j[1] + m_i[2]*m_j[2]
                F_mag = C * (m_i_norm * m_j_norm) / (r_norm**4)
                for k in range(3):
                    force[i, k] -= F_mag * (
                        mi_d_r * m_j[k] + mj_d_r * m_i[k]
                        - (5.0 * mj_d_r * mi_d_r - mi_d_mj) * r_hat[k])
                T_mag = (1.0/3.0) * C * (m_i_norm * m_j_norm) / (r_norm**3)
                mi_X_mj = np.cross(m_i, m_j)
                mi_X_r  = np.cross(m_i, r_hat)
                for k in range(3):
                    torque[i, k] += T_mag * (3.0 * mj_d_r * mi_X_r[k] - mi_X_mj[k])

            # ── z-field induced dipole force & torque ─────────────────────────
            ez_i_norm = np.sqrt(E_mom[i,0]**2 + E_mom[i,1]**2 + E_mom[i,2]**2)
            ez_j_norm = np.sqrt(E_mom[j,0]**2 + E_mom[j,1]**2 + E_mom[j,2]**2)
            if ez_i_norm > 1e-30 and ez_j_norm > 1e-30:
                ez_i = E_mom[i, :] / ez_i_norm
                ez_j = E_mom[j, :] / ez_j_norm
                ezi_d_r  = ez_i[0]*r_hat[0] + ez_i[1]*r_hat[1] + ez_i[2]*r_hat[2]
                ezj_d_r  = ez_j[0]*r_hat[0] + ez_j[1]*r_hat[1] + ez_j[2]*r_hat[2]
                ezi_d_ezj = ez_i[0]*ez_j[0] + ez_i[1]*ez_j[1] + ez_i[2]*ez_j[2]
                Fz_mag = C_z * (ez_i_norm * ez_j_norm) / (r_norm**4)
                for k in range(3):
                    force[i, k] -= Fz_mag * (
                        ezi_d_r * ez_j[k] + ezj_d_r * ez_i[k]
                        - (5.0 * ezj_d_r * ezi_d_r - ezi_d_ezj) * r_hat[k])
                Tz_mag = (1.0/3.0) * C_z * (ez_i_norm * ez_j_norm) / (r_norm**3)
                ezi_X_ezj = np.cross(ez_i, ez_j)
                ezi_X_r   = np.cross(ez_i, r_hat)
                for k in range(3):
                    torque[i, k] += Tz_mag * (3.0 * ezj_d_r * ezi_X_r[k] - ezi_X_ezj[k])

            # ── steric repulsion (only between real particles, j < N) ─────────
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

    # ── external magnetic torque (separate loop, matching original) ──────────
    for i in prange(N):
        for k in range(3):
            torque[i, k] += B_torque[i, k]

    return force, torque


# =============================================================================
if __name__ == '__main__':
    main()