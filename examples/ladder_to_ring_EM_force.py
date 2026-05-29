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
  - External magnetic torque (B × m)

Each n_plot steps a snapshot PNG is saved showing particles as 2-D discs
with an orientation arrow (projection of the body x-axis into the xy plane).

Time-dependent protocol:  B_z is stepped up at specified simulation times
to drive the chain-to-ring transition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from functools import partial
from numba import njit, prange
from scipy.spatial.transform import Rotation

from body import Body
from pyStokesianDynamics import pyStokesianDynamics


# =============================================================================
def main():
# =============================================================================

    # ── Physical parameters ───────────────────────────────────────────────────
    a   = 2.25          # particle radius (µm)
    eta = 8.9e-4        # fluid viscosity
    kT  = 0.0041419464  # thermal energy
    g   = 0.28041       # gravitational acceleration (z)
    Stoch = True        # include stochastic forces/torques (Brownian motion)

    # ── Magnetic / electric field parameters ─────────────────────────────────
    mu_dipole  = 8.6    # permanent dipole (aJ/mT)
    B_0        = 0.92   # in-plane rotating field amplitude (mT)
    B_freq     = 80.0   # rotation frequency (Hz)
    RB_0       = 48.22  # susceptibility prefactor (has susceptibility hard-coded in)
    # Below is how RB_0 was caclulated.
    # https://www.wolframalpha.com/input?i=convert+4*pi*%282.25+um%29%5E3+*+1.27+*%281+Militesla%29%2F%283*%284*pi*1e-7%29+Henry%2Fm%29+in+attoJoules%2FMilitessla
    chi_exp    = 1.27   # magnetic susceptibility
    C          = 0.3    # dipole-dipole coupling constant (in-plane)
    # Below is how C was caclulated
    # https://www.wolframalpha.com/input?i=%283%2F%284*pi%29%29*+%281+attoJoule+%2Fmillitesla+%29%5E2+*+%284*pi*1e-7+Henry%2Fm%29+%2F+%281+um%29%5E4+to+pN
    C_z        = 0.3    # dipole-dipole coupling constant (z field)

    # ── Time-dependent B_z protocol ───────────────────────────────────────────
    # List of (time_s, B_z_mT) steps; B_z takes the value from the last
    # entry whose time <= current simulation time.
    B_z_schedule = [
        (0.0,  0.535),
        (7.0,  0.635),
        (22.0, 0.645),
        (30.0, 0.655),
        (38.0, 0.665),
        (46.0, 0.670),
    ]
    t_end = 54.0        # stop time (s)

    # ── Interaction parameters ────────────────────────────────────────────────
    firm_delta          = 1e-3
    debye_firm          = 2.0 * a * firm_delta / np.log(10.0)
    repulsion_firm      = 0.0331
    repulsion_soft      = 0.0    # soft Yukawa off
    debye_soft          = 0.225

    # ── Box geometry (open — no periodicity) ─────────────────────────────────
    L            = np.array([0.0, 0.0, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    # ── Simulation parameters ─────────────────────────────────────────────────
    dt      = 6.25e-5   # timestep (s)
    n_steps = int(t_end / dt)
    n_plot  = 10*160       # plot frame every n_plot steps
    solver_tolerance = 5e-3

    # ── Output ────────────────────────────────────────────────────────────────
    if Stoch:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ladder_to_ring_frames_stochastic')
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ladder_to_ring_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f"N = 6 particles, dt = {dt}, n_steps = {n_steps}, t_end = {t_end} s")

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
    B_freq_y = 2 * B_freq
    def m_rot_fn(t):
        return np.array([np.cos(2 * np.pi * B_freq   * t),
                         np.sin(2 * np.pi * B_freq_y * t),
                         0.0])

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
        )

        solver.Update_Bodies_Trap(FT_calc, stochastic=Stoch, print_residual=False)

        # ── Per-step diagnostics ──────────────────────────────────────────────
        r_now = np.array([b.location for b in bodies])
        min_sep = _min_separation(r_now)
        min_z   = r_now[:, 2].min()
        if not np.all(np.isfinite(r_now)) or min_sep < 0.5*a or min_z < 0.1*a:
            print(f"\n*** GEOMETRY WARNING at step {step}, t={t_sim:.5f} ***")
            print(f"  min pair separation = {min_sep:.4e}  (2a = {2*a:.4f})")
            print(f"  min z               = {min_z:.4e}  (a = {a:.4f})")
            print(f"  positions:\n{r_now}")

        if step % n_plot == 0:
            elapsed = time.perf_counter() - t_wall
            print(f"  step {step:7d}/{n_steps}  t={t_sim:.4f} s  "
                  f"B_z={B_z:.3f} mT  elapsed={elapsed:.1f}s")
            plot_frame(bodies, a, frame_idx, t_sim, B_z, out_dir)
            frame_idx += 1

    print(f"\nSimulation complete in {time.perf_counter()-t_wall:.1f}s")


# =============================================================================
# Visualisation
# =============================================================================
def plot_frame(bodies, a, step, t_sim, B_z, out_dir):
    """
    Plot particles as 2-D discs with orientation arrows projected into xy plane.
    Cornflower-blue fill, blue edge (lw=3.5).  Pink arrow (lw=3.5) shows the
    projection of the body x-axis (first column of rotation matrix) into xy.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for b in bodies:
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
        scale  = a * 0.75                 # arrow length relative to radius

        gold = np.array([255,165,0]) /255.0
        ax.annotate('', xy=(x + scale * dx, y + scale * dy),
                    xytext=(x - scale * dx, y - scale * dy),
                    arrowprops=dict(arrowstyle='->', color=gold,
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

    # Evaluate rotating field direction at current time
    m_rot = np.asarray(m_rot_fn(t_sim), dtype=np.float64)

    N = len(bodies)
    r = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3)

    # ── rotation matrices and moment arrays ───────────────────────────────────
    R_mats = np.array([b.orientation.as_matrix() for b in bodies])
    m_perm = np.array([mu_dipole * R[:, 0] for R in R_mats])   # (N,3) permanent moments
    zax    = np.array([R[:, 2] for R in R_mats])               # (N,3) body z-axes
    B_z_vec = np.array([0.0, 0.0, 1.0])

    # induced moments (linear response, ignoring self-field correction)
    induced_mom = RB_0 * B_0 * m_rot[np.newaxis, :]              # (N,3)  broadcast
    moments     = m_perm + np.tile(induced_mom, (N, 1))           # (N,3)
    E_mom       = np.tile(RB_0 * B_z * B_z_vec, (N, 1))          # (N,3)  same for all

    # magnetic torques from external field: tau = m × B_applied
    B_torque = B_0 * np.cross(moments, m_rot[np.newaxis, :])  # (N,3)

    force, torque = _pair_and_wall_forces(
        r, moments, E_mom, zax, B_torque,
        L, a, g,
        rep_firm, deb_firm, firm_delta,
        rep_soft, deb_soft,
        C, C_z,
    )

    force, torque = _pair_and_wall_forces(
        r, moments, E_mom, zax, B_torque,
        L, a, g,
        rep_firm, deb_firm, firm_delta,
        rep_soft, deb_soft,
        C, C_z,
    )

    # ── Diagnostics: print if any force/torque is suspiciously large ──────────
    f_max = np.abs(force).max()
    t_max = np.abs(torque).max()
    if f_max > 1e3 or t_max > 1e3 or not np.isfinite(f_max) or not np.isfinite(t_max):
        print(f"\n*** LARGE/NAN FORCE/TORQUE at t={t_sim:.5f} ***")
        print(f"  |m_rot| = {np.linalg.norm(m_rot):.4f}  m_rot = {m_rot}")
        print(f"  max |force| = {f_max:.4e}   max |torque| = {t_max:.4e}")
        print(f"  positions:\n{r}")
        print(f"  separations (min): {_min_separation(r):.4e}")
        print(f"  moments:\n{moments}")
        print(f"  E_mom:\n{E_mom}")
        print(f"  B_torque:\n{B_torque}")
        print(f"  forces:\n{force}")
        print(f"  torques:\n{torque}")

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


@njit(parallel=True, fastmath=True)
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
