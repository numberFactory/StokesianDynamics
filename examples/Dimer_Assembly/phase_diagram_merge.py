"""
Merge phase-diagram sweep: original structure/sweep-loop preserved from the
pasted phase-diagram script, with these changes from the ladder_to_ring code:

  1. Solver: Lub_Solver -> pyStokesianDynamics, using its construction /
     attribute-setting / Set_R_Mats() / Update_Bodies_Trap(FT_calc,
     stochastic=..., print_residual=...) calling convention.

  2. Force function: multi_bodies_functions.force_torque_calculator_sort_by_bodies
     -> force_torque_calculator (copied in below, unchanged from the
     validated ladder_to_ring version -- includes the optional
     self-consistent mutual-polarizability solve). use_permanent_moment and
     mutual_polarizability_mag are both toggles, defaulting to OFF, matching
     the reference phase-diagram script (no permanent moment, no mutual
     polarizability there).

  3. No input file: all physical parameters (a, eta, kT, g, firm_delta,
     repulsion_firm, solver_tolerance, dt) are hardcoded, identical to
     ladder_to_ring_EM_force.py's own values -- argparse/read_input have
     been removed entirely, matching ladder_to_ring's self-contained style.

Everything else -- the manual Nmer/mer setup, is_merged, the beta/
separation sweep loop and its early-exit-on-timeout assumption, periodic
plotting and config-file writing, nostdout -- is kept as close to the
original phase-diagram script as possible.

Field model (unchanged from before, different from ladder_to_ring's own
Lissajous in-plane field): the old script's B_vec (beta-angle-parameterized,
with a DC z component) + separate freq/time_s is wrapped as a single
B_field_fn(t) closure per (beta, sep) trial, since that's the interface
force_torque_calculator expects -- this reproduces the exact same field,
just packaged differently:

    B(t) = B_strength * [sin(beta)*cos(2*pi*freq*t),
                          sin(beta)*sin(2*pi*freq*t),
                          cos(beta)]

One genuinely new piece, not a straight port: `phase_mat` was created in
the original but never actually filled in or saved -- that's added here
(0 = no merge/timeout, 1 = merge, 2 = cluster), plus a saved .npy and a
rendered heatmap, since that's the actual deliverable of "a phase diagram."

Body construction is also necessarily different: ladder_to_ring's Body/
pyStokesianDynamics are built for single-blob rigid spheres (no vertex
file, no separate reference configuration), unlike the old multi-blob
body.Body(location, orientation, reference_configuration, a) constructor.
Bodies are now built with ladder_to_ring's Body(location=..., orientation=...)
and scipy Rotation instead of the custom Quaternion class -- read_vertex_file
/ blob.vertex are no longer needed and have been removed.
"""

import numpy as np
import scipy.sparse.linalg as spla
from functools import partial
import sys
import os
import time
import copy
import contextlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numba import njit, prange
from scipy.spatial.transform import Rotation

# Project imports -- single explicit path, matching ladder_to_ring.py's own
# convention. Adjust this one line if body/pyStokesianDynamics don't live
# under '../../src' relative to this file in your actual layout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from body import Body
from pyStokesianDynamics import pyStokesianDynamics


def is_merged(r_vecs, Nmer, a, sep):
    mer_1 = r_vecs[0:Nmer,:]
    mer_2 = r_vecs[Nmer::,:]
    ##########
    merge=0
    cluster=0
    ##########
    num_close = 0
    for v1 in mer_1:
        for v2 in mer_2:
            dist = np.linalg.norm(v1-v2)
            if ((dist/(2.0*a))-1.0) < sep: #0.5*(sep/a)
                num_close += 1
    if num_close == 1:
        merge = 1
    if num_close > 1:
        cluster = 1
    return merge, cluster


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def plot_mers_frame(fig, bodies, Nmer, a, B_field_fn, t_sim, out_path="mers.png"):
    """
    Render mers.png: an x-z snapshot of the two mers with a pseudo-3D depth
    cue and the current B-field drawn on every particle.

    - y-axis of the plot (== physical z, the wall-normal direction) starts
      at 0 (the wall) rather than being symmetric around 0 -- particles
      can't have z < 0, so the old symmetric range wasted half the plot on
      physically impossible space.
    - Perspective: particles are drawn back-to-front (farthest physical-y
      first, closest last), so the one nearer the camera correctly
      occludes the farther one, and each particle's *apparent* radius is
      scaled by simple pinhole-camera perspective (radius ~ 1/distance)
      using its actual y-coordinate, rather than every particle using the
      same fixed radius `a` regardless of depth.
    - Every particle gets a thick orange-gold arrow (solid triangular head)
      showing the current B-field, projected from the full 3-D field
      vector onto the x-z plane (a field with some y-component renders
      shorter, same "raw xy/xz-projection" convention used for the
      field/moment arrows in ladder_to_ring.py's plot_frame) -- total
      length up to 0.9x that particle's own apparent diameter. Draw order
      is disc-then-arrow for each particle in far-to-near sequence (i.e.
      bottom disc, bottom arrow, top disc, top arrow for the two-particle
      case): every arrow sits on top of its OWN disc, but a farther
      particle's disc+arrow pair is still entirely beneath a closer
      particle's disc+arrow pair.

    Camera model (a visualization choice -- there's no physical camera in
    the simulation itself -- tune d_cam below if the perspective effect
    looks too subtle or too strong): positioned at y = +d_cam, looking
    toward -y, so larger y = closer to the camera. d_cam is set well
    outside the visible x/z window so perspective is present but not
    distorting.
    """
    fig.clf()
    ax = fig.add_subplot(111)

    r_vecs   = np.array([b.location for b in bodies])
    y_coords = r_vecs[:, 1]

    view_half_width = 3 * 2 * Nmer * a
    d_cam = 10.0 * view_half_width               # camera distance -- tune to taste
    dist  = d_cam - y_coords                      # distance from camera, per particle
    apparent_radius = a * d_cam / dist            # perspective-scaled radius

    B_full = np.asarray(B_field_fn(t_sim), dtype=np.float64)
    B_mag  = np.linalg.norm(B_full)

    gold          = '#D4A017'                     # orange-gold field arrow
    blue_fill,    blue_edge    = 'cornflowerblue', '#26466D'   # mer 1
    magenta_fill, magenta_edge = '#D1618F',        '#8B3A62'   # mer 2 (soft reddish magenta)

    order = np.argsort(y_coords)   # farthest (smallest y) first, closest (largest y) last

    for rank, idx in enumerate(order):
        v   = r_vecs[idx]
        r_i = apparent_radius[idx]
        # draw order: bottom disc, bottom arrow, top disc, top arrow -- i.e.
        # each arrow sits on top of its OWN disc, but a farther particle's
        # disc+arrow pair is still entirely beneath a closer particle's.
        z_sphere, z_arrow = 10 + 2*rank, 11 + 2*rank

        fill, edge = (blue_fill, blue_edge) if idx < Nmer else (magenta_fill, magenta_edge)
        cir = plt.Circle((v[0], v[2]), r_i, facecolor=fill, edgecolor=edge,
                          linewidth=4, fill=True, zorder=z_sphere)
        ax.add_patch(cir)

        if B_mag > 1e-12:
            max_len = 0.9 * (2.0 * r_i)                # TOTAL length, up to 0.9x this particle's diameter
            half_span = 0.5 * max_len
            scale = half_span / B_mag
            dx, dz = B_full[0]*scale, B_full[2]*scale  # projected onto the x-z plane
            ax.annotate('', xy=(v[0]+dx, v[2]+dz), xytext=(v[0]-dx, v[2]-dz),
                        arrowprops=dict(arrowstyle='-|>', color=gold, lw=4.0,
                                         mutation_scale=22),
                        zorder=z_arrow)

    ax.set_xlim(-view_half_width, view_half_width)
    ax.set_ylim(0, view_half_width)                    # z starts at the wall, not below it
    ax.set_aspect('equal')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')
    fig.savefig(out_path)


# =============================================================================
# Magnetic force / torque calculator -- swapped in from ladder_to_ring.py,
# unchanged. use_permanent_moment and mutual_polarizability_mag both default
# to False (see force_torque_calculator below).
# =============================================================================
@njit(fastmath=True, parallel=True)
def _magnetic_mobility_numba(moments, r, L, a, chi, RB_0):
    """
    Numba matvec kernel for the magnetic-mobility linear operator A, where

        (A @ x)_i  =  x_i / RB_0  +  sum_{j != i} K_ij @ x_j

    K_ij is the near-field-regularized point-dipole field kernel per unit
    moment (far field: standard dipole tensor scaled by chi/3 * a^3 / r^3;
    near field, r <= 2a: a regularized polynomial). The x_i/RB_0 "self" term
    is intentional: it is what makes A represent "the trial moment's own
    single-particle response, plus the mutual dipole coupling from every
    other particle," so that solving A @ m_ind = RHS gives the
    self-consistent induced moment (see _solve_induced_moments_mutual).
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
    Self-consistently solve for the induced moment of every particle,
    accounting for mutual dipole coupling. Works for a general (in-plane or
    fully 3-D, e.g. with a DC z-component) applied field.

    RHS = B_applied - D[mom_perm], built as A(mom_perm) - mom_perm/RB_0 to
    cancel A's identity self-term (spurious when applied to mom_perm rather
    than the unknown m_ind -- see _magnetic_mobility_numba's docstring).
    When mom_perm is identically zero (use_permanent_moment=False), this
    correction is a no-op and RHS reduces to simply B_applied.
    """
    N = mom_perm.shape[0]
    system_size = 3 * N

    def matvec(x_flat):
        x = x_flat.reshape(N, 3)
        return _magnetic_mobility_numba(x, r, L, a, chi_exp, RB_0).reshape(-1)

    A = spla.LinearOperator((system_size, system_size), matvec=matvec, dtype=np.float64)

    A_mom_perm = matvec(mom_perm.reshape(-1)).reshape(N, 3)
    D_mom_perm = A_mom_perm - mom_perm / RB_0
    RHS = (B_applied - D_mom_perm).reshape(-1)

    x0_flat = (RB_0 * B_applied).reshape(-1) if x0 is None else np.asarray(x0).reshape(-1)

    m_ind, info = spla.gmres(A, RHS, x0=x0_flat, rtol=rtol, restart=restart,
                              maxiter=maxiter)

    return m_ind.reshape(N, 3)


def _compute_moments(R_mats, r, t_sim, B_field_fn, mu_dipole, RB_0, chi_exp, L, a,
                      use_permanent_moment, mutual_polarizability_mag,
                      gmres_rtol=1e-6, gmres_restart=50, gmres_maxiter=100):
    """
    Compute the permanent moment, applied field, and induced moment (mutual
    or simple, per mutual_polarizability_mag) for the current body
    configuration and a general applied field B_field_fn(t_sim) -> (3,).

    use_permanent_moment=False (the default): m_perm is identically zero for
    every particle -- purely paramagnetic/induced-moment particles, no
    permanent dipole, matching the reference phase-diagram script's model.
    """
    N = r.shape[0]
    B_applied = np.tile(np.asarray(B_field_fn(t_sim), dtype=np.float64), (N, 1))  # (N,3)

    if use_permanent_moment:
        m_perm = np.array([mu_dipole * R[:, 0] for R in R_mats])   # (N,3)
    else:
        m_perm = np.zeros((N, 3))

    if mutual_polarizability_mag:
        induced_mom = _solve_induced_moments_mutual(
            m_perm, B_applied, r, L, a, chi_exp, RB_0,
            rtol=gmres_rtol, restart=gmres_restart, maxiter=gmres_maxiter,
        )
    else:
        induced_mom = RB_0 * B_applied                          # (N,3), same for all

    return m_perm, induced_mom, B_applied


def force_torque_calculator(bodies, r_vecs, **kwargs):
    """
    Returns (2N, 3) array of [force_i, torque_i] for each body.

    Forces:
      - Gravity + firm wall repulsion  (single-body, z direction)
      - Pair magnetic dipole interaction (from B_field_fn(t), possibly with
        a DC z-component)
      - Pair steric repulsion (firm + soft Yukawa)
      - External magnetic torque  B x m

    kwargs must include B_field_fn: a callable (t_sim) -> (3,) field vector
    (amplitude included). use_permanent_moment (default False) and
    mutual_polarizability_mag (default False) are both explicit toggles.
    """
    a          = kwargs['a']
    g          = kwargs['g']
    L          = kwargs['L']
    B_field_fn = kwargs['B_field_fn']
    mu_dipole  = kwargs.get('mu_dipole', 0.0)
    RB_0       = kwargs['RB_0']
    chi_exp    = kwargs['chi_exp']
    C          = kwargs['C']
    C_z        = kwargs.get('C_z', 0.0)
    B_z        = kwargs.get('B_z', 0.0)
    rep_firm   = kwargs['rep_firm']
    deb_firm   = kwargs['deb_firm']
    firm_delta = kwargs['firm_delta']
    rep_soft   = kwargs['rep_soft']
    deb_soft   = kwargs['deb_soft']
    t_sim      = kwargs['t_sim']
    use_permanent_moment      = kwargs.get('use_permanent_moment', False)
    mutual_polarizability_mag = kwargs.get('mutual_polarizability_mag', False)
    gmres_rtol     = kwargs.get('gmres_rtol', 1e-6)
    gmres_restart  = kwargs.get('gmres_restart', 50)
    gmres_maxiter  = kwargs.get('gmres_maxiter', 100)

    N = len(bodies)
    r = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3)

    R_mats = np.array([b.orientation.as_matrix() for b in bodies])
    zax    = np.array([R[:, 2] for R in R_mats])               # (N,3) body z-axes
    B_z_vec = np.array([0.0, 0.0, 1.0])

    m_perm, induced_mom, B_applied = _compute_moments(
        R_mats, r, t_sim, B_field_fn, mu_dipole, RB_0, chi_exp, L, a,
        use_permanent_moment, mutual_polarizability_mag,
        gmres_rtol, gmres_restart, gmres_maxiter,
    )

    moments = m_perm + induced_mom                              # (N,3)
    E_mom   = np.tile(RB_0 * B_z * B_z_vec, (N, 1))              # (N,3) -- zero by default

    B_torque = np.cross(moments, B_applied)                     # (N,3)

    force, torque = _pair_and_wall_forces(
        r, moments, E_mom, zax, B_torque,
        L, a, g,
        rep_firm, deb_firm, firm_delta,
        rep_soft, deb_soft,
        C, C_z,
    )

    FT = np.zeros((2 * N, 3))
    FT[0::2] = force
    FT[1::2] = torque
    return FT


@njit(fastmath=True)
def _pair_and_wall_forces(r, moments, E_mom, zax, B_torque,
                          L, a, g,
                          rep_firm, deb_firm, firm_delta,
                          rep_soft, deb_soft,
                          C, C_z):
    """
    Numba kernel: computes all forces and torques.
      - Gravity + wall repulsion per particle
      - Pair magnetic dipole
      - Pair electric-field-induced dipole (z moments; zero by default here)
      - Pair steric repulsion
      - External magnetic torque (from B_torque array)
    """
    N      = r.shape[0]
    force  = np.zeros((N, 3))
    torque = np.zeros((N, 3))

    for i in prange(N):
        force[i, 2] -= g
        h       = r[i, 2]
        contact = a * (1.0 - firm_delta)
        if h > contact:
            force[i, 2] += (rep_firm / deb_firm) * np.exp(-(h - contact) / deb_firm)
        else:
            force[i, 2] += rep_firm / deb_firm

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
            r_hat = dr / r_norm

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

    for i in prange(N):
        for k in range(3):
            torque[i, k] += B_torque[i, k]

    return force, torque


if __name__ == '__main__':
    # ── Physical parameters -- identical to ladder_to_ring_EM_force.py ────────
    # No input file: this driver is self-contained, matching ladder_to_ring's
    # own style, rather than reading from a data.main config.
    a   = 2.25          # particle radius (µm)
    eta = 8.9e-4         # fluid viscosity
    kT  = 0.0041419464   # thermal energy
    g   = 0.28041        # gravitational acceleration (z)

    firm_delta      = 5e-5 #1e-3
    repulsion_firm  = 0.0331
    solver_tolerance = 5e-3
    dt = 5.0e-5 #6.25e-5

    # ── Magnetic / interaction parameters -- also identical to ladder_to_ring,
    # except mu_dipole/RB_0/chi_exp/C have no equivalent in an input-file
    # driven setup, so they were already hardcoded here rather than read
    # from a config file.
    mu_dipole  = 8.6     # permanent dipole (aJ/mT) -- unused unless use_permanent_moment=True
    RB_0       = 48.22   # susceptibility prefactor
    chi_exp    = 1.27    # magnetic susceptibility
    C          = 0.3     # dipole-dipole coupling constant

    use_permanent_moment      = False   # toggle, default off
    mutual_polarizability_mag = False   # toggle, default off
    gmres_rtol    = 1e-6
    gmres_restart = 100
    gmres_maxiter = 100

    repulsion_soft = 0.0     # soft Yukawa off, matching ladder_to_ring's default
    debye_soft     = 0.225

    debye_length_delta = 2.0*a*firm_delta/np.log(1.0e1)
    repulsion_strength_delta = repulsion_firm

    # ── open (non-periodic) box, z_max sized as in ladder_to_ring ─────────────
    L = np.array([0.0, 0.0, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    ############################
    Nmer = 2
    ############################

    # ── base (sep=0) locations: 2*Nmer touching particles along x ────────────
    struct_locations = []
    for k in range(-2*Nmer+1,2*Nmer,2):
        struct_locations.append(np.array([1.0*k*a,0.0,1.01*a]))
    num_bodies_struct = 2*Nmer

    # Create each body -- single-blob, ladder_to_ring style (Body(location=,
    # orientation=), no reference configuration / vertex file, identity
    # orientation since there's no permanent moment by default so
    # orientation doesn't affect the physics unless use_permanent_moment is
    # turned on).
    base_bodies = []
    for i in range(num_bodies_struct):
        b = Body(location=struct_locations[i].copy(), orientation=Rotation.identity())
        base_bodies.append(b)

    num_bodies = len(base_bodies)

    # ── solver, constructed once and reused (bodies/R-mats refreshed per trial) ─
    solver = pyStokesianDynamics(
        bodies=base_bodies, a=a, eta=eta,
        periodic_length=L, z_max=z_max_solver,
        debye_length=firm_delta,
    )
    solver.kT                  = kT
    solver.dt                  = dt
    solver.tolerance           = solver_tolerance
    solver.num_rejections_wall = 0
    solver.num_rejections_jump = 0

    total_rej = 0
    # amount of time that B-field is off and chain can equilibriate
    equilib_time = 0.0
    B_strength = 4.5 #4.0 #3.5 #3.0
    freq = 5.0

    betas = np.arange(65,50,-1) #[52] #
    time_out_time = 5.0
    merge_min_time = 0.2
    merge_sep_thresh = 0.01
    n_steps = int(time_out_time/dt) + 10   # upper bound; time_out_time is what actually governs stopping

    sep_array = [1.1,1.3,1.5,1.7,1.9,2.1] #[1.3] #
    num_seps = len(sep_array)
    phase_mat = np.zeros((len(betas),num_seps))

    cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cfg_data')
    os.makedirs(cfg_dir, exist_ok=True)

    f_name_base = 'Phase_Vid_Det_Nmer_'+str(Nmer)+'_Bstrength_'+str(B_strength)
    fig = plt.figure(1)

    for bb in range(len(betas)):
        beta_set = betas[bb]*(np.pi/180.0)
        sin_b = B_strength * np.sin(beta_set)
        cos_b = B_strength * np.cos(beta_set)

        # B_field_fn: the conical field this beta implies, wrapped as the
        # single (3,) function force_torque_calculator expects. Same field
        # as the original B_vec/B_rot, just packaged as one closure:
        #   B(t) = [sin_b*cos(2*pi*freq*t), sin_b*sin(2*pi*freq*t), cos_b]
        def B_field_fn(t, sin_b=sin_b, cos_b=cos_b):
            return np.array([sin_b*np.cos(2*np.pi*freq*t),
                              sin_b*np.sin(2*np.pi*freq*t),
                              cos_b])

        print(betas[bb])
        time_out = 0
        sep = 0
        for sk, sep_a in enumerate(sep_array):
            print("sep_a = ", sep_a)
            print("sk = ", sk)
            if time_out != 0:
                phase_mat[bb, sk] = 0
                break
            merge_steps = 0
            ###################
            new_bodies = []
            sep = sep_a*a #+0.1*a
            for k, b in enumerate(base_bodies):
                n_b = copy.deepcopy(b)
                new_bodies.append(n_b)
                if k < Nmer:
                    new_bodies[k].location[0] -= 0.5*sep
                else:
                    new_bodies[k].location[0] += 0.5*sep
            bodies = new_bodies
            solver.bodies = new_bodies
            solver.Set_R_Mats()
            ###################
            outcome = None
            for n in range(n_steps):
                time_s = n*dt + 0.05
                if n % 1000 == 0:
                    print("time_s = %f" % time_s)
                    plot_mers_frame(fig, bodies, Nmer, a, B_field_fn, time_s)
                ###############################
                # time out check
                ###############################
                if time_s > time_out_time:
                    time_out = 1
                    break

                FT_calc = partial(
                    force_torque_calculator,
                    a=a, g=g, L=L,
                    B_field_fn=B_field_fn,
                    mu_dipole=mu_dipole, RB_0=RB_0,
                    chi_exp=chi_exp, C=C,
                    rep_firm=repulsion_strength_delta, deb_firm=debye_length_delta,
                    firm_delta=firm_delta,
                    rep_soft=repulsion_soft, deb_soft=debye_soft,
                    t_sim=time_s,
                    use_permanent_moment=use_permanent_moment,
                    mutual_polarizability_mag=mutual_polarizability_mag,
                    gmres_rtol=gmres_rtol,
                    gmres_restart=gmres_restart,
                    gmres_maxiter=gmres_maxiter,
                )

                r_vecs = np.array([b.location for b in bodies])

                if (n % 4 == 0) and (time_s > 0.0):
                    f_name = os.path.join(
                        cfg_dir,
                        'config_'+f_name_base+'_beta_'+str(betas[bb])+'_sep_'+str(sep_a)+'.txt')
                    B_rot = B_field_fn(time_s)
                    with open(f_name, 'a') as f:
                        f.write('%f\n' % (time_s))
                        f.write('%f %f %f\n' % (B_rot[0], B_rot[1], B_rot[2]))
                        for pp, v in enumerate(r_vecs):
                            if pp < Nmer:
                                f.write('%f %f %f\n' % (v[0], v[1], v[2]))
                            else:
                                f.write('%f %f %f\n' % (v[0], v[1], v[2]))

                merge, cluster = is_merged(r_vecs, Nmer, a, merge_sep_thresh)

                if merge and (time_s > merge_min_time):
                    print("merge")
                    print("time_s = %f" % time_s)
                    plot_mers_frame(fig, bodies, Nmer, a, B_field_fn, time_s)
                    outcome = 'merge'
                    break
                if cluster:
                    print("cluster")
                    outcome = 'cluster'
                    break

                with nostdout():
                    solver.Update_Bodies_Trap(FT_calc, stochastic=False, print_residual=False)

                if solver.num_rejections_wall > 0:
                    print(("Number of rejected timesteps wall: %s" % solver.num_rejections_wall))
                if solver.num_rejections_jump > 0:
                    print(("Number of rejected timesteps jump: %s" % solver.num_rejections_jump))

            if outcome == 'merge':
                phase_mat[bb, sk] = 1
            elif outcome == 'cluster':
                phase_mat[bb, sk] = 2
            else:
                phase_mat[bb, sk] = 0
                # time_out was set to 1 inside the loop in this case, so the
                # `if time_out != 0` check at the top of the next sk
                # iteration will skip remaining (larger) separations.

    # ── save raw outcome matrix + a rendered phase-diagram heatmap ───────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(out_dir, f'phase_mat_Nmer_{Nmer}.npy'), phase_mat)

    # Symbols rather than a heatmap: X = no merge (phase_mat == 0), filled
    # teal dot with a dark teal outline = merge. "cluster" (phase_mat == 2)
    # is drawn the same as "merge" (phase_mat == 1) here -- both are cases
    # where the two mers ended up joined, just via a single bridging contact
    # vs. multiple simultaneous contacts -- since only two symbol categories
    # were asked for. Say the word if you'd rather cluster get its own
    # marker instead of being folded into "merge".
    #
    # Plotted via explicit (x, y) scatter coordinates (betas_grid/seps_grid
    # below), not an image array -- this is also what fixes the axis flip:
    # imshow's origin='lower' assumed phase_mat's rows increased with beta,
    # but betas is built in DESCENDING order (np.arange(65,50,-1) -> 65
    # down to 51), so row 0 (beta=65) was landing at the bottom of the
    # (ascending-labeled) y-axis instead of the top. Scatter has no such
    # row-order assumption -- each point is placed at its own true (sep,
    # beta) coordinate regardless of what order the arrays were built in.
    betas_grid, seps_grid = np.meshgrid(betas, sep_array, indexing='ij')
    no_merge_mask = (phase_mat == 0)
    merge_mask    = (phase_mat > 0)

    teal      = '#0f9c8f'
    dark_teal = '#0a5c54'

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(seps_grid[no_merge_mask], betas_grid[no_merge_mask],
                marker='x', color='black', s=70, linewidths=2, label='no merge')
    ax2.scatter(seps_grid[merge_mask], betas_grid[merge_mask],
                marker='o', facecolors=teal, edgecolors=dark_teal,
                linewidths=1.8, s=90, label='merge')
    ax2.set_xlabel('initial separation (multiples of a)')
    ax2.set_ylabel('beta (deg)')
    ax2.set_title(f'Merge phase diagram, Nmer={Nmer}')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f'phase_diagram_Nmer_{Nmer}.png'), dpi=150)
    plt.close(fig2)
