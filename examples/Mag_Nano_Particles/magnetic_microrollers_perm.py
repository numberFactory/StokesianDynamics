"""
magnetic_suspension_perm.py
---------------------------
Stokesian Dynamics simulation of magnetic microparticles (a = 0.1 µm)
with PERMANENT magnetic moments only (no induced moment).

Changes from ladder_to_ring_EM_force.py:
  1) Induced moment removed — only permanent dipole interactions.
  2) Particle radius a = 0.1 µm.
  3) Random initial condition at packing fraction phi=0.4, N=50000,
     with mg = 2*kbt so particles are dispersed above the wall.
     Box size L solved from phi and N.  z_max = 10 * (2a), monitored.
  4) Every N_vel steps, linear velocities at a uniform grid of passive
     tracer points are computed via Mdot (forces + torques from real
     particles, zero force on tracers, linear velocity only extracted).
  5) Every N_plot steps, pyvista renders the periodically-wrapped
     particle positions as spheres above z=0.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import numpy as np
import time
from functools import partial
from numba import njit, prange
from scipy.spatial.transform import Rotation

from body import Body
from pyStokesianDynamics import pyStokesianDynamics

try:
    import pyvista as pv
    try:
        pv.start_xvfb()      # headless rendering — only needed if no display
    except Exception:
        pass                 # already have a display, or xvfb not needed
    pv.OFF_SCREEN = True
    HAS_PYVISTA = True
except Exception as e:
    HAS_PYVISTA = False
    print(f"WARNING: pyvista not available — 3-D plots will be skipped. ({e})")



# =============================================================================
def main():
# =============================================================================

    # ── Physical parameters ───────────────────────────────────────────────────
    a   = 1.0155 #25            # particle radius (µm)
    eta = 0.957e-3         # fluid viscosity (Pa·s)
    kbt = 0.0040749841   # thermal energy kT (pN·µm)
    Stoch = True        # Brownian motion on/off

    # ── Sedimentation: mg = 2*kbt ─────────────────────────────────────────────
    # g here is the net gravitational force (buoyancy-corrected) in pN
    g = 0.0303
    g_place = 0.0303   # stronger gravity for placement to reduce initial z_max

    # ── Magnetic parameters (permanent moment only) ───────────────────────────
    # mu_dipole = 2e-15 A·m^2 = 2 aJ/mT
    mu_dipole = 0.5          # permanent dipole magnitude (aJ/mT)
    B_0       = 4.0         # rotating field amplitude (mT)
    B_freq    = 9.0         # rotation frequency (Hz)
    # Coupling constant C = (3/(4*pi)) * mu0 * mu_dipole^2 / (r)^4
    # With mu_dipole in aJ/mT, distances in µm, forces in pN:
    # C ~ (3/(4*pi))* (1 attoJoule /millitesla )^2 * (4*pi*1e-7 Henry/m) / (1 um)^4 to pN
    C  = 0.3
    print(f"Dipole coupling C = {C:.4e} pN")

    # ── No induced moment — C_z removed entirely ─────────────────────────────

    # ── Interaction / steric parameters ──────────────────────────────────────
    firm_delta     = 1e-2
    debye_firm     = 2.0 * a * firm_delta / np.log(10.0)
    repulsion_firm = 0.0163               # firm repulsion strength
    repulsion_soft = 0.0326
    debye_soft     = 0.0406

    # ── Box geometry from packing fraction and N ──────────────────────────────
    N    = 2048
    phi  = 0.4
    # phi = N * (4/3 pi a^3) / (Lx * Ly * z_max)
    # Choose Lx = Ly = L_xy (square cross-section), z_max = 10*(2a)
    z_max_particles = 3.5 * (2.0 * a)         # initial guess: 10 diameters
    A_box   = N * np.pi * a**2 / phi
    L_xy    = float(np.sqrt(A_box))
    print(f"N={N}, phi={phi:.2f}")
    print(f"Box: Lx=Ly={L_xy:.4f} µm,  z_max={z_max_particles:.4f} µm")

    L             = np.array([L_xy, L_xy, 0.0])   # periodic in x,y; open in z
    #L             = np.array([0.0, 0.0, 0.0])
    z_max_solver  = z_max_particles                # SD solver cutoff

    # ── Time parameters ───────────────────────────────────────────────────────
    t_end   = 10.0              # stop time (s)
    dt      = 0.01     
    n_steps = int(t_end / dt)
    n_plot       = 100   # 3-D pyvista snapshot every n_plot steps
    n_vel        = 50    # particle velocity diagnostics every n_vel steps
    solver_tolerance = 5e-3

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mag_rollers_frames')
    os.makedirs(out_dir, exist_ok=True)

    # ── Rotating field — rotates in the yz plane ──────────────────────────────
    def m_rot_fn(t):
        return np.array([np.cos(2 * np.pi * B_freq * t),
                         0.0,
                         np.sin(2 * np.pi * B_freq * t)])

    # ── Place particles ───────────────────────────────────────────────────────
    print("Placing particles ...")
    t0_place = time.perf_counter()
    positions = place_particles(N, a, kbt, g_place, 0.99*L_xy, 0.99*L_xy, seed=42)
    print(f"  done in {time.perf_counter()-t0_place:.1f}s  "
          f"z range: [{positions[:,2].min():.4f}, {positions[:,2].max():.4f}] µm")

    # Random orientations
    rng = np.random.default_rng()
    bodies = []
    for pos in positions:
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        ori = Rotation.from_quat([q[1], q[2], q[3], q[0]])   # scipy: (x,y,z,w)
        bodies.append(Body(location=pos.copy(), orientation=ori))

    # ── Initialise solver ─────────────────────────────────────────────────────
    solver = pyStokesianDynamics(
        bodies=bodies, a=a, eta=eta,
        periodic_length=L, z_max=z_max_solver,
        debye_length=debye_firm,
    )
    solver.kT                  = kbt
    solver.dt                  = dt
    solver.tolerance           = solver_tolerance
    solver.num_rejections_wall = 0
    solver.num_rejections_jump = 0
    solver.Set_R_Mats()

    print(f"\nStarting simulation: N={N}, dt={dt}, n_steps={n_steps}, "
          f"t_end={t_end} s")
    print(f"  g={g:.6f} pN,  kbt={kbt:.6f} pN·µm,  mg/kbt={g/kbt:.2f}")
    print(f"  mu_dipole={mu_dipole:.4e} aJ/mT,  B_0={B_0} mT\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    t_wall    = time.perf_counter()

    for step in range(n_steps):
        t_sim = step * dt

        FT_calc = partial(
            force_torque_calculator,
            a=a, g=g, L=L,
            B_0=B_0,
            mu_dipole=mu_dipole,
            C=C,
            rep_firm=repulsion_firm, deb_firm=debye_firm,
            firm_delta=firm_delta,
            rep_soft=repulsion_soft, deb_soft=debye_soft,
            t_sim=t_sim,
            m_rot_fn=m_rot_fn,
        )



        r_before = np.array([b.location for b in bodies])      # (N, 3)

        solver.Update_Bodies_Trap(FT_calc, stochastic=Stoch, print_residual=True)
        #solver.print_timings()

        r_after  = np.array([b.location for b in bodies])      # (N, 3)

        # ── Monitor z_max ─────────────────────────────────────────────────────
        z_max_now = r_after[:, 2].max()
        print(f"  [step {step}] z_max is {(z_max_now + a):.4f} µm, z_solver_max is {z_max_solver:.4f} µm")

        # ── Particle velocity diagnostics ─────────────────────────────────────
        if step % n_vel == 0 and step > 0:
            speeds = np.linalg.norm((r_after - r_before) / dt, axis=1)  # (N,)
            elapsed = time.perf_counter() - t_wall
            print(f"  [vel] step={step:7d} t={t_sim:.4f}s  "
                  f"speed: min={speeds.min():.4e}  "
                  f"max={speeds.max():.4e}  "
                  f"median={np.median(speeds):.4e}  "
                  f"mean={speeds.mean():.4e}  "
                  f"elapsed={elapsed:.1f}s")

        # ── 3-D pyvista snapshot + data output ───────────────────────────────
        if step % n_plot == 0:
            speeds  = np.linalg.norm((r_after - r_before) / dt, axis=1)
            elapsed = time.perf_counter() - t_wall
            print(f"  [plot] step={step:7d}/{n_steps}  t={t_sim:.4f}s  "
                  f"z_max={z_max_now:.4f}  elapsed={elapsed:.1f}s")
            plot_frame_3d(bodies, speeds, a, L_xy, L, frame_idx, t_sim, out_dir)

            # r_vectors — shape (N, 3)
            np.savetxt(
                os.path.join(out_dir, f'r_vectors_{frame_idx:07d}.txt'),
                r_after,
                header='x y z',
                fmt='%.8e',
            )

            # velocities:
            vels = (r_after - r_before) / dt
            np.savetxt(
                os.path.join(out_dir, f'vels_{frame_idx:07d}.txt'),
                vels,
                header='Vx Vy Vz',
                fmt='%.8e',
            )

            # x-column of each rotation matrix — shape (N, 3)
            R_x = np.array([b.orientation.as_matrix()[:, 0] for b in bodies])
            np.savetxt(
                os.path.join(out_dir, f'orientation_x_{frame_idx:07d}.txt'),
                R_x,
                header='Rx_x Rx_y Rx_z',
                fmt='%.8e',
            )

            frame_idx += 1

    print(f"\nSimulation complete in {time.perf_counter()-t_wall:.1f}s")


# =============================================================================
# Force / torque calculator  (permanent moment only — no induced moment)
# =============================================================================
def force_torque_calculator(bodies, r_vecs, **kwargs):
    """
    Returns (2N, 3) array of [force_i, torque_i] for each body.

    Forces:
      - Gravity + firm wall repulsion  (single-body, z direction)
      - Pair magnetic dipole interaction (permanent moment only)
      - Pair steric repulsion (firm Yukawa)
      - External magnetic torque: tau = m_perm × B_applied
    """
    a          = kwargs['a']
    g          = kwargs['g']
    L          = kwargs['L']
    B_0        = kwargs['B_0']
    mu_dipole  = kwargs['mu_dipole']
    C          = kwargs['C']
    rep_firm   = kwargs['rep_firm']
    deb_firm   = kwargs['deb_firm']
    firm_delta = kwargs['firm_delta']
    rep_soft   = kwargs['rep_soft']
    deb_soft   = kwargs['deb_soft']
    t_sim      = kwargs['t_sim']
    m_rot_fn   = kwargs['m_rot_fn']

    m_rot = np.asarray(m_rot_fn(t_sim), dtype=np.float64)

    N = len(bodies)
    r = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3)

    # Rotation matrices and permanent moments (body x-axis in lab frame)
    R_mats = np.array([b.orientation.as_matrix() for b in bodies])
    m_perm = np.array([mu_dipole * R[:, 0] for R in R_mats])   # (N,3)

    # External torque: tau_i = m_perm_i × B_applied
    B_applied = B_0 * m_rot                                     # (3,)
    B_torque  = np.cross(m_perm, B_applied[np.newaxis, :])      # (N,3)


    force, torque = _pair_and_wall_forces_perm(
        r, m_perm, B_torque,
        L, a, g,
        rep_firm, deb_firm, firm_delta,
        rep_soft, deb_soft,
        C,
    )

    # # check for large forces and overlapping particles
    # for i in range(N):
    #     f_mag = np.linalg.norm(force[i])
    #     if f_mag > 1e3:
    #         print(f"WARNING: large force on particle {i}: {f_mag:.4e} pN")
    #     for j in range(i+1, N):
    #         dr = r[j] - r[i]
    #         for k in range(3):
    #             if L[k] > 0:
    #                 dr[k] -= round(dr[k] / L[k]) * L[k]
    #         r_norm = np.linalg.norm(dr)
    #         if r_norm < 2.0 * a:
    #             print(f"WARNING: particles {i} and {j} are overlapping "
    #                   f"(r={r_norm:.4e} µm)")

    FT = np.zeros((2 * N, 3))
    FT[0::2] = force
    FT[1::2] = torque
    return FT


@njit(parallel=True, fastmath=True)
def _pair_and_wall_forces_perm(r, m_perm, B_torque,
                                L, a, g,
                                rep_firm, deb_firm, firm_delta,
                                rep_soft, deb_soft,
                                C):
    """
    Numba kernel — permanent-moment only version.
    All induced-dipole (E_mom / C_z) terms removed.
    """
    N      = r.shape[0]
    force  = np.zeros((N, 3))
    torque = np.zeros((N, 3))

    # ── gravity + wall repulsion ──────────────────────────────────────────────
    for i in prange(N):
        force[i, 2] -= g
        h       = r[i, 2]
        contact = a * (1.0 - firm_delta)
        if h > contact:
            force[i, 2] += (rep_firm / deb_firm) * np.exp(-(h - contact) / deb_firm)
        else:
            force[i, 2] += rep_firm / deb_firm

    # ── pairwise: magnetic dipole + steric ────────────────────────────────────
    for i in prange(N):
        for j in range(N):
            if i == j:
                continue

            # minimum-image displacement r_j - r_i
            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r[j, k] - r[i, k]
                if L[k] > 0:
                    dr[k] -= round(dr[k] / L[k]) * L[k]

            r_norm = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            if r_norm < 1e-12:
                continue
            r_hat = dr / r_norm

            # ── permanent-moment dipole force & torque ────────────────────────
            m_i_norm = np.sqrt(m_perm[i,0]**2 + m_perm[i,1]**2 + m_perm[i,2]**2)
            m_j_norm = np.sqrt(m_perm[j,0]**2 + m_perm[j,1]**2 + m_perm[j,2]**2)

            m_i = m_perm[i, :] / m_i_norm
            m_j = m_perm[j, :] / m_j_norm
            mi_d_r  = m_i[0]*r_hat[0] + m_i[1]*r_hat[1] + m_i[2]*r_hat[2]
            mj_d_r  = m_j[0]*r_hat[0] + m_j[1]*r_hat[1] + m_j[2]*r_hat[2]
            mi_d_mj = m_i[0]*m_j[0]   + m_i[1]*m_j[1]   + m_i[2]*m_j[2]
            F_mag   = C * (m_i_norm * m_j_norm) / (r_norm**4)
            if F_mag > 1e-7:
                for k in range(3):
                    force[i, k] -= F_mag * (
                        mi_d_r * m_j[k] + mj_d_r * m_i[k]
                        + (mi_d_mj - 5.0 * mj_d_r * mi_d_r) * r_hat[k])

                T_mag   = (1.0 / 3.0) * C * (m_i_norm * m_j_norm) / (r_norm**3)
                mi_X_mj = np.cross(m_i, m_j)
                mi_X_r  = np.cross(m_i, r_hat)
                for k in range(3):
                    torque[i, k] += T_mag * (3.0 * mj_d_r * mi_X_r[k] - mi_X_mj[k])

            # ── steric repulsion ──────────────────────────────────────────────
            offset_firm = 2.0 * a * (1.0 - firm_delta)
            for k in range(3):
                if r_norm > offset_firm:
                    force[i, k] -= (
                        (rep_firm / deb_firm) *
                        np.exp(-(r_norm - offset_firm) / deb_firm) / r_norm
                    ) * dr[k]
                else:
                    force[i, k] -= (rep_firm / deb_firm / r_norm) * dr[k]
                if rep_soft > 0.0:
                    offset_soft = 2.0 * a
                    if r_norm > offset_soft:
                        force[i, k] -= (
                            (rep_soft / deb_soft) *
                            np.exp(-(r_norm - offset_soft) / deb_soft) / r_norm
                        ) * dr[k]
                    else:
                        force[i, k] -= (rep_soft / deb_soft / r_norm) * dr[k]

    # ── external torque from applied field ────────────────────────────────────
    for i in prange(N):
        for k in range(3):
            torque[i, k] += B_torque[i, k]

    return force, torque


# =============================================================================
# Tracer velocity grid
# =============================================================================
def make_tracer_grid(Lx, Ly, z_max, nx=10, ny=10, nz=5):
    """
    Return (nx*ny*nz, 3) array of uniformly-spaced tracer positions
    covering [0,Lx] x [0,Ly] x [0, z_max], cell-centred in each direction.
    """
    xs = np.linspace(0.0, Lx, nx, endpoint=False) + Lx / (2 * nx)
    ys = np.linspace(0.0, Ly, ny, endpoint=False) + Ly / (2 * ny)
    zs = np.linspace(0.0, z_max, nz, endpoint=False) + z_max / (2 * nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def compute_tracer_velocities(solver, r_vecs_flat, Lambda_s, tracer_points):
    """
    Compute linear velocities at passive tracer points using Mdot.

    The real particles contribute forces AND torques (Lambda_s contains
    both, interleaved as [fx,fy,fz, tx,ty,tz, ...] in the SD convention).
    Tracers are passive (zero force and zero torque); only their linear
    velocities are extracted from the result.

    Parameters
    ----------
    solver       : pyStokesianDynamics instance (provides Mdot)
    r_vecs_flat  : (3N,) flattened real-particle positions
    Lambda_s     : (6N,) flattened force/torque vector for real particles
    tracer_points: (M, 3) array of tracer positions

    Returns
    -------
    u, v, w : each (M,) — x, y, z linear velocity at each tracer point
    """
    M = tracer_points.shape[0]

    tracer_flat   = tracer_points.flatten()   # (3M,)
    tracer_lambda = np.zeros(6 * M)           # zero force + torque on tracers

    all_points = np.concatenate([r_vecs_flat,  tracer_flat  ])
    all_lambda = np.concatenate([Lambda_s,     tracer_lambda])

    all_vel = solver.Mdot(all_lambda, all_points)

    # Each particle contributes 6 entries in all_vel: [vx, vy, vz, ox, oy, oz]
    N_real     = len(r_vecs_flat) // 3
    tracer_vel = all_vel[6 * N_real:]         # skip real-particle entries

    u = tracer_vel[0::6]
    v = tracer_vel[1::6]
    w = tracer_vel[2::6]
    return u, v, w


# =============================================================================
# Visualisation — pyvista 3-D spheres
# =============================================================================
def plot_frame_3d(bodies, speeds, a, L_xy, L, frame_idx, t_sim, out_dir):
    """
    Render particles as spheres above z=0 using pyvista.
    Particles are wrapped periodically into [0,Lx] x [0,Ly].
    Colour encodes speed. Three-point lighting with shadow casting.
    """
    if not HAS_PYVISTA:
        print("pyvista not available — skipping 3-D plot.")
        return

    Lx, Ly = float(L[0]), float(L[1])
    locs = np.array([b.location for b in bodies], dtype=float)

    # periodic wrap in x and y
    if Lx > 0:
        locs[:, 0] = locs[:, 0] % Lx
    if Ly > 0:
        locs[:, 1] = locs[:, 1] % Ly

    cloud = pv.PolyData(locs)
    cloud['z_height'] = locs[:, 2]
    cloud['speeds']   = speeds

    spheres = cloud.glyph(
        geom=pv.Sphere(radius=a, theta_resolution=24, phi_resolution=24),
        scale=False,
        orient=False,
    )

    z_cam = 10.0 * a

    pl = pv.Plotter(off_screen=True, window_size=(1200, 900),
                    lighting='none')
    pl.set_background('black')

    # ── Lighting — positional light required for shadow casting ───────────────
    # Key light: high, to the left, positional so shadows are cast
    key = pv.Light(
        position=(L_xy * 0.0, -L_xy * 0.5, z_cam * 5.0),
        focal_point=(L_xy / 2, L_xy / 2, 0.0),
        color='white',
        intensity=1.0,
        positional=True,
        cone_angle=60,
        exponent=2,
    )
    pl.add_light(key)

    # Fill light: softer, from the right — camera-type so no shadow from this
    fill = pv.Light(
        position=(L_xy * 2.0, L_xy * 1.0, z_cam * 2.0),
        focal_point=(L_xy / 2, L_xy / 2, z_cam / 2),
        color='white',
        intensity=0.35,
        light_type='camera light',
    )
    pl.add_light(fill)

    # Rim light: cool tint from behind, traces silhouettes
    rim = pv.Light(
        position=(L_xy / 2, L_xy * 2.5, -z_cam * 0.3),
        focal_point=(L_xy / 2, L_xy / 2, z_cam / 2),
        color='lightblue',
        intensity=0.2,
        light_type='camera light',
    )
    pl.add_light(rim)

    # ── Enable shadows — must come before add_mesh ────────────────────────────
    #pl.enable_shadows()

    # ── Floor plane ───────────────────────────────────────────────────────────
    floor = pv.Plane(
    center=(L_xy / 2, L_xy / 2, 0.0),
    direction=(0, 0, 1),
    i_size=L_xy * 1.0,    # 3x larger than domain
    j_size=L_xy *1.0,
    i_resolution=200, j_resolution=200,
)
    pl.add_mesh(floor, color='#cccccc', opacity=1.0,
                ambient=0.3, diffuse=0.7, specular=0.1)

    # ── Spheres ───────────────────────────────────────────────────────────────
    c_low  = max(0.0, speeds.mean() - 2 * speeds.std())
    c_high = speeds.mean() + 2 * speeds.std()

    pl.add_mesh(
        spheres,
        scalars='speeds',
        cmap='plasma',
        clim=[c_low, c_high],
        show_scalar_bar=True,
        scalar_bar_args={'title': '|V| (µm/s)', 'color': 'white'},
        ambient=0.1,
        diffuse=0.7,
        specular=0.5,
        specular_power=40,
        smooth_shading=True,
    )

    # ── Camera ────────────────────────────────────────────────────────────────
    pl.camera_position = [
        (L_xy / 2, -0.85 * L_xy * 1.5, z_cam * 3),
        (L_xy / 2,  0.85 * L_xy / 2,   z_cam / 2),
        (0, 0, 1),
    ]

    pl.add_title(f't = {t_sim:.4f} s   z_max = {locs[:,2].max():.3f} µm',
                 font_size=14, color='white')

    fname = os.path.join(out_dir, f'frame3d_{frame_idx:07d}.png')
    pl.screenshot(fname)
    pl.close()
    print(f"    3-D frame saved → {fname}")


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
if __name__ == '__main__':
    main()
