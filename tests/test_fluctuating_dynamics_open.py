"""
test_fluctuating_dynamics_open.py
----------------------------------
Fluctuating Stokesian Dynamics test — open (non-periodic) geometry with
a soft confinement potential replacing periodic boundaries.

Confinement:
  - In x and y: linear restoring force for particles outside [0, Lx] x [0, Ly]
    Force = -4*kT * displacement_outside_box  (per unit length)
  - Wall: exponential repulsion as before (z direction)
  - Gravity: as before

libMobility is initialised with NBody (open boundaries) so there is no
periodicity in the hydrodynamics either.

Everything else (MSD computation, neighbour list, parameters) is identical
to test_fluctuating_dynamics.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.spatial as spatial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from functools import partial
from numba import njit, prange, objmode
from scipy.spatial.transform import Rotation

from body import Body
from pyStokesianDynamics import pyStokesianDynamics


# =============================================================================
# Parameters  (same as periodic version)
# =============================================================================
a             = 1.395
eta           = 1.4e-3
kT            = 0.004075
g             = 0.0592
dt            = 0.25
t_max         = 1e2
n_steps       = int(t_max / dt)
n_save        = 2

# Box size — same as periodic version but now used only for confinement
Lx, Ly        = 320.0, 320.0
L_open        = np.array([0.0, 0.0, 0.0])   # no periodicity
z_max_solver  = 4.0 * (2.0 * a)

phi           = 0.34
N             = max(1, int(phi * Lx * Ly / (np.pi * a**2)))

firm_delta     = 1e-2
debye_firm     = 2.0 * a * firm_delta / np.log(10.0)
repulsion_firm = 0.0163
repulsion_soft = 0.0
debye_soft     = 0.1395

# Confinement spring constant: F = -k * displacement_outside
k_conf        = 4.0 * kT   # force per unit length outside box

pair_cutoff   = 2.0 * (2.0 * a)
buffer_skin   = pair_cutoff
nl_cutoff     = pair_cutoff + buffer_skin
rebuild_tol   = buffer_skin / 2.0

print(f"N = {N} particles, phi = {phi:.3f}, L = ({Lx},{Ly}) [open], "
      f"dt = {dt}, n_steps = {n_steps}")
print(f"Confinement spring: k = 4*kT = {k_conf:.4e} per unit length")


# =============================================================================
# MSD via FFT autocorrelation
# =============================================================================
def autocorrFFT(x):
    N_   = len(x)
    F    = np.fft.fft(x, n=2 * N_)
    PSD  = F * np.conjugate(F)
    res  = np.fft.ifft(PSD)
    res  = res[:N_].real
    n    = N_ * np.ones(N_) - np.arange(N_)
    return res / n


@njit(fastmath=True)
def msd_fft1d(r):
    N_  = len(r)
    D   = np.square(r)
    D   = np.append(D, 0.0)
    with objmode(S2='float64[:]'):
        S2 = autocorrFFT(r)
    Q   = 2.0 * D.sum()
    S1  = np.zeros(N_)
    for m in range(N_):
        Q      = Q - D[m - 1] - D[N_ - m]
        S1[m]  = Q / (N_ - m)
    return S1 - 2.0 * S2


@njit(parallel=True, fastmath=True)
def msd_matrix(matrix):
    Nrows, Ncols = matrix.shape
    MSDs = np.zeros((Nrows, Ncols))
    for i in prange(Nrows):
        MSDs[i, :] = msd_fft1d(matrix[i, :])
    return MSDs


# =============================================================================
# Force kernels
# =============================================================================
@njit(parallel=True, fastmath=True)
def _wall_force_numba(r, a, g, rep_firm, deb_firm, firm_delta):
    N_  = r.shape[0]
    f   = np.zeros((N_, 3))
    deb = 0.5 * deb_firm
    for i in prange(N_):
        f[i, 2] -= g
        h        = r[i, 2]
        contact  = a * (1.0 - firm_delta)
        if h > contact:
            f[i, 2] += (rep_firm / deb) * np.exp(-(h - contact) / deb)
        else:
            f[i, 2] += rep_firm / deb
    return f


@njit(parallel=True, fastmath=True)
def _confinement_force_numba(r, Lx, Ly, k_conf):
    """
    Linear restoring force for particles outside [0, Lx] x [0, Ly].
    F_x = -k_conf * max(0, x - Lx)  +  k_conf * max(0, -x)
    Similarly for y. No force in z.
    """
    N_  = r.shape[0]
    f   = np.zeros((N_, 3))
    for i in prange(N_):
        x = r[i, 0]
        y = r[i, 1]
        if x > Lx:
            f[i, 0] -= k_conf * (x - Lx)
        elif x < 0.0:
            f[i, 0] -= k_conf * x       # x < 0, so this is positive (pushes right)
        if y > Ly:
            f[i, 1] -= k_conf * (y - Ly)
        elif y < 0.0:
            f[i, 1] -= k_conf * y
    return f


@njit(parallel=True, fastmath=True)
def _pair_force_numba(r, pair_cutoff, a, rep_firm, deb_firm, firm_delta,
                      rep_soft, deb_soft, neighbors, offsets):
    """Pair forces — no periodic wrapping (open geometry)."""
    N_ = r.shape[0]
    f  = np.zeros((N_, 3))
    for i in prange(N_):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = neighbors[offsets[i] + kk]
            if j == i:
                continue
            dr = np.zeros(3)
            for d in range(3):
                dr[d] = r[j, d] - r[i, d]
            rn = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            if rn > pair_cutoff:
                continue
            rn_safe     = max(rn, 1e-12)
            offset_firm = 2.0 * a * (1.0 - firm_delta)
            if rn > offset_firm:
                fmag = -(rep_firm / deb_firm) * np.exp(
                    -(rn - offset_firm) / deb_firm) / rn_safe
            else:
                fmag = -(rep_firm / deb_firm) / rn_safe
            for d in range(3):
                f[i, d] += fmag * dr[d]
    return f


def build_neighbour_list(r, cutoff):
    """Build buffered flat neighbour list — no periodic wrapping."""
    tree    = spatial.cKDTree(r)
    pairs   = tree.query_ball_tree(tree, cutoff)
    Np      = r.shape[0]
    offsets = np.zeros(Np + 1, dtype=np.int64)
    for i in range(Np):
        offsets[i + 1] = offsets[i] + len(pairs[i])
    neighbors = np.concatenate(pairs).ravel().astype(np.int64) \
        if offsets[-1] > 0 else np.empty(0, dtype=np.int64)
    return neighbors, offsets


def needs_rebuild(r_now, r_nl_ref, tol):
    """True if any particle has moved more than tol since last rebuild."""
    return bool(np.any(np.linalg.norm(r_now - r_nl_ref, axis=1) > tol))


def force_torque_calculator(bodies, r_vecs, a, g, Lx, Ly, k_conf,
                             pair_cutoff, rep_firm, deb_firm, firm_delta,
                             rep_soft, deb_soft, neighbors, offsets):
    r  = np.asarray(r_vecs, dtype=np.float64).reshape(-1, 3)
    f  = _wall_force_numba(r, a, g, rep_firm, deb_firm, firm_delta)
    f += _confinement_force_numba(r, Lx, Ly, k_conf)
    f += _pair_force_numba(r, pair_cutoff, a, rep_firm, deb_firm, firm_delta,
                           rep_soft, deb_soft, neighbors, offsets)
    FT       = np.zeros((2 * r.shape[0], 3))
    FT[0::2] = f
    return FT


# =============================================================================
# Initialise particles — non-overlapping, z ~ exponential
# =============================================================================
rng          = np.random.default_rng(42)
z_mean       = kT / g
min_sep      = 2.0 * a
max_attempts = 1000 * N

positions = []
attempts  = 0
while len(positions) < N and attempts < max_attempts:
    attempts += 1
    x     = rng.uniform(0.0, Lx)
    y     = rng.uniform(0.0, Ly)
    z_gap = rng.exponential(z_mean)
    z     = a + z_gap
    ok    = True
    for p in positions:
        dx = x - p[0]
        dy = y - p[1]
        dz = z - p[2]
        if dx**2 + dy**2 + dz**2 < min_sep**2:
            ok = False
            break
    if ok:
        positions.append([x, y, z])

if len(positions) < N:
    raise RuntimeError(
        f"Could only place {len(positions)}/{N} non-overlapping particles "
        f"after {max_attempts} attempts. Reduce phi or increase box size.")

identity_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
bodies       = [Body(location=np.array(p), orientation=identity_rot)
                for p in positions]

# =============================================================================
# Initialise solver — open geometry (NBody)
# =============================================================================
solver = pyStokesianDynamics(
    bodies=bodies, a=a, eta=eta,
    periodic_length=L_open,        # [0,0,0] → NBody open solver
    z_max=z_max_solver,
    debye_length=firm_delta,
)
solver.kT                  = kT
solver.dt                  = dt
solver.tolerance           = 1e-4
solver.num_rejections_wall = 0
solver.num_rejections_jump = 0
solver.Set_R_Mats()

# =============================================================================
# Initial neighbour list (no periodic wrapping)
# =============================================================================
r_now = np.array([b.location for b in bodies])
nl_nbrs, nl_offsets = build_neighbour_list(r_now, nl_cutoff)
r_nl_ref = r_now.copy()

FT_calc = partial(
    force_torque_calculator,
    a=a, g=g, Lx=Lx, Ly=Ly, k_conf=k_conf,
    pair_cutoff=pair_cutoff,
    rep_firm=repulsion_firm, deb_firm=debye_firm, firm_delta=firm_delta,
    rep_soft=repulsion_soft, deb_soft=debye_soft,
    neighbors=nl_nbrs, offsets=nl_offsets,
)

# =============================================================================
# Trajectory storage
# =============================================================================
n_frames      = n_steps // n_save + 1
traj_r        = np.zeros((N, n_frames))
frame_idx     = 0
save_times    = np.zeros(n_frames)

r_init        = np.array([b.location for b in bodies])
traj_r[:, 0]  = np.sqrt(r_init[:, 0]**2 + r_init[:, 1]**2)
save_times[0] = 0.0

# =============================================================================
# Main loop
# =============================================================================
nl_rebuilds = 0
t_wall      = time.perf_counter()

for step in range(1, n_steps + 1):
    r_now = np.array([b.location for b in bodies])

    if needs_rebuild(r_now, r_nl_ref, rebuild_tol):
        nl_nbrs, nl_offsets = build_neighbour_list(r_now, nl_cutoff)
        r_nl_ref = r_now.copy()
        FT_calc  = partial(
            force_torque_calculator,
            a=a, g=g, Lx=Lx, Ly=Ly, k_conf=k_conf,
            pair_cutoff=pair_cutoff,
            rep_firm=repulsion_firm, deb_firm=debye_firm, firm_delta=firm_delta,
            rep_soft=repulsion_soft, deb_soft=debye_soft,
            neighbors=nl_nbrs, offsets=nl_offsets,
        )
        nl_rebuilds += 1

    np.savez('debug_positions.npz', step=step, time=step*dt, positions=r_now)
    solver.Update_Bodies_Trap(FT_calc, stochastic=True)
    #solver.print_timings()

    if step % n_save == 0:
        frame_idx += 1
        r_cur = np.array([b.location for b in bodies])
        traj_r[:, frame_idx] = np.sqrt(r_cur[:, 0]**2 + r_cur[:, 1]**2)
        save_times[frame_idx] = step * dt

        if frame_idx % 50 == 0:
            elapsed = time.perf_counter() - t_wall
            print(f"  frame {frame_idx:5d}/{n_frames-1}  t={step*dt:.0f}  "
                  f"NL rebuilds={nl_rebuilds}  "
                  f"rej_wall={solver.num_rejections_wall}  "
                  f"rej_jump={solver.num_rejections_jump}  "
                  f"elapsed={elapsed:.0f}s")

print(f"\nSimulation complete in {time.perf_counter()-t_wall:.1f}s  "
      f"NL rebuilds={nl_rebuilds}")

# =============================================================================
# MSD
# =============================================================================
print("Computing MSD via FFT...")
all_msds = msd_matrix(traj_r)
msd_mean = all_msds.mean(axis=0)
msd_std  = all_msds.std(axis=0)
msd_err  = 2.0 * msd_std / np.sqrt(N)
lag_dt   = n_save * dt
lag_times = np.arange(n_frames) * lag_dt

# diffusion coefficient from first non-zero lag
# MSD_parallel(t) = 4 D t  (2D in the x-y plane)
# 2 instead of 4 because we are looking at the radial coordinate r = sqrt(x^2 + y^2)
D_meas  = msd_mean[1] / (2.0 * lag_times[1])
D0_free = kT / (6.0 * np.pi * eta * a)
print(f"Measured D_parallel = {D_meas:.4e}  (free D0 = {D0_free:.4e})")

mask   = (lag_times > 0) & (lag_times <= 1e3)
t_pl   = lag_times[mask]
msd_pl = msd_mean[mask]
err_pl = msd_err[mask]

fig, ax = plt.subplots(figsize=(7, 5))
ax.fill_between(t_pl, msd_pl - err_pl, msd_pl + err_pl,
                alpha=0.25, color='steelblue',
                label=r'$\pm 2\sigma/\sqrt{N}$')
ax.plot(t_pl, msd_pl, color='steelblue', lw=1.5, label='MSD (parallel)')
ax.plot(t_pl, 4.0 * D0_free * t_pl, 'k--', lw=0.8,
        label=r'$4D_0 t$ (free, 2D)')
ax.plot(t_pl, 4.0 * D_meas  * t_pl, 'r:',  lw=1.0,
        label=rf'$4D_{{\rm meas}}t$  ($D={D_meas:.2e}$)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(t_pl.min(), 1e3)
ax.set_xlabel(r'$\tau$ (simulation units)', fontsize=13)
ax.set_ylabel(r'MSD$_\parallel = \langle \Delta r^2 \rangle$', fontsize=13)
ax.set_title(f'Parallel MSD (open) — N={N}, phi={phi:.2f}, kT={kT}', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', ls=':', alpha=0.4)
fig.tight_layout()
plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'msd_fluctuating_open.png')
fig.savefig(plot_path, dpi=150)
print(f"MSD plot saved to {plot_path}")
