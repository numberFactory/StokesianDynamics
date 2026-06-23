"""
test_fluctuating_dynamics.py
----------------------------
Self-contained fluctuating Stokesian Dynamics test for N point particles
above a single wall with periodic boundaries in x and y.

Forces:
  - Gravity (z direction)
  - Exponential wall repulsion (firm-contact potential)
  - Exponential pair repulsion (firm + soft Yukawa)

A buffered neighbour list (cutoff = 2 * pair_cutoff) is maintained in
the main loop and only rebuilt when any particle has moved more than
half the buffer shell thickness since the last rebuild.

After the run, the parallel MSD (radial coordinate r = sqrt(x^2+y^2))
is computed per-particle using FFT autocorrelation, then averaged
pointwise across particles. The diffusion coefficient is extracted
from the first non-zero lag point. Error bars are 2*std/sqrt(N).
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
def main():
# =============================================================================

    # ── Physical parameters ───────────────────────────────────────────────────
    a   = 1.395        # particle radius
    eta = 1.4e-3       # fluid viscosity
    kT  = 0.004075     # thermal energy
    g   = 0.0592       # gravitational acceleration (z)

    # ── Interaction parameters ────────────────────────────────────────────────
    firm_delta     = 1e-2
    debye_firm     = 2.0 * a * firm_delta / np.log(10.0)
    repulsion_firm = 0.0163
    repulsion_soft = 0.0
    debye_soft     = 0.1395

    # ── Box geometry ──────────────────────────────────────────────────────────
    Lx, Ly       = 160.0, 160.0
    L            = np.array([Lx, Ly, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    # ── Packing fraction ──────────────────────────────────────────────────────
    phi = 0.66
    N   = max(1, int(phi * Lx * Ly / (np.pi * a**2)))

    # ── Simulation parameters ─────────────────────────────────────────────────
    dt      = 0.25 #use dt = 0.05 to get a more accurate MSD at short times (which will match those reported in https://doi.org/10.1103/PhysRevX.14.041016)
    t_max   = 1.1e3
    n_steps = int(t_max / dt)
    n_save  = 1          # save radial coordinate every n_save steps
    solver_tolerance = 5e-3

    # ── Neighbour list parameters ─────────────────────────────────────────────
    pair_cutoff = 2.0 * (2.0 * a)
    buffer_skin = pair_cutoff
    nl_cutoff   = pair_cutoff + buffer_skin
    rebuild_tol = buffer_skin / 2.0

    # ── Output ────────────────────────────────────────────────────────────────
    out_dir    = os.path.dirname(os.path.abspath(__file__))
    out_prefix = os.path.join(out_dir, f'msd_phi{phi:.2f}_L{Lx:.0f}x{Ly:.0f}')

    print(f"N = {N} particles, phi = {phi:.3f}, L = ({Lx},{Ly}), "
          f"dt = {dt}, n_steps = {n_steps}")

    # ── Initialise particles ──────────────────────────────────────────────────
    positions    = place_particles(N, a, kT, g, Lx, Ly, seed=42)
    print(f"Placed {len(positions)}/{N} particles.")
    identity_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    bodies       = [Body(location=np.array(p), orientation=identity_rot)
                    for p in positions]

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

    # ── Initial neighbour list and force calculator ───────────────────────────
    r_now     = np.array([b.location for b in bodies])
    r_wrapped = wrap_positions(r_now, L)
    nl_nbrs, nl_offsets = build_neighbour_list(r_wrapped, L, nl_cutoff)
    r_nl_ref  = r_now.copy()

    FT_calc = partial(force_torque_calculator,
                      a=a, g=g, L=L, pair_cutoff=pair_cutoff,
                      rep_firm=repulsion_firm, deb_firm=debye_firm,
                      firm_delta=firm_delta,
                      rep_soft=repulsion_soft, deb_soft=debye_soft,
                      neighbors=nl_nbrs, offsets=nl_offsets)

    # ── Trajectory storage ────────────────────────────────────────────────────
    n_frames   = n_steps // n_save + 1
    traj_r     = np.zeros((N, n_frames))
    frame_idx  = 0
    r_init     = np.array([b.location for b in bodies])
    traj_r[:, 0] = np.sqrt(r_init[:, 0]**2 + r_init[:, 1]**2)

    # ── Main loop ─────────────────────────────────────────────────────────────
    nl_rebuilds = 0
    t_wall      = time.perf_counter()

    for step in range(1, n_steps + 1):
        r_now     = np.array([b.location for b in bodies])
        r_wrapped = wrap_positions(r_now, L)

        if needs_rebuild(r_now, r_nl_ref, L, rebuild_tol):
            nl_nbrs, nl_offsets = build_neighbour_list(r_wrapped, L, nl_cutoff)
            r_nl_ref = r_now.copy()
            FT_calc  = partial(force_torque_calculator,
                               a=a, g=g, L=L, pair_cutoff=pair_cutoff,
                               rep_firm=repulsion_firm, deb_firm=debye_firm,
                               firm_delta=firm_delta,
                               rep_soft=repulsion_soft, deb_soft=debye_soft,
                               neighbors=nl_nbrs, offsets=nl_offsets)
            nl_rebuilds += 1

        np.savez('debug_positions.npz', step=step, time=step*dt, positions=r_now)
        solver.Update_Bodies_Trap(FT_calc, stochastic=True, print_residual=False)
        #solver.print_timings()

        if step % n_save == 0:
            frame_idx += 1
            r_cur = np.array([b.location for b in bodies])
            traj_r[:, frame_idx] = np.sqrt(r_cur[:, 0]**2 + r_cur[:, 1]**2)

            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_wall
                print(f"  frame {frame_idx:5d}/{n_frames-1}  t={step*dt:.0f}  "
                      f"NL rebuilds={nl_rebuilds}  "
                      f"rej_wall={solver.num_rejections_wall}  "
                      f"rej_jump={solver.num_rejections_jump}  "
                      f"elapsed={elapsed:.0f}s")

                traj_so_far   = traj_r[:, :frame_idx + 1]
                all_msds_now  = msd_matrix(traj_so_far)
                msd_mean_now  = all_msds_now.mean(axis=0)
                msd_err_now   = 2.0 * all_msds_now.std(axis=0) / np.sqrt(N)
                lag_times_now = np.arange(frame_idx + 1) * (n_save * dt)
                plot_and_save_msd(msd_mean_now, msd_err_now, lag_times_now,
                                  out_prefix, phi, N, kT, eta, a,
                                  tag=f'frame{frame_idx:06d}', t_sim=step * dt)

    print(f"\nSimulation complete in {time.perf_counter()-t_wall:.1f}s  "
          f"NL rebuilds={nl_rebuilds}")

    # ── Final MSD ─────────────────────────────────────────────────────────────
    print("Computing final MSD via FFT...")
    all_msds  = msd_matrix(traj_r)
    msd_mean  = all_msds.mean(axis=0)
    msd_err   = 2.0 * all_msds.std(axis=0) / np.sqrt(N)
    lag_times = np.arange(n_frames) * (n_save * dt)
    t_fit     = lag_times[1:4]
    msd_fit   = msd_mean[1:4]
    coeffs, _, _, _ = np.linalg.lstsq(
        np.column_stack([t_fit, t_fit**2]), msd_fit, rcond=None)
    D_meas    = coeffs[0] / 2.0
    D0_free   = kT / (6.0 * np.pi * eta * a)
    print(f"Measured D_parallel = {D_meas:.4e}  (free D0 = {D0_free:.4e})")
    plot_and_save_msd(msd_mean, msd_err, lag_times,
                      out_prefix, phi, N, kT, eta, a, tag='final')
    print(f"Final MSD saved to {out_prefix}_final.{{npz,png}}")


# =============================================================================
# MSD plot / save
# =============================================================================
def plot_and_save_msd(msd_mean, msd_err, lag_times, out_prefix, phi, N,
                      kT, eta, a, tag, t_sim=None):
    """
    Save npz and png for the given MSD arrays.
    Overlays experimental MSD if available for the current phi.
    """
    D0_free = kT / (6.0 * np.pi * eta * a)
    # Quadratic fit to first 3 non-zero lag points: MSD ≈ 2*D*t + c*t^2
    # Fit in log space is unreliable at short times; use linear lstsq instead.
    if len(lag_times) >= 4 and lag_times[1] > 0:
        t_fit   = lag_times[1:4]
        msd_fit = msd_mean[1:4]
        # design matrix: [t, t^2]
        A       = np.column_stack([t_fit, t_fit**2])
        coeffs, _, _, _ = np.linalg.lstsq(A, msd_fit, rcond=None)
        D_meas  = coeffs[0] / 2.0   # MSD = 2*D*t + c*t^2 => D = slope/2
    elif lag_times[1] > 0:
        D_meas  = msd_mean[1] / (2.0 * lag_times[1])
    else:
        D_meas  = float('nan')

    np.savez(f'{out_prefix}.npz',
             msd_mean=msd_mean, msd_err=msd_err,
             lag_times=lag_times, N=N, phi=phi,
             D_meas=D_meas, D0_free=D0_free)

    mask = (lag_times > 0) & (lag_times <= 1e3)
    if not mask.any():
        return
    t_pl   = lag_times[mask]
    msd_pl = msd_mean[mask]
    err_pl = msd_err[mask]

    title = f'MSD  N={N}, phi={phi:.2f}'
    if t_sim is not None:
        title += f', t={t_sim:.0f}'

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(t_pl, msd_pl - err_pl, msd_pl + err_pl,
                    alpha=0.25, color='steelblue',
                    label=r'$\pm 2\sigma/\sqrt{N}$')
    ax.plot(t_pl, msd_pl, '--', color='steelblue', lw=1.5,
            label='MSD sim (parallel)')
    if np.isfinite(D_meas):
        ax.plot(t_pl, 2.0 * D_meas * t_pl, 'k:', lw=1.0,
                label=rf'$4D_{{\rm meas}}t$  ($D={D_meas:.2e}$)')

    t_exp, msd_exp = _get_exp_msd(phi)
    if t_exp is not None:
        mask_exp = (t_exp > 0) & (t_exp <= 1e3)
        ax.plot(t_exp[mask_exp], msd_exp[mask_exp],
                '-o', ms=4, color='darkorange', lw=3.0, zorder=5,
                label=f'Exp. phi={phi:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t_pl.min(), 1e3)
    ax.set_xlabel(r'$\tau$ (simulation units)', fontsize=13)
    ax.set_ylabel(r'MSD$_\parallel = \langle \Delta r^2 \rangle$', fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}.png', dpi=150)
    plt.close(fig)


# =============================================================================
# MSD via FFT autocorrelation
# =============================================================================
def autocorrFFT(x):
    """Normalised autocorrelation of 1D array x via FFT (convention A)."""
    N_  = len(x)
    F   = np.fft.fft(x, n=2 * N_)
    PSD = F * np.conjugate(F)
    res = np.fft.ifft(PSD)
    res = res[:N_].real
    n   = N_ * np.ones(N_) - np.arange(N_)
    return res / n


@njit(fastmath=True)
def msd_fft1d(r):
    """FFT-based MSD for a single 1D trajectory r (length T)."""
    N  = len(r)
    D  = np.square(r)
    D  = np.append(D, 0.0)
    with objmode(S2='float64[:]'):
        S2 = autocorrFFT(r)
    Q  = 2.0 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q     = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2.0 * S2


@njit(parallel=True, fastmath=True)
def msd_matrix(matrix):
    """
    Compute FFT-based MSD for each row (shape N_particles x T_frames).
    Returns MSD matrix of same shape.
    """
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
    N   = r.shape[0]
    f   = np.zeros((N, 3))
    deb = 0.5 * deb_firm
    for i in prange(N):
        f[i, 2] -= g
        h        = r[i, 2]
        contact  = a * (1.0 - firm_delta)
        if h > contact:
            f[i, 2] += (rep_firm / deb) * np.exp(-(h - contact) / deb)
        else:
            f[i, 2] += rep_firm / deb
    return f


@njit(parallel=True, fastmath=True)
def _pair_force_numba(r, L, a, rep_firm, deb_firm, firm_delta,
                      rep_soft, deb_soft, pair_cutoff, neighbors, offsets):
    N = r.shape[0]
    f = np.zeros((N, 3))
    for i in prange(N):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = neighbors[offsets[i] + kk]
            if j == i:
                continue
            dr = np.zeros(3)
            for d in range(3):
                dr[d] = r[j, d] - r[i, d]
                if L[d] > 0:
                    dr[d] -= int(dr[d] / L[d] + 0.5 * (
                        int(dr[d] > 0) - int(dr[d] < 0))) * L[d]
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
            if rep_soft > 0.0:
                offset_soft = 2.0 * a
                if rn > offset_soft:
                    smag = -(rep_soft / deb_soft) * np.exp(
                        -(rn - offset_soft) / deb_soft) / rn_safe
                else:
                    smag = -(rep_soft / deb_soft) / rn_safe
                for d in range(3):
                    f[i, d] += smag * dr[d]
    return f


def wrap_positions(r, L):
    """Return a copy of (N,3) positions wrapped into [0, L) for periodic dims."""
    r_w = r.copy()
    for dim in range(3):
        if L[dim] > 0:
            r_w[:, dim] = r_w[:, dim] % L[dim]
    return r_w


def build_neighbour_list(r_wrapped, L, cutoff):
    """Build buffered flat neighbour list from wrapped positions."""
    bs      = np.array([L[0] if L[0] > 0 else 1e30,
                        L[1] if L[1] > 0 else 1e30,
                        1e30])
    tree    = spatial.cKDTree(r_wrapped, boxsize=1.001 * bs)
    pairs   = tree.query_ball_tree(tree, cutoff)
    Np      = r_wrapped.shape[0]
    offsets = np.zeros(Np + 1, dtype=np.int64)
    for i in range(Np):
        offsets[i + 1] = offsets[i] + len(pairs[i])
    neighbors = np.concatenate(pairs).ravel().astype(np.int64) \
        if offsets[-1] > 0 else np.empty(0, dtype=np.int64)
    return neighbors, offsets


def needs_rebuild(r_now, r_nl_ref, L, tol):
    """True if any particle has moved more than tol (minimum image) since rebuild."""
    d = r_now - r_nl_ref
    for dim in range(2):
        if L[dim] > 0:
            d[:, dim] -= np.round(d[:, dim] / L[dim]) * L[dim]
    return bool(np.any(np.linalg.norm(d, axis=1) > tol))


def force_torque_calculator(bodies, r_vecs, a, g, L, pair_cutoff,
                             rep_firm, deb_firm, firm_delta,
                             rep_soft, deb_soft, neighbors, offsets):
    """Returns (2N,3) [force_i, torque_i]. Uses pre-built neighbour list."""
    r        = np.asarray(r_vecs, dtype=np.float64).reshape(-1, 3)
    f        = _wall_force_numba(r, a, g, rep_firm, deb_firm, firm_delta)
    f       += _pair_force_numba(r, L, a, rep_firm, deb_firm, firm_delta,
                                 rep_soft, deb_soft, pair_cutoff,
                                 neighbors, offsets)
    FT       = np.zeros((2 * r.shape[0], 3))
    FT[0::2] = f
    return FT


# =============================================================================
# Particle placement
# =============================================================================
@njit(parallel=True, fastmath=True)
def _check_overlaps(positions, x, y, z, min_sep_sq, n_placed, Lx, Ly):
    """Parallel overlap check with minimum-image wrapping in x and y."""
    overlaps = False
    for i in prange(n_placed):
        dx = x - positions[i, 0]
        dy = y - positions[i, 1]
        dz = z - positions[i, 2]
        if Lx > 0.0:
            dx -= round(dx / Lx) * Lx
        if Ly > 0.0:
            dy -= round(dy / Ly) * Ly
        if dx*dx + dy*dy + dz*dz < min_sep_sq:
            overlaps = True
    return overlaps


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
            x     = rng.uniform(0.0, Lx if Lx > 0 else 1.0)
            y     = rng.uniform(0.0, Ly if Ly > 0 else 1.0)
            z     = a + rng.exponential(z_mean)
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
# Experimental MSD data
# Thorneywork & Mackay — https://doi.org/10.1103/PhysRevX.14.041016
# Each table entry = 2 * MSD_parallel;  stored here as MSD = value / 2.
# =============================================================================
_EXP_PHI = np.array([0.02, 0.08, 0.16, 0.29, 0.34, 0.43, 0.52, 0.63, 0.66])

_EXP_T = np.array([
    0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 6, 7, 8, 9.5, 11, 12.5,
    14.5, 16.5, 19, 22, 25, 29, 33, 38, 44, 50.5, 58, 66.5, 76.5, 88,
    101.5, 116.5, 134, 154, 177, 203.5, 234, 269.5, 310, 356.5, 409.5,
    471, 542, 623, 716.5, 824, 947.5,
])

_EXP_MSD = np.array([
    [0.0868786736727541,  0.0811814360557076, 0.0732367588998956, 0.0664781607241859, 0.0649954491744,    0.0546196633020209, 0.053164209136343,  0.0403713289867451, 0.0370239972183891],
    [0.175738977998964,   0.164070232190311,  0.147889090159221,  0.133526865623559,  0.130455661981887,  0.108622841000876,  0.105122047458105,  0.0778352945992654, 0.07065183239013  ],
    [0.264292436232838,   0.246445961308918,  0.222336230436119,  0.199914854472008,  0.195108505536563,  0.161380258278236,  0.155402999254902,  0.112888782185394,  0.101484199584803 ],
    [0.352874345557503,   0.328482720272941,  0.296473524180094,  0.265820591473002,  0.259056986024403,  0.213259995516959,  0.204256458171688,  0.146085797956125,  0.130243321708071 ],
    [0.441098899606716,   0.410409954498356,  0.370374769041301,  0.331206629051439,  0.322440160876072,  0.264324763630819,  0.251921587152452,  0.177800185561488,  0.157333542171872 ],
    [0.52898057269913,    0.492184710568706,  0.444160946170739,  0.396265978931877,  0.385471475087028,  0.314783099531038,  0.298671081161577,  0.208311456015695,  0.183137369456852 ],
    [0.61655896225786,    0.573917834292706,  0.517819649702655,  0.461002977019876,  0.447896724402983,  0.364591161751855,  0.344538271368291,  0.237756360253775,  0.207779833943979 ],
    [0.703926721847952,   0.655506722782662,  0.591496974563293,  0.525510924711458,  0.509890348752772,  0.413811488633707,  0.38968786867399,   0.266274580966704,  0.231458611123113 ],
    [0.791891081313041,   0.737144870737402,  0.665038774053072,  0.589555098102336,  0.571435052169974,  0.462520102288019,  0.434081344775382,  0.294011098883758,  0.254266571620017 ],
    [0.967686315780234,   0.90013993269007,   0.811763861295088,  0.716767785528134,  0.69340679901767,   0.558481369047533,  0.521002865800685,  0.347390614500428,  0.297721285159871 ],
    [1.05555642328384,    0.981553653476312,  0.884926463131354,  0.77996562796108,   0.75380862293722,   0.605833887556175,  0.563620728905012,  0.373151765128246,  0.31846032928802  ],
    [1.2312117335818,     1.14478793535187,   1.03066365443738,   0.905610213715602,  0.873963038727266,  0.69941072403572,   0.647553900845309,  0.423228643415681,  0.358307565984299 ],
    [1.40745550638788,    1.30835420176995,   1.17624459631922,   1.03033149936619,   0.993173510410742,  0.791664759734791,  0.729823099173026,  0.471572348631057,  0.396259705707489 ],
    [1.67293910253892,    1.55314335375111,   1.39428668834006,   1.21604252282035,   1.17001396779874,   0.927682975454273,  0.850330280882776,  0.541079016284914,  0.450441359219353 ],
    [1.93776637434797,    1.79762934219217,   1.61137506523197,   1.4001725037146,    1.34453517687978,   1.06106928744683,   0.967502870407812,  0.60776655878543,   0.501722316853706 ],
    [2.20170378078571,    2.04126518089836,   1.82695980502487,   1.5824505793603,    1.51694392509941,   1.19224278008126,   1.08221155855274,   0.672090030524882,  0.550547972471751 ],
    [2.55289792079046,    2.36638805642949,   2.11287980623318,   1.82249846386644,   1.74379975542347,   1.36359270086091,   1.23205319059188,   0.754710874135537,  0.612529474686724 ],
    [2.90414931671642,    2.69177346610138,   2.39707344730973,   2.05969485608729,   1.96847072253944,   1.53194252351703,   1.37868526790385,   0.834307370300539,  0.671885446421092 ],
    [3.34313093329382,    3.09782012508071,   2.75007456141895,   2.35283813805334,   2.24550269857062,   1.73897131882663,   1.55804855506251,   0.930206596019685,  0.74303595910093  ],
    [3.86793714948975,    3.58434945810827,   3.16978242927572,   2.70052940343666,   2.57337912962869,   1.98214823815577,   1.76857410864693,   1.04129122999038,   0.824807065687041 ],
    [4.39078688021191,    4.0713249999406,    3.5863604112121,    3.04457548182523,   2.89695448190292,   2.22109572583651,   1.97471549111182,   1.14900927486991,   0.903742288596519 ],
    [5.09111040657461,    4.71731488245305,   4.13662499945957,   3.49907189017553,   3.3230627722167,    2.53325367486434,   2.24443912803959,   1.2890915245166,    1.00556972395272  ],
    [5.78515542458162,    5.35843916487378,   4.68247396695169,   3.94965506868391,   3.74530635119983,   2.83884636971126,   2.510060075818,     1.4256518850172,    1.10454522701066  ],
    [6.65036003288592,    6.15840005320517,   5.35835867243159,   4.50533786100141,   4.26674057624375,   3.21492753862889,   2.83569146499476,   1.59269147670772,   1.22459421832395  ],
    [7.70780525849094,    7.11369413684059,   6.16576942691208,   5.16188856803568,   4.88415442748577,   3.65811310025135,   3.22148387705622,   1.78865982659854,   1.36526330696697  ],
    [8.86644495749324,    8.14551701162966,   7.03831880276658,   5.8654675426017,    5.54396019590245,   4.1260209599968,    3.63521192237035,   1.99590683565611,   1.51433451952258  ],
    [10.2034671627172,    9.3350661717212,    8.04193919852012,   6.66341997415905,   6.29717751313796,   4.65630451246583,   4.10829220608304,   2.23053144409677,   1.68382572219583  ],
    [11.720073226655,     10.6795545846139,   9.17328345878865,   7.55296658401439,   7.14401989023319,   5.25096833107197,   4.63979324004518,   2.49336436975303,   1.87422882397029  ],
    [13.4844330314038,    12.2362125617288,   10.492206874079,    8.58568878209489,   8.13060745883306,   5.94192350906068,   5.26413513898123,   2.80018194633152,   2.09386412360428  ],
    [15.5138012191151,    13.9946812480499,   11.9939211863456,   9.75305244150671,   9.25526513009904,   6.7243367687033,    5.97524590588017,   3.14627717001996,   2.34325926880463  ],
    [17.871968791727,     16.0585891922564,   13.755493574536,    11.1030627864185,   10.5624698306932,   7.62700139518517,   6.80386997724844,   3.54417287298521,   2.6355056068531   ],
    [20.4392185829286,    18.3665458966568,   15.7388930006376,   12.5894113104433,   11.9960234216531,   8.62009010733314,   7.72099147882357,   3.98143458319333,   2.96203286017631  ],
    [23.3556460286582,    21.0559161504663,   18.0429311252854,   14.3065900139866,   13.6491959814441,   9.76777820939959,   8.77802943702702,   4.48472122392531,   3.34303596860473  ],
    [26.6106124369596,    24.1120933433756,   20.671114762213,    16.2433176630194,   15.5175423588604,   11.0640660977708,   9.98648322740558,   5.05942008572231,   3.77768191195541  ],
    [30.3696085370867,    27.628120626449,    23.6900529045949,   18.4705957671652,   17.6505217941523,   12.5418776677611,   11.3716093181446,   5.7291981964347,    4.28306888004766  ],
    [34.7510331447315,    31.5951502252214,   27.1869183696485,   20.9989676152347,   20.0757140924404,   14.2356024182825,   12.9612353799771,   6.50300007610229,   4.8678584631572   ],
    [39.7421078903801,    36.0820844646022,   31.2292153478394,   23.9009834739343,   22.8499453203894,   16.1630639205054,   14.8016756170298,   7.38159542212014,   5.54261200388029  ],
    [45.580967057262,     41.3125680429184,   35.8670850021179,   27.274532623811,    26.0554590683251,   18.3845715263776,   16.9490796899905,   8.392903644939,     6.3328437198428   ],
    [52.357864850444,     47.4417611444292,   41.0591086638179,   31.0156702730393,   29.6708077982658,   20.897730377471,    19.3963461429172,   9.56070270415908,   7.23552732392666  ],
    [59.8627517075298,    54.4194148804208,   46.8684356808982,   35.3050567740477,   33.8581390356538,   23.7866750939495,   22.1448820302229,   10.9110911944064,   8.27686048254701  ],
    [68.3692095277022,    62.3365548866045,   53.3206012297264,   40.1717735257059,   38.6008020769934,   27.0721828322376,   25.2436675473375,   12.4607830161717,   9.48286040334868  ],
    [78.0393322134494,    71.5369760466356,   60.7815295162781,   45.762523039144,    44.059231873321,    30.8118550393562,   28.8636280236722,   14.2170389627483,   10.8736327294343  ],
    [89.5215513263167,    81.6742424518502,   69.4030257491507,   52.1944694428289,   50.2144458427821,   35.0829569144851,   32.9822815499314,   16.2413086598507,   12.4710517736035  ],
    [103.376577495411,    93.2726779235743,   79.5028257676777,   59.5994833348946,   57.2342189622603,   39.8674319535071,   37.5707556277038,   18.5479620319854,   14.2620168550702  ],
    [120.028109938036,    107.102738243904,   91.4239433618642,   67.6708478316377,   65.8765927470285,   45.4368901744459,   42.9684355617069,   21.3754526189108,   16.3701209722769  ],
    [138.595585772359,    123.829174820632,   105.245075050535,   76.7920769082152,   76.2747017581318,   51.7967440617664,   49.2834238958463,   24.6770910708216,   18.9217704074296  ],
    [156.835151682324,    142.777977734586,   120.615327426969,   86.5121870442846,   88.3336504642584,   59.1044816827387,   56.6153116064194,   28.3242705843719,   21.8828010603831  ],
]) / 2.0   # table stores 2*MSD; divide by 2 to get MSD


def _get_exp_msd(phi_sim, tol=0.005):
    """
    Return (t, msd) for the experimental phi closest to phi_sim,
    or (None, None) if none is within tol.
    """
    diffs = np.abs(_EXP_PHI - phi_sim)
    idx   = np.argmin(diffs)
    if diffs[idx] > tol:
        return None, None
    return _EXP_T, _EXP_MSD[:, idx]


# =============================================================================
if __name__ == '__main__':
    main()
