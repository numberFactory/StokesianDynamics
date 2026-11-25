"""
test_fluctuating_dynamics_gpu.py
--------------------------------
GPU version of test_fluctuating_dynamics.py: the same self-contained
fluctuating Stokesian Dynamics test for N point particles above a single wall
with periodic boundaries in x and y, run with cupyStokesianDynamics.

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

main() follows test_fluctuating_dynamics.main(), and everything between
main() and the GPU section (MSD, forces, placement, experimental data) is
copied verbatim from test_fluctuating_dynamics.py.  The GPU section at the
bottom adds, behind command-line flags:

  --forces     gpu (default): the forces as a CUDA kernel with the same
               expressions as the numba calculator; cpu: the numba calculator
  --check      component checks of the GPU solver against the CPU reference
               (lubrication matrices, Brownian factor, preconditioner, one
               solve with each GMRES back end, one deterministic step,
               orientation update, periodic wrap, GPU forces vs numba)
  --profile    per-method times, call counts and device allocations of the
               GPU and CPU solvers (--solvers gpu,cpu)
  --factor-bench  where the factorisation time goes: cuDSS plan / factorize /
               solve for each reordering algorithm and thread count, one plan
               for both matrices, the noise from a backward-only cuDSS solve,
               CHOLMOD analyze / factorize for each ordering
  --no-run     skip the diffusion run
  --reference  compare the final MSD with a CPU run of test_fluctuating_dynamics.py

  python test_fluctuating_dynamics_gpu.py                                 # diffusion run
  python test_fluctuating_dynamics_gpu.py --check --profile --no-run
  python test_fluctuating_dynamics_gpu.py --reference msd_phi0.66_L160x160.npz
  python test_fluctuating_dynamics_gpu.py --L 60 --t-max 50               # quick run
  python test_fluctuating_dynamics_gpu.py --forces cpu                    # numba forces
  python test_fluctuating_dynamics_gpu.py --profile --no-run --solvers gpu --gmres-backend scipy_gpu
  nsys profile -t cuda,nvtx --stats=true \\
      python test_fluctuating_dynamics_gpu.py --profile --no-run --solvers gpu
"""

import sys
import os
# One BLAS / OpenMP thread (before numpy is imported): scipy's GMRES loop on the
# host makes many short BLAS calls on 6N vectors, and threading them over all
# cores made each solve ~250 ms slower on a 32-core machine.  An OMP_NUM_THREADS
# given on the command line is kept.
os.environ.setdefault('OMP_NUM_THREADS', '1')
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
from cupyStokesianDynamics import cupyStokesianDynamics


# =============================================================================
def main():
# =============================================================================
    args = parse_args()

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
    Lx, Ly       = args.L, args.L              # 160 x 160 by default
    L            = np.array([Lx, Ly, 0.0])
    z_max_solver = 2.0 * (2.0 * a)

    # ── Packing fraction ──────────────────────────────────────────────────────
    phi = 0.66
    N   = max(1, int(phi * Lx * Ly / (np.pi * a**2)))

    # ── Simulation parameters ─────────────────────────────────────────────────
    dt      = 0.25 #use dt = 0.05 to get a more accurate MSD at short times (which will match those reported in https://doi.org/10.1103/PhysRevX.14.041016)
    t_max   = args.t_max                       # 1.1e3 by default
    n_steps = int(t_max / dt)
    n_save  = 1          # save radial coordinate every n_save steps
    solver_tolerance = 5e-3

    # ── GPU options: solver (defaults: scipy GMRES, cluster PC) and forces ───
    solver_options = dict(gmres_backend=args.gmres_backend,
                          gmres_restart=args.gmres_restart,
                          pc_backend=args.pc_backend,
                          lub_skin=args.skin,
                          sync_timings=args.sync_timings,
                          pc_host_threads=args.pc_host_threads,
                          pc_replan_in_place=args.pc_replan_in_place,
                          pc_gap=args.pc_gap, pc_max_cluster=args.pc_max_cluster,
                          noise_backend=args.noise_backend)
    force_calc = (gpu_force_torque_calculator if args.forces == 'gpu'
                  else on_host(force_torque_calculator))

    # ── Neighbour list parameters ─────────────────────────────────────────────
    pair_cutoff = 2.0 * (2.0 * a)
    buffer_skin = pair_cutoff
    nl_cutoff   = pair_cutoff + buffer_skin
    rebuild_tol = buffer_skin / 2.0

    # ── Output ────────────────────────────────────────────────────────────────
    out_dir    = os.path.dirname(os.path.abspath(__file__))
    out_prefix = os.path.join(out_dir, f'msd_phi{phi:.2f}_L{Lx:.0f}x{Ly:.0f}_gpu')

    print(f"N = {N} particles, phi = {phi:.3f}, L = ({Lx},{Ly}), "
          f"dt = {dt}, n_steps = {n_steps}")

    # ── Initialise particles ──────────────────────────────────────────────────
    positions    = place_particles(N, a, kT, g, Lx, Ly, seed=42)
    print(f"Placed {len(positions)}/{N} particles.")
    identity_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    bodies       = [Body(location=np.array(p), orientation=identity_rot)
                    for p in positions]

    # ── Optional checks and profiling (GPU section at the bottom) ─────────────
    setup = dict(a=a, eta=eta, kT=kT, g=g, firm_delta=firm_delta,
                 debye_firm=debye_firm, repulsion_firm=repulsion_firm,
                 repulsion_soft=repulsion_soft, debye_soft=debye_soft,
                 L=L, z_max=z_max_solver, phi=phi, N=N, dt=dt,
                 tolerance=solver_tolerance, pair_cutoff=pair_cutoff,
                 nl_cutoff=nl_cutoff, rebuild_tol=rebuild_tol,
                 positions=positions, solver_options=solver_options,
                 forces=args.forces)
    status = 0
    if args.check:
        status = run_checks(setup, args)
    if args.profile:
        run_profile(setup, args, out_prefix)
    if args.factor_bench:
        run_factor_bench(setup, args)
    if args.no_run:
        return status

    # ── Initialise solver ─────────────────────────────────────────────────────
    solver = cupyStokesianDynamics(
        bodies=bodies, a=a, eta=eta,
        periodic_length=L, z_max=z_max_solver,
        debye_length=firm_delta, **solver_options,
    )
    solver.kT                  = kT
    solver.dt                  = dt
    solver.tolerance           = solver_tolerance
    solver.num_rejections_wall = 0
    solver.num_rejections_jump = 0
    solver.Set_R_Mats()

    # ── Initial neighbour list and force calculator ───────────────────────────
    r_now     = to_numpy(solver.positions)       # positions live on the GPU
    r_wrapped = wrap_positions(r_now, L)
    nl_nbrs, nl_offsets = build_neighbour_list(r_wrapped, L, nl_cutoff)
    r_nl_ref  = r_now.copy()

    FT_calc = partial(force_calc,
                      a=a, g=g, L=L, pair_cutoff=pair_cutoff,
                      rep_firm=repulsion_firm, deb_firm=debye_firm,
                      firm_delta=firm_delta,
                      rep_soft=repulsion_soft, deb_soft=debye_soft,
                      neighbors=nl_nbrs, offsets=nl_offsets)

    # ── Trajectory storage ────────────────────────────────────────────────────
    n_frames   = n_steps // n_save + 1
    traj_r     = np.zeros((N, n_frames))
    frame_idx  = 0
    r_init     = to_numpy(solver.positions)
    traj_r[:, 0] = np.sqrt(r_init[:, 0]**2 + r_init[:, 1]**2)

    # ── Main loop ─────────────────────────────────────────────────────────────
    nl_rebuilds = 0
    t_wall      = time.perf_counter()

    for step in range(1, n_steps + 1):
        r_now     = to_numpy(solver.positions)
        r_wrapped = wrap_positions(r_now, L)

        if needs_rebuild(r_now, r_nl_ref, L, rebuild_tol):
            nl_nbrs, nl_offsets = build_neighbour_list(r_wrapped, L, nl_cutoff)
            r_nl_ref = r_now.copy()
            FT_calc  = partial(force_calc,
                               a=a, g=g, L=L, pair_cutoff=pair_cutoff,
                               rep_firm=repulsion_firm, deb_firm=debye_firm,
                               firm_delta=firm_delta,
                               rep_soft=repulsion_soft, deb_soft=debye_soft,
                               neighbors=nl_nbrs, offsets=nl_offsets)
            nl_rebuilds += 1

        if args.debug_every and step % args.debug_every == 0:
            np.savez('debug_positions.npz', step=step, time=step*dt, positions=r_now)
        solver.Update_Bodies_Trap(FT_calc, stochastic=True, print_residual=False)
        #solver.print_timings()

        if step % n_save == 0:
            frame_idx += 1
            r_cur = to_numpy(solver.positions)
            traj_r[:, frame_idx] = np.sqrt(r_cur[:, 0]**2 + r_cur[:, 1]**2)

            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t_wall
                print(f"  frame {frame_idx:5d}/{n_frames-1}  t={step*dt:.0f}  "
                      f"NL rebuilds={nl_rebuilds}  "
                      f"rej_wall={solver.num_rejections_wall}  "
                      f"rej_jump={solver.num_rejections_jump}  "
                      f"elapsed={elapsed:.0f}s  {gpu_progress(solver, elapsed, step)}")

                traj_so_far   = traj_r[:, :frame_idx + 1]
                all_msds_now  = msd_matrix(traj_so_far)
                msd_mean_now  = all_msds_now.mean(axis=0)
                msd_err_now   = 2.0 * all_msds_now.std(axis=0) / np.sqrt(N)
                lag_times_now = np.arange(frame_idx + 1) * (n_save * dt)
                plot_and_save_msd(msd_mean_now, msd_err_now, lag_times_now,
                                  out_prefix, phi, N, kT, eta, a,
                                  tag=f'frame{frame_idx:06d}', t_sim=step * dt)

    t_run = time.perf_counter() - t_wall
    print(f"\nSimulation complete in {t_run:.1f}s  "
          f"NL rebuilds={nl_rebuilds}")
    solver.print_timings()

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
                      out_prefix + '_final', phi, N, kT, eta, a, tag='final')
    print(f"Final MSD saved to {out_prefix}_final.{{npz,png}}")

    # ── Comparison with experiment / a CPU run (GPU section at the bottom) ────
    report_run(solver, msd_mean, msd_err, lag_times, D_meas, phi, out_prefix,
               args, t_run, n_steps)
    return status


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
    for i in prange(N):
        f[i, 2] -= g
        h        = r[i, 2]
        contact  = a * (1.0 - firm_delta)
        if h > contact:
            f[i, 2] += (rep_firm / deb_firm) * np.exp(-(h - contact) / deb_firm)
        else:
            f[i, 2] += rep_firm / deb_firm
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


# ##############################################################################
# ##############################################################################
#
#    GPU test section: command line, GPU force calculator, host/device
#    adapters, comparison with experiment and with a CPU run, component
#    checks, profiling
#
# ##############################################################################
# ##############################################################################
import argparse
import faulthandler
import gc
import json
import traceback
from collections import defaultdict
import scipy.sparse as sp

faulthandler.enable()     # a crash inside a native library prints the Python stack


def parse_args():
    ap = argparse.ArgumentParser(description='GPU fluctuating-dynamics test',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--check', action='store_true', help='run the component checks first')
    ap.add_argument('--profile', action='store_true', help='profile the solvers first')
    ap.add_argument('--factor-bench', action='store_true',
                    help='benchmark the factorisation options (cuDSS, CHOLMOD) first')
    ap.add_argument('--no-run', action='store_true', help='skip the diffusion run')
    ap.add_argument('--solvers', default='gpu,cpu',
                    help='solvers for --check/--profile: gpu (cupyStokesianDynamics), '
                         'cpu (pyStokesianDynamics)')
    ap.add_argument('--forces', default='gpu', choices=['gpu', 'cpu'],
                    help='gpu: CUDA kernel with the expressions of _wall_force_numba / '
                         '_pair_force_numba (check C8); cpu: the numba calculator on the host')
    ap.add_argument('--L', type=float, default=160.0, help='box side Lx = Ly')
    ap.add_argument('--t-max', type=float, default=1.1e3)
    ap.add_argument('--gmres-backend', default='scipy', choices=['scipy', 'scipy_gpu', 'cupy'],
                    help='scipy: scipy on the host, operators on the GPU (as the CPU solver); '
                         'scipy_gpu: the same algorithm on device arrays; cupy: cupyx gmres')
    ap.add_argument('--gmres-restart', type=int, default=None,
                    help='default 300 for scipy / scipy_gpu, 10 for cupy')
    ap.add_argument('--pc-backend', default='auto',
                    choices=['auto', 'nvmath', 'cholesky_solver', 'cholmod_host', 'clusters'],
                    help='auto = clusters: near-contact clusters inverted as small dense '
                         'blocks (no cuDSS); nvmath / cholesky_solver: cuDSS; cholmod_host')
    ap.add_argument('--pc-gap', type=float, default=0.1,
                    help='clusters: pairs closer than this (units of a) form clusters; 0 = block Jacobi')
    ap.add_argument('--pc-max-cluster', type=int, default=64,
                    help='clusters: a larger cluster gets a smaller gap (there only), then is '
                         'cut into chunks of at most this many particles')
    ap.add_argument('--noise-backend', default='auto',
                    choices=['auto', 'pairwise', 'cudss', 'cholmod'],
                    help='Brownian increments: pairwise (default, auto): pair by pair on the '
                         'GPU, no factorisation; cudss (implies --pc-backend nvmath) / cholmod: '
                         'Cholesky factor of Delta_R')
    ap.add_argument('--pc-replan-in-place', action='store_true',
                    help='nvmath: plan a new pattern on the same DirectSolver')
    ap.add_argument('--pc-host-threads', type=int, default=None,
                    help="host threads for cuDSS's reordering with --pc-backend nvmath")
    ap.add_argument('--skin', type=float, default=0.0,
                    help='lubrication neighbour-list skin, units of a')
    ap.add_argument('--sync-timings', action='store_true',
                    help='synchronise the GPU around the solver timings (print_timings)')
    ap.add_argument('--profile-steps', type=int, default=10)
    ap.add_argument('--warmup-steps', type=int, default=2)
    ap.add_argument('--check-tol', type=float, default=None,
                    help='GMRES tolerance of the solve and step checks (default: 100x the '
                         'measured precision of the mobility product, at least 1e-8)')
    ap.add_argument('--debug-every', type=int, default=0,
                    help='write debug_positions.npz every k steps (the CPU test: every step)')
    ap.add_argument('--reference', default=None,
                    help='MSD npz written by test_fluctuating_dynamics.py to compare with')
    args = ap.parse_args()
    args.solvers = [s.strip() for s in args.solvers.split(',') if s.strip()]
    unknown = [s for s in args.solvers if s not in SOLVER_NAMES]
    if unknown:
        ap.error(f"--solvers: unknown {', '.join(unknown)} (choose from "
                 f"{', '.join(SOLVER_NAMES)})")
    return args


# =============================================================================
# GPU force calculator
# =============================================================================
FORCE_KERNEL = r"""
// Wall + pair forces of test_fluctuating_dynamics.py (gravity, firm exponential
// wall repulsion, firm + soft exponential pair repulsion) on the GPU.
// Same expressions as _wall_force_numba / _pair_force_numba.  One thread per
// particle; neighbours from the buffered CSR list (offsets/neighbors, int64,
// the list contains i itself).  Output FT is the 6N vector [F_1 T_1 F_2 T_2 ...].
extern "C" __global__ void ft_wall_pair(
    const int N, const double* __restrict__ r,
    const long long* __restrict__ offsets, const long long* __restrict__ nbrs,
    const double Lx, const double Ly, const double Lz,
    const double a, const double g,
    const double rep_firm, const double deb_firm, const double firm_delta,
    const double rep_soft, const double deb_soft, const double pair_cutoff,
    double* __restrict__ FT)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  const double L[3] = {Lx, Ly, Lz};
  // wall (gravity + firm repulsion)
  double fw = 0.0;
  fw -= g;
  const double h = r[3*i + 2];
  const double contact = a * (1.0 - firm_delta);
  if (h > contact) fw += (rep_firm / deb_firm) * exp(-(h - contact) / deb_firm);
  else             fw += rep_firm / deb_firm;
  // pairs
  double f[3] = {0.0, 0.0, 0.0};
  const double offset_firm = 2.0 * a * (1.0 - firm_delta);
  const double offset_soft = 2.0 * a;
  for (long long kk = offsets[i]; kk < offsets[i + 1]; ++kk) {
    const long long j = nbrs[kk];
    if (j == i) continue;
    double dr[3];
    for (int d = 0; d < 3; ++d) {
      dr[d] = r[3*j + d] - r[3*i + d];
      if (L[d] > 0) {
        const double s = (double)((dr[d] > 0) - (dr[d] < 0));
        dr[d] -= (double)((long long)(dr[d] / L[d] + 0.5 * s)) * L[d];
      }
    }
    const double rn = sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
    if (rn > pair_cutoff) continue;
    const double rn_safe = fmax(rn, 1e-12);
    double fmag;
    if (rn > offset_firm) fmag = -(rep_firm / deb_firm) * exp(-(rn - offset_firm) / deb_firm) / rn_safe;
    else                  fmag = -(rep_firm / deb_firm) / rn_safe;
    for (int d = 0; d < 3; ++d) f[d] += fmag * dr[d];
    if (rep_soft > 0.0) {
      double smag;
      if (rn > offset_soft) smag = -(rep_soft / deb_soft) * exp(-(rn - offset_soft) / deb_soft) / rn_safe;
      else                  smag = -(rep_soft / deb_soft) / rn_safe;
      for (int d = 0; d < 3; ++d) f[d] += smag * dr[d];
    }
  }
  FT[6*i + 0] = f[0];
  FT[6*i + 1] = f[1];
  FT[6*i + 2] = fw + f[2];
  FT[6*i + 3] = 0.0;
  FT[6*i + 4] = 0.0;
  FT[6*i + 5] = 0.0;
}
"""


class _ForceKernelState(object):
    '''The compiled kernel and a device copy of the current neighbour list.'''
    kernel = None
    nl_host = None
    nl_dev = None


def gpu_force_torque_calculator(bodies, r_vecs, a, g, L, pair_cutoff,
                                rep_firm, deb_firm, firm_delta,
                                rep_soft, deb_soft, neighbors, offsets):
    '''GPU version of force_torque_calculator, same arguments.  r_vecs: (N, 3)
    device positions; returns the (6N,) device vector [F_1 T_1 F_2 T_2 ...].
    The neighbour list from build_neighbour_list is uploaded once per rebuild.'''
    import cupy
    st = _ForceKernelState
    if st.kernel is None:
        st.kernel = cupy.RawKernel(FORCE_KERNEL, 'ft_wall_pair')
    if st.nl_host is None or st.nl_host[0] is not neighbors or st.nl_host[1] is not offsets:
        st.nl_dev = (cupy.asarray(neighbors, dtype=cupy.int64),
                     cupy.asarray(offsets, dtype=cupy.int64))
        st.nl_host = (neighbors, offsets)            # held, so the identity test is safe
    nbrs_d, offs_d = st.nl_dev
    r = cupy.ascontiguousarray(cupy.asarray(r_vecs, dtype=cupy.float64).reshape(-1, 3))
    N = r.shape[0]
    FT = cupy.empty(6 * N, dtype=cupy.float64)
    f8 = np.float64
    st.kernel(((N + 127) // 128,), (128,),
              (np.int32(N), r, offs_d, nbrs_d, f8(L[0]), f8(L[1]), f8(L[2]), f8(a), f8(g),
               f8(rep_firm), f8(deb_firm), f8(firm_delta), f8(rep_soft), f8(deb_soft),
               f8(pair_cutoff), FT))
    return FT


# =============================================================================
# Host / device adapters
# =============================================================================
def to_numpy(x):
    '''device or host array -> numpy'''
    if isinstance(x, np.ndarray):
        return x
    import cupy
    return cupy.asnumpy(x)


def on_host(calc):
    '''force_torque_calculator for device positions: the numba calculator runs
    on the host (one transfer each way per call).'''
    def host_calc(bodies, r_vecs, *args, **kw):
        return calc(bodies, to_numpy(r_vecs), *args, **kw)
    return host_calc


def gpu_sync():
    try:
        import cupy
        cupy.cuda.get_current_stream().synchronize()
    except Exception:
        pass


def free_gpu_memory():
    gc.collect()
    try:
        import cupy
        gpu_sync()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def gpu_progress(solver, elapsed, step):
    mv = [h['matvecs'] for h in solver.gmres_history[-100:]]
    return (f"({1e3 * elapsed / step:.0f} ms/step, "
            f"{np.mean(mv) if mv else 0.0:.1f} GMRES applications/solve)")


# =============================================================================
# After the run: comparison with experiment and with a CPU run
# =============================================================================
def report_run(solver, msd_mean, msd_err, lag_times, D_meas, phi, out_prefix,
               args, seconds, n_steps):
    mv = [h['matvecs'] for h in solver.gmres_history]
    summary = {'D_meas': float(D_meas), 'seconds': seconds,
               'ms_per_step': 1e3 * seconds / max(n_steps, 1),
               'rejections': [int(solver.num_rejections_wall), int(solver.num_rejections_jump)],
               'gmres_backend': solver.gmres_backend, 'gmres_restart': solver.gmres_restart,
               'gmres_applications_per_solve': float(np.mean(mv)) if mv else None,
               'pc_backend': solver.PC.name, 'noise': _noise_label(solver),
               'lub_skin': solver.lub_skin,
               'forces': args.forces}
    summary['experiment'] = compare_with_experiment(msd_mean, lag_times, phi)
    if args.reference:
        summary['reference'] = compare_with_reference(msd_mean, msd_err, lag_times, D_meas,
                                                      args.reference)
    with open(out_prefix + '_run.json', 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"run summary written to {out_prefix}_run.json")
    return summary


def compare_with_experiment(msd_mean, lag_times, phi):
    t_exp, msd_exp = _get_exp_msd(phi)
    if t_exp is None:
        print('no experimental data for this packing fraction')
        return None
    m = (lag_times > 0) & np.isfinite(msd_mean) & (msd_mean > 0)
    t, y = lag_times[m], msd_mean[m]
    sel = (t_exp >= t.min()) & (t_exp <= 0.1 * t.max())
    if not sel.any():
        return None
    ratio = np.exp(np.interp(np.log(t_exp[sel]), np.log(t), np.log(y))) / msd_exp[sel]
    rms = float(np.sqrt(np.mean(np.log(ratio) ** 2)))
    print(f"MSD sim/exp for t in [{t_exp[sel].min():g}, {t_exp[sel].max():g}] (lags <= 10% "
          f"of the run): mean {ratio.mean():.3f}, rms log deviation {rms:.3f}")
    for tt in (0.5, 1, 5, 10, 50, 100, 500):
        j = np.flatnonzero(np.isclose(t_exp[sel], tt))
        if j.size:
            print(f"   t = {tt:6g}: sim/exp = {ratio[j[0]]:.3f}")
    return {'t': t_exp[sel].tolist(), 'ratio': ratio.tolist(), 'rms_log_dev': rms}


def compare_with_reference(msd_mean, msd_err, lag_times, D_meas, ref_path):
    '''Reference = npz written by test_fluctuating_dynamics.py (a CPU run),
    interpolated (log-log) at 25 lags up to 10% of the shorter run, so the
    runs may differ in dt and length.'''
    ref = np.load(ref_path)
    t_r, m_r, e_r = ref['lag_times'], ref['msd_mean'], ref['msd_err']
    keep = (t_r > 0) & (m_r > 0)
    t_r, m_r, e_r = t_r[keep], m_r[keep], e_r[keep]
    m = (lag_times > 0) & (msd_mean > 0)
    t, y, e = lag_times[m], msd_mean[m], msd_err[m]
    t_lo, t_hi = max(t_r.min(), t.min()), 0.1 * min(t_r.max(), t.max())
    if t_hi <= t_lo:
        print('reference and run do not overlap in lag time')
        return None
    lags = np.geomspace(t_lo, t_hi, 25)
    loglog = lambda tt, x, v: np.exp(np.interp(np.log(tt), np.log(x), np.log(v)))
    y_new, y_ref = loglog(lags, t, y), loglog(lags, t_r, m_r)
    sig = np.hypot(np.interp(lags, t, e), np.interp(lags, t_r, e_r)) / 2.0   # err = 2 sigma/sqrt(N)
    z = (y_new - y_ref) / np.where(sig > 0, sig, np.inf)
    rel = y_new / y_ref - 1.0
    D_ref = float(ref['D_meas']) if 'D_meas' in ref else float('nan')
    frac = float(np.mean(np.abs(z) <= 3.0))
    ok = frac >= 0.9 and abs(D_meas / D_ref - 1.0) < 0.05
    print(f"vs reference {ref_path}: |z| <= 3 at {100*frac:.0f}% of {lags.size} lags in "
          f"[{t_lo:g}, {t_hi:g}], max |rel diff| {np.abs(rel).max():.3f}, "
          f"D_meas {D_meas:.4e} vs {D_ref:.4e} ({100*(D_meas/D_ref - 1):+.1f}%)")
    print('REFERENCE COMPARISON:', 'PASS' if ok else 'CHECK',
          '(|z| <= 3 at >= 90% of the lags and D within 5%; the lags are correlated, '
          'so treat it as a guide)')
    return {'lags': lags.tolist(), 'z': z.tolist(), 'rel': rel.tolist(), 'D_ref': D_ref,
            'pass': bool(ok)}


# =============================================================================
# Solvers for the checks and the profile
# =============================================================================
SOLVER_NAMES = {'gpu': 'cupyStokesianDynamics', 
                'cpu': 'pyStokesianDynamics'}


def make_solver(kind, setup, tolerance=None, **overrides):
    '''Solver of the given kind at setup['positions'] (fresh Body objects).'''
    identity_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    bodies = [Body(location=np.array(p), orientation=identity_rot) for p in setup['positions']]
    common = dict(bodies=bodies, a=setup['a'], eta=setup['eta'], periodic_length=setup['L'],
                  z_max=setup['z_max'], debye_length=setup['firm_delta'])
    if kind == 'gpu':
        s = cupyStokesianDynamics(**common, **{**setup['solver_options'], **overrides})
    elif kind == 'cpu':
        from pyStokesianDynamics import pyStokesianDynamics
        s = pyStokesianDynamics(**common)
    else:
        raise ValueError(kind)
    s.kT = setup['kT']
    s.dt = setup['dt']
    s.tolerance = setup['tolerance'] if tolerance is None else tolerance
    s.num_rejections_wall = 0
    s.num_rejections_jump = 0
    s.Set_R_Mats()
    return s, bodies


def make_FT_calc(setup, r_unwrapped, calc=None):
    '''calc (default force_torque_calculator) with the buffered neighbour list built at r.'''
    L = setup['L']
    nbrs, offs = build_neighbour_list(wrap_positions(r_unwrapped, L), L, setup['nl_cutoff'])
    return partial(calc or force_torque_calculator, a=setup['a'], g=setup['g'], L=L,
                   pair_cutoff=setup['pair_cutoff'], rep_firm=setup['repulsion_firm'],
                   deb_firm=setup['debye_firm'], firm_delta=setup['firm_delta'],
                   rep_soft=setup['repulsion_soft'], deb_soft=setup['debye_soft'],
                   neighbors=nbrs, offsets=offs)


def force_calculator(kind, forces):
    '''The force calculator for each solver's FT_calc convention.'''
    if kind == 'cpu':
        return force_torque_calculator                   # numpy in and out
    if forces == 'gpu':
        return gpu_force_torque_calculator               # device in and out
    return on_host(force_torque_calculator)              # device in, numpy out


def solver_positions(kind, s, bodies):
    return to_numpy(s.positions) if kind == 'gpu' else np.array([b.location for b in bodies])


# =============================================================================
# Component checks (--check)
# =============================================================================
class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=''):
        self.rows.append((name, status, detail))
        print(f"  [{status:4s}] {name}: {detail}", flush=True)

    def value(self, name, ok, detail):
        self.add(name, 'PASS' if ok else 'FAIL', detail)


def run_checks(setup, args):
    '''Runs C1-C8; returns 1 if any check failed, else 0.'''
    print('\n=== check ===', flush=True)
    rep = Report()
    for name, check in [('C1 lubrication matrices', check_lubrication),
                        ('C2/C3 Brownian factor and preconditioner', check_factors),
                        ('C4/C5 linear solve and step', check_solve_and_step),
                        ('C6/C7 orientation update and wrap', check_orientation_and_wrap),
                        ('C8 GPU forces', check_forces)]:
        try:
            check(rep, setup, args)
        except Exception as e:
            rep.add(name, 'FAIL', repr(e))
            traceback.print_exc()
        free_gpu_memory()
    counts = defaultdict(int)
    for row in rep.rows:
        counts[row[1]] += 1
    print('check summary:', dict(counts), flush=True)
    return int(counts['FAIL'] > 0)


def _host_csc(M):
    '''device CSR of a symmetric matrix -> scipy CSC with the same arrays'''
    return sp.csc_matrix((to_numpy(M.data), to_numpy(M.indices), to_numpy(M.indptr)),
                         shape=M.shape)


def _cpp_neighbours(r, L, cut):
    tree = spatial.cKDTree(r, boxsize=L if np.any(L > 0) else None,
                           balanced_tree=False, compact_nodes=False)
    return [np.array([i for i in tree.query_ball_point(r[j], r=cut) if i > j], dtype=np.int32)
            for j in range(len(r))]


def _timed(f, *args, **kw):
    gpu_sync()
    t0 = time.perf_counter()
    out = f(*args, **kw)
    gpu_sync()
    return out, time.perf_counter() - t0


def check_lubrication(rep, setup, args):
    '''C1: fused GPU assembly vs lubrication.cpp.'''
    import cupy
    from lubrication_cupy import LubricationAssembler, BlockPattern
    a, eta, debye = setup['a'], setup['eta'], setup['firm_delta']
    try:
        from StokesianDynamics import Lubrication as CppLub
    except Exception as e:
        CppLub = None
        rep.add('C1 lubrication vs C++', 'SKIP', f'StokesianDynamics not importable ({e})')
    syn, L_syn = synthetic_pairs(a)
    for tag, r, L in [('initial configuration', wrap_positions(setup['positions'], setup['L']),
                       setup['L']),
                      ('synthetic, all fit branches', wrap_positions(syn, L_syn), L_syn)]:
        pairs = spatial.cKDTree(r, boxsize=L).query_pairs(4.5 * a, output_type='ndarray')
        asm = LubricationAssembler(debye, a, eta)
        asm.set_pattern(BlockPattern(len(r), pairs))
        r_d = cupy.asarray(r)
        asm.assemble(r_d, L)                               # first call compiles the kernels
        _, t_gpu = _timed(asm.assemble, r_d, L)
        gpu_mb, gpu_sup = _host_csc(asm.R_MB), _host_csc(asm.R_SupS)
        rep.value(f'C1 exact symmetry ({tag})', abs(gpu_sup - gpu_sup.T).max() == 0.0, '')
        timing = f'fused GPU assembly {1e3*t_gpu:.2f} ms'
        if CppLub is not None:
            nl = _cpp_neighbours(r, L, 4.5 * a)
            (ref_mb, ref_sup), t_cpp = _timed(CppLub(debye).ResistCSC_both,
                                              [x for x in r], nl, a, eta, L)
            for name, A, B in [('R_MB', gpu_mb, ref_mb), ('R_Sup', gpu_sup, ref_sup),
                               ('Delta_R', gpu_sup - gpu_mb, ref_sup - ref_mb)]:
                D = (A - B).tocoo()
                err = abs(D.data).max() / abs(B).max() if D.nnz else 0.0
                rep.value(f'C1 {name} vs lubrication.cpp ({tag})', err < 1e-12,
                          f'max|diff|/max|A| = {err:.2e}')
            timing += f', C++ ResistCSC_both {1e3*t_cpp:.1f} ms'
        rep.add(f'C1 timing ({tag})', 'INFO', timing)


def _factor_error(fac, A):
    '''||L L^T - P A P^T|| / ||A|| for a scikit-sparse factor (LL' = P A P')'''
    Lf, p = fac.L(), fac.P()
    return sp.linalg.norm(Lf @ Lf.T - A[p][:, p]) / sp.linalg.norm(A)


def _moved(setup, sigma, seed=4):
    rng = np.random.default_rng(seed)
    r = setup['positions'] + rng.normal(0, sigma, setup['positions'].shape) * np.array([1, 1, 0.1])
    r[:, 2] = np.maximum(r[:, 2], setup['a'] * 1.001)
    return r


def _cluster_matrix_host(s):
    '''B of the cluster preconditioner, built on the host from R_Sup + shift and
    the kept pairs (off-diagonal blocks of the other pairs removed), and whether
    the kept pairs follow the rule: all have max(gap, debye_cut) < pc_gap, no
    cluster exceeds pc_max_cluster, and every cluster of the pairs below pc_gap
    that fits is kept whole.'''
    from scipy.sparse.csgraph import connected_components
    A = _host_csc(s.LC.R_SupS).tocoo()
    P, n = s.LC.pattern.pairs, s.LC.pattern.N
    keep = s.PC.kept_pairs
    km = sp.coo_matrix((np.ones(2 * keep.sum()), (np.r_[P[keep, 0], P[keep, 1]],
                                                  np.r_[P[keep, 1], P[keep, 0]])), shape=(n, n)).tocsr()
    bi, bj = A.row // 6, A.col // 6
    sel = (bi == bj) | (np.asarray(km[bi, bj]).ravel() > 0)
    B = sp.csr_matrix((A.data[sel], (A.row[sel], A.col[sel])), shape=A.shape)
    strong = np.maximum(s.LC.pair_gaps(), s.LC.debye_cut) < s.PC.gap

    def sizes(pr):
        G = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
        nc, lab = connected_components(G, directed=False)
        return np.bincount(lab, minlength=nc), lab
    sz1, _ = sizes(P[keep])
    sz0, lab0 = sizes(P[strong])
    fits = sz0[lab0[P[:, 0]]] <= s.PC.max_cluster
    rule = (not np.any(keep & ~strong)) and sz1.max() <= s.PC.max_cluster and np.all(keep[strong & fits])
    return B, rule


def _noise_identity(s, seed=1):
    '''Largest relative violation of z^T A z = w^T w and z1^T A z0 = w1^T w0
    for z = half_solve(w), A = Delta_R + shift: zero exactly when the
    Brownian increment A z has covariance A.'''
    import cupy
    A, ver = s.LC.Delta_RS, s.LC.pattern_version
    rng = np.random.default_rng(seed)
    w0, w1 = rng.standard_normal(A.shape[0]), rng.standard_normal(A.shape[0])
    z0, z1 = s.DR_factor.half_solve(w0, A, ver), s.DR_factor.half_solve(w1, A, ver)
    Az0 = A @ z0
    ww = float(w0 @ w0)
    return max(abs(float(cupy.dot(z0, Az0)) - ww), abs(float(cupy.dot(z1, Az0)) - float(w1 @ w0))) / ww


def _noise_sampled(s, K=1000, seed=3):
    '''Pairwise noise: variance of x^T y over K draws vs x^T (Delta_R + shift) x
    for 3 random x.  Returns (max |ratio - 1|, limit = 5 standard errors).'''
    import cupy
    A = s.LC.Delta_RS
    X = cupy.asarray(np.random.default_rng(seed).standard_normal((A.shape[0], 3)))
    want = cupy.asnumpy(cupy.sum(X * (A @ X), axis=0))
    proj = np.array([cupy.asnumpy(X.T @ s.DR_factor.sample(s.LC, s.r_vecs, s.periodic_length))
                     for _ in range(K)])
    return float(np.abs(proj.var(axis=0) / want - 1.0).max()), 5.0 * np.sqrt(2.0 / K)


def _noise_label(s):
    f = s.DR_factor
    if getattr(f, 'fallback', None) is not None:
        return 'CHOLMOD (cuDSS fell back)'
    return {'CuDSSNoiseFactor': 'cuDSS', 'PairwiseNoise': 'pairwise'}.get(type(f).__name__,
                                                                          'CHOLMOD')


def _n_plans(s):
    label = _noise_label(s)
    if label == 'pairwise':
        return s.LC.n_pattern_builds
    return s.PC.n_plans if label == 'cuDSS' else s.DR_factor.n_analyses


def check_factors(rep, setup, args):
    '''C2: Brownian increments (pairwise: sampled variances and no clipped
    blocks; Cholesky back ends: sampling identity, and for CHOLMOD the explicit
    factor; again after the particles move, through both code paths).
    C3: preconditioner residual.'''
    import cupy
    s, _ = make_solver('gpu', setup)
    n = 6 * len(setup['positions'])
    label = _noise_label(s)
    if label == 'pairwise':
        _, t_first = _timed(s.DR_factor.sample, s.LC, s.r_vecs, s.periodic_length)
        _, t_fast = _timed(s.DR_factor.sample, s.LC, s.r_vecs, s.periodic_length)
        dev, lim = _noise_sampled(s)
        rep.value('C2 pairwise noise: var(x^T y) = x^T (Delta_R + shift) x', dev < lim,
                  f'max |ratio - 1| = {dev:.3f} over 1000 draws (limit {lim:.3f})')
        rep.value('C2 pairwise noise: every pair and wall block PSD', s.DR_factor.clipped == 0,
                  f'{s.DR_factor.clipped} clipped blocks')
        rep.add('C2 timing', 'INFO', f'Brownian increment {1e3*t_fast:.2f} ms per draw '
                                     f'(first call {1e3*t_first:.1f} ms, kernel compilation)')
    else:
        W = np.random.default_rng(1).standard_normal(n)
        y_d, t_first = _timed(s.DR_factor.sqrt_times, W, s.LC.Delta_RS, s.LC.pattern_version)
        label = _noise_label(s)
        ident = _noise_identity(s)
        rep.value(f'C2 sampling identity z^T A z = w^T w ({label})', ident < 1e-10, f'{ident:.2e}')
        if label.startswith('CHOLMOD (cuDSS'):
            rep.add('C2 Brownian factor on the GPU', 'WARN', 'cuDSS backward solve failed its '
                    'check; CHOLMOD on the host is used (see the warning above)')
        fac = getattr(s.DR_factor, 'fac', None) or getattr(getattr(s.DR_factor, 'fallback', None),
                                                           'fac', None)
        if fac is not None:
            y_ref, t_L = _timed(lambda w: fac.apply_Pt(fac.L() @ w), W)
            err = np.linalg.norm(to_numpy(y_d) - y_ref) / np.linalg.norm(y_ref)
            rep.value('C2 A P^T L^-T W == apply_Pt(L() W) (CHOLMOD, same factor)', err < 1e-10,
                      f'{err:.2e}')
        _, t_fast = _timed(s.DR_factor.sqrt_times, W, s.LC.Delta_RS, s.LC.pattern_version)
        rep.add('C2 timing', 'INFO', f'Brownian increment {1e3*t_first:.1f} ms with '
                                     f'factorisation, {1e3*t_fast:.1f} ms with the factor '
                                     f'reused ({label})')
    b = cupy.asarray(np.random.default_rng(2).standard_normal(n))
    x = s.PC.solve(b)
    res = float(cupy.linalg.norm(s.R_Sup @ x - b) / cupy.linalg.norm(b))
    if s.PC.name == 'clusters':
        B, rule = _cluster_matrix_host(s)
        res_B = np.linalg.norm(B @ to_numpy(x) - to_numpy(b)) / np.linalg.norm(to_numpy(b))
        rep.value('C3 cluster preconditioner solves B exactly (B built independently)',
                  res_B < 1e-10 and rule,
                  f'||B x - b||/||b|| = {res_B:.1e}, selection rule {"ok" if rule else "VIOLATED"}; '
                  f'{s.PC.n_clusters} clusters, largest {s.PC.largest} particles, gap {s.PC.gap:.3f}a, '
                  f'{s.PC.n_split} split ({s.PC.n_chunked} cut into chunks), '
                  f'{int(s.PC.kept_pairs.sum())} pairs kept')
        rep.add('C3 cluster preconditioner vs the full matrix', 'INFO',
                f'||(R_Sup + shift) B^-1 b - b||/||b|| = {res:.2f} (B is an approximation; '
                f'GMRES counts show its effect)')
    else:
        rep.value(f'C3 preconditioner residual ({s.PC.name})', res < 1e-8, f'{res:.2e}')
    del s
    W = np.random.default_rng(1).standard_normal(n)
    for skin, sigma, case in [(0.0, 0.1, 'pair list changed'),
                              (1.0, 0.02, 'lub_skin=1a, pattern kept')]:
        s, _ = make_solver('gpu', setup, lub_skin=skin)
        s.DR_factor.sample(s.LC, s.r_vecs, s.periodic_length)          # noise at the start
        n0 = _n_plans(s)
        s.Set_R_Mats(cupy.asarray(_moved(setup, sigma)))
        s.PC.solve(b)                     # the preconditioner factor in between, as in a step
        label = _noise_label(s)
        if label == 'pairwise':
            dev, lim = _noise_sampled(s)
            path = 'pattern kept' if _n_plans(s) == n0 else 'new pattern'
            ok = dev < lim and s.DR_factor.clipped == 0
            detail = f'max |var ratio - 1| = {dev:.3f} (limit {lim:.3f}), {path}'
        else:
            ident = _noise_identity(s)
            path = 'plan / analysis reused' if _n_plans(s) == n0 else 'new plan / analysis'
            detail = f'z^T A z = w^T w to {ident:.1e} ({path})'
            ok = ident < 1e-10
            fac = getattr(s.DR_factor, 'fac', None)
            if fac is not None:
                err = _factor_error(fac, _host_csc(s.LC.Delta_RS))
                detail += f', ||L L^T - P A P^T||/||A|| = {err:.2e}'
                ok = ok and err < 1e-12
        rep.value(f'C2 Brownian increments after a move ({case}, {label})', ok, detail)
        del s


CHECK_ITERATIONS = 1000      # GMRES iteration budget of every check solve


def _check_budget(kind, s):
    '''its_out giving about CHECK_ITERATIONS GMRES iterations: cupyx counts
    iterations, scipy (CPU solver, scipy_gpu, scipy) counts restart cycles.'''
    if kind == 'gpu' and s.gmres_backend != 'cupy':
        return -(-CHECK_ITERATIONS // s.gmres_restart)
    if kind == 'cpu':
        return -(-CHECK_ITERATIONS // min(300, 6 * len(s.bodies)))
    return CHECK_ITERATIONS


def _limit_iterations(kind, s):
    '''Every Lubrication_solve of s (also inside Update_Bodies_Trap) gets the budget.'''
    s.Lubrication_solve = partial(s.Lubrication_solve, its_out=_check_budget(kind, s))


def mobility_precision(s):
    '''Floor below which no GMRES residual can go: the relative linearity
    error ||M(x+y) - Mx - My|| / ||M(x+y)|| and repeatability ||Mx - Mx'|| / ||Mx||
    of the mobility product, and the dtype libMobility returns.'''
    import cupy
    rng = np.random.default_rng(6)
    n = 6 * len(s.bodies)
    x, y = (cupy.asarray(rng.standard_normal(n)) for _ in range(2))
    mx, my, mxy, mx_again = (s.Wall_Mobility_Mult(v) for v in (x, y, x + y, x))
    lin = float(cupy.linalg.norm(mxy - mx - my) / cupy.linalg.norm(mxy))
    repeat = float(cupy.linalg.norm(mx_again - mx) / cupy.linalg.norm(mx))
    F, T = s.gpu.split_FT(x)
    return lin, repeat, s.solver.Mdot(forces=F, torques=T)[0].dtype


def _counted_solve(s, Xm, its_out):
    '''Lubrication_solve with counts of (I + M dR) applications and PC solves.'''
    timer = MethodTimer(sync=True)
    timer.wrap(s, 'IpMDR_Mult')
    timer.wrap(s, 'IpMDR_PC')
    u, dt = _timed(s.Lubrication_solve, None, Xm, its_out=its_out)
    timer.unwrap_all()
    return to_numpy(u), dt, timer.n['IpMDR_Mult'], timer.n['IpMDR_PC']


def check_solve_and_step(rep, setup, args):
    '''C4: one deterministic solve with the GPU solver (each GMRES back end)
    and the CPU solver, at a tolerance the mobility product can
    reach, each limited to about CHECK_ITERATIONS iterations; every solution's
    true residual is evaluated with the GPU operator.  C5: one deterministic
    trapezoidal step, GPU vs CPU.'''
    import cupy
    r0 = setup['positions']
    FT = make_FT_calc(setup, r0)(None, wrap_positions(r0, setup['L'])).ravel()
    s0, _ = make_solver('gpu', setup)
    lin, repeat, dtype = mobility_precision(s0)
    floor = max(lin, repeat, np.finfo(np.float64).eps)
    tol = args.check_tol if args.check_tol else max(1e-8, 100.0 * floor)
    rep.add('C4 mobility product precision', 'INFO' if repeat < 1e-4 else 'WARN',
            f'libMobility returns {dtype}; linearity error {lin:.1e}, repeatability {repeat:.1e}'
            f' -> check tolerance {tol:.0e}'
            + ('' if repeat < 1e-4 else ' (Mdot is not repeatable: race between libMobility '
               'and CuPy streams?)'))
    b = s0.Wall_Mobility_Mult(cupy.asarray(FT))               # RHS of the solve, M Xm
    b_norm = float(cupy.linalg.norm(b))

    def true_residual(u):
        return float(cupy.linalg.norm(s0.IpMDR_Mult(cupy.asarray(u)) - b)) / b_norm

    sols, iters = {}, {}
    cases = [('gpu', m, dict(gmres_backend=m, gmres_restart=None))
             for m in ('cupy', 'scipy_gpu', 'scipy')]
    if 'cpu' in args.solvers:
        cases.append(('cpu', 'cpu', {}))
    for kind, label, kw in cases:
        name = f'C4 solve, {SOLVER_NAMES[kind]} ({label})'
        try:
            s, _ = make_solver(kind, setup, tolerance=tol, **kw)
            Xm = FT.copy() if kind == 'cpu' else cupy.asarray(FT)
            u, dt, n_mv, n_pc = _counted_solve(s, Xm, _check_budget(kind, s))
            res = true_residual(u)
            sols[label] = u
            detail = (f'{1e3*dt:.1f} ms, {n_mv} applications of (I + M dR), {n_pc} PC solves, '
                      f'residual {res:.1e} (tol {tol:.0e})')
            if kind == 'gpu':
                iters[label] = s.gmres_history[-1]['iterations']
                info = s.gmres_history[-1]['info']
                rep.value(name, info == 0, detail + ('' if info == 0 else
                          f'; not converged in ~{CHECK_ITERATIONS} iterations (info={info})'))
            else:
                rep.add(name, 'INFO', detail)
            del s
        except Exception as e:
            rep.add(name, 'SKIP', repr(e))
        free_gpu_memory()
    if iters.get('scipy_gpu') is not None and iters.get('scipy') is not None:
        a_, b_ = iters['scipy_gpu'], iters['scipy']
        rep.value('C4 scipy_gpu port vs scipy: same iterations',
                  abs(a_ - b_) <= max(1, 0.05 * max(a_, b_)), f'{a_} vs {b_}')
    ref = sols.get('cpu')
    for label in ('cupy', 'scipy_gpu', 'scipy'):
        if ref is None or label not in sols:
            continue
        err = np.linalg.norm(sols[label] - ref) / np.linalg.norm(ref)
        rep.value(f'C4 solution, GPU ({label}) vs CPU', err < 1e3 * tol, f'rel diff {err:.2e}')
    s0 = b = None                       # release the reference solver before C5
    free_gpu_memory()
    if 'cpu' in args.solvers:
        moved = {}
        for kind in ('gpu', 'cpu'):
            s, bodies = make_solver(kind, setup, tolerance=tol)
            _limit_iterations(kind, s)
            FT_calc = make_FT_calc(setup, r0, force_calculator(kind, setup['forces']))
            s.Update_Bodies_Trap(FT_calc, stochastic=False)
            moved[kind] = solver_positions(kind, s, bodies) - r0
            del s
        err = np.abs(moved['gpu'] - moved['cpu']).max() / np.abs(moved['cpu']).max()
        rep.value(f"C5 deterministic step, GPU ({setup['forces']} forces) vs CPU", err < 1e3 * tol,
                  f'max|dr_gpu - dr_cpu| / max|dr_cpu| = {err:.2e}')


def check_orientation_and_wrap(rep, setup, args):
    '''C6: orientation update vs Body.update.  C7: periodic wrap edge cases.'''
    import cupy
    from cupyStokesianDynamics import GPUBackend
    gpu = GPUBackend()
    rng = np.random.default_rng(5)
    n, dt = 16, 0.25
    R0 = Rotation.random(n, random_state=2)
    theta = rng.normal(size=(n, 3)) * 0.4
    ref = []
    for k in range(n):
        b = Body(location=np.zeros(3), orientation=R0[k])
        b.update(np.zeros(3), theta[k], target='current')
        ref.append(b.orientation.as_quat())
    ref = np.array(ref)
    v = np.zeros(6 * n)
    v.reshape(n, 6)[:, 3:] = theta / dt
    v_d = cupy.asarray(v)
    q = to_numpy(gpu.rotate_trap(cupy.asarray(R0.as_quat()), v_d, v_d, dt))
    err = np.minimum(np.abs(q - ref).max(1), np.abs(q + ref).max(1)).max()
    if err < 1e-12:
        rep.add('C6 orientation update vs Body.update', 'PASS', f'max|dq| = {err:.1e}')
    else:
        rep.add('C6 orientation update vs Body.update', 'WARN',
                f'max|dq| = {err:.1e}: Body.update uses another convention (orientations '
                f'only; the dynamics of spheres is unaffected)')
    L = setup['L']
    r = rng.uniform(-3 * L[0], 3 * L[0], (64, 3))
    r[:4, 0] = [-1e-17, L[0], 2 * L[0], -L[0]]
    w = to_numpy(gpu.wrap(cupy.asarray(r), L))
    ok = np.all((w[:, :2] >= 0) & (w[:, :2] < L[:2])) and np.array_equal(w[:, 2], r[:, 2])
    rep.value('C7 periodic wrap into [0, L)', ok, 'including x = -1e-17, L, 2L, -L')


def check_forces(rep, setup, args):
    '''C8: gpu_force_torque_calculator vs force_torque_calculator (numba) at the
    initial configuration, and with particles pushed into the wall-contact
    zone and the soft repulsion switched on (every branch of the kernel).'''
    import cupy
    L = setup['L']
    r_contact = setup['positions'].copy()
    r_contact[:5, 2] = setup['a'] * (1.0 - setup['firm_delta']) - 0.01
    for tag, r_unwrapped, soft in [('initial configuration', setup['positions'], None),
                                   ('wall contact, soft repulsion on', r_contact, 0.004)]:
        r_w = wrap_positions(r_unwrapped, L)
        nbrs, offs = build_neighbour_list(r_w, L, setup['nl_cutoff'])
        kw = dict(a=setup['a'], g=setup['g'], L=L, pair_cutoff=setup['pair_cutoff'],
                  rep_firm=setup['repulsion_firm'], deb_firm=setup['debye_firm'],
                  firm_delta=setup['firm_delta'],
                  rep_soft=setup['repulsion_soft'] if soft is None else soft,
                  deb_soft=setup['debye_soft'], neighbors=nbrs, offsets=offs)
        ref = force_torque_calculator(None, r_w, **kw).ravel()
        r_d = cupy.asarray(r_w)
        gpu_force_torque_calculator(None, r_d, **kw)               # first call compiles
        got, t_gpu = _timed(gpu_force_torque_calculator, None, r_d, **kw)
        _, t_cpu = _timed(on_host(force_torque_calculator), None, r_d, **kw)
        err = np.abs(to_numpy(got) - ref).max() / np.abs(ref).max()
        rep.value(f'C8 GPU forces vs numba ({tag})', err < 1e-12,
                  f'max|diff|/max|FT| = {err:.2e}; GPU {1e3*t_gpu:.2f} ms, '
                  f'numba incl. transfers {1e3*t_cpu:.2f} ms')


def synthetic_pairs(a):
    '''Isolated pairs at controlled gaps and heights hitting every branch of
    the fits (Debye clamp, AT asymptotics, rational fits, near and beyond the
    4.5a cutoff; wall near-contact asymptotics, mid range, far field).'''
    gaps = [5e-5, 1e-3, 4e-3, 5.2e-3, 6e-3, 2e-2, 7e-2, 0.2, 0.6, 1.2, 2.0, 2.45, 2.49, 2.6]
    hts = [1e-5, 1e-3, 9e-3, 0.02, 0.03, 0.05, 0.11, 0.25, 0.39, 0.45, 1.0, 3.3, 5.2, 6.0, 7.5]
    rng = np.random.default_rng(3)
    pos = []
    for n, gap in enumerate(gaps):
        base = np.array([12.0 * n * a + 5 * a, 7.0 * a, (1.0 + hts[n % len(hts)]) * a])
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        pos.append(base)
        pos.append(base + (2.0 + gap) * a * u)
    pos = np.array(pos)
    pos[:, 2] = np.maximum(pos[:, 2], (1.0 + 1e-5) * a)
    return pos, np.array([len(gaps) * 12 * a + 10 * a, 20 * a, 0.0])


# =============================================================================
# Factorisation benchmark (--factor-bench)
# =============================================================================
def _short(e, n=140):
    s = repr(e).replace('\n', ' ')
    return s if len(s) <= n else s[:n] + '...'


def block_ordering(pattern):
    """Fill-reducing ordering computed on the particle graph (CHOLMOD on the
    N x N block pattern, 36x fewer nonzeros than the matrix) and expanded to
    the 6 degrees of freedom of each particle."""
    from sksparse import cholmod
    N = pattern.N
    G = sp.csc_matrix((np.ones(pattern.nb), pattern.bcol, pattern.bptr), shape=(N, N))
    p = cholmod.analyze(G).P()
    return (6 * p[:, None] + np.arange(6)[None, :]).ravel()


def _permuted_host(Ah, perm):
    B = Ah[perm][:, perm].tocsc()
    B.sort_indices()
    return B


def _reordering_algs():
    import nvmath.sparse.advanced as nvsp
    Alg = getattr(nvsp, 'DirectSolverReorderingAlg', None) or getattr(nvsp, 'DirectSolverAlgType')
    return Alg, list(Alg)


def _direct_solver(A, b, handle=None):
    """DirectSolver configured as in cholesky_solver (optionally on a given handle)."""
    import nvmath.sparse.advanced as nvsp
    from cupyStokesianDynamics import _cudss_threading_lib

    def make(mtlib):
        opts = dict(sparse_system_type=nvsp.DirectSolverMatrixType.SPD,
                    sparse_system_view=nvsp.DirectSolverMatrixViewType.FULL,
                    multithreading_lib=mtlib)
        if handle is not None:
            opts['handle'] = handle
        return nvsp.DirectSolver(A, b, options=nvsp.DirectSolverOptions(**opts),
                                 execution=nvsp.ExecutionCUDA())
    try:
        return make(_cudss_threading_lib())
    except Exception:
        return make(None)


def _cudss_case(label, A, alg=None, other=None, noise=False, user_perm=None):
    '''Plan / factorize / solve with cuDSS on a copy of A.  other: refactorise
    with the values of a matrix with the same pattern (values updated in
    place, keeping the plan).  noise: Brownian increment from a backward-only
    solve of the loaded factor, checked with z^T A z = w^T w.  user_perm:
    ordering given to cuDSS instead of its reordering.  Returns cuDSS's
    row permutation.'''
    import cupy
    from cupyStokesianDynamics import _cudss_backward
    n = A.shape[0]
    b = cupy.asarray(np.random.default_rng(8).standard_normal(n))
    A_own = A.copy()
    s, perm_out = None, None
    try:
        s, t_make = _timed(_direct_solver, A_own, b)
        if alg is not None:
            s.plan_config.reordering_algorithm = alg
        if user_perm is not None:
            import nvmath.bindings.cudss as cudss
            p = np.ascontiguousarray(user_perm, dtype=np.int32)
            cudss.data_set(s.handle, s.data_ptr, int(cudss.DataParam.USER_PERM), p.ctypes.data,
                           p.nbytes)
        _, t_plan = _timed(s.plan)
        try:
            perm_out = np.array(s.plan_info.row_permutation, dtype=np.int64)
        except Exception:
            pass
        _, t_fac = _timed(s.factorize)
        s.solve()
        x, t_solve = _timed(s.solve)
        res = float(cupy.linalg.norm(A @ x - b) / cupy.linalg.norm(b))
        line = (f'plan {1e3*t_plan:6.1f} ms   factorize {1e3*t_fac:6.1f} ms   '
                f'solve {1e3*t_solve:5.2f} ms   residual {res:.0e}')
        try:
            line += f'   factor ~{s.plan_info.memory_estimates[0] / 2**20:.0f} MB'
        except Exception:
            pass
        M = A
        if other is not None:
            A_own.data[...] = other.data
            _, t_ref = _timed(s.factorize)
            s.reset_operands(b=b)
            x2 = s.solve()
            res2 = float(cupy.linalg.norm(other @ x2 - b) / cupy.linalg.norm(b))
            line += (f'\n  {"":46s}same plan, Delta_R + shift values: factorize {1e3*t_ref:.1f} ms, '
                     f'residual {res2:.0e}')
            M = other
        if noise:
            rng = np.random.default_rng(9)
            w0, w1 = (cupy.asarray(rng.standard_normal(n)) for _ in range(2))
            z0, t_bwd = _timed(_cudss_backward, s, w0)
            z1 = _cudss_backward(s, w1)
            Mz0 = M @ z0
            ww = float(w0 @ w0)
            e = max(abs(float(z0 @ Mz0) - ww), abs(float(z1 @ Mz0) - float(w1 @ w0))) / ww
            line += (f'\n  {"":46s}Brownian increment from a backward-only solve: '
                     f'{1e3*t_bwd:.2f} ms, z^T A z = w^T w to {e:.0e} -> '
                     f'{"valid" if e < 1e-8 else "NOT valid"}')
        print(f'  {label:44s}  {line}', flush=True)
    except Exception as e:
        print(f'  {label:44s}  SKIP: {_short(e)}', flush=True)
    finally:
        if s is not None:
            try:
                s.free()
            except Exception:
                pass
    return perm_out


def _cudss_replan(A1, A2, cycles=3):
    """Alternating patterns A1, A2 (as when the pair list changes every
    Set_R_Mats): plan + factorize on a new DirectSolver each time (current)
    vs reset_operands + plan on the same DirectSolver."""
    import cupy
    b = cupy.asarray(np.random.default_rng(8).standard_normal(A1.shape[0]))
    for mode in ('new DirectSolver per pattern', 'same DirectSolver, re-planned'):
        times, res, s = [], 0.0, None
        try:
            for k in range(2 * cycles):
                A = (A1 if k % 2 == 0 else A2).copy()
                if mode.startswith('new') or s is None:
                    if s is not None:
                        s.free()
                    (s, t_c) = _timed(_direct_solver, A, b)
                else:
                    _, t_c = _timed(s.reset_operands, a=A)
                _, t_p = _timed(s.plan)
                _, t_f = _timed(s.factorize)
                _, t_f2 = _timed(s.factorize)          # second factorisation, same plan
                s.reset_operands(b=b)
                x = s.solve()
                res = max(res, float(cupy.linalg.norm(A @ x - b) / cupy.linalg.norm(b)))
                if k > 0:                              # the first one warms up
                    times.append((t_c, t_p, t_f, t_f2))
            t = 1e3 * np.mean(times, axis=0)
            print(f'  {mode:44s}  create/reset {t[0]:5.1f} ms   plan {t[1]:6.1f} ms   '
                  f'factorize {t[2]:5.1f} ms (again {t[3]:5.1f} ms)   residual {res:.0e}',
                  flush=True)
        except Exception as e:
            print(f'  {mode:44s}  SKIP: {_short(e)}', flush=True)
        finally:
            if s is not None:
                try:
                    s.free()
                except Exception:
                    pass


def _cholmod_cases(Ah, perm):
    from sksparse import cholmod
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None
    for method in ('default', 'amd', 'metis', 'nesdis'):
        label = f'ordering {method}'
        try:
            fac, t_an = _timed(cholmod.analyze, Ah, ordering_method=method)
            _, t_fac = _timed(fac.cholesky_inplace, Ah)
            line = (f'analyze {1e3*t_an:7.1f} ms   factorize {1e3*t_fac:6.1f} ms   '
                    f'nnz(L) {fac.L().nnz / 1e6:.1f} M')
            if method == 'default' and threadpool_limits is not None:
                with threadpool_limits(limits=os.cpu_count()):
                    _, t_all = _timed(fac.cholesky_inplace, Ah)
                line += f'   factorize with {os.cpu_count()} BLAS threads {1e3*t_all:.1f} ms'
            print(f'  {label:44s}  {line}', flush=True)
        except Exception as e:
            print(f'  {label:44s}  SKIP: {_short(e)}', flush=True)
    try:
        Bh, t_perm = _timed(_permuted_host, Ah, perm)
        fac, t_an = _timed(cholmod.analyze, Bh, ordering_method='natural')
        _, t_fac = _timed(fac.cholesky_inplace, Bh)
        print(f'  {"block ordering, then natural":44s}  analyze {1e3*t_an:7.1f} ms   '
              f'factorize {1e3*t_fac:6.1f} ms   nnz(L) {fac.L().nnz / 1e6:.1f} M   '
              f'(host permutation {1e3*t_perm:.1f} ms)', flush=True)
    except Exception as e:
        print(f'  {"block ordering, then natural":44s}  SKIP: {_short(e)}', flush=True)


def run_factor_bench(setup, args):
    """Where the factorisation time goes, at the initial configuration.  Every
    line is independent; a failing option prints SKIP."""
    print('\n=== factorisation benchmark ===', flush=True)
    import cupy
    s, _ = make_solver('gpu', setup)
    pattern = s.LC.pattern
    A_pc, A_dr = s.LC.R_SupS.copy(), s.LC.Delta_RS.copy()
    s.Set_R_Mats(cupy.asarray(_moved(setup, 0.1)))      # another sparsity pattern
    A_pc2 = s.LC.R_SupS.copy()
    del s
    versions = []
    try:
        import nvmath
        versions.append(f'nvmath {nvmath.__version__}')
        import nvmath.bindings.cudss as cudss
        versions.append('cuDSS ' + '.'.join(str(cudss.get_property(k)) for k in range(3)))
    except Exception:
        pass
    print(f"  n = {A_pc.shape[0]}, nnz = {A_pc.nnz}; OMP_NUM_THREADS="
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')}, {os.cpu_count()} cores; "
          + ', '.join(versions), flush=True)
    perm, t_ord = _timed(block_ordering, pattern)
    print(f'  block ordering on the {pattern.N}-particle graph: {1e3*t_ord:.1f} ms', flush=True)

    print('cuDSS (nvmath DirectSolver), R_Sup + shift; times per factorisation:', flush=True)
    try:
        Alg, algs = _reordering_algs()
    except Exception as e:
        print(f'  SKIP: nvmath not usable ({_short(e)})')
        algs = []
    by_name = {a.name: a for a in algs}
    default = by_name.get('DEFAULT', by_name.get('ALG_DEFAULT'))
    perm_cudss = None
    if algs:
        perm_cudss = _cudss_case('reordering default (current)', A_pc, alg=default, other=A_dr,
                                 noise=True)
        for name in ('AMD', 'ALG_3', 'NESTED_DISSECTION'):
            if name in by_name:
                _cudss_case(f'reordering {name}', A_pc, alg=by_name[name])
        print('cuDSS, a new pattern every time (averages over 5 re-plans):', flush=True)
        _cudss_replan(A_pc, A_pc2)
        print('cuDSS with a given ordering (USER_PERM: no reordering in plan):', flush=True)
        if perm_cudss is not None:
            _cudss_case("cuDSS's own ordering of this pattern", A_pc, alg=default,
                        user_perm=perm_cudss, other=A_dr, noise=True)
        _cudss_case('block ordering', A_pc, alg=default, user_perm=perm)
        _cudss_case('block ordering, inverse', A_pc, alg=default, user_perm=np.argsort(perm))
    print('Cluster preconditioner (pc_backend clusters; no cuDSS), per build / per solve:', flush=True)
    for gap in (0.0, 0.05, 0.1, 0.2):
        try:
            s3, _ = make_solver('gpu', setup, pc_backend='clusters', pc_gap=gap)
            b3 = s3.gpu.xp.asarray(np.random.default_rng(4).standard_normal(6 * len(setup['positions'])))
            s3.PC.factorize(s3.R_Sup, s3.LC.pattern_version)            # warm-up
            s3.PC.solve(b3)
            _, t_b = _timed(s3.PC.factorize, s3.R_Sup, s3.LC.pattern_version)
            _, t_s = _timed(s3.PC.solve, b3)
            print(f'  {"gap < %.2fa" % gap if gap else "block Jacobi":44s}  build {1e3*t_b:6.2f} ms   '
                  f'solve {1e3*t_s:5.2f} ms   {s3.PC.n_clusters} clusters, largest '
                  f'{s3.PC.largest}, {s3.PC.n_split} split', flush=True)
            del s3
        except Exception as e:
            print(f'  {"clusters gap %.2f" % gap:44s}  SKIP: {_short(e)}', flush=True)
    print('Pairwise Brownian increments (default; no factorisation):', flush=True)
    try:
        s2, _ = make_solver('gpu', setup)
        s2.LC.brownian(s2.r_vecs, s2.periodic_length)            # compile
        _, t_pw = _timed(s2.LC.brownian, s2.r_vecs, s2.periodic_length)
        print(f'  {"pairwise":44s}  {1e3*t_pw:.2f} ms per draw, clipped blocks '
              f'{s2.LC.noise_clipped}', flush=True)
        del s2
    except Exception as e:
        print(f'  {"pairwise":44s}  SKIP: {_short(e)}', flush=True)
    print('CHOLMOD, Delta_R + shift (Brownian noise):', flush=True)
    _cholmod_cases(_host_csc(A_dr), perm)


# =============================================================================
# Profiling (--profile)
# =============================================================================
class MethodTimer(object):
    '''Inclusive per-method wall time with device synchronisation, plus NVTX
    ranges (visible in nsys).'''

    def __init__(self, sync=True):
        self.sync = sync
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self._installed = []
        try:
            from cupy.cuda import nvtx
            self._nvtx = nvtx
        except Exception:
            self._nvtx = None

    def timed(self, f, label):
        def wrapper(*a, **k):
            if self.sync:
                gpu_sync()
            if self._nvtx is not None:
                self._nvtx.RangePush(label)
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                if self.sync:
                    gpu_sync()
                self.t[label] += time.perf_counter() - t0
                self.n[label] += 1
                if self._nvtx is not None:
                    self._nvtx.RangePop()
        return wrapper

    def wrap(self, obj, name, label=None):
        f = getattr(obj, name, None)
        if f is None:
            return False
        try:
            setattr(obj, name, self.timed(f, label or name))
        except (AttributeError, TypeError):       # extension objects: use TimedProxy
            return False
        self._installed.append((obj, name))
        return True

    def unwrap_all(self):
        for obj, name in self._installed:
            try:
                delattr(obj, name)                 # instance attribute -> class method again
            except AttributeError:
                pass
        self._installed.clear()


class TimedProxy(object):
    '''Forwards to `obj`, timing the listed methods (for the libMobility solver
    and the C++ Lubrication object, which cannot be patched).'''

    def __init__(self, obj, labels, timer):
        object.__setattr__(self, '_obj', obj)
        for m, label in labels.items():
            if hasattr(obj, m):
                object.__setattr__(self, m, timer.timed(getattr(obj, m), label))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_obj'), name)


class AllocCounter(object):
    '''Counts the device allocations requested through CuPy's allocator.'''

    def __init__(self):
        import cupy
        self.cupy = cupy
        self.pool = cupy.get_default_memory_pool()
        self.n = self.bytes = 0
        self.peak = self.pool.total_bytes()

    def __call__(self, nbytes):
        self.n += 1
        self.bytes += nbytes
        mem = self.pool.malloc(nbytes)
        self.peak = max(self.peak, self.pool.total_bytes())
        return mem

    def __enter__(self):
        self.cupy.cuda.set_allocator(self)
        return self

    def __exit__(self, *exc):
        self.cupy.cuda.set_allocator(self.pool.malloc)
        return False


# solver method -> label
METHODS = {m: m for m in ('Update_Bodies_Trap', 'Set_R_Mats', '_build_neighbour_list',
                          'Lub_Mobility_Root_RHS', 'Lubrication_solve', 'Wall_Mobility_Mult',
                          'IpMDR_Mult', 'IpMDR_PC', 'Check_Update_With_Jump_Trap',
                          'put_r_vecs_in_periodic_box', 'project_to_periodic_image')}
LUBRICATION = {'assemble': 'lubrication_assembly', 'ResistCSC_both': 'lubrication_assembly',
               'set_pairs': 'lubrication_pattern', 'brownian': 'pairwise noise'}
MOBILITY = {m: 'libMobility.' + m for m in ('Mdot', 'sqrtMdotW', 'divM', 'setPositions')}
FACTORS = {'nvmath': {'_new_solver': 'cuDSS create', '_plan': 'cuDSS plan',
                      '_factorize': 'cuDSS factorize', '_solve': 'cuDSS solve',
                      'backward': 'cuDSS backward'},
           'cholesky_solver': {'factorize': 'cuDSS plan+factorize', 'solve': 'cuDSS solve'},
           'cholmod_host': {'factorize': 'CHOLMOD factorize (PC)', 'solve': 'CHOLMOD solve (PC)'},
           'clusters': {'factorize': 'clusters build', '_structure': 'clusters host',
                        '_fill': 'clusters fill', '_invert': 'clusters inverse',
                        'solve': 'clusters solve'}}


def instrument(s, timer):
    for name, label in METHODS.items():
        timer.wrap(s, name, label)
    for name, label in FACTORS.get(getattr(getattr(s, 'PC', None), 'name', None), {}).items():
        timer.wrap(s.PC, name, label)
    if type(getattr(s, 'DR_factor', None)).__name__ == 'CholmodFactor':
        timer.wrap(s.DR_factor, 'factorize', 'CHOLMOD factorize (noise)')
    proxied = {name: label for name, label in LUBRICATION.items()
               if hasattr(s.LC, name) and not timer.wrap(s.LC, name, label)}
    if proxied:
        s.LC = TimedProxy(s.LC, proxied, timer)
    s.solver = TimedProxy(s.solver, MOBILITY, timer)


def uninstrument(s, timer):
    timer.unwrap_all()
    for attr in ('solver', 'LC'):
        v = getattr(s, attr, None)
        if isinstance(v, TimedProxy):
            setattr(s, attr, object.__getattribute__(v, '_obj'))


def gpu_solver_counters(s):
    extra = {}
    if getattr(s.PC, 'name', '') == 'clusters':
        extra = dict(largest_cluster=s.PC.largest, split_clusters=s.PC.n_split)
    return dict(**extra, pattern_builds=s.LC.n_pattern_builds, assemblies=s.LC.n_assemblies,
                pc_plans=s.PC.n_plans, pc_factorizations=s.PC.n_factorizations,
                pc_solves=s.PC.n_solves, brownian_analyses=s.DR_factor.n_analyses,
                brownian_factorizations=s.DR_factor.n_factorizations)


def profile_solver(kind, setup, args):
    '''K instrumented steps (synchronised per-method times, call counts, device
    allocations), then K plain steps for the step time.'''
    print(f'\n--- profiling {SOLVER_NAMES[kind]} ---', flush=True)
    np.random.seed(42)
    extra = dict(sync_timings=True) if kind == 'gpu' else {}
    t0 = time.perf_counter()
    s, bodies = make_solver(kind, setup, **extra)
    out = {'solver': SOLVER_NAMES[kind], 'setup_s': time.perf_counter() - t0,
           'forces': 'cpu' if kind == 'cpu' else setup['forces']}
    calc = force_calculator(kind, setup['forces'])
    FT_calc = make_FT_calc(setup, setup['positions'], calc)
    r_nl_ref = setup['positions'].copy()

    def step():
        nonlocal FT_calc, r_nl_ref
        s.Update_Bodies_Trap(FT_calc, stochastic=True)
        r_now = solver_positions(kind, s, bodies)        # also synchronises
        if needs_rebuild(r_now, r_nl_ref, setup['L'], setup['rebuild_tol']):
            FT_calc, r_nl_ref = make_FT_calc(setup, r_now, calc), r_now.copy()

    for _ in range(args.warmup_steps):                  # kernel compilation, cuDSS warm-up
        step()
    if hasattr(s, 'timings'):
        s.timings = {k: 0.0 for k in s.timings}
        s._n_steps_timed = 0
    if kind == 'gpu':
        s.gmres_history.clear()
        c0 = gpu_solver_counters(s)
    K = args.profile_steps

    # pass 1: instrumented
    timer = MethodTimer(sync=True)
    instrument(s, timer)
    if kind == 'gpu':
        with AllocCounter() as alloc:
            for _ in range(K):
                step()
        out.update(device_allocations_per_step=alloc.n / K,
                   device_alloc_MB_per_step=alloc.bytes / K / 2**20,
                   pool_high_water_MB=alloc.peak / 2**20)
    else:
        for _ in range(K):
            step()
    uninstrument(s, timer)
    out['methods'] = {k: {'ms_per_step': 1e3 * v / K, 'calls_per_step': timer.n[k] / K}
                      for k, v in sorted(timer.t.items(), key=lambda kv: -kv[1])}
    if hasattr(s, 'timings'):
        out['timings_ms_per_step'] = {k: 1e3 * v / K for k, v in s.timings.items()}
        s.print_timings()
    if kind == 'gpu':
        c1 = gpu_solver_counters(s)
        out['counters_per_step'] = {k: (c1[k] - c0[k]) / K for k in c1
                                    if k not in ('largest_cluster', 'cluster_gap')}
        out['counters_per_step'].update({k: c1[k] for k in ('largest_cluster', 'cluster_gap')
                                         if k in c1})
        out['gmres_applications_per_solve'] = [h['matvecs'] for h in s.gmres_history]
        s.sync_timings = False

    # pass 2: plain step time (one synchronisation per step)
    times = []
    for _ in range(K):
        if kind != 'cpu':
            gpu_sync()
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)
    out['step_ms'] = 1e3 * float(np.mean(times))
    out['step_ms_std'] = 1e3 * float(np.std(times))
    out['rejections'] = [int(s.num_rejections_wall), int(s.num_rejections_jump)]
    print(f"  step time {out['step_ms']:.1f} +- {out['step_ms_std']:.1f} ms", flush=True)
    return out


def print_profile_table(results):
    kinds = [k for k in ('gpu', 'cpu') if k in results]
    print('\n' + ' ' * 42 + ''.join(f'{SOLVER_NAMES[k]:>28s}' for k in kinds))

    def row(label, f):
        cells = []
        for k in kinds:
            try:
                cells.append(f(results[k]))
            except Exception:
                cells.append('-')
        print(f'{label:42s}' + ''.join(f'{c:>28s}' for c in cells))

    row('step time [ms]', lambda r: f"{r['step_ms']:.1f}")
    if 'gpu' in kinds:
        row('step time / GPU solver', lambda r: f"{r['step_ms'] / results['gpu']['step_ms']:.1f}x")
    for name in ['Set_R_Mats', 'lubrication_assembly', 'lubrication_pattern',
                 '_build_neighbour_list', 'Lub_Mobility_Root_RHS', 'Lubrication_solve',
                 'IpMDR_Mult', 'IpMDR_PC', 'Wall_Mobility_Mult', 'libMobility.Mdot',
                 'libMobility.sqrtMdotW', 'libMobility.divM', 'libMobility.setPositions',
                 'cuDSS create', 'cuDSS plan', 'cuDSS factorize', 'cuDSS solve',
                 'cuDSS backward', 'cuDSS plan+factorize', 'CHOLMOD factorize (noise)',
                 'pairwise noise', 'clusters build', 'clusters host', 'clusters fill',
                 'clusters inverse', 'clusters solve',
                 'Check_Update_With_Jump_Trap', 'put_r_vecs_in_periodic_box',
                 'project_to_periodic_image']:
        row(f'  {name} [ms | calls]',
            lambda r, nm=name: f"{r['methods'][nm]['ms_per_step']:.1f} | "
                               f"{r['methods'][nm]['calls_per_step']:.1f}")
    row('device allocations / step', lambda r: f"{r['device_allocations_per_step']:.0f}")
    row('device MB allocated / step', lambda r: f"{r['device_alloc_MB_per_step']:.0f}")
    row('memory-pool high water [MB]', lambda r: f"{r['pool_high_water_MB']:.0f}")
    row('rejections (wall, jump)', lambda r: f"{r['rejections']}")
    print('(per step; inclusive, synchronised method times; IpMDR_Mult calls = GMRES '
          'applications of I + M dR)')
    if 'gpu' in kinds:
        print('GPU solver, per step:', ', '.join(
            f'{k} {v:g}' for k, v in results['gpu']['counters_per_step'].items()))


def run_profile(setup, args, out_prefix):
    print('\n=== profile ===', flush=True)
    results = {}
    for kind in args.solvers:
        try:
            results[kind] = profile_solver(kind, setup, args)
        except Exception as e:
            print(f'  {SOLVER_NAMES[kind]} failed: {e!r}')
            traceback.print_exc()
        free_gpu_memory()                        # the solver is gone once profile_solver returns
    if results:
        print_profile_table(results)
        meta = {k: setup[k] for k in ('N', 'phi', 'dt', 'tolerance')}
        meta.update(setup['solver_options'], profile_steps=args.profile_steps)
        with open(out_prefix + '_profile.json', 'w') as f:
            json.dump({'params': meta, 'results': results}, f, indent=1, default=float)
        print(f'profile written to {out_prefix}_profile.json')
    return results


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
