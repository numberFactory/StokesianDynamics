'''
Class to handle Lubrication solve on the GPU.

GPU version of pyStokesianDynamics: same constructor arguments, same public
methods, same predictor-corrector scheme.  The class reads like the CPU
version; everything GPU specific (CuPy kernels, sparse-matrix and
direct-solver plumbing, GMRES implementations) is collected below the line of
#'s at the bottom of this file.

Differences from pyStokesianDynamics
  * Positions (and orientations) live on the GPU: self.r (N x 3, unwrapped)
    and self.q (N x 4 quaternions, scalar last).  The Body objects are a
    mirror, updated by push_bodies()/pull_bodies(), or every step with
    sync_bodies=True.
  * FT_calc(bodies, r_vecs) receives the wrapped positions as an (N, 3) device
    array and may return numpy or cupy, (6N,) or (2N, 3).
  * GMRES: gmres_backend = 'scipy' (default: scipy's gmres on the host, as
    the CPU solver calls it, operators on the GPU), 'scipy_gpu' (the same
    algorithm ported to device arrays) or 'cupy' (cupyx gmres, restart 10).
  * Brownian increments Delta_R^{1/2} W are drawn pair by pair on the GPU
    (Delta_R is a sum of positive semi-definite pair and wall blocks), with
    no factorisation of Delta_R; a Cholesky factor (cuDSS or CHOLMOD) is
    available as an option.
  * Preconditioner (default pc_backend='clusters'): R_Sup + shift restricted
    to clusters of near-contact particles, inverted exactly as small batched
    dense blocks on the GPU; no sparse direct solver.  With pc_backend='nvmath'
    one cuDSS plan per sparsity pattern, numeric factors when they are needed;
    plans / symbolic analyses are reused only while the sparsity pattern is
    unchanged.
'''
import importlib
import os
import time
import warnings

import numpy as np
import scipy.spatial as spatial
from scipy.spatial.transform import Rotation

# Native libraries are loaded in the same order as in cupyStokesianDynamics:
# CHOLMOD (scikit-sparse), CuPy, the cuDSS wrapper, then libMobility.  Shared
# libraries and OpenMP runtimes are bound by whichever module loads first, so
# keep this order.  Missing optional modules are skipped (CPU validation).
for _module in ('sksparse.cholmod', 'cupy', 'cupyx.scipy.sparse',
                'cupyx.scipy.sparse.linalg', 'cholesky_solver'):
    try:
        importlib.import_module(_module)
    except Exception:
        pass

from libMobility import NBody, DPStokes
from lubrication_cupy import LubricationAssembler


class cupyStokesianDynamics(object):
    '''
    Class to handle Lubrication solve (GPU)
    '''

    def __init__(self, bodies, a, eta, periodic_length,
                 z_max, debye_length=1e-4, allowChangingBoxSize=False,
                 gmres_backend='scipy', gmres_restart=None, pc_backend='auto',
                 lub_skin=0.0, sync_bodies=False, track_orientation=True,
                 predictor_guess='last', jump_threshold=4.0,
                 sync_timings=False, pc_host_threads=None, pc_replan_in_place=False,
                 pc_gap=0.1, pc_max_cluster=64,
                 noise_backend='auto',
                 mobility=None, gpu=None):
        '''
        Constructor. Initialises lubrication and libMobility solver objects.

        GPU-version options
          gmres_backend     'scipy' (default) | 'scipy_gpu' | 'cupy'
          gmres_restart     default 10 for 'cupy', 300 (as the CPU solver)
                            for the scipy variants
          pc_backend        'auto' = 'clusters' (near-contact clusters, batched
                            dense inverses, no sparse direct solver) | 'nvmath'
                            | 'cholesky_solver' | 'cholmod_host'
          pc_gap, pc_max_cluster  pc_backend='clusters': pairs with a gap below
                            pc_gap (units of a) couple particles into clusters;
                            inside a cluster larger than pc_max_cluster the gap
                            is halved there, then it is cut into chunks
          noise_backend     Brownian increments with covariance Delta_R + shift:
                            'pairwise' (default, 'auto'): pair by pair on the
                            GPU, no factorisation | 'cudss': backward solve
                            with the factor of the nvmath plan | 'cholmod'
          lub_skin          lubrication neighbour-list skin, units of a
                            (0: exact 4.5a list on every call)
          sync_bodies       mirror positions/orientations into `bodies` every step
          track_orientation integrate orientations
          predictor_guess   'last' (previous corrector velocity) | 'zero'
          jump_threshold    reject a step that moves a particle > jump_threshold*a
          sync_timings      synchronise the GPU around the timed regions
          pc_host_threads   host threads for cuDSS's reordering with
                            pc_backend='nvmath' (default: the threading
                            layer's, i.e. OMP_NUM_THREADS)
          pc_replan_in_place  pc_backend='nvmath': plan a new pattern on the
                            same DirectSolver instead of a new one
          mobility, gpu     libMobility-like object / device backend to use
        '''
        self.bodies          = bodies
        self.periodic_length = np.asarray(periodic_length, dtype=np.float64)
        self.tolerance       = 1e-4
        self.eta             = eta
        self.a               = a
        self.kT              = 0.0041419464
        self.dt              = 1.0
        self.cutoff          = 4.5 # DO NOT CHANGE, FITS ARE HARD-CODED TO THIS VALUE
        self.debye_length    = debye_length

        self.Delta_R  = None
        self.R_MB     = None
        self.R_Sup    = None   # R_Sup + small diagonal (factorised for the preconditioner)
        self.vel_last = None

        # Initialize perturbation matrix diagonal
        small_F = 6.0 * np.pi * self.eta * self.a          *self.tolerance
        small_T = 8.0 * np.pi * self.eta * self.a**3       *self.tolerance

        self.small_diag = np.tile(
            np.array([small_F, small_F, small_F,
                    small_T, small_T, small_T]),
            len(self.bodies)
        )

        # GPU machinery (defined at the bottom of this file)
        self.gpu       = gpu if gpu is not None else GPUBackend()
        self.LC        = self.gpu.make_lubrication(debye_length, a, eta, small_F, small_T)
        if noise_backend == 'cudss' and pc_backend == 'auto':
            pc_backend = 'nvmath'             # the cuDSS noise shares the nvmath plan
        self.PC        = make_preconditioner(pc_backend, self.gpu, host_threads=pc_host_threads,
                                             replan_in_place=pc_replan_in_place, lc=self.LC,
                                             gap=pc_gap, max_cluster=pc_max_cluster)
        self.DR_factor = make_noise_factor(noise_backend, self.PC, self.gpu, self.LC)

        # Initialise libMobility solver
        if mobility is not None:
            self.solver = mobility
        else:
            L = self.periodic_length
            if L[0] <= 0 and L[1] <= 0:
                self.solver = NBody("open", "open", "single_wall")
                self.solver.setParameters(wallHeight=0.0)
            else:
                self.solver = DPStokes("periodic", "periodic", "single_wall")
                self.solver.setParameters(
                    Lx=L[0], Ly=L[1], zmin=0.0, zmax=z_max,
                    allowChangingBoxSize=allowChangingBoxSize
                )
            self.solver.initialize(viscosity=eta, hydrodynamicRadius=a,
                                   includeAngular=True)

        # GPU-version options
        if gmres_backend not in GMRES_DEFAULT_RESTART:
            raise ValueError(f'gmres_backend must be one of {sorted(GMRES_DEFAULT_RESTART)}')
        if predictor_guess not in ('last', 'zero'):
            raise ValueError("predictor_guess must be 'last' or 'zero'")
        self.gmres_backend     = gmres_backend
        self.gmres_restart     = int(gmres_restart if gmres_restart is not None
                                     else GMRES_DEFAULT_RESTART[gmres_backend])
        self.lub_skin          = float(lub_skin)
        self.sync_bodies       = bool(sync_bodies)
        self.track_orientation = bool(track_orientation)
        self.predictor_guess   = predictor_guess
        self.jump_threshold    = float(jump_threshold)
        self.sync_timings      = bool(sync_timings)

        # State on the GPU: unwrapped positions and orientations
        self.r = self.gpu.to_device(self._body_locations())
        self.q = None
        if self.track_orientation:
            quat = self._body_quaternions()
            if quat is None:
                self.track_orientation = False
            else:
                self.q = self.gpu.to_device(quat)
        self.r_vecs        = None    # wrapped positions of the current matrices
        self._mats_current = False   # matrices built at self.r?
        self._r_nl_ref     = None    # positions of the last neighbour-list build

        self.num_rejections_wall = 0
        self.num_rejections_jump = 0
        self.gmres_history       = []   # one record per Lubrication_solve call

        # Cumulative timing for Update_Bodies_Trap components (seconds)
        self.timings = {
            'set_r_mats':  0.0,
            'stochastic':  0.0,
            'ft_calc':     0.0,
            'solve_pred':  0.0,
            'solve_corr':  0.0,
        }
        self._n_steps_timed = 0

    def project_to_periodic_image(self, r, L):
        '''
        Project vectors r (..., 3), device arrays, to the minimal image
        representation centered around (0,0,0) and of size L=(Lx, Ly, Lz).
        If any dimension of L is equal or smaller than zero the box is
        assumed to be infinite in that direction.
        '''
        return self.gpu.min_image(r, L)

    def put_r_vecs_in_periodic_box(self, r_vecs, L):
        '''
        Return a copy of the (N,3) device positions wrapped into [0, L) in
        the periodic directions.
        '''
        return self.gpu.wrap(r_vecs, L)

    def Set_R_Mats(self, r_vecs=None):
        '''
        Build lubrication resistance matrices in sparse CSR format on the GPU
        and factorise the preconditioner.  Sets self.R_MB, self.R_Sup
        (+ small diagonal), self.Delta_R.  The Cholesky factor of
        Delta_R + small diagonal (Brownian increments) is recomputed when the
        next Lub_Mobility_Root_RHS call needs it.
        '''
        self._mats_current = r_vecs is None
        if r_vecs is None:
            r_vecs = self.r
        r_vecs = self.put_r_vecs_in_periodic_box(r_vecs, self.periodic_length)
        self.r_vecs = r_vecs

        self.solver.setPositions(r_vecs.flatten())

        # neighbour list (pairs within cutoff + skin), rebuilt only when needed;
        # the CSR pattern only changes when the pair list changes
        if self._neighbour_list_expired(r_vecs):
            L      = self.periodic_length
            r_np   = self.gpu.to_host(r_vecs)
            box    = np.where(L > 0, L, 0.0) if np.any(L > 0) else None
            r_tree = spatial.cKDTree(r_np, boxsize=box,
                                     balanced_tree=False, compact_nodes=False)
            pairs  = r_tree.query_pairs((self.cutoff + self.lub_skin) * self.a,
                                        output_type='ndarray')
            self.LC.set_pairs(len(r_np), pairs)
            self._r_nl_ref = r_vecs

        self.LC.assemble(r_vecs, self.periodic_length)
        self.R_MB, self.R_Sup, self.Delta_R = self.LC.R_MB, self.LC.R_SupS, self.LC.Delta_R

        self.PC.factorize(self.R_Sup, self.LC.pattern_version)
        self.DR_factor.invalidate()

    def Wall_Mobility_Mult(self, X, r_vecs=None):
        '''
        Multiply a vector X of forces and torques by the RPB mobility.
        X should be formatted as [F_1 T_1 F_2 T_2 ...]^T (device array).
        '''
        if r_vecs is not None:
            self.solver.setPositions(r_vecs.flatten())

        F, T = self.gpu.split_FT(X)

        U, W = self.solver.Mdot(forces=F, torques=T)

        return self.gpu.merge_UW(U, W)

    def IpMDR_Mult(self, X):
        '''
        Returns (I + M_RPY * Delta_R) * X
        '''
        D_R       = self.Delta_R @ X
        M_Delta_R = self.Wall_Mobility_Mult(D_R)
        return X + M_Delta_R

    def IpMDR_PC(self, X):
        '''
        Returns (R_Sup + small diagonal)^{-1} * R_MB * X
        (GPU direct solver, factorised in Set_R_Mats).
        '''
        RHS = self.R_MB @ X
        Y_F = self.PC.solve(RHS)
        return Y_F

    def Lubrication_solve(self, X, Xm, X0=None, print_residual=False, its_out=1000):
        '''
        Solve the lubrication problem using GMRES.
        Computes U = [I + M_RPY * Delta_R]^{-1} * (X + M_RPY * Xm).
        Requires Set_R_Mats() to have been called first.  Device arrays in/out.
        '''
        if self.Delta_R is None:
            self.Set_R_Mats()

        xp = self.gpu.xp
        num_particles = len(self.bodies)

        RHS = xp.zeros(6 * num_particles)
        if Xm is not None:
            RHS += self.Wall_Mobility_Mult(Xm)
        if X is not None:
            RHS += X.ravel()

        RHS_norm = float(xp.linalg.norm(RHS))
        if RHS_norm == 0:
            return xp.zeros(6 * num_particles)  # A is invertible, so U = 0
        RHS = RHS / RHS_norm

        if X0 is not None:
            X0 = X0 / RHS_norm

        # Accept when ||RHS - A U|| <= tolerance (||RHS|| = 1), as the CPU solver.
        U, info, n_iter, n_matvec, res_list = gmres_solve(
            self.gmres_backend, self.IpMDR_Mult, self.IpMDR_PC, RHS, X0,
            tol=self.tolerance, restart=self.gmres_restart, its_out=its_out,
            gpu=self.gpu)

        U = U * RHS_norm
        self.gmres_history.append({'iterations': n_iter, 'matvecs': n_matvec, 'info': info})

        if print_residual:
            if n_iter is None:     # cupyx reports neither iterations nor residuals
                print(f'GMRES ({self.gmres_backend}): {n_matvec} operator applications, info={info}')
            else:
                final = f'{res_list[-1]:.3e}' if res_list else 'n/a (X0 already converged)'
                print(f'GMRES ({self.gmres_backend}): {n_iter} iterations, '
                      f'info={info}, final residual={final}')

        return U

    def Lub_Mobility_Root_RHS(self):
        '''
        Returns RHS_Xm = sqrt(2kT/dt) * Delta_R^{1/2} * W1
            and RHS_X  = sqrt(2kT/dt) * M^{1/2} * W
        for use in Lubrication_solve to compute the square root of the
        lubrication-corrected mobility (device arrays).
        '''
        prefactor           = np.sqrt(2.0 * self.kT / self.dt)

        # Delta_R^{1/2} * W1: Gaussian with covariance Delta_R + shift, drawn
        # pair by pair (default) or from a Cholesky factor (noise_backend)
        DRhalf  = self.DR_factor.sample(self.LC, self.r_vecs, self.periodic_length)

        # M^{1/2} * W via sqrtMdotW — W is generated internally by libMobility
        sqrtM_W_U, sqrtM_W_W = self.solver.sqrtMdotW()
        sqrtM_W = self.gpu.merge_UW(sqrtM_W_U, sqrtM_W_W)

        return prefactor * DRhalf, prefactor * sqrtM_W

    def Update_Bodies_Trap(self, FT_calc, stochastic=True, print_residual=False):
        '''
        Predictor-corrector (trapezoidal) timestep update.
        stochastic=True  : includes Brownian noise and drift terms.
        stochastic=False : purely deterministic, force-driven.
        FT_calc(bodies, r_vecs) receives the wrapped positions (device array).
        Returns (reject_wall, reject_jump).
        '''
        num_particles = len(self.bodies)

        # positions edited through the Body objects?
        if self.sync_bodies:
            self.pull_bodies()

        # save initial configuration
        r_old, q_old = self.r, self.q

        # wrapped positions for predictor (matrices from the end of the last step)
        if not self._mats_current:
            self.Set_R_Mats()
        r_vecs = self.r_vecs

        # ── FT_calc (predictor) ───────────────────────────────────────────
        _t0 = self._clock()
        FT  = self.gpu.as_vector(FT_calc(self.bodies, r_vecs), 6 * num_particles)
        self.timings['ft_calc'] += self._clock() - _t0

        # ── stochastic RHS ────────────────────────────────────────────────
        if stochastic:
            _t0 = self._clock()
            Root_Xm, Root_X = self.Lub_Mobility_Root_RHS()
            MXm   = self.Wall_Mobility_Mult(Root_Xm)
            Mhalf = Root_X + MXm

            div_U, div_W = self.solver.divM()
            div_UW = self.gpu.merge_UW(div_U, div_W)
            D_M = 2.0 * self.kT * div_UW
            self.timings['stochastic'] += self._clock() - _t0
        else:
            Mhalf = None
            D_M   = None

        # ── predictor solve ───────────────────────────────────────────────
        _t0   = self._clock()
        X0    = self.vel_last if self.predictor_guess == 'last' else None
        vel_p = self.Lubrication_solve(X=Mhalf, Xm=FT, X0=X0, print_residual=print_residual)
        self.timings['solve_pred'] += self._clock() - _t0

        # predictor positions: r_old + vel_p * dt
        r_pred = self.gpu.advance(r_old, vel_p, self.dt)

        # ── Set_R_Mats at corrector positions ─────────────────────────────
        _t0 = self._clock()
        self.Set_R_Mats(r_vecs=r_pred)
        self.timings['set_r_mats'] += self._clock() - _t0

        # ── FT_calc (corrector) ───────────────────────────────────────────
        _t0  = self._clock()
        FT_C = self.gpu.as_vector(FT_calc(self.bodies, self.r_vecs), 6 * num_particles)
        self.timings['ft_calc'] += self._clock() - _t0

        # ── corrector solve ───────────────────────────────────────────────
        RHS_X_C = (D_M + Mhalf) if stochastic else None
        _t0   = self._clock()
        vel_c = self.Lubrication_solve(X=RHS_X_C, Xm=FT_C, X0=vel_p, print_residual=print_residual)
        self.vel_last = vel_c
        self.timings['solve_corr'] += self._clock() - _t0

        # trapezoidal average → new positions (and orientations)
        r_new = self.gpu.advance_trap(r_old, vel_p, vel_c, self.dt)
        q_new = self.gpu.rotate_trap(q_old, vel_p, vel_c, self.dt) if self.track_orientation else None

        reject_wall, reject_jump = self.Check_Update_With_Jump_Trap(r_new, r_old)
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        # accept or reject
        if (reject_wall + reject_jump) == 0:
            self.r, self.q = r_new, q_new

        # ── Set_R_Mats for next step ──────────────────────────────────────
        _t0 = self._clock()
        self.Set_R_Mats()
        self.timings['set_r_mats'] += self._clock() - _t0

        if self.sync_bodies:
            self.push_bodies()

        self._n_steps_timed += 1
        return reject_wall, reject_jump

    def print_timings(self):
        '''Print mean per-step timings for Update_Bodies_Trap components.'''
        n = max(self._n_steps_timed, 1)
        total = sum(self.timings.values())
        print(f"\n--- Update_Bodies_Trap timings ({n} steps"
              f"{', GPU-synchronised' if self.sync_timings else ''}) ---")
        for key, val in self.timings.items():
            print(f"  {key:<14s}  {val/n*1e3:8.2f} ms/step  "
                  f"({100*val/max(total, 1e-300):.1f}%)")
        print(f"  {'total':<14s}  {total/n*1e3:8.2f} ms/step")
        mv = [h['matvecs'] for h in self.gmres_history]
        if mv:
            print(f"  GMRES ({self.gmres_backend}, restart {self.gmres_restart}): "
                  f"{np.mean(mv):.1f} operator applications per solve (max {max(mv)})")

    def Check_Update_With_Jump_Trap(self, r_new, r_old):
        '''
        Reject a timestep if any particle crosses the wall (z < 0) or moves
        more than jump_threshold*a (4a, as the CPU solver) in a single step.
        As in the CPU version the first offending particle decides.
        Returns (reject_wall, reject_jump).
        '''
        threshold = self.jump_threshold * self.a
        code = self.gpu.to_host(self.gpu.jump_codes(r_new, r_old, threshold))
        bad  = np.flatnonzero(code)
        if bad.size == 0:
            return 0, 0

        k = bad[0]
        if code[k] == 1:
            print("Rejected timestep: wall crossing.")
            return 1, 0

        disp = np.linalg.norm(self.gpu.to_host(r_new[k] - r_old[k]))
        print(f"Rejected timestep: large jump ({disp:.4f} > {threshold:.4f}).")
        return 0, 1

    # ── GPU version only: state access and Body mirror ─────────────────────
    @property
    def positions(self):
        '''Unwrapped positions, (N, 3) device array.'''
        return self.r

    @property
    def orientations(self):
        '''Quaternions (scalar last), (N, 4) device array; None without
        track_orientation.'''
        return self.q

    def push_bodies(self):
        '''Write the GPU state into the Body objects (b.location in place and,
        with track_orientation, b.orientation).'''
        loc = self.gpu.to_host(self.r)
        for b, x in zip(self.bodies, loc):
            if isinstance(b.location, np.ndarray) and b.location.shape == (3,):
                np.copyto(b.location, x)
            else:
                b.location = x.copy()
        if self.track_orientation:
            rots = Rotation.from_quat(self.gpu.to_host(self.q))
            for k, b in enumerate(self.bodies):
                b.orientation = rots[k]
        self._r_pushed = loc

    def pull_bodies(self):
        '''Re-read b.location; if it was changed outside the solver, take the
        new positions (the matrices are rebuilt before the next step).
        Returns True when something changed.'''
        loc = self._body_locations()
        if np.array_equal(loc, getattr(self, '_r_pushed', None)):
            return False
        if np.array_equal(loc, self.gpu.to_host(self.r)):
            self._r_pushed = loc
            return False
        self.r = self.gpu.to_device(loc)
        self._r_pushed = loc
        self._mats_current = False
        return True

    @property
    def isolated(self):
        '''Particles above 4.5a with no higher-index neighbour within 4.5a
        (as the CPU solver; not used by the solve).'''
        if self.LC.pattern is None:
            return []
        pos = self.gpu.to_host(self.r_vecs)
        pr  = self.LC.pattern.pairs
        has_upper = np.zeros(len(pos), dtype=bool)
        if pr.size:
            d = self.gpu.to_host(self.project_to_periodic_image(
                self.gpu.to_device(pos[pr[:, 1]] - pos[pr[:, 0]]), self.periodic_length))
            has_upper[pr[np.linalg.norm(d, axis=1) <= self.cutoff * self.a, 0]] = True
        return list(np.flatnonzero((pos[:, 2] >= self.cutoff * self.a) & ~has_upper))

    def _body_locations(self):
        return np.array([b.location for b in self.bodies], dtype=np.float64).reshape(-1, 3)

    def _body_quaternions(self):
        try:
            return np.ascontiguousarray(
                Rotation.concatenate([b.orientation for b in self.bodies]).as_quat())
        except Exception as e:
            warnings.warn(f'orientation tracking disabled ({e})')
            return None

    def _neighbour_list_expired(self, r_vecs):
        if self.LC.pattern is None or self.lub_skin <= 0.0:
            return True
        d2 = self.gpu.max_min_image_disp2(r_vecs, self._r_nl_ref, self.periodic_length)
        return d2 > (0.5 * self.lub_skin * self.a) ** 2

    def _clock(self):
        if self.sync_timings:
            self.gpu.synchronize()
        return time.perf_counter()


# ##############################################################################
# ##############################################################################
#
#    GPU-specific implementation details
#
#    1. Device backend: CuPy element-wise kernels, transfers
#    2. Preconditioner back ends: cuDSS (cholesky_solver or nvmath), CHOLMOD,
#       near-contact clusters
#    3. Brownian increments: pair by pair (default), CHOLMOD or cuDSS factor
#    4. GMRES implementations
#
#    The lubrication matrices themselves are assembled by
#    lubrication_cupy.LubricationAssembler (fused CUDA kernels).
#
# ##############################################################################
# ##############################################################################


# =============================================================================
# 1. Device backend
# =============================================================================
# CuPy ElementwiseKernel sources: name -> (in_params, out_params, body)
ELEMENTWISE_KERNELS = {
    # wrap positions into [0, L) in periodic directions (robust at x -> L^-)
    'wrap': ('T r, T Lx, T Ly, T Lz', 'T out', r"""
        const int c = (int)(i % 3);
        const T Lc = (c == 0) ? Lx : ((c == 1) ? Ly : Lz);
        T x = r;
        if (Lc > 0) {
            x = x - floor(x / Lc) * Lc;
            if (x >= Lc) x -= Lc;
            if (x < 0) x += Lc;
        }
        out = x;
        """),
    # minimum image (round half away from zero, as the CPU code)
    'min_image': ('T d, T Lx, T Ly, T Lz', 'T out', r"""
        const int c = (int)(i % 3);
        const T Lc = (c == 0) ? Lx : ((c == 1) ? Ly : Lz);
        T x = d;
        if (Lc > 0) x -= round(x / Lc) * Lc;
        out = x;
        """),
    # squared minimum-image displacement per component
    'dmin2': ('T a, T b, T Lx, T Ly, T Lz', 'T out', r"""
        const int c = (int)(i % 3);
        const T Lc = (c == 0) ? Lx : ((c == 1) ? Ly : Lz);
        T d = a - b;
        if (Lc > 0) d -= Lc * rint(d / Lc);
        out = d * d;
        """),
    # [F1 T1 F2 T2 ...] -> F, T
    'split_ft': ('raw T x', 'T f, T t', r"""
        const ptrdiff_t p = i / 3, c = i - 3 * p;
        f = x[6 * p + c];
        t = x[6 * p + 3 + c];
        """),
    # U, W -> [U1 W1 U2 W2 ...]
    'merge_uw': ('T u, T w', 'raw T out', r"""
        const ptrdiff_t p = i / 3, c = i - 3 * p;
        out[6 * p + c] = u;
        out[6 * p + 3 + c] = w;
        """),
    # r + v_lin dt                                     (size = 3N)
    'advance': ('raw T r, raw T v, T dt', 'raw T out', r"""
        const ptrdiff_t p = i / 3, c = i - 3 * p;
        out[i] = r[i] + v[6 * p + c] * dt;
        """),
    # r + 0.5 (vp + vc)_lin dt                         (size = 3N)
    'advance_trap': ('raw T r, raw T vp, raw T vc, T dt', 'raw T out', r"""
        const ptrdiff_t p = i / 3, c = i - 3 * p;
        const T v = 0.5 * (vc[6 * p + c] + vp[6 * p + c]);
        out[i] = r[i] + v * dt;
        """),
    # 0 ok, 1 wall crossing (z < 0), 2 jump |rn - r|^2 > jump2   (size = N)
    'jump_codes': ('raw T rn, raw T r, T jump2', 'raw int32 code', r"""
        T e2 = 0;
        for (int c = 0; c < 3; ++c) {
            const T e = rn[3 * i + c] - r[3 * i + c];
            e2 += e * e;
        }
        code[i] = (rn[3 * i + 2] < 0) ? 1 : ((e2 > jump2) ? 2 : 0);
        """),
    # q_new = exp(dt w_trap) * q_old  (space frame, scipy scalar-last)   (size = N)
    'rotate_trap': ('raw T q, raw T vp, raw T vc, T dt', 'raw T qn', r"""
        T th[3];
        for (int c = 0; c < 3; ++c)
            th[c] = (0.5 * (vc[6 * i + 3 + c] + vp[6 * i + 3 + c])) * dt;
        const T ang = sqrt(th[0] * th[0] + th[1] * th[1] + th[2] * th[2]);
        T s;
        if (ang <= 1e-3) { const T a2 = ang * ang; s = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0; }
        else s = sin(0.5 * ang) / ang;
        const T dx = s * th[0], dy = s * th[1], dz = s * th[2], dw = cos(0.5 * ang);
        const T qx = q[4 * i], qy = q[4 * i + 1], qz = q[4 * i + 2], qw = q[4 * i + 3];
        const T nx = dw * qx + qw * dx + (dy * qz - dz * qy);
        const T ny = dw * qy + qw * dy + (dz * qx - dx * qz);
        const T nz = dw * qz + qw * dz + (dx * qy - dy * qx);
        const T nw = dw * qw - (dx * qx + dy * qy + dz * qz);
        const T inv = 1.0 / sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        qn[4 * i] = nx * inv; qn[4 * i + 1] = ny * inv;
        qn[4 * i + 2] = nz * inv; qn[4 * i + 3] = nw * inv;
        """),
}


class DeviceBackend(object):
    '''
    Array-module-agnostic part of the backend.  Subclasses provide xp,
    sparse, _ew() (element-wise kernel launch), transfers and sparse_linalg.
    '''

    def _ew(self, name, *args, size=None):
        raise NotImplementedError

    def _L(self, L):
        return [float(x) if x > 0 else 0.0 for x in L]

    def wrap(self, r, L):
        r = self.xp.ascontiguousarray(r, dtype=self.xp.float64)
        out = self.xp.empty_like(r)
        self._ew('wrap', r, *self._L(L), out)
        return out

    def min_image(self, d, L):
        d = self.xp.ascontiguousarray(d, dtype=self.xp.float64)
        out = self.xp.empty_like(d)
        self._ew('min_image', d, *self._L(L), out)
        return out

    def max_min_image_disp2(self, a, b, L):
        d2 = self.xp.empty_like(a)
        self._ew('dmin2', a, b, *self._L(L), d2)
        return float(d2.reshape(-1, 3).sum(axis=1).max())

    def split_FT(self, X):
        n3 = X.size // 2
        F = self.xp.empty(n3, dtype=X.dtype)
        T = self.xp.empty(n3, dtype=X.dtype)
        self._ew('split_ft', X, F, T)
        return F, T

    def merge_UW(self, U, W):
        # float64 even if libMobility was built in single precision
        U = self.xp.asarray(U, dtype=self.xp.float64).reshape(-1)
        W = self.xp.asarray(W, dtype=self.xp.float64).reshape(-1)
        out = self.xp.empty(2 * U.size, dtype=self.xp.float64)
        self._ew('merge_uw', U, W, out)
        return out

    def advance(self, r, v, dt):
        out = self.xp.empty_like(r)
        self._ew('advance', r, v, float(dt), out, size=r.size)
        return out

    def advance_trap(self, r, vp, vc, dt):
        out = self.xp.empty_like(r)
        self._ew('advance_trap', r, vp, vc, float(dt), out, size=r.size)
        return out

    def rotate_trap(self, q, vp, vc, dt):
        out = self.xp.empty_like(q)
        self._ew('rotate_trap', q, vp, vc, float(dt), out, size=q.shape[0])
        return out

    def jump_codes(self, rn, r, threshold):
        code = self.xp.empty(r.shape[0], dtype=self.xp.int32)
        self._ew('jump_codes', rn, r, float(threshold) ** 2, code, size=r.shape[0])
        return code

    def as_vector(self, x, n):
        v = self.xp.asarray(x, dtype=self.xp.float64).reshape(-1)
        if v.size != n:
            raise ValueError(f'expected {n} entries, got {v.size}')
        return v

    def make_lubrication(self, *args, **kw):
        return LubricationAssembler(*args, xp=self.xp, sparse=self.sparse, **kw)


class GPUBackend(DeviceBackend):
    '''CuPy implementation.'''

    def __init__(self):
        import cupy
        import cupyx
        import cupyx.scipy.sparse
        self.cp = cupy
        self.xp = cupy
        self.sparse = cupyx.scipy.sparse
        self._cupyx = cupyx
        self._kernels = {k: cupy.ElementwiseKernel(i, o, body, 'sdf_' + k)
                         for k, (i, o, body) in ELEMENTWISE_KERNELS.items()}

    def _ew(self, name, *args, size=None):
        if size is None:
            self._kernels[name](*args)
        else:
            self._kernels[name](*args, size=size)

    @property
    def sparse_linalg(self):
        import cupyx.scipy.sparse.linalg
        return cupyx.scipy.sparse.linalg

    def synchronize(self):
        self.cp.cuda.get_current_stream().synchronize()

    def to_host(self, a, out=None):
        if out is None:
            return self.cp.asnumpy(a)
        return self.cp.asnumpy(a, out=out)

    def to_device(self, h):
        return self.cp.asarray(h)

    def empty_pinned(self, n, dtype):
        return self._cupyx.empty_pinned((n,), dtype=dtype)


# =============================================================================
# 2. Preconditioner back ends: factorize(A, pattern_version), solve(b)
# =============================================================================
class PCCholeskySolver(object):
    '''cuDSS wrapper of the original solver (cholesky_solver.CholeskySolver):
    analysis + factorisation on every Set_R_Mats call.  Used exactly as
    cupyStokesianDynamics uses it: every factorisation gets a fresh CSR copy
    of the matrix (the original passed R_Sup_shifted.tocsr()), which is kept
    alive with the factor, and the new factor is built before the previous
    one is released.  The wrapper never sees the assembler's persistent
    buffers, which are overwritten in place.'''
    name = 'cholesky_solver'

    def __init__(self, gpu):
        from cholesky_solver import CholeskySolver
        self._cls = CholeskySolver
        self._s = self._A = None
        self.n_plans = self.n_factorizations = self.n_solves = 0

    def factorize(self, A, pattern_version):
        A_own = A.copy()
        s = self._cls(A_own)
        self._s, self._A = s, A_own         # the previous factor is released here
        self.n_plans += 1
        self.n_factorizations += 1

    def solve(self, b):
        self.n_solves += 1
        return self._s.solve(b)


def _cudss_threading_lib():
    '''cuDSS threading layer: the one cholesky_solver located, else $CUDSS_THREADING_LIB.'''
    try:
        from cholesky_solver import _MULTITHREADING_LIB
        if _MULTITHREADING_LIB:
            return _MULTITHREADING_LIB
    except Exception:
        pass
    return os.environ.get('CUDSS_THREADING_LIB') or None


class PCNvmath(object):
    '''
    cuDSS through nvmath-python's DirectSolver (what cholesky_solver wraps,
    with the same options: SPD, full view, the cuDSS threading layer).

    One plan (reordering + symbolic factorisation) per sparsity pattern,
    kept while the pattern is unchanged, and shared by the two SPD matrices
    of a step: R_Sup + shift (this preconditioner) and Delta_R + shift
    (Brownian increments, CuDSSNoiseFactor), which have the same pattern.
    The numeric factor of whichever matrix is needed next is computed on
    demand: its values are copied into the solver's own matrix (in place, as
    nvmath requires for keeping the plan) and factorize() runs.  One cuDSS
    library handle serves all plans.  host_threads sets the threads of
    cuDSS's host-side reordering.  replan_in_place: a new pattern is planned
    on the same DirectSolver (reset_operands + plan) instead of a new one.
    '''
    name = 'nvmath'

    def __init__(self, gpu, host_threads=None, replan_in_place=False):
        import nvmath.sparse.advanced as nvsp
        self._nvsp = nvsp
        self.replan_in_place = bool(replan_in_place)
        self._xp = gpu.xp
        self._mtlib = _cudss_threading_lib()
        self._host_threads = host_threads
        self._solver = self._A = self._src = None
        self._version = None
        self._loaded = None                 # which matrix the numeric factor holds
        import nvmath.bindings.cudss as cudss
        self._cudss = cudss
        self._handle = cudss.create()       # fails without a usable cuDSS: 'auto' moves on
        self.n_plans = self.n_factorizations = self.n_solves = 0

    # ---- plan and numeric factor (shared with CuDSSNoiseFactor) ------------
    def ensure_plan(self, A, pattern_version):
        '''Plan for the sparsity pattern of A unless pattern_version is current.'''
        if self._solver is not None and pattern_version == self._version:
            return False
        A_own = A.copy()                    # cuDSS never sees the assembler's buffers
        solver = None
        if self.replan_in_place and self._solver is not None:
            try:                            # keep the DirectSolver (handle, data object)
                self._solver.reset_operands(a=A_own)
                self._plan(self._solver)
                solver = self._solver
            except Exception:               # not supported: new DirectSolver from now on
                self.replan_in_place = False
        if solver is None:
            solver = self._new_solver(A_own)
            self._plan(solver)
            if self._solver is not None:
                self._solver.free()
        self._solver, self._A = solver, A_own
        self._version = pattern_version
        self._loaded = None
        self.n_plans += 1
        return True

    def _new_solver(self, A_own):
        nvsp = self._nvsp
        b = self._xp.zeros(A_own.shape[0], dtype=self._xp.float64)

        def make(mtlib, handle):
            opts = dict(sparse_system_type=nvsp.DirectSolverMatrixType.SPD,
                        sparse_system_view=nvsp.DirectSolverMatrixViewType.FULL,
                        multithreading_lib=mtlib)
            if handle is not None:
                opts['handle'] = handle
            return nvsp.DirectSolver(A_own, b, options=nvsp.DirectSolverOptions(**opts),
                                     execution=nvsp.ExecutionCUDA())
        for mtlib, handle in ((self._mtlib, self._handle), (None, self._handle)):
            try:
                solver = make(mtlib, handle)
                break
            except Exception:               # threading layer rejected (as cholesky_solver)
                continue
        else:
            solver = make(None, None)       # raise the real error
        if self._host_threads:
            solver.plan_config.host_nthreads = int(self._host_threads)
        return solver

    def _plan(self, solver):
        solver.plan()

    def values_changed(self):
        self._loaded = None

    def load(self, A, key):
        '''Numeric factor of A (same pattern as the plan) unless key is loaded.'''
        if self._loaded == key:
            return False
        self._A.data[...] = A.data          # in place: nvmath keeps the plan
        self._factorize()
        self._loaded = key
        return True

    def _factorize(self):
        self._solver.factorize()

    def backward(self, w):
        '''z = P^T L^-T w for the loaded factor (P A P^T = L L^T).'''
        return _cudss_backward(self._solver, self._xp.ascontiguousarray(w, dtype=self._xp.float64))

    # ---- preconditioner interface -------------------------------------------
    def factorize(self, A, pattern_version):
        '''New values of R_Sup + shift; the numeric factor follows at the next solve.'''
        self.ensure_plan(A, pattern_version)
        self.values_changed()
        self._src = A

    def solve(self, b):
        if self.load(self._src, 'pc'):
            self.n_factorizations += 1
        self.n_solves += 1
        return self._solve(b)

    def _solve(self, b):
        self._solver.reset_operands(b=self._xp.ascontiguousarray(b))
        return self._solver.solve()

    def __del__(self):
        try:
            if self._solver is not None:
                self._solver.free()
            if self._handle is not None:
                self._cudss.destroy(self._handle)
        except Exception:
            pass


def _cudss_backward(solver, w):
    '''cuDSS backward substitution with permutation (phases SOLVE_BWD |
    SOLVE_BWD_PERM) for the factor held by an nvmath DirectSolver.  nvmath
    only exposes the full solve, so this calls cuDSS on the DirectSolver's
    handle, config, data and matrix objects.  The RHS and the solution buffer
    both hold w, so the result does not depend on which one cuDSS reads.'''
    import cupy
    import nvmath.bindings.cudss as cudss
    x = w.copy()
    solver.reset_operands(b=w)
    cudss.matrix_set_values(solver.x_ptr, x.data.ptr)
    cudss.set_stream(solver.handle, cupy.cuda.get_current_stream().ptr)
    cudss.execute(solver.handle, int(cudss.Phase.SOLVE_BWD) | int(cudss.Phase.SOLVE_BWD_PERM),
                  solver.config_ptr, solver.data_ptr, solver.a_ptr, solver.x_ptr, solver.b_ptr)
    return x


class PCHostCholmod(object):
    '''CHOLMOD on the host (two transfers per solve).  Fallback / debugging.'''
    name = 'cholmod_host'

    def __init__(self, gpu):
        self._gpu = gpu
        self._f = CholmodFactor(gpu)
        self.n_plans = self.n_factorizations = self.n_solves = 0

    def factorize(self, A, pattern_version):
        self.n_plans += int(self._f.factorize(A, pattern_version))
        self.n_factorizations += 1

    def solve(self, b):
        self.n_solves += 1
        return self._gpu.to_device(self._f.fac(self._gpu.to_host(b)))


# cluster sizes a group is padded to: 1, 2, 3, 4, 6, 8, 12, 16, ... (at most 1.5x padding)
_CLUSTER_SIZES = np.array(sorted({1 << k for k in range(16)} | {3 << k for k in range(15)}))


class PCClusters(object):
    '''
    Preconditioner from clusters of near-contact particles, without a sparse
    direct solver.

    B = R_Sup + shift with the off-diagonal blocks of the pairs that are not
    kept removed; the diagonal blocks are kept whole.  B is block diagonal over
    the connected components of the kept pairs ("clusters"), so B^-1 is
    exact: one small dense inverse per cluster.

    Kept pairs: gap < gap (units of a; gaps are clamped at the fits'
    debye_cut, so closer pairs and overlaps count as touching).  Inside a
    cluster with more than max_cluster particles, and only there, the
    threshold is halved until it fits; what the threshold cannot split (pairs
    at the clamp, e.g. chains of overlapping particles) is cut into connected
    chunks of at most max_cluster particles along a breadth-first order.

    Dropping a pair's off-diagonal block leaves its diagonal blocks, which are
    positive definite (every pair's R_Sup contribution and every wall block
    is), so B is SPD for any set of kept pairs.  IpMDR_PC applies B^-1 R_MB, so
    the preconditioned operator is K times the one with the full factor,
    K = B^-1 (R_Sup + shift).  On a non-overlapping N = 2763, phi = 0.66
    configuration K's eigenvalues lie in [0.62, 1.26] for gap = 0.1a (largest
    cluster 22 particles) and in [0.25, 1.73] for gap = 0 (block Jacobi).

    Layout: clusters are grouped by padded size (1, 2, 3, 4, 6, 8, 12, ...
    particles, padded with identity), all groups live in one flat buffer, and
    each group is inverted in one batched call.  A solve is one gather, one
    batched matrix-vector product per group and one scatter.

    factorize = _structure (host: gaps, clusters, index arrays)
              + _fill (device: the kept blocks of A into the flat buffer)
              + _invert (device: batched inverses).
    '''
    name = 'clusters'

    def __init__(self, gpu, lc, gap=0.1, max_cluster=64):
        from scipy.sparse import csgraph
        self._csgraph = csgraph
        self.gpu, self.lc = gpu, lc
        self.gap, self.max_cluster = float(gap), int(max_cluster)
        self.gap_used = self.gap
        self.largest = self.n_clusters = self.n_split = self.n_chunked = None
        self.gap_min = None                  # smallest threshold used inside split clusters
        self.kept_pairs = None               # host bool per pattern pair
        self.n_plans = self.n_factorizations = self.n_solves = 0
        self._groups, self._dof, self._n = [], None, 0

    # ----------------------------------------------------------------- build
    def factorize(self, A, pattern_version):
        st = self._structure()
        D = self._fill(A, st)
        self._invert(D, st)
        self.n_plans += 1
        self.n_factorizations += 1

    def _components(self, N, pairs):
        import scipy.sparse as sp
        G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
        return self._csgraph.connected_components(G, directed=False)

    def _select(self, gaps):
        '''Kept pairs (see the class docstring), with the final clusters.'''
        P, N, cap = self.lc.pattern.pairs, self.lc.pattern.N, self.max_cluster
        keep = gaps < self.gap
        g, self.n_split, self.n_chunked, self.gap_min = self.gap, 0, 0, None
        while True:
            n_c, labels = self._components(N, P[keep])
            sizes = np.bincount(labels, minlength=n_c)
            big = sizes > cap
            if not big.any():
                return keep, n_c, labels, sizes
            if g == self.gap:
                self.n_split = int(big.sum())
            inside = keep & big[labels[P[:, 0]]]       # kept pairs of oversized clusters
            if 0.5 * g > self.lc.debye_cut:           # halve the threshold there only
                g *= 0.5
                self.gap_min = g
                keep = keep & ~(inside & (gaps >= g))
                continue
            # the threshold cannot split these any further: cut along a BFS order
            import scipy.sparse as sp
            kp = P[inside]
            G = sp.coo_matrix((np.ones(len(kp)), (kp[:, 0], kp[:, 1])), shape=(N, N)).tocsr()
            cid = labels.copy()
            nxt = n_c
            for c in np.flatnonzero(big):
                root = int(np.flatnonzero(labels == c)[0])
                order = self._csgraph.breadth_first_order(G, root, directed=False,
                                                          return_predecessors=False)
                k = np.arange(len(order)) // cap
                cid[order] = nxt + k
                nxt += int(k[-1]) + 1
            self.n_chunked = int(big.sum())
            keep = keep & (cid[P[:, 0]] == cid[P[:, 1]])

    def _structure(self):
        p = self.lc.pattern
        N = p.N
        gaps = np.maximum(self.lc.pair_gaps(), self.lc.debye_cut)
        keep, n_c, labels, sizes = self._select(gaps)
        self.kept_pairs = keep
        self.largest, self.n_clusters = int(sizes.max()), int(n_c)
        # padded size, group and position in the group of every cluster
        S = _CLUSTER_SIZES[np.searchsorted(_CLUSTER_SIZES, sizes)]
        groups, gidx, batch = np.unique(S, return_inverse=True, return_counts=True)
        cord = np.argsort(gidx, kind='stable')
        qpos = np.empty(n_c, dtype=np.int64)
        qpos[cord] = np.arange(n_c) - np.repeat(np.cumsum(batch) - batch, batch)
        ng = 6 * groups.astype(np.int64)
        mat_off = np.cumsum(batch * ng * ng) - batch * ng * ng
        slot_off = np.cumsum(batch * ng) - batch * ng
        c_n = ng[gidx]
        c_mat = mat_off[gidx] + qpos * c_n * c_n
        c_slot = slot_off[gidx] + qpos * c_n
        # local index of each particle in its cluster
        order = np.argsort(labels, kind='stable')
        loc = np.empty(N, dtype=np.int64)
        loc[order] = np.arange(N) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        # slot -> dof (padding slots -> 6N, a zero entry of the padded vector)
        n_slots = int(np.sum(batch * ng))
        dof = np.full(n_slots, 6 * N, dtype=np.int64)
        dof[(c_slot[labels] + 6 * loc)[:, None] + np.arange(6)] = 6 * np.arange(N)[:, None] + np.arange(6)
        # identity on the diagonal of the padding
        npad = 6 * (S - sizes)
        cr = np.repeat(np.arange(n_c), npad)
        ip = 6 * sizes[cr] + (np.arange(int(npad.sum())) - np.repeat(np.cumsum(npad) - npad, npad))
        pad_pos = c_mat[cr] + ip * c_n[cr] + ip
        # kept blocks: diagonal blocks and both blocks of every kept pair
        kept_b = np.concatenate([keep, [False]])[np.where(p.bpair >= 0, p.bpair, len(keep))]
        kb = np.flatnonzero((p.brow == p.bcol) | kept_b)
        bi, bj = p.brow[kb].astype(np.int64), p.bcol[kb].astype(np.int64)
        start = p.bptr[bi].astype(np.int64)
        deg = p.bptr[bi + 1].astype(np.int64) - start
        nb = c_n[labels[bi]]
        # entry (lr, lc) of block kb: A.data[36 start + 6 deg lr + 6 (kb - start) + lc] (CSR layout)
        # goes to D[c_mat + (6 loc_i + lr) n + 6 loc_j + lc]
        blocks = np.stack([c_mat[labels[bi]] + 6 * loc[bi] * nb + 6 * loc[bj], nb,
                           36 * start + 6 * (kb - start), 6 * deg])
        return dict(blocks=blocks, pad_pos=pad_pos, dof=dof, mat_total=int(np.sum(batch * ng * ng)),
                    groups=[(int(mo), int(so), int(b), int(n))
                            for mo, so, b, n in zip(mat_off, slot_off, batch, ng)], n=6 * N)

    def _fill(self, A, st):
        xp, dev = self.gpu.xp, self.gpu.to_device
        D = xp.zeros(st['mat_total'], dtype=xp.float64)
        if st['pad_pos'].size:
            D[dev(st['pad_pos'])] = 1.0
        bl = dev(st['blocks'])
        r = xp.arange(6, dtype=bl.dtype)
        lr, lc = r[None, :, None], r[None, None, :]
        dst = bl[0][:, None, None] + bl[1][:, None, None] * lr + lc
        src = bl[2][:, None, None] + bl[3][:, None, None] * lr + lc
        D[dst.ravel()] = A.data[src.ravel()]
        self._dof = dev(st['dof'])
        return D

    def _invert(self, D, st):
        xp = self.gpu.xp
        self._groups = [(so, b, n, xp.linalg.inv(D[mo:mo + b * n * n].reshape(b, n, n)))
                        for mo, so, b, n in st['groups']]
        self._n = st['n']

    # ----------------------------------------------------------------- solve
    def solve(self, b):
        xp = self.gpu.xp
        self.n_solves += 1
        x = xp.zeros(self._n + 1, dtype=xp.float64)        # last slot: padding (zero)
        x[:self._n] = b
        xs = x[self._dof]
        ys = xp.empty_like(xs)
        for so, nb, n, inv in self._groups:
            ys[so:so + nb * n] = xp.matmul(inv, xs[so:so + nb * n].reshape(nb, n, 1)).reshape(-1)
        y = xp.empty(self._n + 1, dtype=xp.float64)
        y[self._dof] = ys
        return y[:self._n]


PC_BACKENDS = {'cholesky_solver': PCCholeskySolver, 'nvmath': PCNvmath,
               'cholmod_host': PCHostCholmod, 'clusters': PCClusters}
_PC_NEEDS = {'cholesky_solver': 'the cholesky_solver module used by cupyStokesianDynamics',
             'nvmath': 'nvmath-python with cuDSS (pip install "nvmath-python[cu12]" nvidia-cudss-cu12)',
             'cholmod_host': 'scikit-sparse', 'clusters': 'scipy'}


def make_preconditioner(name, gpu, host_threads=None, replan_in_place=False, lc=None,
                        gap=0.1, max_cluster=64):
    ''''auto' = 'clusters' (no sparse direct solver; 38% faster steps than
    cuDSS on the N = 2763 test).  Without an assembler: nvmath, then
    cholesky_solver, then host CHOLMOD.'''
    def build(cand):
        if cand == 'nvmath':
            return PCNvmath(gpu, host_threads, replan_in_place)
        if cand == 'clusters':
            return PCClusters(gpu, lc, gap, max_cluster)
        return PC_BACKENDS[cand](gpu)

    if name != 'auto':
        if name not in PC_BACKENDS:
            raise ValueError(f'unknown pc_backend {name!r}; choose from {sorted(PC_BACKENDS)} or auto')
        try:
            return build(name)
        except ImportError as e:
            raise ImportError(f'pc_backend={name!r} needs {_PC_NEEDS[name]} ({e})') from e
    if lc is not None:
        return build('clusters')
    errors = []
    for cand in ('nvmath', 'cholesky_solver'):
        try:
            return build(cand)
        except Exception as e:
            errors.append(f'{cand}: {e}')
    warnings.warn('no GPU direct solver available (' + '; '.join(errors) +
                  '); using host CHOLMOD for the preconditioner (slow)')
    return PCHostCholmod(gpu)


# =============================================================================
# 3. Brownian increments with covariance Delta_R + shift: pair by pair
#    (PairwiseNoise, default), or from a Cholesky factor: CHOLMOD on the host,
#    or cuDSS sharing the nvmath preconditioner's plan
# =============================================================================
class CholmodFactor(object):
    '''
    Host CHOLMOD factor of a symmetric device matrix.  The symbolic analysis
    is reused only for the same sparsity-pattern version: scikit-sparse
    requires an identical pattern, and reusing an analysis across patterns
    (as the old GPU solver did when only the shape matched) gives a wrong
    factor.  Values go to the host through a pinned buffer.
    '''

    def __init__(self, gpu):
        self.gpu = gpu
        self.fac = None
        self.valid = False
        self._version = None
        self._vals = None
        self._indptr = self._indices = None
        self.n_analyses = self.n_factorizations = 0

    def invalidate(self):
        self.valid = False

    def factorize(self, A, pattern_version):
        '''Factorise A; returns True when a new symbolic analysis was done.'''
        import scipy.sparse as sp
        from sksparse import cholmod
        n, nnz = A.shape[0], A.data.size
        if self._vals is None or nnz > self._vals.size or 2 * nnz < self._vals.size:
            self._vals = self.gpu.empty_pinned(int(1.25 * nnz) + 16, np.float64)
        vals = self._vals[:nnz]
        self.gpu.to_host(A.data, out=vals)
        new_pattern = self.fac is None or pattern_version != self._version
        if new_pattern:
            self._indptr = np.asarray(self.gpu.to_host(A.indptr))
            self._indices = np.asarray(self.gpu.to_host(A.indices))
        Ah = sp.csc_matrix((vals, self._indices, self._indptr), shape=(n, n))
        if new_pattern:
            self.fac = cholmod.analyze(Ah)
            self._version = pattern_version
            self.n_analyses += 1
        self.fac.cholesky_inplace(Ah)
        self.n_factorizations += 1
        self.valid = True
        return new_pattern

    def half_solve(self, W, A, pattern_version):
        '''z = P^T L^-T W (device array), so that A z = P^T L W.'''
        if not self.valid:
            self.factorize(A, pattern_version)
        z = self.fac.apply_Pt(self.fac.solve_Lt(self.gpu.to_host(W) if not
                                                isinstance(W, np.ndarray) else W,
                                                use_LDLt_decomposition=False))
        return self.gpu.to_device(np.ascontiguousarray(z))

    def sqrt_times(self, W, A, pattern_version):
        '''
        P^T L W for the factor of A (LL' = P A P'), i.e. what
        fac.apply_Pt(fac.L() @ W) returns, evaluated as A (P^T L^-T W): one
        triangular solve on the host and one SpMV on the GPU, without forming
        L (fac.L() copies the factor and converts it to scipy every call).
        '''
        try:
            return A @ self.half_solve(W, A, pattern_version)
        except TypeError:                   # very old scikit-sparse without the flag
            fac = self.fac
            return self.gpu.to_device(np.ascontiguousarray(fac.apply_Pt(fac.L() @ W)))

    def sample(self, lc, r_vecs, periodic_length):
        '''Delta_R^{1/2} W1 with W1 = np.random.randn (as the CPU solver).'''
        W1 = np.random.randn(lc.Delta_RS.shape[0])
        return self.sqrt_times(W1, lc.Delta_RS, lc.pattern_version)


class CuDSSNoiseFactor(object):
    '''
    Brownian increments from the cuDSS factor of Delta_R + shift, sharing the
    nvmath preconditioner's plan (the two matrices have the same pattern):
    the Delta_R values are factorised on the GPU on demand, z = P^T L^-T W
    comes from cuDSS's backward substitution, and A z = P^T L W has
    covariance A.  The first use checks z^T A z = w^T w (true exactly when
    A z has covariance A); if that fails, or cuDSS's solve phases are not
    reachable, CHOLMOD on the host takes over with a warning.
    '''

    def __init__(self, pc, gpu):
        self.pc = pc
        self.gpu = gpu
        self.fallback = None
        self.checked = False
        self.n_analyses = 0                 # the plan belongs to the preconditioner
        self.n_factorizations = 0

    def invalidate(self):
        self.pc.values_changed()
        if self.fallback is not None:
            self.fallback.invalidate()

    def factorize(self, A, pattern_version):
        if self.fallback is not None:
            return self.fallback.factorize(A, pattern_version)
        self.pc.ensure_plan(A, pattern_version)
        if self.pc.load(A, 'dr'):
            self.n_factorizations += 1
        return False

    def half_solve(self, W, A, pattern_version):
        '''z = P^T L^-T W (device array), so that A z = P^T L W.'''
        if self.fallback is not None:
            return self.fallback.half_solve(W, A, pattern_version)
        try:
            self.factorize(A, pattern_version)
            if not self.checked:
                self._check(A)
            return self.pc.backward(self.gpu.to_device(np.ascontiguousarray(W))
                                    if isinstance(W, np.ndarray) else W)
        except Exception as e:
            self._use_cholmod(f'cuDSS backward solve unavailable ({e!r})')
            return self.fallback.half_solve(W, A, pattern_version)

    def sqrt_times(self, W, A, pattern_version):
        return A @ self.half_solve(W, A, pattern_version)

    def sample(self, lc, r_vecs, periodic_length):
        '''Delta_R^{1/2} W1 with W1 = np.random.randn (as the CPU solver).'''
        W1 = np.random.randn(lc.Delta_RS.shape[0])
        return self.sqrt_times(W1, lc.Delta_RS, lc.pattern_version)

    def _check(self, A):
        xp = self.gpu.xp
        rng = np.random.default_rng(12345)
        w0, w1 = (self.gpu.to_device(rng.standard_normal(A.shape[0])) for _ in range(2))
        z0, z1 = self.pc.backward(w0), self.pc.backward(w1)
        Az0 = A @ z0
        ww = float(xp.dot(w0, w0))
        err = max(abs(float(xp.dot(z0, Az0)) - ww), abs(float(xp.dot(z1, Az0)) - float(xp.dot(w1, w0))))
        if not err <= 1e-8 * ww:
            raise RuntimeError(f'z^T A z = w^T w violated by {err / ww:.1e}')
        self.checked = True

    def _use_cholmod(self, why):
        warnings.warn(f'Brownian factor: {why}; using CHOLMOD on the host')
        self.fallback = CholmodFactor(self.gpu)


class PairwiseNoise(object):
    '''
    Brownian increments with covariance Delta_R + shift, drawn pair by pair
    by the lubrication kernels (LubricationAssembler.brownian): Delta_R is a
    sum of positive semi-definite 12x12 pair blocks and 6x6 wall blocks, so
    y = sum_p chol(P_p) xi_p + sum_i chol(W_i + shift) eta_i has exactly that
    covariance, without factorising Delta_R.  The normals come from the
    device RNG (cupy.random; seed with cupy.random.seed).
    '''
    n_analyses = 0
    n_factorizations = 0

    def __init__(self, lc):
        self.lc = lc
        self.n_samples = 0

    def invalidate(self):
        pass

    def sample(self, lc, r_vecs, periodic_length):
        self.n_samples += 1
        return lc.brownian(r_vecs, periodic_length)

    @property
    def clipped(self):
        '''Pair/wall blocks that were not positive semi-definite in the last draw.'''
        return self.lc.noise_clipped


def make_noise_factor(name, pc, gpu, lc):
    ''''auto' = 'pairwise'; 'cudss' needs the nvmath preconditioner.'''
    if name not in ('auto', 'pairwise', 'cudss', 'cholmod'):
        raise ValueError("noise_backend must be 'auto', 'pairwise', 'cudss' or 'cholmod'")
    if name in ('auto', 'pairwise'):
        return PairwiseNoise(lc)
    if name == 'cholmod':
        return CholmodFactor(gpu)
    if not isinstance(pc, PCNvmath):
        raise ValueError("noise_backend='cudss' needs pc_backend='nvmath'")
    return CuDSSNoiseFactor(pc, gpu)


# =============================================================================
# 4. GMRES implementations
# =============================================================================
# name -> default restart length
GMRES_DEFAULT_RESTART = {'cupy': 10, 'scipy_gpu': 300, 'scipy': 300}


def gmres_solve(method, matvec, psolve, b, x0, tol, restart, its_out, gpu):
    '''
    Solve A x = b (b normalised to unit norm) until ||b - A x|| <= tol.
      'cupy'      cupyx.scipy.sparse.linalg.gmres
      'scipy_gpu' gmres_scipy_port (scipy's algorithm on device arrays)
      'scipy'     scipy.sparse.linalg.gmres on the host, operators on the GPU
    Returns (x, info, iterations or None, operator applications, residuals).
    '''
    n = b.size
    count = [0]

    def mv(x):
        count[0] += 1
        return matvec(x)

    restart = min(restart, n)
    if method == 'cupy':
        x, info, it, res = _gmres_cupy(mv, psolve, b, x0, tol, restart, its_out, gpu)
    elif method == 'scipy_gpu':
        x, info, it, res = gmres_scipy_port(mv, psolve, b, x0, tol=tol, restart=restart,
                                            maxiter=min(its_out, n), xp=gpu.xp,
                                            to_host=gpu.to_host)
    elif method == 'scipy':
        x, info, it, res = _gmres_scipy_host(mv, psolve, b, x0, tol, restart,
                                             min(its_out, n), gpu)
    else:
        raise ValueError(f'unknown GMRES method {method!r}')
    return x, info, it, count[0], res


def _cupy_gmres_tolerance_kwargs(gmres, tol):
    '''Absolute tolerance for cupyx gmres across versions:
    CuPy <= 13: gmres(A, b, x0, tol=1e-5, ..., atol=None) -> atol = max(atol, tol |b|)
    CuPy >= 14: gmres(A, b, x0, *, rtol=1e-5, atol=0.0)   -> atol = max(atol, rtol |b|)'''
    import inspect
    rel = 'rtol' if 'rtol' in inspect.signature(gmres).parameters else 'tol'
    return {rel: 0.0, 'atol': float(tol)}


def _gmres_cupy(matvec, psolve, b, x0, tol, restart, maxiter, gpu):
    '''
    cupyx gmres tests convergence only after a full restart cycle (hence the
    short default restart), and it returns M y while starting from y = x0,
    i.e. it would replace a solution guess x0 by M x0.  A guess is therefore
    used through the residual equation A d = b - A x0, solved from d0 = 0.
    '''
    xp, spla = gpu.xp, gpu.sparse_linalg
    n = b.size
    if x0 is not None:
        b = b - matvec(x0)
        if float(xp.linalg.norm(b)) <= tol:
            return x0.copy(), 0, None, []
    A = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    M = spla.LinearOperator((n, n), matvec=psolve, dtype=np.float64)
    d, info = spla.gmres(A, b, x0=None, M=M, restart=restart, maxiter=maxiter,
                         **_cupy_gmres_tolerance_kwargs(spla.gmres, tol))
    return (d if x0 is None else x0 + d), info, None, []


def _gmres_scipy_host(matvec, psolve, b, x0, tol, restart, maxiter, gpu):
    '''scipy.sparse.linalg.gmres called exactly as in
    pyStokesianDynamics.Lubrication_solve; each operator application moves
    one vector to the device and one back.  Run with one BLAS thread
    (OMP_NUM_THREADS=1): the loop makes many short BLAS calls, and threading
    them over many cores costs far more than it saves.'''
    import scipy.sparse.linalg as spla
    n = b.size

    def on_device(op):
        def f(x):
            return gpu.to_host(op(gpu.to_device(np.ascontiguousarray(x, dtype=np.float64))))
        return f

    A = spla.LinearOperator((n, n), matvec=on_device(matvec), dtype=np.float64)
    M = spla.LinearOperator((n, n), matvec=on_device(psolve), dtype=np.float64)
    res = []
    x, info = spla.gmres(A, gpu.to_host(b), x0=None if x0 is None else gpu.to_host(x0),
                         rtol=0.0, atol=tol, M=M, maxiter=maxiter, restart=restart,
                         callback=res.append, callback_type='pr_norm')
    return gpu.to_device(np.ascontiguousarray(x)), info, len(res), res


def gmres_scipy_port(matvec, psolve, b, x0=None, *, tol, restart, maxiter, xp, to_host,
                     callback=None):
    '''
    scipy.sparse.linalg.gmres (scipy >= 1.12, checked line by line against
    1.18) with the vectors on the device; called with rtol = 0, atol = tol as
    in pyStokesianDynamics.  Same algorithm: left preconditioning, modified
    Gram-Schmidt, LAPACK lartg Givens rotations, inner exit when the
    preconditioned residual estimate presid <= ptol, outer exit on the true
    residual ||b - A x|| <= atol, ptol adaptation (scipy gh-8400).  The small
    Hessenberg problem stays on the host; the MGS coefficients of one column
    are copied to the host together (one transfer per iteration).
    Returns (x, info, inner iterations, preconditioned residuals / |b|).
    '''
    from scipy.linalg import get_lapack_funcs
    n = b.shape[0]
    x = xp.zeros_like(b) if x0 is None else xp.array(x0, dtype=b.dtype, copy=True)
    bnrm2 = float(xp.linalg.norm(b))
    atol = max(float(tol), 0.0 * bnrm2)                    # rtol = 0
    if bnrm2 == 0:
        return xp.array(b, copy=True), 0, 0, []
    eps = np.finfo(np.float64).eps
    restart = min(restart, n)

    Mb_nrm2 = float(xp.linalg.norm(psolve(b)))
    ptol_max_factor = 1.
    ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
    presid = 0.

    lartg = get_lapack_funcs('lartg', dtype=np.float64)
    v = xp.empty((restart + 1, n), dtype=b.dtype)           # Krylov basis (device)
    h = np.zeros((restart, restart + 1))                    # Hessenberg (host, as scipy)
    givens = np.zeros((restart, 2))
    hcol = xp.empty(restart + 2, dtype=b.dtype)             # one column's MGS output
    residuals = []
    inner_iter = 0
    rnorm = np.inf
    for iteration in range(maxiter):
        if iteration == 0:
            r = b - matvec(x) if bool(x.any()) else xp.array(b, copy=True)
            if float(xp.linalg.norm(r)) < atol:            # Are we done?
                return x, 0, 0, residuals

        v[0] = psolve(r)
        tmp = float(xp.linalg.norm(v[0]))
        v[0] *= (1 / tmp)
        # RHS of the Hessenberg problem
        S = np.zeros(restart + 1)
        S[0] = tmp

        breakdown = False
        for col in range(restart):
            av = matvec(v[col])
            w = psolve(av)

            # Modified Gram-Schmidt (coefficients stay on the device)
            hcol[col + 2] = xp.linalg.norm(w)                 # h0
            for k in range(col + 1):
                t = xp.dot(v[k], w)
                hcol[k] = t
                w -= t * v[k]
            hcol[col + 1] = xp.linalg.norm(w)                 # h1
            hc = np.array(to_host(hcol[:col + 3]), dtype=np.float64)
            h0, h1 = hc[col + 2], hc[col + 1]
            h[col, :col + 2] = hc[:col + 2]
            v[col + 1] = w

            # Exact solution indicator
            if h1 <= eps * h0:
                h[col, col + 1] = 0
                breakdown = True
            else:
                v[col + 1] *= (1 / h1)

            # apply past Givens rotations to current h column
            for k in range(col):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = h[col, [k, k + 1]]
                h[col, [k, k + 1]] = [c * n0 + s * n1, -s * n0 + c * n1]

            # get and apply current rotation to h and S
            c, s, mag = lartg(h[col, col], h[col, col + 1])
            givens[col, :] = [c, s]
            h[col, [col, col + 1]] = mag, 0

            # S[col+1] component is always 0
            tmp = -s * S[col]
            S[[col, col + 1]] = [c * S[col], tmp]
            presid = np.abs(tmp)
            inner_iter += 1
            residuals.append(presid / bnrm2)
            if callback is not None:
                callback(presid / bnrm2)

            if presid <= ptol or breakdown:
                break

        # Solve h(col, col) upper triangular system and allow pseudo-solve
        # singular cases (as scipy)
        if h[col, col] == 0:
            S[col] = 0

        y = np.zeros([col + 1])
        y[:] = S[:col + 1]
        for k in range(col, 0, -1):
            if y[k] != 0:
                y[k] /= h[k, k]
                tmp = y[k]
                y[:k] -= tmp * h[k, :k]
        if y[0] != 0:
            y[0] /= h[0, 0]

        x += xp.asarray(y) @ v[:col + 1]

        r = b - matvec(x)
        rnorm = float(xp.linalg.norm(r))

        if rnorm <= atol:
            break
        elif breakdown:
            # Reached breakdown (= exact solution), but the external
            # tolerance check failed. Bail out with failure.
            break
        elif presid <= ptol:
            # Inner loop passed but outer didn't
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

        ptol = presid * min(ptol_max_factor, atol / rnorm)

    info = 0 if (rnorm <= atol) else maxiter
    return x, info, inner_iter, residuals
