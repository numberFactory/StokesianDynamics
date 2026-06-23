"""
cupyStokesianDynamics.py
------------------------
GPU-accelerated Stokesian Dynamics solver using lubrication_cupy.

  - Particle positions stored as (N,3) CuPy array on GPU
  - libMobility receives and returns CuPy arrays — no host transfer in Mdot
  - Lubrication matrices (R_MB, R_Sup, Delta_R) live on GPU as cupyx CSC
  - Neighbour list stored as GPU arrays (j_gpu, k_gpu) for fast reuse
  - Preconditioner: GPU cuDSS Cholesky via CholeskySolver (fully on device)
  - GMRES: cupyx gmres on GPU with callback for residuals (fully GPU)
  - sqrt(Delta_R)*W: CPU CHOLMOD, result immediately moved to GPU
"""

import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sp
from sksparse.cholmod import analyze as cholmod_analyze
import copy

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla

from lubrication_cupy import Lubrication as LubricationCuPy
from cholesky_solver  import CholeskySolver
from libMobility import NBody, DPStokes


class _GmresCounter:
    """Callable passed as the GMRES callback; records residual norms."""
    def __init__(self):
        self.niter     = 0
        self.residuals = []

    def __call__(self, rk):
        self.residuals.append(rk)
        self.niter += 1


class cupyStokesianDynamics:
    """
    GPU-accelerated Stokesian Dynamics solver.

    Particle positions are (N,3) CuPy arrays.  libMobility accepts CuPy
    arrays natively — no host transfers in Wall_Mobility_Mult.  GMRES and
    the Cholesky preconditioner solve stay entirely on GPU.
    The Brownian force square-root is computed on CPU (CHOLMOD) then
    moved to GPU.
    """

    def __init__(self, bodies, a, eta, periodic_length,
                 z_max, debye_length=1e-4, allowChangingBoxSize=False):
        self.bodies             = bodies
        self.periodic_length    = cp.asarray(periodic_length, dtype=cp.float64)
        self.periodic_length_np = np.asarray(periodic_length, dtype=np.float64)
        self.tolerance          = 1e-4
        self.eta                = eta
        self.a                  = a
        self.kT                 = 0.0041419464
        self.dt                 = 1.0
        self.debye_length       = debye_length

        # GPU lubrication matrices (cupyx CSC, float64)
        self.R_MB    = None
        self.R_Sup   = None
        self.Delta_R = None
        self.isolated = []

        # Initialize perturbation matrix diagonal
        small_F = 6.0 * cp.pi * self.eta * self.a       * self.tolerance
        small_T = 8.0 * cp.pi * self.eta * self.a**3    * self.tolerance
        self.small_diag = cp.tile(
            cp.array([small_F, small_F, small_F,
                    small_T, small_T, small_T]),
            len(self.bodies)
        )

        # GPU Cholesky preconditioner for R_Sup
        self._chol_pc = None

        # CPU CHOLMOD factor for Delta_R^{1/2}
        self._chol_dr_sym = None   # symbolic (cached while N unchanged)
        self._chol_dr_fac = None   # numeric  (rebuilt each Set_R_Mats)

        # GPU neighbour list arrays
        self._r_gpu  = None   # (N,3) in units of a, float64
        self._j_gpu  = None   # flat pair source indices, int32
        self._k_gpu  = None   # flat pair dest   indices, int32
        self._pl_gpu = None   # periodic_length (3,), float64

        self.LC = LubricationCuPy(debye_length, dtype=cp.float64)

        L = self.periodic_length_np
        if L[0] <= 0 and L[1] <= 0:
            self.solver = NBody("open", "open", "single_wall")
            self.solver.setParameters(wallHeight=0.0)
        else:
            self.solver = DPStokes("periodic", "periodic", "single_wall")
            self.solver.setParameters(
                Lx=L[0], Ly=L[1], zmin=0.0, zmax=z_max,
                allowChangingBoxSize=allowChangingBoxSize)
        self.solver.initialize(viscosity=eta, hydrodynamicRadius=a,
                               includeAngular=True)

        self.num_rejections_wall = 0
        self.num_rejections_jump = 0

    # =========================================================================
    # Periodic box utilities (GPU)
    # =========================================================================
    def put_r_vecs_in_periodic_box_gpu(self, r_gpu):
        """Wrap (N,3) CuPy positions into [0, L) on GPU."""
        r = r_gpu.copy()
        for i in range(3):
            if float(self.periodic_length[i]) > 0.0:
                r[:, i] = r[:, i] - cp.floor(
                    r[:, i] / self.periodic_length[i]) * self.periodic_length[i]
        return r

    def project_to_periodic_image_gpu(self, dr_gpu):
        """Minimum-image displacement vector (3,) on GPU."""
        d = dr_gpu.copy()
        for i in range(3):
            li = float(self.periodic_length[i])
            if li > 0.0:
                d[i] = d[i] - cp.round(d[i] / li) * li
        return d

    # =========================================================================
    # Neighbour list
    # =========================================================================
    def _build_neighbour_list(self, r_gpu):
        """
        Build upper-triangle neighbour list within 4.5a.

        cKDTree runs on CPU (one .get() for the tree build only).
        Flat pair arrays and scaled positions are stored as GPU members.
        libMobility positions updated via CuPy array.

        Parameters
        ----------
        r_gpu : cp.ndarray (N,3) — positions in physical units
        """
        N           = r_gpu.shape[0]
        cutoff_dist = 4.5 * self.a
        L_np        = self.periodic_length_np

        r_np    = r_gpu.get()   # single transfer for cKDTree
        boxsize = L_np if np.all(L_np > 0) else None
        tree    = spatial.cKDTree(r_np, boxsize=boxsize,
                                  balanced_tree=False, compact_nodes=False)

        neighbors = []
        self.isolated = []
        for j in range(N):
            idx   = tree.query_ball_point(r_np[j], r=cutoff_dist)
            upper = [i for i in idx if i > j]
            neighbors.append(np.array(upper, dtype=np.int32))
            if r_np[j, 2] >= cutoff_dist and not upper:
                self.isolated.append(j)

        nb_sizes = np.array([len(nb) for nb in neighbors], dtype=np.int32)
        if nb_sizes.sum() > 0:
            j_np = np.repeat(np.arange(N, dtype=np.int32), nb_sizes)
            k_np = np.concatenate([nb for nb in neighbors if len(nb) > 0])
        else:
            j_np = np.empty(0, dtype=np.int32)
            k_np = np.empty(0, dtype=np.int32)

        self._r_gpu  = (r_gpu / self.a).astype(cp.float64)
        self._j_gpu  = cp.asarray(j_np)
        self._k_gpu  = cp.asarray(k_np)
        self._pl_gpu = self.periodic_length.astype(cp.float64)

        # libMobility accepts CuPy — no host transfer
        self.solver.setPositions(r_gpu.flatten())

        return neighbors

    # =========================================================================
    # Lubrication matrix construction
    # =========================================================================
    def Set_R_Mats(self, r_gpu=None):
        """
        Build GPU lubrication resistance matrices and Cholesky factors.

        Parameters
        ----------
        r_gpu : cp.ndarray (N,3) or None
            Particle positions.  If None, reads from self.bodies.
        """
        if r_gpu is None:
            r_gpu = cp.asarray(
                np.array([b.location for b in self.bodies], dtype=np.float64))

        r_gpu = self.put_r_vecs_in_periodic_box_gpu(r_gpu)
        N     = r_gpu.shape[0]
        n_dof = 6 * N

        neighbors   = self._build_neighbour_list(r_gpu)
        r_vecs_list = [r_gpu[j].get() for j in range(N)]   # fast-path fallback

        R_MB_gpu, R_Sup_gpu = self.LC.ResistCSC_both(
            r_vecs_list, neighbors, self.a, self.eta,
            r_gpu=self._r_gpu, j_gpu=self._j_gpu,
            k_gpu=self._k_gpu, pl_gpu=self._pl_gpu)

        if R_MB_gpu.nnz == 0:
            R_MB_gpu  = cpsp.diags(self.small_diag, 0, format='csc')
        if R_Sup_gpu.nnz == 0:
            R_Sup_gpu = cpsp.diags(self.small_diag, 0, format='csc')

        self.R_MB    = R_MB_gpu
        self.R_Sup   = R_Sup_gpu
        self.Delta_R = R_Sup_gpu - R_MB_gpu

        # ── GPU Cholesky preconditioner (cuDSS) ───────────────────────────
        R_Sup_shifted = R_Sup_gpu + cpsp.diags(
            self.small_diag, 0, format='csc')
        self._chol_pc = CholeskySolver(R_Sup_shifted.tocsr())

        # ── CPU CHOLMOD factor for Delta_R^{1/2} ──────────────────────────
        Delta_R_cpu = sp.csc_matrix(
            (self.Delta_R.data.get(),
             self.Delta_R.indices.get(),
             self.Delta_R.indptr.get()),
            shape=(n_dof, n_dof))

        Delta_R_shifted_cpu = Delta_R_cpu + sp.diags(
            self.small_diag, 0, format='csc')

        if (self._chol_dr_sym is None or
                Delta_R_shifted_cpu.shape != self._chol_dr_sym.D().shape):
            self._chol_dr_sym = cholmod_analyze(Delta_R_shifted_cpu)
        self._chol_dr_fac = self._chol_dr_sym.cholesky(Delta_R_shifted_cpu)

    # =========================================================================
    # Linear operators  (GPU in → GPU out throughout)
    # =========================================================================
    def Wall_Mobility_Mult(self, X_gpu):
        """
        M_RPY * X.  GPU in → GPU out.
        libMobility accepts and returns CuPy — no host transfer.

        X_gpu : cp.ndarray (6N,)  [F1 T1 F2 T2 ...]
        """
        N  = len(self.bodies)
        FT = X_gpu.reshape(N, 6)
        U, W = self.solver.Mdot(forces=FT[:, 0:3].flatten(),
                                torques=FT[:, 3:6].flatten())
        return cp.concatenate(
            (U.reshape(N, 3), W.reshape(N, 3)), axis=1).flatten()

    def IpMDR_Mult(self, X_gpu):
        """(I + M * Delta_R) * X.  GPU in → GPU out."""
        return X_gpu + self.Wall_Mobility_Mult(self.Delta_R @ X_gpu)

    def IpMDR_PC(self, X_gpu):
        """
        Preconditioner: R_Sup^{-1} * R_MB * X, isolated dofs passed through.
        GPU cuDSS Cholesky — stays entirely on device.
        """
        RHS_gpu = self.R_MB @ X_gpu
        Y_gpu = self._chol_pc.solve(RHS_gpu)
        # for k in self.isolated:
        #     RHS_gpu[6*k:6*k+6] = 0.0
        # Y_gpu = self._chol_pc.solve(RHS_gpu)
        # for k in self.isolated:
        #     Y_gpu[6*k:6*k+6] = X_gpu[6*k:6*k+6]
        # return Y_gpu

    # =========================================================================
    # GMRES solve
    # =========================================================================
    def Lubrication_solve(self, X_gpu, Xm_gpu, X0_gpu=None,
                          print_residual=False, its_out=1000):
        """
        Solve (I + M * Delta_R) U = X + M * Xm using GPU GMRES.

        X_gpu, Xm_gpu : cp.ndarray (6N,) or None
        X0_gpu        : cp.ndarray (6N,) or None  — initial guess on GPU
        Returns       : cp.ndarray (6N,)
        """
        if self.Delta_R is None:
            self.Set_R_Mats()

        n_dof = 6 * len(self.bodies)

        RHS_gpu = cp.zeros(n_dof, dtype=cp.float64)
        if Xm_gpu is not None:
            RHS_gpu += self.Wall_Mobility_Mult(Xm_gpu)
        if X_gpu is not None:
            RHS_gpu += X_gpu

        RHS_norm = float(cp.linalg.norm(RHS_gpu))
        if RHS_norm > 0:
            RHS_gpu = RHS_gpu / RHS_norm

        if X0_gpu is not None and RHS_norm > 0:
            X0_gpu = X0_gpu / RHS_norm

        A  = cpspla.LinearOperator((n_dof, n_dof),
                                   matvec=self.IpMDR_Mult, dtype=cp.float64)
        PC = cpspla.LinearOperator((n_dof, n_dof),
                                   matvec=self.IpMDR_PC,   dtype=cp.float64)

        counter = _GmresCounter()
        U_gpu, info = cpspla.gmres(
            A, RHS_gpu, x0=X0_gpu, tol=self.tolerance, M=PC,
            maxiter=its_out,
            restart=min(100, n_dof),
            callback=counter)

        if RHS_norm > 0:
            U_gpu = U_gpu * RHS_norm

        if print_residual:
            last = counter.residuals[-1] if counter.residuals else float('nan')
            print(f'GMRES: {counter.niter} iters, '
                  f'info={info}, residual={last:.3e}')

        return U_gpu

    # =========================================================================
    # Brownian force
    # =========================================================================
    def Lub_Mobility_Root_RHS(self):
        """
        Returns (RHS_Xm, RHS_X) as cp.ndarray on GPU.

        RHS_Xm = sqrt(2kT/dt) * Delta_R^{1/2} * W1   (CPU CHOLMOD → GPU)
        RHS_X  = sqrt(2kT/dt) * M^{1/2} * W           (libMobility GPU → GPU)
        """
        N         = len(self.bodies)
        n_dof     = 6 * N
        prefactor = np.sqrt(2.0 * self.kT / self.dt)

        W1     = np.random.randn(n_dof)
        fac    = self._chol_dr_fac
        DRhalf = fac.apply_Pt(fac.L().dot(W1))

        sqrtM_W_U, sqrtM_W_W = self.solver.sqrtMdotW()
        sqrtM_W = cp.concatenate(
            (sqrtM_W_U.reshape(N, 3), sqrtM_W_W.reshape(N, 3)),
            axis=1).flatten()

        return (cp.asarray(prefactor * DRhalf),
                prefactor * sqrtM_W)

    # =========================================================================
    # Timestep
    # =========================================================================
    def Update_Bodies_Trap(self, FT_calc, stochastic=True):
        """
        Predictor-corrector (trapezoidal) timestep.

        FT_calc(bodies, r_gpu) → cp.ndarray (6N,) forces/torques on GPU.
        Returns (reject_wall, reject_jump).
        """
        for b in self.bodies:
            np.copyto(b.location_old, b.location)
            b.orientation_old = copy.copy(b.orientation)

        r_gpu  = self.put_r_vecs_in_periodic_box_gpu(cp.asarray(
            np.array([b.location for b in self.bodies], dtype=np.float64)))
        FT_gpu = FT_calc(self.bodies, r_gpu)

        if stochastic:
            Root_Xm, Root_X = self.Lub_Mobility_Root_RHS()
            Mhalf_gpu = Root_X + self.Wall_Mobility_Mult(Root_Xm)
            div_U, div_W = self.solver.divM()
            D_M_gpu = 2.0 * self.kT * cp.concatenate(
                (div_U.reshape(len(self.bodies), 3),
                 div_W.reshape(len(self.bodies), 3)), axis=1).flatten()
        else:
            Mhalf_gpu = None
            D_M_gpu   = None

        # predictor
        vel_p_gpu = self.Lubrication_solve(X_gpu=Mhalf_gpu, Xm_gpu=FT_gpu)
        vel_p_np  = vel_p_gpu.get()

        for k, b in enumerate(self.bodies):
            b.location    = b.location_old.copy()
            b.orientation = copy.copy(b.orientation_old)
            b.update(vel_p_np[6*k:6*k+3]   * self.dt,
                     vel_p_np[6*k+3:6*k+6] * self.dt, target='current')

        # rebuild at corrector positions
        r_c_gpu = self.put_r_vecs_in_periodic_box_gpu(cp.asarray(
            np.array([b.location for b in self.bodies], dtype=np.float64)))
        self.Set_R_Mats(r_gpu=r_c_gpu)

        FT_C_gpu = FT_calc(self.bodies, r_c_gpu)
        RHS_X_C  = (D_M_gpu + Mhalf_gpu) if stochastic else None

        vel_c_gpu = self.Lubrication_solve(
            X_gpu=RHS_X_C, Xm_gpu=FT_C_gpu, X0_gpu=vel_p_gpu)
        vel_c_np  = vel_c_gpu.get()

        vel_trap_np = 0.5 * (vel_c_np + vel_p_np)
        for k, b in enumerate(self.bodies):
            b.location_new    = b.location_old.copy()
            b.orientation_new = copy.copy(b.orientation_old)
            b.update(vel_trap_np[6*k:6*k+3]   * self.dt,
                     vel_trap_np[6*k+3:6*k+6] * self.dt, target='new')

        reject_wall, reject_jump = self.Check_Update_With_Jump_Trap()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        if (reject_wall + reject_jump) == 0:
            for b in self.bodies:
                b.update(np.zeros(3), np.zeros(3), target='current')
                np.copyto(b.location, b.location_new)
                b.orientation = copy.copy(b.orientation_new)
        else:
            for b in self.bodies:
                np.copyto(b.location, b.location_old)
                b.orientation = copy.copy(b.orientation_old)

        self.Set_R_Mats()
        return reject_wall, reject_jump

    def Check_Update_With_Jump_Trap(self):
        for b in self.bodies:
            if b.location_new[2] < 0:
                print("Rejected timestep: wall crossing.")
                return 1, 0
            r    = self.project_to_periodic_image_gpu(
                cp.asarray(b.location_new - b.location_old)).get()
            disp = np.linalg.norm(r)
            if disp > 2 * self.a:
                print(f"Rejected timestep: large jump ({disp:.4f}).")
                return 0, 1
        return 0, 0
