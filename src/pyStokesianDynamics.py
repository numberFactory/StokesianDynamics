'''
Class to handle Lubrication solve
'''
import numpy as np
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
from functools import partial
import copy
import time
import sys
import pyamg
import scipy.sparse as sp
from sksparse.cholmod import cholesky
from StokesianDynamics import Lubrication
from libMobility import NBody, DPStokes
from numba import jit, njit, prange


class pyStokesianDynamics(object):
    '''
    Class to handle Lubrication solve
    '''

    def __init__(self, bodies, a, eta, periodic_length,
                 z_max, debye_length=1e-4, allowChangingBoxSize=False):
        '''
        Constructor. Initialises lubrication and libMobility solver objects.
        '''
        self.bodies          = bodies
        self.periodic_length = periodic_length
        self.tolerance       = 1e-4
        self.eta             = eta
        self.a               = a
        self.kT              = 0.0041419464
        self.dt              = 1.0
        self.cutoff          = 4.5 # DO NOT CHANGE, FITS ARE HARD-CODED TO THIS VALUE
        self.debye_length    = debye_length

        self.Delta_R  = None
        self.R_MB     = None
        self.R_Sup    = None
        self.isolated = []   # particles far from wall and all neighbours
        self.vel_last = None

        # Initialize perturbation matrix diagonal
        small_F = 6.0 * np.pi * self.eta * self.a          *self.tolerance
        small_T = 8.0 * np.pi * self.eta * self.a**3       *self.tolerance

        self.small_diag = np.tile(
            np.array([small_F, small_F, small_F,
                    small_T, small_T, small_T]),
            len(self.bodies)
        )

        self.LC = Lubrication(debye_length)

        # Initialise libMobility solver
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
        Project a vector r to the minimal image representation
        centered around (0,0,0) and of size L=(Lx, Ly, Lz). If
        any dimension of L is equal or smaller than zero the
        box is assumed to be infinite in that direction.
        '''
        if L is None:
            exit()
        for i in range(3):
            if L[i] > 0:
                r[i] = r[i] - int(r[i] / L[i] + 0.5 *
                                  (int(r[i] > 0) - int(r[i] < 0))) * L[i]
        return r

    def put_r_vecs_in_periodic_box(self, r_vecs_np, L):
        r_vecs = np.copy(r_vecs_np)
        for r_vec in r_vecs:
            for i in range(3):
                if L[i] > 0:
                    while r_vec[i] < 0:
                        r_vec[i] += L[i]
                    while r_vec[i] >= L[i]:
                        r_vec[i] -= L[i]
        return r_vecs

    # TODO: the neighbour list is 30% of this function. The rest is building the sparse matrices.
    def Set_R_Mats(self, r_vecs_np=None):
        '''
        Build lubrication resistance matrices in sparse CSC format.
        Sets self.R_MB, self.R_Sup, self.Delta_R, and self.isolated.
        '''
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = list(self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length))
        r_vecs = [np.asarray(r, dtype=np.float64) for r in r_vecs]

        num_particles = len(r_vecs)

        self.solver.setPositions(np.array(r_vecs).flatten())

        r_tree = spatial.cKDTree(np.array(r_vecs), boxsize=self.periodic_length,
                                 balanced_tree=False, compact_nodes=False)

        # build neighbour list (upper triangle) and isolated particle list
        neighbors     = []
        self.isolated = []
        for j in range(num_particles):
            idx   = r_tree.query_ball_point(r_vecs[j], r=self.cutoff * self.a)
            upper = [i for i in idx if i > j]
            neighbors.append(np.array(upper, dtype=np.int32))
            # isolated: above wall cutoff height and no pair neighbours
            if r_vecs[j][2] >= self.cutoff * self.a and not upper:
                self.isolated.append(j)

        self.R_MB, self.R_Sup = self.LC.ResistCSC_both(
            r_vecs, neighbors, self.a, self.eta, self.periodic_length)

        if self.R_MB.nnz == 0:
            self.R_MB  = sp.diags(self.small_diag, 0, format='csc')
        if self.R_Sup.nnz == 0:
            self.R_Sup = sp.diags(self.small_diag, 0, format='csc')

        self.Delta_R = self.R_Sup - self.R_MB

    def Wall_Mobility_Mult(self, X, r_vecs_np=None):
        '''
        Multiply a vector X of forces and torques by the RPB mobility.
        X should be formatted as [F_1 T_1 F_2 T_2 ...]^T.
        '''
        if r_vecs_np is not None:
            self.solver.setPositions(np.array(r_vecs_np).flatten())

        num_particles = len(self.bodies)
        FT = X.reshape(num_particles, 6)
        F  = FT[:, 0:3].flatten()
        T  = FT[:, 3:6].flatten()

        U, W = self.solver.Mdot(forces=F, torques=T)

        UW = np.concatenate((U.reshape(num_particles, 3),
                             W.reshape(num_particles, 3)), axis=1)
        return UW.flatten()

    def IpMDR_Mult(self, X):
        '''
        Returns (I + M_RPY * Delta_R) * X
        '''
        D_R       = self.Delta_R.dot(X)
        M_Delta_R = self.Wall_Mobility_Mult(D_R)
        return X + M_Delta_R

    def IpMDR_PC(self, X, R_fact=None):
        '''
        Returns (R_fact)^{-1} * X except for particles in self.isolated,
        which return X unchanged.
        '''
        RHS = self.R_MB.dot(X)
        Y_F = R_fact(RHS)
        # for k in self.isolated:
        #     RHS[6*k:6*k+6] = 0.0
        # Y_F = R_fact(RHS)
        # for k in self.isolated:
        #     Y_F[6*k:6*k+6] = X[6*k:6*k+6]
        return Y_F

    def Lubrication_solve(self, X, Xm, X0=None, print_residual=False, its_out=1000):
        '''
        Solve the lubrication problem using GMRES.
        Computes U = [I + M_RPY * Delta_R]^{-1} * (X + M * Xm).
        Requires Set_R_Mats() to have been called first.
        '''
        if self.Delta_R is None:
            self.Set_R_Mats()

        num_particles = len(self.bodies)

        RHS = np.zeros(6 * num_particles)
        if Xm is not None:
            RHS += self.Wall_Mobility_Mult(Xm)
        if X is not None:
            RHS += X.ravel()

        RHS_norm = np.linalg.norm(RHS)
        if RHS_norm > 0:
            RHS = RHS / RHS_norm

        Eig_Shift_R_Sup = self.R_Sup + sp.diags(self.small_diag, 0, format='csc')
        
        factor = cholesky(Eig_Shift_R_Sup)

        PC = spla.LinearOperator(
            (6 * num_particles, 6 * num_particles),
            matvec=partial(self.IpMDR_PC, R_fact=factor), dtype='float64')

        A = spla.LinearOperator(
            (6 * num_particles, 6 * num_particles),
            matvec=self.IpMDR_Mult, dtype='float64')

        if X0 is not None:
            X0 = X0 / RHS_norm

        res_list = []
        U_gmres, info = pyamg.krylov.fgmres(
            A, RHS, x0=X0, tol=self.tolerance, M=PC,
            maxiter=min(its_out, A.shape[0]),
            restart=min(300, A.shape[0]),
            residuals=res_list)

        if RHS_norm > 0:
            U_gmres = U_gmres * RHS_norm

        if print_residual:
            print(f'GMRES: {len(res_list)} iterations, '
                  f'info={info}, final residual={res_list[-1]:.3e}')

        return U_gmres
    

    def Lub_Mobility_Root_RHS(self):
        '''
        Returns RHS_Xm = sqrt(2kT/dt) * Delta_R^{1/2} * W1
            and RHS_X  = sqrt(2kT/dt) * M^{1/2} * W
        for use in Lubrication_solve to compute the square root of the
        lubrication-corrected mobility.
        '''
        num_particles       = len(self.bodies)
        Dim                 = 6*num_particles
        W1                  = np.random.randn(Dim)
        prefactor           = np.sqrt(2.0 * self.kT / self.dt)

        # Delta_R^{1/2} * W1 via Cholesky of shifted Delta_R
        Eig_Shift_DR = self.Delta_R + sp.diags(
            self.small_diag, 0, format='csc')


        factor  = cholesky(Eig_Shift_DR)
        DRhalf  = factor.apply_Pt(factor.L().dot(W1))

        # M^{1/2} * W via sqrtMdotW — W is generated internally by libMobility
        sqrtM_W_U, sqrtM_W_W = self.solver.sqrtMdotW()
        sqrtM_W = np.concatenate(
            (sqrtM_W_U.reshape(num_particles, 3),
             sqrtM_W_W.reshape(num_particles, 3)), axis=1
        ).flatten()

        return prefactor * DRhalf, prefactor * sqrtM_W
    
    def Update_Bodies_Trap(self, FT_calc, stochastic=True, print_residual=False):
        '''
        Predictor-corrector (trapezoidal) timestep update.
        stochastic=True  : includes Brownian noise and drift terms.
        stochastic=False : purely deterministic, force-driven.
        Returns (reject_wall, reject_jump).
        '''
        L = self.periodic_length

        # save initial configuration into _old slots
        for b in self.bodies:
            np.copyto(b.location_old, b.location)
            b.orientation_old = copy.copy(b.orientation)

        # wrapped positions for predictor
        r_vecs_np = [b.location for b in self.bodies]
        r_vecs    = self.put_r_vecs_in_periodic_box(r_vecs_np, L)

        # ── FT_calc (predictor) ───────────────────────────────────────────
        _t0 = time.perf_counter()
        FT  = FT_calc(self.bodies, r_vecs).flatten()
        self.timings['ft_calc'] += time.perf_counter() - _t0

        # ── stochastic RHS ────────────────────────────────────────────────
        if stochastic:
            _t0 = time.perf_counter()
            Root_Xm, Root_X = self.Lub_Mobility_Root_RHS()
            MXm   = self.Wall_Mobility_Mult(Root_Xm)
            Mhalf = Root_X + MXm

            div_U, div_W = self.solver.divM()
            div_UW = np.concatenate(
                (div_U.reshape(len(self.bodies), 3),
                 div_W.reshape(len(self.bodies), 3)), axis=1).flatten()
            D_M = 2.0 * self.kT * div_UW
            self.timings['stochastic'] += time.perf_counter() - _t0
        else:
            Mhalf = None
            D_M   = None

        # ── predictor solve ───────────────────────────────────────────────
        _t0   = time.perf_counter()
        vel_p = self.Lubrication_solve(X=Mhalf, Xm=FT, X0=self.vel_last, print_residual=print_residual)
        self.timings['solve_pred'] += time.perf_counter() - _t0

        # update current slots from _old + predictor velocity
        for k, b in enumerate(self.bodies):
            b.location    = b.location_old.copy()
            b.orientation = copy.copy(b.orientation_old)
            b.update(vel_p[6*k:6*k+3] * self.dt,
                     vel_p[6*k+3:6*k+6] * self.dt,
                     target='current')

        # ── Set_R_Mats at corrector positions ─────────────────────────────
        r_vecs_np_c = [b.location for b in self.bodies]
        r_vecs_c    = self.put_r_vecs_in_periodic_box(r_vecs_np_c, L)
        _t0 = time.perf_counter()
        self.Set_R_Mats(r_vecs_np=r_vecs_c)
        self.timings['set_r_mats'] += time.perf_counter() - _t0

        # ── FT_calc (corrector) ───────────────────────────────────────────
        _t0  = time.perf_counter()
        FT_C = FT_calc(self.bodies, r_vecs_c).flatten()
        self.timings['ft_calc'] += time.perf_counter() - _t0

        # ── corrector solve ───────────────────────────────────────────────
        RHS_X_C = (D_M + Mhalf) if stochastic else None
        _t0   = time.perf_counter()
        vel_c = self.Lubrication_solve(X=RHS_X_C, Xm=FT_C, X0=vel_p, print_residual=print_residual)
        self.vel_last = vel_c
        self.timings['solve_corr'] += time.perf_counter() - _t0

        # trapezoidal average → write into _new slots
        vel_trap = 0.5 * (vel_c + vel_p)
        for k, b in enumerate(self.bodies):
            b.location_new    = b.location_old.copy()
            b.orientation_new = copy.copy(b.orientation_old)
            b.update(vel_trap[6*k:6*k+3] * self.dt,
                     vel_trap[6*k+3:6*k+6] * self.dt,
                     target='new')

        reject_wall, reject_jump = self.Check_Update_With_Jump_Trap()
        self.num_rejections_wall += reject_wall
        self.num_rejections_jump += reject_jump

        # accept or reject
        if (reject_wall + reject_jump) == 0:
            for b in self.bodies:
                b.update(np.zeros(3), np.zeros(3), target='current')
                np.copyto(b.location, b.location_new)
                b.orientation = copy.copy(b.orientation_new)
        else:
            for b in self.bodies:
                np.copyto(b.location, b.location_old)
                b.orientation = copy.copy(b.orientation_old)

        # ── Set_R_Mats for next step ──────────────────────────────────────
        _t0 = time.perf_counter()
        self.Set_R_Mats()
        self.timings['set_r_mats'] += time.perf_counter() - _t0

        self._n_steps_timed += 1
        return reject_wall, reject_jump

    def print_timings(self):
        '''Print mean per-step timings for Update_Bodies_Trap components.'''
        n = max(self._n_steps_timed, 1)
        total = sum(self.timings.values())
        print(f"\n--- Update_Bodies_Trap timings ({n} steps) ---")
        for key, val in self.timings.items():
            print(f"  {key:<14s}  {val/n*1e3:8.2f} ms/step  "
                  f"({100*val/total:.1f}%)")
        print(f"  {'total':<14s}  {total/n*1e3:8.2f} ms/step")
    
    
    def Check_Update_With_Jump_Trap(self):
        '''
        Reject a timestep if any particle crosses the wall (z < 0) or
        moves more than 2a in a single step.
        Returns (reject_wall, reject_jump).
        '''
        for b in self.bodies:
            if b.location_new[2] < 0:
                print("Rejected timestep: wall crossing.")
                return 1, 0

            r    = self.project_to_periodic_image(
                b.location_new - b.location_old, self.periodic_length)
            disp = np.linalg.norm(r)
            if disp > 4.0 * self.a:
                print(f"Rejected timestep: large jump ({disp:.4f} > {4*self.a:.4f}).")
                return 0, 1

        return 0, 0