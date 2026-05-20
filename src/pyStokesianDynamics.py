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
                 z_max, cutoff=4.5, debye_length=1e-4, allowChangingBoxSize=False):
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
        self.cutoff          = cutoff
        self.cutoff_wall     = 1.0e10
        self.debye_length    = debye_length

        self.Delta_R = None
        self.R_MB    = None
        self.R_Sup   = None

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
        self.solver.initialize(viscosity=eta, hydrodynamicRadius=a, includeAngular=True)

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
                    while r_vec[i] > L[i]:
                        r_vec[i] -= L[i]
        return r_vecs

    # TODO: neighborlist is ~30% of this calc, the rest is the ResistCSC_both call
    def Set_R_Mats(self, r_vecs_np=None):
        '''
        Build lubrication resistance matrices in sparse CSC format.
        Sets self.R_MB, self.R_Sup, and self.Delta_R = R_Sup - R_MB.
        '''
        if r_vecs_np is None:
            r_vecs_np = [b.location for b in self.bodies]
        r_vecs = list(self.put_r_vecs_in_periodic_box(
            r_vecs_np, self.periodic_length))
        r_vecs = [np.asarray(r, dtype=np.float64) for r in r_vecs]

        num_particles = len(r_vecs)
        small = 0.5 * 6.0 * np.pi * self.eta * self.a * self.tolerance

        self.solver.setPositions(np.array(r_vecs).flatten())

        r_tree = spatial.cKDTree(np.array(r_vecs), boxsize=self.periodic_length,
                                 balanced_tree=False, compact_nodes=False)
        neighbors = []
        for j in range(num_particles):
            idx = r_tree.query_ball_point(r_vecs[j], r=self.cutoff * self.a)
            neighbors.append(np.array([i for i in idx if i > j], dtype=np.int32))

        self.R_MB, self.R_Sup = self.LC.ResistCSC_both(
            r_vecs, neighbors, self.a, self.eta,
            self.cutoff, self.cutoff_wall, self.periodic_length)

        if self.R_MB.nnz == 0:
            self.R_MB  = sp.diags(small * np.ones(6 * num_particles), 0, format='csc')
        if self.R_Sup.nnz == 0:
            self.R_Sup = sp.diags(small * np.ones(6 * num_particles), 0, format='csc')

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