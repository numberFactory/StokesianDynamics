"""
test_lubrication.py
--------------------
Brief test of the Lubrication nanobind module.
Creates two particles above a wall at z=0 and computes the lubrication
resistance COO matrix entries, then prints a summary.
"""

import numpy as np
import sys
sys.path.insert(0, '.')   # ensure the built module is found if not installed

from Lubrication_Class import Lubrication

# =============================================================================
# Physical parameters
# =============================================================================
a        = 1.0          # particle radius
eta      = 1.0          # fluid viscosity
d_cut    = 1e-4         # Debye cutoff (minimum gap before clamping)
cutoff   = 4.0          # pair lubrication cutoff in units of a (centre-centre)
wall_cutoff = 4.0       # wall lubrication cutoff in units of a (centre-wall)

# Periodic box: no periodicity (set to 0)
periodic_length = np.array([0.0, 0.0, 0.0])

# =============================================================================
# Particle positions: two particles above the wall at z=0
# Particle 1: directly above wall
# Particle 2: nearby, slightly further from wall
# Both within lubrication range of each other and the wall
# =============================================================================
z1 = 1.05 * a     # just above wall (centre-wall gap = 0.05a)
z2 = 1.10 * a     # slightly higher

r1 = np.array([0.0,    0.0, z1])
r2 = np.array([2.05*a, 0.0, z2])   # centre-centre distance ~ 2.05a (near contact)

r_vectors = [r1, r2]

# Neighbour list: particle 0 sees particle 1 and vice versa
n_list = [
    np.array([1], dtype=np.int32),   # neighbours of particle 0
    np.array([0], dtype=np.int32),   # neighbours of particle 1
]

# =============================================================================
# Initialise lubrication object
# =============================================================================
print("Initialising Lubrication object...")
lub = Lubrication(d_cut)
print("Done.\n")

# =============================================================================
# Test 1: Full pair + wall resistance matrix (Stokesian Dynamics / Sup)
# =============================================================================
data_sup, rows_sup, cols_sup = [], [], []

lub.ResistCOO(
    r_vectors, n_list,
    a, eta, cutoff, wall_cutoff,
    periodic_length,
    True,             # Sup_if_true = True -> use Stokesian Dynamics scalars
    data_sup, rows_sup, cols_sup
)

data_sup = np.array(data_sup)
rows_sup = np.array(rows_sup)
cols_sup = np.array(cols_sup)

print("=== Stokesian Dynamics (Sup) resistance matrix ===")
print(f"  Number of nonzero entries : {len(data_sup)}")
print(f"  Max |value|               : {np.max(np.abs(data_sup)):.6e}")
print(f"  Min |value|               : {np.min(np.abs(data_sup)):.6e}")
print(f"  Row range                 : [{rows_sup.min()}, {rows_sup.max()}]")
print(f"  Col range                 : [{cols_sup.min()}, {cols_sup.max()}]\n")

# =============================================================================
# Test 2: Wall-only resistance matrix
# =============================================================================
data_wall, rows_wall, cols_wall = [], [], []

lub.ResistCOO_wall(
    r_vectors,
    a, eta, wall_cutoff,
    periodic_length,
    True,             # Sup scalars
    data_wall, rows_wall, cols_wall
)

data_wall = np.array(data_wall)
rows_wall = np.array(rows_wall)
cols_wall = np.array(cols_wall)

print("=== Wall-only resistance matrix ===")
print(f"  Number of nonzero entries : {len(data_wall)}")
print(f"  Max |value|               : {np.max(np.abs(data_wall)):.6e}")
print(f"  Min |value|               : {np.min(np.abs(data_wall)):.6e}\n")

# =============================================================================
# Test 3: Single pair resistance matrix printout via ResistPairSup_py
# =============================================================================
r_vec   = r2 - r1
r_norm  = np.linalg.norm(r_vec) / a
r_hat   = r_vec / np.linalg.norm(r_vec)

print("=== Single pair resistance matrix (ResistPairSup_py) ===")
print(f"  Centre-centre distance : {r_norm:.4f} a")
print(f"  r_hat                  : {r_hat}")
lub.ResistPairSup_py(r_norm, a, eta, r_hat)

# =============================================================================
# Test 4: Reconstruct sparse matrix and check symmetry
# =============================================================================
from scipy.sparse import coo_matrix

n_dof  = 2 * 6   # 2 particles x 6 DOF each
R_sup  = coo_matrix((data_sup, (rows_sup, cols_sup)), shape=(n_dof, n_dof)).toarray()

sym_err = np.max(np.abs(R_sup - R_sup.T))
print(f"\n=== Symmetry check (Sup full matrix) ===")
print(f"  Max |R - R^T| : {sym_err:.6e}")
if sym_err < 1e-10:
    print("  PASS: matrix is symmetric.")
else:
    print("  WARNING: matrix is not symmetric — check sign conventions.")