"""
test_lubrication.py
--------------------
Brief test of the Lubrication nanobind module.
Creates two particles above a wall at z=0 and computes the lubrication
resistance CSC matrices, then prints a summary.
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from StokesianDynamics import Lubrication

# =============================================================================
# Physical parameters
# =============================================================================
a               = 1.0
eta             = 1.0
d_cut           = 1e-2
cutoff          = 4.5
wall_cutoff     = 1.0e10
periodic_length = np.array([0.0, 0.0, 0.0])

# =============================================================================
# Particle positions
# =============================================================================
z1 = 1.05 * a
z2 = 1.10 * a
r1 = np.array([0.0, 0.0, z1])
r2 = np.array([2.05 * a, 0.0, z2])
r_vectors = [r1, r2]

n_list = [
    np.array([1], dtype=np.int32),
    np.array([0], dtype=np.int32),
]

n_dof = 2 * 6

# =============================================================================
# Helpers
# =============================================================================
def print_matrix_summary(name, mat):
    dense = mat.toarray()
    print(f"=== {name} ===")
    print(f"  Shape                     : {mat.shape}")
    print(f"  Number of nonzero entries : {mat.nnz}")
    print(f"  Max |value|               : {np.max(np.abs(dense)):.6e}")
    print(f"  Min |value|               : {np.min(np.abs(dense[dense != 0])):.6e}")

def check_symmetry(name, mat):
    err = np.max(np.abs(mat.toarray() - mat.toarray().T))
    print(f"=== Symmetry check: {name} ===")
    print(f"  Max |R - R^T| : {err:.6e}")
    if err < 1e-10:
        print("  PASS: matrix is symmetric.\n")
    else:
        print("  WARNING: matrix is not symmetric.\n")

def check_equal(name_a, name_b, mat_a, mat_b, tol=1e-12):
    err = np.max(np.abs((mat_a - mat_b).toarray()))
    print(f"=== Agreement check: {name_a} vs {name_b} ===")
    print(f"  Max |difference| : {err:.6e}")
    if err < tol:
        print("  PASS: matrices are identical.\n")
    else:
        print("  WARNING: matrices differ.\n")

# =============================================================================
# Initialise
# =============================================================================
print("Initialising Lubrication object...")
lub = Lubrication(d_cut)
print("Done.\n")

# =============================================================================
# Test 1: R_Sup via ResistCSC
# =============================================================================
R_sup = lub.ResistCSC(r_vectors, n_list, a, eta, cutoff, wall_cutoff,
                      periodic_length, True)
print_matrix_summary("R_Sup (ResistCSC)", R_sup)
check_symmetry("R_Sup", R_sup)

# =============================================================================
# Test 2: R_MB via ResistCSC
# =============================================================================
R_mb = lub.ResistCSC(r_vectors, n_list, a, eta, cutoff, wall_cutoff,
                     periodic_length, False)
print_matrix_summary("R_MB (ResistCSC)", R_mb)
check_symmetry("R_MB", R_mb)

# =============================================================================
# Test 3: ResistCSC_both — single-pass combined computation
# =============================================================================
R_mb_both, R_sup_both = lub.ResistCSC_both(r_vectors, n_list, a, eta,
                                            cutoff, wall_cutoff, periodic_length)
print_matrix_summary("R_MB (ResistCSC_both)", R_mb_both)
check_symmetry("R_MB_both", R_mb_both)
print_matrix_summary("R_Sup (ResistCSC_both)", R_sup_both)
check_symmetry("R_Sup_both", R_sup_both)

# =============================================================================
# Test 4: Agreement between ResistCSC and ResistCSC_both
# =============================================================================
check_equal("R_MB (ResistCSC)", "R_MB (ResistCSC_both)",   R_mb,  R_mb_both)
check_equal("R_Sup (ResistCSC)", "R_Sup (ResistCSC_both)", R_sup, R_sup_both)

# =============================================================================
# Test 5: Delta_R
# =============================================================================
Delta_R      = (R_sup      - R_mb     ).toarray()
Delta_R_both = (R_sup_both - R_mb_both).toarray()
print(f"=== Delta_R = R_Sup - R_MB ===")
print(f"  Max |Delta_R|               : {np.max(np.abs(Delta_R)):.6e}")
print(f"  Min |Delta_R|               : {np.min(np.abs(Delta_R)):.6e}")
print(f"  Max |Delta_R - Delta_R_both|: {np.max(np.abs(Delta_R - Delta_R_both)):.6e}")
print(f"\n  Full Delta_R matrix:")
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(Delta_R)

# =============================================================================
# Test 6: Comparison to stored reference
# =============================================================================
Delta_R_ref = np.load("./tests/Delta_R_test.npz")["Delta_R"]
err_ref      = np.max(np.abs(Delta_R      - Delta_R_ref))
err_ref_both = np.max(np.abs(Delta_R_both - Delta_R_ref))
print(f"\n=== Comparison to reference Delta_R ===")
print(f"  Max |Delta_R (ResistCSC)      - ref| : {err_ref:.6e}")
print(f"  Max |Delta_R (ResistCSC_both) - ref| : {err_ref_both:.6e}")

# =============================================================================
# TODO:
# -Replace the coefficients with the working AK coefficients and check against the old code
# -Profile for large systems to see if neighbour list is slowing things down