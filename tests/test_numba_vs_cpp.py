"""
test_numba_vs_cpp.py
--------------------
Compares the Numba lubrication implementation against the C++ nanobind
implementation for:
  a) Matrix correctness: max absolute and relative differences in R_MB,
     R_Sup, and Delta_R.
  b) Performance: wall-clock timing over n_trials random configurations.

Uses a standalone Set_R_Mats that mirrors pyStokesianDynamics.Set_R_Mats
but accepts both a C++ Lubrication object and the Numba Lubrication object,
so the same neighbour list is passed to both.
"""
import sys
import os
import time
import numpy as np
import scipy.sparse as sp
from scipy import spatial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication as LubricationCPP
from lubrication_numba  import Lubrication as LubricationNumba

# =============================================================================
# Parameters
# =============================================================================
N          = 50        # particles per trial (keep small for correctness check)
N_perf     = 10000       # particles for performance test
a          = 1.0
eta        = 1.0 / (6.0 * np.pi)
phi        = 0.3
z0         = 1.5 * a
z_max      = 10.0 * a
cutoff     = 4.5
cutoff_wall= 4.5       # h/a at which wall corrections are applied
d_cut      = 1e-2      # Debye cutoff
n_trials   = 10        # trials for timing
rng        = np.random.default_rng(0)

# =============================================================================
# Helpers
# =============================================================================
def make_positions(n, phi, z0, periodic):
    L_xy = np.sqrt(n * np.pi * a**2 / phi)
    if periodic:
        xy = rng.uniform(0.0, L_xy, size=(n, 2))
    else:
        xy = rng.uniform(-L_xy / 2, L_xy / 2, size=(n, 2))
    z = np.full(n, z0)
    return [np.array([xy[i,0], xy[i,1], z[i]]) for i in range(n)], L_xy


def build_neighbour_list(r_vecs, cutoff_dist, periodic_length):
    """Upper-triangle neighbour list (i < j pairs within cutoff_dist)."""
    boxsize = periodic_length if np.all(periodic_length > 0) else None
    tree    = spatial.cKDTree(np.array(r_vecs), boxsize=boxsize,
                              balanced_tree=False, compact_nodes=False)
    neighbors = []
    for j in range(len(r_vecs)):
        idx   = tree.query_ball_point(r_vecs[j], r=cutoff_dist)
        upper = np.array([i for i in idx if i > j], dtype=np.int32)
        neighbors.append(upper)
    return neighbors


def set_r_mats(lc, r_vecs, neighbors, a, eta, cutoff, cutoff_wall,
               periodic_length):
    """
    Standalone Set_R_Mats: calls ResistCSC_both and returns (R_MB, R_Sup, Delta_R).
    Works with both the C++ and Numba Lubrication objects.
    """
    r_vecs_f = [np.asarray(r, dtype=np.float64) for r in r_vecs]
    pl       = np.asarray(periodic_length, dtype=np.float64)
    R_MB, R_Sup = lc.ResistCSC_both(
        r_vecs_f, neighbors, a, eta, cutoff, cutoff_wall, pl)
    n = len(r_vecs)
    small = 0.5 * 6.0 * np.pi * eta * a * 1e-4
    if R_MB.nnz  == 0: R_MB  = sp.diags(small*np.ones(6*n), 0, format='csc')
    if R_Sup.nnz == 0: R_Sup = sp.diags(small*np.ones(6*n), 0, format='csc')
    Delta_R = R_Sup - R_MB
    return R_MB, R_Sup, Delta_R


def sparse_max_diff(A, B):
    """Max absolute and relative difference between two sparse matrices."""
    D    = (A - B).tocsr()
    D.eliminate_zeros()
    if D.nnz == 0:
        return 0.0, 0.0
    abs_diff = np.max(np.abs(D.data))
    ref      = np.max(np.abs(A.data)) if A.nnz > 0 else 1.0
    return abs_diff, abs_diff / (ref + 1e-30)


def summarise(name, arr_ms):
    print(f"  {name:<28} {arr_ms.mean():>9.2f} ± {arr_ms.std():>7.2f}"
          f"  [{arr_ms.min():>8.2f}, {arr_ms.max():>8.2f}] ms")


# =============================================================================
# Instantiate both solvers
# =============================================================================
print("Instantiating C++ Lubrication ...")
lc_cpp  = LubricationCPP(d_cut)

print("Instantiating Numba Lubrication (first call triggers JIT) ...")
t0 = time.perf_counter()
lc_nb   = LubricationNumba(d_cut)
print(f"  JIT warmup: {(time.perf_counter()-t0)*1e3:.0f} ms\n")

# =============================================================================
# (a) Correctness check — small N, single trial, both periodic and open
# =============================================================================
print("=" * 60)
print("(a)  CORRECTNESS  —  N={}, one trial".format(N))
print("=" * 60)

for label, periodic in [("periodic", True), ("open", False)]:
    positions, L_xy = make_positions(N, phi, z0, periodic)
    pl = np.array([L_xy, L_xy, 0.0]) if periodic else np.zeros(3)
    neighbors = build_neighbour_list(positions, cutoff * a, pl)

    R_MB_cpp,  R_Sup_cpp,  dR_cpp  = set_r_mats(lc_cpp, positions, neighbors,
                                                 a, eta, cutoff, cutoff_wall, pl)
    R_MB_nb,   R_Sup_nb,   dR_nb   = set_r_mats(lc_nb,  positions, neighbors,
                                                 a, eta, cutoff, cutoff_wall, pl)

    print(f"\n  [{label}]")
    for name, A, B in [("R_MB",    R_MB_cpp,  R_MB_nb),
                       ("R_Sup",   R_Sup_cpp,  R_Sup_nb),
                       ("Delta_R", dR_cpp,     dR_nb)]:
        abs_d, rel_d = sparse_max_diff(A, B)
        flag = "OK" if abs_d < 1e-10 else ("WARN" if abs_d < 1e-6 else "FAIL")
        print(f"    {name:<10}  abs_diff={abs_d:.3e}  rel_diff={rel_d:.3e}  [{flag}]")
        print(f"              cpp nnz={A.nnz}  nb nnz={B.nnz}")

# =============================================================================
# (b) Performance — larger N, n_trials trials
# =============================================================================
print()
print("=" * 60)
print(f"(b)  PERFORMANCE  —  N={N_perf}, {n_trials} trials")
print("=" * 60)

for label, periodic in [("periodic", True), ("open", False)]:
    pl_base = None

    t_cpp = []
    t_nb  = []

    for trial in range(n_trials):
        positions, L_xy = make_positions(N_perf, phi, z0, periodic)
        pl = np.array([L_xy, L_xy, 0.0]) if periodic else np.zeros(3)
        neighbors = build_neighbour_list(positions, cutoff * a, pl)

        t0 = time.perf_counter()
        set_r_mats(lc_cpp, positions, neighbors, a, eta, cutoff, cutoff_wall, pl)
        t_cpp.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        set_r_mats(lc_nb,  positions, neighbors, a, eta, cutoff, cutoff_wall, pl)
        t_nb.append((time.perf_counter() - t0) * 1e3)

        if (trial + 1) % max(1, n_trials // 5) == 0:
            print(f"  [{label}] trial {trial+1}/{n_trials} done")

    t_cpp = np.array(t_cpp)
    t_nb  = np.array(t_nb)

    print(f"\n  [{label}]  (mean ± std  [min, max]  ms)")
    summarise("C++ nanobind",  t_cpp)
    summarise("Numba @njit",   t_nb)
    speedup = t_cpp.mean() / t_nb.mean()
    print(f"  Speedup (cpp/numba): {speedup:.2f}×")

print()
print("=" * 60)
print("Tests complete.")
print("=" * 60)
