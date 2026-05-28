"""
test_cupy_cpp.py
----------------
Compares CuPy (float32 and float64) lubrication implementations against
the C++ nanobind implementation for:
  a) Matrix correctness: max absolute and relative differences in R_MB,
     R_Sup, and Delta_R.
  b) Performance at N=1000 and N=10000.
  c) CuPy section-level profiling via profile_lubrication_cupy.

Note: NEIGHBOUR_CUTOFF (4.5a) is used only for neighbour list construction.
The lubrication codes now hardcode their own cutoffs internally.
"""
import sys
import os
import time
import numpy as np
import cupy as cp
from scipy import spatial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics        import Lubrication as LubricationCPP
from lubrication_cupy         import Lubrication as LubricationCuPy
from profile_lubrication_cupy import Lubrication as LubricationCuPyProfile

# =============================================================================
# Parameters
# =============================================================================
N_correct        = 50
N_perf           = 2000
N_large          = 10000
a                = 1.0
eta              = 1.0 / (6.0 * np.pi)
phi              = 0.3
z0               = 1.5 * a
NEIGHBOUR_CUTOFF = 4.5   # used only for cKDTree neighbour list construction
d_cut            = 1e-2
n_trials         = 10
rng              = np.random.default_rng(0)

# =============================================================================
# Helpers
# =============================================================================
def make_positions(n, phi, z0, periodic):
    L_xy = np.sqrt(n * np.pi * a**2 / phi)
    xy   = rng.uniform(0.0, L_xy, size=(n, 2)) if periodic \
           else rng.uniform(-L_xy/2, L_xy/2, size=(n, 2))
    z    = np.full(n, z0)
    return [np.array([xy[i,0], xy[i,1], z[i]]) for i in range(n)], L_xy


def build_neighbour_list(r_vecs, periodic_length):
    """Upper-triangle neighbour list within NEIGHBOUR_CUTOFF."""
    boxsize = periodic_length if np.all(periodic_length > 0) else None
    tree    = spatial.cKDTree(np.array(r_vecs), boxsize=boxsize,
                              balanced_tree=False, compact_nodes=False)
    neighbors = []
    for j in range(len(r_vecs)):
        idx   = tree.query_ball_point(r_vecs[j], r=NEIGHBOUR_CUTOFF * a)
        upper = np.array([i for i in idx if i > j], dtype=np.int32)
        neighbors.append(upper)
    return neighbors


def build_gpu_arrays(r_vecs, n_list, a, periodic_length):
    N        = len(r_vecs)
    r_np     = np.asarray(r_vecs, dtype=np.float64).reshape(N, 3) / a
    nb_sizes = np.array([len(nb) for nb in n_list], dtype=np.int32)
    if nb_sizes.sum() > 0:
        j_np = np.repeat(np.arange(N, dtype=np.int32), nb_sizes)
        k_np = np.concatenate([np.asarray(nb, dtype=np.int32)
                                for nb in n_list if len(nb) > 0])
    else:
        j_np = np.empty(0, dtype=np.int32)
        k_np = np.empty(0, dtype=np.int32)
    return (cp.asarray(r_np), cp.asarray(j_np), cp.asarray(k_np),
            cp.asarray(periodic_length, dtype=cp.float64))


def to_cpu_sparse(M):
    """Convert cupyx CSC to scipy CSC if needed."""
    if hasattr(M, 'get'):
        import scipy.sparse as sp
        return sp.csc_matrix((M.data.get(), M.indices.get(), M.indptr.get()),
                             shape=M.shape)
    return M


def set_r_mats(lc, r_vecs, neighbors, a, eta,
               periodic_length, r_gpu=None, j_gpu=None, k_gpu=None,
               pl_gpu=None):
    """Call ResistCSC_both on any solver, return CPU scipy CSC matrices."""
    r_vecs_f = [np.asarray(r, dtype=np.float64) for r in r_vecs]
    pl       = np.asarray(periodic_length, dtype=np.float64)
    if isinstance(lc, LubricationCuPy):
        R_MB, R_Sup = lc.ResistCSC_both(
            r_vecs_f, neighbors, a, eta,
            r_gpu=r_gpu, j_gpu=j_gpu, k_gpu=k_gpu, pl_gpu=pl_gpu)
    else:
        # C++ nanobind: positional periodic_length
        R_MB, R_Sup = lc.ResistCSC_both(r_vecs_f, neighbors, a, eta, pl)
    R_MB  = to_cpu_sparse(R_MB)
    R_Sup = to_cpu_sparse(R_Sup)
    n     = len(r_vecs)
    import scipy.sparse as sp
    small = 0.5 * 6.0 * np.pi * eta * a * 1e-4
    if R_MB.nnz  == 0: R_MB  = sp.diags(small*np.ones(6*n), 0, format='csc')
    if R_Sup.nnz == 0: R_Sup = sp.diags(small*np.ones(6*n), 0, format='csc')
    Delta_R = R_Sup - R_MB
    return R_MB, R_Sup, Delta_R


def sparse_max_diff(A, B):
    D = (A - B).tocsr(); D.eliminate_zeros()
    if D.nnz == 0: return 0.0, 0.0
    abs_diff = np.max(np.abs(D.data))
    ref      = np.max(np.abs(A.data)) if A.nnz > 0 else 1.0
    return abs_diff, abs_diff / (ref + 1e-30)


def summarise(name, arr_ms):
    print(f"  {name:<28} {arr_ms.mean():>9.2f} ± {arr_ms.std():>7.2f}"
          f"  [{arr_ms.min():>8.2f}, {arr_ms.max():>8.2f}] ms")


# =============================================================================
# Instantiate solvers
# =============================================================================
print("Instantiating C++ Lubrication ...")
lc_cpp = LubricationCPP(d_cut)

print("Instantiating CuPy Lubrication (float64) ...")
t0 = time.perf_counter()
lc_cp_f64 = LubricationCuPy(d_cut, dtype=cp.float64)
print(f"  GPU warmup: {(time.perf_counter()-t0)*1e3:.0f} ms")

print("Instantiating CuPy Lubrication (float32) ...")
t0 = time.perf_counter()
lc_cp_f32 = LubricationCuPy(d_cut, dtype=cp.float32)
print(f"  GPU warmup: {(time.perf_counter()-t0)*1e3:.0f} ms")

print("Instantiating CuPy Profile Lubrication ...")
lc_cp_prof = LubricationCuPyProfile(d_cut)
print()

solvers = [
    ("C++ nanobind",  lc_cpp,    None),
    ("CuPy float64",  lc_cp_f64, cp.float64),
    ("CuPy float32",  lc_cp_f32, cp.float32),
]

# =============================================================================
# (a) Correctness check
# =============================================================================
print("=" * 65)
print(f"(a)  CORRECTNESS  —  N={N_correct}, one trial")
print("=" * 65)

for label, periodic in [("periodic", True), ("open", False)]:
    positions, L_xy = make_positions(N_correct, phi, z0, periodic)
    pl        = np.array([L_xy, L_xy, 0.0]) if periodic else np.zeros(3)
    neighbors = build_neighbour_list(positions, pl)
    r_gpu, j_gpu, k_gpu, pl_gpu = build_gpu_arrays(positions, neighbors, a, pl)

    # C++ is the reference
    R_MB_ref, R_Sup_ref, dR_ref = set_r_mats(
        lc_cpp, positions, neighbors, a, eta, pl)

    print(f"\n  [{label}]")
    for sname, lc, _ in solvers[1:]:
        R_MB, R_Sup, dR = set_r_mats(
            lc, positions, neighbors, a, eta, pl,
            r_gpu=r_gpu, j_gpu=j_gpu, k_gpu=k_gpu, pl_gpu=pl_gpu)
        print(f"    vs {sname}:")
        for name, A, B in [("R_MB",    R_MB_ref, R_MB),
                            ("R_Sup",   R_Sup_ref, R_Sup),
                            ("Delta_R", dR_ref,    dR)]:
            abs_d, rel_d = sparse_max_diff(A, B)
            flag = "OK" if abs_d < 1e-10 else ("WARN" if abs_d < 1e-6 else "FAIL")
            print(f"      {name:<10} abs={abs_d:.3e}  rel={rel_d:.3e}"
                  f"  nnz cpp={A.nnz} impl={B.nnz}  [{flag}]")

# =============================================================================
# (b) Performance
# =============================================================================
def run_perf(n_particles, n_trials, label_suffix=""):
    print()
    print("=" * 65)
    print(f"(b)  PERFORMANCE  —  N={n_particles}, {n_trials} trials{label_suffix}")
    print("=" * 65)

    for domain_label, periodic in [("periodic", True), ("open", False)]:
        timings = {sname: [] for sname, _, _ in solvers}

        for trial in range(n_trials):
            positions, L_xy = make_positions(n_particles, phi, z0, periodic)
            pl        = np.array([L_xy, L_xy, 0.0]) if periodic else np.zeros(3)
            neighbors = build_neighbour_list(positions, pl)
            r_gpu, j_gpu, k_gpu, pl_gpu = build_gpu_arrays(positions, neighbors, a, pl)

            for sname, lc, _ in solvers:
                is_cupy = isinstance(lc, LubricationCuPy)
                gpu_kw  = dict(r_gpu=r_gpu, j_gpu=j_gpu,
                               k_gpu=k_gpu, pl_gpu=pl_gpu) if is_cupy else {}
                t0 = time.perf_counter()
                set_r_mats(lc, positions, neighbors, a, eta, pl, **gpu_kw)
                timings[sname].append((time.perf_counter() - t0) * 1e3)

            if (trial + 1) % max(1, n_trials // 5) == 0:
                print(f"  [{domain_label}] trial {trial+1}/{n_trials} done")

        print(f"\n  [{domain_label}]  (mean ± std  [min, max]  ms)")
        cpp_mean = np.array(timings["C++ nanobind"]).mean()
        for sname, _, _ in solvers:
            summarise(sname, np.array(timings[sname]))
        for sname, _, _ in solvers[1:]:
            arr   = np.array(timings[sname])
            ratio = arr.mean() / cpp_mean
            tag   = f"{1/ratio:.2f}× faster" if ratio < 1.0 else f"{ratio:.2f}× slower"
            print(f"  {sname} vs C++: {tag}")


run_perf(N_perf,  n_trials)
run_perf(N_large, n_trials=3, label_suffix="  *** large scale ***")

# =============================================================================
# (c) CuPy section profiling
# =============================================================================
def run_profile(n_particles, n_trials, periodic):
    domain = "periodic" if periodic else "open"
    print()
    print("=" * 65)
    print(f"(c)  CUPY PROFILE  —  N={n_particles}, {n_trials} trials  [{domain}]")
    print("=" * 65)

    lc_cp_prof.reset_profile()

    for trial in range(n_trials):
        positions, L_xy = make_positions(n_particles, phi, z0, periodic)
        pl        = np.array([L_xy, L_xy, 0.0]) if periodic else np.zeros(3)
        neighbors = build_neighbour_list(positions, pl)
        r_gpu, j_gpu, k_gpu, pl_gpu = build_gpu_arrays(positions, neighbors, a, pl)
        set_r_mats(lc_cp_prof, positions, neighbors, a, eta, pl,
                   r_gpu=r_gpu, j_gpu=j_gpu, k_gpu=k_gpu, pl_gpu=pl_gpu)
        if (trial + 1) % max(1, n_trials // 5) == 0:
            print(f"  trial {trial+1}/{n_trials} done")

    lc_cp_prof.print_profile()


run_profile(N_perf,  n_trials=10, periodic=True)
run_profile(N_large, n_trials=5,  periodic=True)

print()
print("=" * 65)
print("Tests complete.")
print("=" * 65)
