"""
test_cholesky_solver.py
-----------------------
Tests CholeskySolver against cupyx spsolve for a particle configuration
similar to the lubrication tests, and times each component.
"""
import sys
import os
import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspla
from scipy import spatial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lubrication_cupy  import Lubrication as LubricationCuPy
from cholesky_solver   import CholeskySolver

# =============================================================================
# Parameters
# =============================================================================
N        = 2000
a        = 1.0
eta      = 1.0 / (6.0 * np.pi)
phi      = 0.3
z0       = 1.5 * a
d_cut    = 1e-2
n_trials = 5
rng      = np.random.default_rng(42)

NEIGHBOUR_CUTOFF = 4.5

# =============================================================================
# Helpers
# =============================================================================
def make_positions(n, phi, z0):
    L_xy = np.sqrt(n * np.pi * a**2 / phi)
    xy   = rng.uniform(0.0, L_xy, size=(n, 2))
    z    = np.full(n, z0)
    return [np.array([xy[i,0], xy[i,1], z[i]]) for i in range(n)], L_xy


def build_neighbour_list(r_vecs, periodic_length):
    boxsize = periodic_length if np.all(periodic_length > 0) else None
    tree    = spatial.cKDTree(np.array(r_vecs), boxsize=boxsize,
                              balanced_tree=False, compact_nodes=False)
    neighbors = []
    for j in range(len(r_vecs)):
        idx   = tree.query_ball_point(r_vecs[j], r=NEIGHBOUR_CUTOFF * a)
        upper = np.array([i for i in idx if i > j], dtype=np.int32)
        neighbors.append(upper)
    return neighbors


def build_gpu_arrays(r_vecs, n_list, a, pl):
    n    = len(r_vecs)
    r_np = np.asarray(r_vecs, dtype=np.float64).reshape(n, 3) / a
    nb_sizes = np.array([len(nb) for nb in n_list], dtype=np.int32)
    if nb_sizes.sum() > 0:
        j_np = np.repeat(np.arange(n, dtype=np.int32), nb_sizes)
        k_np = np.concatenate([np.asarray(nb, dtype=np.int32)
                                for nb in n_list if len(nb) > 0])
    else:
        j_np = np.empty(0, dtype=np.int32)
        k_np = np.empty(0, dtype=np.int32)
    return (cp.asarray(r_np), cp.asarray(j_np), cp.asarray(k_np),
            cp.asarray(pl, dtype=cp.float64))


def gpu_event_time(fn):
    """Run fn(), return (result, elapsed_ms) using CUDA Events."""
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    result = fn()
    end.record(); end.synchronize()
    return result, cp.cuda.get_elapsed_time(start, end)


# =============================================================================
# Build R_Sup (CSC)
# =============================================================================
print("=" * 60)
print(f"  CholeskySolver test  —  N={N} particles")
print("=" * 60)

print(f"\nBuilding particle configuration ...")
positions, L_xy = make_positions(N, phi, z0)
pl        = np.array([L_xy, L_xy, 0.0])
neighbors = build_neighbour_list(positions, pl)
r_gpu, j_gpu, k_gpu, pl_gpu = build_gpu_arrays(positions, neighbors, a, pl)

lc = LubricationCuPy(d_cut, dtype=cp.float64)

print(f"Building R_Sup via lubrication_cupy ...")
t0 = time.perf_counter()
_, R_Sup_csc = lc.ResistCSC_both(
    [np.asarray(r, dtype=np.float64) for r in positions],
    neighbors, a, eta,
    r_gpu=r_gpu, j_gpu=j_gpu, k_gpu=k_gpu, pl_gpu=pl_gpu)
cp.cuda.Device().synchronize()
t_build = (time.perf_counter() - t0) * 1e3
n_dof   = R_Sup_csc.shape[0]
print(f"  R_Sup (CSC): shape={R_Sup_csc.shape}, nnz={R_Sup_csc.nnz}, "
      f"build={t_build:.2f} ms")

# =============================================================================
# CSC -> CSR conversion (required by cuDSS / CholeskySolver)
# =============================================================================
print(f"\nConverting CSC -> CSR ...")
t_csc2csr_trials = []
for _ in range(n_trials):
    _, t_ms = gpu_event_time(lambda: R_Sup_csc.tocsr())
    t_csc2csr_trials.append(t_ms)
R_Sup_csr = R_Sup_csc.tocsr()
t_conv = np.array(t_csc2csr_trials)
print(f"  CSC -> CSR:  {t_conv.mean():.3f} ± {t_conv.std():.3f} ms"
      f"  [{t_conv.min():.3f}, {t_conv.max():.3f}]")

# Random RHS vectors
b_single = cp.asarray(rng.standard_normal(n_dof), dtype=cp.float64)
b_mat    = cp.asarray(rng.standard_normal((n_dof, n_trials)), dtype=cp.float64)

# =============================================================================
# (1) CholeskySolver: factorisation timing
# =============================================================================
print(f"\n--- CholeskySolver (input: CSR) ---")

# Warmup
chol = CholeskySolver(R_Sup_csr)

t_factor_trials = []
for _ in range(n_trials):
    _, t_ms = gpu_event_time(lambda: CholeskySolver(R_Sup_csr))
    t_factor_trials.append(t_ms)
chol = CholeskySolver(R_Sup_csr)
t_f  = np.array(t_factor_trials)
print(f"  Factorisation:  {t_f.mean():.2f} ± {t_f.std():.2f} ms"
      f"  [{t_f.min():.2f}, {t_f.max():.2f}]")

# Single RHS solve — steady state
t_solve_trials = []
for _ in range(n_trials):
    _, t_ms = gpu_event_time(lambda: chol.solve(b_single))
    t_solve_trials.append(t_ms)
x_chol = chol.solve(b_single)
t_s = np.array(t_solve_trials)
print(f"  Solve (1 RHS, {n_trials} trials):")
print(f"    {t_s.mean():.3f} ± {t_s.std():.3f} ms"
      f"  [{t_s.min():.3f}, {t_s.max():.3f}]")

# Repeated single RHS with different vectors — matches preconditioner use
t_repeat_trials = []
for trial in range(n_trials):
    b_trial = cp.ascontiguousarray(b_mat[:, trial])
    _, t_ms = gpu_event_time(lambda: chol.solve(b_trial))
    t_repeat_trials.append(t_ms)
t_rep = np.array(t_repeat_trials)
print(f"  Solve (repeated single RHS, varying b, {n_trials} trials):")
print(f"    {t_rep.mean():.3f} ± {t_rep.std():.3f} ms"
      f"  [{t_rep.min():.3f}, {t_rep.max():.3f}]")

# =============================================================================
# (2) Reference: cupyx spsolve
# =============================================================================
print(f"\n--- cupyx spsolve (reference, CSC input) ---")
t_spsolve_trials = []
for _ in range(n_trials):
    _, t_ms = gpu_event_time(
        lambda: cpspla.spsolve(R_Sup_csc.astype(cp.float64), b_single))
    t_spsolve_trials.append(t_ms)
x_spsolve = cpspla.spsolve(R_Sup_csc.astype(cp.float64), b_single)
t_sp = np.array(t_spsolve_trials)
print(f"  spsolve (1 RHS, {n_trials} trials):")
print(f"    {t_sp.mean():.3f} ± {t_sp.std():.3f} ms"
      f"  [{t_sp.min():.3f}, {t_sp.max():.3f}]")

# =============================================================================
# (3) Correctness
# =============================================================================
print(f"\n--- Correctness ---")

diff    = float(cp.linalg.norm(x_chol - x_spsolve))
ref     = float(cp.linalg.norm(x_spsolve))
rel_err = diff / (ref + 1e-30)
flag    = "OK" if rel_err < 1e-6 else ("WARN" if rel_err < 1e-3 else "FAIL")
print(f"  vs spsolve:  abs_err={diff:.3e}  rel_err={rel_err:.3e}  [{flag}]")

residual = R_Sup_csc.astype(cp.float64) @ x_chol - b_single
rel_res  = float(cp.linalg.norm(residual) / cp.linalg.norm(b_single))
flag_r   = "OK" if rel_res < 1e-6 else ("WARN" if rel_res < 1e-3 else "FAIL")
print(f"  Residual ||Ax-b||/||b||: {rel_res:.3e}  [{flag_r}]")

# =============================================================================
# (4) CPU CHOLMOD path — full pipeline starting from GPU CSC
# =============================================================================
print()
print("=" * 60)
print("  CPU CHOLMOD comparison")
print("=" * 60)

import scipy.sparse as sp
from sksparse.cholmod import analyze

# b on CPU for CHOLMOD
b_single_cpu = b_single.get()

# --- GPU CSC -> CPU scipy CSC (transfer) ---
print(f"\nGPU CSC -> CPU scipy CSC transfer ...")
t_gpu2cpu_trials = []
for _ in range(n_trials):
    t0 = time.perf_counter()
    R_cpu = sp.csc_matrix(
        (R_Sup_csc.data.get(), R_Sup_csc.indices.get(), R_Sup_csc.indptr.get()),
        shape=R_Sup_csc.shape)
    t_gpu2cpu_trials.append((time.perf_counter() - t0) * 1e3)
R_cpu = sp.csc_matrix(
    (R_Sup_csc.data.get(), R_Sup_csc.indices.get(), R_Sup_csc.indptr.get()),
    shape=R_Sup_csc.shape)
t_xfer = np.array(t_gpu2cpu_trials)
print(f"  GPU->CPU transfer:  {t_xfer.mean():.3f} ± {t_xfer.std():.3f} ms"
      f"  [{t_xfer.min():.3f}, {t_xfer.max():.3f}]")

# --- CHOLMOD symbolic analysis (done once per sparsity pattern) ---
print(f"\nCHOLMOD symbolic analysis ...")
t_sym_trials = []
for _ in range(n_trials):
    t0 = time.perf_counter()
    sym = analyze(R_cpu)
    t_sym_trials.append((time.perf_counter() - t0) * 1e3)
sym = analyze(R_cpu)
t_sym = np.array(t_sym_trials)
print(f"  Symbolic analysis:  {t_sym.mean():.3f} ± {t_sym.std():.3f} ms"
      f"  [{t_sym.min():.3f}, {t_sym.max():.3f}]")

# --- CHOLMOD numeric factorisation ---
print(f"\nCHOLMOD numeric factorisation ...")
t_num_trials = []
for _ in range(n_trials):
    t0 = time.perf_counter()
    fac = sym.cholesky(R_cpu)
    t_num_trials.append((time.perf_counter() - t0) * 1e3)
fac = sym.cholesky(R_cpu)
t_num = np.array(t_num_trials)
print(f"  Numeric factor:     {t_num.mean():.3f} ± {t_num.std():.3f} ms"
      f"  [{t_num.min():.3f}, {t_num.max():.3f}]")

# --- CHOLMOD solve (single RHS, repeated) ---
print(f"\nCHOLMOD solve ...")
t_chol_solve_trials = []
for trial in range(n_trials):
    b_trial_cpu = b_mat[:, trial].get()
    t0 = time.perf_counter()
    _ = fac.solve_A(b_trial_cpu)
    t_chol_solve_trials.append((time.perf_counter() - t0) * 1e3)
x_cholmod = fac.solve_A(b_single_cpu)
t_chol_s = np.array(t_chol_solve_trials)
print(f"  Solve (1 RHS, {n_trials} trials, varying b):")
print(f"    {t_chol_s.mean():.3f} ± {t_chol_s.std():.3f} ms"
      f"  [{t_chol_s.min():.3f}, {t_chol_s.max():.3f}]")

# --- Correctness vs GPU CholeskySolver ---
print(f"\n--- Correctness (CHOLMOD vs GPU CholeskySolver) ---")
x_chol_cpu = x_chol.get()
diff_cm    = float(np.linalg.norm(x_cholmod - x_chol_cpu))
ref_cm     = float(np.linalg.norm(x_cholmod))
rel_cm     = diff_cm / (ref_cm + 1e-30)
flag_cm    = "OK" if rel_cm < 1e-6 else ("WARN" if rel_cm < 1e-3 else "FAIL")
print(f"  CHOLMOD vs GPU Cholesky: rel_err={rel_cm:.3e}  [{flag_cm}]")

res_cpu   = R_cpu @ x_cholmod - b_single_cpu
rel_res_c = float(np.linalg.norm(res_cpu) / np.linalg.norm(b_single_cpu))
flag_rc   = "OK" if rel_res_c < 1e-6 else ("WARN" if rel_res_c < 1e-3 else "FAIL")
print(f"  CHOLMOD residual ||Ax-b||/||b||: {rel_res_c:.3e}  [{flag_rc}]")

# --- Summary ---
print()
print("=" * 60)
print("  Timing summary")
print("=" * 60)
print(f"  {'Operation':<40} {'GPU (ms)':>10}  {'CPU (ms)':>10}")
print(f"  {'-'*62}")
print(f"  {'Build R_Sup CSC (lubrication_cupy)':<40} {t_build:>10.2f}  {'n/a':>10}")
print(f"  {'CSC -> CSR (GPU, for cuDSS)':<40} {t_conv.mean():>10.3f}  {'n/a':>10}")
print(f"  {'GPU CSC -> CPU CSC (PCIe transfer)':<40} {'n/a':>10}  {t_xfer.mean():>10.3f}")
print(f"  {'Factorisation (cuDSS / CHOLMOD total)':<40} {t_f.mean():>10.2f}  {(t_sym+t_num).mean():>10.3f}")
print(f"    {'  symbolic (CHOLMOD only)':<38} {'n/a':>10}  {t_sym.mean():>10.3f}")
print(f"    {'  numeric':<38} {'n/a':>10}  {t_num.mean():>10.3f}")
print(f"  {'Solve (1 RHS, preconditioner-style)':<40} {t_rep.mean():>10.3f}  {t_chol_s.mean():>10.3f}")
print()
print("=" * 60)
print("Test complete.")
print("=" * 60)
