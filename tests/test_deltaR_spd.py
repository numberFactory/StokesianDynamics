"""
test_deltaR_spd.py
------------------
Checks that:
  - Delta_R = R_Sup - R_MB is symmetric positive (semi-)definite
  - R_Sup is symmetric positive (semi-)definite

for both the C++ and CuPy (float32 and float64) lubrication implementations.

Tests:
  b) Two particles far from the wall: log-spaced pair separation grid.
  c) One particle: log-spaced wall-height grid.

For each test, the minimum eigenvalue is computed at every grid point.
If all min eigenvalues >= -tol the test passes ([PASS]).
Otherwise a figure is saved showing the min eigenvalue vs separation/height,
with failing points highlighted in red ([FAIL]).
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication as LubricationCPP
import cupy as cp
from lubrication_cupy import Lubrication as LubricationCuPy

# =============================================================================
# Parameters
# =============================================================================
a          = 1.0
eta        = 1.0 / (6.0 * np.pi)
d_cut      = 1e-2      # very small Debye cutoff
N_grid     = 1000
WALL_FAR   = 1.0e3 * a  # height of both particles above wall (test b)
tol        = -1e-10   # eigenvalue tolerance (small negatives from rounding OK)

# log-spaced separation grids
pair_seps  = np.logspace(np.log10(d_cut), 1,              N_grid)  # eps = r-2a
wall_hts   = np.logspace(np.log10(d_cut), np.log10(20),  N_grid)  # eps = h-a

# =============================================================================
# Helpers
# =============================================================================

def to_cpu(M):
    """Convert cupyx CSC to scipy CSC if needed."""
    if hasattr(M, 'get'):
        import scipy.sparse as sp
        return sp.csc_matrix((M.data.get(), M.indices.get(), M.indptr.get()),
                             shape=M.shape)
    return M


def build_r_vecs(positions):
    """List of (3,) arrays."""
    return [np.array(p, dtype=np.float64) for p in positions]


def get_matrices(lc, r_vecs, n_list, a, eta):
    """Return (R_MB, R_Sup, Delta_R) as CPU scipy sparse matrices."""
    pl = np.zeros(3)
    if hasattr(lc, 'dtype'):
        R_MB, R_Sup = lc.ResistCSC_both(r_vecs, n_list, a, eta,
                                        periodic_length=pl)
    else:
        R_MB, R_Sup = lc.ResistCSC_both(r_vecs, n_list, a, eta, pl)
    R_MB  = to_cpu(R_MB)
    R_Sup = to_cpu(R_Sup)
    return R_MB, R_Sup, R_Sup - R_MB


def min_eig(M):
    """
    Minimum eigenvalue of a sparse matrix M, evaluated in float64.
    Returns -inf on NaN/Inf or non-convergence.
    """
    A = M.toarray().astype(np.float64)
    A = 0.5 * (A + A.T)          # symmetrise
    if not np.isfinite(A).all():
        return -np.inf
    try:
        return float(np.linalg.eigvalsh(A).min())
    except np.linalg.LinAlgError:
        return -np.inf


def run_test(label, lc, eps_grid, build_config, test_name, out_prefix,
             matrix_name):
    """
    Run an SPD test for one matrix (Delta_R or R_Sup) over a grid.

    matrix_name : 'Delta_R' or 'R_Sup'
    """
    tag = f"{label}[{matrix_name}]"
    print(f"  {tag:<28} ... ", end='', flush=True)
    min_eigs = np.empty(len(eps_grid))

    for i, eps in enumerate(eps_grid):
        r_vecs, n_list = build_config(eps)
        R_MB, R_Sup, dR = get_matrices(lc, r_vecs, n_list, a, eta)
        M = dR if matrix_name == 'Delta_R' else R_Sup
        min_eigs[i] = min_eig(M)

    failing  = min_eigs < tol
    nan_pts  = ~np.isfinite(min_eigs)
    if not failing.any():
        print(f"[PASS]  (min eig = {min_eigs.min():.3e})")
        return True

    n_fail       = failing.sum()
    n_nan        = nan_pts.sum()
    nan_str      = f", {n_nan} non-finite" if n_nan > 0 else ""
    finite_vals  = min_eigs[np.isfinite(min_eigs)]
    finite_min   = finite_vals.min() if finite_vals.size > 0 else np.nan
    fail_eps     = eps_grid[failing]
    smallest_fail = fail_eps.min()
    largest_fail  = fail_eps.max()
    print(f"[FAIL]  ({n_fail}/{len(eps_grid)} pts negative{nan_str}, "
          f"min eig = {finite_min:.3e}, "
          f"gap ∈ [{smallest_fail:.3e}, {largest_fail:.3e}])")

    x_label = 'pair gap ε = r/a − 2' if test_name == 'pair' \
              else 'wall gap ε = h/a − 1'
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(eps_grid[~failing], min_eigs[~failing],
                'b.', ms=3, label='positive')
    ax.semilogx(eps_grid[failing],  min_eigs[failing],
                'r.', ms=5, label='negative (FAIL)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'min eigenvalue of {matrix_name}')
    ax.set_title(f'{label} {matrix_name} — SPD check ({test_name})')
    ax.legend()
    fig.tight_layout()
    fname = f'{out_prefix}_{test_name}_{matrix_name}_fail.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"         figure saved: {fname}")
    return False


def run_spd_tests(label, lc, eps_grid, build_config, test_name, out_prefix):
    """Run both Delta_R and R_Sup SPD tests, return True only if both pass."""
    ok_dr  = run_test(label, lc, eps_grid, build_config, test_name,
                      out_prefix, 'Delta_R')
    ok_sup = run_test(label, lc, eps_grid, build_config, test_name,
                      out_prefix, 'R_Sup')
    return ok_dr and ok_sup


def pair_config(eps):
    """Two particles at height WALL_FAR/a, separated by 2a + eps in x."""
    r      = 2.0 * a + eps
    r_vecs = build_r_vecs([[0.0, 0.0, WALL_FAR], [r, 0.0, WALL_FAR]])
    n_list = [np.array([1], dtype=np.int32), np.array([], dtype=np.int32)]
    return r_vecs, n_list


def wall_config(eps):
    """Single particle at height a + eps above the wall."""
    r_vecs = build_r_vecs([[0.0, 0.0, a + eps]])
    n_list = [np.array([], dtype=np.int32)]
    return r_vecs, n_list


# =============================================================================
# Solvers to test
# =============================================================================
print("Initialising solvers ...")
lc_cpp    = LubricationCPP(d_cut)
lc_cp_f64 = LubricationCuPy(d_cut, dtype=cp.float64)
lc_cp_f32 = LubricationCuPy(d_cut, dtype=cp.float32)

solvers = [
    ("C++      ", lc_cpp,    "cpp"),
    ("CuPy f64 ", lc_cp_f64, "cupy_f64"),
    ("CuPy f32 ", lc_cp_f32, "cupy_f32"),
]

out_dir  = os.path.dirname(os.path.abspath(__file__))
all_pass = True

# =============================================================================
# (b) Pair test
# =============================================================================
print()
print("=" * 65)
print("(b)  PAIR SPD TEST — two particles far from wall")
print(f"     wall height = {WALL_FAR:.0e}a, "
      f"gap ε ∈ [{pair_seps[0]:.1e}, {pair_seps[-1]:.1e}], "
      f"N={N_grid}")
print("=" * 65)

for label, lc, prefix in solvers:
    ok = run_spd_tests(label, lc, pair_seps, pair_config, 'pair',
                       os.path.join(out_dir, prefix))
    all_pass &= ok

# =============================================================================
# (c) Wall test
# =============================================================================
print()
print("=" * 65)
print("(c)  WALL SPD TEST — single particle, varying height")
print(f"     height ε ∈ [{wall_hts[0]:.1e}, {wall_hts[-1]:.1e}], "
      f"N={N_grid}")
print("=" * 65)

for label, lc, prefix in solvers:
    ok = run_spd_tests(label, lc, wall_hts, wall_config, 'wall',
                       os.path.join(out_dir, prefix))
    all_pass &= ok

print()
print("=" * 65)
print("All tests passed." if all_pass else "Some tests FAILED — see figures.")
print("=" * 65)
