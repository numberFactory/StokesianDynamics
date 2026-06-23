import nvmath
import nvmath.sparse.advanced as nvsp
import cupyx.scipy.sparse as cpsp
import cupy as cp
import logging
import os
import glob

# Suppress the nvmath multithreading library warning — we handle it ourselves.
logging.getLogger('nvmath').setLevel(logging.ERROR)


def _find_multithreading_lib():
    """
    Locate cuDSS's prebuilt threading layer shim (libcudss_mtlayer_gomp.so).
    This is NOT libgomp.so itself — cuDSS requires its own wrapper library
    that ships with the cuDSS distribution package.

    Searches the active conda environment and common CUDA installation paths.
    Returns the full path string, or None if not found.
    """
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    cuda_home    = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', ''))

    search_dirs = []
    if conda_prefix:
        search_dirs += [
            os.path.join(conda_prefix, 'lib'),
            os.path.join(conda_prefix, 'lib', 'python3.12', 'site-packages',
                         'nvidia', 'cudss', 'lib'),
        ]
    if cuda_home:
        search_dirs.append(os.path.join(cuda_home, 'lib64'))
    search_dirs += ['/usr/local/cuda/lib64', '/usr/lib/x86_64-linux-gnu']

    # cuDSS ships libcudss_mtlayer_gomp.so (GNU OpenMP shim)
    # and libcudss_mtlayer_omp.so (generic OpenMP shim) on Linux
    shim_names = [
        'libcudss_mtlayer_gomp.so',
        'libcudss_mtlayer_gomp.so.0',
        'libcudss_mtlayer_omp.so',
        'libcudss_mtlayer_omp.so.0',
    ]
    for name in shim_names:
        for d in search_dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    # fallback: glob for any versioned shim, prefer shorter version suffix
    for pat in ['libcudss_mtlayer_gomp.so*', 'libcudss_mtlayer_omp.so*']:
        for d in search_dirs:
            hits = sorted(glob.glob(os.path.join(d, pat)),
                          key=lambda p: len(p))   # shortest = least versioned
            if hits:
                return hits[0]
    return None


_MULTITHREADING_LIB = _find_multithreading_lib()


class CholeskySolver:
    """
    Sparse Cholesky solver via NVIDIA cuDSS (through nvmath-python).

    Accepts a cupyx CSR or CSC SPD matrix. cuDSS requires CSR format;
    if CSC is passed it is converted to CSR automatically (with a copy).
    For best performance, pass CSR directly to avoid the conversion cost.
    """
    def __init__(self, A):
        if isinstance(A, cpsp.csc_matrix):
            A = A.tocsr()
        if not isinstance(A, cpsp.csr_matrix):
            raise TypeError("A must be a cupyx csr_matrix or csc_matrix")
        n = A.shape[0]
        # placeholder RHS — shape only matters for planning
        b_dummy = cp.zeros(n, dtype=cp.float64)
        if _MULTITHREADING_LIB is not None:
            import warnings
            warnings.warn(
                f"CholeskySolver: using threading library {_MULTITHREADING_LIB}",
                RuntimeWarning, stacklevel=2)
        # build stateful solver — this does symbolic factorisation + numeric
        def _make_solver(threading_lib):
            return nvsp.DirectSolver(
                A, b_dummy,
                options=nvsp.DirectSolverOptions(
                    sparse_system_type=nvsp.DirectSolverMatrixType.SPD,
                    sparse_system_view=nvsp.DirectSolverMatrixViewType.FULL,
                    multithreading_lib=threading_lib),
                execution=nvsp.ExecutionCUDA())

        try:
            self._solver = _make_solver(_MULTITHREADING_LIB)
        except Exception:
            # cuDSS rejected the threading library — fall back to none
            self._solver = _make_solver(None)
        self._solver.plan()
        self._solver.factorize()
        self._n = n

    def solve(self, b: cp.ndarray) -> cp.ndarray:
        self._solver.reset_operands(b=cp.ascontiguousarray(b))
        return self._solver.solve()

    def factor(self):
        raise NotImplementedError(
            "cuDSS does not expose the sparse L factor directly. "
            "Use solve() for triangular solves.")