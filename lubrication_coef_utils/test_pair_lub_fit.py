"""
test_pair_fit.py
----------------
Reads rational fit coefficients from pair_sup_scalar_fits_and_cutoffs.txt,
evaluates the fitted pair resistance matrix over a range of separations,
and compares to the C++ Lubrication class output.

Efficiency: powers of epsilon and log(1/eps) are computed once per
separation, then reused across all 10 scalar evaluations.

Usage:
    python test_pair_fit.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from libMobility import NBody

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication

# =============================================================================
# Parameters
# =============================================================================
a      = 1.0
eta    = 1.0/6.0/np.pi
d_cut  = 1e-4
cutoff = 4.5

solver = NBody("open", "open", "single_wall")
solver.setParameters(wallHeight=0.0)
solver.initialize(viscosity=eta, hydrodynamicRadius=a,
                               includeAngular=True)

def form_rpy_mobility(r1,r2):
    Id = np.eye(12)
    positions = np.vstack((r1,r2))
    solver.setPositions(positions.flatten())
    Mob = np.zeros((12, 12))
    for i in range(12):
        FT = Id[:,i].reshape(2, 6)
        F  = FT[:, 0:3].flatten()
        T  = FT[:, 3:6].flatten()

        U, W = solver.Mdot(forces=F, torques=T)

        UW = np.concatenate((U.reshape(2, 3),
                             W.reshape(2, 3)), axis=1)
        Mob[:, i] = UW.flatten()
    return Mob

mob_factor = np.array([
    6.0 * np.pi * eta * a,
    6.0 * np.pi * eta * a**2,
    6.0 * np.pi * eta * a**3,
])
f0, f1, f2 = mob_factor

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
z_safe          = 1e3 * a

# =============================================================================
# Load fit coefficients
# =============================================================================
fit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pair_sup_scalar_fits_and_cutoffs_higher_order.txt')

scalar_names = ['X11A','X12A','Y11A','Y12A','Y11B','Y12B',
                'X11C','X12C','Y11C','Y12C']

fits = {}   # name -> {singular, crossover, p_coeffs, c_coeffs}
with open(fit_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        name     = parts[0]
        singular = parts[1] == 'True'
        crossover= float(parts[2])
        coeffs   = np.array([float(x) for x in parts[3:]])
        n_num, n_den = 6, 5
        fits[name] = {
            'singular' : singular,
            'crossover': crossover,
            'p'        : coeffs[:n_num],
            'c'        : coeffs[n_num:],
        }

print(f"Loaded fits for: {list(fits.keys())}")

# =============================================================================
# Asymptotic formulas (AT region, for eps < crossover)
# =============================================================================
def asym_scalar(name, eps, li):
    """Evaluate AT asymptotic. eps and li=log(1/eps) are pre-computed arrays."""
    d = {
        'X11A':  0.995419  + 0.25/eps  + 0.225*li   + 0.0267857*eps*li,
        'X12A': -0.350153  - 0.25/eps  - 0.225*li   - 0.0267857*eps*li,
        'Y11A':  0.998317  + 0.166667*li,
        'Y12A': -0.273652  - 0.166667*li,
        'Y11B': (-0.666667)*( 0.23892    - 0.25*li  - 0.125*eps*li),
        'Y12B': ( 0.666667)*(-0.00162268 + 0.25*li  + 0.125*eps*li),
        'X11C': (1.33333)*( 1.0518    - 0.125*eps*li),
        'X12C': (1.33333)*(-0.150257  + 0.125*eps*li),
        'Y11C': (1.33333)*( 0.702834  + 0.2*li    + 0.188*eps*li),
        'Y12C': (1.33333)*(-0.027464  + 0.05*li   + 0.062*eps*li),
    }
    return d[name]

# =============================================================================
# Evaluate all 10 scalars at a single epsilon efficiently.
# Powers eps^0..eps^5 and c^2 terms are computed once and shared.
# =============================================================================
def eval_all_scalars(eps_val):
    """
    Evaluate all 10 fitted scalars at a single separation eps_val.
    Returns dict name -> scalar value.
    """
    li = np.log(1.0 / max(eps_val, 1e-300))

    # pre-compute eps powers once: [eps^0, eps^1, ..., eps^5]
    n_num = 6
    eps_pows = np.array([eps_val**i for i in range(n_num)])

    # pre-compute Q denominator terms once per coefficient set
    # Q(eps) = 1 + sum(c_i^2 * eps^(i+1), i=0..4)
    n_den   = 5
    eps_q   = np.array([eps_val**(i+1) for i in range(n_den)])

    result = {}
    for name in scalar_names:
        f  = fits[name]
        co = f['crossover']

        if eps_val < co:
            result[name] = asym_scalar(name, eps_val, li)
        else:
            p   = f['p']
            c   = f['c']
            P   = np.dot(p, eps_pows)
            Q   = 1.0 + np.dot(c**2, eps_q)

            if f['singular']:
                S = 0.25/eps_val if name == 'X11A' else -0.25/eps_val
                result[name] = S + P / (Q * eps_val)
            else:
                result[name] = P / Q

    return result

# =============================================================================
# Build 12x12 pair resistance matrix from scalars (r_hat = [1, 0, 0])
# =============================================================================
def build_R_pair(scalars):
    """
    Assemble the 12x12 pair resistance matrix for two particles along x-axis.
    r_hat = [1, 0, 0] so:
      squeeze = diag(1,0,0)
      shear   = diag(0,1,1)
      vort    = [[0,0,0],[0,0,-1],[0,1,0]]
    """
    X11A = scalars['X11A']; Y11A = scalars['Y11A']
    Y11B = scalars['Y11B']; X11C = scalars['X11C']; Y11C = scalars['Y11C']
    X12A = scalars['X12A']; Y12A = scalars['Y12A']
    Y12B = scalars['Y12B']; X12C = scalars['X12C']; Y12C = scalars['Y12C']

    # 3x3 building blocks (r_hat=[1,0,0])
    def TT_self(XA, YA):    # f0 * (XA*squeeze + YA*shear)
        return f0 * np.diag([XA, YA, YA])

    def TT_cross(XA, YA):
        return f0 * np.diag([XA, YA, YA])

    def TR(YB, sign):       # sign * f1 * YB * vort
        v = np.array([[0,0,0],[0,0,-1],[0,1,0]], dtype=float)
        return sign * f1 * YB * v

    def RR_self(XC, YC):    # f2 * (XC*squeeze + YC*shear)
        return f2 * np.diag([XC, YC, YC])

    def RR_cross(XC, YC):
        return f2 * np.diag([XC, YC, YC])

    R = np.zeros((12, 12))

    # jj block (0:6, 0:6)
    R[0:3, 0:3] = TT_self(X11A, Y11A)
    R[0:3, 3:6] = TR(Y11B, -1)
    R[3:6, 0:3] = TR(Y11B, +1)
    R[3:6, 3:6] = RR_self(X11C, Y11C)

    # kk block (6:12, 6:12)
    R[6:9,  6:9]  = TT_self(X11A, Y11A)
    R[6:9,  9:12] = TR(Y11B, +1)
    R[9:12, 6:9]  = TR(Y11B, -1)
    R[9:12, 9:12] = RR_self(X11C, Y11C)

    # jk block (0:6, 6:12)
    R[0:3, 6:9]  = TT_cross(X12A, Y12A)
    R[0:3, 9:12] = TR(Y12B, -1)
    R[3:6, 6:9]  = TR(Y12B, -1)
    R[3:6, 9:12] = RR_cross(X12C, Y12C)

    # kj block (6:12, 0:6)
    R[6:9,  0:3] = TT_cross(X12A, Y12A)
    R[6:9,  3:6] = TR(Y12B, +1)
    R[9:12, 0:3] = TR(Y12B, +1)
    R[9:12, 3:6] = RR_cross(X12C, Y12C)

    return R

# =============================================================================
# Compute over a range of separations and compare to C++ output
# =============================================================================
# np.array([2.505e-01]) #
gaps    = np.logspace(np.log10(d_cut * 1.01), np.log10(cutoff - 2.0 - 1e-6), 500)
r_norms = gaps + 2.0

lub = Lubrication(d_cut)

rel_errors = np.zeros(len(gaps))
bad_eps = []
bad_norms = []

for idx, (eps_val, r_norm) in enumerate(zip(gaps, r_norms)):
    # --- fitted matrix ---
    scalars = eval_all_scalars(eps_val)
    R_sup_fit   = build_R_pair(scalars)

    # --- C++ matrix ---
    r1 = np.array([0.0,      0.0, z_safe], dtype=np.float64)
    r2 = np.array([r_norm*a, 0.0, z_safe], dtype=np.float64)
    n_list = [np.array([1], dtype=np.int32),
              np.array([],  dtype=np.int32)]
    R_sup_cpp = lub.ResistCSC([r1, r2], n_list, a, eta,
                          cutoff, 0.0, periodic_length, True).toarray()
    
    R_mb_cpp  = lub.ResistCSC([r1, r2], n_list, a, eta,
                            cutoff, 0.0, periodic_length, False).toarray()
    
    M_rpy_libm = form_rpy_mobility(r1, r2)
    R_rpy_libm = np.linalg.pinv(M_rpy_libm)
    
    # print each 6x6 block for debugging
    #print(f"\n=== Separation eps={eps_val:.3e} (r_norm={r_norm:.3f}) ===")
    # only print 4 digits for readability
    #np.set_printoptions(precision=4, suppress=True)
    # print("Fitted R (12x12):")
    # print(R_sup_fit[0:6, 0:6])
    # print("\nC++ R (12x12):")
    # print(R_sup_cpp[0:6, 0:6])
    # print("\nDifference (mb - RPY):")
    # print(np.linalg.norm(R_mb_cpp-R_rpy_libm)/np.linalg.norm(R_rpy_libm))

    # eigs_fit = np.linalg.eigvalsh(R_sup_fit-R_rpy_libm)
    # eigs_cpp = np.linalg.eigvalsh(R_sup_cpp-R_rpy_libm)
    # print("\nEigenvalues of (fit - MB):")
    # print(eigs_fit)
    # print("\nEigenvalues of (C++ - MB):")
    # print(eigs_cpp)
    # sys.exit(0)

    # compare only the 12x12 pair block (wall contribution is suppressed)
    norm_cpp = np.linalg.norm(R_sup_cpp, ord=2)
    rel_errors[idx] = (np.linalg.norm(R_sup_fit - R_sup_cpp, ord=2) /
                       (norm_cpp + 1e-300))
    
    DeltaR_fit = R_sup_fit - R_mb_cpp # R_rpy_libm
    DeltaR_cpp = R_sup_cpp - R_mb_cpp
    # check if positive definite
    eigvals = np.linalg.eigvalsh(DeltaR_fit)
    if np.any(eigvals <= 0):
        print(f"Warning: R_sup_fit - R_mb_cpp is not positive definite at eps={eps_val:.3e}")
        print(f"  Min eigenvalue: {eigvals.min():.3e}")
        bad_eps.append(eps_val)
        bad_norms.append(rel_errors[idx])

    eigvals = np.linalg.eigvalsh(DeltaR_cpp)
    if np.any(eigvals <= 0):
        print(f"[CPP] Warning: R_sup_cpp - R_mb_cpp is not positive definite at eps={eps_val:.3e}")
        print(f"  Min eigenvalue: {eigvals.min():.3e}")

# =============================================================================
# Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(gaps, rel_errors, lw=2.5)
ax.scatter(bad_eps, bad_norms, color='red', marker='x', label='Non-PD fit (R_sup_fit - R_mb_cpp)')
ax.axhline(1e-3, color='k', lw=1.5, linestyle=':', label='$10^{-3}$')
ax.axhline(1e-6, color='k', lw=1.5, linestyle='--', label='$10^{-6}$')
ax.axvline(d_cut, color='grey', lw=1.5, linestyle='--',
           label=f'training boundary ($\\epsilon={d_cut}$)')
ax.set_xlabel(r'$\epsilon = r/a - 2$', fontsize=13)
ax.set_ylabel(r'$\|R_\mathrm{fit} - R_\mathrm{C++}\|_2 \;/\; \|R_\mathrm{C++}\|_2$',
              fontsize=12)
ax.set_title('Pair resistance matrix: rational fit vs C++ lubrication', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nMax relative error: {rel_errors.max():.3e}")
print(f"Mean relative error: {rel_errors.mean():.3e}")
