"""
check_pair_scalars.py
---------------------
Calls the C++ Lubrication class to extract pair resistance scalars and
compares them against the rational fits stored in the coefficient file.
No new fitting is done — this is purely a visual diagnostic.

Usage:
    python check_pair_scalars.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication

# =============================================================================
# Parameters
# =============================================================================
a      = 1.0
eta    = 1.0 / 6.0 / np.pi
d_cut  = 1e-4
cutoff = 4.5

mob_factor = np.array([
    6.0 * np.pi * eta * a,
    6.0 * np.pi * eta * a**2,
    6.0 * np.pi * eta * a**3,
])
f0, f1, f2 = mob_factor

gaps    = np.logspace(np.log10(d_cut * 1.01), np.log10(cutoff - 2.0 - 1e-6), 500)
r_norms = gaps + 2.0
eps     = gaps

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
z_safe          = 1e3 * a
lub             = Lubrication(d_cut)

# =============================================================================
# Extract scalars from C++ code
# =============================================================================
X11A = np.zeros(len(r_norms)); Y11A = np.zeros(len(r_norms))
Y11B = np.zeros(len(r_norms)); X11C = np.zeros(len(r_norms))
Y11C = np.zeros(len(r_norms)); X12A = np.zeros(len(r_norms))
Y12A = np.zeros(len(r_norms)); Y12B = np.zeros(len(r_norms))
X12C = np.zeros(len(r_norms)); Y12C = np.zeros(len(r_norms))

for idx, r_norm in enumerate(r_norms):
    r1 = np.array([0.0,      0.0, z_safe], dtype=np.float64)
    r2 = np.array([r_norm*a, 0.0, z_safe], dtype=np.float64)
    n_list = [np.array([1], dtype=np.int32),
              np.array([],  dtype=np.int32)]
    R = lub.ResistCSC([r1, r2], n_list, a, eta,
                      cutoff, 0.0, periodic_length, True).toarray()
    X11A[idx] = R[0, 0]  / f0;  Y11A[idx] = R[1, 1]  / f0
    Y11B[idx] = R[1, 5]  / f1;  X11C[idx] = R[3, 3]  / f2
    Y11C[idx] = R[4, 4]  / f2;  X12A[idx] = R[0, 6]  / f0
    Y12A[idx] = R[1, 7]  / f0;  Y12B[idx] = R[1, 11] / f1
    X12C[idx] = R[3, 9]  / f2;  Y12C[idx] = R[4, 10] / f2

data_dict = {
    'X11A': X11A, 'Y11A': Y11A, 'Y11B': Y11B,
    'X11C': X11C, 'Y11C': Y11C, 'X12A': X12A,
    'Y12A': Y12A, 'Y12B': Y12B, 'X12C': X12C, 'Y12C': Y12C,
}

# =============================================================================
# AT asymptotic formulas
# =============================================================================
def asym_scalar(name, e):
    li = np.log(1.0 / np.maximum(e, 1e-300))
    d = {
        'X11A':  0.995419  + 0.25/e    + 0.225*li   + 0.0267857*e*li,
        'X12A': -0.350153  - 0.25/e    - 0.225*li   - 0.0267857*e*li,
        'Y11A':  0.998317  + 0.166667*li,
        'Y12A': -0.273652  - 0.166667*li,
        'Y11B': (-0.666667)*( 0.23892    - 0.25*li  - 0.125*e*li),
        'Y12B': ( 0.666667)*(-0.00162268 + 0.25*li  + 0.125*e*li),
        'X11C': (1.33333)*( 1.0518    - 0.125*e*li),
        'X12C': (1.33333)*(-0.150257  + 0.125*e*li),
        'Y11C': (1.33333)*( 0.702834  + 0.2*li    + 0.188*e*li),
        'Y12C': (1.33333)*(-0.027464  + 0.05*li   + 0.062*e*li),
    }
    return d[name]

# =============================================================================
# Load fit coefficients and evaluate the blended fit
# =============================================================================
fit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pair_sup_scalar_fits_and_cutoffs_higher_order.txt')

scalar_names = ['X11A','X12A','Y11A','Y12A','Y11B','Y12B',
                'X11C','X12C','Y11C','Y12C']

fits_loaded = {}
with open(fit_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts    = line.split()
        name     = parts[0]
        singular = parts[1] == 'True'
        crossover= float(parts[2])
        coeffs   = np.array([float(x) for x in parts[3:]])
        n_num, n_den = 6, 5
        fits_loaded[name] = {
            'singular' : singular,
            'crossover': crossover,
            'p'        : coeffs[:n_num],
            'c'        : coeffs[n_num:],
        }

print(f"Loaded fits for: {list(fits_loaded.keys())}")

def eval_fit(name, e):
    """Evaluate the blended fit for scalar 'name' at epsilon array e."""
    e   = np.asarray(e, dtype=float)
    fit = fits_loaded[name]
    co  = fit['crossover']
    p   = fit['p']
    c   = fit['c']
    n_num = len(p)
    n_den = len(c)

    # rational fit evaluation (vectorised)
    P = sum(p[i] * e**i for i in range(n_num))
    Q = 1.0 + sum(c[i]**2 * e**(i+1) for i in range(n_den))

    if fit['singular']:
        # X11A: S_sing = +0.25/eps,  X12A: S_sing = -0.25/eps
        S_sing = 0.25/e if name == 'X11A' else -0.25/e
        rat_val = S_sing + P / (Q * e)
    else:
        rat_val = P / Q

    asym_val = asym_scalar(name, e)
    return np.where(e >= co, rat_val, asym_val)

fit_dict = {name: eval_fit(name, eps) for name in scalar_names}

# =============================================================================
# Plot helpers
# =============================================================================
lw_d   = 3.0
lw_fit = 2.0
lw_at  = 1.5

pairs_A = [('$X_{11}^A$','X11A'), ('$Y_{11}^A$','Y11A'),
           ('$X_{12}^A$','X12A'), ('$Y_{12}^A$','Y12A')]
pairs_B = [('$Y_{11}^B$','Y11B'), ('$Y_{12}^B$','Y12B')]
pairs_C = [('$X_{11}^C$','X11C'), ('$Y_{11}^C$','Y11C'),
           ('$X_{12}^C$','X12C'), ('$Y_{12}^C$','Y12C')]
all_pairs = pairs_A + pairs_B + pairs_C

groups = [
    ('A scalars (TT)',          pairs_A),
    ('B scalars (TR coupling)', pairs_B),
    ('C scalars (RR)',          pairs_C),
    ('All scalars',             all_pairs),
]

# =============================================================================
# Figure 1: C++ data vs fit vs asymptotic
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('C++ scalars (solid) vs rational fit (dotted) vs AT asymptotic (dashed)',
              fontsize=13)

for ax, (title, pairs) in zip(axes1.flat, groups):
    for label, name in pairs:
        ln, = ax.loglog(eps, np.abs(data_dict[name]), lw=lw_d,  label=label)
        ax.loglog(eps, np.abs(fit_dict[name]),  lw=lw_fit,
                  color=ln.get_color(), linestyle=':')
        ax.loglog(eps, np.abs(asym_scalar(name, eps)), lw=lw_at,
                  color=ln.get_color(), linestyle='--')
    ax.plot([], [], 'k-',  lw=lw_d,  label='C++ code')
    ax.plot([], [], 'k:',  lw=lw_fit, label='fit (file)')
    ax.plot([], [], 'k--', lw=lw_at,  label='AT asymptotic')
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel('|scalar|',    fontsize=12)
    ax.set_title(title,          fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()

# =============================================================================
# Figure 2: relative error between C++ and fit
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Relative error: |C++ - fit| / |C++|', fontsize=14)

for ax, (title, pairs) in zip(axes2.flat, groups):
    for label, name in pairs:
        rel_err = np.abs((data_dict[name] - fit_dict[name]) /
                         (np.abs(data_dict[name]) + 1e-30))
        ax.loglog(eps, rel_err, lw=2.5, label=label)
    ax.axhline(1e-3, color='k', lw=1.5, linestyle=':', label='$10^{-3}$')
    ax.axhline(1e-6, color='k', lw=1.5, linestyle='--', label='$10^{-6}$')
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel('relative error', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'scalar':>8}  {'max |C++ - fit| / |C++|':>25}  {'crossover':>12}")
print("-" * 52)
for name in scalar_names:
    rel = np.max(np.abs((data_dict[name] - fit_dict[name]) /
                        (np.abs(data_dict[name]) + 1e-30)))
    co  = fits_loaded[name]['crossover']
    print(f"{name:>8}  {rel:>25.3e}  {co:>12.4e}")

plt.show()
