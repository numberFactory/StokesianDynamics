"""
plot_lub_scalars.py
--------------------
Extracts the scalar resistance coefficients from the Lubrication C++ class,
fits each with a rational function on the training grid [d_cut, cutoff-2],
and switches to the exact AT asymptotic below d_cut.

Usage:
    python plot_lub_scalars.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication

# =============================================================================
# Parameters
# =============================================================================
a      = 1.0
eta    = 1.0/6.0/np.pi
d_cut  = 5.0e-3
cutoff = 4.5

mob_factor = np.array([
    6.0 * np.pi * eta * a,
    6.0 * np.pi * eta * a**2,
    6.0 * np.pi * eta * a**3,
])
f0, f1, f2 = mob_factor

gaps    = np.unique(np.concatenate([
    np.logspace(np.log10(d_cut * 1.01), np.log10(cutoff - 2.0 - 1e-6), 300),
    np.linspace(0.08, 0.25, 200), np.linspace(2.3, 2.5- 1e-6, 20)  # extra density in WS/JO transition only
]))
r_norms = gaps + 2.0
eps     = gaps

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
z_safe          = 1e3 * a
lub             = Lubrication(d_cut)

# =============================================================================
# Extract scalars
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
# Known asymptotic functions (full AT forms from ATResistMatrix in C++)
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

asym_eps = {name: asym_scalar(name, eps) for name in data_dict}

# =============================================================================
# Load existing fit coefficients as initial guesses (if file exists)
# =============================================================================
in_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'pair_sup_scalar_fits_and_cutoffs_higher_order.txt')

p0_dict = {}   # name -> popt array from previous fit
if os.path.isfile(in_path):
    print(f"Loading initial guesses from {in_path}")
    with open(in_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts  = line.split()
            nm     = parts[0]
            coeffs = np.array([float(x) for x in parts[3:]])
            p0_dict[nm] = coeffs
    print(f"  Loaded guesses for: {list(p0_dict.keys())}")
else:
    print("No existing fit file found — using default initial guesses.")

# =============================================================================
# Rational fit: scalar = S(eps) + P(eps)/Q(eps)
# Q(eps) = 1 + sum(c_i^2 * eps^i) > 0 always
# X11A, X12A: subtract 1/eps singularity, scale residual by eps
# All others: fit directly
# =============================================================================
def make_model_pos_Q(n_num, n_den):
    def model(e, *params):
        p = params[:n_num]
        c = params[n_num:]
        P = sum(p[i] * e**i for i in range(n_num))
        Q = 1.0 + sum(c[i]**2 * e**(i+1) for i in range(n_den))
        return P / Q
    return model

def fit_direct(eps, data, name, n_num=6, n_den=5):
    model = make_model_pos_Q(n_num, n_den)
    if name in p0_dict and len(p0_dict[name]) == n_num + n_den:
        p0 = p0_dict[name].copy()
    else:
        p0 = np.zeros(n_num + n_den); p0[0] = np.median(data)
    try:
        popt, _ = curve_fit(model, eps, data, p0=p0,
                            maxfev=200000, method='trf',
                            ftol=1e-12, xtol=1e-12)
    except RuntimeError:
        print("  WARNING: did not converge"); popt = p0
    recon   = model(eps, *popt)
    rel_err = np.max(np.abs((recon - data) / (np.abs(data) + 1e-30)))
    _p, _m  = popt, model
    return (lambda e, m=_m, p=_p: m(e, *p)), recon, rel_err, popt

def fit_with_singular(eps, data, S_sing, name, n_num=6, n_den=5):
    residual = (data - S_sing(eps)) * eps
    model    = make_model_pos_Q(n_num, n_den)
    if name in p0_dict and len(p0_dict[name]) == n_num + n_den:
        p0 = p0_dict[name].copy()
    else:
        p0 = np.zeros(n_num + n_den); p0[0] = np.median(residual)
    try:
        popt, _ = curve_fit(model, eps, residual, p0=p0,
                            maxfev=200000, method='trf',
                            ftol=1e-12, xtol=1e-12)
    except RuntimeError:
        print("  WARNING: did not converge"); popt = p0
    recon   = S_sing(eps) + model(eps, *popt) / eps
    rel_err = np.max(np.abs((recon - data) / (np.abs(data) + 1e-30)))
    _S, _p, _m = S_sing, popt, model
    return (lambda e, s=_S, m=_m, p=_p: s(e) + m(e, *p) / e), recon, rel_err, popt

# =============================================================================
# Find optimal crossover for each scalar: minimise |rat(eps) - asym(eps)|
# on a fine grid, pick the eps where they agree best.
# =============================================================================
eps_cross = np.logspace(np.log10(d_cut * 1.01),
                        np.log10(cutoff - 2.0 - 1e-6), 5000)

def find_crossover(name, rat_fn):
    rat_vals  = rat_fn(eps_cross)
    asym_vals = asym_scalar(name, eps_cross)
    rel_diff  = np.abs(rat_vals - asym_vals) / (np.abs(asym_vals) + 1e-30)
    idx       = np.argmin(rel_diff)
    return eps_cross[idx], rel_diff[idx]

def make_blended(name, rat_fn):
    co, min_err = find_crossover(name, rat_fn)
    print(f"    crossover at eps={co:.4e}, relative mismatch={min_err:.2e}")
    def fn(e):
        e      = np.asarray(e, dtype=float)
        result = np.where(e >= co, rat_fn(e), asym_scalar(name, e))
        return result
    return fn, co

print("Fitting rational functions...")
fits        = {}
callables   = {}
crossovers  = {}
popts       = {}   # store fitted coefficients for saving

for name, data, singular, S_func in [
    ('X11A', X11A, True,  lambda e: 0.25/e),
    ('X12A', X12A, True,  lambda e: -0.25/e),
    ('Y11A', Y11A, False, None), ('Y12A', Y12A, False, None),
    ('Y11B', Y11B, False, None), ('Y12B', Y12B, False, None),
    ('X11C', X11C, False, None), ('X12C', X12C, False, None),
    ('Y11C', Y11C, False, None), ('Y12C', Y12C, False, None),
]:
    print(f"  {name}...", end=' ')
    if singular:
        rat_fn, recon, rel_err, popt = fit_with_singular(eps, data, S_func, name)
    else:
        rat_fn, recon, rel_err, popt = fit_direct(eps, data, name)
    print(f"max relative error = {rel_err:.2e}")
    fits[name]      = recon
    popts[name]     = popt
    blended_fn, co  = make_blended(name, rat_fn)
    callables[name] = blended_fn
    crossovers[name]= co

# =============================================================================
# Plot helpers
# =============================================================================
lw_d   = 3.0
lw_at  = 2.0
lw_fit = 2.0

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
# Figure 1: data (solid) + asymptotic (dashed) + fit (dotted)
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Sup lubrication scalars: data, asymptotic, fit', fontsize=14)

for ax, (title, pairs) in zip(axes1.flat, groups):
    for label, name in pairs:
        ln, = ax.loglog(eps, np.abs(data_dict[name]),  lw=lw_d,  label=label)
        ax.loglog(eps, np.abs(asym_eps[name]),          lw=lw_at,
                  color=ln.get_color(), linestyle='--')
        ax.loglog(eps, np.abs(fits[name]),              lw=lw_fit,
                  color=ln.get_color(), linestyle=':')
    ax.plot([], [], 'k-',  lw=lw_d,  label='data')
    ax.plot([], [], 'k--', lw=lw_at, label='asymptotic')
    ax.plot([], [], 'k:',  lw=lw_fit,label='fit')
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel('|scalar|',    fontsize=12)
    ax.set_title(title,          fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
print("\nFigure 1 ready.")

# =============================================================================
# Figure 2: fit relative error on training grid
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Rational fit residuals: |fit - data| / |data|', fontsize=14)

for ax, (title, pairs) in zip(axes2.flat, groups):
    for label, name in pairs:
        rel_err = np.abs((fits[name] - data_dict[name]) /
                         (np.abs(data_dict[name]) + 1e-30))
        ax.loglog(eps, rel_err, lw=2.5, label=label)
    ax.axhline(1e-3, color='k', lw=1.5, linestyle=':',  label='$10^{-3}$')
    ax.axhline(1e-6, color='k', lw=1.5, linestyle='--', label='$10^{-6}$')
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel('relative error', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
print("Figure 2 ready.")

# =============================================================================
# Figure 3: blended fit vs asymptotic on fine grid [1e-6, 2.5]
# Below d_cut: exact asymptotic. Above d_cut: rational fit.
# =============================================================================
eps_f  = np.logspace(-6, np.log10(cutoff - 2.0 - 1e-6), 2000)
asym_f = {name: asym_scalar(name, eps_f) for name in data_dict}

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle(
    r'Blended fit vs asymptotic ($\epsilon \in [10^{-6}, 2.5]$) — '
    r'crossover at optimal $\epsilon$ per scalar',
    fontsize=11)

for ax, (title, pairs) in zip(axes3.flat, groups):
    for label, name in pairs:
        fit_f = callables[name](eps_f)
        ln,   = ax.loglog(eps_f, np.abs(fit_f),       lw=lw_d,  label=label)
        ax.loglog(eps_f, np.abs(asym_f[name]),          lw=lw_at,
                  color=ln.get_color(), linestyle='--')
        ax.axvline(crossovers[name], color=ln.get_color(),
                   lw=1.0, linestyle=':')
    ax.plot([], [], 'k-',  lw=lw_d,  label='blended fit')
    ax.plot([], [], 'k--', lw=lw_at, label='asymptotic')
    ax.plot([], [], 'k:',  lw=1.0,   label='crossover')
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel('|scalar|',    fontsize=12)
    ax.set_title(title,          fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
print("Figure 3 ready.")

# crossover table
print(f"\n{'scalar':>8}  {'crossover eps':>16}")
print("-" * 28)
for name, co in crossovers.items():
    print(f"{name:>8}  {co:>16.4e}")

# =============================================================================
# Save rational fit coefficients and crossover epsilons to a text file.
#
# Format:
#   One block per scalar, separated by blank lines.
#   Each block begins with "# <name>" followed by key: value pairs:
#     singular   : bool — whether a 1/eps singular part was subtracted
#     crossover  : float — epsilon below which the AT asymptotic is used
#     n_num      : int — number of numerator coefficients P(eps)
#     n_den      : int — number of denominator coefficients c_i (Q > 0 by construction)
#     p_coeffs   : space-separated floats — numerator polynomial coefficients p_0..p_{n_num-1}
#     c_coeffs   : space-separated floats — denominator squared-root coefficients c_0..c_{n_den-1}
#
#   The rational fit evaluates as:
#     For non-singular scalars:  scalar(eps) = P(eps) / Q(eps)
#     For singular scalars (X11A, X12A):
#         scalar(eps) = S_sing(eps) + P(eps) / (Q(eps) * eps)
#     where P(eps) = sum(p_i * eps^i, i=0..n_num-1)
#           Q(eps) = 1 + sum(c_i^2 * eps^(i+1), i=0..n_den-1)
#           S_sing for X11A  = +0.25 / eps
#           S_sing for X12A  = -0.25 / eps
#   Below the crossover epsilon, use the AT asymptotic formula directly.
# =============================================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pair_sup_scalar_fits_and_cutoffs_higher_order.txt')

scalar_specs = [
    ('X11A', True), ('X12A', True),
    ('Y11A', False), ('Y12A', False), ('Y11B', False), ('Y12B', False),
    ('X11C', False), ('X12C', False), ('Y11C', False), ('Y12C', False),
]

with open(out_path, 'w') as f:
    f.write("# Sup lubrication scalar rational fit coefficients\n")
    f.write("# Fit: scalar = P/Q  (non-singular)  or  S(eps) + P/(Q*eps)  (singular: X11A, X12A)\n")
    f.write("# S(X11A)=+0.25/eps, S(X12A)=-0.25/eps\n")
    f.write("# P(eps) = sum(p_i * eps^i, i=0..5),  Q(eps) = 1 + sum(c_i^2 * eps^(i+1), i=0..4)\n")
    f.write("# Below crossover eps use AT asymptotic instead.\n")
    f.write("# name  singular  crossover  p_0..p_5  c_0..c_4\n")

    for name, singular in scalar_specs:
        popt       = popts[name]
        n_num, n_den = 6, 5
        co         = crossovers[name]
        coeffs_str = ' '.join(f'{v:.15e}' for v in popt)
        f.write(f"{name}  {singular}  {co:.6e}  {coeffs_str}\n")

print(f"\nFit coefficients saved to {out_path}")

plt.show()