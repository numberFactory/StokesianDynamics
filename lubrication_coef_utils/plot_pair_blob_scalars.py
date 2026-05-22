"""
plot_pair_mb_scalars.py
-----------------------
Extracts MB pair resistance scalars from the C++ Lubrication class,
fits each with a rational function (no asymptotic formula available),
and compares against the RPY mobility-derived resistance as a reference.

Usage:
    python plot_pair_mb_scalars.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from libMobility import NBody

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication

# =============================================================================
# Parameters
# =============================================================================
a      = 1.0
eta    = 1.0 / 6.0 / np.pi
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
    np.linspace(0.08, 0.25, 200),
    np.linspace(2.3,  2.5 - 1e-6, 20),
]))
r_norms = gaps + 2.0
eps     = gaps

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
z_safe          = 1e3 * a
lub             = Lubrication(d_cut)

# =============================================================================
# RPY mobility solver — used to compute reference resistance
# =============================================================================
solver = NBody("open", "open", "single_wall")
solver.setParameters(wallHeight=0.0)
solver.initialize(viscosity=eta, hydrodynamicRadius=a, includeAngular=True)

def form_rpy_mobility(r1, r2):
    Id        = np.eye(12)
    positions = np.vstack((r1, r2))
    solver.setPositions(positions.flatten())
    Mob = np.zeros((12, 12))
    for i in range(12):
        FT   = Id[:, i].reshape(2, 6)
        F    = FT[:, 0:3].flatten()
        T    = FT[:, 3:6].flatten()
        U, W = solver.Mdot(forces=F, torques=T)
        UW   = np.concatenate((U.reshape(2, 3), W.reshape(2, 3)), axis=1)
        Mob[:, i] = UW.flatten()
    return Mob

# =============================================================================
# Extract MB scalars from C++ and RPY resistance from libMobility
# =============================================================================
print("Extracting MB scalars and RPY resistance...")

X11A_mb = np.zeros(len(r_norms)); Y11A_mb = np.zeros(len(r_norms))
Y11B_mb = np.zeros(len(r_norms)); X11C_mb = np.zeros(len(r_norms))
Y11C_mb = np.zeros(len(r_norms)); X12A_mb = np.zeros(len(r_norms))
Y12A_mb = np.zeros(len(r_norms)); Y12B_mb = np.zeros(len(r_norms))
X12C_mb = np.zeros(len(r_norms)); Y12C_mb = np.zeros(len(r_norms))

# RPY resistance scalars (same index mapping, from pinv of Mob)
X11A_rpy = np.zeros(len(r_norms)); Y11A_rpy = np.zeros(len(r_norms))
Y11B_rpy = np.zeros(len(r_norms)); X11C_rpy = np.zeros(len(r_norms))
Y11C_rpy = np.zeros(len(r_norms)); X12A_rpy = np.zeros(len(r_norms))
Y12A_rpy = np.zeros(len(r_norms)); Y12B_rpy = np.zeros(len(r_norms))
X12C_rpy = np.zeros(len(r_norms)); Y12C_rpy = np.zeros(len(r_norms))

for idx, r_norm in enumerate(r_norms):
    r1 = np.array([0.0,      0.0, z_safe], dtype=np.float64)
    r2 = np.array([r_norm*a, 0.0, z_safe], dtype=np.float64)
    n_list = [np.array([1], dtype=np.int32),
              np.array([],  dtype=np.int32)]

    # MB resistance from C++
    R_mb = lub.ResistCSC([r1, r2], n_list, a, eta,
                         cutoff, 0.0, periodic_length, False).toarray()
    X11A_mb[idx] = R_mb[0, 0]  / f0;  Y11A_mb[idx] = R_mb[1, 1]  / f0
    Y11B_mb[idx] = R_mb[1, 5]  / f1;  X11C_mb[idx] = R_mb[3, 3]  / f2
    Y11C_mb[idx] = R_mb[4, 4]  / f2;  X12A_mb[idx] = R_mb[0, 6]  / f0
    Y12A_mb[idx] = R_mb[1, 7]  / f0;  Y12B_mb[idx] = R_mb[1, 11] / f1
    X12C_mb[idx] = R_mb[3, 9]  / f2;  Y12C_mb[idx] = R_mb[4, 10] / f2

    # RPY resistance from pinv(Mob)
    Mob   = form_rpy_mobility(r1, r2)
    R_rpy = np.linalg.pinv(Mob)
    X11A_rpy[idx] = R_rpy[0, 0]  / f0;  Y11A_rpy[idx] = R_rpy[1, 1]  / f0
    Y11B_rpy[idx] = R_rpy[1, 5]  / f1;  X11C_rpy[idx] = R_rpy[3, 3]  / f2
    Y11C_rpy[idx] = R_rpy[4, 4]  / f2;  X12A_rpy[idx] = R_rpy[0, 6]  / f0
    Y12A_rpy[idx] = R_rpy[1, 7]  / f0;  Y12B_rpy[idx] = R_rpy[1, 11] / f1
    X12C_rpy[idx] = R_rpy[3, 9]  / f2;  Y12C_rpy[idx] = R_rpy[4, 10] / f2

data_dict = {
    'X11A': X11A_mb, 'Y11A': Y11A_mb, 'Y11B': Y11B_mb,
    'X11C': X11C_mb, 'Y11C': Y11C_mb, 'X12A': X12A_mb,
    'Y12A': Y12A_mb, 'Y12B': Y12B_mb, 'X12C': X12C_mb, 'Y12C': Y12C_mb,
}
rpy_dict = {
    'X11A': X11A_rpy, 'Y11A': Y11A_rpy, 'Y11B': Y11B_rpy,
    'X11C': X11C_rpy, 'Y11C': Y11C_rpy, 'X12A': X12A_rpy,
    'Y12A': Y12A_rpy, 'Y12B': Y12B_rpy, 'X12C': X12C_rpy, 'Y12C': Y12C_rpy,
}
print("Done.")

# =============================================================================
# Load existing MB fit coefficients as initial guesses (if file exists)
# =============================================================================
in_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'pair_mb_scalar_fits.txt')

p0_dict = {}
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
    print("No existing MB fit file — using default initial guesses.")

# =============================================================================
# Rational fits — all MB scalars are finite at eps=0, fit directly.
# X11A_mb has a 1/eps singularity (same as Sup); X12A_mb does too.
# All others: fit directly with P/Q.
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

print("Fitting rational functions...")
fits      = {}
callables = {}
popts     = {}

for name, data, singular, S_func in [
    ('X11A', X11A_mb, True,  lambda e: 0.25/e),
    ('X12A', X12A_mb, True,  lambda e: -0.25/e),
    ('Y11A', Y11A_mb, False, None), ('Y12A', Y12A_mb, False, None),
    ('Y11B', Y11B_mb, False, None), ('Y12B', Y12B_mb, False, None),
    ('X11C', X11C_mb, False, None), ('X12C', X12C_mb, False, None),
    ('Y11C', Y11C_mb, False, None), ('Y12C', Y12C_mb, False, None),
]:
    print(f"  {name}...", end=' ')
    if singular:
        rat_fn, recon, rel_err, popt = fit_with_singular(eps, data, S_func, name)
    else:
        rat_fn, recon, rel_err, popt = fit_direct(eps, data, name)
    print(f"max relative error = {rel_err:.2e}")
    fits[name]      = recon
    callables[name] = rat_fn
    popts[name]     = popt

# =============================================================================
# Plot helpers
# =============================================================================
lw_d   = 3.0
lw_rpy = 2.0
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
# Figure 1: MB data (solid) + RPY resistance (dashed) + fit (dotted)
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('MB scalars: C++ data (solid), RPY resistance (dashed), fit (dotted)',
              fontsize=13)

for ax, (title, pairs) in zip(axes1.flat, groups):
    for label, name in pairs:
        ln, = ax.loglog(eps, np.abs(data_dict[name]), lw=lw_d,  label=label)
        ax.loglog(eps, np.abs(rpy_dict[name]),         lw=lw_rpy,
                  color=ln.get_color(), linestyle='--')
        ax.loglog(eps, np.abs(fits[name]),              lw=lw_fit,
                  color=ln.get_color(), linestyle=':')
    ax.plot([], [], 'k-',  lw=lw_d,  label='MB (C++)')
    ax.plot([], [], 'k--', lw=lw_rpy, label='RPY resistance')
    ax.plot([], [], 'k:',  lw=lw_fit, label='rational fit')
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
fig2.suptitle('MB rational fit residuals: |fit - data| / |data|', fontsize=14)

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
# Figure 3: relative error between MB (C++) and RPY resistance
# =============================================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('Relative difference: |MB - RPY| / |RPY|', fontsize=14)

for ax, (title, pairs) in zip(axes3.flat, groups):
    for label, name in pairs:
        rel_diff = np.abs((data_dict[name] - rpy_dict[name]) /
                          (np.abs(rpy_dict[name]) + 1e-30))
        ax.loglog(eps, rel_diff, lw=2.5, label=label)
    ax.axhline(1e-3, color='k', lw=1.5, linestyle=':',  label='$10^{-3}$')
    ax.set_xlabel(r'$\epsilon$', fontsize=12)
    ax.set_ylabel('relative error', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
print("Figure 3 ready.")

# =============================================================================
# Figure 4: minimum eigenvalue of Delta_R = R_sup - R_mb vs epsilon
# Checks positive semi-definiteness of the lubrication correction.
# R_sup is taken from the C++ code (using the rational fit implementation).
# =============================================================================
print("Computing Delta_R eigenvalues...")

min_eigs = np.zeros(len(r_norms))

for idx, r_norm in enumerate(r_norms):
    r1 = np.array([0.0,      0.0, z_safe], dtype=np.float64)
    r2 = np.array([r_norm*a, 0.0, z_safe], dtype=np.float64)
    n_list = [np.array([1], dtype=np.int32),
              np.array([],  dtype=np.int32)]

    R_sup = lub.ResistCSC([r1, r2], n_list, a, eta,
                          cutoff, 0.0, periodic_length, True).toarray()
    R_mb  = lub.ResistCSC([r1, r2], n_list, a, eta,
                          cutoff, 0.0, periodic_length, False).toarray()

    Delta_R      = R_sup - R_mb
    min_eigs[idx] = np.linalg.eigvalsh(Delta_R).min()

print("Done.")

fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.suptitle(r'Min eigenvalue of $\Delta R = R_\mathrm{sup} - R_\mathrm{mb}$ vs $\epsilon$',
              fontsize=13)

pos_mask = min_eigs >= 0
neg_mask = min_eigs <  0

if pos_mask.any():
    ax4.semilogy(eps[pos_mask],  min_eigs[pos_mask],
                 'o', color='steelblue', ms=3, label='positive (PD)')
if neg_mask.any():
    ax4.semilogy(eps[neg_mask], -min_eigs[neg_mask],
                 'x', color='red', ms=5, label='negative (NOT PD) — plotted as |val|')

ax4.axhline(0, color='k', lw=1.0, linestyle='--')
ax4.set_xlabel(r'$\epsilon = r/a - 2$', fontsize=12)
ax4.set_ylabel(r'$\lambda_\mathrm{min}(\Delta R)$', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, which='both', alpha=0.3)

n_neg = neg_mask.sum()
print(f"\nDelta_R positivity: {n_neg} / {len(r_norms)} separations have negative min eigenvalue.")
if n_neg > 0:
    print(f"  Worst violation: lambda_min = {min_eigs[neg_mask].min():.3e} "
          f"at eps = {eps[neg_mask][np.argmin(min_eigs[neg_mask])]:.4e}")
else:
    print("  PASS: Delta_R is positive semi-definite at all tested separations.")

plt.tight_layout()
print("Figure 4 ready.")

# =============================================================================
# Save MB fit coefficients
# =============================================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pair_mb_scalar_fits.txt')

scalar_specs = [
    ('X11A', True), ('X12A', True),
    ('Y11A', False), ('Y12A', False), ('Y11B', False), ('Y12B', False),
    ('X11C', False), ('X12C', False), ('Y11C', False), ('Y12C', False),
]

with open(out_path, 'w') as f:
    f.write("# MB pair lubrication scalar rational fit coefficients\n")
    f.write("# Fit: scalar = P/Q  (non-singular)  or  S(eps) + P/(Q*eps)  (singular: X11A, X12A)\n")
    f.write("# S(X11A)=+0.25/eps, S(X12A)=-0.25/eps\n")
    f.write("# P(eps) = sum(p_i * eps^i, i=0..5),  Q(eps) = 1 + sum(c_i^2 * eps^(i+1), i=0..4)\n")
    f.write("# No asymptotic fallback — rational fit used over full range.\n")
    f.write("# name  singular  d_cut  p_0..p_5  c_0..c_4\n")
    for name, singular in scalar_specs:
        co         = d_cut
        coeffs_str = ' '.join(f'{v:.15e}' for v in popts[name])
        f.write(f"{name}  {singular}  {co:.6e}  {coeffs_str}\n")

print(f"\nMB fit coefficients saved to {out_path}")

plt.show()
