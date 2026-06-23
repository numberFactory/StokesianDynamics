"""
chimera_wall_scalars.py
-----------------------
Builds chimera wall resistance scalars composed of three regions:
  1. Near-contact: exact AT asymptotic  (eps < asym_cutoff)
  2. Midfield:     fit to 2562-blob data (asym_cutoff <= eps <= rpy_cutoff)
  3. Far-field:    RPY resistance        (eps > rpy_cutoff)

Cutoffs and fit functions are taken from the manually optimised values
in the original plotting code.

Figure 1: each scalar with chimera + components + asymptotics + RPY
Figure 2: chimera scalars only, with red dots where Delta_R has a negative eigenvalue
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from libMobility import NBody

# =============================================================================
# Load 2562-blob reference data
# Columns: h  Xa  Ya  Yb  Xc  Yc  (full scalars)
# =============================================================================
ref_file = "./resistance_coeffs/res_scalars_wall_MB_2562.txt"
ref_data = np.loadtxt(ref_file)
sort_idx = np.argsort(ref_data[:, 0])
ref_data = ref_data[sort_idx]
ref_h    = ref_data[:, 0]
ref_eps  = ref_h - 1.0

ref_dict = {
    'Xa_corr': ref_data[:, 1] - 1.0,
    'Ya_corr': ref_data[:, 2] - 1.0,
    'Yb':      ref_data[:, 3]*1.0,
    'XcPlus':  ref_data[:, 4]*1.00 - 4.0/3.0,
    'YcPlus':  ref_data[:, 5]*1.00 - 4.0/3.0,
}
print(f"Loaded {ref_data.shape[0]} rows  h: {ref_h.min():.4f}..{ref_h.max():.4f}")

# =============================================================================
# RPY scalars on a dense grid that extends to h=30
# =============================================================================
a   = 1.0
eta = 1.0 / (6.0 * np.pi)
f0  = 6.0 * np.pi * eta * a
f1  = 6.0 * np.pi * eta * a**2
f2  = 6.0 * np.pi * eta * a**3

solver_wall = NBody("open", "open", "single_wall")
solver_wall.setParameters(wallHeight=0.0)
solver_wall.initialize(viscosity=eta, hydrodynamicRadius=a, includeAngular=True)

def form_mob_single(h):
    pos = np.array([0.0, 0.0, h], dtype=np.float64)
    solver_wall.setPositions(pos)
    M = np.zeros((6, 6)); Id = np.eye(6)
    for i in range(6):
        U, W = solver_wall.Mdot(forces=Id[i, :3].copy(), torques=Id[i, 3:].copy())
        M[:3, i] = U;  M[3:, i] = W
    return M

def rpy_scalar_at_h(h):
    R = np.linalg.inv(form_mob_single(h))
    return {
        'Xa_corr': R[2, 2] / f0 - 1.0,
        'Ya_corr': R[0, 0] / f0 - 1.0,
        'Yb':      R[0, 4] / f1,
        'XcPlus':  R[5, 5] / f2 - 4.0/3.0,
        'YcPlus':  R[3, 3] / f2 - 4.0/3.0,
    }

def build_R_corr_scaled_local(Xa_c, Ya_c, Yb, XcP, YcP):
    R = np.zeros((6, 6))
    R[0,0]=f0*Ya_c; R[1,1]=f0*Ya_c; R[2,2]=f0*Xa_c
    R[3,3]=f2*YcP;  R[4,4]=f2*YcP;  R[5,5]=f2*XcP
    R[0,4]= f1*Yb;  R[4,0]= f1*Yb
    R[1,3]=-f1*Yb;  R[3,1]=-f1*Yb
    return R

# =============================================================================
# Preprocessing: enforce SPD of Delta_R on 2562-blob scalars
#
# For each height h in the reference grid:
#   1. Build R_ref (6x6) from 2562 scalars, R_rpy (6x6) from RPY
#   2. Delta_R = R_ref - R_rpy
#   3. Eigen-decompose Delta_R; zero negative eigenvalues -> Delta_R_trunc
#   4. R_trunc = R_rpy + Delta_R_trunc; extract truncated scalars
# All subsequent fitting uses ref_dict_trunc.
# =============================================================================
print("\nPreprocessing: truncating negative eigenvalues of Delta_R on 2562 data...")
n_truncated    = 0
ref_dict_trunc = {k: ref_dict[k].copy() for k in ref_dict}

for idx, h in enumerate(ref_h):
    rv    = rpy_scalar_at_h(h)
    R_rpy = build_R_corr_scaled_local(
        rv['Xa_corr'], rv['Ya_corr'], rv['Yb'], rv['XcPlus'], rv['YcPlus'])
    R_ref = build_R_corr_scaled_local(
        ref_dict['Xa_corr'][idx], ref_dict['Ya_corr'][idx],
        ref_dict['Yb'][idx],      ref_dict['XcPlus'][idx],
        ref_dict['YcPlus'][idx])

    Delta            = R_ref - R_rpy
    eigvals, eigvecs = np.linalg.eigh(Delta)

    if eigvals.min() < 0:
        n_truncated      += 1
        eigvals_trunc     = np.maximum(eigvals, 0.0)
        Delta_trunc       = eigvecs @ np.diag(eigvals_trunc) @ eigvecs.T
        R_trunc           = R_rpy + Delta_trunc
        ref_dict_trunc['Xa_corr'][idx] = R_trunc[2, 2] / f0
        ref_dict_trunc['Ya_corr'][idx] = R_trunc[0, 0] / f0
        ref_dict_trunc['Yb'][idx]      = R_trunc[0, 4] / f1
        ref_dict_trunc['XcPlus'][idx]  = R_trunc[5, 5] / f2
        ref_dict_trunc['YcPlus'][idx]  = R_trunc[3, 3] / f2

print(f"  Truncated {n_truncated}/{len(ref_h)} heights  "
      f"({100*n_truncated/len(ref_h):.1f}% of reference grid)")

# Replace ref_dict with SPD-enforced version for all downstream fitting
ref_dict = ref_dict_trunc

# Dense grid out to h=30
rpy_h   = np.unique(np.concatenate([
    ref_h,
    np.logspace(np.log10(ref_h.max()), np.log10(30.0), 60),
]))
rpy_eps = rpy_h - 1.0

print("Extracting RPY scalars to h=30...")
rpy_vals = {k: np.zeros(len(rpy_h)) for k in ref_dict}
for idx, h in enumerate(rpy_h):
    R = np.linalg.inv(form_mob_single(h))
    rpy_vals['Xa_corr'][idx] = R[2, 2] / f0 - 1.0
    rpy_vals['Ya_corr'][idx] = R[0, 0] / f0 - 1.0
    rpy_vals['Yb'][idx]      = R[0, 4] / f1
    rpy_vals['XcPlus'][idx]  = R[5, 5] / f2 - 4.0/3.0
    rpy_vals['YcPlus'][idx]  = R[3, 3] / f2 - 4.0/3.0
print("Done.")

# =============================================================================
# Asymptotic formulas
# =============================================================================
def asym_wall(name, e):
    e  = np.maximum(e, 1e-300)
    le = np.log(e)
    d  = {
        'Xa_corr': 1.0/e - (1.0/5.0)*le + 0.971280 - 1.0,
        'Ya_corr': -(8.0/15.0)*le + 0.9588 - 1.0,
        'Yb':      (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.0195 - (0.4576-0.15)*e),
        'XcPlus':  (4.0/3.0)*(1.2020569 - 3.0*(np.pi**2/6.0-1.0)*e) - 4.0/3.0,
        'YcPlus':  (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*e) - 4.0/3.0,
    }
    return d[name]

def farfield_corr(name, h):
    d = {'Xa_corr': -9.0/(8.0*h), 'Ya_corr': -9.0/(16.0*h)}
    return d.get(name, np.zeros_like(h))

# =============================================================================
# Fit functions (from manual optimisation)
# =============================================================================
def rational_fit(x, y, num_deg=3, denom_deg=5):
    def rat_func(x, *params):
        n = num_deg + 1
        return np.polyval(params[:n][::-1], x) / np.polyval(params[n:][::-1], x)
    popt, _ = curve_fit(rat_func, x, y, p0=np.ones(num_deg+denom_deg+2), method='trf')
    return lambda x: rat_func(x, *popt)

def PowerLaw_fit(x, y, num_terms=7, power_start=-5):
    def fit_func(x, *params):
        result = np.zeros_like(x, dtype=float)
        for n in range(num_terms):
            result += params[n] * (1.0+x)**(power_start-n)
        return result
    popt, _ = curve_fit(fit_func, x, y, p0=np.ones(num_terms), method='trf')
    return lambda x: fit_func(x, *popt)

# Cutoffs (eps_w):  asym valid below asym_cutoff, RPY used above rpy_cutoff
cutoffs = {
    'Xa_corr': dict(asym=2.0549e-1, rpy=9.0),
    'Ya_corr': dict(asym=2.9118e-2,  rpy=6.44),
    # 'Yb':      dict(asym=1.3e-1,    rpy=7.0),
    'Yb':      dict(asym=1.30825e-1,    rpy=4.15),
    # 'XcPlus':  dict(asym=2.5e-3,    rpy=0.86),
    # 'YcPlus':  dict(asym=4.789e-2,  rpy=1.8204),
    'XcPlus':  dict(asym=2.5e-3,    rpy=0.55),
    'YcPlus':  dict(asym=4.56e-2,  rpy=3.5),
}

# Build fit functions on 2562 data
print("\nFitting midfield functions...")
fit_funcs = {}

# Xa_corr
name = 'Xa_corr'
mask = ref_eps > 0.1
fit_funcs[name] = rational_fit(ref_eps[mask], ref_dict[name][mask], num_deg=3, denom_deg=4)

# Ya_corr
name = 'Ya_corr'
mask = ref_eps > 0.01
#fit_funcs[name] = PowerLaw_fit(ref_eps[mask], ref_dict[name][mask], num_terms=4, power_start=-1)
fit_funcs[name] = rational_fit(ref_eps[mask], 1.0*ref_dict[name][mask], num_deg=4, denom_deg=5)

# Yb
name = 'Yb'
mask = ref_eps > 0.1
#fit_funcs[name] = PowerLaw_fit(ref_eps[mask], ref_dict[name][mask], num_terms=4, power_start=-5)
fit_funcs[name] = rational_fit(ref_eps[mask], ref_dict[name][mask], num_deg=4, denom_deg=5)

# XcPlus
name = 'XcPlus'
limit_XcPlus = (4.0/3.0)*(1.2020569) - 4.0/3.0
small_eps_extra = np.logspace(-4, -2.5, 200)
ref_v_extra  = np.concatenate([asym_wall(name, small_eps_extra), ref_dict[name]])
ref_eps_extra = np.concatenate([small_eps_extra, ref_eps])
fit_funcs[name] = rational_fit(ref_eps_extra, ref_v_extra - limit_XcPlus,
                                num_deg=3, denom_deg=6)
_f_Xc = fit_funcs[name]
fit_funcs[name] = lambda e, _f=_f_Xc: _f(e) + limit_XcPlus

# YcPlus
name = 'YcPlus'
mask = ref_eps > 0.01
#fit_funcs[name] = rational_fit(ref_eps[mask], ref_dict[name][mask], num_deg=3, denom_deg=6)
fit_funcs[name] = rational_fit(ref_eps[mask], 1.0*ref_dict[name][mask], num_deg=3, denom_deg=6)

print("Fits complete.")

# =============================================================================
# Build chimera curves on a unified fine grid
# Grid: from very small eps to h=30
# =============================================================================
chi_eps = np.unique(np.concatenate([
    np.logspace(-4, np.log10(ref_eps.min()), 80),
    ref_eps,
    rpy_eps[rpy_eps > ref_eps.max()],
]))
chi_h = chi_eps + 1.0

def build_R_corr_scaled(Xa_c, Ya_c, Yb, XcP, YcP):
    return build_R_corr_scaled_local(Xa_c, Ya_c, Yb, XcP, YcP)

chimera = {k: np.zeros(len(chi_eps)) for k in ref_dict}
for i, (e, h) in enumerate(zip(chi_eps, chi_h)):
    rpy_at_h = None   # computed lazily only if needed
    for name in ref_dict:
        co   = cutoffs[name]
        e_lo = co['asym']
        e_hi = co['rpy']
        if e <= e_lo:
            chimera[name][i] = asym_wall(name, np.array([e]))[0]
        elif e <= e_hi:
            chimera[name][i] = fit_funcs[name](np.array([e]))[0]
        else:
            if rpy_at_h is None:
                rpy_at_h = rpy_scalar_at_h(h)
            chimera[name][i] = rpy_at_h[name]

print("Chimera cutoffs:")
for name in ref_dict:
    co = cutoffs[name]
    print(f"  {name}: asym eps<{co['asym']:.4f}, fit {co['asym']:.4f}..{co['rpy']:.4f}, "
          f"RPY eps>{co['rpy']:.4f}")

# =============================================================================
# SPD check on chimera: build Delta_R and find negative eigenvalues
# Delta_R is the resistance correction matrix (6x6) minus RPY correction
# We check if Delta_R = R_chimera_corr - R_RPY_corr is SPD
# =============================================================================
print("\nChecking SPD of Delta_R on chimera grid...")
min_eigs    = np.zeros(len(chi_eps))
bad_eps_spd = []

for i, (e, h) in enumerate(zip(chi_eps, chi_h)):
    R_chi = build_R_corr_scaled(
        chimera['Xa_corr'][i], chimera['Ya_corr'][i],
        chimera['Yb'][i],      chimera['XcPlus'][i],  chimera['YcPlus'][i])
    R_rpy_v = rpy_scalar_at_h(h)
    R_rpy = build_R_corr_scaled(
        R_rpy_v['Xa_corr'], R_rpy_v['Ya_corr'],
        R_rpy_v['Yb'],      R_rpy_v['XcPlus'],  R_rpy_v['YcPlus'])
    Delta  = R_chi - R_rpy
    lam    = np.linalg.eigvalsh(Delta)
    min_eigs[i] = lam.min()
    if lam.min() < 0:
        bad_eps_spd.append(e)

n_bad = len(bad_eps_spd)
if bad_eps_spd:
    print(f"  First non-SPD point at eps={min(bad_eps_spd):.4e} "
          f"(h={min(bad_eps_spd)+1:.4f}), min eig={min_eigs[chi_eps==min(bad_eps_spd)][0]:.4e}")
print(f"Largest negative eigenvalue of Delta_R: {min_eigs.min():.2e}")
print(f"Smallest epsilon with negative Delta_R eigenvalue: "
      f"{min(bad_eps_spd):.4e}" if bad_eps_spd else "No negative eigenvalues found.")
print(f"  {n_bad}/{len(chi_eps)} chimera points have negative Delta_R eigenvalue")

# =============================================================================
# Figure 1: each scalar — chimera + 2562 data + RPY + asymptotics
# =============================================================================
scalar_info = [
    ('$X_a$ corr', 'Xa_corr', True),
    ('$Y_a$ corr', 'Ya_corr', True),
    ('$Y_b$',      'Yb',      False),
    ('$X_c^+$',    'XcPlus',  False),
    ('$Y_c^+$',    'YcPlus',  False),
]

eps_fine = np.logspace(np.log10(chi_eps.min()), np.log10(chi_eps.max()), 500)
h_fine   = eps_fine + 1.0

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Wall resistance scalars: chimera, MB 2562, RPY, asymptotics', fontsize=13)

for ax, (label, name, has_ff) in zip(axes1.flat, scalar_info):
    co    = cutoffs[name]
    ref_v = ref_dict[name]
    rpy_v = rpy_vals[name]
    chi_v = chimera[name]

    # chimera
    mask = np.abs(chi_v) > 1e-15
    ax.loglog(chi_eps[mask], np.abs(chi_v[mask]),
              lw=3.0, color='purple', label='chimera')

    # MB 2562
    mask = np.abs(ref_v) > 1e-15
    ax.loglog(ref_eps[mask], np.abs(ref_v[mask]),
              lw=1.5, color='steelblue', linestyle='--', label='MB 2562')

    # RPY (to h=30)
    mask = np.abs(rpy_v) > 1e-15
    ax.loglog(rpy_eps[mask], np.abs(rpy_v[mask]),
              lw=1.5, color='darkorange', linestyle='--', label='RPY')

    # near-contact asymptotic
    asym_v = asym_wall(name, eps_fine)
    mask_nc = (eps_fine < co['asym']*3) & (np.abs(asym_v) > 1e-15)
    ax.loglog(eps_fine[mask_nc], np.abs(asym_v[mask_nc]),
              lw=1.5, color='tomato', linestyle=':', label='near-contact asym.')

    # far-field Faxen (Xa, Ya only)
    if has_ff:
        ff = farfield_corr(name, h_fine)
        mask_ff = (eps_fine > 1.0) & (np.abs(ff) > 1e-15)
        ax.loglog(eps_fine[mask_ff], np.abs(ff[mask_ff]),
                  lw=1.5, color='seagreen', linestyle='-.', label='Faxen $A/h$')

    # cutoff verticals
    ax.axvline(co['asym'], color='tomato',   lw=0.8, linestyle=':')
    ax.axvline(co['rpy'],  color='darkorange', lw=0.8, linestyle=':')

    ax.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=11)
    ax.set_ylabel('|scalar|', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes1.flat[-1].set_visible(False)
plt.tight_layout()

# =============================================================================
# Figure 2: chimera - RPY difference + red dots for non-SPD points
# =============================================================================
print("\nComputing RPY values on chimera grid for Figure 2...")
rpy_on_chi = {k: np.zeros(len(chi_eps)) for k in ref_dict}
for i, (e, h) in enumerate(zip(chi_eps, chi_h)):
    rv = rpy_scalar_at_h(h)
    for k in ref_dict:
        rpy_on_chi[k][i] = rv[k]
print("Done.")

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Chimera $-$ RPY  (linear scale), red dots = non-SPD $\\Delta R$',
              fontsize=13)

bad_eps_arr = np.array(bad_eps_spd) if bad_eps_spd else np.array([])

for ax, (label, name, _) in zip(axes2.flat, scalar_info):
    diff = chimera[name] - rpy_on_chi[name]
    ax.plot(chi_eps, diff, lw=2.0, color='purple', label='chimera $-$ RPY')
    ax.axhline(0, color='k', lw=0.8, linestyle='--')

    # red dots at non-SPD eps values
    if len(bad_eps_arr) > 0:
        bad_diff = np.interp(bad_eps_arr, chi_eps, diff)
        ax.scatter(bad_eps_arr, bad_diff, s=30, color='red', zorder=5,
                   label=f'non-SPD ({n_bad} pts)')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=11)
    ax.set_ylabel('$\\Delta$ scalar', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

axes2.flat[-1].set_visible(False)
plt.tight_layout()

plt.show()
