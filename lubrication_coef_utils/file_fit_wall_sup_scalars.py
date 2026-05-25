"""
plot_wall_scalars_vs_rpy.py
----------------------------
Plots the 2562-blob resistance scalars, AT asymptotics, and RPY resistance,
then in a second figure plots the difference (2562 - RPY) on a linear scale.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from libMobility import NBody

# =============================================================================
# Load 2562-blob reference resistance scalars
# Columns: h  Xa  Ya  Yb  Xc  Yc  (full scalars, not corrections)
# =============================================================================
ref_file = "./resistance_coeffs/res_scalars_wall_MB_2562.txt"
#ref_file = "./resistance_coeffs/res_scalars_wall_MB_10242_full.txt"
ref_data    = np.loadtxt(ref_file)
ref_h       = ref_data[:, 0]
ref_eps     = ref_h - 1.0
sort_idx    = np.argsort(ref_h)
ref_eps     = ref_eps[sort_idx]
ref_h       = ref_h[sort_idx]

ref_dict = {
    'Xa_corr': ref_data[sort_idx, 1] - 1.0,
    'Ya_corr': ref_data[sort_idx, 2] - 1.0,
    'Yb':      ref_data[sort_idx, 3],
    'XcPlus':  ref_data[sort_idx, 4]*1.0005 - 4.0/3.0,
    'YcPlus':  ref_data[sort_idx, 5]*1.0005 - 4.0/3.0,
}
print(f"Loaded {ref_data.shape[0]} rows from {ref_file}  "
      f"(h: {ref_h.min():.4f} .. {ref_h.max():.4f})")

# =============================================================================
# RPY single-wall resistance scalars via libMobility
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

print("Extracting RPY scalars on 2562-blob height grid...")
rpy_dict = {k: np.zeros(len(ref_h)) for k in ref_dict}
for idx, h in enumerate(ref_h):
    R = np.linalg.inv(form_mob_single(h))
    rpy_dict['Xa_corr'][idx] = R[2, 2] / f0 - 1.0
    rpy_dict['Ya_corr'][idx] = R[0, 0] / f0 - 1.0
    rpy_dict['Yb'][idx]      = R[0, 4] / f1
    rpy_dict['XcPlus'][idx]  = R[5, 5] / f2 - 4.0/3.0
    rpy_dict['YcPlus'][idx]  = R[3, 3] / f2 - 4.0/3.0
print("Done.")

# =============================================================================
# Asymptotics
# =============================================================================
def asym_wall(name, e):
    e  = np.maximum(e, 1e-300)
    le = np.log(e)
    d  = {
        'Xa_corr': 1.0/e - (1.0/5.0)*le + 0.971280 - 1.0,
        'Ya_corr': -(8.0/15.0)*le + 0.9588 - 1.0,
        'Yb':      (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.4576*e),
        'XcPlus':  (4.0/3.0)*(1.2020569 - 3.0*(np.pi**2/6.0-1.0)*e) - 4.0/3.0,
        'YcPlus':  (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*e) - 4.0/3.0,
    }
    return d[name]

def farfield_corr(name, h):
    d = {'Xa_corr': -9.0/(8.0*h), 'Ya_corr': -9.0/(16.0*h)}
    return d.get(name, np.zeros_like(h))

eps_fine = np.logspace(np.log10(ref_eps.min()*0.8), np.log10(ref_eps.max()*1.2), 500)
h_fine   = eps_fine + 1.0

##################################
# code idea: 
# fit to 2562 data, only use it where you need to, switch to RPY where they agree
# then use asyptotics for the near-contact limit
##################################
# =============================================================================
# Rational fits
# =============================================================================
def rational_fit(x, y, num_deg=3, denom_deg=5):
    from scipy.optimize import curve_fit
    def rat_func(x, *params):
        num_params = num_deg + 1
        denom_params = denom_deg + 1
        num = np.polyval(params[:num_params][::-1], x)
        denom = np.polyval(params[num_params:][::-1], x)
        return num / denom
    p0 = np.ones(num_deg + denom_deg + 2)  # Initial guess
    popt, _ = curve_fit(rat_func, x, y, p0=p0, method='trf')
    return lambda x: rat_func(x, *popt)

def PowerLaw_fit(x,y,num_terms=7, power_start=-5):
    from scipy.optimize import curve_fit
    def fit_func(x, *params):
        result = np.zeros_like(x)
        for n in range(num_terms):
            result += params[n] * (1.0+x)**(power_start-n)
        return result
    p0 = np.ones(num_terms)  # Initial guess
    popt, _ = curve_fit(fit_func, x, y, p0=p0, method='trf')
    return lambda x: fit_func(x, *popt)






# =============================================================================
# Figure 1: log-log  —  2562 scalars + RPY + asymptotics
# =============================================================================
scalar_info = [
    ('$X_a$ corr', 'Xa_corr', True),
    ('$Y_a$ corr', 'Ya_corr', True),
    ('$Y_b$',      'Yb',      False),
    ('$X_c^+$',    'XcPlus',  False),
    ('$Y_c^+$',    'YcPlus',  False),
]

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Wall resistance scalars: 2562-blob, RPY, AT asymptotics', fontsize=13)

for ax, (label, name, has_ff) in zip(axes1.flat, scalar_info):
    ref_v = ref_dict[name]
    rpy_v = rpy_dict[name]

    mask_ref = np.abs(ref_v) > 1e-15
    mask_rpy = np.abs(rpy_v) > 1e-15

    ax.loglog(ref_eps[mask_ref], np.abs(ref_v[mask_ref]),
              lw=2.5, color='steelblue', label='MB 2562')
    ax.loglog(ref_eps[mask_rpy], np.abs(rpy_v[mask_rpy]),
              lw=2.0, color='darkorange', linestyle='--', label='RPY')
    
    # fit a rational function to the 2562 data for this scalar
    if name == 'Yb':
        Yb_RPY_cutoff = 7.0
        Yb_asym_cutoff = 1.3e-1
        mask_Yb = ref_eps > 0.1
        fit_func = PowerLaw_fit(ref_eps[mask_Yb], ref_v[mask_Yb])
        ax.loglog(ref_eps[mask_Yb], np.abs(fit_func(ref_eps[mask_Yb])),
                    lw=3.8, color='mediumpurple', linestyle=':', label='Yb fit')
    elif name == 'XcPlus':
        Xc_RPY_cutoff = 0.86
        Xc_asym_cutoff = 2.5e-3
        mask_XcPlus = ref_eps > 0.001
        limit_XcPlus = ((4.0/3.0)*(1.2020569 - 3.0*(np.pi**2/6.0-1.0)*0.0) - 4.0/3.0)*1.0005
        small_eps_extra = np.logspace(-4, -2.5, 200)
        small_eps_asym = asym_wall(name, small_eps_extra)
        ref_v_extra = np.concatenate([small_eps_asym, ref_v])
        ref_eps_extra = np.concatenate([small_eps_extra, ref_eps])

        fit_func = rational_fit(ref_eps_extra, ref_v_extra-limit_XcPlus, num_deg=3, denom_deg=3)
        ax.loglog(ref_eps, np.abs(fit_func(ref_eps)+limit_XcPlus),
                    lw=3.8, color='mediumpurple', linestyle=':', label='rational fit')
        # fit_func = PowerLaw_fit(ref_eps[mask_XcPlus], ref_v[mask_XcPlus]-limit_XcPlus,power_start=-3, num_terms=5)
        # ax.loglog(ref_eps[mask_XcPlus], np.abs(fit_func(ref_eps[mask_XcPlus])+limit_XcPlus),
        #             lw=3.8, color='mediumpurple', linestyle=':', label='Xc+ fit')
    elif name == 'YcPlus':
        Yc_RPY_cutoff = 1.8204
        Yc_asym_cutoff = 4.789e-2
        mask_Xa = ref_eps > 0.1
        fit_func = rational_fit(ref_eps[mask_Xa], ref_v[mask_Xa], num_deg=2, denom_deg=5)
        ax.loglog(ref_eps, np.abs(fit_func(ref_eps)),
                    lw=3.8, color='mediumpurple', linestyle=':', label='rational fit')
    elif name == 'Xa_corr':
        Xa_RPY_cutoff = 9.0
        Xa_asym_cutoff = 2.0549e-1
        # mask_Xa = ref_eps > 0.1
        # fit_func = PowerLaw_fit(ref_eps[mask_Xa], ref_v[mask_Xa], power_start=-1, num_terms=6)
        # ax.loglog(ref_eps[mask_Xa], np.abs(fit_func(ref_eps[mask_Xa])),
        #             lw=3.8, color='mediumpurple', linestyle=':', label='Xa corr fit')
        mask_Xa = ref_eps > 0.1
        fit_func = rational_fit(ref_eps[mask_Xa], ref_v[mask_Xa],num_deg=3, denom_deg=4)
        ax.loglog(ref_eps, np.abs(fit_func(ref_eps)),
                    lw=3.8, color='mediumpurple', linestyle=':', label='rational fit')
    elif name == 'Ya_corr':
        Ya_RPY_cutoff = 7.871
        Ya_asym_cutoff = 4.654e-2
        mask_Xa = ref_eps > 0.1
        fit_func = rational_fit(ref_eps[mask_Xa], ref_v[mask_Xa],num_deg=2, denom_deg=3)
        ax.loglog(ref_eps, np.abs(fit_func(ref_eps)),
                    lw=3.8, color='mediumpurple', linestyle=':', label='rational fit')
    # else:
        # fit_func = rational_fit(ref_eps[mask_ref], ref_v[mask_ref])
        # ax.loglog(ref_eps, np.abs(fit_func(ref_eps)),
        #             lw=3.8, color='mediumpurple', linestyle=':', label='rational fit')

    # near-contact asymptotic
    asym = asym_wall(name, eps_fine)
    mask_nc = (eps_fine < 0.5) & (np.abs(asym) > 1e-15)
    ax.loglog(eps_fine[mask_nc], np.abs(asym[mask_nc]),
              lw=1.8, color='tomato', linestyle=':', label='near-contact asym.')

    # far-field (Xa, Ya only)
    if has_ff:
        ff = farfield_corr(name, h_fine)
        mask_ff = (eps_fine > 1.0) & (np.abs(ff) > 1e-15)
        ax.loglog(eps_fine[mask_ff], np.abs(ff[mask_ff]),
                  lw=1.8, color='seagreen', linestyle='-.', label='Faxen $A/h$')

    ax.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=11)
    ax.set_ylabel('|scalar|', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

axes1.flat[-1].set_visible(False)
plt.tight_layout()



plt.show()