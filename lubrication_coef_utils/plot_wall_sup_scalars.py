"""
plot_wall_sup_scalars.py
------------------------
Diagnostic only: plots wall Sup resistance scalars from C++ and RPY
alongside near-contact and far-field asymptotics. No fitting.

Usage:
    python plot_wall_sup_scalars.py
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
eta    = 1.0 / 6.0 / np.pi
d_cut  = 1e-2
cutoff = 4.5

mob_factor = np.array([
    6.0 * np.pi * eta * a,
    6.0 * np.pi * eta * a**2,
    6.0 * np.pi * eta * a**3,
])
f0, f1, f2 = mob_factor

h_min   = 1.0 + d_cut * 1.01
h_max   = 20.0 * a
heights = np.unique(np.concatenate([
    np.logspace(np.log10(h_min), np.log10(1.1),   50),
    np.logspace(np.log10(1.1),   np.log10(h_max), 200),
]))
epsh = heights - 1.0

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
lub             = Lubrication(d_cut)

# =============================================================================
# Extract wall Sup scalars from C++
# =============================================================================
print("Extracting wall Sup scalars from C++...")

Xa_corr = np.zeros(len(heights))
Ya_corr = np.zeros(len(heights))
Yb_wall = np.zeros(len(heights))
XcPlus  = np.zeros(len(heights))
YcPlus  = np.zeros(len(heights))

for idx, h in enumerate(heights):
    r1     = np.array([0.0, 0.0, h],   dtype=np.float64)
    r2     = np.array([1e6, 0.0, h],   dtype=np.float64)
    n_list = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]
    R_wall = lub.ResistCSC([r1, r2], n_list, a, eta,
                           cutoff, 1e10, periodic_length, True).toarray()
    Xa_corr[idx] = R_wall[2, 2] / f0
    Ya_corr[idx] = R_wall[0, 0] / f0
    Yb_wall[idx] = R_wall[0, 4] / f1
    XcPlus[idx]  = R_wall[5, 5] / f2
    YcPlus[idx]  = R_wall[3, 3] / f2
print("Done.")

# =============================================================================
# Extract RPY single-wall resistance scalars from libMobility
# =============================================================================
solver_wall = NBody("open", "open", "single_wall")
solver_wall.setParameters(wallHeight=0.0)
solver_wall.initialize(viscosity=eta, hydrodynamicRadius=a, includeAngular=True)

def form_mob_single(h):
    pos = np.array([0.0, 0.0, h], dtype=np.float64)
    solver_wall.setPositions(pos)
    M  = np.zeros((6, 6))
    Id = np.eye(6)
    for i in range(6):
        U, W = solver_wall.Mdot(forces=Id[i, :3].copy(), torques=Id[i, 3:].copy())
        M[:3, i] = U;  M[3:, i] = W
    return M

print("Extracting RPY single-wall resistance scalars...")
Xa_corr_rpy = np.zeros(len(heights))
Ya_corr_rpy = np.zeros(len(heights))
Yb_rpy      = np.zeros(len(heights))
XcPlus_rpy  = np.zeros(len(heights))
YcPlus_rpy  = np.zeros(len(heights))

for idx, h in enumerate(heights):
    R = np.linalg.inv(form_mob_single(h))
    Xa_corr_rpy[idx] = R[2, 2] / f0 - 1.0
    Ya_corr_rpy[idx] = R[0, 0] / f0 - 1.0
    Yb_rpy[idx]      = R[0, 4] / f1
    XcPlus_rpy[idx]  = max(R[5, 5] / f2 - 4.0/3.0, 0.0)
    YcPlus_rpy[idx]  = max(R[3, 3] / f2 - 4.0/3.0, 0.0)
print("Done.")

# =============================================================================
# Asymptotic formulas
# =============================================================================
def asym_wall(name, e):
    """Near-contact AT asymptotic. e = h - 1 = eps_w."""
    e  = np.maximum(e, 1e-300)
    le = np.log(e)
    d = {
        'Xa_corr': 1.0/e - (1.0/5.0)*le + 0.971280 - 1.0,
        'Ya_corr': -(8.0/15.0)*le + 0.9588 - 1.0,
        'Yb':      (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.4576*e),
        'XcPlus':  np.maximum((4.0/3.0)*(1.2020569
                               - 3.0*(np.pi**2/6.0-1.0)*e) - 4.0/3.0, 0.0),
        'YcPlus':  np.maximum((4.0/3.0)*(-(2.0/5.0)*le
                               + 0.3817 + 1.4578*e) - 4.0/3.0, 0.0),
    }
    return d[name]

# C++ asymptotic validity limits (from WallResistMatrix source)
cpp_limit = {
    'Xa_corr': 0.18, 'Ya_corr': 0.01,
    'Yb': 0.1275,    'XcPlus': 0.01, 'YcPlus': 0.1,
}

# Known far-field forms (Faxen)
def farfield_known(name, h):
    d = {'Xa_corr': -9.0/(8.0*h), 'Ya_corr': -9.0/(16.0*h)}
    return d[name]

# Far-field power-law fits to RPY data
def fit_powerlaw(h, data, h_lo=2.0):
    nonzero = np.abs(data) > 1e-12
    if nonzero.sum() < 5: return None, None
    h_hi = h[nonzero].max()
    mask = (h > h_lo) & (h <= h_hi) & nonzero
    if mask.sum() < 5: return None, None
    coeffs = np.polyfit(np.log(h[mask]), np.log(np.abs(data[mask])), 1)
    return np.exp(coeffs[1]) * np.sign(np.median(data[mask])), coeffs[0]

print("\nPower-law fits to RPY far-field:")
rpy_vals = {'Yb': Yb_rpy, 'XcPlus': XcPlus_rpy, 'YcPlus': YcPlus_rpy}
powerlaw_fits = {}
for name in ['Yb', 'XcPlus', 'YcPlus']:
    C, alpha = fit_powerlaw(heights, rpy_vals[name])
    powerlaw_fits[name] = (C, alpha)
    if C is not None:
        print(f"  {name:>8}: ({C:+.4e}) * h^({alpha:.4f})")

# =============================================================================
# Plot: one subplot per scalar
# =============================================================================
h_fine   = np.logspace(np.log10(h_min), np.log10(h_max * 2), 3000)
eps_fine = h_fine - 1.0

scalar_info = [
    ('$X_a$ corr', 'Xa_corr', Xa_corr, Xa_corr_rpy, True),
    ('$Y_a$ corr', 'Ya_corr', Ya_corr, Ya_corr_rpy, True),
    ('$Y_b$',      'Yb',      Yb_wall, Yb_rpy,      False),
    ('$X_c^+$',    'XcPlus',  XcPlus,  XcPlus_rpy,  False),
    ('$Y_c^+$',    'YcPlus',  YcPlus,  YcPlus_rpy,  False),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Wall Sup scalars: C++ data, RPY resistance, near-contact asymptotic, far-field',
             fontsize=13)

for ax, (label, name, data_cpp, data_rpy, known_ff) in zip(axes.flat, scalar_info):
    limit = cpp_limit[name]

    # C++ data
    mask = np.abs(data_cpp) > 1e-15
    ax.loglog(epsh[mask], np.abs(data_cpp[mask]),
              lw=2.5, color='steelblue', label='C++ data')

    # RPY resistance
    mask = np.abs(data_rpy) > 1e-15
    ax.loglog(epsh[mask], np.abs(data_rpy[mask]),
              lw=2.0, color='darkorange', linestyle='--', label='RPY resistance')

    # near-contact asymptotic (plot up to 2x the validity limit)
    eps_nc  = eps_fine[eps_fine <= limit * 2]
    asym_nc = asym_wall(name, eps_nc)
    mask_nc = np.abs(asym_nc) > 1e-15
    ax.loglog(eps_nc[mask_nc], np.abs(asym_nc[mask_nc]),
              lw=2.0, color='tomato', linestyle='--', label='near-contact asym.')
    ax.axvline(limit, color='tomato', lw=1.0, linestyle=':',
               label=f'validity limit $\\epsilon_w={limit}$')

    # far-field
    eps_far = eps_fine[eps_fine > 1.0]
    h_far   = eps_far + 1.0
    if known_ff:
        ff = farfield_known(name, h_far)
        mask_ff = np.abs(ff) > 1e-15
        ax.loglog(eps_far[mask_ff], np.abs(ff[mask_ff]),
                  lw=2.0, color='seagreen', linestyle='-.', label='Faxen $-9/(8h)$')
    else:
        C, alpha = powerlaw_fits.get(name, (None, None))
        if C is not None:
            ff = C * h_far**alpha
            mask_ff = np.abs(ff) > 1e-15
            ax.loglog(eps_far[mask_ff], np.abs(ff[mask_ff]),
                      lw=2.0, color='seagreen', linestyle='-.',
                      label=f'RPY power law $h^{{{alpha:.2f}}}$')

    ax.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=11)
    ax.set_ylabel('|scalar|',                fontsize=11)
    ax.set_title(label,                      fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes.flat[-1].set_visible(False)
plt.tight_layout()
plt.show()