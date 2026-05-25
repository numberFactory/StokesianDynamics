"""
fit_rpy_wall_scalars.py
-----------------------
Computes RPY single-wall resistance scalars on a dense grid,
fits each with a rational function P(u)/Q(u) in u = 1/(1 + h/h_scale),
and plots the results.

Grid: logspaced eps in [1e-4, 1e4] with extra linspaced points for eps < 0.1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from libMobility import NBody

# =============================================================================
# Parameters
# =============================================================================
a   = 1.0
eta = 1.0 / (6.0 * np.pi)
f0  = 6.0 * np.pi * eta * a
f1  = 6.0 * np.pi * eta * a**2
f2  = 6.0 * np.pi * eta * a**3

# =============================================================================
# Height grid: logspaced eps in [1e-4, 1e4] + linspaced near-contact
# =============================================================================
eps_log   = np.logspace(-4, 4, 100)
eps_lin   = np.linspace(1e-4, 0.1, 60)   # dense near-contact
eps_grid  = np.unique(np.concatenate([eps_lin, eps_log]))
h_grid    = eps_grid + 1.0

print(f"Grid: {len(eps_grid)} points,  eps in [{eps_grid.min():.1e}, {eps_grid.max():.1e}]")

# =============================================================================
# Compute RPY scalars
# =============================================================================
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

print("Computing RPY scalars...")
rpy = {k: np.zeros(len(h_grid)) for k in
       ['Xa_corr','Ya_corr','Yb','XcPlus','YcPlus']}

for i, h in enumerate(h_grid):
    R = np.linalg.inv(form_mob_single(h))
    rpy['Xa_corr'][i] = R[2,2]/f0 - 1.0
    rpy['Ya_corr'][i] = R[0,0]/f0 - 1.0
    rpy['Yb'][i]      = R[0,4]/f1
    rpy['XcPlus'][i]  = R[5,5]/f2 - 4.0/3.0
    rpy['YcPlus'][i]  = R[3,3]/f2 - 4.0/3.0
print("Done.")

# =============================================================================
# Rational fit: P(u)/Q(u),  u = 1/(1 + h/h_scale)
#
# P(u) = sum_{i=0}^{n_num-1} p_i * u^(p_min + i)   [P(0)=0: decays at large h]
# Q(u) = 1 + sum_{i=0}^{n_den-1} q_i^2 * u^(i+1)   [Q > 0 always]
#
# Sanathanan-Koerner iteration: each step is a linear lstsq
# =============================================================================
H_SCALE = 2.0
N_NUM   = 5
N_DEN   = 4
N_ITER  = 30

# Far-field power (leading decay of correction)
p_min = {'Xa_corr': 1, 'Ya_corr': 1, 'Yb': 4, 'XcPlus': 3, 'YcPlus': 3}

def sk_fit(u, data, p_start, n_num, n_den, n_iter=N_ITER, weights=None):
    """Sanathanan-Koerner rational fit. Returns (p_coeffs, q_coeffs)."""
    w  = np.ones(len(u)) if weights is None else np.asarray(weights)
    pc = np.zeros(n_num)
    qc = np.zeros(n_den)
    for _ in range(n_iter):
        P_cols = np.column_stack([u**(p_start+i) / w for i in range(n_num)])
        Q_cols = np.column_stack([data * u**(i+1) / w for i in range(n_den)])
        A      = np.hstack([P_cols, -Q_cols])
        coeffs, _, _, _ = np.linalg.lstsq(A, data/w, rcond=None)
        pc = coeffs[:n_num]
        qc = coeffs[n_num:]
        Q  = 1.0 + sum(qc[i] * u**(i+1) for i in range(n_den))
        w  = np.abs(Q) + 1e-10
    return pc, qc

def eval_rat(u, pc, qc, p_start):
    P = sum(pc[i] * u**(p_start+i) for i in range(len(pc)))
    Q = 1.0 + sum(qc[i] * u**(i+1) for i in range(len(qc)))
    return P / Q

def fit_scalar(name, h_data, data, h_scale=H_SCALE,
               n_num=N_NUM, n_den=N_DEN):
    u      = 1.0 / (1.0 + h_data / h_scale)
    ps     = p_min[name]
    pc, qc = sk_fit(u, data, ps, n_num, n_den)
    recon  = eval_rat(u, pc, qc, ps)
    rel_err = np.max(np.abs((recon - data) / (np.abs(data) + 1e-30)))

    # check Q > 0
    Q_vals = 1.0 + sum(qc[i] * u**(i+1) for i in range(n_den))
    if np.any(Q_vals <= 0):
        print(f"  WARNING: {name}: Q has non-positive values on training grid")

    def fn(h_eval):
        h_eval = np.asarray(h_eval, dtype=float)
        u_eval = 1.0 / (1.0 + h_eval / h_scale)
        return eval_rat(u_eval, pc, qc, ps)

    return fn, recon, rel_err, pc, qc

# =============================================================================
# Fit all scalars
# =============================================================================
print(f"\nFitting rational functions (h_scale={H_SCALE}, "
      f"n_num={N_NUM}, n_den={N_DEN}, SK iters={N_ITER})...")

fit_fns   = {}
fit_recon = {}
fit_pc    = {}
fit_qc    = {}

for name in ['Xa_corr','Ya_corr','Yb','XcPlus','YcPlus']:
    data = rpy[name]
    # only fit where the scalar is meaningfully large
    valid = np.abs(data) > 1e-3
    h_v   = h_grid[valid]
    d_v   = data[valid]
    fn, recon, rel_err, pc, qc = fit_scalar(name, h_v, d_v)
    large   = np.abs(d_v) > 1e-3
    err     = np.zeros_like(d_v)
    err[large]  = np.abs((recon[large]  - d_v[large])  / (np.abs(d_v[large])  + 1e-30))
    err[~large] = np.abs( recon[~large] - d_v[~large])
    print(f"  {name:>8}: max err = {err.max():.2e}  "
          f"({valid.sum()} training pts, {large.sum()} with |data|>1e-3)")
    fit_fns[name]   = fn
    fit_recon[name] = (h_v - 1.0, recon, d_v)
    fit_pc[name]    = pc
    fit_qc[name]    = qc

# =============================================================================
# Figure 1: RPY data + fit
# =============================================================================
scalar_info = [
    ('$X_a$ corr', 'Xa_corr'),
    ('$Y_a$ corr', 'Ya_corr'),
    ('$Y_b$',      'Yb'),
    ('$X_c^+$',    'XcPlus'),
    ('$Y_c^+$',    'YcPlus'),
]

h_fine   = np.logspace(np.log10(h_grid.min()), np.log10(h_grid.max()), 2000)
eps_fine = h_fine - 1.0

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle(f'RPY wall scalars + rational fit  '
              f'(h_scale={H_SCALE}, {N_NUM}/{N_DEN})', fontsize=13)

for ax, (label, name) in zip(axes1.flat, scalar_info):
    data = rpy[name]
    mask = np.abs(data) > 1e-15
    ax.loglog(eps_grid[mask], np.abs(data[mask]),
              'o', ms=3, color='darkorange', label='RPY data', zorder=3)

    fit_v = fit_fns[name](h_fine)
    mask_f = np.abs(fit_v) > 1e-15
    ax.loglog(eps_fine[mask_f], np.abs(fit_v[mask_f]),
              '-', lw=2.5, color='purple', label='rational fit')

    ax.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=11)
    ax.set_ylabel('|scalar|', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

axes1.flat[-1].set_visible(False)
plt.tight_layout()

# =============================================================================
# Figure 2: fit residuals
# =============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Rational fit residuals: |fit - RPY| / |RPY|', fontsize=13)

for ax, (label, name) in zip(axes2.flat, scalar_info):
    eps_v, recon_v, data_v = fit_recon[name]
    # relative error where |data| > 1e-3, absolute elsewhere
    large   = np.abs(data_v) > 1e-3
    err     = np.zeros_like(data_v)
    err[large]  = np.abs((recon_v[large]  - data_v[large])  / (np.abs(data_v[large])  + 1e-30))
    err[~large] = np.abs( recon_v[~large] - data_v[~large])
    label2  = 'rel err (|data|>1e-3) / abs err (|data|<=1e-3)'
    ax.loglog(eps_v, err + 1e-30, lw=2.0, color='steelblue')
    if np.any(~large):
        ax.loglog(eps_v[~large], err[~large], 'o', ms=3,
                  color='tomato', label='abs err region')
    ax.axhline(1e-3, color='k', lw=1.2, linestyle=':', label='$10^{-3}$')
    ax.axhline(1e-6, color='k', lw=1.2, linestyle='--', label='$10^{-6}$')
    ax.set_xlabel(r'$\epsilon_w$', fontsize=11)
    ax.set_ylabel('error', fontsize=11)
    ax.set_title(f'{label}\n{label2}', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes2.flat[-1].set_visible(False)
plt.tight_layout()

# =============================================================================
# Save coefficients
# =============================================================================
out_path = 'rpy_wall_scalar_fits.txt'
with open(out_path, 'w') as f:
    f.write("# RPY wall scalar rational fit coefficients\n")
    f.write(f"# u = 1/(1 + h/h_scale),  h_scale={H_SCALE}\n")
    f.write("# P(u) = sum(p_i * u^(p_min+i), i=0..n_num-1)  [P(0)=0]\n")
    f.write("# Q(u) = 1 + sum(q_i^2 * u^(i+1), i=0..n_den-1)  [Q>0]\n")
    f.write(f"# n_num={N_NUM}  n_den={N_DEN}\n")
    f.write(f"# p_min={p_min}\n")
    f.write("# name  p_min  p_0..p_{n_num-1}  q_0..q_{n_den-1}\n")
    for name in ['Xa_corr','Ya_corr','Yb','XcPlus','YcPlus']:
        pm  = p_min[name]
        pc  = fit_pc[name]
        qc  = fit_qc[name]
        cs  = ' '.join(f'{v:.15e}' for v in list(pc) + list(qc))
        f.write(f"{name}  {pm}  {cs}\n")
print(f"\nCoefficients saved to {out_path}")

plt.show()