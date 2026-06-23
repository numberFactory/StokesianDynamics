"""
fit_wall_sup_scalars.py
-----------------------
Three-region blended fit for wall Sup resistance scalars:

  Region 1 (eps < eps_lo):        exact AT near-contact asymptotic
  Region 2 (eps_lo <= h <= h_hi): rational fit P(u)/Q(u) via
                                  Sanathanan-Koerner iteration (linear lstsq)
  Region 3 (h > h_hi):            C/h^p  (integer p, C from RPY tail)

h_hi chosen to minimise mismatch between regions 2 and 3 near 20*a.

Rational fit:
  u = 1/(1 + h/h_scale)
  P(u) = sum p_i * u^(p_min+i),  i=0..n_num-1   [P(0)=0]
  Q(u) = 1 + sum q_i * u^(i+1),  i=0..n_den-1

SK iteration: each step is a linear lstsq — fast, no curve_fit needed.

Usage:
    python fit_wall_sup_scalars.py
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
h_max   = 25.0 * a
heights = np.unique(np.concatenate([
    np.logspace(np.log10(h_min), np.log10(1.1),   100),
    np.logspace(np.log10(1.1),   np.log10(h_max), 500),
]))
epsh = heights - 1.0

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
lub             = Lubrication(d_cut)

# =============================================================================
# Extract wall Sup scalars from C++ (training data)
# =============================================================================
print("Extracting wall Sup scalars from C++...")
Xa_corr = np.zeros(len(heights)); Ya_corr = np.zeros(len(heights))
Yb_wall = np.zeros(len(heights)); XcPlus  = np.zeros(len(heights))
YcPlus  = np.zeros(len(heights))

for idx, h in enumerate(heights):
    r1     = np.array([0.0, 0.0, h], dtype=np.float64)
    r2     = np.array([1e6, 0.0, h], dtype=np.float64)
    n_list = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]
    R_wall = lub.ResistCSC([r1, r2], n_list, a, eta,
                           cutoff, 1e10, periodic_length, True).toarray()
    Xa_corr[idx] = R_wall[2, 2] / f0
    Ya_corr[idx] = R_wall[0, 0] / f0
    Yb_wall[idx] = R_wall[0, 4] / f1
    XcPlus[idx]  = R_wall[5, 5] / f2
    YcPlus[idx]  = R_wall[3, 3] / f2
print("Done.")

cpp_dict = {
    'Xa_corr': Xa_corr, 'Ya_corr': Ya_corr, 'Yb': Yb_wall,
    'XcPlus':  XcPlus,  'YcPlus':  YcPlus,
}

# =============================================================================
# RPY — far-field coefficients only
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

print("Extracting RPY scalars (far-field C only)...")
Xa_corr_rpy = np.zeros(len(heights)); Ya_corr_rpy = np.zeros(len(heights))
Yb_rpy      = np.zeros(len(heights)); XcPlus_rpy  = np.zeros(len(heights))
YcPlus_rpy  = np.zeros(len(heights))

for idx, h in enumerate(heights):
    R = np.linalg.inv(form_mob_single(h))
    Xa_corr_rpy[idx] = R[2, 2] / f0 - 1.0
    Ya_corr_rpy[idx] = R[0, 0] / f0 - 1.0
    Yb_rpy[idx]      = R[0, 4] / f1
    XcPlus_rpy[idx]  = max(R[5, 5] / f2 - 4.0/3.0, 0.0)
    YcPlus_rpy[idx]  = max(R[3, 3] / f2 - 4.0/3.0, 0.0)
print("Done.")

rpy_dict = {
    'Xa_corr': Xa_corr_rpy, 'Ya_corr': Ya_corr_rpy, 'Yb': Yb_rpy,
    'XcPlus':  XcPlus_rpy,  'YcPlus':  YcPlus_rpy,
}

# =============================================================================
# Asymptotic formulas
# =============================================================================
def asym_wall(name, e):
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

cpp_limit = {
    'Xa_corr': 0.18, 'Ya_corr': 0.01,
    'Yb': 0.1275,    'XcPlus': 0.01, 'YcPlus': 0.1,
}
ff_power  = {'Xa_corr': 1, 'Ya_corr': 1, 'Yb': 4, 'XcPlus': 3, 'YcPlus': 3}
ff_exact  = {'Xa_corr': -9.0/8.0, 'Ya_corr': -9.0/16.0}

def farfield_eval(name, h, C1, C2):
    """
    Two-term far-field:
      Xa_corr, Ya_corr: C1/h   + C2/h^2
      Yb:               C1/h^4 + C2/h^5
      XcPlus, YcPlus:   C1/h^3 + C2/h^4
    """
    p = ff_power[name]
    return C1 / h**p + C2 / h**(p + 1)

# =============================================================================
# Step 1: fit three-term far-field from RPY tail
#   Xa_corr, Ya_corr: C1/h + C2/h^2 + C3/h^3  with C1 fixed to exact Faxen
#   Yb:               C1/h^4 + C2/h^5 + C3/h^6  (all fitted)
#   XcPlus, YcPlus:   C1/h^3 + C2/h^4 + C3/h^5  (all fitted)
# =============================================================================
H_FF_LO = 5.0
ff_C1 = {}
ff_C2 = {}

print("\nFar-field three-term fits from RPY tail (h > {:.1f}):".format(H_FF_LO))
for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
    rpy  = rpy_dict[name]
    mask = heights > H_FF_LO
    h_v  = heights[mask];  d_v = rpy[mask]
    p    = ff_power[name]

    if name in ('Xa_corr', 'Ya_corr'):
        # fit both terms: C1/h + C2/h^2
        A               = np.column_stack([1.0 / h_v,
                                           1.0 / h_v**2])
        coeffs, _, _, _ = np.linalg.lstsq(A, d_v, rcond=None)
        ff_C1[name]     = float(coeffs[0])
        ff_C2[name]     = float(coeffs[1])
        print(f"  {name:>8}: {ff_C1[name]:.6e}/h + {ff_C2[name]:.6e}/h^2"
              f"  (exact Faxen = {ff_exact[name]:.6e}/h)")
    else:
        # fit two terms: C1/h^p + C2/h^(p+1)
        A               = np.column_stack([1.0 / h_v**p, 1.0 / h_v**(p+1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, d_v, rcond=None)
        ff_C1[name]     = float(coeffs[0])
        ff_C2[name]     = float(coeffs[1])
        print(f"  {name:>8}: {ff_C1[name]:.6e}/h^{p} + {ff_C2[name]:.6e}/h^{p+1}")

# =============================================================================
# Far-field singularity analysis
# f(h) = C1/h^p + C2/h^(p+1)  =  h*(C1 + C2/h) / h^(p+1)
#                               =  (C1*h + C2) / h^(p+1)
# Numerator zero (singularity in the rational form): h = -C2/C1
# Denominator h^(p+1) only zero at h=0 (unphysical).
# A positive zero means a singularity in the physical domain h>0.
# =============================================================================
print("\nFar-field rational form singularity analysis:")
print("  f(h) = (C1*h + C2) / h^(p+1)")
print(f"  {'name':>8}  {'p':>3}  {'C1':>14}  {'C2':>14}  "
      f"{'zero h=-C2/C1':>16}  {'in domain?':>10}")
print("  " + "-"*74)
for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
    p  = ff_power[name]
    C1 = ff_C1[name]
    C2 = ff_C2[name]
    if abs(C1) > 1e-30:
        h_zero   = -C2 / C1
        in_domain = "YES (h>0)" if h_zero > 0 else "no  (h<0)"
    else:
        h_zero    = float('inf')
        in_domain = "no  (C1~0)"
    print(f"  {name:>8}  {p:>3}  {C1:>14.6e}  {C2:>14.6e}  "
          f"{h_zero:>16.4f}  {in_domain:>10}")

# =============================================================================
# Far-field accuracy check at very large h (absolute error)
# =============================================================================
print("\nFar-field accuracy check (fit vs RPY, absolute error):")
print(f"  {'name':>8}  {'h':>10}  {'fit':>14}  {'RPY':>14}  {'abs err':>12}")
print("  " + "-"*66)
for h_check in [1e3, 1e5, 1e7]:
    R_rpy_check    = np.linalg.inv(form_mob_single(h_check))
    rpy_vals_check = {
        'Xa_corr': R_rpy_check[2, 2] / f0 - 1.0,
        'Ya_corr': R_rpy_check[0, 0] / f0 - 1.0,
        'Yb':      R_rpy_check[0, 4] / f1,
        'XcPlus':  max(R_rpy_check[5, 5] / f2 - 4.0/3.0, 0.0),
        'YcPlus':  max(R_rpy_check[3, 3] / f2 - 4.0/3.0, 0.0),
    }
    for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
        fit_val = farfield_eval(name, h_check, ff_C1[name], ff_C2[name])
        rpy_val = rpy_vals_check[name]
        abs_err = abs(fit_val - rpy_val)
        print(f"  {name:>8}  {h_check:>10.0e}  {fit_val:>14.6e}"
              f"  {rpy_val:>14.6e}  {abs_err:>12.3e}")

# =============================================================================
# Step 2: Sanathanan-Koerner rational fit
#
#   P(u)/Q(u),  u = 1/(1 + h/h_scale)
#   P(u) = sum p_i * u^(p_start+i),  i=0..n_num-1
#   Q(u) = 1 + sum q_i * u^(i+1),    i=0..n_den-1
#
# SK iteration: minimise ||data/Q - P/Q||^2 by iterating
#   weights <- |Q_prev|
#   [P_cols | -data*Q_cols] @ [p; q] = data   (linear lstsq each step)
# =============================================================================
H_SCALE = 1.0
N_NUM   = 5
N_DEN   = 4
N_ITER  = 30    # SK iterations

def sk_fit(u, data, p_start, n_num, n_den, n_iter=N_ITER,
           point_weights=None):
    """
    Sanathanan-Koerner rational fit. Returns (p_coeffs, q_coeffs).
    point_weights: optional per-point importance weights (e.g. large for
    boundary-pinning points). Multiplied into the SK weights each iteration.
    """
    n = len(u)
    sk_weights = np.ones(n)
    p_coeffs   = np.zeros(n_num)
    q_coeffs   = np.zeros(n_den)
    pw         = np.ones(n) if point_weights is None else np.asarray(point_weights)

    for _ in range(n_iter):
        w = pw * sk_weights
        P_cols = np.column_stack(
            [u**(p_start + i) / w for i in range(n_num)])
        Q_cols = np.column_stack(
            [data * u**(i+1) / w for i in range(n_den)])
        A = np.hstack([P_cols, -Q_cols])
        b = data / w
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        p_coeffs = coeffs[:n_num]
        q_coeffs = coeffs[n_num:]
        Q_val    = 1.0 + sum(q_coeffs[i] * u**(i+1) for i in range(n_den))
        sk_weights = np.abs(Q_val) + 1e-10

    return p_coeffs, q_coeffs

def eval_rational(u, p_coeffs, q_coeffs, p_start):
    P = sum(p_coeffs[i] * u**(p_start + i) for i in range(len(p_coeffs)))
    Q = 1.0 + sum(q_coeffs[i] * u**(i+1)  for i in range(len(q_coeffs)))
    return P / Q

def fit_midfield(name, h_data, data, h_hi,
                 h_scale=H_SCALE, n_num=N_NUM, n_den=N_DEN):
    """
    Fit C++ data in [h_lo, h_hi] with SK rational function.
    Two boundary-pinning points are appended with weight = N_fit:
      - left  boundary at h_lo: value = AT asymptotic
      - right boundary at h_hi: value = far-field power law
    """
    h_lo  = 1.0 + cpp_limit[name]
    mask  = (h_data >= h_lo) & (h_data <= h_hi) & (np.abs(data) > 1e-15)
    h_fit = h_data[mask];  d_fit = data[mask]
    if h_fit.shape[0] < n_num + n_den + 1:
        return None, None, None, np.inf

    v_left  = asym_wall(name, np.array([cpp_limit[name]]))[0]
    v_right = farfield_eval(name, h_hi, ff_C1[name], ff_C2[name])
    bw      = float(len(h_fit))

    h_aug = np.concatenate([h_fit,   [h_lo],   [h_hi]])
    d_aug = np.concatenate([d_fit,   [v_left],  [v_right]])
    pw    = np.concatenate([np.ones(len(h_fit)), [bw], [bw]])

    u_aug   = 1.0 / (1.0 + h_aug / h_scale)
    p_start = ff_power[name]
    p_c, q_c = sk_fit(u_aug, d_aug, p_start, n_num, n_den, point_weights=pw)

    u_fit   = 1.0 / (1.0 + h_fit / h_scale)
    recon   = eval_rational(u_fit, p_c, q_c, p_start)
    rel_err = np.max(np.abs((recon - d_fit) / (np.abs(d_fit) + 1e-30)))
    Q_vals  = 1.0 + sum(q_c[i] * u_fit**(i+1) for i in range(n_den))
    q_ok    = bool(np.all(Q_vals > 0))

    return p_c, q_c, q_ok, rel_err

# =============================================================================
# Step 3: scan h_hi, pick minimum mismatch between regions 2 and 3
# =============================================================================
H_HI_MIN  = 5.0
H_HI_MAX  = 25.0
N_HI_SCAN = 60

print("\nFitting midfield (SK rational) and finding optimal h_hi...")

optimal_h_hi = {}
mid_p_coeffs = {}
mid_q_coeffs = {}

for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
    data = cpp_dict[name]
    h_candidates = np.linspace(H_HI_MIN, H_HI_MAX, N_HI_SCAN)

    best_mismatch = np.inf
    best_h_hi     = 20.0
    best_pc       = None
    best_qc       = None

    for h_hi in h_candidates:
        pc, qc, q_ok, rel_err = fit_midfield(name, heights, data, h_hi)
        if pc is None:
            continue

        # evaluate at h_hi
        u_hi    = 1.0 / (1.0 + h_hi / H_SCALE)
        rat_val = eval_rational(np.array([u_hi]), pc, qc, ff_power[name])[0]
        ff_val  = farfield_eval(name, h_hi, ff_C1[name], ff_C2[name])
        ref     = max(abs(rat_val), abs(ff_val), 1e-30)
        mismatch = abs(rat_val - ff_val) / ref

        # penalise if Q goes negative on training interval
        if not q_ok:
            mismatch += 10.0

        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_h_hi     = h_hi
            best_pc       = pc
            best_qc       = qc

    optimal_h_hi[name] = best_h_hi
    mid_p_coeffs[name] = best_pc
    mid_q_coeffs[name] = best_qc
    print(f"  {name:>8}: h_hi={best_h_hi:.2f}  mismatch={best_mismatch:.3e}")

# =============================================================================
# Build blended callables
# =============================================================================
def make_blended(name):
    limit  = cpp_limit[name]
    h_hi   = optimal_h_hi[name]
    pc     = mid_p_coeffs[name]
    qc     = mid_q_coeffs[name]
    ps     = ff_power[name]
    hs     = H_SCALE

    def fn(h_eval):
        h_eval   = np.asarray(h_eval, dtype=float)
        eps_eval = h_eval - 1.0
        result   = np.empty_like(h_eval)
        near = eps_eval < limit
        far  = h_eval > h_hi
        mid  = (~near) & (~far)
        if near.any():
            result[near] = asym_wall(name, eps_eval[near])
        if mid.any():
            u = 1.0 / (1.0 + h_eval[mid] / hs)
            result[mid] = eval_rational(u, pc, qc, ps)
        if far.any():
            result[far] = farfield_eval(name, h_eval[far], ff_C1[name], ff_C2[name])
        return result

    return fn

fit_callables = {n: make_blended(n)
                 for n in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']}

# residuals
print("\nResiduals on midfield training data:")
fit_recons = {}
for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
    data  = cpp_dict[name]
    limit = cpp_limit[name]
    h_hi  = optimal_h_hi[name]
    valid = (heights >= 1.0+limit) & (heights <= h_hi) & (np.abs(data) > 1e-15)
    h_v   = heights[valid];  d_v = data[valid]
    recon = fit_callables[name](h_v)
    rel_e = np.max(np.abs((recon - d_v) / (np.abs(d_v) + 1e-30)))
    print(f"  {name:>8}: max rel err = {rel_e:.2e}")
    fit_recons[name] = (h_v, recon, d_v)

# =============================================================================
# Figures
# =============================================================================
scalar_info = [
    ('$X_a$ corr', 'Xa_corr', Xa_corr),
    ('$Y_a$ corr', 'Ya_corr', Ya_corr),
    ('$Y_b$',      'Yb',      Yb_wall),
    ('$X_c^+$',    'XcPlus',  XcPlus),
    ('$Y_c^+$',    'YcPlus',  YcPlus),
]

h_ext   = np.logspace(np.log10(1.0+1e-5), np.log10(h_max*2), 3000)
eps_ext = h_ext - 1.0

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Three-region SK rational fit vs C++ data and RPY', fontsize=13)

for ax, (label, name, data_cpp) in zip(axes1.flat, scalar_info):
    limit = cpp_limit[name];  h_hi = optimal_h_hi[name]
    fit_v = fit_callables[name](h_ext)
    d_rpy = rpy_dict[name]

    mask = np.abs(data_cpp) > 1e-15
    ax.loglog(epsh[mask], np.abs(data_cpp[mask]),
              lw=2.5, color='steelblue', label='C++ data')
    mask = np.abs(d_rpy) > 1e-15
    ax.loglog(epsh[mask], np.abs(d_rpy[mask]),
              lw=1.5, color='darkorange', linestyle='--', label='RPY (ref)')
    mask = np.abs(fit_v) > 1e-15
    ax.loglog(eps_ext[mask], np.abs(fit_v[mask]),
              lw=2.5, color='purple', label='SK rational fit')
    eps_nc = eps_ext[eps_ext <= limit*2]
    anc    = asym_wall(name, eps_nc)
    ax.loglog(eps_nc[np.abs(anc)>1e-15], np.abs(anc[np.abs(anc)>1e-15]),
              lw=1.5, color='tomato', linestyle=':', label='near-contact asym.')
    ax.axvline(limit,       color='tomato',   lw=0.8, linestyle=':')
    ax.axvline(h_hi - 1.0, color='seagreen', lw=0.8, linestyle=':',
               label=f'h_hi={h_hi:.1f}')
    ax.set_xlabel(r'$\epsilon_w = h/a-1$', fontsize=11)
    ax.set_ylabel('|scalar|', fontsize=11)
    ax.set_title(f'{label}  (h_hi={h_hi:.1f})', fontsize=11)
    ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.3)

axes1.flat[-1].set_visible(False)
plt.tight_layout(); print("Figure 1 ready.")

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('SK rational fit residuals: |fit - C++| / |C++|', fontsize=13)

for ax, (label, name, _) in zip(axes2.flat, scalar_info):
    h_v, recon_v, data_v = fit_recons[name]
    rel_err = np.abs((recon_v - data_v) / (np.abs(data_v) + 1e-30))
    ax.loglog(h_v - 1.0, rel_err, lw=2.5)
    ax.axhline(1e-3, color='k', lw=1.5, linestyle=':',  label='$10^{-3}$')
    ax.axhline(1e-6, color='k', lw=1.5, linestyle='--', label='$10^{-6}$')
    ax.set_xlabel(r'$\epsilon_w$', fontsize=11)
    ax.set_ylabel('relative error', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

axes2.flat[-1].set_visible(False)
plt.tight_layout(); print("Figure 2 ready.")

# =============================================================================
# Standalone function: build 6x6 wall Sup resistance matrix from fitted scalars
#
# The wall resistance matrix has the structure (from WallResistMatrix in C++):
#   R_wall[0,0] = f0*(Ya-1)    R_wall[2,2] = f0*(Xa-1)
#   R_wall[3,3] = f2*YcPlus    R_wall[4,4] = f2*YcPlus
#   R_wall[5,5] = f2*XcPlus
#   R_wall[0,4] = f1*Yb        R_wall[1,3] = -f1*Yb
#   R_wall[3,0] = f1*Yb        (symmetric)
# i.e. the 6x6 block for particle j:
#   dofs: [Ux, Uy, Uz, Wx, Wy, Wz]
# =============================================================================
def build_wall_R_fit(h_val, mob_factor):
    """
    Build the 6x6 wall Sup resistance correction matrix from the fitted scalars.
    h_val: height of particle centre above wall (in units of a).
    mob_factor: [f0, f1, f2] = [6*pi*eta*a, 6*pi*eta*a^2, 6*pi*eta*a^3]
    Returns a 6x6 numpy array.
    """
    f0_, f1_, f2_ = mob_factor
    h = np.asarray([h_val], dtype=float)

    Xa_c  = fit_callables['Xa_corr'](h)[0]   # = Xa - 1
    Ya_c  = fit_callables['Ya_corr'](h)[0]   # = Ya - 1
    Yb_   = fit_callables['Yb'](h)[0]
    XcP   = fit_callables['XcPlus'](h)[0]    # = max(Xc - 4/3, 0)
    YcP   = fit_callables['YcPlus'](h)[0]    # = max(Yc - 4/3, 0)

    R = np.zeros((6, 6))
    # TT blocks
    R[0, 0] = f0_ * Ya_c
    R[1, 1] = f0_ * Ya_c
    R[2, 2] = f0_ * Xa_c
    # RR blocks
    R[3, 3] = f2_ * YcP
    R[4, 4] = f2_ * YcP
    R[5, 5] = f2_ * XcP
    # TR coupling (Yb)
    R[0, 4] =  f1_ * Yb_;  R[4, 0] =  f1_ * Yb_
    R[1, 3] = -f1_ * Yb_;  R[3, 1] = -f1_ * Yb_
    return R

# =============================================================================
# SPD check: Delta_R = R_sup_fit - R_RPY_wall
# Computed at the same height grid used for data extraction.
# =============================================================================
print("\nChecking SPD of Delta_R = R_sup_fit - R_RPY_wall ...")

min_eigs  = np.zeros(len(heights))
bad_h     = []
bad_eps   = []

for idx, h in enumerate(heights):
    R_fit = build_wall_R_fit(h, mob_factor)

    # RPY wall resistance (6x6)
    R_rpy_mob = np.linalg.inv(form_mob_single(h))

    # compare fit correction vs RPY correction directly
    R_rpy_corr       = np.zeros((6, 6))
    R_rpy_corr[0, 0] = R_rpy_mob[0, 0] / f0 - 1.0
    R_rpy_corr[1, 1] = R_rpy_mob[1, 1] / f0 - 1.0
    R_rpy_corr[2, 2] = R_rpy_mob[2, 2] / f0 - 1.0
    R_rpy_corr[3, 3] = max(R_rpy_mob[3, 3] / f2 - 4.0/3.0, 0.0)
    R_rpy_corr[4, 4] = max(R_rpy_mob[4, 4] / f2 - 4.0/3.0, 0.0)
    R_rpy_corr[5, 5] = max(R_rpy_mob[5, 5] / f2 - 4.0/3.0, 0.0)
    R_rpy_corr[0, 4] = R_rpy_mob[0, 4] / f1;  R_rpy_corr[4, 0] = R_rpy_corr[0, 4]
    R_rpy_corr[1, 3] = R_rpy_mob[1, 3] / f1;  R_rpy_corr[3, 1] = R_rpy_corr[1, 3]

    # scale to resistance units for SPD check
    R_fit_scaled      = R_fit.copy()
    R_fit_scaled[0,0] *= f0; R_fit_scaled[1,1] *= f0; R_fit_scaled[2,2] *= f0
    R_fit_scaled[3,3] *= f2; R_fit_scaled[4,4] *= f2; R_fit_scaled[5,5] *= f2
    R_fit_scaled[0,4] *= f1; R_fit_scaled[4,0] *= f1
    R_fit_scaled[1,3] *= f1; R_fit_scaled[3,1] *= f1

    R_rpy_scaled      = R_rpy_corr.copy()
    R_rpy_scaled[0,0] *= f0; R_rpy_scaled[1,1] *= f0; R_rpy_scaled[2,2] *= f0
    R_rpy_scaled[3,3] *= f2; R_rpy_scaled[4,4] *= f2; R_rpy_scaled[5,5] *= f2
    R_rpy_scaled[0,4] *= f1; R_rpy_scaled[4,0] *= f1
    R_rpy_scaled[1,3] *= f1; R_rpy_scaled[3,1] *= f1

    Delta = R_fit_scaled - R_rpy_scaled
    eigv  = np.linalg.eigvalsh(Delta)
    min_eigs[idx] = eigv.min()
    if eigv.min() < 0:
        bad_h.append(h)
        bad_eps.append(h - 1.0)

n_bad = len(bad_h)
print(f"  {n_bad} / {len(heights)} heights have Delta_R not SPD")
if n_bad > 0:
    print(f"  Worst: lambda_min = {min_eigs.min():.3e} at h = {heights[np.argmin(min_eigs)]:.3f}")
else:
    print("  PASS: Delta_R is SPD at all tested heights.")

# =============================================================================
# Figure 3: 2-norm of fitted vs RPY wall correction matrices, with SPD markers
# =============================================================================
print("Building Figure 3 (matrix norms + SPD check)...")

norm_fit = np.zeros(len(heights))
norm_rpy = np.zeros(len(heights))

for idx, h in enumerate(heights):
    R_fit = build_wall_R_fit(h, mob_factor)
    R_rpy_mob = np.linalg.inv(form_mob_single(h))
    R_rpy_corr = np.zeros((6, 6))
    R_rpy_corr[0,0] = (R_rpy_mob[0,0]/f0 - 1.0)*f0
    R_rpy_corr[1,1] = (R_rpy_mob[1,1]/f0 - 1.0)*f0
    R_rpy_corr[2,2] = (R_rpy_mob[2,2]/f0 - 1.0)*f0
    R_rpy_corr[3,3] = max(R_rpy_mob[3,3]/f2 - 4.0/3.0, 0.0)*f2
    R_rpy_corr[4,4] = max(R_rpy_mob[4,4]/f2 - 4.0/3.0, 0.0)*f2
    R_rpy_corr[5,5] = max(R_rpy_mob[5,5]/f2 - 4.0/3.0, 0.0)*f2
    R_rpy_corr[0,4] = R_rpy_mob[0,4];  R_rpy_corr[4,0] = R_rpy_corr[0,4]
    R_rpy_corr[1,3] = R_rpy_mob[1,3];  R_rpy_corr[3,1] = R_rpy_corr[1,3]

    R_fit_scaled = R_fit.copy()
    R_fit_scaled[0,0]*=f0; R_fit_scaled[1,1]*=f0; R_fit_scaled[2,2]*=f0
    R_fit_scaled[3,3]*=f2; R_fit_scaled[4,4]*=f2; R_fit_scaled[5,5]*=f2
    R_fit_scaled[0,4]*=f1; R_fit_scaled[4,0]*=f1
    R_fit_scaled[1,3]*=f1; R_fit_scaled[3,1]*=f1

    norm_fit[idx] = np.linalg.norm(R_fit_scaled, ord=2)
    norm_rpy[idx] = np.linalg.norm(R_rpy_corr,   ord=2)

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.suptitle(r'Wall correction $\|R\|_2$: fit vs RPY, with non-SPD markers',
              fontsize=13)

mask_fit = norm_fit > 1e-15
mask_rpy = norm_rpy > 1e-15
ax3.loglog(epsh[mask_fit], norm_fit[mask_fit],
           lw=2.5, color='purple', label='fit $\\|R_{sup}^{wall}\\|_2$')
ax3.loglog(epsh[mask_rpy], norm_rpy[mask_rpy],
           lw=2.0, color='darkorange', linestyle='--',
           label='RPY $\\|R_{RPY}^{wall}\\|_2$')

if n_bad > 0:
    bad_eps_arr = np.array(bad_eps)
    bad_fit_norm = np.array([norm_fit[np.argmin(np.abs(heights - h))]
                             for h in bad_h])
    ax3.scatter(bad_eps_arr, bad_fit_norm, marker='x', s=80,
                color='red', zorder=5,
                label=f'non-SPD $\\Delta R$ ({n_bad} pts)')

ax3.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=12)
ax3.set_ylabel(r'$\|R^{wall}\|_2$', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, which='both', alpha=0.3)
plt.tight_layout()
print("Figure 3 ready.")


# =============================================================================
# Save coefficients
# =============================================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'wall_sup_scalar_fits.txt')

with open(out_path, 'w') as f:
    f.write("# Wall Sup scalar three-region SK rational fit\n")
    f.write("# Region 1 (eps<eps_lo):     exact AT asymptotic\n")
    f.write("# Region 2 (eps_lo<=h<=h_hi): P(u)/Q(u), u=1/(1+h/h_scale)\n")
    f.write("#   P(u)=sum(p_i*u^(p_min+i)), Q(u)=1+sum(q_i*u^(i+1))\n")
    f.write("# Region 3 (h>h_hi): C1/h^p + C2/h^(p+1)\n")
    f.write("#   Xa_corr, Ya_corr: both C1,C2 fitted (C1 ~ exact Faxen)\n")
    f.write("#   Yb, XcPlus, YcPlus: both fitted from RPY tail\n")
    f.write(f"# h_scale={H_SCALE}  n_num={N_NUM}  n_den={N_DEN}\n")
    f.write(f"# ff_power={ff_power}\n")
    f.write(f"# cpp_limits={cpp_limit}\n")
    f.write("# name  p_min  C1  C2  h_lo  h_hi  p_0..p_{n_num-1}  q_0..q_{n_den-1}\n")
    for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
        C1   = ff_C1[name]
        C2   = ff_C2[name]
        h_hi = optimal_h_hi[name]
        h_lo = 1.0 + cpp_limit[name]
        pm   = ff_power[name]
        pc   = mid_p_coeffs[name]
        qc   = mid_q_coeffs[name]
        cs   = ' '.join(f'{v:.15e}' for v in list(pc) + list(qc))
        f.write(f"{name}  {pm}  {C1:.15e}  {C2:.15e}  "
                f"{h_lo:.6e}  {h_hi:.6e}  {cs}\n")

print(f"\nCoefficients saved to {out_path}")
plt.show()