"""
fit_deltaR_wall_scalars.py
--------------------------
Fits the Delta_R = R_sup - R_RPY wall resistance correction using
rational functions in u = 1/(1 + h/h_scale).

Strategy:
  1. RPY scalars computed entirely from hardcoded rational fits (no libMobility)
  2. Load 2562-blob scalars; truncate negative eigenvalues of Delta_R (SPD preprocessing)
  3. Fit Delta_R scalars (not raw Sup scalars) to rational functions over a
     prescribed intermediate range [asym_cutoff, rpy_cutoff]
  4. Chimera: Delta_R = asym - RPY_fit  (near),  rational fit (mid),  0 (far)
     -> Sup = RPY_fit + Delta_R
  5. Monitor (not enforce) SPD conditions on Delta_R

Usage:
    python fit_deltaR_wall_scalars.py
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Parameters
# =============================================================================
a   = 1.0
eta = 1.0 / (6.0 * np.pi)
f0  = 6.0 * np.pi * eta * a
f1  = 6.0 * np.pi * eta * a**2
f2  = 6.0 * np.pi * eta * a**3

# =============================================================================
# 1. Hardcoded RPY rational fit coefficients
#    u = 1/(1 + h/h_scale),  h_scale=2.0
#    P(u) = sum(p_i * u^(p_min+i), i=0..4)
#    Q(u) = 1 + sum(q_i * u^(i+1), i=0..3)
# =============================================================================
H_SCALE_RPY = 2.0

RPY_COEFFS = {
    'Xa_corr': dict(p_min=1, p=[
        5.624811222421287e-01, -1.902140800834350e+00,  2.115451678636365e+00,
       -7.635996664143829e-01, -4.837500295844031e-03],
        q=[-4.944641980681546e+00,  9.159688702109271e+00,
           -7.511281900597266e+00,  2.291593234815973e+00]),
    'Ya_corr': dict(p_min=1, p=[
        2.812105339266384e-01, -1.115099808672645e+00,  1.464702518976790e+00,
       -6.325034511693657e-01, -6.999899056304174e-03],
        q=[-5.248201918594638e+00,  1.035754559569218e+01,
           -9.127978546090679e+00,  3.037666763264918e+00]),
    'Yb':      dict(p_min=4, p=[
       -7.392648118347235e-03,  3.368845461666825e-03,  1.612424189552847e-02,
       -1.298745899757570e-02,  9.455887027999391e-03],
        q=[-5.169293792382640e+00,  1.003175394314084e+01,
           -8.681787623181577e+00,  2.834997295430568e+00]),
    'XcPlus':  dict(p_min=3, p=[
        2.098387553712979e-02, -3.553195250282171e-02,  7.166486859377513e-03,
       -9.653580363206946e-03,  9.591411297605018e-03],
        q=[-4.632481897836154e+00,  7.956808419346241e+00,
           -6.033709799853070e+00,  1.716055087204692e+00]),
    'YcPlus':  dict(p_min=3, p=[
        5.238992353228791e-02, -1.079457649669765e-01,  3.418699552920573e-02,
       -2.084108061392004e-02,  5.170400576390023e-02],
        q=[-5.003405409201031e+00,  9.348095205572779e+00,
           -7.751092728329362e+00,  2.416658256123458e+00]),
}

def eval_rat(h, coeffs, h_scale=H_SCALE_RPY):
    """Evaluate rational fit at heights h."""
    h   = np.asarray(h, dtype=float)
    u   = 1.0 / (1.0 + h / h_scale)
    pm  = coeffs['p_min']
    p   = coeffs['p']
    q   = coeffs['q']
    P   = sum(p[i] * u**(pm + i) for i in range(len(p)))
    Q   = 1.0 + sum(q[i] * u**(i+1) for i in range(len(q)))
    return P / Q

def rpy_fit(name, h):
    return eval_rat(h, RPY_COEFFS[name])

# =============================================================================
# 2. Load 2562-blob scalars and truncate Delta_R negative eigenvalues
# =============================================================================
ref_file = "./resistance_coeffs/res_scalars_wall_MB_2562.txt"
ref_data = np.loadtxt(ref_file)
sort_idx = np.argsort(ref_data[:, 0])
ref_data = ref_data[sort_idx]
ref_h    = ref_data[:, 0]
ref_eps  = ref_h - 1.0

ref_raw = {
    'Xa_corr': ref_data[:, 1] - 1.0,
    'Ya_corr': ref_data[:, 2] - 1.0,
    'Yb':      ref_data[:, 3],
    'XcPlus':  ref_data[:, 4]*1.000 - 4.0/3.0,
    'YcPlus':  ref_data[:, 5]*1.000 - 4.0/3.0,
}
print(f"Loaded {len(ref_h)} rows  h: {ref_h.min():.4f}..{ref_h.max():.4f}")

def build_R_corr(Xa_c, Ya_c, Yb, XcP, YcP):
    R = np.zeros((6, 6))
    R[0,0]=f0*Ya_c; R[1,1]=f0*Ya_c; R[2,2]=f0*Xa_c
    R[3,3]=f2*YcP;  R[4,4]=f2*YcP;  R[5,5]=f2*XcP
    R[0,4]= f1*Yb;  R[4,0]= f1*Yb
    R[1,3]=-f1*Yb;  R[3,1]=-f1*Yb
    return R

print("Preprocessing: truncating negative Delta_R eigenvalues...")
n_trunc   = 0
ref_trunc = {k: ref_raw[k].copy() for k in ref_raw}

for idx, h in enumerate(ref_h):
    rv = {k: float(rpy_fit(k, h)) for k in ref_raw}
    R_rpy = build_R_corr(rv['Xa_corr'], rv['Ya_corr'],
                         rv['Yb'], rv['XcPlus'], rv['YcPlus'])
    R_ref = build_R_corr(ref_raw['Xa_corr'][idx], ref_raw['Ya_corr'][idx],
                         ref_raw['Yb'][idx], ref_raw['XcPlus'][idx],
                         ref_raw['YcPlus'][idx])
    Delta = R_ref - R_rpy
    eigvals, eigvecs = np.linalg.eigh(Delta)
    if eigvals.min() < 0:
        n_trunc += 1
        Delta_trunc = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T
        R_trunc     = R_rpy + Delta_trunc
        ref_trunc['Xa_corr'][idx] = R_trunc[2,2] / f0
        ref_trunc['Ya_corr'][idx] = R_trunc[0,0] / f0
        ref_trunc['Yb'][idx]      = R_trunc[0,4] / f1
        ref_trunc['XcPlus'][idx]  = R_trunc[5,5] / f2
        ref_trunc['YcPlus'][idx]  = R_trunc[3,3] / f2

print(f"  Truncated {n_trunc}/{len(ref_h)} heights  "
      f"({100*n_trunc/len(ref_h):.1f}%)")

# Delta_R scalars (truncated 2562 - RPY fit) on reference grid
delta_ref = {}
for name in ref_raw:
    rpy_v = np.array([float(rpy_fit(name, h)) for h in ref_h])
    delta_ref[name] = ref_trunc[name] - rpy_v
    if name == 'Yb':
        scale = 1.05
        scale_mask = (ref_trunc[name] <= 1e-3)
        ref_trunc[name][scale_mask] *= scale
        delta_ref[name] = ref_trunc[name] - rpy_v


# =============================================================================
# Asymptotic formulas (updated)
# =============================================================================
def asym_wall(name, e):
    e  = np.maximum(e, 1e-300)
    le = np.log(e)
    d  = {
        'Xa_corr': 1.0/e - (1.0/5.0)*le + 0.971280 - 1.0,
        'Ya_corr': -(8.0/15.0)*le + 0.9588 - 1.0,
        'Yb':      (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.029
                               - (0.4576-0.2)*e),
        'XcPlus':  (4.0/3.0)*(1.20206 - 3.0*(np.pi**2/6.0-1.0)*e) - 4.0/3.0,
        'YcPlus':  (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*e) - 4.0/3.0,
    }
    return d[name]

def delta_asym(name, e, h):
    """Near-contact Delta_R: asymptotic - RPY_fit."""
    return asym_wall(name, e) - rpy_fit(name, h)

# =============================================================================
# 3. Tunable cutoffs and SK rational fit of Delta_R in midfield
# =============================================================================
cutoffs = {
    'Xa_corr': dict(asym=2.0549e-1, rpy=7.0),
    'Ya_corr': dict(asym=2.9118e-2, rpy=5.6),
    'Yb':      dict(asym=1.0e-1, rpy=3.4),
    'XcPlus':  dict(asym=3.0e-3,    rpy=4.0e-1),
    'YcPlus':  dict(asym=4.56e-2,   rpy=5.0),
}

# Per-scalar rational fit degrees (n_num, n_den) — tune independently
fit_degrees = {
    'Xa_corr': (5, 4),
    'Ya_corr': (4, 4),
    'Yb':      (5, 4),
    'XcPlus':  (5, 4),
    'YcPlus':  (5, 4),
}

H_SCALE_FIT = 0.5
N_ITER_SK   = 25

def sk_fit(u, data, p_start, n_num, n_den, n_iter=N_ITER_SK, weights=None):
    w  = np.ones(len(u)) if weights is None else np.asarray(weights, dtype=float)
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

def eval_fit(u, pc, qc, p_start):
    P = sum(pc[i] * u**(p_start+i) for i in range(len(pc)))
    Q = 1.0 + sum(qc[i] * u**(i+1) for i in range(len(qc)))
    return P / Q

# p_min for Delta_R fits (same as RPY since both decay similarly)
p_min_delta = {'Xa_corr': 1, 'Ya_corr': 1, 'Yb': 4, 'XcPlus': 3, 'YcPlus': 3}

print(f"\nFitting Delta_R rational functions (h_scale={H_SCALE_FIT})...")

fit_pc  = {}
fit_qc  = {}

for name in cutoffs:
    co       = cutoffs[name]
    e_lo     = co['asym']
    e_hi     = co['rpy']
    n_num, n_den = fit_degrees[name]
    mask_mid = (ref_eps >= e_lo) & (ref_eps <= e_hi) & (np.abs(delta_ref[name]) > 1e-15)
    h_fit    = ref_h[mask_mid]
    d_fit    = delta_ref[name][mask_mid]

    if mask_mid.sum() < n_num + n_den + 1:
        print(f"  {name:>8}: too few points ({mask_mid.sum()}) — skipping")
        fit_pc[name] = np.zeros(n_num)
        fit_qc[name] = np.zeros(n_den)
        continue

    u_fit  = 1.0 / (1.0 + h_fit / H_SCALE_FIT)
    ps     = p_min_delta[name]

    pc, qc = sk_fit(u_fit, d_fit, ps, n_num, n_den)

    recon  = eval_fit(u_fit, pc, qc, ps)
    rel_err = np.where(np.abs(d_fit) > 1e-3,
                   np.abs(recon - d_fit) / np.abs(d_fit),
                   np.abs(recon - d_fit))
    rel    = np.max(rel_err)
    print(f"  {name:>8}: n_num={n_num} n_den={n_den}  "
          f"max rel err = {rel:.2e}  ({mask_mid.sum()} pts)")
    fit_pc[name] = pc
    fit_qc[name] = qc

# =============================================================================
# 4. Build chimera on a fine grid
# =============================================================================
chi_eps = np.unique(np.concatenate([
    np.logspace(-4, np.log10(ref_eps.min()), 60),
    ref_eps,
    np.logspace(np.log10(ref_eps.max()), 2, 60),
]))
chi_h = chi_eps + 1.0

chimera_delta = {k: np.zeros(len(chi_eps)) for k in cutoffs}
chimera_sup   = {k: np.zeros(len(chi_eps)) for k in cutoffs}

for i, (e, h) in enumerate(zip(chi_eps, chi_h)):
    rpy_v = {k: float(rpy_fit(k, h)) for k in cutoffs}
    for name in cutoffs:
        co   = cutoffs[name]
        e_lo = co['asym']
        e_hi = co['rpy']
        if e < e_lo:
            dR = delta_asym(name, np.array([e]), h)[0]
        elif e <= e_hi:
            u  = 1.0 / (1.0 + h / H_SCALE_FIT)
            ps = p_min_delta[name]
            dR = eval_fit(np.array([u]), fit_pc[name], fit_qc[name], ps)[0]
        else:
            dR = 0.0
        chimera_delta[name][i] = dR
        chimera_sup[name][i]   = rpy_v[name] + dR


# =============================================================================
# 5. Monitor SPD conditions (do NOT enforce)
# =============================================================================
print("\nSPD monitor on chimera grid:")
n_neg = 0
n_neg_diag = {k: 0 for k in cutoffs}
n_schur_viol = 0
min_eigs = np.zeros(len(chi_eps))

for i, (e, h) in enumerate(zip(chi_eps, chi_h)):
    Xa = chimera_delta['Xa_corr'][i]
    Ya = chimera_delta['Ya_corr'][i]
    Yb = chimera_delta['Yb'][i]
    Xc = chimera_delta['XcPlus'][i]
    Yc = chimera_delta['YcPlus'][i]
    Delta = build_R_corr(Xa, Ya, Yb, Xc, Yc)
    eigv  = np.linalg.eigvalsh(Delta)
    min_eigs[i] = eigv.min()
    if eigv.min() < 0:
        n_neg += 1
    for name, val, scale in [('Xa_corr', Xa, f0), ('Ya_corr', Ya, f0),
                               ('Yb', Yb, 0.0),
                               ('XcPlus', Xc, f2), ('YcPlus', Yc, f2)]:
        if name != 'Yb' and val * scale < 0:
            n_neg_diag[name] += 1
    schur = f0*Ya * f2*Yc - (f1*Yb)**2
    if schur < 0:
        n_schur_viol += 1

print(f"  Negative eigenvalue: {n_neg}/{len(chi_eps)} points")
if n_neg > 0:
    neg_mask = min_eigs < 0
    print(f"  Min eigenvalue range: {min_eigs[neg_mask].min():.2e} .. "
          f"{min_eigs[neg_mask].max():.2e}")
for k, v in n_neg_diag.items():
    print(f"  Negative diagonal ({k}): {v} points")
print(f"  Schur violation (Ya*Yc < Yb^2 scaled): {n_schur_viol} points")

# collect Schur violation eps for plotting
schur_viol_mask = np.array([
    f0*chimera_delta['Ya_corr'][i] * f2*chimera_delta['YcPlus'][i]
    - (f1*chimera_delta['Yb'][i])**2 < 0
    for i in range(len(chi_eps))
])
schur_viol_eps = chi_eps[schur_viol_mask]

# =============================================================================
# 6. Save fitting coefficients to file
# =============================================================================
def save_fit_coeffs(out_path):
    """
    Save all fitting parameters to a text file:
      - h_scale for the Delta_R rational fits
      - cutoffs (asym and rpy) for each scalar
      - fit degrees (n_num, n_den) for each scalar
      - p_min for each scalar
      - rational fit coefficients p_0..p_{n_num-1}, q_0..q_{n_den-1}

    Format:
      # header comments
      h_scale_fit  <value>
      # name  asym_cutoff  rpy_cutoff  p_min  n_num  n_den  p_0..  q_0..
      Xa_corr  <asym>  <rpy>  <p_min>  <n_num>  <n_den>  <p_0> .. <q_0> ..
      ...
    """
    with open(out_path, 'w') as f:
        f.write("# Delta_R wall scalar rational fit coefficients\n")
        f.write("# Chimera: Delta_R = asym-RPY (eps<asym_cut), "
                "rational fit (asym_cut<=eps<=rpy_cut), 0 (eps>rpy_cut)\n")
        f.write("# Sup scalar = RPY_fit + Delta_R\n")
        f.write("#\n")
        f.write(f"# h_scale_fit = {H_SCALE_FIT}  (for Delta_R rational fits)\n")
        f.write(f"# h_scale_rpy = {H_SCALE_RPY}  (for RPY rational fits)\n")
        f.write("# u = 1/(1 + h/h_scale)\n")
        f.write("# P(u) = sum(p_i * u^(p_min+i), i=0..n_num-1)\n")
        f.write("# Q(u) = 1 + sum(q_i * u^(i+1), i=0..n_den-1)\n")
        f.write("#\n")
        f.write("h_scale_fit  {:.15e}\n".format(H_SCALE_FIT))
        f.write("#\n")
        f.write("# name  asym_cutoff  rpy_cutoff  p_min  n_num  n_den  "
                "p_0 .. p_{n_num-1}  q_0 .. q_{n_den-1}\n")

        for name in ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']:
            co      = cutoffs[name]
            e_lo    = co['asym']
            e_hi    = co['rpy']
            pm      = p_min_delta[name]
            n_num, n_den = fit_degrees[name]
            pc      = fit_pc[name]
            qc      = fit_qc[name]
            p_str   = '  '.join(f'{v:.15e}' for v in pc)
            q_str   = '  '.join(f'{v:.15e}' for v in qc)
            f.write(f"{name}  {e_lo:.10e}  {e_hi:.10e}  {pm}  "
                    f"{n_num}  {n_den}  {p_str}  {q_str}\n")

    print(f"\nFit coefficients saved to: {out_path}")

save_fit_coeffs("wall_deltaR_scalar_fits.txt")

# =============================================================================
# Figures
# =============================================================================
scalar_info = [
    ('$X_a$ corr', 'Xa_corr'),
    ('$Y_a$ corr', 'Ya_corr'),
    ('$Y_b$',      'Yb'),
    ('$X_c^+$',    'XcPlus'),
    ('$Y_c^+$',    'YcPlus'),
]

h_fine   = np.logspace(np.log10(chi_h.min()), np.log10(chi_h.max()), 2000)
eps_fine = h_fine - 1.0

# Figure 1: Delta_R — reference data, fit, and component curves
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('$\\Delta R$ scalars: 2562-trunc (dots), rational fit (line), '
              'asym-RPY (dashed)', fontsize=12)

for ax, (label, name) in zip(axes1.flat, scalar_info):
    co   = cutoffs[name]
    e_lo = co['asym']
    e_hi = co['rpy']

    ax.semilogy(ref_eps, np.abs(delta_ref[name]) + 1e-20, '.',
                ms=3, color='steelblue', label='2562 trunc $\\Delta R$')

    mask_mid = (chi_eps >= e_lo) & (chi_eps <= e_hi)
    u_mid = 1.0 / (1.0 + chi_h[mask_mid] / H_SCALE_FIT)
    ps    = p_min_delta[name]
    fit_v = eval_fit(u_mid, fit_pc[name], fit_qc[name], ps)
    ax.semilogy(chi_eps[mask_mid], np.abs(fit_v) + 1e-20,
                lw=2.5, color='purple', label='rational fit')

    eps_nc = eps_fine[eps_fine < e_lo*3]
    h_nc   = eps_nc + 1.0
    dR_nc  = delta_asym(name, eps_nc, h_nc)
    ax.semilogy(eps_nc, np.abs(dR_nc) + 1e-20,
                '--', lw=1.8, color='tomato', label='asym $-$ RPY')

    ax.axvline(e_lo, color='tomato',   lw=0.8, linestyle=':',
               label=f'asym cut ({e_lo:.3f})')
    ax.axvline(e_hi, color='seagreen', lw=0.8, linestyle=':',
               label=f'rpy cut ({e_hi:.2f})')

    if len(schur_viol_eps) > 0:
        viol_chi_vals = np.abs(np.interp(schur_viol_eps, chi_eps,
                                         chimera_delta[name])) + 1e-20
        ax.scatter(schur_viol_eps, viol_chi_vals, s=8, color='red',
                   zorder=5, label=f'Schur viol. ({len(schur_viol_eps)} pts)')

    ax.set_xlabel(r'$\epsilon_w$', fontsize=11)
    ax.set_ylabel('$|\\Delta R|$', fontsize=11)
    ax.set_xscale('log')
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes1.flat[-1].set_visible(False)
plt.tight_layout()

# Figure 2: chimera Sup scalars + 2562 raw + RPY fit
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Chimera Sup scalars vs 2562-raw, RPY fit', fontsize=13)

for ax, (label, name) in zip(axes2.flat, scalar_info):
    sup_v  = chimera_sup[name]
    raw_v  = ref_raw[name]

    h_rpy_ext = np.logspace(np.log10(chi_h.min()), 2, 800)
    rpy_ext   = np.array([float(rpy_fit(name, h)) for h in h_rpy_ext])
    mask_rpy  = np.abs(rpy_ext) > 1e-15
    ax.loglog(h_rpy_ext[mask_rpy] - 1.0, np.abs(rpy_ext[mask_rpy]),
              '--', lw=1.5, color='darkorange', label='RPY fit', zorder=2)

    mask_r = np.abs(raw_v) > 1e-15
    ax.loglog(ref_eps[mask_r], np.abs(raw_v[mask_r]),
              '.', ms=3, color='steelblue', label='2562 raw', zorder=3)

    mask_s = np.abs(sup_v) > 1e-15
    ax.loglog(chi_eps[mask_s], np.abs(sup_v[mask_s]),
              lw=2.5, color='purple', label='chimera Sup', zorder=4)

    ax.set_xlabel(r'$\epsilon_w$', fontsize=11)
    ax.set_ylabel('|scalar|', fontsize=11)
    ax.set_title(label, fontsize=11)
    co = cutoffs[name]
    ax.axvline(co['asym'], color='tomato',   lw=0.8, linestyle=':',
               label=f"asym cut ({co['asym']:.3f})")
    ax.axvline(co['rpy'],  color='seagreen', lw=0.8, linestyle=':',
               label=f"rpy cut ({co['rpy']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes2.flat[-1].set_visible(False)
plt.tight_layout()

plt.show()
