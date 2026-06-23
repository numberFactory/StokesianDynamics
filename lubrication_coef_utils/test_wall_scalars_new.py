"""
test_wall_scalars.py
--------------------
Standalone test: reads RPY fits (hardcoded) and Delta_R fits from
wall_deltaR_scalar_fits.txt, constructs the chimera Sup scalars, and plots:

Figure 1: R_sup, R_rpy, and asymptotics for each scalar (log-log)
Figure 2: minimum eigenvalue of Delta_R = R_sup_corr - R_rpy_corr vs eps
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Physical parameters
# =============================================================================
a   = 1.0
eta = 1.0 / (6.0 * np.pi)
f0  = 6.0 * np.pi * eta * a
f1  = 6.0 * np.pi * eta * a**2
f2  = 6.0 * np.pi * eta * a**3

# =============================================================================
# RPY rational fits (from rpy_wall_scalar_fits_corrected.txt)
#    u = 1/(1 + h/h_scale),  h_scale=0.5
#    P(u) = sum(p_i * u^(p_min+i), i=0..n_num-1)
#    Q(u) = 1 + sum(q_i * u^(i+1), i=0..n_den-1)
# =============================================================================
H_SCALE_RPY = 0.5

RPY_COEFFS = {
    'Xa_corr': dict(p_min=1, p=[
        2.249994311366312e+00, -9.302942542409326e+00,  1.018898977062156e+01,
       -1.461983146752135e-01],
        q=[-7.384833913349599e+00,  1.974855818648951e+01,
           -2.040618954066672e+01,  4.353438654549962e+00]),
    'Ya_corr': dict(p_min=1, p=[
        1.124980798925478e+00, -5.161218663062001e+00,  7.210326948819593e+00,
       -2.621851150554913e+00],
        q=[-6.713754414307387e+00,  1.706822775135723e+01,
           -1.933894500702774e+01,  7.413436778017460e+00]),
    'Yb':      dict(p_min=4, p=[
       -2.019389402582813e+00,  7.475392505257506e+00, -3.877269126222068e+00,
       -1.028617914833345e+00],
        q=[-8.683501976306239e+00,  2.835163248843032e+01,
           -4.234435891678831e+01,  2.531363165467517e+01]),
    'XcPlus':  dict(p_min=3, p=[
        1.333114620704834e+00, -4.215726080748013e+00, -5.775019233806103e-02,
        1.969213190855149e-01],
        q=[-6.164932696556416e+00,  1.247407137449759e+01,
           -1.135905628832895e+01,  6.141510604219781e+00]),
    'YcPlus':  dict(p_min=3, p=[
        3.339371808444304e+00, -1.406703500455225e+01,  8.727632354389423e+00,
        9.801957547519073e+00],
        q=[-7.179264725936619e+00,  1.782208382807917e+01,
           -1.884878181254139e+01,  8.912717591979181e+00]),
}

def eval_rat(h, coeffs, h_scale=H_SCALE_RPY):
    h   = np.asarray(h, dtype=float)
    u   = 1.0 / (1.0 + h / h_scale)
    pm  = coeffs['p_min']
    p   = coeffs['p']
    q   = coeffs['q']
    P   = sum(p[i] * u**(pm + i) for i in range(len(p)))
    Q   = 1.0 + sum(q[i]    * u**(i+1) for i in range(len(q)))
    return P / Q

def rpy_fit(name, h):
    return eval_rat(h, RPY_COEFFS[name])

# =============================================================================
# Asymptotic formulas
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
    return asym_wall(name, e) - rpy_fit(name, h)

# =============================================================================
# Delta_R fit parameters (hardcoded from wall_deltaR_scalar_fits.txt)
#   h_scale_fit = 0.5
#   u = 1/(1 + h/h_scale_fit)
#   P(u) = sum(p_i * u^(p_min+i), i=0..n_num-1)
#   Q(u) = 1 + sum(q_i * u^(i+1), i=0..n_den-1)
#   Chimera: Delta_R = asym-RPY (eps<asym_cut), fit (asym_cut<=eps<=rpy_cut), 0 (far)
# =============================================================================
h_scale_fit = 0.5

delta_coeffs = {
    'Xa_corr': dict(
        asym_cut = 2.0549e-01,
        rpy_cut  = 7.0,
        p_min    = 1,
        pc = [3.390440744007662e-03,  -1.292707770613473e-01,  1.621030547158851e+00],
        qc = [-9.934759618046485e+00,  3.852782844267631e+01,  -6.919767486523838e+01,  4.812890822657162e+01],
    ),
    'Ya_corr': dict(
        asym_cut = 2.9118e-02,
        rpy_cut  = 5.6,
        p_min    = 1,
        pc = [-1.728773665526388e-03,  8.013222079624960e-03,  3.309887895636077e-01,
              -9.973447439024367e-01],
        qc = [-1.076837737152707e+01,  4.552251214892676e+01, -8.993351368693166e+01,
               6.985624347396001e+01],
    ),
    'Yb': dict(
        asym_cut = 1.0e-01,
        rpy_cut  = 3.4,
        p_min    = 4,
        pc = [-1.170746516949502e+00,  2.632828108588735e+01, -2.037613634506775e+02,
               6.606004672829661e+02, -7.648826135646909e+02],
        qc = [-1.458911690535435e+01,  7.822372987710401e+01, -1.822480390025351e+02,
               1.556764973800673e+02],
    ),
    'XcPlus': dict(
        asym_cut = 9.7e-03,
        rpy_cut  = 4.0e-01,
        p_min    = 3,
        pc = [-6.769668477673338e-02,  4.251981921512744e-01, -4.761218699317399e-01,
              -5.662068770688917e-01],
        qc = [-7.986015731409064e+00,  2.107903159501786e+01, -1.836099133069141e+01],
    ),
    'YcPlus': dict(
        asym_cut = 4.56e-02,
        rpy_cut  = 5.0,
        p_min    = 3,
        pc = [ 4.777639575123545e-02, -1.200574927120337e+00,  1.143761945785224e+01,
              -4.755342967407084e+01,  7.229862934356090e+01],
        qc = [-1.420293268437043e+01,  7.365993492906483e+01, -1.644241955903594e+02,
               1.329482129469521e+02],
    ),
}

def eval_delta_fit(name, h):
    """Evaluate Delta_R rational fit at height h."""
    h  = np.asarray(h, dtype=float)
    d  = delta_coeffs[name]
    u  = 1.0 / (1.0 + h / h_scale_fit)
    pm = d['p_min'];  pc = d['pc'];  qc = d['qc']
    P  = sum(pc[i] * u**(pm+i) for i in range(len(pc)))
    Q  = 1.0 + sum(qc[i] * u**(i+1) for i in range(len(qc)))
    return P / Q

def sup_scalar(name, h):
    """Chimera Sup scalar at height h (scalar or array)."""
    h   = np.asarray(h, dtype=float)
    e   = h - 1.0
    d   = delta_coeffs[name]
    e_lo = d['asym_cut']
    e_hi = d['rpy_cut']
    out  = np.empty_like(h)
    for i in range(h.size):
        ei = e.flat[i]; hi = h.flat[i]
        rpy_v = float(rpy_fit(name, hi))
        if ei < e_lo:
            dR = float(delta_asym(name, np.array([ei]), hi)[0])
        elif ei <= e_hi:
            dR = float(eval_delta_fit(name, hi))
        else:
            dR = 0.0
        out.flat[i] = rpy_v + dR
    return out if out.ndim > 0 else float(out)

# =============================================================================
# Evaluation grid
# =============================================================================
# eps_grid = np.unique(np.concatenate([
#     np.logspace(-4, -1, 20),
#     np.geomspace(0.02, 0.03, 2000),
#     np.logspace(-1,  1, 20),
#     np.logspace( 1,  3,  20),
# ]))
eps_grid = np.geomspace(1e-1, 0.38, 100000)
h_grid = eps_grid + 1.0

# =============================================================================
# Build R_sup, R_rpy, Delta_R arrays
# =============================================================================
names = ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']

sup_vals = {k: sup_scalar(k, h_grid) for k in names}
rpy_vals = {k: np.array([float(rpy_fit(k, h)) for h in h_grid]) for k in names}
asym_vals = {k: asym_wall(k, eps_grid) for k in names}

def build_R_corr(Xa_c, Ya_c, Yb, XcP, YcP):
    R = np.zeros((6, 6))
    R[0,0]=f0*Ya_c; R[1,1]=f0*Ya_c; R[2,2]=f0*Xa_c
    R[3,3]=f2*YcP;  R[4,4]=f2*YcP;  R[5,5]=f2*XcP
    R[0,4]= f1*Yb;  R[4,0]= f1*Yb
    R[1,3]=-f1*Yb;  R[3,1]=-f1*Yb
    return R

print("Computing min eigenvalue of Delta_R on fine grid...")
min_eig = np.zeros(len(eps_grid))
for i in range(len(eps_grid)):
    R_sup = build_R_corr(sup_vals['Xa_corr'][i], sup_vals['Ya_corr'][i],
                         sup_vals['Yb'][i],       sup_vals['XcPlus'][i],
                         sup_vals['YcPlus'][i])
    R_rpy = build_R_corr(rpy_vals['Xa_corr'][i], rpy_vals['Ya_corr'][i],
                         rpy_vals['Yb'][i],       rpy_vals['XcPlus'][i],
                         rpy_vals['YcPlus'][i])
    Delta       = R_sup - R_rpy
    min_eig[i]  = np.linalg.eigvalsh(Delta).min()
print(f"  min eig range: {min_eig.min():.3e} .. {min_eig.max():.3e}")
print(f"  n_negative: {(min_eig < 0).sum()}/{len(eps_grid)}")

# =============================================================================
# Figure 1: R_sup, R_rpy, asymptotics
# =============================================================================
scalar_info = [
    ('$X_a$ corr', 'Xa_corr'),
    ('$Y_a$ corr', 'Ya_corr'),
    ('$Y_b$',      'Yb'),
    ('$X_c^+$',    'XcPlus'),
    ('$Y_c^+$',    'YcPlus'),
]

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Wall Sup scalars: chimera (purple), RPY fit (orange), '
              'asymptotic (red)', fontsize=13)

for ax, (label, name) in zip(axes1.flat, scalar_info):
    d    = delta_coeffs[name]
    e_lo = d['asym_cut']
    e_hi = d['rpy_cut']

    # RPY
    mask = np.abs(rpy_vals[name]) > 1e-15
    ax.loglog(eps_grid[mask], np.abs(rpy_vals[name][mask]),
              '--', lw=1.8, color='darkorange', label='RPY fit')

    # Sup chimera
    mask = np.abs(sup_vals[name]) > 1e-15
    ax.loglog(eps_grid[mask], np.abs(sup_vals[name][mask]),
              lw=2.5, color='purple', label='chimera Sup')

    # asymptotic (near-contact only)
    mask_nc = (eps_grid < e_lo * 4) & (np.abs(asym_vals[name]) > 1e-15)
    ax.loglog(eps_grid[mask_nc], np.abs(asym_vals[name][mask_nc]),
              ':', lw=1.8, color='tomato', label='asymptotic')

    ax.axvline(e_lo, color='tomato',   lw=0.8, linestyle=':',
               label=f'asym cut ({e_lo:.3f})')
    ax.axvline(e_hi, color='seagreen', lw=0.8, linestyle=':',
               label=f'rpy cut ({e_hi:.2f})')

    ax.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=11)
    ax.set_ylabel('|scalar|', fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

axes1.flat[-1].set_visible(False)
plt.tight_layout()

# =============================================================================
# Figure 2: min eigenvalue of Delta_R vs eps
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.suptitle(r'Min eigenvalue of $\Delta R = R_{sup} - R_{RPY}$ vs $\epsilon_w$',
              fontsize=13)

ax2.semilogx(eps_grid, min_eig, lw=2.0, color='steelblue', label='min eig')
ax2.axhline(0, color='k', lw=1.0, linestyle='--')

# shade negative regions
neg = min_eig < 0
if neg.any():
    ax2.fill_between(eps_grid, min_eig, 0,
                     where=neg, color='red', alpha=0.3,
                     label=f'negative ({neg.sum()} pts)')

ax2.set_xlabel(r'$\epsilon_w = h/a - 1$', fontsize=12)
ax2.set_ylabel(r'$\lambda_{\min}(\Delta R)$', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', alpha=0.3)
plt.tight_layout()

plt.show()
