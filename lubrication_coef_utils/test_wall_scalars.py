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
# RPY rational fits (hardcoded)
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
    h  = np.asarray(h, dtype=float)
    u  = 1.0 / (1.0 + h / h_scale)
    pm = coeffs['p_min']
    p  = coeffs['p'];  q = coeffs['q']
    P  = sum(p[i] * u**(pm+i) for i in range(len(p)))
    Q  = 1.0 + sum(q[i] * u**(i+1) for i in range(len(q)))
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
        pc = [-4.373370078409577e-04,  4.424560646016040e-03, -7.638777986146561e-02,
               9.810003614859475e+00, -3.385902988985162e+01],
        qc = [-1.105160500583412e+01,  4.616461895203592e+01, -8.736869111978241e+01,
               6.400332372662409e+01],
    ),
    'Ya_corr': dict(
        asym_cut = 2.9118e-02,
        rpy_cut  = 5.6,
        p_min    = 1,
        pc = [-1.802661869524016e-03,  9.468660270547602e-03,  3.218091451875633e-01,
              -9.812949264604398e-01],
        qc = [-1.077793936835893e+01,  4.559152441171614e+01, -9.009502525501624e+01,
               6.997760797588727e+01],
    ),
    'Yb': dict(
        asym_cut = 1.0e-01,
        rpy_cut  = 3.4,
        p_min    = 4,
        pc = [-1.171720418904620e+00,  2.634024743369987e+01, -2.037770517911233e+02,
               6.604148051813611e+02, -7.644275364225128e+02],
        qc = [-1.458404137489715e+01,  7.817171888830218e+01, -1.820761238925333e+02,
               1.554914960487163e+02],
    ),
    'XcPlus': dict(
        asym_cut = 3.0e-03,
        rpy_cut  = 4.0e-01,
        p_min    = 3,
        pc = [-1.363325750623006e-01,  1.661064454119597e+00, -7.478587775449657e+00,
               1.474504992172720e+01, -1.073379968056573e+01],
        qc = [-1.156154685363075e+01,  5.011347178843301e+01, -9.652115572123826e+01,
               6.970392297095620e+01],
    ),
    'YcPlus': dict(
        asym_cut = 4.56e-02,
        rpy_cut  = 5.0,
        p_min    = 3,
        pc = [ 4.774546185054501e-02, -1.200098819024352e+00,  1.143510786982436e+01,
              -4.754910568257404e+01,  7.229905586544142e+01],
        qc = [-1.420388909734784e+01,  7.366905048843127e+01, -1.644525346135324e+02,
               1.329770561654655e+02],
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
eps_grid = np.unique(np.concatenate([
    np.logspace(-4, -1, 120),
    np.logspace(-1,  1, 200),
    np.logspace( 1,  3,  80),
]))
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
