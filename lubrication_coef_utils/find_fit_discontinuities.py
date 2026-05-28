"""
find_fit_discontinuities.py
---------------------------
Evaluates the RPY and Delta_R rational fits on a very fine grid (spacing ~1e-6)
and detects sharp changes / discontinuities by looking for large values of
|df/dh| / |f| (relative derivative).  Plots each fit with problem points
marked in red and prints their gap locations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

# =============================================================================
# Coefficients
# =============================================================================
H_SCALE_RPY = 0.5
h_scale_fit  = 0.5

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

delta_coeffs = {
    'Xa_corr': dict(
        asym_cut = 2.0549e-01, rpy_cut = 7.0, p_min = 1,
        pc = [3.390440744007662e-03,  -1.292707770613473e-01,  1.621030547158851e+00],
        qc = [-9.934759618046485e+00,  3.852782844267631e+01,  -6.919767486523838e+01,  4.812890822657162e+01],
    ),
    'Ya_corr': dict(
        asym_cut = 2.9118e-02, rpy_cut = 5.6, p_min = 1,
        pc = [-1.728773665526388e-03,  8.013222079624960e-03,  3.309887895636077e-01,
              -9.973447439024367e-01],
        qc = [-1.076837737152707e+01,  4.552251214892676e+01, -8.993351368693166e+01,
               6.985624347396001e+01],
    ),
    'Yb': dict(
        asym_cut = 1.0e-01, rpy_cut = 3.4, p_min = 4,
        pc = [-1.170746516949502e+00,  2.632828108588735e+01, -2.037613634506775e+02,
               6.606004672829661e+02, -7.648826135646909e+02],
        qc = [-1.458911690535435e+01,  7.822372987710401e+01, -1.822480390025351e+02,
               1.556764973800673e+02],
    ),
    'XcPlus': dict(
        asym_cut = 9.7e-03, rpy_cut = 4.0e-01, p_min = 3,
        pc = [-6.769668477673338e-02,  4.251981921512744e-01, -4.761218699317399e-01,
              -5.662068770688917e-01],
        qc = [-7.986015731409064e+00,  2.107903159501786e+01, -1.836099133069141e+01],
    ),
    'YcPlus': dict(
        asym_cut = 4.56e-02, rpy_cut = 5.0, p_min = 3,
        pc = [ 4.777639575123545e-02, -1.200574927120337e+00,  1.143761945785224e+01,
              -4.755342967407084e+01,  7.229862934356090e+01],
        qc = [-1.420293268437043e+01,  7.365993492906483e+01, -1.644241955903594e+02,
               1.329482129469521e+02],
    ),
}

# =============================================================================
# Evaluation functions
# =============================================================================
def eval_rpy(h, d, h_scale=H_SCALE_RPY):
    h = np.asarray(h, dtype=float)
    u = 1.0 / (1.0 + h / h_scale)
    P = sum(d['p'][i] * u**(d['p_min'] + i) for i in range(len(d['p'])))
    Q = 1.0 + sum(d['q'][i] * u**(i + 1)   for i in range(len(d['q'])))
    return P / Q


def eval_dr(h, d, h_scale=h_scale_fit):
    h = np.asarray(h, dtype=float)
    u = 1.0 / (1.0 + h / h_scale)
    P = sum(d['pc'][i] * u**(d['p_min'] + i) for i in range(len(d['pc'])))
    Q = 1.0 + sum(d['qc'][i] * u**(i + 1)    for i in range(len(d['qc'])))
    return P / Q


# =============================================================================
# Discontinuity detection
# =============================================================================
from scipy.ndimage import generic_filter

def find_problem_points(h_vals, f_vals, threshold_multiplier=50.0,
                        window=2000, trim_edges=50):
    """
    Detect genuine discontinuities by comparing |Δf/Δh| to a sliding
    median rather than the global median.  A genuine pole will be orders
    of magnitude above its local neighbourhood; a steeply-sloped but smooth
    region will have a high local median and not be flagged.

    threshold_multiplier: flag if rate > this * local_median
    window:               half-width of sliding median window (samples)
    trim_edges:           ignore this many samples at each end
    """
    dh       = np.diff(h_vals)
    df       = np.diff(f_vals)
    abs_rate = np.abs(df / dh)
    finite   = abs_rate[np.isfinite(abs_rate)]
    if len(finite) == 0:
        return np.array([], dtype=int)
    abs_rate = np.where(np.isfinite(abs_rate), abs_rate, finite.max() * 10)

    # fast sliding median via scipy
    local_median = generic_filter(abs_rate, np.median,
                                  size=2 * window + 1, mode='nearest')
    local_median = np.maximum(local_median, 1e-30)
    ratio        = abs_rate / local_median

    n        = len(abs_rate)
    interior = np.zeros(n, dtype=bool)
    interior[trim_edges: n - trim_edges] = True
    spikes   = np.where((ratio > threshold_multiplier) & interior)[0]

    if len(spikes) == 0:
        return spikes

    clusters, cluster = [[spikes[0]]], [spikes[0]]
    for s in spikes[1:]:
        if s - cluster[-1] <= 5:
            cluster.append(s)
        else:
            clusters.append(cluster)
            cluster = [s]
    clusters.append(cluster)
    return np.array([c[np.argmax(ratio[c])] for c in clusters[:-1]] +
                    [clusters[-1][np.argmax(ratio[clusters[-1]])]])


# =============================================================================
# Main: scan each fit, detect problems, plot
# =============================================================================
names = ['Xa_corr', 'Ya_corr', 'Yb', 'XcPlus', 'YcPlus']
N_POINTS = 2_000_000   # fine grid spacing ~1e-6 over typical ranges

print("=" * 65)
print("Scanning fits for discontinuities on fine grid")
print("=" * 65)

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle("RPY (top) and Delta_R (bottom) fits — red = problem points",
             fontsize=13)

for col, name in enumerate(names):
    # ── RPY ──────────────────────────────────────────────────────────────
    ax_rpy = axes[0, col]
    h_rpy  = 1.0 + np.linspace(1e-4, 10.0, N_POINTS)
    f_rpy  = eval_rpy(h_rpy, RPY_COEFFS[name])

    prob_rpy = find_problem_points(h_rpy, f_rpy)
    if prob_rpy.size:
        print(f"\nRPY  {name}:")
        for idx in prob_rpy:
            print(f"  problem near h = {h_rpy[idx]:.6e}  "
                  f"(eps = {h_rpy[idx]-1:.6e})  f = {f_rpy[idx]:.4e}")

    # clip for plotting
    f_plot = np.clip(f_rpy, -1e4, 1e4)
    ax_rpy.plot(h_rpy - 1, f_plot, lw=0.8, color='steelblue')
    if prob_rpy.size:
        ax_rpy.plot(h_rpy[prob_rpy] - 1, f_plot[prob_rpy],
                    'ro', ms=6, zorder=5, label='problem')
        ax_rpy.legend(fontsize=7)
    ax_rpy.set_title(f'RPY {name}', fontsize=9)
    ax_rpy.set_xlabel(r'$\epsilon = h-1$', fontsize=8)
    ax_rpy.grid(True, which='both', alpha=0.3)
    ax_rpy.set_xlim(h_rpy[0] - 1, h_rpy[-1] - 1)

    # ── Delta_R ───────────────────────────────────────────────────────────
    ax_dr = axes[1, col]
    d     = delta_coeffs[name]
    h_lo  = d['asym_cut'] + 1.0
    h_hi  = d['rpy_cut']  + 1.0
    h_dr  = np.linspace(h_lo, h_hi, N_POINTS)
    f_dr  = eval_dr(h_dr, d)

    prob_dr = find_problem_points(h_dr, f_dr)
    if prob_dr.size:
        print(f"\nDR   {name}  [eps in ({d['asym_cut']:.4e}, {d['rpy_cut']:.4e})]:")
        for idx in prob_dr:
            print(f"  problem near h = {h_dr[idx]:.6e}  "
                  f"(eps = {h_dr[idx]-1:.6e})  f = {f_dr[idx]:.4e}")

    f_plot_dr = np.clip(f_dr, -1e4, 1e4)
    ax_dr.plot(h_dr - 1, f_plot_dr, lw=0.8, color='darkorange')
    if prob_dr.size:
        ax_dr.plot(h_dr[prob_dr] - 1, f_plot_dr[prob_dr],
                   'ro', ms=6, zorder=5, label='problem')
        ax_dr.legend(fontsize=7)
    ax_dr.set_title(f'DR {name}', fontsize=9)
    ax_dr.set_xlabel(r'$\epsilon = h-1$', fontsize=8)
    ax_dr.grid(True, which='both', alpha=0.3)
    ax_dr.set_xlim(h_dr[0] - 1, h_dr[-1] - 1)

if not any(find_problem_points(
        np.linspace(1e-4, 10.0, N_POINTS),
        eval_rpy(np.linspace(1e-4, 10.0, N_POINTS), RPY_COEFFS[n])).size
        for n in names):
    print("\nNo RPY problems detected.")
if not any(find_problem_points(
        np.linspace(delta_coeffs[n]['asym_cut']+1, delta_coeffs[n]['rpy_cut']+1, N_POINTS),
        eval_dr(np.linspace(delta_coeffs[n]['asym_cut']+1, delta_coeffs[n]['rpy_cut']+1, N_POINTS),
                delta_coeffs[n])).size
        for n in names):
    print("No Delta_R problems detected.")

print(f"\n{'='*65}")
plt.tight_layout()
plt.show()
