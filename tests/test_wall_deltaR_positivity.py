"""
test_wall_deltaR_positivity.py
------------------------------
Sweeps a single particle's wall gap from very small to large and plots the
smallest real eigenvalue of Delta_R = R_Sup - R_MB as a function of gap.

This isolates whether the negativity is a fit accuracy/cancellation problem
at small gaps in the wall lubrication corrections.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

from StokesianDynamics import Lubrication

# =============================================================================
# Parameters — match your simulation
# =============================================================================
a            = 1.395
eta          = 1.4e-3
debye_length = 1e-2
L_open       = np.array([0.0, 0.0, 0.0])   # no periodicity — wall only

# Gap sweep: h = (z - a) / a, i.e. surface-to-surface gap normalised by a
# Focus on small gaps where the problem was observed (h ~ 0.02 - 0.03)
n_points   = 2000
h_min      = 0.02    # very small gap
h_max      = 0.03     # up to cutoff
h_vals     = np.geomspace(h_min, h_max, n_points)   # log-spaced

# Threshold below which we consider an eigenvalue genuinely negative
# (filters floating-point noise which appears as ~1e-90 negatives in eigs)
NEG_THRESHOLD = -1e-10

LC = Lubrication(debye_length)


def min_real_eigenvalue(A):
    # Use dense solver — the 6x6 single-particle matrix is tiny, and
    # eigs produces ~1e-90 spurious negatives for near-zero eigenvalues.
    return float(np.linalg.eigvals(A.toarray()).real.min())


# =============================================================================
# Sweep single particle at varying wall gaps, no pair neighbours
# =============================================================================
lam_DR    = np.zeros(n_points)
lam_RSup  = np.zeros(n_points)
lam_RMB   = np.zeros(n_points)

no_neighbors = [np.array([], dtype=np.int32)]

for i, h in enumerate(h_vals):
    z   = a * (1.0 + h)           # particle centre height
    r   = [np.array([0.0, 0.0, z])]

    R_MB, R_Sup = LC.ResistCSC_both(r, no_neighbors, a, eta, L_open)
    DR = R_Sup - R_MB

    lam_DR[i]   = min_real_eigenvalue(DR)
    lam_RSup[i] = min_real_eigenvalue(R_Sup)
    lam_RMB[i]  = min_real_eigenvalue(R_MB)

# =============================================================================
# Find crossover gap
# =============================================================================
neg_mask = lam_DR < NEG_THRESHOLD
if neg_mask.any():
    h_cross = h_vals[neg_mask].max()
    h_onset = h_vals[neg_mask].min()
    print(f"Delta_R is NEGATIVE (< {NEG_THRESHOLD:.0e}) for gap h in "
          f"[{h_onset:.4e}, {h_cross:.4e}]")
    print(f"  z range: z/a in [{1+h_onset:.4f}, {1+h_cross:.4f}]")
    print(f"  gap range: {h_onset*a:.4f} to {h_cross*a:.4f} (physical units)")
    print(f"  Most negative eigenvalue: {lam_DR[neg_mask].min():.6e}")
    print(f"  At gap h = {h_vals[np.argmin(lam_DR)]:.6e}")
else:
    print(f"Delta_R is positive (> {NEG_THRESHOLD:.0e}) for all tested gaps.")

print(f"\nMin lam(R_Sup) over sweep: {lam_RSup.min():.6e}  (should be > 0)")
print(f"Min lam(R_MB)  over sweep: {lam_RMB.min():.6e}  (should be > 0)")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax = axes[0]
ax.semilogx(h_vals, lam_DR,   'b-',  lw=1.5, label=r'$\lambda_{\min}(\Delta R)$')
ax.axhline(0, color='k', lw=0.8, ls='--')
if neg_mask.any():
    ax.axvspan(h_vals[neg_mask].min(), h_vals[neg_mask].max(),
               alpha=0.15, color='red', label=f'Negative (< {NEG_THRESHOLD:.0e})')
ax.set_ylabel(r'$\lambda_{\min}$', fontsize=12)
ax.set_title(r'Single-particle $\Delta R$ eigenvalue vs wall gap', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which='both', ls=':', alpha=0.4)

ax = axes[1]
ax.loglog(h_vals, np.abs(lam_RSup), 'g-',  lw=1.5, label=r'$\lambda_{\min}(R_{\rm Sup})$')
ax.loglog(h_vals, np.abs(lam_RMB),  'r--', lw=1.5, label=r'$\lambda_{\min}(R_{\rm MB})$')
ax.set_xlabel(r'Gap $h = (z/a - 1)$', fontsize=12)
ax.set_ylabel(r'$|\lambda_{\min}|$', fontsize=12)
ax.set_title(r'$R_{\rm Sup}$ and $R_{\rm MB}$ eigenvalues (log scale)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, which='both', ls=':', alpha=0.4)

fig.tight_layout()
plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'wall_deltaR_positivity.png')
fig.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
