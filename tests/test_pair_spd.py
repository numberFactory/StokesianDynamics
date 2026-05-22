"""
test_pair_spd.py
----------------
Tests that Delta_R = R_sup - R_mb is symmetric positive semi-definite (SPSD)
for two particles at varying separations, far from the wall.

Usage:
    python test_pair_spd.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from StokesianDynamics import Lubrication

# =============================================================================
# Parameters
# =============================================================================
a      = 1.0
eta    = 1.0 / 6.0 / np.pi
d_cut  = 5e-3
cutoff = 4.5

periodic_length = np.array([0.0, 0.0, 0.0], dtype=np.float64)
z_safe          = 1e6 * a   # far from wall — wall contributions zero

gaps    = np.logspace(np.log10(d_cut * 1.01), np.log10(cutoff - 2.0 - 1e-6), 1000)
r_norms = gaps + 2.0

lub = Lubrication(d_cut)

# =============================================================================
# Compute Delta_R and check SPD at each separation
# =============================================================================
print(f"Testing {len(gaps)} separations in eps = [{gaps[0]:.3e}, {gaps[-1]:.3e}]")
print(f"Particles at z = {z_safe:.1e} (wall effects negligible)\n")

min_eigs     = np.zeros(len(gaps))
sym_errors   = np.zeros(len(gaps))

n_list = [np.array([1], dtype=np.int32),
          np.array([],  dtype=np.int32)]

for idx, r_norm in enumerate(r_norms):
    r1 = np.array([0.0,      0.0, z_safe], dtype=np.float64)
    r2 = np.array([r_norm*a, 0.0, z_safe], dtype=np.float64)

    R_sup = lub.ResistCSC([r1, r2], n_list, a, eta,
                          cutoff, 0.0, periodic_length, True ).toarray()
    R_mb  = lub.ResistCSC([r1, r2], n_list, a, eta,
                          cutoff, 0.0, periodic_length, False).toarray()

    Delta_R = R_sup - R_mb

    sym_errors[idx]   = np.max(np.abs(Delta_R - Delta_R.T)) / (np.max(np.abs(Delta_R)) + 1e-300)
    min_eigs[idx]     = np.linalg.eigvalsh(Delta_R).min()

# =============================================================================
# Summary
# =============================================================================
n_non_sym = np.sum(sym_errors  > 1e-10)
n_non_spd = np.sum(min_eigs    < 0.0)

print(f"Symmetry violations (|Delta_R - Delta_R^T|/|Delta_R| > 1e-10): "
      f"{n_non_sym} / {len(gaps)}")
print(f"SPD violations (min eigenvalue < 0):                            "
      f"{n_non_spd} / {len(gaps)}")

if n_non_spd > 0:
    worst_idx = np.argmin(min_eigs)
    print(f"\n  Worst violation: lambda_min = {min_eigs[worst_idx]:.3e} "
          f"at eps = {gaps[worst_idx]:.4e}")
    print("  [FAIL] Delta_R is NOT positive semi-definite at all separations.")
else:
    print(f"\n  Min eigenvalue across all separations: {min_eigs.min():.3e}")
    print("  [PASS] Delta_R is positive semi-definite at all tested separations.")

if n_non_sym > 0:
    print(f"  [FAIL] Delta_R is not symmetric at {n_non_sym} separations.")
else:
    print("  [PASS] Delta_R is symmetric at all tested separations.")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(r'$\Delta R = R_\mathrm{sup} - R_\mathrm{mb}$ pair test (no wall)',
             fontsize=13)

# Panel 1: minimum eigenvalue
ax = axes[0]
pos_mask = min_eigs >= 0
neg_mask = min_eigs <  0
if pos_mask.any():
    ax.semilogy(gaps[pos_mask], min_eigs[pos_mask],
                '.', color='steelblue', ms=3, label='positive')
if neg_mask.any():
    ax.semilogy(gaps[neg_mask], -min_eigs[neg_mask],
                'x', color='red', ms=5, label='negative (plotted as |val|)')
ax.set_xlabel(r'$\epsilon = r/a - 2$', fontsize=12)
ax.set_ylabel(r'$\lambda_\mathrm{min}(\Delta R)$', fontsize=12)
ax.set_title('Min eigenvalue of $\\Delta R$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

# Panel 2: symmetry error
ax = axes[1]
ax.loglog(gaps, sym_errors, '.', color='steelblue', ms=3)
ax.axhline(1e-10, color='k', lw=1.5, linestyle='--', label='$10^{-10}$ threshold')
ax.set_xlabel(r'$\epsilon = r/a - 2$', fontsize=12)
ax.set_ylabel(r'$\|\Delta R - \Delta R^T\| / \|\Delta R\|$', fontsize=12)
ax.set_title('Symmetry error of $\\Delta R$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()
