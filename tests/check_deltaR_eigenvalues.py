"""
check_deltaR_eigenvalues.py
---------------------------
Reads a debug_positions.npz file, builds a neighbour list, forms Delta_R
for each interacting pair, and checks for negative real eigenvalues.

Usage:
    python check_deltaR_eigenvalues.py debug_positions.npz
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.spatial as spatial
import scipy.sparse.linalg as spla

from StokesianDynamics import Lubrication
import scipy.sparse as sp

# =============================================================================
# Parameters — match your simulation
# =============================================================================
a            = 1.395
eta          = 1.4e-3
debye_length = 1e-2
Lx, Ly       = 256.0, 256.0
L            = np.array([Lx, Ly, 0.0])
PAIR_CUTOFF  = 4.5 * a   # hard-coded in lubrication fits


def wrap_positions(r, L):
    r_w = r.copy()
    for dim in range(3):
        if L[dim] > 0:
            r_w[:, dim] = r_w[:, dim] % L[dim]
    return r_w


def build_neighbour_list(r, L, cutoff):
    bs   = np.array([L[0] if L[0] > 0 else 1e30,
                     L[1] if L[1] > 0 else 1e30,
                     1e30])
    tree = spatial.cKDTree(r, boxsize=1.001 * bs)
    N    = r.shape[0]
    neighbors = []
    for j in range(N):
        idx   = tree.query_ball_point(r[j], r=cutoff)
        upper = [i for i in idx if i > j]
        neighbors.append(np.array(upper, dtype=np.int32))
    return neighbors


def min_real_eigenvalue(A):
    if A.shape[0] < 2:
        return float(np.linalg.eigvals(A.toarray())[0].real)
    try:
        eigvals, _ = spla.eigs(A, k=1, which='SR')
        return eigvals[0].real
    except Exception:
        # fall back to dense for small or unconverged cases
        return float(np.linalg.eigvals(A.toarray()).real.min())


# =============================================================================
# Load positions
# =============================================================================
npz_path = sys.argv[1] if len(sys.argv) > 1 else 'debug_positions.npz'
data     = np.load(npz_path)
step     = int(data['step'])
t        = float(data['time'])
r_now    = data['positions']   # (N, 3) unwrapped

print(f"Loaded: step={step}, time={t:.3f}, N={r_now.shape[0]} particles")

r_wrapped = wrap_positions(r_now, L)
N         = r_wrapped.shape[0]

# =============================================================================
# Build full Delta_R and check its smallest eigenvalue first
# =============================================================================
LC = Lubrication(debye_length)
r_vecs_list = [r_wrapped[j] for j in range(N)]
neighbors   = build_neighbour_list(r_wrapped, L, PAIR_CUTOFF)

R_MB, R_Sup = LC.ResistCSC_both(r_vecs_list, neighbors, a, eta, L)
Delta_R     = R_Sup - R_MB

print(f"\nFull Delta_R: shape={Delta_R.shape}, nnz={Delta_R.nnz}")
lam_full = min_real_eigenvalue(Delta_R)
print(f"Smallest real eigenvalue (full Delta_R): {lam_full:.6e}")
if lam_full < 0:
    print("  *** NEGATIVE eigenvalue in full Delta_R ***")

# =============================================================================
# Check Delta_R for each interacting pair individually
# =============================================================================
print(f"\nChecking Delta_R for each pair...")
n_pairs   = 0
n_neg     = 0

for j in range(N):
    for k in neighbors[j]:
        n_pairs += 1
        pair_idx = [j, k]

        # single-pair neighbour list: particle j has neighbour k only
        nb_pair = [np.array([1], dtype=np.int32),   # j's upper neighbour: k (local index 1)
                   np.array([], dtype=np.int32)]     # k has no upper neighbours in this pair

        r_pair = [r_wrapped[j], r_wrapped[k]]

        R_MB_p, R_Sup_p = LC.ResistCSC_both(r_pair, nb_pair, a, eta, L)
        DR_p = R_Sup_p - R_MB_p

        lam = min_real_eigenvalue(DR_p)

        if lam < 0:
            n_neg += 1
            # minimum-image separation
            dr      = r_wrapped[k] - r_wrapped[j]
            for dim in range(2):
                if L[dim] > 0:
                    dr[dim] -= round(dr[dim] / L[dim]) * L[dim]
            sep     = np.linalg.norm(dr)
            gap_j   = r_wrapped[j, 2] - a
            gap_k   = r_wrapped[k, 2] - a
            print(f"\n  Pair ({j}, {k})")
            print(f"    Periodic separation:  |r_jk| = {sep:.6f}  ({sep/a:.4f} a)")
            print(f"    dr vector (min image):        {dr}")
            print(f"    Wall gap j: z_j - a = {gap_j:.6f}  ({gap_j/a:.4f} a)")
            print(f"    Wall gap k: z_k - a = {gap_k:.6f}  ({gap_k/a:.4f} a)")
            print(f"    Smallest real eigenvalue:     {lam:.6e}")

            # ── Per-particle wall contribution ────────────────────────────
            # Each particle alone (no pair neighbours) — wall correction only
            for label, idx in [('j', j), ('k', k)]:
                r_single  = [r_wrapped[idx]]
                nb_single = [np.array([], dtype=np.int32)]
                R_MB_s, R_Sup_s = LC.ResistCSC_both(
                    r_single, nb_single, a, eta, L)
                DR_s = R_Sup_s - R_MB_s
                lam_s = min_real_eigenvalue(DR_s)
                flag  = '  *** NEGATIVE ***' if lam_s < 0 else ''
                print(f"    Wall-only Delta_R particle {label} "
                      f"(gap={r_wrapped[idx,2]-a:.4f}): "
                      f"lam_min = {lam_s:.6e}{flag}")

            # ── Pair far from wall — isolate pair resistance ───────────────
            # Shift both particles to z = 100*a, keeping their lateral
            # separation and relative z identical. At this height the wall
            # corrections are negligible so any negative eigenvalue must
            # come from the pair lubrication itself.
            z_far   = 100.0 * a
            r_j_far = r_wrapped[j].copy(); r_j_far[2] = z_far
            r_k_far = r_wrapped[k].copy(); r_k_far[2] = r_wrapped[k, 2] \
                      - r_wrapped[j, 2] + z_far   # preserve relative z

            r_pair_far = [r_j_far, r_k_far]
            nb_pair_far = [np.array([1], dtype=np.int32),
                           np.array([], dtype=np.int32)]

            # Use an open (non-periodic) geometry so the large z doesn't
            # interact with the wall in the solver — periodic_length z=0
            L_open = np.array([0.0, 0.0, 0.0])
            R_MB_f, R_Sup_f = LC.ResistCSC_both(
                r_pair_far, nb_pair_far, a, eta, L_open)
            DR_f  = R_Sup_f - R_MB_f
            lam_f = min_real_eigenvalue(DR_f)
            flag  = '  *** NEGATIVE ***' if lam_f < 0 else ''
            print(f"    Pair-only Delta_R (wall removed, z={z_far:.1f}): "
                  f"lam_min = {lam_f:.6e}{flag}")

            if lam_f >= 0 and lam < 0:
                print(f"    => Negative eigenvalue comes from WALL interaction")
            elif lam_f < 0:
                print(f"    => Negative eigenvalue comes from PAIR lubrication")

print(f"\nChecked {n_pairs} pairs. Found {n_neg} with negative eigenvalue.")

