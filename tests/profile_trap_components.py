"""
profile_trap_components.py
--------------------------
Profiles the individual components of the Stokesian Dynamics trap step:
  a) Wall_Mobility_Mult
  b) DRhalf computation  (Delta_R^{1/2} * W1 via CHOLMOD)
  c) Mhalf computation   (sqrt(M)*W via libMobility)
  d) IpMDR_PC            (preconditioner apply)
  e) IpMDR_Mult          (operator apply)
  f) FT_calc             (force/torque evaluation)

Uses the same parameters and initial particle positions as test_fluctuating_dynamics.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.spatial as spatial
import scipy.sparse as sp
import time
from functools import partial
from numba import njit, prange
from scipy.spatial.transform import Rotation
from sksparse.cholmod import cholesky

from body import Body
from pyStokesianDynamics import pyStokesianDynamics

# =============================================================================
# Parameters — identical to test_fluctuating_dynamics.py
# =============================================================================
a             = 1.395
eta           = 1.4e-3
kT            = 0.004075
g             = 0.0592
dt            = 0.25
firm_delta    = 1e-2
debye_firm    = 2.0 * a * firm_delta / np.log(10.0)
repulsion_firm = 0.0163
repulsion_soft = 0.0
debye_soft    = 0.1395

Lx, Ly        = 256.0, 256.0
L             = np.array([Lx, Ly, 0.0])
z_max_solver  = 4.0 * (2.0 * a)
phi           = 0.34
N             = max(1, int(phi * Lx * Ly / (np.pi * a**2)))

pair_cutoff   = 2.0 * (2.0 * a)
buffer_skin   = pair_cutoff
nl_cutoff     = pair_cutoff + buffer_skin

N_REPEATS     = 10   # number of timing repetitions per component

print(f"N = {N} particles, phi = {phi:.3f}")

# =============================================================================
# Force kernels — identical to test_fluctuating_dynamics.py
# =============================================================================
@njit(parallel=True, fastmath=True)
def _wall_force_numba(r, a, g, rep_firm, deb_firm, firm_delta):
    N   = r.shape[0]
    f   = np.zeros((N, 3))
    deb = 0.5 * deb_firm
    for i in prange(N):
        f[i, 2] -= g
        h        = r[i, 2]
        contact  = a * (1.0 - firm_delta)
        if h > contact:
            f[i, 2] += (rep_firm / deb) * np.exp(-(h - contact) / deb)
        else:
            f[i, 2] += rep_firm / deb
    return f


@njit(parallel=True, fastmath=True)
def _pair_force_numba(r, L, a, rep_firm, deb_firm, firm_delta,
                      rep_soft, deb_soft, pair_cutoff, neighbors, offsets):
    N = r.shape[0]
    f = np.zeros((N, 3))
    for i in prange(N):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = neighbors[offsets[i] + kk]
            if j == i:
                continue
            dr = np.zeros(3)
            for d in range(3):
                dr[d] = r[j, d] - r[i, d]
                if L[d] > 0:
                    dr[d] -= int(dr[d] / L[d] + 0.5 * (
                        int(dr[d] > 0) - int(dr[d] < 0))) * L[d]
            rn = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            if rn > pair_cutoff:
                continue
            rn_safe     = max(rn, 1e-12)
            offset_firm = 2.0 * a * (1.0 - firm_delta)
            if rn > offset_firm:
                fmag = -(rep_firm / deb_firm) * np.exp(
                    -(rn - offset_firm) / deb_firm) / rn_safe
            else:
                fmag = -(rep_firm / deb_firm) / rn_safe
            for d in range(3):
                f[i, d] += fmag * dr[d]
    return f


def wrap_positions(r, L):
    r_w = r.copy()
    for dim in range(3):
        if L[dim] > 0:
            r_w[:, dim] = r_w[:, dim] % L[dim]
    return r_w


def build_neighbour_list(r_wrapped, L, cutoff):
    bs      = np.array([L[0] if L[0] > 0 else 1e30,
                        L[1] if L[1] > 0 else 1e30,
                        1e30])
    tree    = spatial.cKDTree(r_wrapped, boxsize=1.001 * bs)
    pairs   = tree.query_ball_tree(tree, cutoff)
    Np      = r_wrapped.shape[0]
    offsets = np.zeros(Np + 1, dtype=np.int64)
    for i in range(Np):
        offsets[i + 1] = offsets[i] + len(pairs[i])
    neighbors = np.concatenate(pairs).ravel().astype(np.int64) \
        if offsets[-1] > 0 else np.empty(0, dtype=np.int64)
    return neighbors, offsets


def force_torque_calculator(bodies, r_vecs, a, g, L, pair_cutoff,
                             rep_firm, deb_firm, firm_delta,
                             rep_soft, deb_soft, neighbors, offsets):
    r  = np.asarray(r_vecs, dtype=np.float64).reshape(-1, 3)
    f  = _wall_force_numba(r, a, g, rep_firm, deb_firm, firm_delta)
    f += _pair_force_numba(r, L, a, rep_firm, deb_firm, firm_delta,
                           rep_soft, deb_soft, pair_cutoff, neighbors, offsets)
    FT       = np.zeros((2 * r.shape[0], 3))
    FT[0::2] = f
    return FT


# =============================================================================
# Initialise particles — identical to test_fluctuating_dynamics.py
# =============================================================================
rng          = np.random.default_rng(42)
z_mean       = kT / g
min_sep      = 2.0 * a
max_attempts = 1000 * N

positions = []
attempts  = 0
while len(positions) < N and attempts < max_attempts:
    attempts += 1
    x     = rng.uniform(0.0, Lx)
    y     = rng.uniform(0.0, Ly)
    z_gap = rng.exponential(z_mean)
    z     = a + z_gap
    ok    = True
    for p in positions:
        dx = x - p[0]; dx -= round(dx / Lx) * Lx
        dy = y - p[1]; dy -= round(dy / Ly) * Ly
        dz = z - p[2]
        if dx**2 + dy**2 + dz**2 < min_sep**2:
            ok = False
            break
    if ok:
        positions.append([x, y, z])

if len(positions) < N:
    raise RuntimeError(f"Could only place {len(positions)}/{N} particles.")

identity_rot = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
bodies       = [Body(location=np.array(p), orientation=identity_rot)
                for p in positions]

# =============================================================================
# Initialise solver and build matrices
# =============================================================================
solver = pyStokesianDynamics(
    bodies=bodies, a=a, eta=eta,
    periodic_length=L, z_max=z_max_solver,
    debye_length=firm_delta,
)
solver.kT        = kT
solver.dt        = dt
solver.tolerance = 1e-4
solver.num_rejections_wall = 0
solver.num_rejections_jump = 0
solver.Set_R_Mats()

r_now     = np.array([b.location for b in bodies])
r_wrapped = wrap_positions(r_now, L)
nl_nbrs, nl_offsets = build_neighbour_list(r_wrapped, L, nl_cutoff)

FT_calc = partial(
    force_torque_calculator,
    a=a, g=g, L=L, pair_cutoff=pair_cutoff,
    rep_firm=repulsion_firm, deb_firm=debye_firm, firm_delta=firm_delta,
    rep_soft=repulsion_soft, deb_soft=debye_soft,
    neighbors=nl_nbrs, offsets=nl_offsets,
)

n_dof = 6 * N
r_vecs = [b.location.copy() for b in bodies]

# Build preconditioner (same as in Lubrication_solve)
small           = 6.0 * np.pi * eta * a * solver.tolerance
Eig_Shift_R_Sup = solver.R_Sup + sp.diags(small * np.ones(n_dof), 0, format='csc')
factor          = cholesky(Eig_Shift_R_Sup)

# Build Delta_R Cholesky for DRhalf
small_dr        = 1e-5 * 6.0 * np.pi * eta * a
Eig_Shift_DR    = solver.Delta_R + sp.diags(small_dr * np.ones(n_dof), 0, format='csc')
factor_dr       = cholesky(Eig_Shift_DR)

# Random test vectors
X_test = np.random.randn(n_dof)
W1     = np.random.randn(n_dof)

# =============================================================================
# Timing helper
# =============================================================================
def time_fn(label, fn, n=N_REPEATS):
    # warmup
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    print(f"  {label:<30s}  {arr.mean():8.2f} ± {arr.std():6.2f} ms  "
          f"[{arr.min():.2f}, {arr.max():.2f}]")
    return arr.mean()

# =============================================================================
# Profile each component
# =============================================================================
print(f"\n{'='*65}")
print(f"  Component timings  ({N_REPEATS} repeats each)")
print(f"{'='*65}")

# a) Wall_Mobility_Mult
t_mob = time_fn("a) Wall_Mobility_Mult",
                lambda: solver.Wall_Mobility_Mult(X_test))

# b) DRhalf = factor_dr.apply_Pt(factor_dr.L().dot(W1))
t_drhalf = time_fn("b) DRhalf (CHOLMOD L*W)",
                   lambda: factor_dr.apply_Pt(factor_dr.L().dot(W1)))

# c) Mhalf = sqrtMdotW via libMobility
def _mhalf():
    sqrtM_W_U, sqrtM_W_W = solver.solver.sqrtMdotW()
    return np.concatenate(
        (sqrtM_W_U.reshape(N, 3), sqrtM_W_W.reshape(N, 3)), axis=1).flatten()
t_mhalf = time_fn("c) Mhalf (sqrtMdotW)",  _mhalf)

# d) IpMDR_PC (preconditioner apply)
def _pc():
    RHS = solver.R_MB.dot(X_test)
    for k in solver.isolated:
        RHS[6*k:6*k+6] = 0.0
    Y = factor(RHS)
    for k in solver.isolated:
        Y[6*k:6*k+6] = X_test[6*k:6*k+6]
    return Y
t_pc = time_fn("d) IpMDR_PC (chol solve)", _pc)

# e) IpMDR_Mult (operator apply = Delta_R matvec + Mdot)
def _mult():
    D_R = solver.Delta_R.dot(X_test)
    return X_test + solver.Wall_Mobility_Mult(D_R)
t_mult = time_fn("e) IpMDR_Mult (DR*x+Mx)", _mult)

# f) FT_calc
t_ft = time_fn("f) FT_calc",
               lambda: FT_calc(bodies, r_vecs))

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*65}")
print(f"  Context from full trap step timings:")
print(f"    stochastic  ~235 ms  (DRhalf + MXm + divM + sqrtMdotW)")
print(f"    solve_pred  ~255 ms  (~GMRES iters * (IpMDR_Mult + IpMDR_PC))")
print(f"    solve_corr  ~257 ms")
print(f"{'='*65}")
print(f"  DRhalf + Mhalf subtotal:  {t_drhalf+t_mhalf:.2f} ms")
print(f"  One GMRES iter estimate:  {t_mult+t_pc:.2f} ms")
print(f"  => GMRES iters to reach 255ms: ~{255/(t_mult+t_pc):.0f}")
