"""
test_pyStokesianDynamics.py
----------------------------
Simple smoke test for pyStokesianDynamics.
Places N particles in a monolayer at z = 1.5*a above a wall with
in-plane packing fraction phi = 0.4, then checks that:
  1. The libMobility solver initialises without error.
  2. Delta_R, R_MB, and R_Sup are formed without error.
  3. Delta_R is a non-empty sparse matrix of the correct shape.

Timing is collected over n_trials random configurations and printed
as mean ± std ± min ± max for each function.

Run with:
    python test_pyStokesianDynamics.py
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
from pyStokesianDynamics import pyStokesianDynamics
from Body import Body

# =============================================================================
# Parameters
# =============================================================================
N        = 10000
a        = 1.0
eta      = 1.0
phi      = 0.4
z0       = 1.5 * a
z_max    = 10.0 * a
n_trials = 10

L_xy = np.sqrt(N * np.pi * a**2 / phi)
rng  = np.random.default_rng(42)


def make_positions(periodic):
    """Return N particle positions as a list of length-3 arrays."""
    if periodic:
        xy = rng.uniform(0.0, L_xy, size=(N, 2))
    else:
        xy = rng.uniform(-L_xy / 2, L_xy / 2, size=(N, 2))
    z = np.full(N, z0)
    return [np.array([xy[i, 0], xy[i, 1], z[i]]) for i in range(N)]


# =============================================================================
# Test helper
# =============================================================================
def run_test(label, periodic):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    if periodic:
        periodic_length = np.array([L_xy, L_xy, 0.0])
    else:
        periodic_length = np.array([0.0, 0.0, 0.0])

    t_init         = []
    t_set_r        = []
    t_wall_mob     = []
    t_delta_r_mult = []

    for trial in range(n_trials):
        positions = make_positions(periodic)
        bodies = [Body(location=p,
                       orientation=Rotation.random(random_state=rng))
                  for p in positions]

        # --- construction ----------------------------------------------------
        try:
            t0 = time.perf_counter()
            sd = pyStokesianDynamics(
                bodies=bodies,
                a=a,
                eta=eta,
                periodic_length=periodic_length,
                z_max=z_max,
            )
            t_init.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"  [FAIL] Trial {trial}: construction raised: {e}")
            return

        # --- Set_R_Mats ------------------------------------------------------
        try:
            t0 = time.perf_counter()
            sd.Set_R_Mats(r_vecs_np=positions)
            t_set_r.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"  [FAIL] Trial {trial}: Set_R_Mats raised: {e}")
            return

        # random force/torque vector for multiply tests
        X = rng.standard_normal(6 * N)

        # --- Wall_Mobility_Mult ----------------------------------------------
        try:
            r_vecs_np = np.array(positions)
            t0 = time.perf_counter()
            _ = sd.Wall_Mobility_Mult(X, r_vecs_np=r_vecs_np)
            t_wall_mob.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"  [FAIL] Trial {trial}: Wall_Mobility_Mult raised: {e}")
            return

        # --- Delta_R @ X (sparse matrix-vector multiply) --------------------
        try:
            t0 = time.perf_counter()
            _ = sd.Delta_R @ X
            t_delta_r_mult.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"  [FAIL] Trial {trial}: Delta_R @ X raised: {e}")
            return

        # correctness checks on the first trial only
        if trial == 0:
            expected_shape = (6 * N, 6 * N)
            for name, mat in [("R_MB",    sd.R_MB),
                               ("R_Sup",  sd.R_Sup),
                               ("Delta_R",sd.Delta_R)]:
                if mat is None:
                    print(f"  [FAIL] {name} is None after Set_R_Mats.")
                elif mat.shape != expected_shape:
                    print(f"  [FAIL] {name} shape {mat.shape} != expected {expected_shape}.")
                elif not sp.issparse(mat):
                    print(f"  [FAIL] {name} is not a sparse matrix.")
                elif mat.nnz == 0:
                    print(f"  [WARN] {name} has no nonzero entries.")
                else:
                    print(f"  [PASS] {name}: shape={mat.shape}, nnz={mat.nnz}.")

        if (trial + 1) % (n_trials//10) == 0:
            print(f"  completed {trial + 1}/{n_trials} trials...")

    # --- timing summary ------------------------------------------------------
    t_init         = np.array(t_init)         * 1e3
    t_set_r        = np.array(t_set_r)        * 1e3
    t_wall_mob     = np.array(t_wall_mob)     * 1e3
    t_delta_r_mult = np.array(t_delta_r_mult) * 1e3
    t_total        = t_init + t_set_r

    print(f"\n  ── Timing over {n_trials} trials ──────────────────────────────────")
    print(f"  {'function':<25} {'mean ms':>10} {'std ms':>10} {'min ms':>10} {'max ms':>10}")
    print(f"  {'-'*67}")
    for name, arr in [("__init__",          t_init),
                      ("Set_R_Mats",        t_set_r),
                      ("Wall_Mobility_Mult",t_wall_mob),
                      ("Delta_R @ X",       t_delta_r_mult),
                      ("total (init+set_r)",t_total)]:
        print(f"  {name:<25} {arr.mean():>10.1f} {arr.std():>10.1f} "
              f"{arr.min():>10.1f} {arr.max():>10.1f}")


# =============================================================================
# Run both cases
# =============================================================================
if __name__ == "__main__":
    run_test("Periodic domain  (DPStokes)", periodic=True)
    run_test("Non-periodic domain (NBody)", periodic=False)
    print(f"\n{'='*55}")
    print("  Tests complete.")
    print(f"{'='*55}\n")