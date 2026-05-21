"""
tetra_lub_mobility_test.py
---------------------------
Test: computes the lubrication-corrected mobility matrix for a tetrahedron
of 4 single-blob rigid bodies at varying heights above a wall, then plots
the relative error against reference blob-discretisation datasets loaded
from ./Tet_Data/.

Results are never saved — always freshly computed to avoid stale data.

Usage:
    python tetra_lub_mobility_test.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pyStokesianDynamics import pyStokesianDynamics
from Body import Body

# =============================================================================
# Physical parameters (from inputfile_tetra.dat)
# =============================================================================
a   = 1.0
eta = 0.05305164769

# =============================================================================
# Tetrahedron geometry (from tetra_test.clones)
# Columns: x y z qw qx qy qz
# =============================================================================
CLONES = np.array([
    [0.0,                  0.0,               4.449489742783178, 1.0, 0.0, 0.0, 0.0],
    [1.732050807568877,    0.0,               2.0,               1.0, 0.0, 0.0, 0.0],
    [-0.866025403784439,  -1.500000000000000, 2.0,               1.0, 0.0, 0.0, 0.0],
    [-0.866025403784439,   1.500000000000000, 2.0,               1.0, 0.0, 0.0, 0.0],
])

# =============================================================================
# Simulation parameters
# =============================================================================
cutoff    = 4.5
Lbig      = 100.0 * a
L         = np.array([Lbig, Lbig, 0.0])
n_heights = 500
z_max     = (n_heights + 1) * 0.01 + 2 * a
TET_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tet_Data")


# =============================================================================
# Build bodies — exactly matching the original code's geometry transform:
#   1. subtract z=2 (the clone file floor) before scaling
#   2. scale by factor
#   3. lift to target height
# Periodic version also shifts to box centre (L/2, L/2).
# =============================================================================
def make_bodies(height, dist):
    factor = (2 * a + dist * a) / 3.0
    bodies = []
    for row in CLONES:
        loc = row[:3].copy()
        qw, qx, qy, qz = row[3], row[4], row[5], row[6]
        orientation = Rotation.from_quat([qx, qy, qz, qw])
        loc[2] -= 2.0
        loc    *= factor
        loc[2] += height
        loc[0] += L[0] / 2.0
        loc[1] += L[1] / 2.0
        bodies.append(Body(location=loc, orientation=orientation))
    return bodies


def make_bodies_open(height, dist):
    factor = (2 * a + dist * a) / 3.0
    bodies = []
    for row in CLONES:
        loc = row[:3].copy()
        qw, qx, qy, qz = row[3], row[4], row[5], row[6]
        orientation = Rotation.from_quat([qx, qy, qz, qw])
        loc[2] -= 2.0
        loc    *= factor
        loc[2] += height
        bodies.append(Body(location=loc, orientation=orientation))
    return bodies


# =============================================================================
# Lubrication-corrected mobility
# =============================================================================
def Form_Lub_Mobility(sd):
    """
    Solves [I + M * Delta_R] * U_k = M * e_k for each unit direction e_k.
    Returns the (6N x 6N) mobility matrix.
    """
    Dim = sd.Delta_R.shape[1]
    I   = np.eye(Dim)
    Mob = np.zeros((Dim, Dim))
    for k in range(Dim):
        vel = sd.Lubrication_solve(X=None, Xm=I[:, k])
        Mob[:, k] = vel
    return Mob


# =============================================================================
# Helpers
# =============================================================================
def load_tet(fname):
    path = os.path.join(TET_DIR, fname)
    if not os.path.isfile(path):
        print(f"  Warning: {fname} not found in Tet_Data/, skipping.")
        return None
    return np.loadtxt(path)


def relative_error(data, ref):
    """Per-row relative error ||M - M_ref|| / ||M_ref||."""
    n   = min(len(data), len(ref))
    Dim = int(np.sqrt(ref.shape[1] - 1))
    err = np.zeros(n)
    for i in range(n):
        M     = data[i, 1:].reshape(Dim, Dim)
        M_ref = ref[i,  1:].reshape(Dim, Dim)
        denom = np.linalg.norm(M_ref)
        if denom > 0:
            err[i] = np.linalg.norm(M - M_ref) / denom
    return err


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':

    print(f"Computing lubrication mobility for {n_heights} heights...")
    print(f"a={a}, eta={eta}, cutoff={cutoff}, L={L}")

    bodies_init = make_bodies(height=1.01, dist=0.01)

    # periodic solver (DPStokes)
    sd_periodic = pyStokesianDynamics(
        bodies=bodies_init,
        a=a,
        eta=eta,
        periodic_length=L,
        z_max=z_max,
        cutoff=cutoff,
    )

    # non-periodic solver (NBody)
    sd_open = pyStokesianDynamics(
        bodies=bodies_init,
        a=a,
        eta=eta,
        periodic_length=np.array([0.0, 0.0, 0.0]),
        z_max=z_max,
        cutoff=cutoff,
    )

    sd_open.tolerance     = 1e-6
    sd_periodic.tolerance = 1e-6

    Mob_list_periodic = []
    Mob_list_open     = []

    for h in range(n_heights):
        height = 1.0 + (h + 1.0) * 0.01
        dist   = (h + 1.0) * 0.01
        print(f"Height {h+1}/{n_heights}: h={height:.3f}, dist={dist:.3f}")

        # periodic
        bodies_p = make_bodies(height, dist)
        sd_periodic.bodies = bodies_p
        sd_periodic.Set_R_Mats(r_vecs_np=[b.location for b in bodies_p])
        Mob_periodic = Form_Lub_Mobility(sd_periodic)
        Mob_list_periodic.append(np.append([height], Mob_periodic.flatten()))

        # non-periodic
        bodies_o = make_bodies_open(height, dist)
        sd_open.bodies = bodies_o
        sd_open.Set_R_Mats(r_vecs_np=[b.location for b in bodies_o])
        Mob_open = Form_Lub_Mobility(sd_open)
        Mob_list_open.append(np.append([height], Mob_open.flatten()))

        if (h + 1) % 50 == 0:
            print(f"  {h+1}/{n_heights} done (height={height:.3f})")

    Mob_of_h_periodic = np.array(Mob_list_periodic)
    Mob_of_h_open     = np.array(Mob_list_open)
    print("Computation complete.")

    # ── PASS/FAIL check against stored reference ──────────────────────────────
    Mob_periodic_ref = load_tet("tetra_lub_periodic_mob.dat")
    Mob_open_ref     = load_tet("tetra_lub_open_mob.dat")

    if Mob_periodic_ref is not None and Mob_open_ref is not None:
        n_pts = min(len(Mob_of_h_periodic), len(Mob_periodic_ref))
        mean_rel_err_per = np.mean(
            relative_error(Mob_of_h_periodic[:n_pts], Mob_periodic_ref[:n_pts]))

        n_pts_o = min(len(Mob_of_h_open), len(Mob_open_ref))
        mean_rel_err_open = np.mean(
            relative_error(Mob_of_h_open[:n_pts_o], Mob_open_ref[:n_pts_o]))

        print(f"Mean relative error (periodic) vs reference: {mean_rel_err_per:.2e}")
        print(f"Mean relative error (open)     vs reference: {mean_rel_err_open:.2e}")

        tolerance = 1e-7
        if mean_rel_err_per < tolerance and mean_rel_err_open < tolerance:
            print("[PASS] Results consistent with reference within tolerance.")
            sys.exit(0)
        else:
            print("[FAIL] Results differ from reference — plotting for diagnosis.")

    # ── Load blob-discretisation reference data and plot ─────────────────────
    ref_data = load_tet("tetra_mb2562_mob.dat")
    mb162    = load_tet("tetra_mb162_mob.dat")
    mb642    = load_tet("tetra_mb642_mob.dat")

    if ref_data is None:
        print("Reference file tetra_mb2562_mob.dat not found — cannot plot.")
        sys.exit(1)

    n_pts   = min(len(Mob_of_h_periodic), len(ref_data))
    epsilon = Mob_of_h_periodic[:n_pts, 0]
    err_per = relative_error(Mob_of_h_periodic[:n_pts], ref_data[:n_pts])

    n_pts_o  = min(len(Mob_of_h_open), len(ref_data))
    err_open = relative_error(Mob_of_h_open[:n_pts_o], ref_data[:n_pts_o])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(epsilon - 1, err_per, '-', linewidth=2.5,
              label='Lubrication corrected (periodic)')
    ax.loglog(Mob_of_h_open[:n_pts_o, 0] - 1, err_open, '-', linewidth=2.5,
              label='Lubrication corrected (non-periodic)')

    if mb162 is not None:
        n = min(len(mb162), len(ref_data))
        ax.loglog(ref_data[:n, 0] - 1,
                  relative_error(mb162[:n], ref_data[:n]),
                  '--', linewidth=2, label='162 blobs')

    if mb642 is not None:
        n = min(len(mb642), len(ref_data))
        ax.loglog(ref_data[:n, 0] - 1,
                  relative_error(mb642[:n], ref_data[:n]),
                  '-.', linewidth=2, label='642 blobs')

    ax.set_xlabel(r'$\epsilon = h/a - 1$', fontsize=14)
    ax.set_ylabel('Relative error in mobility', fontsize=14)
    ax.set_title('Tetrahedron mobility — relative error vs 2562-blob reference',
                 fontsize=13)
    ax.set_xlim([0.01, 2.5])
    ax.legend(fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(TET_DIR, 'tetra_mobility_error.png')
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()
