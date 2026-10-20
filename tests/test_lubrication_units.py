"""
test_lubrication_units.py
-------------------------
Regression tests for the pair-separation units bug in lubrication.cpp.

ResistCSC / ResistCSC_both used to divide the pair separation r_j - r_k by the
radius a only in periodic directions. In open directions r_norm stayed in
physical units while PAIR_CUTOFF (4.5) and the pair fits are in units of a, so
for a != 1 pairs near contact got (almost) no lubrication, and in a doubly
periodic box (L = [Lx, Ly, 0]) pairs at different heights got a mixed-unit
separation. The existing tests use a = 1 and equal heights, which hides it.

Checks (all with a != 1 and particles at different heights):
  1. a contact pair in an open box gets the same (scaled) lubrication as a = 1
  2. dimensional scaling: R(a x; a, a L) = S R(x; 1, L) S, with S = sqrt(a) for
     translational and a^(3/2) for rotational dofs, for open, doubly periodic
     and triply periodic boxes, MB and Sup, ResistCSC and ResistCSC_both
  3. open box == large periodic boxes when no pair crosses a boundary
  4. minimum image: a pair split across a periodic boundary == the same pair
     inside the box
  5. ResistCSC_both returns the same matrices as two ResistCSC calls
  6. matrices are symmetric
  7. a touching pair moving as a rigid body feels a finite force as the gap
     closes (checks the signs of the near-contact couplings)

Run with pytest, or directly: python test_lubrication_units.py
Imports the installed StokesianDynamics package, or a lubrication module
built in place (put its build directory on PYTHONPATH).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    from StokesianDynamics import Lubrication
except ImportError:
    from lubrication import Lubrication

ETA = 8.9e-4
D_CUT = 1e-6            # small, so the Debye clamp does not affect the checks
RADII = (0.5, 2.25, 7.0)


def _nlist(n):
    """Upper-triangle neighbour lists with every pair (all pairs are checked)."""
    return [np.arange(j + 1, n, dtype=np.int32) for j in range(n)]


def _R(lub, r, a, L, sup):
    r = [np.ascontiguousarray(x, dtype=np.float64) for x in r]
    return lub.ResistCSC(r, _nlist(len(r)), a, ETA,
                         np.asarray(L, dtype=np.float64), sup).toarray()


def _R_both(lub, r, a, L):
    r = [np.ascontiguousarray(x, dtype=np.float64) for x in r]
    MB, Sup = lub.ResistCSC_both(r, _nlist(len(r)), a, ETA,
                                 np.asarray(L, dtype=np.float64))
    return MB.toarray(), Sup.toarray()


def _cluster(seed=0):
    """Positions in units of a: a small cluster near the wall with near-contact
    pairs, a pair stacked vertically, and pairs beyond the 4.5a cutoff."""
    rng = np.random.default_rng(seed)
    x = np.array([[0.0, 0.0, 1.2],
                  [2.03, 0.10, 1.4],      # gap ~0.04a, height difference
                  [0.30, -2.05, 1.1],     # gap ~0.07a
                  [2.10, 0.20, 3.5],      # above particle 1 (vertical pair)
                  [4.90, 1.00, 1.6],      # beyond the cutoff from most others
                  [-1.20, 1.90, 2.3]])
    x[:, :2] += rng.uniform(-0.01, 0.01, (len(x), 2))
    return x + np.array([20.0, 20.0, 0.0])     # inside a 40a periodic box


def _scaling(a, n):
    s = np.tile(np.r_[np.full(3, np.sqrt(a)), np.full(3, a**1.5)], n)
    return np.outer(s, s)


def test_contact_pair_lubrication_independent_of_radius():
    """The regression itself: a contact pair in an open box with a != 1."""
    eps = 0.01
    vals = {}
    for a in (1.0,) + RADII:
        lub = Lubrication(D_CUT)
        r = [np.array([0.0, 0.0, 50.0 * a]), np.array([(2.0 + eps) * a, 0.0, 50.0 * a])]
        dR = _R(lub, r, a, [0, 0, 0], True) - _R(lub, r, a, [0, 0, 0], False)
        vals[a] = dR[0, 6] / (6.0 * np.pi * ETA * a)     # squeeze coupling X12A
    # near contact X12A ~ -1/(4 eps) (-25.4 at eps = 0.01)
    assert vals[1.0] < -20.0, vals
    for a in RADII:
        assert abs(vals[a] - vals[1.0]) < 1e-9 * abs(vals[1.0]), \
            f"a={a}: squeeze coupling {vals[a]:.4f}, expected {vals[1.0]:.4f} (as for a=1)"


def test_dimensional_scaling():
    x = _cluster()
    n = len(x)
    lub = Lubrication(D_CUT)
    for L in ([0, 0, 0], [40.0, 40.0, 0.0], [40.0, 40.0, 40.0]):
        L = np.asarray(L)
        R1 = {sup: _R(lub, x, 1.0, L, sup) for sup in (False, True)}
        for a in RADII:
            S = _scaling(a, n)
            for sup in (False, True):
                Ra = _R(lub, a * x, a, a * L, sup)
                err = np.abs(Ra - S * R1[sup]).max() / np.abs(S * R1[sup]).max()
                assert err < 1e-10, f"L={L.tolist()}, a={a}, {'Sup' if sup else 'MB'}: rel. err {err:.2e}"
            MB, Sup = _R_both(lub, a * x, a, a * L)
            for nm, M, sup in (('MB', MB, False), ('Sup', Sup, True)):
                err = np.abs(M - S * R1[sup]).max() / np.abs(S * R1[sup]).max()
                assert err < 1e-10, f"ResistCSC_both, L={L.tolist()}, a={a}, {nm}: rel. err {err:.2e}"


def test_open_equals_periodic_without_wrapping():
    lub = Lubrication(D_CUT)
    for a in RADII:
        x = a * _cluster(1)
        ref = {sup: _R(lub, x, a, [0, 0, 0], sup) for sup in (False, True)}
        for L in ([40.0 * a, 40.0 * a, 0.0], [40.0 * a, 40.0 * a, 40.0 * a], [40.0 * a, 0.0, 0.0]):
            for sup in (False, True):
                R = _R(lub, x, a, L, sup)
                err = np.abs(R - ref[sup]).max() / np.abs(ref[sup]).max()
                assert err < 1e-12, f"a={a}, L={L}, {'Sup' if sup else 'MB'}: rel. err {err:.2e}"


def test_minimum_image_across_boundary():
    lub = Lubrication(D_CUT)
    for a in RADII:
        Lx = 30.0 * a
        L = [Lx, Lx, 0.0]
        inside = [np.array([10.0, 10.0, 1.3]) * a, np.array([12.04, 10.3, 1.6]) * a]
        split = [np.array([Lx - 1.0 * a, 10.0 * a, 1.3 * a]), np.array([1.04 * a, 10.3 * a, 1.6 * a])]
        for sup in (False, True):
            A = _R(lub, inside, a, L, sup)
            B = _R(lub, split, a, L, sup)
            err = np.abs(A - B).max() / np.abs(A).max()
            assert err < 1e-10, f"a={a}, {'Sup' if sup else 'MB'}: rel. err {err:.2e}"


def test_symmetric():
    lub = Lubrication(D_CUT)
    for a in RADII:
        x = a * _cluster(2)
        for L in ([0, 0, 0], [40.0 * a, 40.0 * a, 0.0]):
            for sup in (False, True):
                R = _R(lub, x, a, L, sup)
                assert np.abs(R - R.T).max() <= 1e-12 * np.abs(R).max()


def test_rigid_doublet_force_stays_finite():
    """Rigid-body motion of a touching pair does not shear the gap, so the
    near-contact singular terms (1/eps, log(1/eps)) must cancel."""
    rng = np.random.default_rng(3)
    for a in RADII:
        lub = Lubrication(1e-9)
        for _ in range(3):
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            W, U0 = rng.normal(size=3), rng.normal(size=3)
            rigid, squeeze, shear = [], [], []
            t = np.cross(W, axis); t /= np.linalg.norm(t)        # tangential direction
            for eps in (1e-2, 1e-3, 1e-4, 1e-5):
                c = np.array([0.0, 0.0, 1e5 * a])
                r = [c - 0.5 * (2.0 + eps) * a * axis, c + 0.5 * (2.0 + eps) * a * axis]
                R = _R(lub, r, a, [0, 0, 0], True)
                V = np.concatenate([U0 + np.cross(W, r[0] - c), W, U0 + np.cross(W, r[1] - c), W])
                rigid.append(np.abs(R @ V).max())
                # controls: sphere 1 alone moving along / across the line of centres
                squeeze.append(np.abs(R @ np.r_[axis, np.zeros(9)]).max())
                shear.append(np.abs(R @ np.r_[t, np.zeros(9)]).max())
            assert max(rigid) < 1.05 * min(rigid), f"a={a}: rigid-body force not bounded: {rigid}"
            assert squeeze[-1] > 100 * squeeze[0], f"a={a}: squeeze force should grow ~1/eps: {squeeze}"
            assert shear[-1] > 1.3 * shear[0], f"a={a}: shear force should grow ~log(1/eps): {shear}"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
