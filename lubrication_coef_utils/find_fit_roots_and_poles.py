"""
find_fit_roots_and_poles.py
---------------------------
For each RPY and Delta_R rational fit, finds all real roots of:
  - the numerator P(u)  → zeros of the fit
  - the denominator Q(u) → poles/singularities of the fit

Only reports roots where the corresponding h = h_scale*(1/u - 1) is:
  a) real  (|Im(h)| < threshold * |Re(h)|)
  b) within the valid fitting range

RPY  valid range: h > 0  (fit used for all positive gaps)
Delta_R valid range: h in [asym_cut + 1, rpy_cut + 1]  (i.e. eps in [asym_cut, rpy_cut])
"""

import numpy as np

IMAG_TOL = 1e-8   # |Im/Re| threshold to consider a root real

# =============================================================================
# RPY fits  (h_scale = 0.5)
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

# =============================================================================
# Delta_R fits  (h_scale = 0.5)
# =============================================================================
h_scale_fit = 0.5

delta_coeffs = {
    'Xa_corr': dict(
        asym_cut = 2.0549e-01, rpy_cut = 7.0, p_min = 1,
        pc = [-3.785206291907196e-04,  2.849533912393731e-03, -6.091201426082193e-02,
               9.742067047171957e+00, -3.375186552831702e+01],
        qc = [-1.105274248411881e+01,  4.617413294274526e+01, -8.739438745579977e+01,
               6.402551939108523e+01],
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
# Helpers
# =============================================================================
def u_to_h(u_root, h_scale):
    """Convert a root in u-space to h = h_scale*(1/u - 1)."""
    return h_scale * (1.0 / u_root - 1.0)


def is_real(z, tol=IMAG_TOL):
    """True if z is numerically real."""
    return abs(z.imag) < tol * max(abs(z.real), 1e-30)


def poly_roots_to_h(coeffs_descending, h_scale, h_lo, h_hi):
    """
    Find roots of a polynomial (coefficients in descending order),
    convert to h, and return those that are real and in (h_lo, h_hi).
    Returns list of (h_real, u_real) tuples.
    """
    roots = np.roots(coeffs_descending)
    hits  = []
    for r in roots:
        if not is_real(r):
            continue
        u = r.real
        if u <= 0 or u >= 1:
            continue          # u must be in (0,1) for h > 0
        h = u_to_h(u, h_scale)
        if h_lo < h < h_hi:
            hits.append((h, u))
    return sorted(hits)


def numerator_poly(coeffs_p, p_min):
    """
    P(u) = sum(p[i] * u^(p_min+i)).
    Returns polynomial coefficients in descending degree order.
    """
    max_deg = p_min + len(coeffs_p) - 1
    poly    = np.zeros(max_deg + 1)
    for i, c in enumerate(coeffs_p):
        poly[max_deg - (p_min + i)] = c
    return poly


def denominator_poly(coeffs_q):
    """
    Q(u) = 1 + sum(q[i] * u^(i+1)).
    Returns polynomial coefficients in descending degree order.
    """
    n    = len(coeffs_q)
    poly = np.zeros(n + 1)
    poly[0] = coeffs_q[-1]
    for i in range(n - 1):
        poly[n - 1 - i] = coeffs_q[i]
    poly[n] = 1.0          # constant term
    # rebuild cleanly: degree n polynomial, highest first
    poly = np.zeros(n + 1)
    for i, c in enumerate(coeffs_q):
        poly[n - (i + 1)] = c
    poly[n] = 1.0
    return poly


def report_roots(label, name, h_scale, p_coeffs, p_min, q_coeffs, h_lo, h_hi):
    """Print zeros and poles of the fit that are real and in (h_lo, h_hi)."""
    p_poly = numerator_poly(p_coeffs, p_min)
    q_poly = denominator_poly(q_coeffs)

    zeros = poly_roots_to_h(p_poly, h_scale, h_lo, h_hi)
    poles = poly_roots_to_h(q_poly, h_scale, h_lo, h_hi)

    has_anything = zeros or poles
    print(f"\n  {label} — {name}  [h in ({h_lo:.4e}, {h_hi:.4e})]"
          + ("" if has_anything else "  (none)"))
    for h, u in zeros:
        print(f"    ZERO  (P=0)  h = {h:.8e}  (u = {u:.8e})")
    for h, u in poles:
        print(f"    POLE  (Q=0)  h = {h:.8e}  (u = {u:.8e})")


# =============================================================================
# Run
# =============================================================================
print("=" * 65)
print("Real zeros and poles of RPY and Delta_R fits within valid range")
print("=" * 65)

print("\n--- RPY fits  (h_scale = 0.5,  valid range: h > 0) ---")
for name, d in RPY_COEFFS.items():
    report_roots("RPY", name,
                 H_SCALE_RPY,
                 d['p'], d['p_min'], d['q'],
                 h_lo=0.0, h_hi=1e6)   # valid for all h > 0

print("\n--- Delta_R fits  (h_scale = 0.5,  valid range: [asym_cut, rpy_cut]) ---")
for name, d in delta_coeffs.items():
    h_lo = d['asym_cut']
    h_hi = d['rpy_cut']
    report_roots("DR", name,
                 h_scale_fit,
                 d['pc'], d['p_min'], d['qc'],
                 h_lo=h_lo, h_hi=h_hi)

print(f"\n{'='*65}")
