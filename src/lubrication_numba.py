"""
lubrication_numba.py
--------------------
Numba @njit reimplementation of the lubrication resistance matrix assembly
from lubrication.cpp. Provides the same ResistCSC interface as the C++ nanobind
module but runs as pure Python/Numba — no compilation step required.

Key design:
  - All inner loops (pair evaluation, rational fits, matrix assembly) are
    compiled with @njit for near-C++ speed.
  - The outer ResistCSC function is also @njit so the full loop is one
    compiled kernel with no Python overhead inside.
  - Sparse triplet assembly happens in a pre-allocated array, converted to
    scipy CSC at the end (unavoidable Python step).
  - Wall scalars use RPY rational fits + Delta_R chimera, identical to C++.
  - Pair scalars use the same rational fit coefficients as the C++ code.
"""
import numpy as np
import numba as nb
from numba import njit, float64, int32
from scipy.sparse import csc_matrix


# =============================================================================
# Constants — rational fit coefficients as module-level arrays
# (Numba can close over module-level numpy arrays in @njit functions)
# =============================================================================

# RPY wall fits: h_scale=2.0, u=1/(1+h/2)
# Stored as (n_scalars, max_coeffs) — padded with zeros
_RPY_HSCALE = 2.0
_RPY_PMIN   = np.array([1, 1, 4, 3, 3], dtype=np.int32)
_RPY_NP     = np.array([5, 5, 5, 5, 5], dtype=np.int32)
_RPY_NQ     = np.array([4, 4, 4, 4, 4], dtype=np.int32)
_RPY_P = np.array([
    [ 5.624811222421287e-01,-1.902140800834350e+00, 2.115451678636365e+00,
     -7.635996664143829e-01,-4.837500295844031e-03],
    [ 2.812105339266384e-01,-1.115099808672645e+00, 1.464702518976790e+00,
     -6.325034511693657e-01,-6.999899056304174e-03],
    [-7.392648118347235e-03, 3.368845461666825e-03, 1.612424189552847e-02,
     -1.298745899757570e-02, 9.455887027999391e-03],
    [ 2.098387553712979e-02,-3.553195250282171e-02, 7.166486859377513e-03,
     -9.653580363206946e-03, 9.591411297605018e-03],
    [ 5.238992353228791e-02,-1.079457649669765e-01, 3.418699552920573e-02,
     -2.084108061392004e-02, 5.170400576390023e-02],
], dtype=np.float64)
_RPY_Q = np.array([
    [-4.944641980681546e+00, 9.159688702109271e+00,-7.511281900597266e+00, 2.291593234815973e+00],
    [-5.248201918594638e+00, 1.035754559569218e+01,-9.127978546090679e+00, 3.037666763264918e+00],
    [-5.169293792382640e+00, 1.003175394314084e+01,-8.681787623181577e+00, 2.834997295430568e+00],
    [-4.632481897836154e+00, 7.956808419346241e+00,-6.033709799853070e+00, 1.716055087204692e+00],
    [-5.003405409201031e+00, 9.348095205572779e+00,-7.751092728329362e+00, 2.416658256123458e+00],
], dtype=np.float64)

# Delta_R fits: h_scale=0.5, u=1/(1+h/0.5)
_DR_HSCALE   = 0.5
_DR_ASYM_CUT = np.array([2.0549e-01, 2.9118e-02, 1.0e-01, 3.0e-03, 4.56e-02], dtype=np.float64)
_DR_RPY_CUT  = np.array([7.0,        5.6,        3.4,     0.4,      5.0     ], dtype=np.float64)
_DR_PMIN     = np.array([1, 1, 4, 3, 3], dtype=np.int32)
_DR_NP       = np.array([5, 4, 5, 5, 5], dtype=np.int32)
_DR_NQ       = np.array([4, 4, 4, 4, 4], dtype=np.int32)
_DR_P = np.zeros((5, 5), dtype=np.float64)
_DR_P[0] = [-4.373370078409577e-04, 4.424560646016040e-03,-7.638777986146561e-02,
             9.810003614859475e+00,-3.385902988985162e+01]
_DR_P[1,:4]=[-1.802661869524016e-03, 9.468660270547602e-03, 3.218091451875633e-01,
             -9.812949264604398e-01]
_DR_P[2] = [-1.171720418904620e+00, 2.634024743369987e+01,-2.037770517911233e+02,
             6.604148051813611e+02,-7.644275364225128e+02]
_DR_P[3] = [-1.363325750623006e-01, 1.661064454119597e+00,-7.478587775449657e+00,
             1.474504992172720e+01,-1.073379968056573e+01]
_DR_P[4] = [ 4.774546185054501e-02,-1.200098819024352e+00, 1.143510786982436e+01,
            -4.754910568257404e+01, 7.229905586544142e+01]
_DR_Q = np.array([
    [-1.105160500583412e+01, 4.616461895203592e+01,-8.736869111978241e+01, 6.400332372662409e+01],
    [-1.077793936835893e+01, 4.559152441171614e+01,-9.009502525501624e+01, 6.997760797588727e+01],
    [-1.458404137489715e+01, 7.817171888830218e+01,-1.820761238925333e+02, 1.554914960487163e+02],
    [-1.156154685363075e+01, 5.011347178843301e+01,-9.652115572123826e+01, 6.970392297095620e+01],
    [-1.420388909734784e+01, 7.366905048843127e+01,-1.644525346135324e+02, 1.329770561654655e+02],
], dtype=np.float64)

# Pair Sup fit coefficients — layout: [crossover, p0..p5, c0..c4]
# Q uses c_i^2 (squared) unlike wall fits
_CF_X11A = np.array([7.006673e-02,
    -1.981052779593092e+01, 2.340828937668049e+03, 4.965834145654573e+05,
     1.139263177405481e+06,-3.107490582481417e+04, 6.123746273694200e+03,
    -5.465471058359071e+02,-1.069447402186549e+03,-6.581289099212643e-09,
    -9.673175257729874e-10,-6.988711115145478e-03], dtype=np.float64)
_CF_X12A = np.array([2.457900e-02,
    -1.593939244959434e-02, 4.445665389795823e+00,-4.078726991721069e+02,
    -1.034351092030956e+04,-8.219652436471222e+03,-1.370173545248133e+05,
     2.488928114444593e-04,-1.258809718612856e+02, 1.407296176369766e-08,
    -4.892039651123681e+02, 3.407984060897670e+02], dtype=np.float64)
_CF_Y11A = np.array([5.228588e-03,
     2.171207126379109e+00, 3.853002949507692e+02, 6.624352544471424e+03,
     1.484697318890232e+04, 7.047767136709095e+03, 6.540531100098950e+03,
    -1.511807925068774e+01, 7.012000660040704e+01,-1.193647614569199e+02,
    -8.041378517885596e+01,-8.122175958097344e+01], dtype=np.float64)
_CF_Y12A = np.array([5.248093e-03,
    -1.461562640575963e+00,-2.507841615126929e+02,-4.331083375391881e+03,
    -9.737921752082497e+03,-4.058329204467756e+02, 3.606943711811880e+01,
    -1.596895122841455e+01, 8.109752269618301e+01,-1.638005603371911e+02,
     1.254673064262033e+02, 5.431104446475567e+00], dtype=np.float64)
_CF_Y11B = np.array([5.313637e-03,
    -1.043087981132569e+00, 5.195160018422946e+03, 1.121528966727754e+06,
     1.355294010638291e+07,-3.532929933499587e+06, 4.458815880568250e+05,
     6.331300459521475e+01,-1.375961122074268e+03,-7.415395959620902e+03,
    -1.414565234165133e+04,-7.015955939280531e+03], dtype=np.float64)
_CF_Y12B = np.array([5.138518e-03,
     1.266855888751514e+00, 6.899463247856546e+02, 4.216236929666354e+04,
     2.147931207416082e+05,-2.876020054138726e+04, 1.086426843098747e+03,
     2.707637792335356e+01,-2.674955385301423e+02, 9.078532792700140e+02,
     9.825987715332953e+02,-5.482196655744529e-01], dtype=np.float64)
_CF_X11C = np.array([5.113070e-03,
     1.397874023016620e+00, 1.345883636860944e+01, 2.434301005382227e+04,
     1.444198486149228e+05, 1.003697333447050e+05, 2.410941641426479e+05,
     3.067204057507511e+00, 1.320239652333805e+02, 3.292073502673707e+02,
    -2.743878779348838e+02,-4.252396254964442e+02], dtype=np.float64)
_CF_X12C = np.array([5.050000e-03,
    -2.022446304463027e-01,-5.956596895307574e+01,-4.801712239322800e+02,
     2.384455230908337e+02,-5.757163499587050e+01, 5.798541137001251e+00,
     1.748413100133951e+01, 5.819390284035368e+01,-5.863429968285631e+01,
    -1.179113239482045e-01,-4.268683173977720e-09], dtype=np.float64)
_CF_Y11C = np.array([5.307046e-03,
     2.802854397382923e+00, 4.462442222692911e+02, 7.021330954933401e+03,
     1.885437964587811e+04, 1.508056246869293e+04,-2.094355307813514e+01,
     1.481417448480583e+01, 6.732984477458869e+01,-1.206040482897068e+02,
    -1.057879287786618e+02,-2.765378408631323e-03], dtype=np.float64)
_CF_Y12C = np.array([1.509115e-02,
    -9.141796920496276e-01, 1.342735109731877e+03, 3.936577516657912e+05,
     3.699261691007629e+06,-1.507356098885682e+06, 2.205561499489647e+05,
     3.866992630227200e+01,-1.228093930062573e+03, 5.906456100095043e+03,
     7.791981385860539e+03, 1.873855300354582e+01], dtype=np.float64)

# Pair MB fit coefficients
_MB_X11A = np.array([5.000000e-03,
    -2.499999982474416e-01, 1.165664399869339e+00, 2.154793952021480e+00,
     1.270149386850713e+00, 2.483649225013684e-01,-4.186019883001905e-05,
    -1.378928715644350e+00, 1.156488416728070e+00, 4.976911470183872e-01,
     4.870465773876030e-06,-4.423392960087700e-07], dtype=np.float64)
_MB_X12A = np.array([5.000000e-03,
     2.500000295508952e-01,-8.497183983207306e-01,-1.951820680989982e-02,
     2.619149841535253e-01,-4.534273305750967e-01,-3.145525674635611e-01,
     8.388694341455940e-01, 1.603695663575000e-06,-6.246243735852574e-01,
     9.857882804654846e-01, 5.012686787659838e-01], dtype=np.float64)
_MB_Y11A = np.array([5.000000e-03,
     1.334448916124861e+00, 1.533276418322284e+00, 2.504280783426653e-02,
     4.777495936484357e-02, 3.630334324421668e-01, 1.237866143758240e-01,
     1.289180844579062e+00, 1.602843358683278e-05, 1.263923257788066e-02,
    -5.997350601777206e-01, 3.521911887962246e-01], dtype=np.float64)
_MB_Y12A = np.array([5.000000e-03,
    -6.167708949219610e-01,-2.537080095626051e-01, 1.307822038490171e-01,
    -1.204261677530989e-01,-1.206649718042400e-01,-8.620245076540467e-05,
    -1.298347445853823e+00,-7.248656296164483e-06,-2.865726715854026e-04,
    -6.865090702813017e-01, 4.033767918608367e-01], dtype=np.float64)
_MB_Y11B = np.array([5.000000e-03,
     1.757300002175146e-01,-1.263482053075663e-01, 5.466331979155107e-02,
     4.145104715957172e-03,-4.855688014271770e-04, 2.869576324453811e-05,
    -1.286512595383362e+00,-1.165742481270486e-04,-4.740203421187537e-03,
    -5.567875571587572e-01,-3.249642869760815e-01], dtype=np.float64)
_MB_Y12B = np.array([5.000000e-03,
     3.445954579556256e-01, 1.715184390725956e-02,-8.224756090070001e-02,
     7.735243406016939e-02, 1.035089219385419e-03,-6.484151006279233e-05,
    -1.279854181851254e+00,-1.070970483681416e-04,-4.365905834933606e-02,
    -4.632099590221615e-01, 2.926235258785276e-01], dtype=np.float64)
_MB_X11C = np.array([5.000000e-03,
     1.354497402277264e+00, 1.685515273189768e+00, 3.267336413581359e-02,
    -1.215634951152663e-02, 9.006177715282025e-01, 9.200311837421076e-01,
     1.136664193298283e+00,-1.011738645854246e-04, 6.378569015548268e-04,
     8.207819113317577e-01, 8.307667300391079e-01], dtype=np.float64)
_MB_X12C = np.array([5.000000e-03,
    -1.693122019768674e-01, 9.231352641200651e-02,-1.787657927723367e-02,
    -1.648168837655593e-02, 2.008204276597000e-03,-1.214848156108869e-04,
    -1.001192015055246e+00,-1.528540714747427e-07, 6.544951602958429e-02,
     3.532367548183314e-01, 2.868536488193641e-01], dtype=np.float64)
_MB_Y11C = np.array([5.000000e-03,
     1.427805390573128e+00, 2.108235342790945e+00, 5.698686101168220e-02,
     1.538429638251989e-02, 4.805901339731146e-01, 2.249137271154556e-01,
     1.286277704062037e+00,-1.490352315543504e-05, 1.373451534868378e-01,
     5.996424201465841e-01, 4.107768773410937e-01], dtype=np.float64)
_MB_Y12C = np.array([5.000000e-03,
     1.331703764524452e-01,-8.242976148134072e-02, 2.876821851261879e-02,
     7.924040945752319e-03,-9.931798750875400e-04, 6.253327411129789e-05,
     1.272211977886686e+00,-5.812561617915775e-04,-1.858813454068888e-01,
    -4.861222841829775e-01,-3.301018709137714e-01], dtype=np.float64)


# =============================================================================
# @njit helper functions
# =============================================================================

@njit(cache=True)
def _eval_wall_rat(h, h_scale, p_min, p, n_p, q, n_q):
    """Evaluate P(u)/Q(u) with u=1/(1+h/h_scale)."""
    u = 1.0 / (1.0 + h / h_scale)
    P = 0.0
    upow = u ** p_min
    for i in range(n_p):
        P += p[i] * upow
        upow *= u
    Q = 1.0
    upow = u
    for i in range(n_q):
        Q += q[i] * upow
        upow *= u
    return P / Q


@njit(cache=True)
def _rpy_wall(s, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale):
    """RPY rational fit for wall scalar s at height h."""
    return _eval_wall_rat(h, rpy_hscale,
                          rpy_pmin[s], rpy_p[s], rpy_np[s],
                          rpy_q[s],   rpy_nq[s])


@njit(cache=True)
def _asym_wall(s, eps):
    """Near-contact AT asymptotic for wall scalar s."""
    le = np.log(eps)
    if s == 0:   # Xa_corr
        return 1.0/eps - (1.0/5.0)*le + 0.971280 - 1.0
    elif s == 1: # Ya_corr
        return -(8.0/15.0)*le + 0.9588 - 1.0
    elif s == 2: # Yb
        return (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.029 - (0.4576-0.2)*eps)
    elif s == 3: # XcPlus
        return (4.0/3.0)*(1.20206 - 3.0*(np.pi**2/6.0-1.0)*eps) - 4.0/3.0
    else:        # YcPlus
        return (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*eps) - 4.0/3.0


@njit(cache=True)
def _delta_R_wall(s, eps, h,
                  rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                  dr_p, dr_q, dr_pmin, dr_np, dr_nq, dr_hscale,
                  dr_asym_cut, dr_rpy_cut):
    """Chimera Delta_R for wall scalar s."""
    if eps > dr_rpy_cut[s]:
        return 0.0
    if eps < dr_asym_cut[s]:
        return (_asym_wall(s, eps)
                - _rpy_wall(s, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale))
    return _eval_wall_rat(h, dr_hscale,
                          dr_pmin[s], dr_p[s], dr_np[s],
                          dr_q[s],    dr_nq[s])


@njit(cache=True)
def _eval_PQ_pair(cf, ep, eq):
    """Evaluate pair rational P/Q (denominator uses c_i^2)."""
    P = (cf[1]*ep[0] + cf[2]*ep[1] + cf[3]*ep[2]
       + cf[4]*ep[3] + cf[5]*ep[4] + cf[6]*ep[5])
    Q = (1.0 + cf[7]*cf[7]*eq[0] + cf[8]*cf[8]*eq[1]
           + cf[9]*cf[9]*eq[2] + cf[10]*cf[10]*eq[3] + cf[11]*cf[11]*eq[4])
    return P / Q


@njit(cache=True)
def _wall_rpy_matrix(h, debye_cut,
                     rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                     f0, f1, f2):
    """6x6 RPY wall correction matrix."""
    eps = h - 1.0
    if eps < debye_cut:
        eps = debye_cut
        h   = 1.0 + eps
    R = np.zeros((6, 6))
    Xa = _rpy_wall(0, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale)
    Ya = _rpy_wall(1, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale)
    Yb = _rpy_wall(2, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale)
    Xc = _rpy_wall(3, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale)
    Yc = _rpy_wall(4, h, rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale)
    R[0,0]=f0*Ya; R[1,1]=f0*Ya; R[2,2]=f0*Xa
    R[3,3]=f2*Yc; R[4,4]=f2*Yc; R[5,5]=f2*Xc
    R[0,4]= f1*Yb; R[4,0]= f1*Yb
    R[1,3]=-f1*Yb; R[3,1]=-f1*Yb
    return R


@njit(cache=True)
def _wall_delta_matrix(h, debye_cut,
                       rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                       dr_p, dr_q, dr_pmin, dr_np, dr_nq, dr_hscale,
                       dr_asym_cut, dr_rpy_cut,
                       f0, f1, f2):
    """6x6 Delta_R wall correction matrix."""
    eps = h - 1.0
    if eps < debye_cut:
        eps = debye_cut
        h   = 1.0 + eps
    R = np.zeros((6, 6))
    dR = np.empty(5)
    for s in range(5):
        dR[s] = _delta_R_wall(s, eps, h,
                              rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                              dr_p, dr_q, dr_pmin, dr_np, dr_nq, dr_hscale,
                              dr_asym_cut, dr_rpy_cut)
    R[0,0]=f0*dR[1]; R[1,1]=f0*dR[1]; R[2,2]=f0*dR[0]
    R[3,3]=f2*dR[4]; R[4,4]=f2*dR[4]; R[5,5]=f2*dR[3]
    R[0,4]= f1*dR[2]; R[4,0]= f1*dR[2]
    R[1,3]=-f1*dR[2]; R[3,1]=-f1*dR[2]
    return R


@njit(cache=True)
def _assemble_pair(R, f0, f1, f2, rhat,
                   X11A, Y11A, Y11B, X11C, Y11C,
                   X12A, Y12A, Y12B, X12C, Y12C):
    """Assemble 12x12 pair resistance matrix from 10 scalars."""
    # squeeze = rhat rhat^T,  shear = I - squeeze,  vort = -[rhat]_x
    sq = np.outer(rhat, rhat)
    sh = np.eye(3) - sq
    vt = np.array([[ 0.0,      rhat[2],-rhat[1]],
                   [-rhat[2],  0.0,     rhat[0]],
                   [ rhat[1],-rhat[0],  0.0    ]])
    vt *= -1.0

    A11 = f0*(X11A*sq + Y11A*sh)
    B11 = -f1*(Y11B*vt)
    A12 = f0*(X12A*sq + Y12A*sh)
    B12 = f1*(Y12B*vt)
    C11 = f2*(X11C*sq + Y11C*sh)
    C12 = f2*(X12C*sq + Y12C*sh)

    R[0:3,  0:3 ] = A11
    R[0:3,  3:6 ] = B11
    R[0:3,  6:9 ] = A12
    R[0:3,  9:12] = B12
    R[3:6,  0:3 ] = -B11
    R[3:6,  3:6 ] = C11
    R[3:6,  6:9 ] = B12
    R[3:6,  9:12] = C12
    R[6:9,  0:3 ] = A12
    R[6:9,  3:6 ] = -B12
    R[6:9,  6:9 ] = A11
    R[6:9,  9:12] = -B11
    R[9:12, 0:3 ] = -B12
    R[9:12, 3:6 ] = C12
    R[9:12, 6:9 ] = B11
    R[9:12, 9:12] = C11


@njit(cache=True)
def _eval_pair_sup(r_norm, debye_cut,
                   cf_X11A, cf_X12A, cf_Y11A, cf_Y12A,
                   cf_Y11B, cf_Y12B, cf_X11C, cf_X12C,
                   cf_Y11C, cf_Y12C):
    """Evaluate all 10 Sup pair scalars at separation r_norm."""
    eps = r_norm - 2.0
    if eps < debye_cut:
        eps = debye_cut
    li   = np.log(1.0 / eps)
    eps2 = eps*eps; eps3=eps2*eps; eps4=eps3*eps; eps5=eps4*eps
    ep = np.array([1.0, eps, eps2, eps3, eps4, eps5])
    eq = np.array([eps, eps2, eps3, eps4, eps5])

    def at_ok(cf, at_val):
        if eps < cf[0]: return at_val
        return _eval_PQ_pair(cf, ep, eq)
    def at_sing(cf, S, at_val):
        if eps < cf[0]: return at_val
        return S + _eval_PQ_pair(cf, ep, eq) / eps

    X11A =  at_sing(cf_X11A,  0.25/eps, 0.995419+0.25/eps+0.225*li+0.0267857*eps*li)
    X12A =  at_sing(cf_X12A, -0.25/eps,-0.350153-0.25/eps-0.225*li-0.0267857*eps*li)
    Y11A =  at_ok  (cf_Y11A,            0.998317+0.166667*li)
    Y12A =  at_ok  (cf_Y12A,           -0.273652-0.166667*li)
    Y11B =  at_ok  (cf_Y11B,           -0.666667*(0.23892-0.25*li-0.125*eps*li))
    Y12B = -at_ok  (cf_Y12B,            0.666667*(-0.00162268+0.25*li+0.125*eps*li))
    X11C =  at_ok  (cf_X11C,            1.33333*(1.0518-0.125*eps*li))
    X12C =  at_ok  (cf_X12C,            1.33333*(-0.150257+0.125*eps*li))
    Y11C =  at_ok  (cf_Y11C,            1.33333*(0.702834+0.2*li+0.188*eps*li))
    Y12C =  at_ok  (cf_Y12C,            1.33333*(-0.027464+0.05*li+0.062*eps*li))
    return X11A, Y11A, Y11B, X11C, Y11C, X12A, Y12A, Y12B, X12C, Y12C


@njit(cache=True)
def _eval_pair_mb(r_norm, debye_cut,
                  mb_X11A, mb_X12A, mb_Y11A, mb_Y12A,
                  mb_Y11B, mb_Y12B, mb_X11C, mb_X12C,
                  mb_Y11C, mb_Y12C):
    """Evaluate all 10 MB pair scalars at separation r_norm."""
    eps = r_norm - 2.0
    if eps < debye_cut:
        eps = debye_cut
    eps2=eps*eps; eps3=eps2*eps; eps4=eps3*eps; eps5=eps4*eps
    ep = np.array([1.0, eps, eps2, eps3, eps4, eps5])
    eq = np.array([eps, eps2, eps3, eps4, eps5])

    X11A =  0.25/eps + _eval_PQ_pair(mb_X11A, ep, eq) / eps
    X12A = -0.25/eps + _eval_PQ_pair(mb_X12A, ep, eq) / eps
    Y11A =  _eval_PQ_pair(mb_Y11A, ep, eq)
    Y12A =  _eval_PQ_pair(mb_Y12A, ep, eq)
    Y11B =  _eval_PQ_pair(mb_Y11B, ep, eq)
    Y12B = -_eval_PQ_pair(mb_Y12B, ep, eq)
    X11C =  _eval_PQ_pair(mb_X11C, ep, eq)
    X12C =  _eval_PQ_pair(mb_X12C, ep, eq)
    Y11C =  _eval_PQ_pair(mb_Y11C, ep, eq)
    Y12C =  _eval_PQ_pair(mb_Y12C, ep, eq)
    return X11A, Y11A, Y11B, X11C, Y11C, X12A, Y12A, Y12B, X12C, Y12C


@njit(cache=True)
def _resist_csc_kernel(r_vecs, neighbor_lists, nb_offsets,
                       debye_cut, cutoff, wall_cutoff,
                       periodic_length, sup_if_true,
                       f0, f1, f2,
                       rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                       dr_p, dr_q, dr_pmin, dr_np, dr_nq, dr_hscale,
                       dr_asym_cut, dr_rpy_cut,
                       cf_X11A, cf_X12A, cf_Y11A, cf_Y12A,
                       cf_Y11B, cf_Y12B, cf_X11C, cf_X12C,
                       cf_Y11C, cf_Y12C,
                       mb_X11A, mb_X12A, mb_Y11A, mb_Y12A,
                       mb_Y11B, mb_Y12B, mb_X11C, mb_X12C,
                       mb_Y11C, mb_Y12C):
    """
    Core loop: returns (rows, cols, vals) triplet arrays for the sparse matrix.
    r_vecs:        (N, 3) float64 blob positions (already divided by a)
    neighbor_lists: flat int32 array of all neighbors concatenated
    nb_offsets:    (N,) int32 number of neighbors per body
    """
    N = r_vecs.shape[0]
    m_eps = 1e-12

    # conservative upper bound on triplets: wall(36*N) + pairs(36*4*max_pairs)
    max_pairs = neighbor_lists.shape[0]
    max_trips = 36*N + 36*4*max_pairs
    rows = np.empty(max_trips, dtype=np.int32)
    cols = np.empty(max_trips, dtype=np.int32)
    vals = np.empty(max_trips, dtype=np.float64)
    n_trips = 0

    R_pair = np.zeros((12, 12))
    nbr_offset = 0

    for j in range(N):
        height = r_vecs[j, 2]  # already /a

        if height < wall_cutoff:
            R_rpy = _wall_rpy_matrix(height, debye_cut,
                                     rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                                     f0, f1, f2)
            if sup_if_true:
                R_dR  = _wall_delta_matrix(height, debye_cut,
                                           rpy_p, rpy_q, rpy_pmin, rpy_np, rpy_nq, rpy_hscale,
                                           dr_p, dr_q, dr_pmin, dr_np, dr_nq, dr_hscale,
                                           dr_asym_cut, dr_rpy_cut,
                                           f0, f1, f2)
                R_wall = R_rpy + R_dR
            else:
                R_wall = R_rpy
            for row in range(6):
                for col in range(6):
                    v = R_wall[row, col]
                    if abs(v) > m_eps:
                        rows[n_trips] = j*6 + row
                        cols[n_trips] = j*6 + col
                        vals[n_trips] = v
                        n_trips += 1

        num_nb = nb_offsets[j]
        for ki in range(num_nb):
            k = neighbor_lists[nbr_offset + ki]
            r_jk = np.empty(3)
            for l in range(3):
                d = r_vecs[j, l] - r_vecs[k, l]
                if periodic_length[l] > 0.0:
                    Ll = periodic_length[l]
                    d -= int(d/Ll + 0.5*(1 if d>0 else -1))*Ll
                r_jk[l] = d
            r_norm = np.sqrt(r_jk[0]**2 + r_jk[1]**2 + r_jk[2]**2)

            if r_norm < cutoff:
                rhat = -r_jk / r_norm
                if sup_if_true:
                    s = _eval_pair_sup(r_norm, debye_cut,
                                       cf_X11A, cf_X12A, cf_Y11A, cf_Y12A,
                                       cf_Y11B, cf_Y12B, cf_X11C, cf_X12C,
                                       cf_Y11C, cf_Y12C)
                else:
                    s = _eval_pair_mb(r_norm, debye_cut,
                                      mb_X11A, mb_X12A, mb_Y11A, mb_Y12A,
                                      mb_Y11B, mb_Y12B, mb_X11C, mb_X12C,
                                      mb_Y11C, mb_Y12C)
                X11A,Y11A,Y11B,X11C,Y11C,X12A,Y12A,Y12B,X12C,Y12C = s
                R_pair[:] = 0.0
                _assemble_pair(R_pair, f0, f1, f2, rhat,
                               X11A,Y11A,Y11B,X11C,Y11C,
                               X12A,Y12A,Y12B,X12C,Y12C)

                # four 6x6 blocks: (j,j), (k,k), (j,k), (k,j)
                dof_r = (j*6, k*6, j*6, k*6)
                dof_c = (j*6, k*6, k*6, j*6)
                blk_r = (0,   6,   0,   6  )
                blk_c = (0,   6,   6,   0  )
                for b in range(4):
                    for row in range(6):
                        for col in range(6):
                            v = R_pair[blk_r[b]+row, blk_c[b]+col]
                            if abs(v) > m_eps:
                                rows[n_trips] = dof_r[b] + row
                                cols[n_trips] = dof_c[b] + col
                                vals[n_trips] = v
                                n_trips += 1

        nbr_offset += num_nb

    return rows[:n_trips], cols[:n_trips], vals[:n_trips]


# =============================================================================
# Public API — mirrors lubrication.Lubrication
# =============================================================================

class Lubrication:
    """
    Numba-accelerated lubrication resistance matrix.
    Same interface as the C++ nanobind Lubrication class.
    """

    def __init__(self, d_cut: float):
        self.debye_cut = d_cut
        # trigger JIT compilation on a tiny dummy call
        self._warmup()

    def _warmup(self):
        """Pre-compile the JIT kernel with a trivial call."""
        r = np.array([[0.0, 0.0, 10.0]], dtype=np.float64)
        nl = np.empty(0, dtype=np.int32)
        np_body = np.zeros(1, dtype=np.int32)
        pl = np.zeros(3, dtype=np.float64)
        _resist_csc_kernel(r, nl, np_body,
                           self.debye_cut, 4.5, 100.0, pl, True,
                           1.0, 1.0, 1.0,
                           _RPY_P, _RPY_Q, _RPY_PMIN, _RPY_NP, _RPY_NQ, _RPY_HSCALE,
                           _DR_P, _DR_Q, _DR_PMIN, _DR_NP, _DR_NQ, _DR_HSCALE,
                           _DR_ASYM_CUT, _DR_RPY_CUT,
                           _CF_X11A, _CF_X12A, _CF_Y11A, _CF_Y12A,
                           _CF_Y11B, _CF_Y12B, _CF_X11C, _CF_X12C,
                           _CF_Y11C, _CF_Y12C,
                           _MB_X11A, _MB_X12A, _MB_Y11A, _MB_Y12A,
                           _MB_Y11B, _MB_Y12B, _MB_X11C, _MB_X12C,
                           _MB_Y11C, _MB_Y12C)

    def ResistCSC_both(self, r_vectors, n_list, a, eta, cutoff,
                       wall_cutoff, periodic_length):
        """
        Build both MB and Sup sparse resistance matrices in one pass.
        Returns (R_MB, R_Sup) as scipy CSC matrices.
        """
        R_MB  = self.ResistCSC(r_vectors, n_list, a, eta, cutoff,
                               wall_cutoff, periodic_length, False)
        R_Sup = self.ResistCSC(r_vectors, n_list, a, eta, cutoff,
                               wall_cutoff, periodic_length, True)
        return R_MB, R_Sup

    def ResistCSC(self, r_vectors, n_list, a, eta, cutoff,
                  wall_cutoff, periodic_length, Sup_if_true):
        """
        Build sparse resistance matrix.

        Parameters
        ----------
        r_vectors : list of (3,) arrays  — blob positions
        n_list    : list of int arrays   — neighbor indices per blob
        a, eta    : float                — blob radius, viscosity
        cutoff    : float                — pair cutoff (in units of a)
        wall_cutoff : float              — wall cutoff (h/a)
        periodic_length : (3,) array     — box lengths (0 = open)
        Sup_if_true : bool               — Sup if True, MB if False
        Returns scipy CSC sparse matrix.
        """
        N = len(r_vectors)
        n_dof = 6 * N
        f0 = 6.0*np.pi*eta*a
        f1 = 6.0*np.pi*eta*a*a
        f2 = 6.0*np.pi*eta*a*a*a

        # pack positions (divide by a) into (N,3) array
        r_vecs = np.zeros((N, 3), dtype=np.float64)
        for j, rv in enumerate(r_vectors):
            r_vecs[j] = np.asarray(rv, dtype=np.float64) / a

        # flatten neighbor lists
        nl_flat   = np.concatenate([np.asarray(nb, dtype=np.int32)
                                    for nb in n_list]) if any(len(nb)>0 for nb in n_list) \
                    else np.empty(0, dtype=np.int32)
        nb_offsets = np.array([len(nb) for nb in n_list], dtype=np.int32)
        pl = np.asarray(periodic_length, dtype=np.float64)
        if pl.shape[0] == 0:
            pl = np.zeros(3, dtype=np.float64)

        rows, cols, vals = _resist_csc_kernel(
            r_vecs, nl_flat, nb_offsets,
            self.debye_cut, cutoff, wall_cutoff, pl, Sup_if_true,
            f0, f1, f2,
            _RPY_P, _RPY_Q, _RPY_PMIN, _RPY_NP, _RPY_NQ, _RPY_HSCALE,
            _DR_P, _DR_Q, _DR_PMIN, _DR_NP, _DR_NQ, _DR_HSCALE,
            _DR_ASYM_CUT, _DR_RPY_CUT,
            _CF_X11A, _CF_X12A, _CF_Y11A, _CF_Y12A,
            _CF_Y11B, _CF_Y12B, _CF_X11C, _CF_X12C,
            _CF_Y11C, _CF_Y12C,
            _MB_X11A, _MB_X12A, _MB_Y11A, _MB_Y12A,
            _MB_Y11B, _MB_Y12B, _MB_X11C, _MB_X12C,
            _MB_Y11C, _MB_Y12C)

        return csc_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
