"""
lubrication_cupy.py
--------------------
CuPy-vectorised lubrication resistance matrix assembly.
All coefficient arrays declared directly as cp.array — no CPU intermediates.
Drop-in replacement for the C++ nanobind Lubrication class.
"""
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp

# =============================================================================
# All coefficients declared directly on GPU
# =============================================================================

# Scalar constants
_RPY_HSCALE  = 0.5
_DR_HSCALE   = 0.5
# pi^2/6 - 1  and  6*pi (precomputed, no np.pi at runtime)
_PI2_6_M1    = 6.449340668482264e-01
_6PI         = 18.84955592153876

# RPY wall fits: u=1/(1+h/0.5), P(u)=sum p_i*u^(pmin+i), Q(u)=1+sum q_i*u^(i+1)
_RPY_PMIN = cp.array([1, 1, 4, 3, 3], dtype=cp.int32)
_RPY_NP   = cp.array([4, 4, 4, 4, 4], dtype=cp.int32)
_RPY_NQ   = cp.array([4, 4, 4, 4, 4], dtype=cp.int32)
_RPY_P = cp.array([
    [ 2.249994311366312e+00,-9.302942542409326e+00, 1.018898977062156e+01,
     -1.461983146752135e-01],
    [ 1.124980798925478e+00,-5.161218663062001e+00, 7.210326948819593e+00,
     -2.621851150554913e+00],
    [-2.019389402582813e+00, 7.475392505257506e+00,-3.877269126222068e+00,
     -1.028617914833345e+00],
    [ 1.333114620704834e+00,-4.215726080748013e+00,-5.775019233806103e-02,
      1.969213190855149e-01],
    [ 3.339371808444304e+00,-1.406703500455225e+01, 8.727632354389423e+00,
      9.801957547519073e+00],
], dtype=cp.float64)
_RPY_Q = cp.array([
    [-7.384833913349599e+00, 1.974855818648951e+01,-2.040618954066672e+01, 4.353438654549962e+00],
    [-6.713754414307387e+00, 1.706822775135723e+01,-1.933894500702774e+01, 7.413436778017460e+00],
    [-8.683501976306239e+00, 2.835163248843032e+01,-4.234435891678831e+01, 2.531363165467517e+01],
    [-6.164932696556416e+00, 1.247407137449759e+01,-1.135905628832895e+01, 6.141510604219781e+00],
    [-7.179264725936619e+00, 1.782208382807917e+01,-1.884878181254139e+01, 8.912717591979181e+00],
], dtype=cp.float64)

# Delta_R fits: u=1/(1+h/0.5)
# XcPlus: asym_cut=9.7e-3, n_p=4, n_q=3 (row padded to match array shape)
_DR_ASYM_CUT = cp.array([2.0549e-01, 2.9118e-02, 1.0e-01, 9.7e-03, 4.56e-02], dtype=cp.float64)
_DR_RPY_CUT  = cp.array([7.0,        5.6,        3.4,     0.4,     5.0     ], dtype=cp.float64)
_DR_PMIN     = cp.array([1, 1, 4, 3, 3], dtype=cp.int32)
_DR_NP       = cp.array([3, 4, 5, 4, 5], dtype=cp.int32)
_DR_NQ       = cp.array([4, 4, 4, 3, 4], dtype=cp.int32)
_DR_P = cp.array([
    [ 3.390440744007662e-03,-1.292707770613473e-01, 1.621030547158851e+00,
      0.0, 0.0],
    [-1.802661869524016e-03, 9.468660270547602e-03, 3.218091451875633e-01,
     -9.812949264604398e-01, 0.0],
    [-1.171720418904620e+00, 2.634024743369987e+01,-2.037770517911233e+02,
      6.604148051813611e+02,-7.644275364225128e+02],
    [-6.769668477673338e-02, 4.251981921512744e-01,-4.761218699317399e-01,
     -5.662068770688917e-01, 0.0],
    [ 4.774546185054501e-02,-1.200098819024352e+00, 1.143510786982436e+01,
     -4.754910568257404e+01, 7.229905586544142e+01],
], dtype=cp.float64)
_DR_Q = cp.array([
    [-9.934759618046485e+00, 3.852782844267631e+01,-6.919767486523838e+01, 4.812890822657162e+01],
    [-1.077793936835893e+01, 4.559152441171614e+01,-9.009502525501624e+01, 6.997760797588727e+01],
    [-1.458404137489715e+01, 7.817171888830218e+01,-1.820761238925333e+02, 1.554914960487163e+02],
    [-7.986015731409064e+00, 2.107903159501786e+01,-1.836099133069141e+01, 0.0],
    [-1.420388909734784e+01, 7.366905048843127e+01,-1.644525346135324e+02, 1.329770561654655e+02],
], dtype=cp.float64)

# Pair Sup fit coefficients [crossover, p0..p5, c0..c4], Q uses c_i^2
_CF_SUP = cp.array([
    [7.006673e-02,-1.981052779593092e+01, 2.340828937668049e+03, 4.965834145654573e+05,
      1.139263177405481e+06,-3.107490582481417e+04, 6.123746273694200e+03,
     -5.465471058359071e+02,-1.069447402186549e+03,-6.581289099212643e-09,
     -9.673175257729874e-10,-6.988711115145478e-03],
    [2.457900e-02,-1.593939244959434e-02, 4.445665389795823e+00,-4.078726991721069e+02,
     -1.034351092030956e+04,-8.219652436471222e+03,-1.370173545248133e+05,
      2.488928114444593e-04,-1.258809718612856e+02, 1.407296176369766e-08,
     -4.892039651123681e+02, 3.407984060897670e+02],
    [5.228588e-03, 2.171207126379109e+00, 3.853002949507692e+02, 6.624352544471424e+03,
      1.484697318890232e+04, 7.047767136709095e+03, 6.540531100098950e+03,
     -1.511807925068774e+01, 7.012000660040704e+01,-1.193647614569199e+02,
     -8.041378517885596e+01,-8.122175958097344e+01],
    [5.248093e-03,-1.461562640575963e+00,-2.507841615126929e+02,-4.331083375391881e+03,
     -9.737921752082497e+03,-4.058329204467756e+02, 3.606943711811880e+01,
     -1.596895122841455e+01, 8.109752269618301e+01,-1.638005603371911e+02,
      1.254673064262033e+02, 5.431104446475567e+00],
    [5.313637e-03,-1.043087981132569e+00, 5.195160018422946e+03, 1.121528966727754e+06,
      1.355294010638291e+07,-3.532929933499587e+06, 4.458815880568250e+05,
      6.331300459521475e+01,-1.375961122074268e+03,-7.415395959620902e+03,
     -1.414565234165133e+04,-7.015955939280531e+03],
    [5.138518e-03, 1.266855888751514e+00, 6.899463247856546e+02, 4.216236929666354e+04,
      2.147931207416082e+05,-2.876020054138726e+04, 1.086426843098747e+03,
      2.707637792335356e+01,-2.674955385301423e+02, 9.078532792700140e+02,
      9.825987715332953e+02,-5.482196655744529e-01],
    [5.113070e-03, 1.397874023016620e+00, 1.345883636860944e+01, 2.434301005382227e+04,
      1.444198486149228e+05, 1.003697333447050e+05, 2.410941641426479e+05,
      3.067204057507511e+00, 1.320239652333805e+02, 3.292073502673707e+02,
     -2.743878779348838e+02,-4.252396254964442e+02],
    [5.050000e-03,-2.022446304463027e-01,-5.956596895307574e+01,-4.801712239322800e+02,
      2.384455230908337e+02,-5.757163499587050e+01, 5.798541137001251e+00,
      1.748413100133951e+01, 5.819390284035368e+01,-5.863429968285631e+01,
     -1.179113239482045e-01,-4.268683173977720e-09],
    [5.307046e-03, 2.802854397382923e+00, 4.462442222692911e+02, 7.021330954933401e+03,
      1.885437964587811e+04, 1.508056246869293e+04,-2.094355307813514e+01,
      1.481417448480583e+01, 6.732984477458869e+01,-1.206040482897068e+02,
     -1.057879287786618e+02,-2.765378408631323e-03],
    [1.509115e-02,-9.141796920496276e-01, 1.342735109731877e+03, 3.936577516657912e+05,
      3.699261691007629e+06,-1.507356098885682e+06, 2.205561499489647e+05,
      3.866992630227200e+01,-1.228093930062573e+03, 5.906456100095043e+03,
      7.791981385860539e+03, 1.873855300354582e+01],
], dtype=cp.float64)

# Pair MB fit coefficients
_CF_MB = cp.array([
    [5e-3,-2.499999982474416e-01, 1.165664399869339e+00, 2.154793952021480e+00,
      1.270149386850713e+00, 2.483649225013684e-01,-4.186019883001905e-05,
     -1.378928715644350e+00, 1.156488416728070e+00, 4.976911470183872e-01,
      4.870465773876030e-06,-4.423392960087700e-07],
    [5e-3, 2.500000295508952e-01,-8.497183983207306e-01,-1.951820680989982e-02,
      2.619149841535253e-01,-4.534273305750967e-01,-3.145525674635611e-01,
      8.388694341455940e-01, 1.603695663575000e-06,-6.246243735852574e-01,
      9.857882804654846e-01, 5.012686787659838e-01],
    [5e-3, 1.334448916124861e+00, 1.533276418322284e+00, 2.504280783426653e-02,
      4.777495936484357e-02, 3.630334324421668e-01, 1.237866143758240e-01,
      1.289180844579062e+00, 1.602843358683278e-05, 1.263923257788066e-02,
     -5.997350601777206e-01, 3.521911887962246e-01],
    [5e-3,-6.167708949219610e-01,-2.537080095626051e-01, 1.307822038490171e-01,
     -1.204261677530989e-01,-1.206649718042400e-01,-8.620245076540467e-05,
     -1.298347445853823e+00,-7.248656296164483e-06,-2.865726715854026e-04,
     -6.865090702813017e-01, 4.033767918608367e-01],
    [5e-3, 1.757300002175146e-01,-1.263482053075663e-01, 5.466331979155107e-02,
      4.145104715957172e-03,-4.855688014271770e-04, 2.869576324453811e-05,
     -1.286512595383362e+00,-1.165742481270486e-04,-4.740203421187537e-03,
     -5.567875571587572e-01,-3.249642869760815e-01],
    [5e-3, 3.445954579556256e-01, 1.715184390725956e-02,-8.224756090070001e-02,
      7.735243406016939e-02, 1.035089219385419e-03,-6.484151006279233e-05,
     -1.279854181851254e+00,-1.070970483681416e-04,-4.365905834933606e-02,
     -4.632099590221615e-01, 2.926235258785276e-01],
    [5e-3, 1.354497402277264e+00, 1.685515273189768e+00, 3.267336413581359e-02,
     -1.215634951152663e-02, 9.006177715282025e-01, 9.200311837421076e-01,
      1.136664193298283e+00,-1.011738645854246e-04, 6.378569015548268e-04,
      8.207819113317577e-01, 8.307667300391079e-01],
    [5e-3,-1.693122019768674e-01, 9.231352641200651e-02,-1.787657927723367e-02,
     -1.648168837655593e-02, 2.008204276597000e-03,-1.214848156108869e-04,
     -1.001192015055246e+00,-1.528540714747427e-07, 6.544951602958429e-02,
      3.532367548183314e-01, 2.868536488193641e-01],
    [5e-3, 1.427805390573128e+00, 2.108235342790945e+00, 5.698686101168220e-02,
      1.538429638251989e-02, 4.805901339731146e-01, 2.249137271154556e-01,
      1.286277704062037e+00,-1.490352315543504e-05, 1.373451534868378e-01,
      5.996424201465841e-01, 4.107768773410937e-01],
    [5e-3, 1.331703764524452e-01,-8.242976148134072e-02, 2.876821851261879e-02,
      7.924040945752319e-03,-9.931798750875400e-04, 6.253327411129789e-05,
      1.272211977886686e+00,-5.812561617915775e-04,-1.858813454068888e-01,
     -4.861222841829775e-01,-3.301018709137714e-01],
], dtype=cp.float64)

# =============================================================================
# GPU-resident index maps for pair block assembly — built once at module load
# =============================================================================
_BLK_R_OFF   = cp.array([0, 6, 0, 6], dtype=cp.int32)
_BLK_C_OFF   = cp.array([0, 6, 6, 0], dtype=cp.int32)
_lr, _lc     = cp.meshgrid(cp.arange(6, dtype=cp.int32),
                            cp.arange(6, dtype=cp.int32), indexing='ij')
_LOCAL_ROW   = _lr.ravel()                                    # (36,)
_LOCAL_COL   = _lc.ravel()
_ENTRY_BLOCK = cp.repeat(cp.arange(4, dtype=cp.int32), 36)   # (144,)
_ENTRY_LROW  = cp.tile(_LOCAL_ROW, 4)                        # (144,)
_ENTRY_LCOL  = cp.tile(_LOCAL_COL, 4)
_G_EMROW     = (_BLK_R_OFF[_ENTRY_BLOCK] + _ENTRY_LROW).astype(cp.int32)
_G_EMCOL     = (_BLK_C_OFF[_ENTRY_BLOCK] + _ENTRY_LCOL).astype(cp.int32)

# =============================================================================
# Default dtype and coefficient cache
# =============================================================================
# Change this to cp.float64 to make float64 the default for all new instances.
_DEFAULT_DTYPE = cp.float32

# Arrays that need to be cast per-dtype.  int32 index maps are dtype-agnostic.
_FLOAT_COEFF_NAMES = (
    '_RPY_P', '_RPY_Q',
    '_DR_P',  '_DR_Q',  '_DR_ASYM_CUT', '_DR_RPY_CUT',
    '_CF_SUP', '_CF_MB',
)

# Cache: dtype -> dict of name -> cast array.  Built on first use per dtype.
_COEFF_CACHE: dict = {}


def _get_coeffs(dtype):
    """
    Return a dict of all float coefficient arrays cast to `dtype`.
    Results are cached so the cast is paid at most once per dtype.
    """
    if dtype not in _COEFF_CACHE:
        g = globals()
        _COEFF_CACHE[dtype] = {
            name: cp.asarray(g[name], dtype=dtype)
            for name in _FLOAT_COEFF_NAMES
        }
    return _COEFF_CACHE[dtype]

# =============================================================================
# GPU helper functions — all dtype-aware via C (cached coeff dict)
# =============================================================================

def _eval_rat_wall_all(h_gpu, h_scale, p_gpu, q_gpu, pmin_arr, np_arr, nq_arr):
    """All 5 wall rational fits simultaneously. dtype follows h_gpu."""
    dtyp   = h_gpu.dtype
    eps_fl = cp.finfo(dtyp).tiny
    u      = 1.0 / (1.0 + h_gpu[None, :] / h_scale)
    pmin_f = pmin_arr.astype(dtyp)[:, None]
    upow   = cp.exp(pmin_f * cp.log(cp.maximum(u, eps_fl)))
    P      = cp.zeros((5, h_gpu.shape[0]), dtype=dtyp)
    for i in range(p_gpu.shape[1]):
        P    += cp.where((np_arr > i)[:, None], p_gpu[:, i:i+1] * upow, 0.0)
        upow  = upow * u
    Q      = cp.ones((5, h_gpu.shape[0]), dtype=dtyp)
    upow   = u.copy()
    for i in range(q_gpu.shape[1]):
        Q    += cp.where((nq_arr > i)[:, None], q_gpu[:, i:i+1] * upow, 0.0)
        upow  = upow * u
    return P / Q


def _rpy_wall_batch(h_gpu, C):
    """All 5 RPY wall scalars. Returns (5, W)."""
    return _eval_rat_wall_all(h_gpu, _RPY_HSCALE,
                              C['_RPY_P'], C['_RPY_Q'],
                              _RPY_PMIN, _RPY_NP, _RPY_NQ)


def _asym_wall_batch(eps_gpu):
    """AT asymptotics for all 5 wall scalars. Returns (5, W)."""
    dtyp = eps_gpu.dtype
    efl = cp.finfo(dtyp).tiny
    le  = cp.log(cp.maximum(eps_gpu, efl))
    out = cp.empty((5, eps_gpu.shape[0]), dtype=dtyp)
    out[0] = 1.0/eps_gpu - (1.0/5.0)*le + 0.971280 - 1.0
    out[1] = -(8.0/15.0)*le + 0.9588 - 1.0
    out[2] = (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.029 - (0.4576-0.2)*eps_gpu)
    out[3] = (4.0/3.0)*(1.20206 - 3.0*6.449340668482264e-01*eps_gpu) - 4.0/3.0
    out[4] = (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*eps_gpu) - 4.0/3.0
    return out


def _delta_R_wall_batch(eps_gpu, h_gpu, rpy_scalars, C):
    """Chimera Delta_R for all 5 wall scalars. Returns (5, W)."""
    dr_fit = _eval_rat_wall_all(h_gpu, _DR_HSCALE,
                                C['_DR_P'], C['_DR_Q'],
                                _DR_PMIN, _DR_NP, _DR_NQ)
    asym   = _asym_wall_batch(eps_gpu)
    e      = eps_gpu[None, :]
    near   = e < C['_DR_ASYM_CUT'][:, None]
    mid    = (~near) & (e <= C['_DR_RPY_CUT'][:, None])
    out    = cp.zeros_like(rpy_scalars)
    out    = cp.where(near, asym - rpy_scalars, out)
    out    = cp.where(mid,  dr_fit,             out)
    return out


def _wall_matrices_batch(h_wall, eps_wall, f0, f1, f2, C):
    """Returns R_rpy (W,6,6) and R_sup (W,6,6) in h_wall.dtype.
    Rational fits always evaluated in float64 to avoid catastrophic cancellation
    between RPY and Delta_R at small gaps, then cast back to working dtype.
    Cost is negligible — only N particles, not M pairs.
    """
    dtyp = h_wall.dtype
    W    = h_wall.shape[0]
    C64  = _get_coeffs(cp.float64)
    h64  = h_wall.astype(cp.float64)
    e64  = eps_wall.astype(cp.float64)
    rpy  = _rpy_wall_batch(h64, C64).astype(dtyp)
    dR   = _delta_R_wall_batch(e64, h64,
                               rpy.astype(cp.float64), C64).astype(dtyp)

    def fill_6x6(Xa, Ya, Yb, Xc, Yc):
        R = cp.zeros((W, 6, 6), dtype=dtyp)
        R[:, 0, 0] = f0*Ya;  R[:, 1, 1] = f0*Ya;  R[:, 2, 2] = f0*Xa
        R[:, 3, 3] = f2*Yc;  R[:, 4, 4] = f2*Yc;  R[:, 5, 5] = f2*Xc
        R[:, 0, 4] =  f1*Yb; R[:, 4, 0] =  f1*Yb
        R[:, 1, 3] = -f1*Yb; R[:, 3, 1] = -f1*Yb
        return R

    R_rpy = fill_6x6(rpy[0], rpy[1], rpy[2], rpy[3], rpy[4])
    R_sup = fill_6x6(rpy[0]+dR[0], rpy[1]+dR[1], rpy[2]+dR[2],
                     rpy[3]+dR[3], rpy[4]+dR[4])
    return R_rpy, R_sup


def _eval_pair_PQ_batch(cf_gpu, eps, ep, eq):
    """P/Q for pair scalars; Q uses c_i^2. Returns (10, M)."""
    p = cf_gpu[:, 1:7]
    c = cf_gpu[:, 7:12]
    P = cp.einsum('si,mi->sm', p, ep)
    Q = 1.0 + cp.einsum('si,mi->sm', c*c, eq)
    return P / Q


def _pair_scalars_batch(r_norms, debye_cut, cf_gpu, is_sup):
    """All 10 pair scalars for M pairs. Returns (10, M)."""
    dtyp = r_norms.dtype
    efl  = cp.finfo(dtyp).tiny
    eps  = cp.maximum(r_norms - 2.0, debye_cut)
    eps2 = eps*eps; eps3=eps2*eps; eps4=eps3*eps; eps5=eps4*eps
    ep   = cp.stack([cp.ones_like(eps), eps, eps2, eps3, eps4, eps5], axis=1)
    eq   = ep[:, 1:]

    raw = _eval_pair_PQ_batch(cf_gpu, eps, ep, eq)
    raw[0] =  0.25/eps + raw[0]/eps
    raw[1] = -0.25/eps + raw[1]/eps

    if is_sup:
        li  = cp.log(1.0 / cp.maximum(eps, efl))
        at  = cp.stack([
             0.995419 + 0.25/eps + 0.225*li + 0.0267857*eps*li,
            -0.350153 - 0.25/eps - 0.225*li - 0.0267857*eps*li,
             0.998317 + 0.166667*li,
            -0.273652 - 0.166667*li,
            -0.666667*(0.23892 - 0.25*li - 0.125*eps*li),
             0.666667*(-0.00162268 + 0.25*li + 0.125*eps*li),
             1.33333*(1.0518 - 0.125*eps*li),
             1.33333*(-0.150257 + 0.125*eps*li),
             1.33333*(0.702834 + 0.2*li + 0.188*eps*li),
             1.33333*(-0.027464 + 0.05*li + 0.062*eps*li),
        ], axis=0)
        use_at = eps[None, :] < cf_gpu[:, 0:1]
        raw    = cp.where(use_at, at, raw)

    raw[5] = -raw[5]
    return raw


def _assemble_pair_triplets(scalars, j_idx, k_idx, n_dof, f0, f1, f2, r_hat_gpu):
    """Build COO triplets for M pairs. Returns (rows, cols, vals)."""
    dtyp = r_hat_gpu.dtype
    M  = r_hat_gpu.shape[0]
    rh = r_hat_gpu

    X11A,X12A,Y11A,Y12A,Y11B,Y12B,X11C,X12C,Y11C,Y12C = [scalars[i] for i in range(10)]

    sq = cp.einsum('mi,mj->mij', rh, rh)
    sh = cp.eye(3, dtype=dtyp)[None] - sq
    vt = cp.zeros((M, 3, 3), dtype=dtyp)
    vt[:, 0, 1] =  rh[:, 2];  vt[:, 0, 2] = -rh[:, 1]
    vt[:, 1, 0] = -rh[:, 2];  vt[:, 1, 2] =  rh[:, 0]
    vt[:, 2, 0] =  rh[:, 1];  vt[:, 2, 1] = -rh[:, 0]
    vt *= -1.0

    def blk(X, Y): return X[:, None, None]*sq + Y[:, None, None]*sh

    A11 = f0*blk(X11A,Y11A);  A12 = f0*blk(X12A,Y12A)
    B11 = -f1*Y11B[:, None, None]*vt
    B12 =  f1*Y12B[:, None, None]*vt
    C11 = f2*blk(X11C,Y11C);  C12 = f2*blk(X12C,Y12C)

    R = cp.zeros((M, 12, 12), dtype=dtyp)
    R[:,  0:3,  0:3] = A11;  R[:,  0:3,  3:6] = B11
    R[:,  0:3,  6:9] = A12;  R[:,  0:3, 9:12] = B12
    R[:,  3:6,  0:3] =-B11;  R[:,  3:6,  3:6] = C11
    R[:,  3:6,  6:9] = B12;  R[:,  3:6, 9:12] = C12
    R[:,  6:9,  0:3] = A12;  R[:,  6:9,  3:6] =-B12
    R[:,  6:9,  6:9] = A11;  R[:,  6:9, 9:12] =-B11
    R[:, 9:12,  0:3] =-B12;  R[:, 9:12,  3:6] = C12
    R[:, 9:12,  6:9] = B11;  R[:, 9:12, 9:12] = C11

    vals = R[:, _G_EMROW, _G_EMCOL]

    j6 = j_idx * 6;  k6 = k_idx * 6
    dof_r = cp.stack([j6, k6, j6, k6], axis=1)
    dof_c = cp.stack([j6, k6, k6, j6], axis=1)
    rows  = dof_r[:, _ENTRY_BLOCK] + _ENTRY_LROW[None, :]
    cols  = dof_c[:, _ENTRY_BLOCK] + _ENTRY_LCOL[None, :]

    rows = rows.ravel();  cols = cols.ravel();  vals = vals.ravel()
    # dtype-appropriate zero threshold
    thr  = 1e-6 if dtyp == cp.float32 else 1e-12
    mask = cp.abs(vals) > thr
    return rows[mask].astype(cp.int32), cols[mask].astype(cp.int32), vals[mask]


def _build_coo(rows_list, cols_list, vals_list, n_dof):
    """Concatenate triplet lists into a cupyx COO matrix."""
    if vals_list:
        r = cp.concatenate(rows_list).astype(cp.int32)
        c = cp.concatenate(cols_list).astype(cp.int32)
        v = cp.concatenate(vals_list)
    else:
        r = cp.empty(0, dtype=cp.int32)
        c = cp.empty(0, dtype=cp.int32)
        v = cp.empty(0, dtype=cp.float32)
    return cpsp.coo_matrix((v, (r, c)), shape=(n_dof, n_dof))


class Lubrication:
    """
    CuPy-vectorised lubrication resistance matrix.
    Drop-in replacement for the C++ nanobind Lubrication class.

    Parameters
    ----------
    d_cut : float
        Debye (minimum epsilon) cutoff.
    dtype : cp.dtype, optional
        Floating-point precision for all GPU computations.
        Default: cp.float32.  Pass cp.float64 for double precision.
        If r_gpu is supplied to ResistCSC_both with a different dtype,
        that dtype takes precedence for that call.
    """

    def __init__(self, d_cut: float, dtype=None):
        self.debye_cut = d_cut
        self.dtype     = dtype if dtype is not None else _DEFAULT_DTYPE
        # Pre-populate the coeff cache for this dtype on construction
        self._C = _get_coeffs(self.dtype)

    # Physical cutoff for pair rational fits — hardcoded because the fits
    # are only valid up to this separation.  Do not override.
    _PAIR_CUTOFF = 4.5

    def ResistCSC(self, r_vectors, n_list, a, eta, cutoff,
                  wall_cutoff, periodic_length, Sup_if_true):
        """wall_cutoff and cutoff args accepted for API compatibility but ignored."""
        R_MB, R_Sup = self.ResistCSC_both(r_vectors, n_list, a, eta,
                                          periodic_length=periodic_length)
        return R_Sup if Sup_if_true else R_MB

    def ResistCSC_both(self, r_vectors, n_list, a, eta,
                       cutoff=None, wall_cutoff=None, periodic_length=None,
                       r_gpu=None, j_gpu=None, k_gpu=None, pl_gpu=None):
        """
        Build both MB and Sup sparse resistance matrices as cupyx CSC on GPU.

        cutoff and wall_cutoff are accepted for API compatibility with the C++
        nanobind class but are ignored:
          - pair cutoff is hardcoded to 4.5a (the fit validity range)
          - wall blocks are always applied to all particles (the rational fits
            decay smoothly to zero in the far field)

        If r_gpu is provided its dtype overrides self.dtype for this call.
        """
        N     = len(r_vectors)
        n_dof = 6 * N

        # Resolve dtype: r_gpu wins if supplied
        dtyp = r_gpu.dtype if r_gpu is not None else self.dtype
        C  = _get_coeffs(dtyp)   # O(1) after first call per dtype

        f0 = cp.dtype(dtyp).type(_6PI * eta * a)
        f1 = cp.dtype(dtyp).type(_6PI * eta * a * a)
        f2 = cp.dtype(dtyp).type(_6PI * eta * a * a * a)

        pl_np = np.asarray(periodic_length if periodic_length is not None
                           else np.zeros(3), dtype=np.float64)
        pl_g  = pl_gpu.astype(dtyp) if pl_gpu is not None \
                else cp.asarray(pl_np, dtype=dtyp)

        if r_gpu is not None:
            r_g = r_gpu.astype(dtyp) if r_gpu.dtype != dtyp else r_gpu
        else:
            r_g = cp.asarray(
                np.asarray(r_vectors, dtype=np.float64).reshape(N, 3) / a,
                dtype=dtyp)

        rows_mb, cols_mb, vals_mb = [], [], []
        rows_sup, cols_sup, vals_sup = [], [], []

        # ── Wall blocks: always applied to ALL N particles ────────────────
        # The rational fits decay smoothly to zero; no cutoff needed.
        h_all   = r_g[:, 2]
        eps_all = cp.maximum(h_all - 1.0, cp.dtype(dtyp).type(self.debye_cut))
        h_all   = 1.0 + eps_all

        R_rpy_blk, R_sup_blk = _wall_matrices_batch(
            h_all, eps_all, f0, f1, f2, C)

        all_idx = cp.arange(N, dtype=cp.int32)
        lr, lc  = cp.meshgrid(cp.arange(6, dtype=cp.int32),
                              cp.arange(6, dtype=cp.int32), indexing='ij')
        lr = lr.ravel(); lc = lc.ravel()
        thr = 1e-6 if dtyp == cp.float32 else 1e-12
        for R_blk, r_list, c_list, v_list in [
                (R_rpy_blk, rows_mb,  cols_mb,  vals_mb),
                (R_sup_blk, rows_sup, cols_sup, vals_sup)]:
            v    = R_blk[:, lr, lc]
            gr   = (all_idx[:, None] * 6 + lr[None, :]).ravel()
            gc   = (all_idx[:, None] * 6 + lc[None, :]).ravel()
            gv   = v.ravel()
            mask = cp.abs(gv) > thr
            if mask.any():
                r_list.append(gr[mask]); c_list.append(gc[mask])
                v_list.append(gv[mask])

        # ── Pair blocks: hardcoded cutoff = 4.5a ─────────────────────────
        if j_gpu is not None:
            j_g = j_gpu;  k_g = k_gpu;  has_pairs = j_g.size > 0
        else:
            nb_sizes  = np.array([len(nb) for nb in n_list], dtype=np.int32)
            has_pairs = nb_sizes.sum() > 0
            if has_pairs:
                j_all = np.repeat(np.arange(N, dtype=np.int32), nb_sizes)
                k_all = np.concatenate([np.asarray(nb, dtype=np.int32)
                                        for nb in n_list if len(nb) > 0])
                j_g = cp.asarray(j_all, dtype=cp.int32)
                k_g = cp.asarray(k_all, dtype=cp.int32)

        if has_pairs:
            d_g = r_g[j_g] - r_g[k_g]
            for l in range(3):
                if pl_np[l] > 0.0:
                    d_g[:, l] -= cp.round(d_g[:, l] / pl_g[l]) * pl_g[l]
            r_norms_g = cp.sqrt((d_g * d_g).sum(axis=1))
            in_cut    = r_norms_g < self._PAIR_CUTOFF
            if in_cut.any():
                j_g = j_g[in_cut];  k_g = k_g[in_cut]
                d_g = d_g[in_cut];  r_norms_g = r_norms_g[in_cut]
                r_hat_g = -d_g / r_norms_g[:, None]

                for is_sup, cf_g, r_list, c_list, v_list in [
                        (False, C['_CF_MB'],  rows_mb,  cols_mb,  vals_mb),
                        (True,  C['_CF_SUP'], rows_sup, cols_sup, vals_sup)]:
                    sc = _pair_scalars_batch(r_norms_g, self.debye_cut,
                                            cf_g, is_sup)
                    pr, pc, pv = _assemble_pair_triplets(
                        sc, j_g, k_g, n_dof, f0, f1, f2, r_hat_g)
                    if pr.size > 0:
                        r_list.append(pr); c_list.append(pc); v_list.append(pv)

        def build_csc(rl, cl, vl):
            if rl:
                r = cp.concatenate(rl).astype(cp.int32)
                c = cp.concatenate(cl).astype(cp.int32)
                v = cp.concatenate(vl)
            else:
                r = cp.empty(0, dtype=cp.int32)
                c = cp.empty(0, dtype=cp.int32)
                v = cp.empty(0, dtype=dtyp)
            return cpsp.coo_matrix((v, (r, c)), shape=(n_dof, n_dof)).tocsc()

        return build_csc(rows_mb, cols_mb, vals_mb), \
               build_csc(rows_sup, cols_sup, vals_sup)
