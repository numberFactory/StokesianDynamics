"""
lubrication_cupy.py
-------------------
Fused GPU assembly of the lubrication resistance matrices.

The sparsity pattern is the 6x6-block pattern {diagonal blocks} U
{(i,k), (k,i) : pair (i,k)}.  It is built once per neighbour list, and two
kernels (one thread per off-diagonal block, one thread per particle) write
the four matrices the solver needs straight into persistent CSR buffers:

    R_MB              lubrication-free (RPY) near-field resistance
    R_SupS            R_Sup + diag(small)       -> preconditioner factorisation
    Delta_R           R_Sup - R_MB              -> GMRES matvec
    Delta_RS          R_Sup - R_MB + diag(small)-> Brownian square root

No host synchronisation, no sorting and no allocation happen during
assembly; the CSR matrices are views of the persistent buffers, so they
refresh in place.

Brownian increments: Delta_R is a sum of positive semi-definite pair blocks
(12x12) and wall blocks (6x6), so ``brownian()`` draws a vector with
covariance Delta_R + diag(small) exactly, pair by pair (a 12x12 Cholesky per
pair and a 6x6 one per particle, in two kernels), without factorising the
global matrix.  Columns inside each row are sorted and the matrices are
exactly symmetric, so the same arrays are also valid CSC arrays.

The kernel arithmetic mirrors ``lubrication.cpp`` operation by operation and
its coefficient tables are generated from that file.

Public API
~~~~~~~~~~
BlockPattern(N, pairs)       host-side block pattern from an upper-triangle pair list
LubricationAssembler(...)    device buffers + kernels; set_pattern(), assemble(),
                             brownian()
KERNEL_SOURCE                the CUDA source (also compiles as C++ for CPU tests)
"""
import numpy as np

# =============================================================================
# CUDA source.  Coefficient tables are generated from lubrication.cpp.
# =============================================================================
KERNEL_SOURCE = r"""
// ---- coefficient tables generated from lubrication.cpp by gen_tables.py ----
// pair fits, rows = X11A X12A Y11A Y12A Y11B Y12B X11C X12C Y11C Y12C
// layout {crossover_or_dcut, p0..p5, c0..c4}; P = sum p_i e^i, Q = 1 + sum c_i^2 e^(i+1)
__constant__ double LUB_CF_SUP[10][12] = {
  {0.07006673, -19.81052779593092, 2340.828937668049, 496583.4145654573, 1139263.177405481, -31074.90582481417, 6123.7462736942, -546.5471058359071, -1069.447402186549, -6.581289099212643e-09, -9.673175257729874e-10, -0.006988711115145478},
  {0.024579, -0.01593939244959434, 4.445665389795823, -407.8726991721069, -10343.51092030956, -8219.652436471222, -137017.3545248133, 0.0002488928114444593, -125.8809718612856, 1.407296176369766e-08, -489.2039651123681, 340.798406089767},
  {0.005228588, 2.171207126379109, 385.3002949507692, 6624.352544471424, 14846.97318890232, 7047.767136709095, 6540.53110009895, -15.11807925068774, 70.12000660040704, -119.3647614569199, -80.41378517885596, -81.22175958097344},
  {0.005248093, -1.461562640575963, -250.7841615126929, -4331.083375391881, -9737.921752082497, -405.8329204467756, 36.0694371181188, -15.96895122841455, 81.097522696183, -163.8005603371911, 125.4673064262033, 5.431104446475567},
  {0.005313637, -1.043087981132569, 5195.160018422946, 1121528.966727754, 13552940.10638291, -3532929.933499587, 445881.588056825, 63.31300459521475, -1375.961122074268, -7415.395959620902, -14145.65234165133, -7015.955939280531},
  {0.005138518, 1.266855888751514, 689.9463247856546, 42162.36929666354, 214793.1207416082, -28760.20054138726, 1086.426843098747, 27.07637792335356, -267.4955385301423, 907.853279270014, 982.5987715332953, -0.5482196655744529},
  {0.00511307, 1.39787402301662, 13.45883636860944, 24343.01005382227, 144419.8486149228, 100369.733344705, 241094.1641426479, 3.067204057507511, 132.0239652333805, 329.2073502673707, -274.3878779348838, -425.2396254964442},
  {0.00505, -0.2022446304463027, -59.56596895307574, -480.17122393228, 238.4455230908337, -57.5716349958705, 5.798541137001251, 17.48413100133951, 58.19390284035368, -58.63429968285631, -0.1179113239482045, -4.26868317397772e-09},
  {0.005307046, 2.802854397382923, 446.2442222692911, 7021.330954933401, 18854.37964587811, 15080.56246869293, -20.94355307813514, 14.81417448480583, 67.3298447745887, -120.6040482897068, -105.7879287786618, -0.002765378408631323},
  {0.01509115, -0.9141796920496276, 1342.735109731877, 393657.7516657912, 3699261.691007629, -1507356.098885682, 220556.1499489647, 38.669926302272, -1228.093930062573, 5906.456100095043, 7791.981385860539, 18.73855300354582}
};
__constant__ double LUB_CF_MB[10][12] = {
  {0.005, -0.2499999982474416, 1.165664399869339, 2.15479395202148, 1.270149386850713, 0.2483649225013684, -4.186019883001905e-05, -1.37892871564435, 1.15648841672807, 0.4976911470183872, 4.87046577387603e-06, -4.4233929600877e-07},
  {0.005, 0.2500000295508952, -0.8497183983207306, -0.01951820680989982, 0.2619149841535253, -0.4534273305750967, -0.3145525674635611, 0.838869434145594, 1.603695663575e-06, -0.6246243735852574, 0.9857882804654846, 0.5012686787659838},
  {0.005, 1.334448916124861, 1.533276418322284, 0.02504280783426653, 0.04777495936484357, 0.3630334324421668, 0.123786614375824, 1.289180844579062, 1.602843358683278e-05, 0.01263923257788066, -0.5997350601777206, 0.3521911887962246},
  {0.005, -0.616770894921961, -0.2537080095626051, 0.1307822038490171, -0.1204261677530989, -0.12066497180424, -8.620245076540467e-05, -1.298347445853823, -7.248656296164483e-06, -0.0002865726715854026, -0.6865090702813017, 0.4033767918608367},
  {0.005, 0.1757300002175146, -0.1263482053075663, 0.05466331979155107, 0.004145104715957172, -0.000485568801427177, 2.869576324453811e-05, -1.286512595383362, -0.0001165742481270486, -0.004740203421187537, -0.5567875571587672, -0.3249642869760815},
  {0.005, 0.3445954579556256, 0.01715184390725956, -0.08224756090070001, 0.0773524340601694, 0.001035089219385419, -6.484151006279233e-05, -1.279854181851254, -0.0001070970483681416, -0.04365905834933606, -0.4632099590221615, 0.2926235258785276},
  {0.005, 1.354497402277264, 1.685515273189768, 0.03267336413581359, -0.01215634951152663, 0.9006177715282025, 0.9200311837421076, 1.136664193298283, -0.0001011738645854246, 0.0006378569015548268, 0.8207819113317577, 0.830766730039108},
  {0.005, -0.1693122019768674, 0.0923135264120065, -0.01787657927723367, -0.01648168837655593, 0.002008204276597, -0.0001214848156108869, -1.001192015055246, -1.528540714747427e-07, 0.06544951602958429, 0.3532367548183314, 0.2868536488193641},
  {0.005, 1.427805390573128, 2.108235342790945, 0.0569868610116822, 0.01538429638251989, 0.4805901339731146, 0.2249137271154556, 1.286277704062037, -1.490352315543504e-05, 0.1373451534868378, 0.5996424201465841, 0.4107768723410937},
  {0.005, 0.1331703764524452, -0.08242976148134072, 0.02876821851261879, 0.007924040945752319, -0.00099317987508754, 6.253327411129789e-05, 1.272211977886686, -0.0005812561617915775, -0.1858813454068888, -0.4861222841829775, -0.3301018709137714}
};
// wall RPY fits, rows = Xa Ya Yb Xc Yc;  u = 1/(1 + h/0.5)
__constant__ double LUB_RPY_P[5][4] = {
  {2.249994311366312, -9.302942542409326, 10.18898977062156, -0.1461983146752135},
  {1.124980798925478, -5.161218663062001, 7.210326948819593, -2.621851150554913},
  {-2.019389402582813, 7.475392505257506, -3.877269126222068, -1.028617914833345},
  {1.333114620704834, -4.215726080748013, -0.05775019233806103, 0.1969213190855149},
  {3.339371808444304, -14.06703500455225, 8.727632354389423, 9.801957547519073}
};
__constant__ double LUB_RPY_Q[5][4] = {
  {-7.384833913349599, 19.74855818648951, -20.40618954066672, 4.353438654549962},
  {-6.713754414307387, 17.06822775135723, -19.33894500702774, 7.41343677801746},
  {-8.683501976306239, 28.35163248843032, -42.34435891678831, 25.31363165467517},
  {-6.164932696556416, 12.47407137449759, -11.35905628832895, 6.141510604219781},
  {-7.179264725936619, 17.82208382807917, -18.84878181254139, 8.912717591979181}
};
__constant__ int LUB_RPY_PMIN[5] = {1, 1, 4, 3, 3};
__constant__ int LUB_RPY_NP[5]   = {4, 4, 4, 4, 4};
__constant__ int LUB_RPY_NQ[5]   = {4, 4, 4, 4, 4};
// wall Delta_R chimera fits (zero padded)
__constant__ double LUB_DR_P[5][5] = {
  {0.003390440744007662, -0.1292707770613473, 1.621030547158851, 0.0, 0.0},
  {-0.001802661869524016, 0.009468660270547602, 0.3218091451875633, -0.9812949264604398, 0.0},
  {-1.17172041890462, 26.34024743369987, -203.7770517911233, 660.4148051813611, -764.4275364225128},
  {-0.06769668477673338, 0.4251981921512744, -0.4761218699317399, -0.5662068770688917, 0.0},
  {0.04774546185054501, -1.200098819024352, 11.43510786982436, -47.54910568257404, 72.29905586544142}
};
__constant__ double LUB_DR_Q[5][4] = {
  {-9.934759618046485, 38.52782844267631, -69.19767486523838, 48.12890822657162},
  {-10.77793936835893, 45.59152441171614, -90.09502525501624, 69.97760797588727},
  {-14.58404137489715, 78.17171888830218, -182.0761238925333, 155.4914960487163},
  {-7.986015731409064, 21.07903159501786, -18.36099133069141, 0.0},
  {-14.20388909734784, 73.66905048843127, -164.4525346135324, 132.9770561654655}
};
__constant__ int LUB_DR_PMIN[5] = {1, 1, 4, 3, 3};
__constant__ int LUB_DR_NP[5]   = {3, 4, 5, 4, 5};
__constant__ int LUB_DR_NQ[5]   = {4, 4, 4, 3, 4};
__constant__ double LUB_DR_ASYM_CUT[5] = {0.20549, 0.029118, 0.1, 0.0097, 0.0456};
__constant__ double LUB_DR_RPY_CUT[5]  = {7.0, 5.6, 3.4, 0.4, 5.0};

// ============================================================================
// Fused lubrication assembly kernels.
//
// The matrices are assembled directly into a CSR array whose pattern is the
// 6x6-block pattern {diagonal blocks} U {(i,k),(k,i) : pair (i,k)}.  For block
// row i with deg_i blocks (diagonal included) starting at block index bptr[i],
// scalar row 6i+lr starts at 36*bptr[i] + lr*6*deg_i and entry (lr,lc) of the
// block with block index b sits at
//         36*bptr[i] + lr*6*deg_i + 6*(b - bptr[i]) + lc .
// Columns inside a row are therefore sorted, so the CSR is canonical and,
// because the matrix is symmetric, the very same arrays are also its CSC.
//
// Four value arrays are written in one pass:
//   vMB  = R_MB
//   vSS  = R_Sup + diag(small)          (preconditioner factorization)
//   vDR  = R_Sup - R_MB                 (Delta_R, used in the GMRES matvec)
//   vDRS = R_Sup - R_MB + diag(small)   (Brownian square root)
//
// Arithmetic mirrors lubrication.cpp operation by operation (same formulas,
// same operand order) so CPU and GPU results agree to rounding.
// ============================================================================

#define LUB_PAIR_CUTOFF 4.5
#define LUB_PI 3.14159265358979323846

// ---------------------------------------------------------------------------
// Pair fits.  SUP selects the table and the asymptotic (AT) fallback.
// ---------------------------------------------------------------------------
template <bool SUP>
__device__ __forceinline__ const double* lub_cf(int s) {
  return SUP ? LUB_CF_SUP[s] : LUB_CF_MB[s];
}

__device__ __forceinline__ double lub_eval_PQ(const double* cf, double e) {
  const double e2 = e*e, e3 = e2*e, e4 = e3*e, e5 = e4*e;
  const double P = cf[1] + cf[2]*e + cf[3]*e2 + cf[4]*e3 + cf[5]*e4 + cf[6]*e5;
  const double Q = 1.0 + cf[7]*cf[7]*e + cf[8]*cf[8]*e2 + cf[9]*cf[9]*e3
                 + cf[10]*cf[10]*e4 + cf[11]*cf[11]*e5;
  return P / Q;
}

// Scalars needed for the self (11) blocks: X11A, Y11A, Y11B, X11C, Y11C.
template <bool SUP>
__device__ __forceinline__ void lub_scalars_self(double eps, double* s) {
  const double li = log(1.0 / eps);
  const double* c;
  c = lub_cf<SUP>(0);   // X11A (singular)
  s[0] = (SUP && eps < c[0]) ? (0.995419 + 0.25/eps + 0.225*li + 0.0267857*eps*li)
                             : (0.25/eps + lub_eval_PQ(c, eps) / eps);
  c = lub_cf<SUP>(2);   // Y11A
  s[1] = (SUP && eps < c[0]) ? (0.998317 + 0.166667*li) : lub_eval_PQ(c, eps);
  c = lub_cf<SUP>(4);   // Y11B
  s[2] = (SUP && eps < c[0]) ? (-0.666667*(0.23892 - 0.25*li - 0.125*eps*li))
                             : lub_eval_PQ(c, eps);
  c = lub_cf<SUP>(6);   // X11C
  s[3] = (SUP && eps < c[0]) ? (1.33333*(1.0518 - 0.125*eps*li)) : lub_eval_PQ(c, eps);
  c = lub_cf<SUP>(8);   // Y11C
  s[4] = (SUP && eps < c[0]) ? (1.33333*(0.702834 + 0.2*li + 0.188*eps*li))
                             : lub_eval_PQ(c, eps);
}

// Scalars needed for the coupling (12) blocks: X12A, Y12A, Y12B, X12C, Y12C.
// The Y12B fit stores -Y12B, so it is negated (both branches, as in the C++).
template <bool SUP>
__device__ __forceinline__ void lub_scalars_pair(double eps, double* s) {
  const double li = log(1.0 / eps);
  const double* c;
  c = lub_cf<SUP>(1);   // X12A (singular)
  s[0] = (SUP && eps < c[0]) ? (-0.350153 - 0.25/eps - 0.225*li - 0.0267857*eps*li)
                             : (-0.25/eps + lub_eval_PQ(c, eps) / eps);
  c = lub_cf<SUP>(3);   // Y12A
  s[1] = (SUP && eps < c[0]) ? (-0.273652 - 0.166667*li) : lub_eval_PQ(c, eps);
  c = lub_cf<SUP>(5);   // Y12B
  s[2] = -((SUP && eps < c[0]) ? (0.666667*(-0.00162268 + 0.25*li + 0.125*eps*li))
                               : lub_eval_PQ(c, eps));
  c = lub_cf<SUP>(7);   // X12C
  s[3] = (SUP && eps < c[0]) ? (1.33333*(-0.150257 + 0.125*eps*li)) : lub_eval_PQ(c, eps);
  c = lub_cf<SUP>(9);   // Y12C
  s[4] = (SUP && eps < c[0]) ? (1.33333*(-0.027464 + 0.05*li + 0.062*eps*li))
                             : lub_eval_PQ(c, eps);
}

// ---------------------------------------------------------------------------
// Wall fits (u = 1/(1 + h/0.5)).
// ---------------------------------------------------------------------------
__device__ __forceinline__ double lub_wall_rat(double h, int p_min, int n_p, int n_q,
                                               const double* p, const double* q) {
  const double u = 1.0 / (1.0 + h / 0.5);
  double P = 0.0, upow = pow(u, (double)p_min);
  for (int i = 0; i < n_p; ++i) { P += p[i] * upow; upow *= u; }
  double Q = 1.0; upow = u;
  for (int i = 0; i < n_q; ++i) { Q += q[i] * upow; upow *= u; }
  return P / Q;
}

__device__ __forceinline__ double lub_wall_asym(int s, double eps) {
  const double le = log(eps);
  switch (s) {
    case 0:  return 1.0/eps - (1.0/5.0)*le + 0.971280 - 1.0;
    case 1:  return -(8.0/15.0)*le + 0.9588 - 1.0;
    case 2:  return (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.029 - (0.4576-0.2)*eps);
    case 3:  return (4.0/3.0)*(1.20206 - 3.0*(LUB_PI*LUB_PI/6.0-1.0)*eps) - 4.0/3.0;
    default: return (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*eps) - 4.0/3.0;
  }
}

// rpy[s] = RPY wall scalar, dr[s] = Delta_R wall scalar (s = Xa Ya Yb Xc Yc).
__device__ __forceinline__ void lub_wall_scalars(double height, double debye_cut,
                                                 double* rpy, double* dr) {
  double eps = height - 1.0, h = height;
  if (eps < debye_cut) { eps = debye_cut; h = 1.0 + eps; }
  for (int s = 0; s < 5; ++s)
    rpy[s] = lub_wall_rat(h, LUB_RPY_PMIN[s], LUB_RPY_NP[s], LUB_RPY_NQ[s],
                          LUB_RPY_P[s], LUB_RPY_Q[s]);
  for (int s = 0; s < 5; ++s) {
    if (eps > LUB_DR_RPY_CUT[s])        dr[s] = 0.0;
    else if (eps < LUB_DR_ASYM_CUT[s])  dr[s] = lub_wall_asym(s, eps) - rpy[s];
    else dr[s] = lub_wall_rat(h, LUB_DR_PMIN[s], LUB_DR_NP[s], LUB_DR_NQ[s],
                              LUB_DR_P[s], LUB_DR_Q[s]);
  }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
// d = (r_k - r_i)/a with the minimum image in periodic directions (L > 0).
// Same operations as pair_separation() in lubrication.cpp (round half away
// from zero, correction applied before dividing by a).
__device__ __forceinline__ void lub_separation(const double* __restrict__ pos, int i, int k,
                                               double a, double Lx, double Ly, double Lz,
                                               double* d) {
  const double L[3] = {Lx, Ly, Lz};
  for (int l = 0; l < 3; ++l) {
    double x = pos[3*k + l] - pos[3*i + l];
    if (L[l] > 0.0) x -= round(x / L[l]) * L[l];
    d[l] = x / a;
  }
}

// [v x] (cross-product matrix) entry (p, q).
__device__ __forceinline__ double lub_cross(const double* v, int p, int q) {
  // [[0,-v2,v1],[v2,0,-v0],[-v1,v0,0]]
  if (p == q) return 0.0;
  if (p == 0) return (q == 1) ? -v[2] :  v[1];
  if (p == 1) return (q == 0) ?  v[2] : -v[0];
  return (q == 0) ? -v[1] : v[0];
}

// index of (p,q) in the packed symmetric 3x3 storage {xx,yy,zz,xy,xz,yz}
__device__ __forceinline__ int lub_sym(int p, int q) {
  if (p == q) return p;
  const int s = p + q;           // (0,1)->1, (0,2)->2, (1,2)->3
  return (s == 1) ? 3 : ((s == 2) ? 4 : 5);
}

// f*(X*sq + Y*sh) for the packed entry (p,q), sq = r r^T, sh = I - sq
__device__ __forceinline__ double lub_xy(double f, double X, double Y, const double* r,
                                         int p, int q) {
  const double sq = r[p]*r[q];
  const double sh = (p == q ? 1.0 : 0.0) - sq;
  return f*(X*sq + Y*sh);
}

__device__ __forceinline__ void lub_write(long long base, long long rstride, int lr, int lc,
                                          double vm, double vs, double shift,
                                          double* __restrict__ vMB, double* __restrict__ vSS,
                                          double* __restrict__ vDR, double* __restrict__ vDRS) {
  const long long p = base + lr*rstride + lc;
  const double dr = vs - vm;
  vMB[p]  = vm;
  vSS[p]  = vs + shift;
  vDR[p]  = dr;
  vDRS[p] = dr + shift;
}

// ---------------------------------------------------------------------------
// Kernel 1: off-diagonal blocks, one thread per block (diagonal blocks return).
//   block (i,k) = [[A12, [c x]], [[c x], C12]]  with r_hat pointing from i to k,
//   A12 = f0 (X12A sq + Y12A sh), c = f1 Y12B r_hat, C12 = f2 (X12C sq + Y12C sh).
// The same formula with r_hat -> -r_hat gives block (k,i) = block(i,k)^T, so
// both halves are produced bitwise symmetric.
// ---------------------------------------------------------------------------
extern "C" __global__ void lub_offdiag(
    const int nb, const double* __restrict__ pos,
    const int* __restrict__ brow, const int* __restrict__ bcol, const int* __restrict__ bptr,
    const double a, const double Lx, const double Ly, const double Lz,
    const double debye_cut, const double f0, const double f1, const double f2,
    double* __restrict__ vMB, double* __restrict__ vSS,
    double* __restrict__ vDR, double* __restrict__ vDRS)
{
  const int b = blockDim.x * blockIdx.x + threadIdx.x;
  if (b >= nb) return;
  const int i = brow[b], k = bcol[b];
  if (i == k) return;
  const int start = bptr[i];
  const long long rstride = 6LL * (bptr[i+1] - start);
  const long long base = 36LL * start + 6LL * (b - start);

  double d[3];
  lub_separation(pos, i, k, a, Lx, Ly, Lz, d);
  const double r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
  if (!(r < LUB_PAIR_CUTOFF)) {       // buffered pattern: pair outside the fit range
    for (int lr = 0; lr < 6; ++lr)
      for (int lc = 0; lc < 6; ++lc)
        lub_write(base, rstride, lr, lc, 0.0, 0.0, 0.0, vMB, vSS, vDR, vDRS);
    return;
  }
  const double rh[3] = {d[0]/r, d[1]/r, d[2]/r};
  const double eps = fmax(r - 2.0, debye_cut);
  double m[5], s[5];
  lub_scalars_pair<false>(eps, m);
  lub_scalars_pair<true>(eps, s);
  // c = f1*(Y12B*r_hat) (operand order of R.block(0,9) = f1*(Y12B*vortMat))
  const double cm[3] = {f1*(m[2]*rh[0]), f1*(m[2]*rh[1]), f1*(m[2]*rh[2])};
  const double cs[3] = {f1*(s[2]*rh[0]), f1*(s[2]*rh[1]), f1*(s[2]*rh[2])};
#pragma unroll
  for (int lr = 0; lr < 6; ++lr) {
#pragma unroll
    for (int lc = 0; lc < 6; ++lc) {
      double vm, vs;
      if (lr < 3 && lc < 3) {
        vm = lub_xy(f0, m[0], m[1], rh, lr, lc);
        vs = lub_xy(f0, s[0], s[1], rh, lr, lc);
      } else if (lr < 3) {
        vm = lub_cross(cm, lr, lc - 3);
        vs = lub_cross(cs, lr, lc - 3);
      } else if (lc < 3) {
        vm = lub_cross(cm, lr - 3, lc);
        vs = lub_cross(cs, lr - 3, lc);
      } else {
        vm = lub_xy(f2, m[3], m[4], rh, lr - 3, lc - 3);
        vs = lub_xy(f2, s[3], s[4], rh, lr - 3, lc - 3);
      }
      lub_write(base, rstride, lr, lc, vm, vs, 0.0, vMB, vSS, vDR, vDRS);
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: diagonal blocks, one thread per particle.  Wall term + the self
// (11) part of every pair in the particle's block row:
//   [[A, [b x]], [[b x]^T, C]],  b_wall = (0,0,-f1 Yb),  b_pair = -f1 Y11B r_hat.
// ---------------------------------------------------------------------------
extern "C" __global__ void lub_diag(
    const int N, const double* __restrict__ pos,
    const int* __restrict__ bcol, const int* __restrict__ bptr, const int* __restrict__ bdiag,
    const double a, const double Lx, const double Ly, const double Lz,
    const double debye_cut, const double f0, const double f1, const double f2,
    const double sF, const double sT,
    double* __restrict__ vMB, double* __restrict__ vSS,
    double* __restrict__ vDR, double* __restrict__ vDRS)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  const int start = bptr[i], end = bptr[i+1];
  const long long rstride = 6LL * (end - start);
  const long long base = 36LL * start + 6LL * (bdiag[i] - start);

  // ---- wall part (WallResistMatrixMB and WallResistMatrixMB + WallDeltaR) ----
  double rpy[5], dw[5];
  lub_wall_scalars(pos[3*i + 2] / a, debye_cut, rpy, dw);
  // packed {xx,yy,zz,xy,xz,yz}; Xa=0 Ya=1 Yb=2 Xc=3 Yc=4
  double mA[6] = {f0*rpy[1], f0*rpy[1], f0*rpy[0], 0.0, 0.0, 0.0};
  double mC[6] = {f2*rpy[4], f2*rpy[4], f2*rpy[3], 0.0, 0.0, 0.0};
  double mb[3] = {0.0, 0.0, -(f1*rpy[2])};
  double sA[6] = {f0*rpy[1] + f0*dw[1], f0*rpy[1] + f0*dw[1], f0*rpy[0] + f0*dw[0], 0.0, 0.0, 0.0};
  double sC[6] = {f2*rpy[4] + f2*dw[4], f2*rpy[4] + f2*dw[4], f2*rpy[3] + f2*dw[3], 0.0, 0.0, 0.0};
  double sb[3] = {0.0, 0.0, -(f1*rpy[2] + f1*dw[2])};

  // ---- pair self terms ----
  for (int bb = start; bb < end; ++bb) {
    const int k = bcol[bb];
    if (k == i) continue;
    double d[3];
    lub_separation(pos, i, k, a, Lx, Ly, Lz, d);
    const double r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    if (!(r < LUB_PAIR_CUTOFF)) continue;
    const double rh[3] = {d[0]/r, d[1]/r, d[2]/r};
    const double eps = fmax(r - 2.0, debye_cut);
    double m[5], s[5];
    lub_scalars_self<false>(eps, m);
    lub_scalars_self<true>(eps, s);
#pragma unroll
    for (int p = 0; p < 3; ++p) {
#pragma unroll
      for (int q = p; q < 3; ++q) {
        const int t = lub_sym(p, q);
        mA[t] += lub_xy(f0, m[0], m[1], rh, p, q);
        sA[t] += lub_xy(f0, s[0], s[1], rh, p, q);
        mC[t] += lub_xy(f2, m[3], m[4], rh, p, q);
        sC[t] += lub_xy(f2, s[3], s[4], rh, p, q);
      }
      // B11 = -f1*(Y11B*vortMat)  ->  b = -f1*(Y11B*r_hat)
      mb[p] += -f1*(m[2]*rh[p]);
      sb[p] += -f1*(s[2]*rh[p]);
    }
  }

#pragma unroll
  for (int lr = 0; lr < 6; ++lr) {
#pragma unroll
    for (int lc = 0; lc < 6; ++lc) {
      double vm, vs;
      if (lr < 3 && lc < 3) {
        vm = mA[lub_sym(lr, lc)];          vs = sA[lub_sym(lr, lc)];
      } else if (lr < 3) {
        vm = lub_cross(mb, lr, lc - 3);    vs = lub_cross(sb, lr, lc - 3);
      } else if (lc < 3) {                 // lower-left = transpose of upper-right
        vm = lub_cross(mb, lc, lr - 3);    vs = lub_cross(sb, lc, lr - 3);
      } else {
        vm = mC[lub_sym(lr-3, lc-3)];      vs = sC[lub_sym(lr-3, lc-3)];
      }
      const double shift = (lr == lc) ? (lr < 3 ? sF : sT) : 0.0;
      lub_write(base, rstride, lr, lc, vm, vs, shift, vMB, vSS, vDR, vDRS);
    }
  }
}

// ---------------------------------------------------------------------------
// Pattern kernels (run only when the block pattern changes).
// ---------------------------------------------------------------------------
extern "C" __global__ void lub_csr_indices(
    const int nb, const int* __restrict__ brow, const int* __restrict__ bcol,
    const int* __restrict__ bptr, int* __restrict__ indices)
{
  const int b = blockDim.x * blockIdx.x + threadIdx.x;
  if (b >= nb) return;
  const int i = brow[b];
  const int start = bptr[i];
  const long long rstride = 6LL * (bptr[i+1] - start);
  const long long base = 36LL * start + 6LL * (b - start);
  const int c0 = 6 * bcol[b];
  for (int lr = 0; lr < 6; ++lr)
    for (int lc = 0; lc < 6; ++lc)
      indices[base + lr*rstride + lc] = c0 + lc;
}

extern "C" __global__ void lub_csr_indptr(
    const int N, const int* __restrict__ bptr, int* __restrict__ indptr)
{
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row > 6*N) return;
  if (row == 6*N) { indptr[row] = 36 * bptr[N]; return; }
  const int i = row / 6, lr = row - 6*i;
  indptr[row] = 36 * bptr[i] + lr * 6 * (bptr[i+1] - bptr[i]);
}

// ---------------------------------------------------------------------------
// Brownian increments with covariance Delta_R + diag(shift), pair by pair.
//
// Delta_R is a sum of pair blocks (12x12, on the dofs of particles i < k) and
// wall blocks (6x6, one per particle), each positive semi-definite.  With
// P_p = B_p B_p^T and W_i + shift = C_i C_i^T (Cholesky),
//     y = sum_p E_p B_p xi_p + sum_i E_i C_i eta_i,    xi, eta ~ N(0, I),
// has covariance sum_p E_p P_p E_p^T + sum_i E_i (W_i + shift) E_i^T
// = Delta_R + diag(shift) exactly, with no global factorisation.
//
// lub_noise_pairs : one thread per pair, ypair[12p + r] = (B_p xi_p)_r
// lub_noise_gather: one thread per particle, out[6i + r] = (C_i eta_i)_r plus
//                   the particle's halves of its pairs' ypair, summed in block
//                   order (deterministic, no atomics)
// The Cholesky factorisations are "semi-definite": a pivot <= tol*max|diag|
// drops its column; a pivot < -tol*max|diag| (block not PSD) also sets flag.
// ---------------------------------------------------------------------------
template <int n>
__device__ __forceinline__ int lub_chol_psd(double* Lp) {
  // Lp: packed lower triangle, entry (r, c) at r*(r+1)/2 + c; in-place factor
  double scale = 0.0;
  for (int r = 0; r < n; ++r) scale = fmax(scale, fabs(Lp[r*(r+1)/2 + r]));
  const double tol = 1e-13 * scale;
  int flag = 0;
  for (int j = 0; j < n; ++j) {
    double d = Lp[j*(j+1)/2 + j];
    for (int t = 0; t < j; ++t) d -= Lp[j*(j+1)/2 + t] * Lp[j*(j+1)/2 + t];
    if (d > tol) {
      const double ljj = sqrt(d);
      Lp[j*(j+1)/2 + j] = ljj;
      for (int r = j + 1; r < n; ++r) {
        double v = Lp[r*(r+1)/2 + j];
        for (int t = 0; t < j; ++t) v -= Lp[r*(r+1)/2 + t] * Lp[j*(j+1)/2 + t];
        Lp[r*(r+1)/2 + j] = v / ljj;
      }
    } else {
      if (d < -tol) flag = 1;
      for (int r = j; r < n; ++r) Lp[r*(r+1)/2 + j] = 0.0;
    }
  }
  return flag;
}

extern "C" __global__ void lub_noise_pairs(
    const int M, const double* __restrict__ pos, const int* __restrict__ pairs,
    const double a, const double Lx, const double Ly, const double Lz,
    const double debye_cut, const double f0, const double f1, const double f2,
    const double* __restrict__ xi, double* __restrict__ ypair, int* __restrict__ flag)
{
  const int p = blockDim.x * blockIdx.x + threadIdx.x;
  if (p >= M) return;
  const int i = pairs[2*p], k = pairs[2*p + 1];
  double d[3];
  lub_separation(pos, i, k, a, Lx, Ly, Lz, d);
  const double r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
  if (!(r < LUB_PAIR_CUTOFF)) {        // buffered pattern: no contribution
    for (int t = 0; t < 12; ++t) ypair[12*p + t] = 0.0;
    flag[p] = 0;
    return;
  }
  const double rh[3] = {d[0]/r, d[1]/r, d[2]/r};
  const double eps = fmax(r - 2.0, debye_cut);
  double m1[5], s1[5], m2[5], s2[5];
  lub_scalars_self<false>(eps, m1);
  lub_scalars_self<true>(eps, s1);
  lub_scalars_pair<false>(eps, m2);
  lub_scalars_pair<true>(eps, s2);
  // Delta_R parts (R_Sup - R_MB) of this pair, as in lub_diag / lub_offdiag:
  //   self block of i: [[A11, [b x]], [[b x]^T, C11]],  b = -f1 Y11B r_hat
  //   self block of k: the same with b -> -b (r_hat -> -r_hat)
  //   coupling (i,k) : [[A12, [c x]], [[c x], C12]],   c = f1 Y12B r_hat
  double b[3], c[3];
  for (int q = 0; q < 3; ++q) {
    b[q] = -f1*(s1[2]*rh[q]) - (-f1*(m1[2]*rh[q]));
    c[q] = f1*(s2[2]*rh[q]) - f1*(m2[2]*rh[q]);
  }
  double S[6][6], X[6][6];               // self block of i, coupling block (i,k)
  for (int u = 0; u < 3; ++u) {
    for (int v = 0; v < 3; ++v) {
      S[u][v]     = lub_xy(f0, s1[0], s1[1], rh, u, v) - lub_xy(f0, m1[0], m1[1], rh, u, v);
      S[3+u][3+v] = lub_xy(f2, s1[3], s1[4], rh, u, v) - lub_xy(f2, m1[3], m1[4], rh, u, v);
      S[u][3+v]   = lub_cross(b, u, v);
      S[3+u][v]   = lub_cross(b, v, u);
      X[u][v]     = lub_xy(f0, s2[0], s2[1], rh, u, v) - lub_xy(f0, m2[0], m2[1], rh, u, v);
      X[3+u][3+v] = lub_xy(f2, s2[3], s2[4], rh, u, v) - lub_xy(f2, m2[3], m2[4], rh, u, v);
      X[u][3+v]   = lub_cross(c, u, v);
      X[3+u][v]   = lub_cross(c, u, v);
    }
  }
  // packed lower triangle of [[S_i, X], [X^T, S_k]];  S_k = S_i with b -> -b
  double Lp[78];
  for (int rr = 0; rr < 12; ++rr) {
    for (int cc = 0; cc <= rr; ++cc) {
      double v;
      if (rr < 6)       v = S[rr][cc];                        // S_i
      else if (cc < 6)  v = X[cc][rr - 6];                    // X^T
      else {                                                  // S_k
        const int u = rr - 6, w = cc - 6;
        v = ((u < 3) != (w < 3)) ? -S[u][w] : S[u][w];
      }
      Lp[rr*(rr+1)/2 + cc] = v;
    }
  }
  flag[p] = lub_chol_psd<12>(Lp);
  for (int rr = 0; rr < 12; ++rr) {
    double y = 0.0;
    for (int cc = 0; cc <= rr; ++cc) y += Lp[rr*(rr+1)/2 + cc] * xi[12*p + cc];
    ypair[12*p + rr] = y;
  }
}

extern "C" __global__ void lub_noise_gather(
    const int N, const double* __restrict__ pos,
    const int* __restrict__ bcol, const int* __restrict__ bptr,
    const int* __restrict__ bpair, const int* __restrict__ bside,
    const double a, const double debye_cut, const double f0, const double f1, const double f2,
    const double sF, const double sT,
    const double* __restrict__ eta, const double* __restrict__ ypair,
    double* __restrict__ out, int* __restrict__ flag)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  // Delta_R wall block (lub_diag: (RPY + DR) - RPY = DR terms) + diag(shift)
  double rpy[5], dw[5];
  lub_wall_scalars(pos[3*i + 2] / a, debye_cut, rpy, dw);
  const double Ad[3] = {f0*dw[1], f0*dw[1], f0*dw[0]};
  const double Cd[3] = {f2*dw[4], f2*dw[4], f2*dw[3]};
  const double bw[3] = {0.0, 0.0, -(f1*dw[2])};
  double Lp[21];
  for (int rr = 0; rr < 6; ++rr) {
    for (int cc = 0; cc <= rr; ++cc) {
      double v;
      if (rr < 3)       v = (rr == cc) ? Ad[rr] + sF : 0.0;
      else if (cc < 3)  v = lub_cross(bw, cc, rr - 3);        // [b x]^T
      else              v = (rr == cc) ? Cd[rr - 3] + sT : 0.0;
      Lp[rr*(rr+1)/2 + cc] = v;
    }
  }
  flag[i] = lub_chol_psd<6>(Lp);
  double y[6];
  for (int rr = 0; rr < 6; ++rr) {
    double v = 0.0;
    for (int cc = 0; cc <= rr; ++cc) v += Lp[rr*(rr+1)/2 + cc] * eta[6*i + cc];
    y[rr] = v;
  }
  for (int bb = bptr[i]; bb < bptr[i+1]; ++bb) {
    const int p = bpair[bb];
    if (p < 0) continue;                 // diagonal block
    const double* yp = ypair + 12*p + 6*bside[bb];
    for (int rr = 0; rr < 6; ++rr) y[rr] += yp[rr];
  }
  for (int rr = 0; rr < 6; ++rr) out[6*i + rr] = y[rr];
}
"""

_PAIR_CUTOFF = 4.5      # fits are only valid to 4.5a -- do not change


def _default_xp():
    import cupy
    return cupy


def _default_sparse():
    import cupyx.scipy.sparse
    return cupyx.scipy.sparse


def _copy_h2d(dst, src):
    """Copy a host array into an existing (contiguous) device buffer view."""
    src = np.ascontiguousarray(src, dtype=dst.dtype)
    if hasattr(dst, 'set'):          # cupy.ndarray
        dst.set(src)
    else:                            # numpy (CPU validation backend)
        dst[...] = src


# =============================================================================
# Block pattern (host side)
# =============================================================================
class BlockPattern:
    """
    6x6 block sparsity pattern of the lubrication matrices.

    Parameters
    ----------
    N     : number of particles
    pairs : (M, 2) integer array of unique pairs (i, k) with i != k.  Order and
            orientation do not matter; self.pairs holds them as sorted (lo, hi).
    version : integer tag; anything that caches a factorisation of a matrix
            with this pattern (cuDSS plan, CHOLMOD symbolic) keys on it.

    Attributes (all numpy int32)
    ----------------------------
    bptr  (N+1,)  block-row pointer
    bcol  (nb,)   block column of each block (sorted inside each row)
    brow  (nb,)   block row of each block
    bdiag (N,)    index of the diagonal block of each row
    bpair (nb,)   pair index (into self.pairs) of each block, -1 for diagonal blocks
    bside (nb,)   0 for block (lo, hi), 1 for block (hi, lo), -1 for diagonal blocks
    nb = N + 2M blocks, nnz = 36 nb scalar entries.
    """

    def __init__(self, N, pairs, version=0):
        N = int(N)
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            lo = np.minimum(pairs[:, 0], pairs[:, 1])
            hi = np.maximum(pairs[:, 0], pairs[:, 1])
            if np.any(lo == hi) or lo.min() < 0 or hi.max() >= N:
                raise ValueError('invalid pair list')
            srt = np.lexsort((hi, lo))
            lo, hi = lo[srt], hi[srt]
        else:
            lo = hi = np.empty(0, dtype=np.int64)
        diag = np.arange(N, dtype=np.int64)
        rows = np.concatenate([lo, hi, diag])
        cols = np.concatenate([hi, lo, diag])
        key = rows * N + cols
        order = np.argsort(key, kind='stable')
        ks = key[order]
        if ks.size > 1 and not np.all(ks[1:] > ks[:-1]):
            raise ValueError('duplicate pairs in pair list')
        self.N = N
        self.n_pairs = int(lo.size)
        self.nb = int(rows.size)
        self.nnz = 36 * self.nb
        self.n_dof = 6 * N
        if self.nnz >= 2**31 - 1:
            raise ValueError('pattern too large for int32 CSR indices')
        self.brow = rows[order].astype(np.int32)
        self.bcol = cols[order].astype(np.int32)
        bptr = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=N), out=bptr[1:])
        self.bptr = bptr.astype(np.int32)
        self.bdiag = np.flatnonzero(self.brow == self.bcol).astype(np.int32)
        self.pairs = np.stack([lo, hi], axis=1).astype(np.int32) if lo.size \
            else np.empty((0, 2), dtype=np.int32)
        # sorted position of every unsorted block: rows = [lo, hi, diag], so the
        # first M are the (lo, hi) blocks of the pairs and the next M the (hi, lo)
        where = np.empty(self.nb, dtype=np.int64)
        where[order] = np.arange(self.nb)
        M = lo.size
        self.bpair = np.full(self.nb, -1, dtype=np.int32)
        self.bside = np.full(self.nb, -1, dtype=np.int32)
        if M:
            self.bpair[where[:M]] = self.bpair[where[M:2 * M]] = np.arange(M, dtype=np.int32)
            self.bside[where[:M]], self.bside[where[M:2 * M]] = 0, 1
        self.version = int(version)

    def scalar_csr_host(self):
        """(indptr, indices) of the scalar CSR on the host (reference/tests)."""
        N, deg = self.N, np.diff(self.bptr).astype(np.int64)
        lr = np.arange(6)
        indptr = np.empty(6 * N + 1, dtype=np.int64)
        indptr[:-1] = (36 * self.bptr[:-1, None].astype(np.int64)
                       + lr[None, :] * 6 * deg[:, None]).ravel()
        indptr[-1] = self.nnz
        indices = np.empty(self.nnz, dtype=np.int64)
        for i in range(N):
            cols = (6 * self.bcol[self.bptr[i]:self.bptr[i + 1]][:, None]
                    + lr[None, :]).ravel()
            start = 36 * self.bptr[i]
            for r in range(6):
                s = start + r * 6 * deg[i]
                indices[s:s + cols.size] = cols
        return indptr.astype(np.int32), indices.astype(np.int32)


# =============================================================================
# Assembler (device side)
# =============================================================================
class LubricationAssembler:
    """
    Persistent device buffers + fused kernels for the lubrication matrices.

    Parameters
    ----------
    debye_cut : minimum gap (units of a) used in the fits
    a, eta    : particle radius and viscosity
    small_F, small_T : diagonal shift added to R_SupS and Delta_RS
                (6 pi eta a tol and 8 pi eta a^3 tol in the solver)
    headroom  : value/index buffers are allocated with this much spare
                capacity so a slightly larger pattern does not reallocate.

    After set_pattern(p):  R_MB, R_SupS, Delta_R, Delta_RS are CSR matrices
    (cupyx) sharing one indptr/indices; assemble(pos, L) refreshes their
    values in place.
    """

    BLOCK = 128

    def __init__(self, debye_cut, a, eta, small_F=0.0, small_T=0.0, *,
                 xp=None, sparse=None, headroom=1.25):
        self.xp = xp if xp is not None else _default_xp()
        self.sparse = sparse if sparse is not None else _default_sparse()
        self.debye_cut = float(debye_cut)
        self.a = float(a)
        self.eta = float(eta)
        # same expressions (and evaluation order) as lubrication.cpp
        self.f0 = 6.0 * np.pi * eta * a
        self.f1 = 6.0 * np.pi * eta * a * a
        self.f2 = 6.0 * np.pi * eta * a * a * a
        self.small_F = float(small_F)
        self.small_T = float(small_T)
        self.headroom = float(headroom)
        self.pattern = None
        self._version = 0
        self._mod = None
        self._cap_nnz = self._cap_nb = self._cap_n = -1
        self.R_MB = self.R_SupS = self.Delta_R = self.Delta_RS = None
        self.n_pattern_builds = 0
        self.n_assemblies = 0

    # ---- kernel plumbing (overridden by the CPU validation backend) ---------
    def _kernel(self, name):
        if self._mod is None:
            self._mod = self.xp.RawModule(code=KERNEL_SOURCE)
        return self._mod.get_function(name)

    def _launch(self, name, n_threads, args):
        if n_threads <= 0:
            return
        grid = ((n_threads + self.BLOCK - 1) // self.BLOCK,)
        self._kernel(name)(grid, (self.BLOCK,), args)

    # ---- buffers -----------------------------------------------------------
    def _ensure_capacity(self, nnz, nb, N):
        # Grow-only with headroom, but also shrink when less than half is used
        # (keeps memory bounded, and scipy -- used by the CPU tests -- copies
        # views of much larger arrays instead of aliasing them).
        xp, h = self.xp, self.headroom
        if nnz > self._cap_nnz or 2 * nnz < self._cap_nnz:
            cap = int(nnz * h) + 36
            self._vals = [xp.empty(cap, dtype=xp.float64) for _ in range(4)]
            self._indices = xp.empty(cap, dtype=xp.int32)
            self._cap_nnz = cap
        if nb > self._cap_nb:
            cap = int(nb * h) + 1
            self._d_brow = xp.empty(cap, dtype=xp.int32)
            self._d_bcol = xp.empty(cap, dtype=xp.int32)
            self._cap_nb = cap
        if N > self._cap_n:
            self._d_bptr = xp.empty(N + 1, dtype=xp.int32)
            self._d_bdiag = xp.empty(N, dtype=xp.int32)
            self._indptr = xp.empty(6 * N + 1, dtype=xp.int32)
            self._cap_n = N

    @staticmethod
    def _same_buffer(x, y):
        if hasattr(x, 'data') and hasattr(x.data, 'ptr'):          # cupy
            return x.data.ptr == y.data.ptr and x.size == y.size
        return np.shares_memory(x, y) and x.size == y.size        # numpy

    def _make_csr(self, data, indices, indptr, n):
        M = self.sparse.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
        # The matrices must be views of the persistent buffers, otherwise
        # assemble() would not update them.  Fail loudly if a sparse library
        # version decides to copy.
        if not (self._same_buffer(M.data, data) and self._same_buffer(M.indices, indices)
                and self._same_buffer(M.indptr, indptr)):
            raise RuntimeError('sparse matrix constructor copied the CSR buffers; '
                               'in-place assembly would be lost')
        try:                          # skip the lazy canonical-format check
            M.has_canonical_format = True
        except (AttributeError, TypeError):
            pass
        return M

    def set_pattern(self, pattern):
        """Upload a BlockPattern and build the scalar CSR structure on device."""
        p = pattern
        self._ensure_capacity(p.nnz, p.nb, p.N)
        nb, N, nnz, n = p.nb, p.N, p.nnz, p.n_dof
        _copy_h2d(self._d_brow[:nb], p.brow)
        _copy_h2d(self._d_bcol[:nb], p.bcol)
        _copy_h2d(self._d_bptr[:N + 1], p.bptr)
        _copy_h2d(self._d_bdiag[:N], p.bdiag)
        self._d_pairs = self.xp.asarray(np.ascontiguousarray(p.pairs.reshape(-1)))
        self._d_bpair = self.xp.asarray(p.bpair)
        self._d_bside = self.xp.asarray(p.bside)
        ind, ptr = self._indices[:nnz], self._indptr[:n + 1]
        self._launch('lub_csr_indices', nb,
                     (np.int32(nb), self._d_brow, self._d_bcol, self._d_bptr, ind))
        self._launch('lub_csr_indptr', n + 1, (np.int32(N), self._d_bptr, ptr))
        v = self._vals
        self.vMB, self.vSS, self.vDR, self.vDRS = (v[0][:nnz], v[1][:nnz],
                                                   v[2][:nnz], v[3][:nnz])
        self.indices, self.indptr = ind, ptr
        self.R_MB = self._make_csr(self.vMB, ind, ptr, n)
        self.R_SupS = self._make_csr(self.vSS, ind, ptr, n)
        self.Delta_R = self._make_csr(self.vDR, ind, ptr, n)
        self.Delta_RS = self._make_csr(self.vDRS, ind, ptr, n)
        self.pattern = p
        self.n_pattern_builds += 1

    def set_pairs(self, N, pairs):
        """
        Use the block pattern of a pair list (any order or orientation).  If
        the list equals the current one nothing happens, so pattern_version
        (the key for cached factorisations) only changes when the sparsity
        pattern does.  Returns True when a new pattern was built.
        """
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            lo = np.minimum(pairs[:, 0], pairs[:, 1])
            hi = np.maximum(pairs[:, 0], pairs[:, 1])
            srt = np.lexsort((hi, lo))
            pairs = np.stack([lo[srt], hi[srt]], axis=1)
        p = self.pattern
        if p is not None and p.N == N and p.pairs.shape == pairs.shape \
                and np.array_equal(p.pairs, pairs):
            return False
        self._version += 1
        self.set_pattern(BlockPattern(N, pairs, version=self._version))
        return True

    @property
    def pattern_version(self):
        return None if self.pattern is None else self.pattern.version

    def assemble(self, pos, periodic_length, length_unit=None):
        """
        Refresh all four matrices for positions ``pos``.

        pos : device (N, 3) float64, C-contiguous, wrapped into the periodic box.
        periodic_length : 3 floats (<= 0: open direction), same units as pos.
        length_unit : scale that makes positions dimensionless (default a).
            Pass 1.0 when pos is already in units of a.
        """
        p = self.pattern
        if p is None:
            raise RuntimeError('set_pattern() must be called first')
        if pos.shape != (p.N, 3) or pos.dtype != np.float64 or not pos.flags.c_contiguous:
            raise ValueError('pos must be a C-contiguous (N,3) float64 array')
        unit = self.a if length_unit is None else float(length_unit)
        L = [float(x) if float(x) > 0.0 else 0.0 for x in periodic_length]
        geo = (np.float64(unit), np.float64(L[0]), np.float64(L[1]), np.float64(L[2]),
               np.float64(self.debye_cut), np.float64(self.f0),
               np.float64(self.f1), np.float64(self.f2))
        outs = (self.vMB, self.vSS, self.vDR, self.vDRS)
        self.pos, self.periodic_length = pos, periodic_length     # last assembled configuration
        self._launch('lub_offdiag', p.nb,
                     (np.int32(p.nb), pos, self._d_brow, self._d_bcol, self._d_bptr)
                     + geo + outs)
        self._launch('lub_diag', p.N,
                     (np.int32(p.N), pos, self._d_bcol, self._d_bptr, self._d_bdiag)
                     + geo + (np.float64(self.small_F), np.float64(self.small_T)) + outs)
        self.n_assemblies += 1

    def brownian(self, pos, periodic_length, normals=None, length_unit=None):
        """
        Gaussian vector with covariance Delta_R + diag(small) (= Delta_RS),
        drawn pair by pair: Delta_R is a sum of positive semi-definite 12x12
        pair blocks and 6x6 wall blocks, so with the Cholesky factors B_p and
        C_i of the pair blocks and of (wall block + small),
            y = sum_p B_p xi_p + sum_i C_i eta_i
        has exactly that covariance.  No global factorisation.

        pos, periodic_length, length_unit : as in assemble() (same positions).
        normals : 12*n_pairs + 6*N standard normals (xi_0, xi_1, ..., eta_0, ...)
                  in the array module of the assembler; drawn if None.
        Returns the (6N,) vector.  noise_clipped counts blocks that were not
        positive semi-definite in this draw (expected 0).
        """
        p = self.pattern
        if p is None:
            raise RuntimeError('set_pattern() must be called first')
        if pos.shape != (p.N, 3) or pos.dtype != np.float64 or not pos.flags.c_contiguous:
            raise ValueError('pos must be a C-contiguous (N,3) float64 array')
        xp, M, N = self.xp, p.n_pairs, p.N
        if normals is None:
            normals = xp.random.standard_normal(12 * M + 6 * N)
        normals = xp.ascontiguousarray(normals, dtype=xp.float64).reshape(-1)
        if normals.size != 12 * M + 6 * N:
            raise ValueError(f'expected {12 * M + 6 * N} normals, got {normals.size}')
        unit = self.a if length_unit is None else float(length_unit)
        L = [float(x) if float(x) > 0.0 else 0.0 for x in periodic_length]
        f8 = np.float64
        ypair = xp.zeros(12 * max(M, 1), dtype=xp.float64)
        flags = xp.zeros(M + N, dtype=xp.int32)
        out = xp.empty(6 * N, dtype=xp.float64)
        self._launch('lub_noise_pairs', M,
                     (np.int32(M), pos, self._d_pairs, f8(unit), f8(L[0]), f8(L[1]), f8(L[2]),
                      f8(self.debye_cut), f8(self.f0), f8(self.f1), f8(self.f2),
                      normals, ypair, flags))
        self._launch('lub_noise_gather', N,
                     (np.int32(N), pos, self._d_bcol, self._d_bptr, self._d_bpair, self._d_bside,
                      f8(unit), f8(self.debye_cut), f8(self.f0), f8(self.f1), f8(self.f2),
                      f8(self.small_F), f8(self.small_T), normals[12 * M:], ypair, out,
                      flags[M:]))
        self._noise_flags = flags
        return out

    def pair_gaps(self):
        """Host array: surface gap / a of every pair of the pattern (self.pattern.pairs)
        at the positions of the last assemble() call (minimum image)."""
        p = self.pattern
        pos = np.asarray(self.pos.get() if hasattr(self.pos, 'get') else self.pos)
        d = pos[p.pairs[:, 1]] - pos[p.pairs[:, 0]]
        for l, Ll in enumerate(self.periodic_length):
            if float(Ll) > 0.0:
                d[:, l] -= np.round(d[:, l] / float(Ll)) * float(Ll)
        return np.sqrt(np.sum(d * d, axis=1)) / self.a - 2.0

    @property
    def noise_clipped(self):
        """Blocks that were not positive semi-definite in the last brownian() call."""
        f = getattr(self, '_noise_flags', None)
        return 0 if f is None else int(f.sum())

    @property
    def R_Sup(self):
        """R_Sup without the diagonal shift (built on demand, for inspection)."""
        n = self.pattern.n_dof
        s = self.xp.tile(self.xp.asarray([self.small_F] * 3 + [self.small_T] * 3,
                                         dtype=self.xp.float64), self.pattern.N)
        return self.R_SupS - self.sparse.diags(s, 0, shape=(n, n), format='csr')
