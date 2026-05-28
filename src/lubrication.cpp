#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> Triplet;

using nb_array_d = nb::ndarray<double, nb::ndim<1>, nb::c_contig>;
using nb_array_i = nb::ndarray<int,    nb::ndim<1>, nb::c_contig>;

// =============================================================================
// Rational fit coefficient layout (shared by Sup and MB):
//   cf[0]        = crossover epsilon (Sup) or d_cut (MB, no fallback)
//   cf[1..6]     = p0..p5  (numerator polynomial)
//   cf[7..11]    = c0..c4  (denominator sqrt coefficients, Q = 1 + sum c_i^2 * eps^(i+1))
//
// Fit forms:
//   non-singular: scalar = P(eps) / Q(eps)
//   singular:     scalar = S_sing + P(eps) / (Q(eps) * eps)
//     X11A: S_sing = +0.25/eps,  X12A: S_sing = -0.25/eps
//
// Sign convention (consistent with Python extraction via R[1,5] and R[1,11]):
//   Y11B: fit stores +Y11B  -> use as-is in AssembleResistMatrix
//   Y12B: fit stores -Y12B  -> negate before passing to AssembleResistMatrix
// =============================================================================

// Shared helper: evaluate P/Q given pre-computed power arrays ep[6] and eq[5]
// cf layout: {crossover_or_dcut, p0..p5, c0..c4}
static inline double eval_PQ(const double* cf,
                              const double ep[6], const double eq[5]) {
  double P = cf[1]*ep[0] + cf[2]*ep[1] + cf[3]*ep[2]
           + cf[4]*ep[3] + cf[5]*ep[4] + cf[6]*ep[5];
  double Q = 1.0 + cf[7]*cf[7]*eq[0] + cf[8]*cf[8]*eq[1]
           + cf[9]*cf[9]*eq[2] + cf[10]*cf[10]*eq[3] + cf[11]*cf[11]*eq[4];
  return P / Q;
}

class Lubrication {
private:
  Matrix WallDeltaR        (double r_norm, double mob_factor[3]);
  Matrix WallResistMatrixMB(double r_norm, double mob_factor[3]);
  Matrix ResistPairSup(double r_norm, double mob_factor[3], Vector3 r_hat);
  Matrix ResistPairMB (double r_norm, double mob_factor[3], Vector3 r_hat);
  void   AssembleResistMatrix(Matrix &R, double mob_factor[3], Vector3 r_hat,
                              double X11A, double Y11A, double Y11B,
                              double X11C, double Y11C, double X12A,
                              double Y12A, double Y12B, double X12C,
                              double Y12C);

public:
  SpMat ResistCSC(nb::list r_vectors, nb::list n_list, double a, double eta,
                  nb_array_d periodic_length, bool Sup_if_true);
  std::pair<SpMat, SpMat> ResistCSC_both(nb::list r_vectors, nb::list n_list,
                                          double a, double eta,
                                          nb_array_d periodic_length);
  double debye_cut;
  // Physical cutoff for pair rational fits — fits are only valid to 4.5a.
  static constexpr double PAIR_CUTOFF = 4.5;
  Lubrication(double d_cut);
};

// =============================================================================
// Constructor
// =============================================================================
Lubrication::Lubrication(double d_cut) {
  debye_cut = d_cut;
  // Wall scalars computed entirely from rational fits — no file I/O needed.
}

// =============================================================================
// Wall rational fit helpers
//
// RPY fits:     u = 1/(1 + h/0.5),   P/Q, raw q_i
// Delta_R fits: u = 1/(1 + h/0.5),   P/Q, raw q_i
//
// Chimera per scalar:
//   eps < asym_cut  -> asym_wall - RPY_fit   (near contact)
//   asym_cut <= eps <= rpy_cut -> Delta_R rational fit
//   eps > rpy_cut   -> Delta_R = 0  (Sup = RPY)
// Sup scalar = RPY_fit + Delta_R
// =============================================================================

static inline double eval_wall_rat(double h, double h_scale,
                                   int p_min, int n_p, int n_q,
                                   const double* p, const double* q) {
  const double u = 1.0 / (1.0 + h / h_scale);
  double P = 0.0, upow = std::pow(u, p_min);
  for (int i = 0; i < n_p; ++i) { P += p[i] * upow; upow *= u; }
  double Q = 1.0; upow = u;
  for (int i = 0; i < n_q; ++i) { Q += q[i] * upow; upow *= u; }
  return P / Q;
}

// RPY rational fits (h_scale = 0.5)
static double rpy_wall(int scalar, double h) {
  static const double H = 0.5;
  static const double Xa_p[]={ 2.249994311366312e+00,-9.302942542409326e+00,
    1.018898977062156e+01,-1.461983146752135e-01};
  static const double Xa_q[]={-7.384833913349599e+00, 1.974855818648951e+01,
   -2.040618954066672e+01, 4.353438654549962e+00};
  static const double Ya_p[]={ 1.124980798925478e+00,-5.161218663062001e+00,
    7.210326948819593e+00,-2.621851150554913e+00};
  static const double Ya_q[]={-6.713754414307387e+00, 1.706822775135723e+01,
   -1.933894500702774e+01, 7.413436778017460e+00};
  static const double Yb_p[]={-2.019389402582813e+00, 7.475392505257506e+00,
   -3.877269126222068e+00,-1.028617914833345e+00};
  static const double Yb_q[]={-8.683501976306239e+00, 2.835163248843032e+01,
   -4.234435891678831e+01, 2.531363165467517e+01};
  static const double Xc_p[]={ 1.333114620704834e+00,-4.215726080748013e+00,
   -5.775019233806103e-02, 1.969213190855149e-01};
  static const double Xc_q[]={-6.164932696556416e+00, 1.247407137449759e+01,
   -1.135905628832895e+01, 6.141510604219781e+00};
  static const double Yc_p[]={ 3.339371808444304e+00,-1.406703500455225e+01,
    8.727632354389423e+00, 9.801957547519073e+00};
  static const double Yc_q[]={-7.179264725936619e+00, 1.782208382807917e+01,
   -1.884878181254139e+01, 8.912717591979181e+00};
  switch(scalar) {
    case 0: return eval_wall_rat(h,H,1,4,4,Xa_p,Xa_q); // Xa_corr = Xa-1
    case 1: return eval_wall_rat(h,H,1,4,4,Ya_p,Ya_q); // Ya_corr = Ya-1
    case 2: return eval_wall_rat(h,H,4,4,4,Yb_p,Yb_q); // Yb
    case 3: return eval_wall_rat(h,H,3,4,4,Xc_p,Xc_q); // XcPlus = Xc-4/3
    default:return eval_wall_rat(h,H,3,4,4,Yc_p,Yc_q); // YcPlus = Yc-4/3
  }
}

// Delta_R chimera (h_scale = 0.5)
static double delta_R_wall(int scalar, double eps, double h) {
  static const double H = 0.5;
  const double le = std::log(eps);
  // asymptotic - RPY_fit
  auto asym_minus_rpy = [&](int s) -> double {
    double asym;
    switch(s) {
      case 0: asym = 1.0/eps - (1.0/5.0)*le + 0.971280 - 1.0; break;
      case 1: asym = -(8.0/15.0)*le + 0.9588 - 1.0; break;
      case 2: asym = (4.0/3.0)*((1.0/10.0)*le + 0.1895 - 0.029 - (0.4576-0.2)*eps); break;
      case 3: asym = (4.0/3.0)*(1.20206 - 3.0*(M_PI*M_PI/6.0-1.0)*eps) - 4.0/3.0; break;
      default:asym = (4.0/3.0)*(-(2.0/5.0)*le + 0.3817 + 1.4578*eps) - 4.0/3.0;
    }
    return asym - rpy_wall(s, h);
  };

  // Xa_corr: asym_cut=0.2055, rpy_cut=7.0, p_min=1, n_p=3, n_q=4
  static const double Xa_ac=2.0549e-01, Xa_rc=7.0;
  static const double Xa_p[]={ 3.390440744007662e-03,-1.292707770613473e-01,
    1.621030547158851e+00};
  static const double Xa_q[]={-9.934759618046485e+00, 3.852782844267631e+01,
   -6.919767486523838e+01, 4.812890822657162e+01};
  // Ya_corr: asym_cut=0.02912, rpy_cut=5.6, p_min=1, n_p=4, n_q=4
  static const double Ya_ac=2.9118e-02, Ya_rc=5.6;
  static const double Ya_p[]={-1.802661869524016e-03, 9.468660270547602e-03,
    3.218091451875633e-01,-9.812949264604398e-01};
  static const double Ya_q[]={-1.077793936835893e+01, 4.559152441171614e+01,
   -9.009502525501624e+01, 6.997760797588727e+01};
  // Yb: asym_cut=0.1, rpy_cut=3.4, p_min=4, n_p=5, n_q=4
  static const double Yb_ac=1.0e-01, Yb_rc=3.4;
  static const double Yb_p[]={-1.171720418904620e+00, 2.634024743369987e+01,
   -2.037770517911233e+02, 6.604148051813611e+02,-7.644275364225128e+02};
  static const double Yb_q[]={-1.458404137489715e+01, 7.817171888830218e+01,
   -1.820761238925333e+02, 1.554914960487163e+02};
  // XcPlus: asym_cut=0.0097, rpy_cut=0.4, p_min=3, n_p=4, n_q=3
  static const double Xc_ac=9.7e-03, Xc_rc=4.0e-01;
  static const double Xc_p[]={-6.769668477673338e-02, 4.251981921512744e-01,
   -4.761218699317399e-01,-5.662068770688917e-01};
  static const double Xc_q[]={-7.986015731409064e+00, 2.107903159501786e+01,
   -1.836099133069141e+01};
  // YcPlus: asym_cut=0.0456, rpy_cut=5.0, p_min=3, n_p=5, n_q=4
  static const double Yc_ac=4.56e-02, Yc_rc=5.0;
  static const double Yc_p[]={ 4.774546185054501e-02,-1.200098819024352e+00,
    1.143510786982436e+01,-4.754910568257404e+01, 7.229905586544142e+01};
  static const double Yc_q[]={-1.420388909734784e+01, 7.366905048843127e+01,
   -1.644525346135324e+02, 1.329770561654655e+02};

  switch(scalar) {
    case 0:
      if (eps < Xa_ac) return asym_minus_rpy(0);
      if (eps > Xa_rc) return 0.0;
      return eval_wall_rat(h,H,1,3,4,Xa_p,Xa_q);
    case 1:
      if (eps < Ya_ac) return asym_minus_rpy(1);
      if (eps > Ya_rc) return 0.0;
      return eval_wall_rat(h,H,1,4,4,Ya_p,Ya_q);
    case 2:
      if (eps < Yb_ac) return asym_minus_rpy(2);
      if (eps > Yb_rc) return 0.0;
      return eval_wall_rat(h,H,4,5,4,Yb_p,Yb_q);
    case 3:
      if (eps < Xc_ac) return asym_minus_rpy(3);
      if (eps > Xc_rc) return 0.0;
      return eval_wall_rat(h,H,3,4,3,Xc_p,Xc_q);
    default:
      if (eps < Yc_ac) return asym_minus_rpy(4);
      if (eps > Yc_rc) return 0.0;
      return eval_wall_rat(h,H,3,5,4,Yc_p,Yc_q);
  }
}

// =============================================================================
// Shared matrix assembly from 10 scalars
// =============================================================================
void Lubrication::AssembleResistMatrix(Matrix &R, double mob_factor[3],
                                       Vector3 r_hat,
                                       double X11A, double Y11A, double Y11B,
                                       double X11C, double Y11C, double X12A,
                                       double Y12A, double Y12B, double X12C,
                                       double Y12C) {
  Matrix3 squeezeMat = r_hat * r_hat.transpose();
  Matrix3 Eye        = Matrix3::Identity();
  Matrix3 shearMat   = Eye - squeezeMat;
  Matrix3 vortMat;
  vortMat << 0.0,       r_hat[2], -r_hat[1],
            -r_hat[2],  0.0,       r_hat[0],
             r_hat[1], -r_hat[0],  0.0;
  vortMat *= -1;

  R.block<3,3>(0,0) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
  R.block<3,3>(0,3) = -mob_factor[1]*(Y11B*vortMat);
  R.block<3,3>(0,6) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
  R.block<3,3>(0,9) = mob_factor[1]*(Y12B*vortMat);
  R.block<3,3>(3,0) = mob_factor[1]*(Y11B*vortMat);
  R.block<3,3>(3,3) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
  R.block<3,3>(3,6) = mob_factor[1]*(Y12B*vortMat);
  R.block<3,3>(3,9) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
  R.block<3,3>(6,0) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
  R.block<3,3>(6,3) = -mob_factor[1]*(Y12B*vortMat);
  R.block<3,3>(6,6) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
  R.block<3,3>(6,9) = mob_factor[1]*(Y11B*vortMat);
  R.block<3,3>(9,0) = -mob_factor[1]*(Y12B*vortMat);
  R.block<3,3>(9,3) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
  R.block<3,3>(9,6) = -mob_factor[1]*(Y11B*vortMat);
  R.block<3,3>(9,9) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
}

// =============================================================================
// Wall resistance matrices — rational fits (no table I/O)
// =============================================================================

// R_RPY wall correction matrix (6x6)
Matrix Lubrication::WallResistMatrixMB(double r_norm, double mob_factor[3]) {
  double epsilon = r_norm - 1.0;
  if (epsilon < debye_cut) { epsilon = debye_cut; r_norm = 1.0 + epsilon; }

  const double Xa_corr = rpy_wall(0, r_norm);
  const double Ya_corr = rpy_wall(1, r_norm);
  const double Yb      = rpy_wall(2, r_norm);
  const double XcPlus  = rpy_wall(3, r_norm);
  const double YcPlus  = rpy_wall(4, r_norm);

  Matrix R(6,6);
  R << mob_factor[0]*Ya_corr, 0, 0, 0,  mob_factor[1]*Yb, 0,
       0, mob_factor[0]*Ya_corr, 0, -mob_factor[1]*Yb, 0, 0,
       0, 0, mob_factor[0]*Xa_corr, 0, 0, 0,
       0, -mob_factor[1]*Yb, 0,  mob_factor[2]*YcPlus, 0, 0,
       mob_factor[1]*Yb, 0, 0, 0,  mob_factor[2]*YcPlus, 0,
       0, 0, 0, 0, 0,  mob_factor[2]*XcPlus;
  return R;
}

// Delta_R wall correction matrix (6x6)
// Each scalar is set to zero if epsilon > rpy_cut for that scalar.
Matrix Lubrication::WallDeltaR(double r_norm, double mob_factor[3]) {
  static const double rpy_cuts[5] = {7.0, 5.6, 3.4, 0.4, 5.0};  // Xa,Ya,Yb,Xc,Yc

  double epsilon = r_norm - 1.0;
  if (epsilon < debye_cut) { epsilon = debye_cut; r_norm = 1.0 + epsilon; }

  double dR[5];
  for (int s = 0; s < 5; ++s)
    dR[s] = (epsilon > rpy_cuts[s]) ? 0.0
                                    : delta_R_wall(s, epsilon, r_norm);

  Matrix R(6,6);
  R << mob_factor[0]*dR[1], 0, 0, 0,  mob_factor[1]*dR[2], 0,
       0, mob_factor[0]*dR[1], 0, -mob_factor[1]*dR[2], 0, 0,
       0, 0, mob_factor[0]*dR[0], 0, 0, 0,
       0, -mob_factor[1]*dR[2], 0,  mob_factor[2]*dR[4], 0, 0,
       mob_factor[1]*dR[2], 0, 0, 0,  mob_factor[2]*dR[4], 0,
       0, 0, 0, 0, 0,  mob_factor[2]*dR[3];
  return R;
}

// =============================================================================
// ResistPairSup: rational fit (Sup scalars) with AT asymptotic fallback.
//
// Coefficient layout: {crossover, p0..p5, c0..c4}
// below crossover -> AT asymptotic; above -> rational fit
// Sign note: Y12B fit stores -Y12B (extracted from R[1,11]/f1 = -Y12B),
//   so we negate it before passing to AssembleResistMatrix.
// =============================================================================
Matrix Lubrication::ResistPairSup(double r_norm, double mob_factor[3],
                                  Vector3 r_hat) {
  Matrix R(12, 12);
  double epsilon = r_norm - 2.0;
  if (epsilon < debye_cut) epsilon = debye_cut;

  static const double cf_X11A[12] = {
    7.006673e-02,
    -1.981052779593092e+01,  2.340828937668049e+03,  4.965834145654573e+05,
     1.139263177405481e+06, -3.107490582481417e+04,  6.123746273694200e+03,
    -5.465471058359071e+02, -1.069447402186549e+03, -6.581289099212643e-09,
    -9.673175257729874e-10, -6.988711115145478e-03 };
  static const double cf_X12A[12] = {
    2.457900e-02,
    -1.593939244959434e-02,  4.445665389795823e+00, -4.078726991721069e+02,
    -1.034351092030956e+04, -8.219652436471222e+03, -1.370173545248133e+05,
     2.488928114444593e-04, -1.258809718612856e+02,  1.407296176369766e-08,
    -4.892039651123681e+02,  3.407984060897670e+02 };
  static const double cf_Y11A[12] = {
    5.228588e-03,
     2.171207126379109e+00,  3.853002949507692e+02,  6.624352544471424e+03,
     1.484697318890232e+04,  7.047767136709095e+03,  6.540531100098950e+03,
    -1.511807925068774e+01,  7.012000660040704e+01, -1.193647614569199e+02,
    -8.041378517885596e+01, -8.122175958097344e+01 };
  static const double cf_Y12A[12] = {
    5.248093e-03,
    -1.461562640575963e+00, -2.507841615126929e+02, -4.331083375391881e+03,
    -9.737921752082497e+03, -4.058329204467756e+02,  3.606943711811880e+01,
    -1.596895122841455e+01,  8.109752269618301e+01, -1.638005603371911e+02,
     1.254673064262033e+02,  5.431104446475567e+00 };
  static const double cf_Y11B[12] = {
    5.313637e-03,
    -1.043087981132569e+00,  5.195160018422946e+03,  1.121528966727754e+06,
     1.355294010638291e+07, -3.532929933499587e+06,  4.458815880568250e+05,
     6.331300459521475e+01, -1.375961122074268e+03, -7.415395959620902e+03,
    -1.414565234165133e+04, -7.015955939280531e+03 };
  static const double cf_Y12B[12] = {   // fit stores -Y12B
    5.138518e-03,
     1.266855888751514e+00,  6.899463247856546e+02,  4.216236929666354e+04,
     2.147931207416082e+05, -2.876020054138726e+04,  1.086426843098747e+03,
     2.707637792335356e+01, -2.674955385301423e+02,  9.078532792700140e+02,
     9.825987715332953e+02, -5.482196655744529e-01 };
  static const double cf_X11C[12] = {
    5.113070e-03,
     1.397874023016620e+00,  1.345883636860944e+01,  2.434301005382227e+04,
     1.444198486149228e+05,  1.003697333447050e+05,  2.410941641426479e+05,
     3.067204057507511e+00,  1.320239652333805e+02,  3.292073502673707e+02,
    -2.743878779348838e+02, -4.252396254964442e+02 };
  static const double cf_X12C[12] = {
    5.050000e-03,
    -2.022446304463027e-01, -5.956596895307574e+01, -4.801712239322800e+02,
     2.384455230908337e+02, -5.757163499587050e+01,  5.798541137001251e+00,
     1.748413100133951e+01,  5.819390284035368e+01, -5.863429968285631e+01,
    -1.179113239482045e-01, -4.268683173977720e-09 };
  static const double cf_Y11C[12] = {
    5.307046e-03,
     2.802854397382923e+00,  4.462442222692911e+02,  7.021330954933401e+03,
     1.885437964587811e+04,  1.508056246869293e+04, -2.094355307813514e+01,
     1.481417448480583e+01,  6.732984477458869e+01, -1.206040482897068e+02,
    -1.057879287786618e+02, -2.765378408631323e-03 };
  static const double cf_Y12C[12] = {
    1.509115e-02,
    -9.141796920496276e-01,  1.342735109731877e+03,  3.936577516657912e+05,
     3.699261691007629e+06, -1.507356098885682e+06,  2.205561499489647e+05,
     3.866992630227200e+01, -1.228093930062573e+03,  5.906456100095043e+03,
     7.791981385860539e+03,  1.873855300354582e+01 };

  const double li    = std::log(1.0 / epsilon);
  const double eps2  = epsilon*epsilon, eps3=eps2*epsilon,
               eps4  = eps3*epsilon,    eps5=eps4*epsilon;
  const double ep[6] = {1.0, epsilon, eps2, eps3, eps4, eps5};
  const double eq[5] = {ep[1], ep[2], ep[3], ep[4], ep[5]};

  // AT asymptotic fallback lambdas
  auto AT_X11A=[&]{return  0.995419+0.25/epsilon+0.225*li+0.0267857*epsilon*li;};
  auto AT_X12A=[&]{return -0.350153-0.25/epsilon-0.225*li-0.0267857*epsilon*li;};
  auto AT_Y11A=[&]{return  0.998317+0.166667*li;};
  auto AT_Y12A=[&]{return -0.273652-0.166667*li;};
  auto AT_Y11B=[&]{return -0.666667*(0.23892-0.25*li-0.125*epsilon*li);};
  auto AT_Y12B=[&]{return  0.666667*(-0.00162268+0.25*li+0.125*epsilon*li);};
  auto AT_X11C=[&]{return  1.33333*(1.0518-0.125*epsilon*li);};
  auto AT_X12C=[&]{return  1.33333*(-0.150257+0.125*epsilon*li);};
  auto AT_Y11C=[&]{return  1.33333*(0.702834+0.2*li+0.188*epsilon*li);};
  auto AT_Y12C=[&]{return  1.33333*(-0.027464+0.05*li+0.062*epsilon*li);};

  // eval helpers — singular subtracts only the 1/eps part, not the full AT
  auto eval_singular = [&](const double* cf, double S_sing,
                            std::function<double()> at_fn) -> double {
    if (epsilon < cf[0]) return at_fn();
    return S_sing + eval_PQ(cf, ep, eq) / epsilon;
  };
  auto eval_regular = [&](const double* cf,
                           std::function<double()> at_fn) -> double {
    if (epsilon < cf[0]) return at_fn();
    return eval_PQ(cf, ep, eq);
  };

  const double X11A =  eval_singular(cf_X11A,  0.25/epsilon, AT_X11A);
  const double X12A =  eval_singular(cf_X12A, -0.25/epsilon, AT_X12A);
  const double Y11A =  eval_regular (cf_Y11A,               AT_Y11A);
  const double Y12A =  eval_regular (cf_Y12A,               AT_Y12A);
  const double Y11B =  eval_regular (cf_Y11B,               AT_Y11B);
  const double Y12B = -eval_regular (cf_Y12B,               AT_Y12B); // fit stores -Y12B
  const double X11C =  eval_regular (cf_X11C,               AT_X11C);
  const double X12C =  eval_regular (cf_X12C,               AT_X12C);
  const double Y11C =  eval_regular (cf_Y11C,               AT_Y11C);
  const double Y12C =  eval_regular (cf_Y12C,               AT_Y12C);

  AssembleResistMatrix(R, mob_factor, r_hat,
                       X11A, Y11A, Y11B, X11C, Y11C,
                       X12A, Y12A, Y12B, X12C, Y12C);
  return R;
}

// =============================================================================
// ResistPairMB: rational fit (MB scalars). No asymptotic fallback.
//
// Coefficient layout: {d_cut, p0..p5, c0..c4}
// cf[0] is d_cut (lower bound of training range) — used as the clamp.
// Sign note: same Y12B convention as Sup — fit stores -Y12B, so negate.
// =============================================================================
Matrix Lubrication::ResistPairMB(double r_norm, double mob_factor[3],
                                 Vector3 r_hat) {
  Matrix R(12, 12);
  double epsilon = r_norm - 2.0;
  if (epsilon < debye_cut) epsilon = debye_cut;

  static const double mb_X11A[12] = {
    5.000000e-03,
    -2.499999982474416e-01,  1.165664399869339e+00,  2.154793952021480e+00,
     1.270149386850713e+00,  2.483649225013684e-01, -4.186019883001905e-05,
    -1.378928715644350e+00,  1.156488416728070e+00,  4.976911470183872e-01,
     4.870465773876030e-06, -4.423392960087700e-07 };
  static const double mb_X12A[12] = {
    5.000000e-03,
     2.500000295508952e-01, -8.497183983207306e-01, -1.951820680989982e-02,
     2.619149841535253e-01, -4.534273305750967e-01, -3.145525674635611e-01,
     8.388694341455940e-01,  1.603695663575000e-06, -6.246243735852574e-01,
     9.857882804654846e-01,  5.012686787659838e-01 };
  static const double mb_Y11A[12] = {
    5.000000e-03,
     1.334448916124861e+00,  1.533276418322284e+00,  2.504280783426653e-02,
     4.777495936484357e-02,  3.630334324421668e-01,  1.237866143758240e-01,
     1.289180844579062e+00,  1.602843358683278e-05,  1.263923257788066e-02,
    -5.997350601777206e-01,  3.521911887962246e-01 };
  static const double mb_Y12A[12] = {
    5.000000e-03,
    -6.167708949219610e-01, -2.537080095626051e-01,  1.307822038490171e-01,
    -1.204261677530989e-01, -1.206649718042400e-01, -8.620245076540467e-05,
    -1.298347445853823e+00, -7.248656296164483e-06, -2.865726715854026e-04,
    -6.865090702813017e-01,  4.033767918608367e-01 };
  static const double mb_Y11B[12] = {
    5.000000e-03,
     1.757300002175146e-01, -1.263482053075663e-01,  5.466331979155107e-02,
     4.145104715957172e-03, -4.855688014271770e-04,  2.869576324453811e-05,
    -1.286512595383362e+00, -1.165742481270486e-04, -4.740203421187537e-03,
    -5.567875571587672e-01, -3.249642869760815e-01 };
  static const double mb_Y12B[12] = {   // fit stores -Y12B
    5.000000e-03,
     3.445954579556256e-01,  1.715184390725956e-02, -8.224756090070001e-02,
     7.735243406016939e-02,  1.035089219385419e-03, -6.484151006279233e-05,
    -1.279854181851254e+00, -1.070970483681416e-04, -4.365905834933606e-02,
    -4.632099590221615e-01,  2.926235258785276e-01 };
  static const double mb_X11C[12] = {
    5.000000e-03,
     1.354497402277264e+00,  1.685515273189768e+00,  3.267336413581359e-02,
    -1.215634951152663e-02,  9.006177715282025e-01,  9.200311837421076e-01,
     1.136664193298283e+00, -1.011738645854246e-04,  6.378569015548268e-04,
     8.207819113317577e-01,  8.307667300391079e-01 };
  static const double mb_X12C[12] = {
    5.000000e-03,
    -1.693122019768674e-01,  9.231352641200651e-02, -1.787657927723367e-02,
    -1.648168837655593e-02,  2.008204276597000e-03, -1.214848156108869e-04,
    -1.001192015055246e+00, -1.528540714747427e-07,  6.544951602958429e-02,
     3.532367548183314e-01,  2.868536488193641e-01 };
  static const double mb_Y11C[12] = {
    5.000000e-03,
     1.427805390573128e+00,  2.108235342790945e+00,  5.698686101168220e-02,
     1.538429638251989e-02,  4.805901339731146e-01,  2.249137271154556e-01,
     1.286277704062037e+00, -1.490352315543504e-05,  1.373451534868378e-01,
     5.996424201465841e-01,  4.107768723410937e-01 };
  static const double mb_Y12C[12] = {
    5.000000e-03,
     1.331703764524452e-01, -8.242976148134072e-02,  2.876821851261879e-02,
     7.924040945752319e-03, -9.931798750875400e-04,  6.253327411129789e-05,
     1.272211977886686e+00, -5.812561617915775e-04, -1.858813454068888e-01,
    -4.861222841829775e-01, -3.301018709137714e-01 };

  const double eps2  = epsilon*epsilon, eps3=eps2*epsilon,
               eps4  = eps3*epsilon,    eps5=eps4*epsilon;
  const double ep[6] = {1.0, epsilon, eps2, eps3, eps4, eps5};
  const double eq[5] = {ep[1], ep[2], ep[3], ep[4], ep[5]};

  // MB: no asymptotic fallback — rational fit used over full range.
  // Singular scalars: S_sing + P/(Q*eps); non-singular: P/Q.
  auto eval_singular_mb = [&](const double* cf, double S_sing) -> double {
    return S_sing + eval_PQ(cf, ep, eq) / epsilon;
  };
  auto eval_regular_mb = [&](const double* cf) -> double {
    return eval_PQ(cf, ep, eq);
  };

  const double X11A =  eval_singular_mb(mb_X11A,  0.25/epsilon);
  const double X12A =  eval_singular_mb(mb_X12A, -0.25/epsilon);
  const double Y11A =  eval_regular_mb (mb_Y11A);
  const double Y12A =  eval_regular_mb (mb_Y12A);
  const double Y11B =  eval_regular_mb (mb_Y11B);
  const double Y12B = -eval_regular_mb (mb_Y12B); // fit stores -Y12B
  const double X11C =  eval_regular_mb (mb_X11C);
  const double X12C =  eval_regular_mb (mb_X12C);
  const double Y11C =  eval_regular_mb (mb_Y11C);
  const double Y12C =  eval_regular_mb (mb_Y12C);

  AssembleResistMatrix(R, mob_factor, r_hat,
                       X11A, Y11A, Y11B, X11C, Y11C,
                       X12A, Y12A, Y12B, X12C, Y12C);
  return R;
}

// =============================================================================
// ResistCSC: build sparse matrix, return as scipy CSC via nanobind
// =============================================================================
SpMat Lubrication::ResistCSC(nb::list r_vectors, nb::list n_list, double a,
                              double eta, nb_array_d periodic_length,
                              bool Sup_if_true) {
  // Pair cutoff hardcoded to PAIR_CUTOFF (4.5a); wall corrections always
  // applied to all particles — the rational fits decay smoothly to zero.
  int num_bodies = (int)r_vectors.size();
  int n_dof      = 6 * num_bodies;
  double mob_factor[3] = {6.0*M_PI*eta*a, 6.0*M_PI*eta*a*a, 6.0*M_PI*eta*a*a*a};
  Vector3 r_jk, r_hat;
  double r_norm, height;
  Matrix R_pair, R_wall;
  const double m_eps = 1e-12;

  std::vector<Triplet> triplets;
  triplets.reserve(num_bodies * 36 * 4);

  std::vector<nb_array_d> r_vecs_cast(num_bodies);
  std::vector<nb_array_i> n_list_cast(num_bodies);
  for (int j = 0; j < num_bodies; j++) {
    r_vecs_cast[j] = nb::cast<nb_array_d>(r_vectors[j]);
    n_list_cast[j] = nb::cast<nb_array_i>(n_list[j]);
  }

  for (int j = 0; j < num_bodies; j++) {
    const nb_array_d &r_j = r_vecs_cast[j];
    height = r_j(2) / a;

    // Wall correction always applied — no cutoff check
    {
      Matrix R_rpy = WallResistMatrixMB(height, mob_factor);
      if (Sup_if_true) {
        R_wall = R_rpy + WallDeltaR(height, mob_factor);
      } else {
        R_wall = R_rpy;
      }
      for (int row = 0; row < 6; row++)
        for (int col = 0; col < 6; col++) {
          double v = R_wall(row, col);
          if (std::fabs(v) > m_eps)
            triplets.emplace_back(row+j*6, col+j*6, v);
        }
    }

    const nb_array_i &neighbors = n_list_cast[j];
    int num_neighbors = (int)neighbors.size();
    if (num_neighbors == 0) continue;

    for (int k_ind = 0; k_ind < num_neighbors; k_ind++) {
      int k = neighbors(k_ind);
      const nb_array_d &r_k = r_vecs_cast[k];
      for (int l = 0; l < 3; ++l) {
        r_jk[l] = r_j(l) - r_k(l);
        if (periodic_length(l) > 0) {
          double Ll = periodic_length(l);
          r_jk[l] -= (int)(r_jk[l]/Ll + 0.5*(int(r_jk[l]>0)-int(r_jk[l]<0)))*Ll;
          r_jk[l] /= a;
        }
      }
      r_norm = r_jk.norm();
      r_hat  = -r_jk / r_norm;

      if (r_norm < PAIR_CUTOFF) {  // hardcoded — fits valid only to 4.5a
        R_pair = Sup_if_true ? ResistPairSup(r_norm, mob_factor, r_hat)
                             : ResistPairMB (r_norm, mob_factor, r_hat);
        const int dof_r[4]={j*6,k*6,j*6,k*6}, dof_c[4]={j*6,k*6,k*6,j*6};
        const int blk_r[4]={0,6,0,6},          blk_c[4]={0,6,6,0};
        for (int b=0;b<4;b++)
          for (int row=0;row<6;row++)
            for (int col=0;col<6;col++) {
              double v = R_pair(blk_r[b]+row, blk_c[b]+col);
              if (std::fabs(v) > m_eps)
                triplets.emplace_back(dof_r[b]+row, dof_c[b]+col, v);
            }
      }
    }
  }

  SpMat Rsp(n_dof, n_dof);
  Rsp.setFromTriplets(triplets.begin(), triplets.end());
  return Rsp;
}

// =============================================================================
// ResistCSC_both: single pass computing both MB and Sup
// =============================================================================
std::pair<SpMat, SpMat>
Lubrication::ResistCSC_both(nb::list r_vectors, nb::list n_list, double a,
                             double eta, nb_array_d periodic_length) {
  // Pair cutoff hardcoded to PAIR_CUTOFF (4.5a); wall corrections always
  // applied to all particles.
  int num_bodies = (int)r_vectors.size();
  int n_dof      = 6 * num_bodies;
  double mob_factor[3] = {6.0*M_PI*eta*a, 6.0*M_PI*eta*a*a, 6.0*M_PI*eta*a*a*a};
  Vector3 r_jk, r_hat;
  double r_norm, height;
  Matrix R_sup, R_mb, R_wall_sup, R_wall_mb;
  const double m_eps = 1e-12;

  std::vector<Triplet> trip_mb, trip_sup;
  trip_mb.reserve(num_bodies * 36 * 4);
  trip_sup.reserve(num_bodies * 36 * 4);

  std::vector<nb_array_d> r_vecs_cast(num_bodies);
  std::vector<nb_array_i> n_list_cast(num_bodies);
  for (int j = 0; j < num_bodies; j++) {
    r_vecs_cast[j] = nb::cast<nb_array_d>(r_vectors[j]);
    n_list_cast[j] = nb::cast<nb_array_i>(n_list[j]);
  }

  auto push_block = [&](std::vector<Triplet> &trips, const Matrix &M,
                        int row_off, int col_off) {
    for (int row=0;row<6;row++)
      for (int col=0;col<6;col++) {
        double v = M(row,col);
        if (std::fabs(v) > m_eps)
          trips.emplace_back(row_off+row, col_off+col, v);
      }
  };

  for (int j = 0; j < num_bodies; j++) {
    const nb_array_d &r_j = r_vecs_cast[j];
    height = r_j(2) / a;

    // Wall correction always applied to all particles — no cutoff check.
    // The rational fits decay smoothly to zero in the far field.
    R_wall_mb  = WallResistMatrixMB(height, mob_factor);
    R_wall_sup = R_wall_mb + WallDeltaR(height, mob_factor);
    push_block(trip_sup, R_wall_sup, j*6, j*6);
    push_block(trip_mb,  R_wall_mb,  j*6, j*6);

    const nb_array_i &neighbors = n_list_cast[j];
    int num_neighbors = (int)neighbors.size();
    if (num_neighbors == 0) continue;

    for (int k_ind = 0; k_ind < num_neighbors; k_ind++) {
      int k = neighbors(k_ind);
      const nb_array_d &r_k = r_vecs_cast[k];
      for (int l = 0; l < 3; ++l) {
        r_jk[l] = r_j(l) - r_k(l);
        if (periodic_length(l) > 0) {
          double Ll = periodic_length(l);
          r_jk[l] -= (int)(r_jk[l]/Ll + 0.5*(int(r_jk[l]>0)-int(r_jk[l]<0)))*Ll;
          r_jk[l] /= a;
        }
      }
      r_norm = r_jk.norm();
      r_hat  = -r_jk / r_norm;

      if (r_norm < PAIR_CUTOFF) {  // hardcoded — fits valid only to 4.5a
        R_sup = ResistPairSup(r_norm, mob_factor, r_hat);
        R_mb  = ResistPairMB (r_norm, mob_factor, r_hat);

        const int dof_r[4]={j*6,k*6,j*6,k*6}, dof_c[4]={j*6,k*6,k*6,j*6};
        const int blk_r[4]={0,6,0,6},          blk_c[4]={0,6,6,0};
        for (int b=0;b<4;b++)
          for (int row=0;row<6;row++)
            for (int col=0;col<6;col++) {
              double vsup = R_sup(blk_r[b]+row, blk_c[b]+col);
              double vmb  = R_mb (blk_r[b]+row, blk_c[b]+col);
              if (std::fabs(vsup) > m_eps)
                trip_sup.emplace_back(dof_r[b]+row, dof_c[b]+col, vsup);
              if (std::fabs(vmb)  > m_eps)
                trip_mb.emplace_back (dof_r[b]+row, dof_c[b]+col, vmb);
            }
      }
    }
  }

  SpMat R_MB_sp(n_dof, n_dof), R_Sup_sp(n_dof, n_dof);
  R_MB_sp.setFromTriplets(trip_mb.begin(),  trip_mb.end());
  R_Sup_sp.setFromTriplets(trip_sup.begin(), trip_sup.end());
  return {R_MB_sp, R_Sup_sp};
}

// =============================================================================
// nanobind module definition
// =============================================================================
using namespace nanobind::literals;

NB_MODULE(lubrication, m) {
  m.doc() = "Lubrication class — nanobind wrapper";
  nb::class_<Lubrication>(m, "Lubrication")
      .def(nb::init<double>(), "d_cut"_a,
           "Construct with Debye cutoff distance.")
      .def("ResistCSC", &Lubrication::ResistCSC,
           "r_vectors"_a, "n_list"_a, "a"_a, "eta"_a,
           "periodic_length"_a, "Sup_if_true"_a,
           "Returns a scipy CSC sparse matrix of the lubrication resistance.")
      .def("ResistCSC_both", &Lubrication::ResistCSC_both,
           "r_vectors"_a, "n_list"_a, "a"_a, "eta"_a,
           "periodic_length"_a,
           "Returns (R_MB, R_Sup) as scipy CSC matrices in a single pair loop.");
}
