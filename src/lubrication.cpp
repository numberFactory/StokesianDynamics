#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>
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

class Lubrication {
private:
  void SetMemberData(std::string fname,
                     std::vector<std::vector<double>> &vec_11,
                     std::vector<std::vector<double>> &vec_12,
                     std::vector<double> &x);
  void SetMemberDataWall(std::string fname,
                         std::vector<std::vector<double>> &vec,
                         std::vector<double> &x, bool reverse);
  std::vector<std::vector<double>> mob_scalars_WS_11, mob_scalars_WS_12;
  std::vector<double> WS_x;
  std::vector<std::vector<double>> mob_scalars_JO_11, mob_scalars_JO_12;
  std::vector<double> JO_x;
  std::vector<std::vector<double>> mob_scalars_MB_11, mob_scalars_MB_12;
  std::vector<double> MB_x;
  std::vector<std::vector<double>> mob_scalars_wall_2562;
  std::vector<double> Wall_2562_x;
  std::vector<std::vector<double>> mob_scalars_wall_MB;
  std::vector<double> Wall_MB_x;
  int FindNearestIndexLower(double r_norm, std::vector<double> &x);
  double LinearInterp(double r_norm, double xL, double xR, double yL,
                      double yR);
  void ResistMatrix(double r_norm, double mob_factor[3], Vector3 r_hat,
                    Matrix &R, bool inv, std::vector<double> &x,
                    const std::vector<std::vector<double>> &vec_11,
                    const std::vector<std::vector<double>> &vec_12);
  void ATResistMatrix(double r_norm, double mob_factor[3], Vector3 r_hat,
                      Matrix &R);
  Matrix WallResistMatrix(double r_norm, double mob_factor[3],
                          std::vector<double> &x,
                          const std::vector<std::vector<double>> &vec);
  Matrix ResistPairSup(double r_norm, double mob_factor[3], Vector3 r_hat);
  Matrix WallResistMatrixMB(double r_norm, double mob_factor[3],
                            std::vector<double> &x,
                            const std::vector<std::vector<double>> &vec);
  Matrix ResistPairMB(double r_norm, double mob_factor[3], Vector3 r_hat);

public:
  SpMat ResistCSC(nb::list r_vectors, nb::list n_list, double a, double eta,
                  double cutoff, double wall_cutoff, nb_array_d periodic_length,
                  bool Sup_if_true);
  std::pair<SpMat, SpMat> ResistCSC_both(nb::list r_vectors, nb::list n_list,
                                          double a, double eta, double cutoff,
                                          double wall_cutoff,
                                          nb_array_d periodic_length);
  double debye_cut;
  Lubrication(double d_cut);
};

Lubrication::Lubrication(double d_cut) {
  debye_cut = d_cut;
  std::string base_dir =
      (std::filesystem::path(__FILE__).parent_path().parent_path()).string();
  base_dir += "/resistance_coeffs/";
  SetMemberData(base_dir + "mob_scalars_WS.txt", mob_scalars_WS_11,
                mob_scalars_WS_12, WS_x);
  SetMemberData(base_dir + "res_scalars_JO.txt", mob_scalars_JO_11,
                mob_scalars_JO_12, JO_x);
  SetMemberDataWall(base_dir + "mob_scalars_wall_MB_2562_eig_thresh.txt",
                    mob_scalars_wall_2562, Wall_2562_x, true);
  SetMemberData(base_dir + "res_scalars_MB_1.txt", mob_scalars_MB_11,
                mob_scalars_MB_12, MB_x);
  SetMemberDataWall(base_dir + "res_scalars_wall_MB.txt", mob_scalars_wall_MB,
                    Wall_MB_x, false);
}

void Lubrication::SetMemberData(std::string fname,
                                std::vector<std::vector<double>> &vec_11,
                                std::vector<std::vector<double>> &vec_12,
                                std::vector<double> &x) {
  std::ifstream ifs(fname);
  double tempval;
  std::vector<double> tempv;

  if (!ifs.fail()) {
    int p = 0;
    int c = -1;
    while (!ifs.eof()) {
      c++;
      ifs >> tempval;
      tempv.push_back(tempval);
      if (c == 5) {
        p++;
        c = -1;
        if (p % 2) {
          vec_11.push_back(tempv);
        } else {
          vec_12.push_back(tempv);
        }
        tempv.clear();
      }
    }
    ifs.close();
  }
  for (auto row : vec_11)
    x.push_back(row[0]);
}

void Lubrication::SetMemberDataWall(std::string fname,
                                    std::vector<std::vector<double>> &vec,
                                    std::vector<double> &x, bool reverse) {
  std::ifstream ifs(fname);
  double tempval;
  std::vector<double> tempv;

  if (!ifs.fail()) {
    int c = -1;
    while (!ifs.eof()) {
      c++;
      ifs >> tempval;
      tempv.push_back(tempval);
      if (c == 5) {
        c = -1;
        if (reverse) {
          vec.insert(vec.begin(), tempv);
        } else {
          vec.push_back(tempv);
        }
        tempv.clear();
      }
    }
    ifs.close();
  }
  for (auto row : vec)
    x.push_back(row[0]);
}

int Lubrication::FindNearestIndexLower(double r_norm, std::vector<double> &x) {
  std::vector<double>::iterator before;
  before = std::lower_bound(x.begin(), x.end(), r_norm);
  if (before == x.begin()) {
    return -1;
  }
  if (before == x.end()) {
    return (int)x.size() - 1;
  }
  --before;
  return (int)std::distance(x.begin(), before);
}

double Lubrication::LinearInterp(double r_norm, double xL, double xR, double yL,
                                 double yR) {
  if (r_norm < xL || r_norm > xR) {
    std::cout << "error in linear interp." << std::endl;
    return 1e100;
  }
  double dydx = (yR - yL) / (xR - xL);
  return yL + dydx * (r_norm - xL);
}

void Lubrication::ResistMatrix(double r_norm, double mob_factor[3],
                               Vector3 r_hat, Matrix &R, bool inv,
                               std::vector<double> &x,
                               const std::vector<std::vector<double>> &vec_11,
                               const std::vector<std::vector<double>> &vec_12) {
  double X11A, Y11A, Y11B, X11C, Y11C;
  double X12A, Y12A, Y12B, X12C, Y12C;

  int Ind = FindNearestIndexLower(r_norm, x);

  if (Ind == -1 || Ind == (int)x.size() - 1) {
    int edge = (Ind == -1) ? 0 : ((int)x.size() - 1);
    X11A = vec_11[edge][1];
    Y11A = vec_11[edge][2];
    Y11B = vec_11[edge][3];
    X11C = vec_11[edge][4];
    Y11C = vec_11[edge][5];
    X12A = vec_12[edge][1];
    Y12A = vec_12[edge][2];
    Y12B = vec_12[edge][3];
    X12C = vec_12[edge][4];
    Y12C = vec_12[edge][5];
  } else {
    double a_11[5], a_12[5];
    double xL = x[Ind], xR = x[Ind + 1];
    for (int i = 0; i < 5; i++) {
      a_11[i] = LinearInterp(r_norm, xL, xR, vec_11[Ind][i + 1],
                             vec_11[Ind + 1][i + 1]);
      a_12[i] = LinearInterp(r_norm, xL, xR, vec_12[Ind][i + 1],
                             vec_12[Ind + 1][i + 1]);
    }
    X11A = a_11[0]; Y11A = a_11[1]; Y11B = a_11[2];
    X11C = a_11[3]; Y11C = a_11[4];
    X12A = a_12[0]; Y12A = a_12[1]; Y12B = a_12[2];
    X12C = a_12[3]; Y12C = a_12[4];
  }

  Matrix3 squeezeMat = r_hat * r_hat.transpose();
  Matrix3 Eye = Matrix3::Identity();
  Matrix3 shearMat = Eye - squeezeMat;
  Matrix3 vortMat;
  vortMat << 0.0, r_hat[2], -r_hat[1], -r_hat[2], 0.0, r_hat[0], r_hat[1],
      -r_hat[0], 0.0;
  vortMat *= -1;

  R.block<3, 3>(0, 0) = mob_factor[0] * (X11A * squeezeMat + Y11A * shearMat);
  R.block<3, 3>(0, 3) = -mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(0, 6) = mob_factor[0] * (X12A * squeezeMat + Y12A * shearMat);
  R.block<3, 3>(0, 9) = mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(3, 0) = mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(3, 3) = mob_factor[2] * (X11C * squeezeMat + Y11C * shearMat);
  R.block<3, 3>(3, 6) = mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(3, 9) = mob_factor[2] * (X12C * squeezeMat + Y12C * shearMat);
  R.block<3, 3>(6, 0) = mob_factor[0] * (X12A * squeezeMat + Y12A * shearMat);
  R.block<3, 3>(6, 3) = -mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(6, 6) = mob_factor[0] * (X11A * squeezeMat + Y11A * shearMat);
  R.block<3, 3>(6, 9) = mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(9, 0) = -mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(9, 3) = mob_factor[2] * (X12C * squeezeMat + Y12C * shearMat);
  R.block<3, 3>(9, 6) = -mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(9, 9) = mob_factor[2] * (X11C * squeezeMat + Y11C * shearMat);

  if (inv) {
    R = R.inverse();
  }
}

void Lubrication::ATResistMatrix(double r_norm, double mob_factor[3],
                                 Vector3 r_hat, Matrix &R) {
  double epsilon = r_norm - 2.0;

  double X11A = 0.995419E0 + (0.25E0) * (1.0 / epsilon) +
                (0.225E0) * log((1.0 / epsilon)) +
                (0.267857E-1) * epsilon * log((1.0 / epsilon));
  double X12A = (-0.350153E0) + (-0.25E0) * (1.0 / epsilon) +
                (-0.225E0) * log((1.0 / epsilon)) +
                (-0.267857E-1) * epsilon * log((1.0 / epsilon));
  double Y11A = 0.998317E0 + (0.166667E0) * log((1.0 / epsilon));
  double Y12A = (-0.273652E0) + (-0.166667E0) * log((1.0 / epsilon));
  double Y11B = (-0.666667E0) * (0.23892E0 + (-0.25E0) * log((1.0 / epsilon)) +
                                 (-0.125E0) * epsilon * log((1.0 / epsilon)));
  double Y12B =
      (-0.666667E0) * ((-0.162268E-2) + (0.25E0) * log((1.0 / epsilon)) +
                       (0.125E0) * epsilon * log((1.0 / epsilon)));
  double X11C =
      0.133333E1 * (0.10518E1 + (-0.125E0) * epsilon * log((1.0 / epsilon)));
  double X12C =
      0.133333E1 * ((-0.150257E0) + (0.125E0) * epsilon * log((1.0 / epsilon)));
  double Y11C = 0.133333E1 * (0.702834E0 + (0.2E0) * log((1.0 / epsilon)) +
                              (0.188E0) * epsilon * log((1.0 / epsilon)));
  double Y12C = 0.133333E1 * ((-0.27464E-1) + (0.5E-1) * log((1.0 / epsilon)) +
                              (0.62E-1) * epsilon * log((1.0 / epsilon)));

  Matrix3 squeezeMat = r_hat * r_hat.transpose();
  Matrix3 Eye = Matrix3::Identity();
  Matrix3 shearMat = Eye - squeezeMat;
  Matrix3 vortMat;
  vortMat << 0.0, r_hat[2], -r_hat[1], -r_hat[2], 0.0, r_hat[0], r_hat[1],
      -r_hat[0], 0.0;
  vortMat *= -1;

  R.block<3, 3>(0, 0) = mob_factor[0] * (X11A * squeezeMat + Y11A * shearMat);
  R.block<3, 3>(0, 3) = -mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(0, 6) = mob_factor[0] * (X12A * squeezeMat + Y12A * shearMat);
  R.block<3, 3>(0, 9) = mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(3, 0) = mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(3, 3) = mob_factor[2] * (X11C * squeezeMat + Y11C * shearMat);
  R.block<3, 3>(3, 6) = mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(3, 9) = mob_factor[2] * (X12C * squeezeMat + Y12C * shearMat);
  R.block<3, 3>(6, 0) = mob_factor[0] * (X12A * squeezeMat + Y12A * shearMat);
  R.block<3, 3>(6, 3) = -mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(6, 6) = mob_factor[0] * (X11A * squeezeMat + Y11A * shearMat);
  R.block<3, 3>(6, 9) = mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(9, 0) = -mob_factor[1] * (Y12B * vortMat);
  R.block<3, 3>(9, 3) = mob_factor[2] * (X12C * squeezeMat + Y12C * shearMat);
  R.block<3, 3>(9, 6) = -mob_factor[1] * (Y11B * vortMat);
  R.block<3, 3>(9, 9) = mob_factor[2] * (X11C * squeezeMat + Y11C * shearMat);
}

Matrix
Lubrication::WallResistMatrix(double r_norm, double mob_factor[3],
                              std::vector<double> &x,
                              const std::vector<std::vector<double>> &vec) {
  double Xa, Ya, Yb, Xc, Yc;
  double epsilon = r_norm - 1.0;
  //double tanh_fact = 1.0;

  if (epsilon < debye_cut) {
    epsilon = debye_cut;
    r_norm = 1.0 + epsilon;
  }

  int Ind = FindNearestIndexLower(r_norm, x);
  if (Ind == -1) {
    Xa = vec[0][1]; Ya = vec[0][2]; Yb = vec[0][3];
    Xc = vec[0][4]; Yc = vec[0][5];
  } else if (Ind == (int)x.size() - 1) {
    Xa = 1.0 - (9.0 / 8.0) * (1.0 / r_norm);
    Ya = 1.0 - (9.0 / 16.0) * (1.0 / r_norm);
    Yb = 0.0; Xc = 0.75; Yc = 0.75;
  } else {
    double a[5], xL = x[Ind], xR = x[Ind + 1];
    for (int i = 0; i < 5; i++)
      a[i] = LinearInterp(r_norm, xL, xR, vec[Ind][i + 1], vec[Ind + 1][i + 1]);
    Xa = a[0]; Ya = a[1]; Yb = a[2]; Xc = a[3]; Yc = a[4];
  }

  double Xa_asym = 1.0 / epsilon - (1.0 / 5.0) * log(epsilon) + 0.971280;
  double Ya_asym = -(8.0 / 15.0) * log(epsilon) + 0.9588;
  double Yb_asym =
      (4. / 3.) * (-(-(1.0 / 10.0) * log(epsilon) - 0.1895) - 0.4576 * epsilon);
  double Xc_asym =
      (4. / 3.) * (1.2020569 - 3.0 * (M_PI * M_PI / 6.0 - 1.0) * epsilon);
  double Yc_asym =
      (4. / 3.) * (-2.0 / 5.0 * log(epsilon) + 0.3817 + 1.4578 * epsilon);

  double denom = Ya * Yc - Yb * Yb;
  double RXa = 1.0 / Xa, RYa = Yc / denom, RYb = -Yb / denom;
  double RXc = 1.0 / Xc, RYc = Ya / denom;

  Xa = (r_norm > 1.1)  ? RXa : Xa_asym;
  Ya = (r_norm > 1.01) ? RYa : Ya_asym;
  Yb = (r_norm > 1.1)  ? RYb : Yb_asym;
  Xc = (r_norm > 1.01) ? RXc : Xc_asym;
  Yc = (r_norm > 1.1)  ? RYc : Yc_asym;

  double XcPlus = fmax((Xc - 4.0 / 3.0), 0.0);
  double YcPlus = fmax((Yc - 4.0 / 3.0), 0.0);

  Matrix R(6, 6);
  R << mob_factor[0] * (Ya - 1.), 0, 0, 0, mob_factor[1] * Yb, 0, 0,
      mob_factor[0] * (Ya - 1.), 0, -mob_factor[1] * Yb, 0, 0, 0, 0,
      mob_factor[0] * (Xa - 1.), 0, 0, 0, 0, -mob_factor[1] * Yb, 0,
      mob_factor[2] * YcPlus, 0, 0, mob_factor[1] * Yb, 0, 0, 0,
      mob_factor[2] * YcPlus, 0, 0, 0, 0, 0, 0, mob_factor[2] * XcPlus;

  // if (fabs(tanh_fact - 1.0) > 1e-10)
  //   R *= tanh_fact;
  return R;
}

Matrix
Lubrication::WallResistMatrixMB(double r_norm, double mob_factor[3],
                                std::vector<double> &x,
                                const std::vector<std::vector<double>> &vec) {
  double Xa, Ya, Yb, Xc, Yc;
  double epsilon = r_norm - 1.0;
  //double tanh_fact = 1.0;

  if (epsilon < debye_cut) {
    epsilon = debye_cut;
    r_norm = 1.0 + epsilon;
  }

  int Ind = FindNearestIndexLower(r_norm, x);
  if (Ind == -1) {
    Xa = vec[0][1]; Ya = vec[0][2]; Yb = vec[0][3];
    Xc = vec[0][4]; Yc = vec[0][5];
  } else if (Ind == (int)x.size() - 1) {
    Xa = 1.0 / (1.0 - (9.0 / 8.0) * (1.0 / r_norm));
    Ya = 1.0 / (1.0 - (9.0 / 16.0) * (1.0 / r_norm));
    Yb = 0.0; Xc = 1.0 / 0.75; Yc = 1.0 / 0.75;
  } else {
    double a[5], xL = x[Ind], xR = x[Ind + 1];
    for (int i = 0; i < 5; i++)
      a[i] = LinearInterp(r_norm, xL, xR, vec[Ind][i + 1], vec[Ind + 1][i + 1]);
    Xa = a[0]; Ya = a[1]; Yb = a[2]; Xc = a[3]; Yc = a[4];
  }

  Matrix R(6, 6);
  R << mob_factor[0] * (Ya - 1.), 0, 0, 0, mob_factor[1] * Yb, 0, 0,
      mob_factor[0] * (Ya - 1.), 0, -mob_factor[1] * Yb, 0, 0, 0, 0,
      mob_factor[0] * (Xa - 1.), 0, 0, 0, 0, -mob_factor[1] * Yb, 0,
      mob_factor[2] * (Yc - 4.0 / 3.0), 0, 0, mob_factor[1] * Yb, 0, 0, 0,
      mob_factor[2] * (Yc - 4.0 / 3.0), 0, 0, 0, 0, 0, 0,
      mob_factor[2] * (Xc - 4.0 / 3.0);

  // if (fabs(tanh_fact - 1.0) > 1e-10)
  //   R *= tanh_fact;
  return R;
}

Matrix Lubrication::ResistPairSup(double r_norm, double mob_factor[3],
                                  Vector3 r_hat) {
  double AT_cutoff = (2.0 + 0.006 - 1e-8);
  double WS_cutoff = (2.0 + 0.1 + 1e-8);
  bool inv;
  double res_factor[3] = {1.0 / mob_factor[0], 1.0 / mob_factor[1],
                          1.0 / mob_factor[2]};
  Matrix R(12, 12);

  double epsilon = r_norm - 2.0;
  //double tanh_fact = 1.0;
  if (epsilon < debye_cut) {
    epsilon = debye_cut;
    r_norm = epsilon + 2.0;
  }

  if (r_norm <= AT_cutoff) {
    ATResistMatrix(r_norm, mob_factor, r_hat, R);
  } else if (r_norm <= WS_cutoff) {
    inv = true;
    ResistMatrix(r_norm, res_factor, r_hat, R, inv, WS_x, mob_scalars_WS_11,
                 mob_scalars_WS_12);
  } else {
    inv = false;
    ResistMatrix(r_norm, mob_factor, r_hat, R, inv, JO_x, mob_scalars_JO_11,
                 mob_scalars_JO_12);
  }

  // if (fabs(tanh_fact - 1.0) > 1e-10)
  //   R *= tanh_fact;
  return R;
}

Matrix Lubrication::ResistPairMB(double r_norm, double mob_factor[3],
                                 Vector3 r_hat) {
  bool inv = false;
  Matrix R(12, 12);

  double epsilon = r_norm - 2.0;
  //double tanh_fact = 1.0;
  if (epsilon < debye_cut) {
    epsilon = debye_cut;
    r_norm = epsilon + 2.0;
  }

  ResistMatrix(r_norm, mob_factor, r_hat, R, inv, MB_x, mob_scalars_MB_11,
               mob_scalars_MB_12);
  // if (fabs(tanh_fact - 1.0) > 1e-10)
  //   R *= tanh_fact;
  return R;
}

// =============================================================================
// ResistCSC: build sparse matrix using Eigen triplets and return directly.
// nanobind's eigen/sparse.h type caster converts SpMat -> scipy CSC matrix.
// =============================================================================
SpMat Lubrication::ResistCSC(nb::list r_vectors, nb::list n_list, double a,
                              double eta, double cutoff, double wall_cutoff,
                              nb_array_d periodic_length, bool Sup_if_true) {
  int num_bodies = (int)r_vectors.size();
  int n_dof      = 6 * num_bodies;
  double mob_factor[3] = {6.0 * M_PI * eta * a,
                          6.0 * M_PI * eta * a * a,
                          6.0 * M_PI * eta * a * a * a};
  Vector3 r_jk, r_hat;
  double r_norm, height;
  Matrix R_pair, R_wall;
  const double m_eps = 1e-12;

  std::vector<Triplet> triplets;
  triplets.reserve(num_bodies * 36 * 4);

  for (int j = 0; j < num_bodies; j++) {
    nb_array_d r_j = nb::cast<nb_array_d>(r_vectors[j]);
    height = r_j(2) / a;

    // wall contribution
    if (height < wall_cutoff) {
      if (Sup_if_true)
        R_wall = WallResistMatrix(height, mob_factor, Wall_2562_x,
                                  mob_scalars_wall_2562);
      else
        R_wall = WallResistMatrixMB(height, mob_factor, Wall_MB_x,
                                    mob_scalars_wall_MB);

      for (int row = 0; row < 6; row++)
        for (int col = 0; col < 6; col++) {
          double v = R_wall(row, col);
          if (std::fabs(v) > m_eps)
            triplets.emplace_back(row + j*6, col + j*6, v);
        }
    }

    // pair contributions
    nb_array_i neighbors = nb::cast<nb_array_i>(n_list[j]);
    int num_neighbors = (int)neighbors.size();
    if (num_neighbors == 0) continue;

    for (int k_ind = 0; k_ind < num_neighbors; k_ind++) {
      int k = neighbors(k_ind);
      nb_array_d r_k = nb::cast<nb_array_d>(r_vectors[k]);

      for (int l = 0; l < 3; ++l) {
        r_jk[l] = r_j(l) - r_k(l);
        if (periodic_length(l) > 0) {
          double Ll = periodic_length(l);
          r_jk[l] -= (int)(r_jk[l] / Ll +
                           0.5 * (int(r_jk[l] > 0) - int(r_jk[l] < 0))) * Ll;
          r_jk[l] /= a;
        }
      }
      r_norm = r_jk.norm();
      r_hat  = -r_jk / r_norm;

      if (r_norm < cutoff) {
        if (Sup_if_true)
          R_pair = ResistPairSup(r_norm, mob_factor, r_hat);
        else
          R_pair = ResistPairMB(r_norm, mob_factor, r_hat);

        // four 6x6 blocks: jj, kk, jk, kj
        const int dof_r[4] = {j*6, k*6, j*6, k*6};
        const int dof_c[4] = {j*6, k*6, k*6, j*6};
        const int blk_r[4] = {0,   6,   0,   6  };
        const int blk_c[4] = {0,   6,   6,   0  };

        for (int b = 0; b < 4; b++)
          for (int row = 0; row < 6; row++)
            for (int col = 0; col < 6; col++) {
              double v = R_pair(blk_r[b] + row, blk_c[b] + col);
              if (std::fabs(v) > m_eps)
                triplets.emplace_back(dof_r[b] + row, dof_c[b] + col, v);
            }
      }
    }
  }

  SpMat R(n_dof, n_dof);
  R.setFromTriplets(triplets.begin(), triplets.end());
  return R;
}

// =============================================================================
// ResistCSC_both: compute MB and Sup matrices in a single pass over all pairs.
// Shared geometric setup (r_jk, r_norm, r_hat, wall matrices) is computed once
// per pair, halving the work compared to calling ResistCSC twice.
// Returns (R_MB, R_Sup) as a pair of scipy CSC matrices.
// =============================================================================
std::pair<SpMat, SpMat>
Lubrication::ResistCSC_both(nb::list r_vectors, nb::list n_list, double a,
                             double eta, double cutoff, double wall_cutoff,
                             nb_array_d periodic_length) {
  int num_bodies = (int)r_vectors.size();
  int n_dof      = 6 * num_bodies;
  double mob_factor[3] = {6.0 * M_PI * eta * a,
                          6.0 * M_PI * eta * a * a,
                          6.0 * M_PI * eta * a * a * a};
  Vector3 r_jk, r_hat;
  double r_norm, height;
  Matrix R_sup, R_mb, R_wall_sup, R_wall_mb;
  const double m_eps = 1e-12;

  std::vector<Triplet> trip_mb, trip_sup;
  trip_mb.reserve(num_bodies * 36 * 4);
  trip_sup.reserve(num_bodies * 36 * 4);

  // helper lambda: push all nonzero entries of a 6x6 block into a triplet list
  auto push_block = [&](std::vector<Triplet> &trips, const Matrix &M,
                        int row_off, int col_off) {
    for (int row = 0; row < 6; row++)
      for (int col = 0; col < 6; col++) {
        double v = M(row, col);
        if (std::fabs(v) > m_eps)
          trips.emplace_back(row_off + row, col_off + col, v);
      }
  };

  for (int j = 0; j < num_bodies; j++) {
    nb_array_d r_j = nb::cast<nb_array_d>(r_vectors[j]);
    height = r_j(2) / a;

    // wall contributions — compute both Sup and MB wall matrices once per particle
    if (height < wall_cutoff) {
      R_wall_sup = WallResistMatrix(height, mob_factor, Wall_2562_x,
                                    mob_scalars_wall_2562);
      R_wall_mb  = WallResistMatrixMB(height, mob_factor, Wall_MB_x,
                                      mob_scalars_wall_MB);
      push_block(trip_sup, R_wall_sup, j*6, j*6);
      push_block(trip_mb,  R_wall_mb,  j*6, j*6);
    }

    // pair contributions
    nb_array_i neighbors = nb::cast<nb_array_i>(n_list[j]);
    int num_neighbors = (int)neighbors.size();
    if (num_neighbors == 0) continue;

    for (int k_ind = 0; k_ind < num_neighbors; k_ind++) {
      int k = neighbors(k_ind);
      nb_array_d r_k = nb::cast<nb_array_d>(r_vectors[k]);

      // shared geometric setup — computed once for both MB and Sup
      for (int l = 0; l < 3; ++l) {
        r_jk[l] = r_j(l) - r_k(l);
        if (periodic_length(l) > 0) {
          double Ll = periodic_length(l);
          r_jk[l] -= (int)(r_jk[l] / Ll +
                           0.5 * (int(r_jk[l] > 0) - int(r_jk[l] < 0))) * Ll;
          r_jk[l] /= a;
        }
      }
      r_norm = r_jk.norm();
      r_hat  = -r_jk / r_norm;

      if (r_norm < cutoff) {
        // compute both resistance matrices for this pair
        R_sup = ResistPairSup(r_norm, mob_factor, r_hat);
        R_mb  = ResistPairMB(r_norm, mob_factor, r_hat);

        const int dof_r[4] = {j*6, k*6, j*6, k*6};
        const int dof_c[4] = {j*6, k*6, k*6, j*6};
        const int blk_r[4] = {0,   6,   0,   6  };
        const int blk_c[4] = {0,   6,   6,   0  };

        for (int b = 0; b < 4; b++) {
          for (int row = 0; row < 6; row++)
            for (int col = 0; col < 6; col++) {
              double vsup = R_sup(blk_r[b] + row, blk_c[b] + col);
              double vmb  = R_mb (blk_r[b] + row, blk_c[b] + col);
              if (std::fabs(vsup) > m_eps)
                trip_sup.emplace_back(dof_r[b] + row, dof_c[b] + col, vsup);
              if (std::fabs(vmb) > m_eps)
                trip_mb.emplace_back (dof_r[b] + row, dof_c[b] + col, vmb);
            }
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
           "Construct Lubrication with a Debye cutoff distance.")
      .def("ResistCSC", &Lubrication::ResistCSC,
           "r_vectors"_a, "n_list"_a, "a"_a, "eta"_a,
           "cutoff"_a, "wall_cutoff"_a, "periodic_length"_a, "Sup_if_true"_a,
           "Returns a scipy CSC sparse matrix of the lubrication resistance.")
      .def("ResistCSC_both", &Lubrication::ResistCSC_both,
           "r_vectors"_a, "n_list"_a, "a"_a, "eta"_a,
           "cutoff"_a, "wall_cutoff"_a, "periodic_length"_a,
           "Returns (R_MB, R_Sup) as scipy CSC matrices in a single pair loop.");
}