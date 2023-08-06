#include <igl/Hit.h>
#include <igl/avg_edge_length.h>
#include <igl/embree/EmbreeIntersector.h>
#include <igl/parallel_for.h>
#include <igl/per_face_normals.h>
#include <igl/ray_mesh_intersect.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <limits>
#include <random>
#include <string>

namespace py = pybind11;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
int randi(int min, int max) {
  std::random_device rd;
  static thread_local std::mt19937 rng(rd());
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

int randi(int max) { return randi(0, max); }

double randd() {
  // std::random_device rd;
  // static thread_local std::mt19937 rng(rd());

  // debug
  static thread_local std::mt19937 rng;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

// https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/common.h#L234
inline double signum(double value) { return std::copysign(1.0, value); }

// https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/field.cpp#L183
std::pair<Eigen::RowVector3d, Eigen::RowVector3d>
compact_orientation_extrinsic_4(const Eigen::RowVector3d &q0,
                                const Eigen::RowVector3d &n0,
                                const Eigen::RowVector3d &q1,
                                const Eigen::RowVector3d &n1) {
  const Eigen::RowVector3d bundle_0[2] = {q0, n0.cross(q0)};
  const Eigen::RowVector3d bundle_1[2] = {q1, n1.cross(q1)};

  double best_score = -std::numeric_limits<double>::infinity();
  int best_0 = 0, best_1 = 0;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double score = std::abs(bundle_0[i].dot(bundle_1[j]));
      if (score > best_score) {
        best_0 = i;
        best_1 = j;
        best_score = score;
      }
    }
  }
  const double dp = bundle_0[best_0].dot(bundle_1[best_1]);
  return std::make_pair(bundle_0[best_0], bundle_1[best_1] * signum(dp));
}

py::list trace_flow_lines(Eigen::Ref<const RowMatrixXd> V,
                          Eigen::Ref<const RowMatrixXi> F,
                          Eigen::Ref<const RowMatrixXd> VN,
                          Eigen::Ref<const RowMatrixXd> Q) {
  std::cout << "trace_flow_lines" << std::endl;

  double avg_edge_length = igl::avg_edge_length(V, F);
  double step_size = avg_edge_length / 2.0;
  double offset = avg_edge_length / 10.0;

  igl::embree::EmbreeIntersector intersector;
  intersector.init(V.cast<float>(), F.cast<int>(), true);

  // int fid = randi(F.rows());
  int fid = 1564;

  // Random a barycentric coordinate
  // https://stackoverflow.com/questions/68493050/sample-uniformly-random-points-within-a-triangle
  double u = randd(), v = randd();
  if (u + v > 1.0) {
    u = 1.0 - u;
    v = 1.0 - v;
  }
  double w = 1.0 - (u + v);
  double weights[3] = {u, v, w};

  Eigen::RowVector3d n;

  n.setZero();
  for (size_t i = 0; i < 3; i++) {
    n += weights[i] * VN.row(F.coeff(fid, i));
  }
  n.normalize();

  // Randomize a tangent vector
  Eigen::RowVector3d q = Eigen::RowVector3d::Random();
  // Project to tangent plane, equivalent to (I - nn^T)q
  q = (q - n * n.dot(q)).normalized();

  // https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/field.cpp#L636
  double weight_sum = 0.0;
  for (size_t i = 0; i < 3; i++) {
    auto pair = compact_orientation_extrinsic_4(q, n, Q.row(F.coeff(fid, 0)),
                                                VN.row(F.coeff(fid, 0)));
    q = weight_sum * pair.first + weights[i] * pair.second;
    q = (q - n * n.dot(q)).normalized();
    weight_sum += weights[i];
  }

  Eigen::RowVector3d o;

  o.setZero();
  for (size_t i = 0; i < 3; i++) {
    o += weights[i] * V.row(F.coeff(fid, i));
  }
  o += offset * n;

  int start_dir = randi(4);
  // Debug
  start_dir = 0;
  switch (start_dir) {
  case 1:
    q = n.cross(q);
    break;
  case 2:
    q = -q;
    break;
  case 3:
    q = -n.cross(q);
    break;
  default:
    break;
  }

  Eigen::RowVector3d o_next = o + step_size * q;
  igl::Hit hit0, hit1;
  bool is_hit0 =
      intersector.intersectRay(o_next.cast<float>(), n.cast<float>(), hit0);
  bool is_hit1 =
      intersector.intersectRay(o_next.cast<float>(), -n.cast<float>(), hit1);

  igl::Hit *hit;
  if (is_hit0 && is_hit1) {
    if (hit0.t < hit1.t) {
      hit = &hit0;
    } else {
      hit = &hit1;
    }
  } else if (is_hit0) {
    hit = &hit0;
  } else if (is_hit1) {
    hit = &hit1;
  } else {
    std::cout << "Failed" << std::endl;
  }

  fid = hit->id;
  weights[0] = hit->u;
  weights[1] = hit->v;
  weights[2] = 1.0 - weights[0] - weights[1];

  std::cout << fid << std::endl;

  py::list return_list;
  return return_list;
}

PYBIND11_MODULE(flow_lines, m) {
  m.doc() = "Trace flow lines";
  m.def("trace", &trace_flow_lines, py::return_value_policy::reference_internal,
        "Sample occlusions");
}
