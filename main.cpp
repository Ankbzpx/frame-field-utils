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
  std::random_device rd;
  static thread_local std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

// https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/common.h#L234
inline double signum(double value) { return std::copysign(1.0, value); }

// https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/field.cpp#L183
std::pair<Eigen::RowVector3d, Eigen::RowVector3d>
compact_orientation_extrinsic_rosy4(const Eigen::RowVector3d& q0,
                                    const Eigen::RowVector3d& n0,
                                    const Eigen::RowVector3d& q1,
                                    const Eigen::RowVector3d& n1) {
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

// Random a barycentric coordinate
// https://stackoverflow.com/questions/68493050/sample-uniformly-random-points-within-a-triangle
void random_point_in_triangle(Eigen::RowVector3d* bary_coord) {
  double u = randd(), v = randd();
  if (u + v > 1.0) {
    u = 1.0 - u;
    v = 1.0 - v;
  }
  double w = 1.0 - (u + v);

  *bary_coord << w, u, v;
}

void vertex_weighted_position(Eigen::RowVector3d* o,
                              const Eigen::RowVector3d& weights, int fid,
                              const RowMatrixXi& F, const RowMatrixXd& V) {
  o->setZero();
  for (size_t i = 0; i < 3; i++) {
    *o += weights[i] * V.row(F.coeff(fid, i));
  }
}

void vertex_weighted_normal(Eigen::RowVector3d* n,
                            const Eigen::RowVector3d& weights, int fid,
                            const RowMatrixXi& F, const RowMatrixXd& VN) {
  n->setZero();
  for (size_t i = 0; i < 3; i++) {
    *n += weights[i] * VN.row(F.coeff(fid, i));
  }
  n->normalize();
}

// Project to tangent plane, equivalent to (I - nn^T)q
void project_tangent(Eigen::RowVector3d* q, const Eigen::RowVector3d& n) {
  *q = (*q - n * n.dot(*q)).normalized();
}

void random_tangent(Eigen::RowVector3d* q, const Eigen::RowVector3d& n) {
  int start_dir = randi(4);
  switch (start_dir) {
    case 1:
      *q = n.cross(*q);
      break;
    case 2:
      *q *= -1;
      break;
    case 3:
      *q = -n.cross(*q);
      break;
    default:
      break;
  }
}

// Vertex weighted compact tangent
// https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/field.cpp#L636
void vertex_weighted_tangent(Eigen::RowVector3d* q, const Eigen::RowVector3d& n,
                             const Eigen::RowVector3d& weights, int fid,
                             const RowMatrixXi& F, const RowMatrixXd& VN,
                             const RowMatrixXd& Q) {
  // Randomize a tangent vector
  *q = Eigen::RowVector3d::Random();
  project_tangent(q, n);

  double weight_sum = 0.0;
  for (size_t i = 0; i < 3; i++) {
    auto pair = compact_orientation_extrinsic_rosy4(
        *q, n, Q.row(F.coeff(fid, 0)), VN.row(F.coeff(fid, 0)));
    *q = weight_sum * pair.first + weights[i] * pair.second;
    project_tangent(q, n);
    weight_sum += weights[i];
  }
}

int trace_step(Eigen::RowVector3d* bary_coord, const Eigen::RowVector3d& o,
               const Eigen::RowVector3d& q, const Eigen::RowVector3d& n,
               const igl::embree::EmbreeIntersector& intersector,
               double step_size) {
  Eigen::RowVector3d o_next = o + step_size * q;
  igl::Hit hit0, hit1;
  bool is_hit0 =
      intersector.intersectRay(o_next.cast<float>(), n.cast<float>(), hit0);
  bool is_hit1 =
      intersector.intersectRay(o_next.cast<float>(), -n.cast<float>(), hit1);

  igl::Hit* hit;
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
    return -1;
  }

  *bary_coord << 1.0 - hit->u - hit->v, hit->u, hit->v;
  return hit->id;
}

void compact_tangent(Eigen::RowVector3d* q, const Eigen::RowVector3d& n,
                     const Eigen::RowVector3d& weights, int fid,
                     const RowMatrixXi& F, const RowMatrixXd& VN,
                     const RowMatrixXd& Q) {
  Eigen::RowVector3d qp;
  vertex_weighted_tangent(&qp, n, weights, fid, F, VN, Q);
  project_tangent(&qp, n);

  Eigen::RowVector3d best_qp = Eigen::RowVector3d::Zero();
  double best_dp = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < 4; i++) {
    double dp = qp.dot(*q);
    if (dp > best_dp) {
      best_dp = dp;
      best_qp = qp;
    }
    qp = n.cross(qp);
  }
  *q = best_qp;
}

py::list trace_flow_lines(Eigen::Ref<const RowMatrixXd> V,
                          Eigen::Ref<const RowMatrixXi> F,
                          Eigen::Ref<const RowMatrixXd> VN,
                          Eigen::Ref<const RowMatrixXd> Q) {
  std::cout << "trace_flow_lines" << std::endl;

  double avg_edge_length = igl::avg_edge_length(V, F);
  double step_size = avg_edge_length / 2.0;
  int n_steps = 20;
  double offset = avg_edge_length / 10.0;

  igl::embree::EmbreeIntersector intersector;
  intersector.init(V.cast<float>(), F.cast<int>(), true);

  py::list return_list;

  RowMatrixXd os, qs;
  os.setZero(n_steps, 3);
  qs.setZero(n_steps, 3);

  // Debug
  int fid = randi(F.rows());

  Eigen::RowVector3d bary_coord;
  random_point_in_triangle(&bary_coord);

  Eigen::RowVector3d n, q, o;
  vertex_weighted_normal(&n, bary_coord, fid, F, VN);
  vertex_weighted_tangent(&q, n, bary_coord, fid, F, VN, Q);
  random_tangent(&q, n);

  for (size_t i = 0; i < n_steps; i++) {
    vertex_weighted_position(&o, bary_coord, fid, F, V);
    o += offset * n;

    // Record traced position
    os.row(i) = o;
    qs.row(i) = q;

    fid = trace_step(&bary_coord, o, q, n, intersector, step_size);
    if (fid < 0) {
      std::cout << "Failed" << std::endl;
      break;
    }
    vertex_weighted_normal(&n, bary_coord, fid, F, VN);
    project_tangent(&q, n);
    compact_tangent(&q, n, bary_coord, fid, F, VN, Q);
  }

  return_list.append(os);
  return_list.append(qs);

  return return_list;
}

PYBIND11_MODULE(flow_lines, m) {
  m.doc() = "Trace flow lines";
  m.def("trace", &trace_flow_lines, py::return_value_policy::reference_internal,
        "Sample occlusions");
}
