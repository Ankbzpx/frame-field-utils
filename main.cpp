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

// https://github.com/wjakob/instant-meshes/blob/7b3160864a2e1025af498c84cfed91cbfb613698/src/common.h#L358
void hsv_to_rgb(Eigen::RowVector3d* rgb, double h, double s, double v) {
  if (s == 0.f) {  // achromatic (grey)
    rgb->setConstant(v);
    return;
  }
  h *= 6;
  int i = std::floor(h);
  double f = h - i;  // fractional part of h
  double p = v * (1 - s);
  double q = v * (1 - s * f);
  double t = v * (1 - s * (1 - f));
  switch (i) {
    case 0:
      *rgb << v, t, p;
      break;
    case 1:
      *rgb << q, v, p;
      break;
    case 2:
      *rgb << p, v, t;
      break;
    case 3:
      *rgb << p, q, v;
      break;
    case 4:
      *rgb << t, p, v;
      break;
    default:
      *rgb << v, p, q;
      break;
  }
}

py::list trace_flow_lines(Eigen::Ref<const RowMatrixXd> V,
                          Eigen::Ref<const RowMatrixXi> F,
                          Eigen::Ref<const RowMatrixXd> VN,
                          Eigen::Ref<const RowMatrixXd> Q, int n_lines,
                          double length_factor, double interval_factor,
                          double offset_factor, double width_factor) {
  double avg_edge_length = igl::avg_edge_length(V, F);

  int n_steps = static_cast<int>(length_factor * interval_factor);
  double step_size = avg_edge_length / (2.0 * interval_factor);

  double offset = avg_edge_length * offset_factor;
  double line_width = avg_edge_length * width_factor;
  double delta = avg_edge_length * (offset_factor * 2.0);

  igl::embree::EmbreeIntersector intersector;
  intersector.init(V.cast<float>(), F.cast<int>(), true);

  std::vector<std::unique_ptr<RowMatrixXd>> positions, bitangents;
  std::vector<int> valid_steps;

  positions.resize(n_lines);
  bitangents.resize(n_lines);
  valid_steps.resize(n_lines);

  igl::parallel_for(
      n_lines,
      [&](int idx) {
        RowMatrixXd os, ts;
        os.setZero(n_steps, 3);
        ts.setZero(n_steps, 3);

        int i = 0;
        int fid = randi(F.rows() - 1);
        double variation = randd() * delta;

        Eigen::RowVector3d bary_coord;
        random_point_in_triangle(&bary_coord);

        Eigen::RowVector3d n, q, o;
        vertex_weighted_normal(&n, bary_coord, fid, F, VN);
        vertex_weighted_tangent(&q, n, bary_coord, fid, F, VN, Q);
        random_tangent(&q, n);

        for (; i < n_steps; i++) {
          vertex_weighted_position(&o, bary_coord, fid, F, V);
          o += (offset + variation) * n;

          // Record traced position
          os.row(i) = o;
          ts.row(i) = n.cross(q);

          fid = trace_step(&bary_coord, o, q, n, intersector, step_size);
          if (fid < 0) {
            break;
          }
          vertex_weighted_normal(&n, bary_coord, fid, F, VN);
          project_tangent(&q, n);
          compact_tangent(&q, n, bary_coord, fid, F, VN, Q);
        }

        positions[idx] = std::make_unique<RowMatrixXd>(os);
        bitangents[idx] = std::make_unique<RowMatrixXd>(ts);
        valid_steps[idx] = i;
      },
      1000);

  int total_valid = 0;
  for (size_t idx = 0; idx < n_lines; idx++) {
    total_valid += valid_steps[idx];
  }
  RowMatrixXd V_stroke, VC_stroke;
  RowMatrixXi F_stroke;
  V_stroke.setZero(2 * total_valid, 3);
  VC_stroke.setZero(2 * total_valid, 3);
  F_stroke.setZero(2 * (total_valid - n_lines), 3);

  int tail_steps = static_cast<int>(2.0 * interval_factor);
  int base = 0;
  for (size_t idx = 0; idx < n_lines; idx++) {
    int n_valid = valid_steps[idx];

    double hue = std::fmod(randi(idx) * M_PI, 1.0);
    Eigen::RowVector3d color;
    hsv_to_rgb(&color, hue, 0.75, 0.5 + 0.35 * randd());

    for (size_t i = 0; i < n_valid; i++) {
      Eigen::RowVector3d pos = positions[idx]->row(i);
      Eigen::RowVector3d t = bitangents[idx]->row(i);

      double w_scale_lower =
          std::sqrt(i > tail_steps ? 1 : i / static_cast<double>(tail_steps));
      double w_scale_upper =
          std::sqrt((n_valid - i) > tail_steps
                        ? 1
                        : (n_valid - i) / static_cast<double>(tail_steps));
      double w_scale =
          w_scale_lower < w_scale_upper ? w_scale_lower : w_scale_upper;

      V_stroke.row(base + 2 * i) = pos + w_scale * line_width * t;
      V_stroke.row(base + 2 * i + 1) = pos - w_scale * line_width * t;

      VC_stroke.row(base + 2 * i) = color;
      VC_stroke.row(base + 2 * i + 1) = color;

      if (i != n_valid - 1) {
        int v0 = base + 2 * i;
        int v1 = base + 2 * i + 1;

        int v2 = base + 2 * (i + 1);
        int v3 = base + 2 * (i + 1) + 1;

        F_stroke.row(base - 2 * idx + 2 * i) << v0, v1, v2;
        F_stroke.row(base - 2 * idx + 2 * i + 1) << v1, v3, v2;
      }
    }
    base += 2 * n_valid;
  }

  py::list return_list;
  return_list.append(V_stroke);
  return_list.append(F_stroke);
  return_list.append(VC_stroke);
  return return_list;
}

PYBIND11_MODULE(flow_lines_bind, m) {
  m.doc() = "Trace flow lines";
  m.def("trace", &trace_flow_lines, py::return_value_policy::reference_internal,
        "Sample occlusions");
}
