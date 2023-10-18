#include <igl/Hit.h>
#include <igl/avg_edge_length.h>
#include <igl/embree/EmbreeIntersector.h>
#include <igl/parallel_for.h>
#include <igl/per_face_normals.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/tet_tet_adjacency.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

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
compact_orientation_extrinsic_rosy4(const Eigen::RowVector3d &q0,
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

// Random a barycentric coordinate
// https://stackoverflow.com/questions/68493050/sample-uniformly-random-points-within-a-triangle
void random_point_in_triangle(Eigen::RowVector3d *bary_coord) {
  double u = randd(), v = randd();
  if (u + v > 1.0) {
    u = 1.0 - u;
    v = 1.0 - v;
  }
  double w = 1.0 - (u + v);

  *bary_coord << w, u, v;
}

void vertex_weighted_position(Eigen::RowVector3d *o,
                              const Eigen::RowVector3d &weights, int fid,
                              const RowMatrixXi &F, const RowMatrixXd &V) {
  o->setZero();
  for (size_t i = 0; i < 3; i++) {
    *o += weights[i] * V.row(F.coeff(fid, i));
  }
}

void vertex_weighted_normal(Eigen::RowVector3d *n,
                            const Eigen::RowVector3d &weights, int fid,
                            const RowMatrixXi &F, const RowMatrixXd &VN) {
  n->setZero();
  for (size_t i = 0; i < 3; i++) {
    *n += weights[i] * VN.row(F.coeff(fid, i));
  }
  n->normalize();
}

// Project to tangent plane, equivalent to (I - nn^T)q
void project_tangent(Eigen::RowVector3d *q, const Eigen::RowVector3d &n) {
  *q = (*q - n * n.dot(*q)).normalized();
}

void random_tangent(Eigen::RowVector3d *q, const Eigen::RowVector3d &n) {
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
void vertex_weighted_tangent(Eigen::RowVector3d *q, const Eigen::RowVector3d &n,
                             const Eigen::RowVector3d &weights, int fid,
                             const RowMatrixXi &F, const RowMatrixXd &VN,
                             const RowMatrixXd &Q) {
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

int trace_step(Eigen::RowVector3d *bary_coord, const Eigen::RowVector3d &o,
               const Eigen::RowVector3d &q, const Eigen::RowVector3d &n,
               const igl::embree::EmbreeIntersector &intersector,
               double step_size) {
  Eigen::RowVector3d o_next = o + step_size * q;
  igl::Hit hit0, hit1;
  bool is_hit0 =
      intersector.intersectRay(o_next.cast<float>(), n.cast<float>(), hit0);
  bool is_hit1 =
      intersector.intersectRay(o_next.cast<float>(), -n.cast<float>(), hit1);

  is_hit0 = is_hit0 && hit0.t < step_size;
  is_hit1 = is_hit1 && hit1.t < step_size;

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
    return -1;
  }

  *bary_coord << 1.0 - hit->u - hit->v, hit->u, hit->v;
  return hit->id;
}

void compact_tangent(Eigen::RowVector3d *q, const Eigen::RowVector3d &n,
                     const Eigen::RowVector3d &weights, int fid,
                     const RowMatrixXi &F, const RowMatrixXd &VN,
                     const RowMatrixXd &Q) {
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

// https://en.wikipedia.org/wiki/Grayscale
float rgb_to_intensity(const Eigen::RowVector3d &rgb) {
  // return (rgb[0] + rgb[1] + rgb[2]) / 3.0;
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

int random_color_idx() {
  double sample = randd();
  if (sample < 0.6) {
    return 0;
  } else if (sample < 0.75) {
    return 1;
  } else if (sample < 0.9) {
    return 2;
  } else {
    return 3;
  }
}

py::list trace_flow_lines(Eigen::Ref<const RowMatrixXd> V,
                          Eigen::Ref<const RowMatrixXi> F,
                          Eigen::Ref<const RowMatrixXd> VN,
                          Eigen::Ref<const RowMatrixXd> Q, int n_lines,
                          int n_steps, double line_length, double line_width,
                          double line_offset) {
  double step_size = line_length / n_steps;
  double delta = line_offset;

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
          o += (line_offset + variation) * n;

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

  RowMatrixXd color_palettes(4, 3);
  color_palettes.row(0) << 70, 25, 89;
  color_palettes.row(1) << 122, 49, 111;
  color_palettes.row(2) << 205, 102, 136;
  color_palettes.row(3) << 174, 216, 204;
  double value_variation = 0.5;

  int n_tail_steps = static_cast<int>(n_steps / 4);
  int base = 0;
  for (size_t idx = 0; idx < n_lines; idx++) {
    int n_valid = valid_steps[idx];

    Eigen::RowVector3d color = color_palettes.row(random_color_idx());
    double intensity = rgb_to_intensity(color);
    double intensity_new =
        intensity + value_variation * intensity * (2 * randd() - 1);
    color *= intensity_new / intensity / 255.0;

    for (size_t i = 0; i < n_valid; i++) {
      Eigen::RowVector3d pos = positions[idx]->row(i);
      Eigen::RowVector3d t = bitangents[idx]->row(i);

      double w_scale_lower = std::sqrt(
          i > n_tail_steps ? 1 : i / static_cast<double>(n_tail_steps));
      double w_scale_upper =
          std::sqrt((n_valid - i) > n_tail_steps
                        ? 1
                        : (n_valid - i) / static_cast<double>(n_tail_steps));
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

py::list tet_edge_one_ring(Eigen::Ref<const RowMatrixXi> T) {
  // Faces 0:012 1:013 2:123 3:203
  Eigen::MatrixXi TT;
  igl::tet_tet_adjacency(T, TT);

  Eigen::MatrixXi TE(6, 2), EF(6, 2);
  // Edges (undirected): 01 02 03 12 13 23
  TE << 0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3;
  // Faces index in tet that is adjacent to its edge: 01 03 13 02 12 23
  EF << 0, 1, 0, 3, 1, 3, 0, 2, 1, 2, 2, 3;

  // Map edge to string for hashing
  auto get_uEid = [](int v0, int v1) {
    return std::to_string(v0) + "_" + std::to_string(v1);
  };

  // Max number of possible undirected edges
  int uE_reserve_size = T.rows() * 6;
  // Track if the edge has been marked
  std::unordered_map<std::string, int> uE2uEid;
  uE2uEid.reserve(uE_reserve_size);
  // Store undirected edge vertex indices
  std::vector<int> uE_vec;
  uE_vec.reserve(uE_reserve_size * 2);
  // Store adjacent tets for each undirected edge
  std::vector<std::vector<int>> uE2T_vec_vec;
  uE2T_vec_vec.reserve(uE_reserve_size);
  // Store if the undirected edge is boundary
  std::vector<int> uE_boundary_mask_vec;
  uE_boundary_mask_vec.reserve(uE_reserve_size);
  // Store if the adjacent tet is boundary w.r.t. current undirected edge
  std::vector<std::vector<bool>> uE2T_boundary_vec_vec;
  uE2T_boundary_vec_vec.reserve(uE_reserve_size);
  // Maps each undirected edge of tet to uE_id
  Eigen::MatrixXi E2uE(T.rows(), 6);
  E2uE.setConstant(-1);
  // Maps each undirected edge of tet to id of adjacent tets w.r.t. the edge
  Eigen::MatrixXi E2T0(T.rows(), 6), E2T1(T.rows(), 6);
  E2T0.setConstant(-1);
  E2T1.setConstant(-1);
  int uE_count = 0;

  for (int i = 0; i < T.rows(); i++) {
    // Each tet has 6 undirected edges
    for (int j = 0; j < 6; j++) {
      int i0 = TE.coeff(j, 0);
      int i1 = TE.coeff(j, 1);

      E2T0.coeffRef(i, j) = TT.coeff(i, EF.coeff(j, 0));
      E2T1.coeffRef(i, j) = TT.coeff(i, EF.coeff(j, 1));

      // If it is on boundary face
      bool is_boundary =
          (E2T0.coeffRef(i, j) == -1) || (E2T1.coeffRef(i, j) == -1);

      int v0 = T.coeff(i, i0), v1 = T.coeff(i, i1);
      if (v0 > v1) {
        int tmp = v1;
        v1 = v0;
        v0 = tmp;
      }
      std::string ue_tag = get_uEid(v0, v1);

      if (uE2uEid.find(ue_tag) == uE2uEid.end()) {
        uE2uEid[ue_tag] = uE_count;
        uE_vec.push_back(v0);
        uE_vec.push_back(v1);
        uE2T_vec_vec.push_back({i});
        uE_boundary_mask_vec.push_back(is_boundary);
        uE2T_boundary_vec_vec.push_back({is_boundary});
        E2uE.coeffRef(i, j) = uE_count;
        uE_count++;
      } else {
        int ue_id = uE2uEid[ue_tag];
        E2uE.coeffRef(i, j) = ue_id;
        uE2T_vec_vec[ue_id].push_back(i);
        uE2T_boundary_vec_vec[ue_id].push_back(is_boundary);

        if (is_boundary) {
          uE_boundary_mask_vec[ue_id] = true;
        }
      }
    }
  }

  RowMatrixXi uE = Eigen::Map<RowMatrixXi>(uE_vec.data(), uE_count, 2);
  Eigen::VectorXi uE_boundary_mask =
      Eigen::Map<Eigen::VectorXi>(uE_boundary_mask_vec.data(), uE_count);

  Eigen::VectorXi uE_mark(uE_count);
  uE_mark.setZero();

  std::unordered_map<int, int> uE2T_size;
  uE2T_size.reserve(uE_count);
  int uE2T_size_total = 0;

  // Sort one-ring
  for (int i = 0; i < T.rows(); i++) {
    for (int j = 0; j < 6; j++) {
      int ue_id = E2uE.coeff(i, j);

      if (uE_mark.coeff(ue_id) == 0) {
        uE_mark.coeffRef(ue_id) = 1;
        const std::vector<int> &T_adj = uE2T_vec_vec[ue_id];

        uE2T_size[ue_id] = T_adj.size();
        uE2T_size_total += T_adj.size();

        // Since the edge is undirected, the traverse order (clockwise /
        // counterclockwise) doesn't matter
        if (T_adj.size() < 3) {
          continue;
        }

        int t_id = i, t_id_end = i, t_id_boundary_first;
        if (uE_boundary_mask.coeff(ue_id) == 1) {
          const std::vector<bool> &T_adj_boundary =
              uE2T_boundary_vec_vec[ue_id];
          bool is_first_set = false;

          for (int k = 0; k < T_adj.size(); k++) {
            if (T_adj_boundary[k]) {
              if (!is_first_set) {
                t_id = T_adj[k];
                t_id_boundary_first = T_adj[k];
                is_first_set = true;
              }
              t_id_end = T_adj[k];
            }
          }
        }

        int t_id_last = -1;
        std::vector<int> T_adj_sorted;
        T_adj_sorted.reserve(T_adj.size());

        bool finished = false;

        while (!finished) {
          int e_id = -1;
          for (int k = 0; k < 6; k++) {
            if (E2uE.coeff(t_id, k) == ue_id) {
              e_id = k;
              break;
            }
          }

          // Next t_id
          int t_id_ = E2T0.coeff(t_id, e_id);
          if (t_id_ == t_id_last) {
            t_id_ = E2T1.coeff(t_id, e_id);
          }
          t_id_last = t_id;
          t_id = t_id_;

          finished = (t_id == t_id_end);
          T_adj_sorted.push_back(t_id);
        }

        if (uE_boundary_mask.coeff(ue_id) == 1) {
          T_adj_sorted.insert(T_adj_sorted.begin(), t_id_boundary_first);
        }

        assert(T_adj.size() == T_adj_sorted.size());

        uE2T_vec_vec[ue_id] = std::move(T_adj_sorted);
      }
    }
  }

  Eigen::VectorXi uE2T(uE2T_size_total);
  Eigen::VectorXi uE2T_cumsum(uE_count + 1);
  uE2T_cumsum.coeffRef(0) = 0;

  for (int i = 0; i < uE_count; i++) {
    int T_adj_size = uE2T_size[i];
    uE2T_cumsum.coeffRef(i + 1) = uE2T_cumsum.coeff(i) + T_adj_size;
    for (int j = 0; j < T_adj_size; j++) {
      uE2T.coeffRef(uE2T_cumsum.coeff(i) + j) = uE2T_vec_vec[i][j];
    }
  }

  py::list return_list;
  return_list.append(uE);
  return_list.append(uE_boundary_mask);
  return_list.append(uE2T);
  return_list.append(uE2T_cumsum);
  return return_list;
}

void transition_matrix(const RowMatrix3d &R_i, const RowMatrix3d &R_j,
                       RowMatrix3d *m) {
  m->setZero();

  for (int j = 0; j < 3; j++) {
    Eigen::RowVector3d dp = R_j.transpose() * R_i.col(j);
    double val = -INFINITY;
    int max_idx;
    double sign;
    for (int i = 0; i < 3; i++) {
      double v = dp.coeff(i);
      double v_abs = std::abs(v);
      if (v_abs > val) {
        val = v_abs;
        max_idx = i;
        sign = (v > 0.0) ? 1.0 : -1.0;
      }
    }
    m->coeffRef(max_idx, j) = sign;
  }
}

Eigen::VectorXi
tet_edge_singular(Eigen::Ref<const RowMatrixXi> uE,
                  Eigen::Ref<const Eigen::VectorXi> uE_boundary_mask,
                  Eigen::Ref<const Eigen::VectorXi> uE2T,
                  Eigen::Ref<const Eigen::VectorXi> uE2T_cumsum,
                  Eigen::Ref<const RowMatrixXd> tetFrames) {

  assert(uE2T.size() == uE2T_cumsum.coeff(uE2T_cumsum.size() - 1));
  assert(uE.rows() == uE_boundary_mask.size() == uE2T_cumsum.size() - 1);
  assert(uE2T.maxCoeff() == tetFrames.rows() - 1);
  assert(tetFrames.cols() == 9);

  Eigen::VectorXi uE_singularity_mask(uE.rows());

  igl::parallel_for(
      uE.rows(),
      [&](int i) {
        if (uE_boundary_mask.coeff(i) == 1) {
          uE_singularity_mask.coeffRef(i) = false;
        } else {
          RowMatrix3d m = RowMatrix3d::Identity(), transition;
          for (int j = uE2T_cumsum.coeff(i); j < uE2T_cumsum.coeff(i + 1);
               j++) {
            int t_i = uE2T.coeff(j);
            // Here we ignore boundary singularity and always assume one ring
            // cycle
            int t_j = uE2T.coeff((j == uE2T_cumsum.coeff(i + 1) - 1)
                                     ? uE2T_cumsum.coeff(i)
                                     : j + 1);

            const RowMatrix3d R_i =
                Eigen::Map<const RowMatrix3d>(tetFrames.row(t_i).data());
            const RowMatrix3d R_j =
                Eigen::Map<const RowMatrix3d>(tetFrames.row(t_j).data());

            transition_matrix(R_i, R_j, &transition);
            m = transition * m;
          }
          bool is_singular = (m - RowMatrix3d::Identity()).norm() > 1e-7;
          uE_singularity_mask.coeffRef(i) = is_singular;
        }
      },
      1000);

  // for (int i = 0; i < uE.rows(); i++) {
  //   if (uE_boundary_mask.coeff(i) == 1) {
  //     uE_singularity_mask.coeffRef(i) = false;
  //     continue;
  //   }

  //   RowMatrix3d m = RowMatrix3d::Identity(), transition;
  //   for (int j = uE2T_cumsum.coeff(i); j < uE2T_cumsum.coeff(i + 1); j++) {
  //     int t_i = uE2T.coeff(j);
  //     // Here we ignore boundary singularity
  //     int t_j = uE2T.coeff(
  //         (j == uE2T_cumsum.coeff(i + 1) - 1) ? uE2T_cumsum.coeff(i) : j +
  //         1);

  //     const RowMatrix3d R_i =
  //         Eigen::Map<const RowMatrix3d>(tetFrames.row(t_i).data());
  //     const RowMatrix3d R_j =
  //         Eigen::Map<const RowMatrix3d>(tetFrames.row(t_j).data());

  //     transition_matrix(R_i, R_j, &transition);
  //     m = transition * m;
  //   }
  //   bool is_singular = (m - RowMatrix3d::Identity()).norm() > 1e-7;
  //   uE_singularity_mask.coeffRef(i) = is_singular;
  // }

  return uE_singularity_mask;
}

PYBIND11_MODULE(flow_lines_bind, m) {
  m.doc() = "Trace flow lines";
  m.def("trace", &trace_flow_lines, py::return_value_policy::reference_internal,
        "Sample occlusions");
  m.def("tet_edge_one_ring", &tet_edge_one_ring,
        py::return_value_policy::reference_internal,
        "Build edge one ring data structure for tetrahedral mesh");
  m.def("tet_edge_singular", &tet_edge_singular,
        py::return_value_policy::reference_internal,
        "Given per tet coordinate frame, compute mask of singularity for "
        "undirected edge");
}
