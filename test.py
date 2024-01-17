import numpy as np
import polyscope as ps
from icecream import ic
import time

import igl
import frame_field_utils


def normalize_aabb(V):
    V = np.copy(V)
    # [0, 1]
    V -= np.mean(V, axis=0, keepdims=True)
    V_max = np.amax(V)
    V_min = np.amin(V)
    V = (V - V_min) / (V_max - V_min)

    # [-0.95, 0.95]
    V -= 0.5
    V *= 1.9
    return V


def test_flowline():
    V = np.load('test_data/V.npy')
    F = np.load('test_data/F.npy')
    VN = np.load('test_data/VN.npy')
    Q = np.load('test_data/Q.npy')
    V = normalize_aabb(V)

    strokes = frame_field_utils.trace(V, F, VN, Q, 4000)

    ps.init()
    ps.register_surface_mesh("mesh", V, F)
    strokes_vis = ps.register_surface_mesh(f"stokes", strokes[0], strokes[1])
    strokes_vis.add_color_quantity('color', strokes[2], enabled=True)
    ps.show()
    ps.remove_all_structures()


def test_tet():
    data = np.load('test_data/prism.npz')
    V = np.float64(data['V'])
    T = np.int64(data['T'])
    Rs_bary: np.array = data['Rs_bary']

    TT, _ = igl.tet_tet_adjacency(T)
    uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, E2uE, E2T = frame_field_utils.tet_edge_one_ring(
        T, TT)
    uE_singularity_mask = frame_field_utils.tet_frame_singularity(
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, Rs_bary)

    ps.init()
    ps.register_curve_network('singularity', V, uE[uE_singularity_mask])
    ps.show()
    ps.remove_all_structures()


def test_sdp():
    num_test_samples = 300000
    np.random.seed(0)
    q = np.random.randn(num_test_samples, 9)

    helper = frame_field_utils.SH4SDPProjectHelper()

    start_time = time.time()
    q = helper.project(q, 1024)
    print(
        f"Project {num_test_samples} test samples using {time.time() - start_time}s"
    )

    assert (np.linalg.norm(q, axis=-1) - 1).max() < 1e-2


def test_dc():
    sdf = lambda x: np.linalg.norm(x) - 0.5
    sdf_grad = lambda x: x

    min_corner = np.array([-1.0, -1.0, -1.0])
    max_corner = np.array([1.0, 1.0, 1.0])
    grid_res = 16

    V, F = frame_field_utils.dual_contouring_serial(sdf,
                                                    sdf_grad,
                                                    min_corner,
                                                    max_corner,
                                                    grid_res,
                                                    grid_res,
                                                    grid_res,
                                                    triangles=True)

    ps.init()
    ps.register_surface_mesh('mesh', V, F)
    ps.show()
    ps.remove_all_structures()


def test_miq():
    data = np.load('test_data/miq.npz')
    V = data['V']
    F = data['F']
    Q = data['Q']

    UV, FUV = frame_field_utils.miq(V, F, Q)
    UV3 = np.hstack([UV, np.ones(len(UV))[:, None]])

    ps.init()
    ps.register_surface_mesh('param', UV3, FUV)
    ps.show()
    ps.remove_all_structures()


if __name__ == '__main__':
    test_flowline()
    test_tet()
    test_sdp()
    test_dc()
    test_miq()
