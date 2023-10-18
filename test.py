import numpy as np
import polyscope as ps
from icecream import ic

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

    uE, uE_boundary_mask, uE2T, uE2T_cumsum = frame_field_utils.tet_edge_one_ring(
        T)
    uE_singularity_mask = frame_field_utils.tet_edge_singularity(
        uE, uE_boundary_mask, uE2T, uE2T_cumsum, Rs_bary.reshape(-1, 9))

    ps.init()
    ps.register_curve_network('singularity', V, uE[uE_singularity_mask])
    ps.show()
    ps.remove_all_structures()


if __name__ == '__main__':
    test_flowline()
    test_tet()
