import numpy as np
import polyscope as ps
from icecream import ic

import flow_lines
import flow_lines_bind


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


if __name__ == '__main__':
    data = np.load('test_data/prism.npz')
    V = np.float64(data['V'])
    T = np.int64(data['T'])
    Rs_bary: np.array = data['Rs_bary']

    uE, uE_boundary_mask, uE2T, uE2T_cumsum = flow_lines_bind.tet_edge_one_ring(
        T)
    uE_singularity_mask = flow_lines_bind.tet_edge_singular(
        uE, uE_boundary_mask, uE2T, uE2T_cumsum, Rs_bary.reshape(-1, 9))
    uE_singularity_mask = uE_singularity_mask.astype(bool)

    ps.init()
    ps.register_curve_network('singularity', V, uE[uE_singularity_mask])
    ps.show()
