import numpy as np
import polyscope as ps
from icecream import ic

import flow_lines


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
    V = np.load('test_data/V.npy')
    F = np.load('test_data/F.npy')
    VN = np.load('test_data/VN.npy')
    Q = np.load('test_data/Q.npy')
    V = normalize_aabb(V)

    strokes = flow_lines.trace(V, F, VN, Q, 4000)

    ps.init()
    ps.register_surface_mesh("mesh", V, F)
    strokes_vis = ps.register_surface_mesh(f"stokes", strokes[0], strokes[1])
    strokes_vis.add_color_quantity('color', strokes[2], enabled=True)
    ps.show()
