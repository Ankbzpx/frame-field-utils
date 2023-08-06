import numpy as np
import polyscope as ps
from icecream import ic

import flow_lines

if __name__ == '__main__':
    V = np.load('test_data/V.npy')
    F = np.load('test_data/F.npy')
    VN = np.load('test_data/VN.npy')
    Q = np.load('test_data/Q.npy')

    strokes = flow_lines.trace(V, F, VN, Q, 200, 20, 5, 0.1, 0.1)

    ps.init()
    ps.register_surface_mesh("mesh", V, F)
    strokes_vis = ps.register_surface_mesh(f"stokes", strokes[0], strokes[1])
    strokes_vis.add_color_quantity('color', strokes[2], enabled=True)
    ps.show()