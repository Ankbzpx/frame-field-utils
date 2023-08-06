import numpy as np
import polyscope as ps
from icecream import ic

import flow_lines

if __name__ == '__main__':
    V = np.load('test_data/V.npy')
    F = np.load('test_data/F.npy')
    VN = np.load('test_data/VN.npy')
    Q = np.load('test_data/Q.npy')

    res = flow_lines.trace(V, F, VN, Q)

    ps.init()
    mesh_vis = ps.register_surface_mesh("mesh", V, F)
    mesh_vis.add_vector_quantity("VN", VN, enabled=True)
    mesh_vis.add_vector_quantity("Q", Q, enabled=True)
    trace_vis = ps.register_point_cloud("trace", res[0])
    trace_vis.add_vector_quantity("q", res[1], enabled=True)
    ps.show()