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

    exit()

    u, v, n, q = res

    w = 1.0 - u - v

    fid = 1564
    FN = np.zeros_like(F, dtype=np.float64)
    Qf = np.zeros_like(F, dtype=np.float64)
    Qf[fid] = q
    FN[fid] = n

    ps.init()
    mesh_vis = ps.register_surface_mesh("mesh", V, F)
    mesh_vis.add_vector_quantity("VN", VN, enabled=True)
    mesh_vis.add_vector_quantity("Q", Q, enabled=True)
    mesh_vis.add_vector_quantity("Qf", Qf, defined_on='faces', enabled=True)
    mesh_vis.add_vector_quantity("FN", FN, defined_on='faces', enabled=True)
    ps.show()