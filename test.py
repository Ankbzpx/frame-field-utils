import numpy as np
import polyscope as ps
from icecream import ic

import flow_lines

if __name__ == '__main__':
    V = np.load('test_data/V.npy')
    F = np.load('test_data/F.npy')
    VN = np.load('test_data/VN.npy')
    Q = np.load('test_data/Q.npy')

    flow_lines.trace(V, F, VN, Q)