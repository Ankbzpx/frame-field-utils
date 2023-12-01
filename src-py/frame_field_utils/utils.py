import numpy as np
import frame_field_utils_bind


def trace(V,
          F,
          VN,
          Q,
          n_lines,
          n_steps=20,
          line_length=1.5e-1,
          line_width=5e-3,
          line_offset=3e-3):
    return frame_field_utils_bind.trace(V, F, VN, Q, n_lines, n_steps,
                                        line_length, line_width, line_offset)


def vertex_tet_adjacency(T):
    return frame_field_utils_bind.vertex_tet_adjacency(T)


def tet_fix_index_order(V, T):
    return frame_field_utils_bind.tet_fix_index_order(V, T)


def tet_edge_one_ring(T, TT):
    uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum, E2uE, E2T0, E2T1 = frame_field_utils_bind.tet_edge_one_ring(
        T, TT)
    return uE, uE_boundary_mask.astype(bool), uE_non_manifold_mask.astype(
        bool), uE2T, uE2T_cumsum, E2uE, np.stack([E2T0, E2T1], -1)


def tet_frame_singularity(uE, uE_boundary_mask, uE_non_manifold_mask, uE2T,
                          uE2T_cumsum, Rs_tet):
    uE_singularity_mask = frame_field_utils_bind.tet_frame_singularity(
        uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum,
        Rs_tet.reshape(-1, 9))
    return uE_singularity_mask.astype(bool)


def tet_comb_frame(T, TT, Rs_tet, params_tet):
    return frame_field_utils_bind.tet_comb_frame(T, TT, Rs_tet.reshape(-1, 9),
                                                 params_tet).reshape(-1, 3, 3)


def tet_uF_map(T, TT, TTi):
    return frame_field_utils_bind.tet_uF_map(T, TT, TTi)


def tet_uE_uF_map(uE, uE_boundary_mask, uE_non_manifold_mask, uE2T, uE2T_cumsum,
                  E2uE, F2uF):
    return frame_field_utils_bind.tet_uE_uF_map(uE, uE_boundary_mask,
                                                uE_non_manifold_mask, uE2T,
                                                uE2T_cumsum, E2uE, F2uF)


def tet_frame_mismatch(T, TT, TTi, Rs_tet):
    return frame_field_utils_bind.tet_frame_mismatch(T, TT, TTi,
                                                     Rs_tet.reshape(-1, 9)) > 0


def tet_reduce(V, VN, V_mask, T):
    return frame_field_utils_bind.tet_reduce(V, VN, V_mask, T)


def dual_contouring_serial(f,
                           f_grad,
                           min_corner,
                           max_corner,
                           nx,
                           ny,
                           nz,
                           constrained=False,
                           triangles=False,
                           root_finding=True):
    return frame_field_utils_bind.dual_contouring_serial(
        f, f_grad, min_corner, max_corner, nx, ny, nz, constrained, triangles,
        root_finding)


class SH4SDPProjectHelper:

    def __init__(self):

        # Reference: Section 4.2 of "Algebraic Representations for Volumetric Frame Fields" by PALMER et al.
        # A coefficients from https://github.com/dpa1mer/arff/blob/master/src/variety/OctaMat.mat

        # SCS compatible format generated using cvxpy
        #
        # import cvxpy as cp
        # Q = cp.Variable((10, 10), symmetric=True)
        # constraints = [Q >> 0]
        # constraints += [cp.trace(A_sdp[i] @ Q) == b_sdp[i] for i in range(16)]
        # prob = cp.Problem(cp.Minimize(cp.trace(Y @ Q)), constraints)
        # data, _, _ = prob.get_problem_data(cp.SCS)

        A_data = np.array([
            1., 240.64225139, 240.64225139, 240.64225139, 120.3211257,
            -240.64225139, -1., 233.00086893, -233.00086893, -1.41421356,
            329.51298889, 329.51298889, -164.75649444, -1.41421356,
            176.13210127, -88.06605064, -264.19815191, -1.41421356, 124.5442032,
            -124.5442032, 62.2721016, -1.41421356, 157.53734059, 157.53734059,
            -39.38433515, -118.15300544, -196.92167573, -157.53734059,
            -1.41421356, 124.5442032, 124.5442032, -186.81630479, -1.41421356,
            176.13210127, -176.13210127, 176.13210127, 176.13210127,
            176.13210127, -1.41421356, 329.51298889, -329.51298889,
            164.75649444, -1.41421356, 233.00086893, -233.00086893,
            233.00086893, -1.41421356, 902.40852733, -1., -638.09918908,
            -1.41421356, 341.07836347, -341.07836347, -341.07836347,
            -341.07836347, -1.41421356, 482.35764745, -482.35764745,
            241.17882373, -1.41421356, -305.06976255, 305.06976255, -1.41421356,
            482.35764745, 482.35764745, -241.17882373, -1.41421356,
            341.07836347, 341.07836347, -1.41421356, 638.09918908, -1.41421356,
            -1.41421356, -225.60213183, -225.60213183, -225.60213183,
            225.60213183, -1., -241.17882373, -241.17882373, -241.17882373,
            -1.41421356, 341.07836347, -341.07836347, 170.53918174,
            170.53918174, -511.61754521, -170.53918174, -1.41421356,
            647.1506935, 647.1506935, -323.57534675, -1.41421356, 341.07836347,
            682.15672694, -341.07836347, -1.41421356, -241.17882373,
            241.17882373, 482.35764745, -1.41421356, -1.41421356, -638.09918908,
            -1.41421356, -257.83100781, -257.83100781, -128.9155039,
            -386.74651171, 128.9155039, -1., -91.15702701, -455.78513505,
            91.15702701, -1.41421356, 922.44425627, 403.56936212, -518.87489415,
            -1.41421356, -455.78513505, 91.15702701, 364.62810804, -1.41421356,
            -386.74651171, 386.74651171, -1.41421356, 241.17882373,
            -241.17882373, -482.35764745, -1.41421356, -341.07836347,
            -341.07836347, -1.41421356, -128.9155039, -902.40852733,
            -32.22887598, -32.22887598, 225.60213183, 32.22887598, -1.,
            -163.06664722, 163.06664722, 122.29998541, -1.41421356,
            -773.49302343, -257.83100781, 515.66201562, -1.41421356,
            91.15702701, -455.78513505, -182.31405402, -1.41421356,
            -341.07836347, -341.07836347, -1.41421356, -482.35764745,
            -482.35764745, 241.17882373, -1.41421356, -618.79441874,
            -618.79441874, -360.96341093, -51.56620156, 257.83100781,
            -25.78310078, -1., -163.06664722, -163.06664722, 40.7666618,
            -1.41421356, 922.44425627, -922.44425627, 864.79149025, 57.65276602,
            57.65276602, 57.65276602, -1.41421356, 647.1506935, -647.1506935,
            323.57534675, -1.41421356, -305.06976255, 305.06976255,
            -305.06976255, -1.41421356, -902.40852733, -128.9155039,
            -676.8063955, -161.14437988, 96.68662793, -96.68662793, -1.,
            -455.78513505, -91.15702701, -91.15702701, -1.41421356,
            341.07836347, -341.07836347, 852.69590868, -511.61754521,
            170.53918174, -170.53918174, -1.41421356, 482.35764745,
            -482.35764745, 241.17882373, -1.41421356, -257.83100781,
            -257.83100781, -515.66201562, -257.83100781, -1., -241.17882373,
            -241.17882373, -241.17882373, -1.41421356, 341.07836347,
            -341.07836347, -341.07836347, -341.07836347, -1.41421356,
            -225.60213183, -225.60213183, -225.60213183, 225.60213183, -1.,
            -638.09918908, -1.41421356, 902.40852733, -1.
        ])

        A_indices = np.array([
            0, 1, 6, 8, 10, 15, 16, 9, 11, 17, 4, 5, 12, 18, 3, 9, 11, 19, 4, 5,
            12, 20, 1, 6, 8, 10, 13, 15, 21, 2, 7, 14, 22, 1, 6, 10, 13, 15, 23,
            2, 7, 14, 24, 8, 10, 13, 25, 15, 26, 14, 27, 8, 10, 13, 15, 28, 2,
            7, 14, 29, 9, 11, 30, 4, 5, 12, 31, 9, 11, 32, 12, 33, 34, 8, 10,
            13, 15, 35, 2, 7, 14, 36, 1, 6, 8, 10, 13, 15, 37, 4, 5, 12, 38, 3,
            9, 11, 39, 4, 5, 12, 40, 41, 12, 42, 1, 6, 8, 10, 13, 43, 2, 7, 14,
            44, 3, 9, 11, 45, 4, 5, 12, 46, 9, 11, 47, 4, 5, 12, 48, 9, 11, 49,
            1, 6, 8, 10, 13, 15, 50, 4, 5, 12, 51, 3, 9, 11, 52, 4, 5, 12, 53,
            3, 11, 54, 4, 5, 12, 55, 1, 6, 8, 10, 13, 15, 56, 2, 7, 14, 57, 1,
            6, 8, 10, 13, 15, 58, 2, 7, 14, 59, 8, 10, 13, 60, 1, 6, 8, 10, 13,
            15, 61, 2, 7, 14, 62, 1, 6, 8, 10, 13, 15, 63, 2, 7, 14, 64, 1, 6,
            8, 13, 65, 2, 7, 14, 66, 8, 10, 13, 15, 67, 8, 10, 13, 15, 68, 14,
            69, 15, 70
        ],
                             dtype=np.int64)

        A_indpter = np.array([
            0, 7, 10, 14, 18, 22, 29, 33, 39, 43, 47, 49, 51, 56, 60, 63, 67,
            70, 72, 73, 78, 82, 89, 93, 97, 101, 102, 104, 110, 114, 118, 122,
            125, 129, 132, 139, 143, 147, 151, 154, 158, 165, 169, 176, 180,
            184, 191, 195, 202, 206, 211, 215, 220, 225, 227, 229
        ],
                             dtype=np.int64)

        b = np.zeros(71, dtype=np.float64)
        b[0] = 1.
        c_base = np.eye(10)[np.triu_indices(10)]

        self.helper = frame_field_utils_bind.SDPHelper(A_data, A_indices,
                                                       A_indpter, b, c_base, 16,
                                                       10)

    def project(self, qs, group_size=1024):
        return self.helper.solve(qs, group_size)
