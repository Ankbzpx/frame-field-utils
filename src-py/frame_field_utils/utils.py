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


def tet_edge_one_ring(T):
    uE, uE_boundary_mask, uE2T, uE2T_cumsum = frame_field_utils_bind.tet_edge_one_ring(
        T)
    return uE, uE_boundary_mask.astype(bool), uE2T, uE2T_cumsum


def tet_edge_singularity(uE, uE_boundary_mask, uE2T, uE2T_cumsum, Rs_tet):
    uE_singularity_mask = frame_field_utils_bind.tet_edge_singularity(
        uE, uE_boundary_mask, uE2T, uE2T_cumsum, Rs_tet.reshape(-1, 9))
    return uE_singularity_mask.astype(bool)
