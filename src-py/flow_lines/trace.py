import flow_lines_bind


def trace(V,
          F,
          VN,
          Q,
          n_lines,
          length_factor=20,
          interval_factor=1,
          offset_factor=0.1,
          width_factor=0.1):
    return flow_lines_bind.trace(V, F, VN, Q, n_lines, length_factor,
                                 interval_factor, offset_factor, width_factor)
