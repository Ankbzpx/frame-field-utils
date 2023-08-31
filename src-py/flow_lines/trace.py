import flow_lines_bind


def trace(V,
          F,
          VN,
          Q,
          n_lines,
          n_steps=20,
          line_length=1.5e-1,
          line_width=5e-3,
          line_offset=3e-3):
    return flow_lines_bind.trace(V, F, VN, Q, n_lines, n_steps, line_length,
                                 line_width, line_offset)
