
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

py::list trace_flow_lines(Eigen::Ref<const RowMatrixXd> V,
                          Eigen::Ref<const RowMatrixXi> F,
                          Eigen::Ref<const RowMatrixXd> VN,
                          Eigen::Ref<const RowMatrixXd> Q) {
  std::cout << "trace_flow_lines" << std::endl;

  py::list return_list;
  return return_list;
}

PYBIND11_MODULE(flow_lines, m) {
  m.doc() = "Trace flow lines";
  m.def("trace", &trace_flow_lines,
        py::return_value_policy::reference_internal, "Sample occlusions");
}
