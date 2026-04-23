#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lane_postprocess.h"

namespace py = pybind11;

PYBIND11_MODULE(lane_cpp, m) {
    m.def("pred2coords_cpp", &pred2coords_cpp,
          py::arg("loc_row"), py::arg("loc_col"),
          py::arg("exist_row"), py::arg("exist_col"),
          py::arg("shape_row"), py::arg("shape_col"),
          py::arg("img_w"), py::arg("img_h"),
          py::arg("row_anchor"), py::arg("col_anchor"),
          py::arg("local_width") = 1,
          "Convert model outputs to lane coordinates (C++ backend)");
}

// cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")