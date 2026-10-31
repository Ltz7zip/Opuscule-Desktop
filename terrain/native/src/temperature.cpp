// temperature.cpp — 气温模块内核 (与 terrain/modules/temperature_module.py 的 Numba 实现逐位对齐)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;
using py::array_t;
using py::array;

// ---------------------------------------------------------------------------
// _ocean_run_lengths_njit: 东西向海洋游程长度 (sea (S,W) bool -> (run_e, run_w) int32)
//   2 遍扫描处理全球经度环绕
// ---------------------------------------------------------------------------
py::tuple ocean_run_lengths(array_t<bool, py::array::forcecast> sea_py,
                            int64_t S, int64_t W) {
    if (sea_py.ndim() != 2)
        throw std::runtime_error("ocean_run_lengths: sea must be 2D");
    auto sea = sea_py.unchecked<2>();
    py::array_t<int32_t> run_e_py({(py::ssize_t)S, (py::ssize_t)W});
    py::array_t<int32_t> run_w_py({(py::ssize_t)S, (py::ssize_t)W});
    auto run_e = run_e_py.mutable_unchecked<2>();
    auto run_w = run_w_py.mutable_unchecked<2>();

    for (int64_t i = 0; i < S; ++i) {
        // East pass: j 从 W-1 递减到 0, 连续 2 遍
        int64_t run = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t j = W - 1; j >= 0; --j) {
                if (sea(i, j))
                    ++run;
                else
                    run = 0;
                run_e(i, j) = (int32_t)run;
            }
        }
        // West pass: j 从 0 到 W-1
        run = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t j = 0; j < W; ++j) {
                if (sea(i, j))
                    ++run;
                else
                    run = 0;
                run_w(i, j) = (int32_t)run;
            }
        }
    }
    return py::make_tuple(run_e_py, run_w_py);
}

void py_register_temperature(py::module_ &m) {
    m.def("ocean_run_lengths", &ocean_run_lengths,
          py::arg("sea"), py::arg("S"), py::arg("W"));
}