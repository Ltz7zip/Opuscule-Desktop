// bindings.cpp — fws_native 模块入口
#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_register_climate(py::module_ &);
void py_register_hydro(py::module_ &);
void py_register_hydro_adv(py::module_ &);
void py_register_temperature(py::module_ &);

PYBIND11_MODULE(fws_native, m) {
    m.doc() = "Fantasy World Simulator native compute kernels (C++)";
    m.def("ping", []() { return std::string("pong"); });
    py_register_climate(m);
    py_register_hydro(m);
    py_register_hydro_adv(m);
    py_register_temperature(m);
}