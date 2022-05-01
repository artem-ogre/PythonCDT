#include <CDT.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace CDT;
namespace py = pybind11;

PYBIND11_MODULE(PythonCDT, m)
{
    // clang-format off
    m.doc() = R"pbdoc(
        PythonCDT module: python bindings for CDT:
        Constrained Delaunay Triangulation
        -----------------------
        .. currentmodule:: PythonCDT
        .. autosummary::
           :toctree: _generate
    )pbdoc";
    // clang-format on

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
