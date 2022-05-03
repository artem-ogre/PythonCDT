/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#include <CDT.h>
#include <VerifyTopology.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <sstream>

namespace py = pybind11;

using coord_t = double;
using V2d = CDT::V2d<coord_t>;
using Triangulation = CDT::Triangulation<coord_t>;

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

    m.attr("NO_NEIGHBOR") = py::int_(CDT::noNeighbor);
    m.attr("NO_VERTEX") = py::int_(CDT::noVertex);

    py::enum_<CDT::VertexInsertionOrder::Enum>(m, "VertexInsertionOrder")
        .value("RANDOMIZED", CDT::VertexInsertionOrder::Randomized)
        .value("AS_PROVIDED", CDT::VertexInsertionOrder::AsProvided);

    py::enum_<CDT::IntersectingConstraintEdges::Enum>(
        m, "IntersectingConstraintEdges")
        .value("IGNORE", CDT::IntersectingConstraintEdges::Ignore)
        .value("RESOLVE", CDT::IntersectingConstraintEdges::Resolve);

    py::class_<V2d>(m, "V2d", py::buffer_protocol())
        .def(py::init<int, int>(), py::arg("x"), py::arg("y"))
        .def(py::init<coord_t, coord_t>(), py::arg("x"), py::arg("y"))
        .def(py::init([](py::buffer b) {
            // Request a buffer descriptor from Python
            py::buffer_info info = b.request();
            // Some sanity checks ...
            if (info.format != py::format_descriptor<coord_t>::format())
                throw std::runtime_error(
                    "Incompatible format: expected a double array!");
            if (info.ndim != 1)
                throw std::runtime_error("Incompatible buffer dimension!");
            // create from buffer
            const coord_t* const ptr = static_cast<coord_t*>(info.ptr);
            return V2d{ptr[0], ptr[1]};
        }))
        .def_readwrite("x", &V2d::x)
        .def_readwrite("y", &V2d::y)
        .def(
            "__repr__",
            [](const V2d& v) {
                std::ostringstream oss;
                oss << "V2d(" << v.x << ", " << v.y << ")";
                return oss.str();
            })
        .def_buffer([](V2d& v) -> py::buffer_info {
            return py::buffer_info(
                &v,
                sizeof(coord_t),
                py::format_descriptor<coord_t>::format(),
                1,
                {2},
                {sizeof(coord_t) * 2});
        });
    PYBIND11_NUMPY_DTYPE(V2d, x, y);

    PYBIND11_NUMPY_DTYPE(CDT::Triangle, vertices, neighbors);
    py::class_<CDT::Triangle>(m, "Triangle")
        .def_readwrite("vertices", &CDT::Triangle::vertices)
        .def_readwrite("neighbors", &CDT::Triangle::neighbors);

    py::class_<CDT::Edge>(m, "Edge", py::buffer_protocol())
        .def(
            py::init<CDT::VertInd, CDT::VertInd>(),
            py::arg("index_vert_a"),
            py::arg("index_vert_b"))
        .def(py::init([](py::buffer b) {
            // Request a buffer descriptor from Python
            py::buffer_info info = b.request();
            // Some sanity checks ...
            if (info.format != py::format_descriptor<CDT::VertInd>::format())
                throw std::runtime_error(
                    "Incompatible format: expected a CDT::VertInd array!");
            if (info.ndim != 1)
                throw std::runtime_error("Incompatible buffer dimension!");
            // create from buffer
            const CDT::VertInd* const ptr =
                static_cast<CDT::VertInd*>(info.ptr);
            return CDT::Edge(ptr[0], ptr[1]);
        }))
        .def_buffer([](CDT::Edge& e) -> py::buffer_info {
            return py::buffer_info(
                &e,
                sizeof(CDT::VertInd),
                py::format_descriptor<coord_t>::format(),
                1,
                {2},
                {sizeof(CDT::VertInd) * 2});
        })
        .def_property_readonly("v1", &CDT::Edge::v1)
        .def_property_readonly("v2", &CDT::Edge::v2)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::hash(py::self))
        .def("__repr__", [](const CDT::Edge& e) {
            std::ostringstream oss;
            oss << "Edge(" << e.v1() << ", " << e.v2() << ")";
            return oss.str();
        });

    py::class_<Triangulation>(m, "Triangulation")
        .def(
            py::init<
                CDT::VertexInsertionOrder::Enum,
                CDT::IntersectingConstraintEdges::Enum,
                coord_t>(),
            py::arg("vertex_insertion_order"),
            py::arg("intersecting_edges_strategy"),
            py::arg("min_dist_to_constraint_edge"))
        .def_readonly("vertices", &Triangulation::vertices)
        .def_readonly("triangles", &Triangulation::triangles)
        .def_readonly("fixed_edges", &Triangulation::fixedEdges)
        .def_readonly("vertices_triangles", &Triangulation::vertTris)
        .def(
            "insert_vertices",
            static_cast<void (Triangulation::*)(const std::vector<V2d>&)>(
                &Triangulation::insertVertices),
            py::arg("vertices"))
        .def(
            "insert_edges",
            static_cast<void (Triangulation::*)(const std::vector<CDT::Edge>&)>(
                &Triangulation::insertEdges),
            py::arg("edges"))
        .def("erase_super_triangle", &Triangulation::eraseSuperTriangle)
        .def("erase_outer_triangles", &Triangulation::eraseOuterTriangles)
        .def(
            "erase_outer_triangles_and_holes",
            &Triangulation::eraseOuterTrianglesAndHoles);

    m.def(
        "verify_topology",
        &CDT::verifyTopology<coord_t>,
        py::arg("triangulation"));
}
