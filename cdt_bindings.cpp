/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

#include <CDT.h>
#include <VerifyTopology.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <sstream>
#include <utility>

namespace py = pybind11;

using coord_t = double;
using V2d = CDT::V2d<coord_t>;
using Triangulation = CDT::Triangulation<coord_t>;

namespace
{

std::string TriInd2str(CDT::TriInd it)
{
    return it != CDT::noNeighbor ? std::to_string(it) : "-";
}

} // namespace

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
            "__eq__",
            [](const V2d& lhs, const V2d& rhs) {
                return lhs.x == rhs.x && lhs.y == rhs.y;
            })
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
        .def_readwrite("neighbors", &CDT::Triangle::neighbors)
        .def(
            "__eq__",
            [](const CDT::Triangle& lhs, const CDT::Triangle& rhs) {
                return std::equal(
                           std::begin(lhs.vertices),
                           std::end(lhs.vertices),
                           std::begin(rhs.vertices)) &&
                       std::equal(
                           std::begin(lhs.neighbors),
                           std::end(lhs.neighbors),
                           std::begin(rhs.neighbors));
            })
        .def("__repr__", [](const CDT::Triangle& tri) {
            std::ostringstream oss;
            const CDT::VerticesArr3 vv = tri.vertices;
            const CDT::VerticesArr3 nn = tri.neighbors;
            oss << "Triangle(vertices(" << vv[0] << ", " << vv[1] << ", "
                << vv[2] << "), neighbors(" << TriInd2str(nn[0]) << ", "
                << TriInd2str(nn[1]) << ", " << TriInd2str(nn[2]) << "))";
            return oss.str();
        });

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
        // vertices
        .def_readonly("vertices", &Triangulation::vertices)
        .def(
            "vertices_count",
            [](const Triangulation& t) { return t.vertices.size(); })
        .def(
            "vertices_iter",
            [](const Triangulation& t) -> py::iterator {
                return py::make_iterator(t.vertices.begin(), t.vertices.end());
            },
            py::keep_alive<0, 1>())
        // triangles
        .def_readonly("triangles", &Triangulation::triangles)
        .def(
            "triangles_count",
            [](const Triangulation& t) { return t.triangles.size(); })
        .def(
            "triangles_iter",
            [](const Triangulation& t) -> py::iterator {
                return py::make_iterator(
                    t.triangles.begin(), t.triangles.end());
            },
            py::keep_alive<0, 1>())
        // fixed edges
        .def_readonly("fixed_edges", &Triangulation::fixedEdges)
        .def(
            "fixed_edges_count",
            [](const Triangulation& t) { return t.fixedEdges.size(); })
        .def(
            "fixed_edges_iter",
            [](const Triangulation& t) -> py::iterator {
                return py::make_iterator(
                    t.fixedEdges.begin(), t.fixedEdges.end());
            },
            py::keep_alive<0, 1>())
        // vertex triangles
        .def_readonly("vertices_triangles", &Triangulation::vertTris)
        .def(
            "vertices_triangles_count",
            [](const Triangulation& t) { return t.vertTris.size(); })
        .def(
            "vertices_triangles_iter",
            [](const Triangulation& t) -> py::iterator {
                return py::make_iterator(t.vertTris.begin(), t.vertTris.end());
            },
            py::keep_alive<0, 1>())
        // overlaps
        .def_readonly("overlap_count", &Triangulation::overlapCount)
        .def(
            "overlap_count_count",
            [](const Triangulation& t) { return t.overlapCount.size(); })
        .def(
            "overlap_count_iter",
            [](const Triangulation& t) -> py::iterator {
                return py::make_iterator(
                    t.overlapCount.begin(), t.overlapCount.end());
            },
            py::keep_alive<0, 1>())
        // piece to originals mapping for edges
        .def_readonly("piece_to_originals", &Triangulation::pieceToOriginals)
        .def(
            "piece_to_originals_count",
            [](const Triangulation& t) { return t.pieceToOriginals.size(); })
        .def(
            "piece_to_originals_iter",
            [](const Triangulation& t) -> py::iterator {
                return py::make_iterator(
                    t.pieceToOriginals.begin(), t.pieceToOriginals.end());
            },
            py::keep_alive<0, 1>())
        // methods
        .def(
            "insert_vertices",
            static_cast<void (Triangulation::*)(const std::vector<V2d>&)>(
                &Triangulation::insertVertices),
            py::arg("vertices"))
        .def(
            "insert_vertices",
            [](Triangulation& t, py::buffer b) {
                const py::buffer_info info = b.request();
                // sanity checks
                if (info.format != py::format_descriptor<coord_t>::format())
                {
                    throw std::runtime_error(
                        "Incompatible format: expected a double array!");
                }
                if (info.ndim != 1 && info.ndim != 2)
                {
                    throw std::runtime_error("Incompatible buffer dimension!");
                }
                if (info.size % 2 != 0)
                {
                    throw std::runtime_error("Buffer must hold even number of "
                                             "coordinates (2 per vertex)!");
                }
                // create from buffer
                struct XY
                {
                    coord_t xy[2];
                };
                const std::size_t n_vert = info.size / 2;
                const XY* const ptr = static_cast<XY*>(info.ptr);
                t.insertVertices(
                    ptr,
                    ptr + n_vert,
                    [](const XY& v) { return v.xy[0]; },
                    [](const XY& v) { return v.xy[1]; });
            },
            py::arg("vertex_buffer"))
        .def(
            "insert_edges",
            static_cast<void (Triangulation::*)(const std::vector<CDT::Edge>&)>(
                &Triangulation::insertEdges),
            py::arg("edges"))
        .def(
            "insert_edges",
            [](Triangulation& t, py::buffer b) {
                const py::buffer_info info = b.request();
                // sanity checks
                if (info.format !=
                    py::format_descriptor<CDT::VertInd>::format())
                {
                    throw std::runtime_error(
                        "Incompatible format: expected a CDT::VertInd array!");
                }
                if (info.ndim != 1 && info.ndim != 2)
                {
                    throw std::runtime_error("Incompatible buffer dimension!");
                }
                if (info.size % 2 != 0)
                {
                    throw std::runtime_error("Buffer must hold even number of "
                                             "CDT::VertInd (2 per edge)!");
                }
                // create from buffer
                struct EdgeData
                {
                    CDT::VertInd vv[2];
                };
                const std::size_t n_vert = info.size / 2;
                const EdgeData* const ptr = static_cast<EdgeData*>(info.ptr);
                t.insertEdges(
                    ptr,
                    ptr + n_vert,
                    [](const EdgeData& e) { return e.vv[0]; },
                    [](const EdgeData& e) { return e.vv[1]; });
            },
            py::arg("edge_buffer"))
        .def(
            "conform_to_edges",
            static_cast<void (Triangulation::*)(const std::vector<CDT::Edge>&)>(
                &Triangulation::conformToEdges),
            py::arg("edges"))
        .def(
            "conform_to_edges",
            [](Triangulation& t, py::buffer b) {
                const py::buffer_info info = b.request();
                // sanity checks
                if (info.format !=
                    py::format_descriptor<CDT::VertInd>::format())
                {
                    throw std::runtime_error(
                        "Incompatible format: expected a CDT::VertInd array!");
                }
                if (info.ndim != 1 && info.ndim != 2)
                {
                    throw std::runtime_error("Incompatible buffer dimension!");
                }
                if (info.size % 2 != 0)
                {
                    throw std::runtime_error("Buffer must hold even number of "
                                             "CDT::VertInd (2 per edge)!");
                }
                // create from buffer
                struct EdgeData
                {
                    CDT::VertInd vv[2];
                };
                const std::size_t n_vert = info.size / 2;
                const EdgeData* const ptr = static_cast<EdgeData*>(info.ptr);
                t.conformToEdges(
                    ptr,
                    ptr + n_vert,
                    [](const EdgeData& e) { return e.vv[0]; },
                    [](const EdgeData& e) { return e.vv[1]; });
            },
            py::arg("edge_buffer"))
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
