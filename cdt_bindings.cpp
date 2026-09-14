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
#include <mutex>
#include <string>
#include <sstream>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using coord_t = double;
using NearPointLocator_t = CDT::LocatorKDTree<coord_t>;
using V2d = CDT::V2d<coord_t>;
using CdtTriangulation = CDT::Triangulation<coord_t, NearPointLocator_t>;

namespace
{

// CDT triangulation shared by Python threads, reachable only through lock()
class Triangulation
{
public:
    Triangulation(
        CDT::VertexInsertionOrder::Enum vertexInsertionOrder,
        CDT::IntersectingConstraintEdges::Enum intersectingEdgesStrategy,
        coord_t minDistToConstraintEdge)
        : m_cdt(
              vertexInsertionOrder,
              intersectingEdgesStrategy,
              minDistToConstraintEdge)
    {}

    // Waits without the GIL, since the thread holding the mutex may need it
    std::pair<CdtTriangulation&, std::unique_lock<std::mutex>> lock()
    {
        py::gil_scoped_release release;
        return {m_cdt, std::unique_lock<std::mutex>(m_mutex)};
    }

private:
    CdtTriangulation m_cdt;
    std::mutex m_mutex;
};

// Binds a container member as property `name`, `name_count()` and `name_iter()`
template <typename Container>
void def_container(
    py::class_<Triangulation>& cls,
    const std::string& name,
    Container CdtTriangulation::*member)
{
    cls.def_property_readonly(
           name.c_str(),
           [member](Triangulation& t) {
               auto [cdt, lock] = t.lock();
               return py::cast(
                   cdt.*member,
                   py::return_value_policy::reference_internal,
                   py::cast(&t));
           })
        .def(
            (name + "_count").c_str(),
            [member](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                return (cdt.*member).size();
            })
        .def(
            (name + "_iter").c_str(),
            [member](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                const Container& container = cdt.*member;
                return py::make_iterator(container.begin(), container.end());
            },
            py::keep_alive<0, 1>());
}

// Binds a vector member as `name_array(*, copy=True)` returning a numpy array
template <typename T>
void def_array(
    py::class_<Triangulation>& cls,
    const std::string& name,
    std::vector<T> CdtTriangulation::*member)
{
    cls.def(
        (name + "_array").c_str(),
        [member](Triangulation& t, bool copy) {
            auto [cdt, lock] = t.lock();
            const std::vector<T>& items = cdt.*member;
            const auto size = static_cast<py::ssize_t>(items.size());
            if (copy)
                return py::array_t<T>(size, items.data());
            // the view's base is the Python wrapper of t, keeping t alive
            py::array_t<T> view(size, items.data(), py::cast(&t));
            view.attr("setflags")(py::arg("write") = false);
            return view;
        },
        py::kw_only(),
        py::arg("copy") = true,
        "Copy as a numpy structured array. copy=False returns a read-only view "
        "of the triangulation's memory instead; it is invalidated by any call "
        "that modifies the triangulation.");
}

template <typename T>
struct BufferPair
{
    T v[2];
};

// Reads a C-contiguous buffer of shape (2N,) or (N, 2) as N pairs
template <typename T>
std::pair<const BufferPair<T>*, std::size_t> buffer_pairs(
    const py::buffer_info& info,
    const std::string& type_name,
    const std::string& item_name)
{
    if (info.format != py::format_descriptor<T>::format())
    {
        throw std::runtime_error(
            "Incompatible format: expected a " + type_name + " array!");
    }
    if (info.ndim != 1 && info.ndim != 2)
    {
        throw std::runtime_error("Incompatible buffer dimension!");
    }
    if (info.ndim == 2 ? info.shape[1] != 2 : info.shape[0] % 2 != 0)
    {
        throw std::runtime_error(
            "Buffer must hold " + type_name + " pairs (2 per " + item_name +
            "): shape (N, 2) or (2N,)!");
    }
    // strides of dimensions with a single element are irrelevant
    py::ssize_t expected_stride = info.itemsize;
    for (py::ssize_t d = info.ndim - 1; d >= 0; --d)
    {
        if (info.shape[d] > 1 && info.strides[d] != expected_stride)
        {
            throw std::runtime_error(
                "Buffer must be C-contiguous: use numpy.ascontiguousarray!");
        }
        expected_stride *= info.shape[d];
    }
    return {static_cast<const BufferPair<T>*>(info.ptr), info.size / 2};
}

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
        .value("AUTO", CDT::VertexInsertionOrder::Auto)
        .value("AS_PROVIDED", CDT::VertexInsertionOrder::AsProvided);

    py::enum_<CDT::IntersectingConstraintEdges::Enum>(
        m, "IntersectingConstraintEdges")
        .value("NOT_ALLOWED", CDT::IntersectingConstraintEdges::NotAllowed)
        .value("TRY_RESOLVE", CDT::IntersectingConstraintEdges::TryResolve)
        .value("DONT_CHECK", CDT::IntersectingConstraintEdges::DontCheck);

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

    py::class_<Triangulation> triangulation(m, "Triangulation");
    triangulation.def(
        py::init<
            CDT::VertexInsertionOrder::Enum,
            CDT::IntersectingConstraintEdges::Enum,
            coord_t>(),
        py::arg("vertex_insertion_order"),
        py::arg("intersecting_edges_strategy"),
        py::arg("min_dist_to_constraint_edge"));
    def_container(triangulation, "vertices", &CdtTriangulation::vertices);
    def_container(triangulation, "triangles", &CdtTriangulation::triangles);
    def_container(triangulation, "fixed_edges", &CdtTriangulation::fixedEdges);
    def_container(
        triangulation, "overlap_count", &CdtTriangulation::overlapCount);
    def_container(
        triangulation,
        "piece_to_originals",
        &CdtTriangulation::pieceToOriginals);
    def_array(triangulation, "vertices", &CdtTriangulation::vertices);
    def_array(triangulation, "triangles", &CdtTriangulation::triangles);
    triangulation
        // methods
        .def(
            "insert_vertices",
            [](Triangulation& t, const std::vector<V2d>& vertices) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.insertVertices(vertices);
            },
            py::arg("vertices"))
        .def(
            "insert_vertices",
            [](Triangulation& t, py::buffer b) {
                const py::buffer_info info = b.request();
                const auto [ptr, n_vert] =
                    buffer_pairs<coord_t>(info, "double", "vertex");
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.insertVertices(
                    ptr,
                    ptr + n_vert,
                    [](const BufferPair<coord_t>& v) { return v.v[0]; },
                    [](const BufferPair<coord_t>& v) { return v.v[1]; });
            },
            py::arg("vertex_buffer"))
        .def(
            "insert_edges",
            [](Triangulation& t, const std::vector<CDT::Edge>& edges) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.insertEdges(edges);
            },
            py::arg("edges"))
        .def(
            "insert_edges",
            [](Triangulation& t, py::buffer b) {
                const py::buffer_info info = b.request();
                const auto [ptr, n_edges] =
                    buffer_pairs<CDT::VertInd>(info, "CDT::VertInd", "edge");
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.insertEdges(
                    ptr,
                    ptr + n_edges,
                    [](const BufferPair<CDT::VertInd>& e) { return e.v[0]; },
                    [](const BufferPair<CDT::VertInd>& e) { return e.v[1]; });
            },
            py::arg("edge_buffer"))
        .def(
            "conform_to_edges",
            [](Triangulation& t, const std::vector<CDT::Edge>& edges) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.conformToEdges(edges);
            },
            py::arg("edges"))
        .def(
            "conform_to_edges",
            [](Triangulation& t, py::buffer b) {
                const py::buffer_info info = b.request();
                const auto [ptr, n_edges] =
                    buffer_pairs<CDT::VertInd>(info, "CDT::VertInd", "edge");
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.conformToEdges(
                    ptr,
                    ptr + n_edges,
                    [](const BufferPair<CDT::VertInd>& e) { return e.v[0]; },
                    [](const BufferPair<CDT::VertInd>& e) { return e.v[1]; });
            },
            py::arg("edge_buffer"))
        .def(
            "erase_super_triangle",
            [](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.eraseSuperTriangle();
            })
        .def(
            "erase_outer_triangles",
            [](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.eraseOuterTriangles();
            })
        .def(
            "erase_outer_triangles_and_holes",
            [](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.eraseOuterTrianglesAndHoles();
            })
        .def(
            "is_finalized",
            [](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                return cdt.isFinalized();
            })
        .def(
            "calculate_triangle_depths",
            [](Triangulation& t) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                return cdt.calculateTriangleDepths();
            })
        .def(
            "remove_triangles",
            [](Triangulation& t, const CDT::TriIndUSet& triangle_indices) {
                auto [cdt, lock] = t.lock();
                py::gil_scoped_release release;
                cdt.removeTriangles(triangle_indices);
            },
            py::arg("triangle_indices"));

    m.def(
        "verify_topology",
        [](Triangulation& t) {
            auto [cdt, lock] = t.lock();
            py::gil_scoped_release release;
            return CDT::verifyTopology(cdt);
        },
        py::arg("triangulation"));
}
