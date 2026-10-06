# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for CDT Python bindings"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import pythoncdt as cdt

DATA_DIR = Path(__file__).parents[1] / "CDT" / "visualizer" / "data"


def test_constants() -> None:
    """Test that constants have proper values"""
    assert cdt.NO_NEIGHBOR == np.iinfo(np.uintc).max, "NO_NEIGHBOR constant has wrong value"
    assert cdt.NO_VERTEX == np.iinfo(np.uintc).max, "NO_VERTEX constant has wrong value"


def test_version() -> None:
    """Test that the version is passed in from the build system"""
    assert cdt.__version__, "Version was not passed in from the build system"


def test_V2d() -> None:
    """Test 2D vector"""
    p = cdt.V2d(42, 42)
    assert p.x == 42 and p.y == 42, "Error in constructing 2D vector with int"
    p = cdt.V2d(42.0, 42.0)
    assert p.x == 42 and p.y == 42, "Error in constructing 2D vector with float"
    p = cdt.V2d(np.array([42.0, 42.0]))
    assert p.x == 42 and p.y == 42, "Error in constructing 2D vector with buffer protocol"

    assert repr(cdt.V2d(1.23, 2)) == "V2d(1.23, 2)", "Wrong __repr__ output for V2d"


def test_Edge() -> None:
    """Test Edge class"""
    e = cdt.Edge(1, 2)
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"
    e = cdt.Edge(2, 1)
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"
    e = cdt.Edge(np.array([2, 1], dtype=np.uintc))
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"

    assert repr(cdt.Edge(1, 2)) == "Edge(1, 2)", "Wrong __repr__ output for Edge"

    ee = [cdt.Edge(2, 3), cdt.Edge(0, 5), cdt.Edge(0, 1)]
    assert sorted(ee) == [cdt.Edge(0, 1), cdt.Edge(0, 5), cdt.Edge(2, 3)], "Edges are ordered wrong"


def test_Triangulation() -> None:
    """Test Triangulation class"""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    assert len(t.vertices) == 0, "Wrong vertex count in empty triangulation"
    assert len(t.triangles) == 0, "Wrong triangle count in empty triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in empty triangulation"

    vv = [cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)]
    t.insert_vertices(vv)
    assert len(t.vertices) == 7, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 9, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in triangulation"

    ee = [cdt.Edge(0, 2)]
    t.insert_edges(ee)
    assert len(t.fixed_edges) == 1, "Wrong fixed edge count in triangulation"
    assert cdt.Edge(0 + 3, 2 + 3) in t.fixed_edges, "Constraint edge was not properly added"

    t.erase_super_triangle()
    assert cdt.Edge(0, 2) in t.fixed_edges, "Constraint edge was not properly added"
    assert len(t.vertices) == 4, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 2, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 1, "Wrong fixed edge count in triangulation"

    # properties give references into the triangulation
    vertices = t.vertices
    vertices[3].x = 42
    assert t.vertices[3] == cdt.V2d(42, -0.5), "Vertex property must reference triangulation's vertices"
    vertices[3].x = 0

    # test retrieving triangulation data using iterators
    assert t.vertices_count() == len(t.vertices), "Wrong vertex count"
    assert t.triangles_count() == len(t.triangles), "Wrong triangle count"
    assert t.fixed_edges_count() == len(t.fixed_edges), "Wrong fixed edge count"
    assert t.overlap_count_count() == len(t.overlap_count), "Wrong number of overlap-count"
    assert t.piece_to_originals_count() == len(t.piece_to_originals), "Wrong piece-to-originals count"
    for i, v in enumerate(t.vertices_iter()):
        assert v == t.vertices[i], "Wrong vertex from iterable"
    for i, tri in enumerate(t.triangles_iter()):
        assert tri == t.triangles[i], "Wrong triangle from iterable"
    for fe in t.fixed_edges_iter():
        assert fe in t.fixed_edges, "Wrong fixed edges from iterable"
    for key, val in t.overlap_count_iter():
        assert t.overlap_count[key] == val, "Wrong overlap-count from iterable"
    for key, val in t.piece_to_originals_iter():
        assert t.piece_to_originals[key] == val, "Wrong piece-to-originals from iterable"

    #  Test resolving fixed edge intersections
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    ee = [cdt.Edge(0, 2), cdt.Edge(1, 3)]
    t.insert_vertices(vv)
    t.insert_edges(ee)
    t.erase_super_triangle()
    assert len(t.vertices) == 5, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 4, "Wrong triangle count in triangulation"


def test_verify_topology() -> None:
    """Test verifying CDT topology"""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices([cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)])
    t.insert_edges([cdt.Edge(0, 2), cdt.Edge(1, 3)])
    assert cdt.verify_topology(t), "Verifying topology produced wrong result"


def test_verify_winding() -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices([cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)])
    assert cdt.verify_winding(t), "Verifying winding produced wrong result"

    t.triangles[0].vertices = t.triangles[0].vertices[::-1]
    assert not cdt.verify_winding(t), "Inverted triangle must not pass winding verification"


def test_triangle_geometry() -> None:
    assert cdt.deg_to_rad(180.0) == np.pi, "Wrong degrees to radians conversion"

    a, b, c = cdt.V2d(0, 0), cdt.V2d(1, 0), cdt.V2d(0, 1)
    assert cdt.area(a, b, c) == 0.5, "Wrong triangle area"
    assert cdt.smallest_angle(a, b, c) == pytest.approx(np.pi / 4), "Wrong smallest triangle angle"
    assert cdt.circumcenter(a, b, c) == cdt.V2d(0.5, 0.5), "Wrong triangle circumcenter"


def read_input_file(input_file):
    with open(input_file) as f:
        n_verts, n_edges = (int(s) for s in f.readline().split())
        verts = [cdt.V2d(*(float(s) for s in f.readline().split())) for _ in range(n_verts)]
        edges = [cdt.Edge(*(int(s) for s in f.readline().split())) for _ in range(n_edges)]
        return verts, edges


def triangulation_md5_checksum(t: cdt.Triangulation) -> str:
    lines = ["OFF", f"{t.vertices_count()} {t.triangles_count()} 0"]
    lines += [f"{v.x} {v.y} 0" for v in t.vertices_iter()]
    lines += ["3 {} {} {}".format(*tri.vertices) for tri in t.triangles_iter()]
    return hashlib.md5("".join(f"{line}\n" for line in lines).encode()).hexdigest()


def test_triangulate_input_file() -> None:
    vv, ee = read_input_file(DATA_DIR / "Constrained Sweden.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.insert_edges(ee)
    t.erase_outer_triangles_and_holes()
    assert triangulation_md5_checksum(t) == "db59c00d9dad866781cd96779e5262b7", "Wrong OFF file contents"


def test_conform_to_edges() -> None:
    vv, ee = read_input_file(DATA_DIR / "ditch.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.conform_to_edges(ee)
    t.erase_outer_triangles_and_holes()
    assert triangulation_md5_checksum(t) == "b64cae39c91a55dd4e23a146eb7df0d3", "Wrong OFF file contents"


def triangle_smallest_angles(t: cdt.Triangulation):
    return [cdt.smallest_angle(*(t.vertices[int(i)] for i in tri.vertices)) for tri in t.triangles_iter()]


def triangle_areas(t: cdt.Triangulation):
    return [cdt.area(*(t.vertices[int(i)] for i in tri.vertices)) for tri in t.triangles_iter()]


def triangulation_with_bad_triangles() -> cdt.Triangulation:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]))
    return t


def triangulation_with_sharp_input_corner() -> cdt.Triangulation:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 1.0]]))
    t.insert_edges(np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uintc))
    return t


@pytest.mark.parametrize(
    "criterion, threshold, is_criterion_fulfilled",
    [
        (
            cdt.RefinementCriterion.SMALLEST_ANGLE,
            cdt.deg_to_rad(20.0),
            lambda t, threshold: min(triangle_smallest_angles(t)) >= threshold,
        ),
        (cdt.RefinementCriterion.LARGEST_AREA, 2.0, lambda t, threshold: max(triangle_areas(t)) <= threshold),
    ],
)
def test_refine_triangles_fulfills_criterion(criterion, threshold, is_criterion_fulfilled) -> None:
    vv, ee = read_input_file(DATA_DIR / "Capital A.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(vv)
    t.insert_edges(ee)

    to_erase = t.collect_outer_triangles_and_holes()
    assert to_erase == t.collect_outer_triangles_and_holes(), "Collecting triangles must not change the triangulation"
    unrefined = t.refine_triangles(1000, criterion, threshold, to_erase)
    gave_up = [
        unrefined.short_edge_triangles,
        unrefined.circumcenter_outside,
        unrefined.circumcenter_on_vertex,
        unrefined.sharp_fixed_corner,
        unrefined.short_edges,
        unrefined.split_vertex_invalid,
    ]
    assert not any(gave_up), f"Refinement gave up: {unrefined}"
    assert t.find_encroached_fixed_edges() == [], "Triangulation has encroached fixed edges"

    t.finalize_triangulation(to_erase)
    assert t.find_unrefined_triangles(criterion, threshold) == [], "Kept triangles must all be refined"
    assert cdt.verify_topology(t) and cdt.verify_winding(t), "Refinement broke the triangulation"
    assert is_criterion_fulfilled(t, threshold), "Kept triangles must fulfill the refinement criterion"


def test_refine_triangles_with_zero_threshold_inserts_nothing() -> None:
    t = triangulation_with_bad_triangles()
    n_vertices_before = t.vertices_count()
    t.refine_triangles(1000, cdt.RefinementCriterion.SMALLEST_ANGLE, 0.0)
    assert t.vertices_count() == n_vertices_before, "Any triangle fulfills a zero threshold"


def test_refine_triangles_stops_at_vertex_budget() -> None:
    t = triangulation_with_bad_triangles()
    n_vertices_before = t.vertices_count()
    t.refine_triangles(5, cdt.RefinementCriterion.SMALLEST_ANGLE, cdt.deg_to_rad(30.0))
    assert t.vertices_count() == n_vertices_before + 5, "Refinement must insert exactly the budgeted vertices"


def test_refine_triangles_reports_sharp_input_corner() -> None:
    t = triangulation_with_sharp_input_corner()
    threshold = cdt.deg_to_rad(20.0)
    to_erase = t.collect_outer_triangles()
    unrefined = t.refine_triangles(1000, cdt.RefinementCriterion.SMALLEST_ANGLE, threshold, to_erase)
    assert unrefined.sharp_fixed_corner > 0, f"Sharp input corner was not reported: {unrefined}"
    assert len(t.find_unrefined_triangles(cdt.RefinementCriterion.SMALLEST_ANGLE, threshold)) == 1, (
        "Triangle left sharp by the input must be found"
    )

    t.finalize_triangulation(to_erase)
    assert t.triangles_count() == 1, "Only the sharp input triangle must be left"


def test_refine_triangles_rejects_to_erase_that_is_not_a_set() -> None:
    t = triangulation_with_bad_triangles()
    with pytest.raises(TypeError):
        t.refine_triangles(10, to_erase=[0, 1])


def test_refining_and_collecting_fail_on_finalized_triangulation() -> None:
    t = triangulation_with_bad_triangles()
    t.erase_super_triangle()
    assert t.is_finalized(), "Triangulation must be finalized"
    for call in [
        lambda: t.refine_triangles(10),
        t.find_encroached_fixed_edges,
        t.collect_super_triangle,
        t.collect_outer_triangles,
        t.collect_outer_triangles_and_holes,
        lambda: t.finalize_triangulation(set()),
    ]:
        with pytest.raises(RuntimeError):
            call()


@pytest.mark.parametrize(
    "vv",
    [
        [cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)],
        np.array([[-1, 0], [0, 0.5], [1, 0], [0, -0.5]], dtype=np.float64),
        np.array([-1, 0, 0, 0.5, 1, 0, 0, -0.5], dtype=np.float64),
    ],
)
def test_insert_vertices(vv) -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(vv)
    assert len(t.vertices) == 7, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 9, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in triangulation"
    assert triangulation_md5_checksum(t) == "db9176f4429942862a7a73155fb55322", "Wrong OFF file contents"


@pytest.mark.parametrize(
    "ee",
    [
        [cdt.Edge(0, 1), cdt.Edge(2, 3), cdt.Edge(3, 4), cdt.Edge(5, 6)],
        np.array([[0, 1], [2, 3], [3, 4], [5, 6]], dtype=np.uintc),
        np.array([0, 1, 2, 3, 3, 4, 5, 6], dtype=np.uintc),
    ],
)
def test_insert_conform_edges(ee) -> None:
    # insert edges
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[0, 0], [4, 0], [5, 1], [2, 1], [-1, 1], [0, 2], [4, 2]], dtype=float))
    t.insert_edges(ee)
    assert len(t.vertices) == 10, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 15, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 4, "Wrong fixed edge count in triangulation"
    assert triangulation_md5_checksum(t) == "639c7a1492b2adb8f25464ec81ff6a00", "Wrong OFF file contents"

    # conform to edges
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[0, 0], [4, 0], [5, 1], [2, 1], [-1, 1], [0, 2], [4, 2]], dtype=float))
    t.conform_to_edges(ee)
    assert len(t.vertices) == 12, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 19, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 6, "Wrong fixed edge count in triangulation"
    assert triangulation_md5_checksum(t) == "9c87b435e247c1658ec0f04af3340dc7", "Wrong OFF file contents"


@pytest.mark.parametrize("copy", [True, False])
def test_arrays(copy) -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    assert t.vertices_array(copy=copy).shape == t.triangles_array(copy=copy).shape == (0,), (
        "Empty triangulation must give empty arrays"
    )

    t.insert_vertices([cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)])
    vertices = t.vertices_array(copy=copy)
    triangles = t.triangles_array(copy=copy)
    assert vertices.tolist() == [(v.x, v.y) for v in t.vertices]
    assert triangles["vertices"].tolist() == [list(tri.vertices) for tri in t.triangles]
    assert triangles["neighbors"].tolist() == [list(tri.neighbors) for tri in t.triangles]

    expected = [vertices.copy(), triangles.copy()]
    del t
    for arr, before in zip([vertices, triangles], expected, strict=True):
        assert arr.flags.owndata == arr.flags.writeable == copy, "Copy must be owned and writeable, view neither"
        assert np.array_equal(arr, before), "Array must stay valid after the triangulation is deleted"


def test_shared_triangulation() -> None:
    """Threads can insert into and read from one triangulation concurrently"""
    vertices = np.random.default_rng(0).random((160_000, 2))
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)

    def insert_and_read(batch) -> None:
        t.insert_vertices(batch)
        assert t.triangles_array()["vertices"].max() < t.vertices_count()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(insert_and_read, np.array_split(vertices, 8)))
    assert t.vertices_count() == len(vertices) + 3, "All vertices plus the super-triangle must be inserted"
    assert cdt.verify_topology(t)


def test_insert_buffers_must_be_contiguous_pairs() -> None:
    """Buffer overloads reject buffers that aren't a C-contiguous run of pairs"""
    vertices = np.array([[0, 0], [9, 9], [1, 0], [9, 9], [0, 1], [9, 9], [1, 1], [9, 9]], dtype=np.float64)
    edges = np.array([[0, 1], [9, 9], [1, 3], [9, 9]], dtype=np.uintc)

    def triangulation():
        return cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)

    for bad in [vertices[::2], vertices[:, ::-1], np.asfortranarray(vertices[::2]), np.zeros((2, 3))]:
        with pytest.raises(RuntimeError):
            triangulation().insert_vertices(bad)
    for insert in [cdt.Triangulation.insert_edges, cdt.Triangulation.conform_to_edges]:
        t = triangulation()
        t.insert_vertices(vertices[::2].copy())
        with pytest.raises(RuntimeError):
            insert(t, edges[::2])

    for good in [vertices[::2].copy(), vertices[::2].ravel()]:
        t = triangulation()
        t.insert_vertices(good)
        assert [(v.x, v.y) for v in t.vertices][3:] == [(0, 0), (1, 0), (0, 1), (1, 1)], "Wrong vertices inserted"
    t.insert_edges(edges[::2].copy())
    assert t.fixed_edges == {cdt.Edge(3, 4), cdt.Edge(4, 6)}, "Wrong edges inserted"
