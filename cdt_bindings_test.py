#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

""" Tests for CDT Python bindings """

import numpy as np
import pytest
import tempfile
import hashlib
import threading

import PythonCDT as cdt


def assert_triangles_array_matches_objects(t: cdt.Triangulation) -> None:
    """triangles_as_array() must agree exactly with the object API, super-triangle included."""
    arr = t.triangles_as_array()
    assert arr.shape == (len(t.triangles),)
    assert arr.dtype.names == ("vertices", "neighbors")
    assert arr["vertices"].shape == (len(t.triangles), 3) and arr["vertices"].dtype == np.uintc
    assert arr["neighbors"].shape == (len(t.triangles), 3) and arr["neighbors"].dtype == np.uintc
    for i, tri in enumerate(t.triangles):
        assert list(tri.vertices) == arr["vertices"][i].tolist(), "triangles_as_array vertices disagree with triangles"
        assert list(tri.neighbors) == arr["neighbors"][i].tolist(), "triangles_as_array neighbors disagree with triangles"


def test_constants() -> None:
    """Test that constants have proper values"""
    assert cdt.NO_NEIGHBOR == np.iinfo(np.uintc).max, "NO_NEIGHBOR constant has wrong value"
    assert cdt.NO_VERTEX == np.iinfo(np.uintc).max, "NO_VERTEX constant has wrong value"


def test_version() -> None:
    """Test that the version is passed in from the build system"""
    assert cdt.__version__ != "dev", "Version was not passed in from the build system"


def test_V2d() -> None:
    """Test 2D vector"""
    p = cdt.V2d(42, 42)
    assert p.x == 42 and p.y == 42, "Error in constructing 2D vector with int"
    p = cdt.V2d(42.0, 42.0)
    assert p.x == 42 and p.y == 42, "Error in constructing 2D vector with float"
    p = cdt.V2d(np.array([42., 42.]))
    assert p.x == 42 and p.y == 42, "Error in constructing 2D vector with buffer protocol"

    assert cdt.V2d(1.23, 2).__repr__() == "V2d(1.23, 2)", "Wrong __repr__ output for V2d"


def test_Edge() -> None:
    """Test Edge class"""
    e = cdt.Edge(1, 2)
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"
    e = cdt.Edge(2, 1)
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"
    e = cdt.Edge(np.array([2, 1], dtype=np.uintc))
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"

    assert cdt.Edge(1, 2).__repr__() == "Edge(1, 2)", "Wrong __repr__ output for Edge"


def test_Triangulation() -> None:
    """Test Triangulation class"""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    assert len(t.vertices) == 0, "Wrong vertex count in empty triangulation"
    assert len(t.triangles) == 0, "Wrong triangle count in empty triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in empty triangulation"
    assert_triangles_array_matches_objects(t)

    vv = [cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)]
    t.insert_vertices(vv)
    assert len(t.vertices) == 7, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 9, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in triangulation"

    ee = [cdt.Edge(0, 2)]
    t.insert_edges(ee)
    assert len(t.fixed_edges) == 1, "Wrong fixed edge count in triangulation"
    assert cdt.Edge(0 + 3, 2 + 3) in t.fixed_edges, "Constraint edge was not properly added"
    assert_triangles_array_matches_objects(t)

    t.erase_super_triangle()
    assert cdt.Edge(0, 2) in t.fixed_edges, "Constraint edge was not properly added"
    assert len(t.vertices) == 4, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 2, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 1, "Wrong fixed edge count in triangulation"
    assert_triangles_array_matches_objects(t)

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
    assert_triangles_array_matches_objects(t)


def test_verify_topology() -> None:
    """Test verifying CDT topology"""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices([cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)])
    t.insert_edges([cdt.Edge(0, 2), cdt.Edge(1, 3)])
    assert cdt.verify_topology(t), "Verifying topology produced wrong result"
    assert_triangles_array_matches_objects(t)


def save_triangulation_as_off(t: cdt.Triangulation, off_file) -> None:
    with open(off_file, "w") as f:
        f.write(f"OFF\n")
        f.write(f"{t.vertices_count()} {t.triangles_count()} 0\n")
        for v in t.vertices_iter():
            f.write(f"{v.x} {v.y} 0\n")
        for tri in t.triangles_iter():
            vv = tri.vertices
            f.write(f"3 {int(vv[0])} {int(vv[1])} {int(vv[2])}\n")


def read_input_file(input_file):
    with open(input_file, "r") as f:
        n_verts, n_edges = (int(s) for s in f.readline().split())
        verts = [cdt.V2d(*(float(s) for s in f.readline().split())) for _ in range(n_verts)]
        edges = [cdt.Edge(*(int(s) for s in f.readline().split())) for _ in range(n_edges)]
        return verts, edges

def md5_checksum(file_path):
    with open(file_path, 'r') as f:
        return hashlib.md5(f.read().encode('utf-8')).hexdigest()

def triangulation_md5_checksum(t: cdt.Triangulation):
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        return md5_checksum(off_file)

def test_triangulate_input_file() -> None:
    vv, ee = read_input_file("CDT/visualizer/data/Constrained Sweden.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.insert_edges(ee)
    t.erase_outer_triangles_and_holes()
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        assert md5_checksum(off_file) == 'db59c00d9dad866781cd96779e5262b7', "Wrong OFF file contents"
    assert_triangles_array_matches_objects(t)


def test_conform_to_edges() -> None:
    vv, ee = read_input_file("CDT/visualizer/data/ditch.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.conform_to_edges(ee)
    t.erase_outer_triangles_and_holes()
    assert triangulation_md5_checksum(t) == 'b64cae39c91a55dd4e23a146eb7df0d3', "Wrong OFF file contents"
    assert_triangles_array_matches_objects(t)


@pytest.mark.parametrize("vv", [[cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)],
                                np.array([[-1, 0], [0, 0.5], [1, 0], [0, -0.5]], dtype=np.float64),
                                np.array([-1, 0, 0, 0.5, 1, 0, 0, -0.5], dtype=np.float64)])
def test_insert_vertices(vv) -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(vv)
    assert len(t.vertices) == 7, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 9, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in triangulation"
    assert triangulation_md5_checksum(t) == 'db9176f4429942862a7a73155fb55322', "Wrong OFF file contents"
    assert_triangles_array_matches_objects(t)


@pytest.mark.parametrize("ee", [[cdt.Edge(0, 1), cdt.Edge(2, 3), cdt.Edge(3, 4), cdt.Edge(5, 6)],
                                np.array([[0, 1], [2, 3], [3, 4], [5, 6]], dtype=np.uintc),
                                np.array([0, 1, 2, 3, 3, 4, 5, 6], dtype=np.uintc)])
def test_insert_conform_edges(ee) -> None:
    # insert edges
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[0, 0], [4, 0], [5, 1], [2, 1], [-1, 1], [0, 2], [4, 2]], dtype=float))
    t.insert_edges(ee)
    assert len(t.vertices) == 10, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 15, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 4, "Wrong fixed edge count in triangulation"
    assert triangulation_md5_checksum(t) == '639c7a1492b2adb8f25464ec81ff6a00', "Wrong OFF file contents"
    assert_triangles_array_matches_objects(t)

    # conform to edges
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[0, 0], [4, 0], [5, 1], [2, 1], [-1, 1], [0, 2], [4, 2]], dtype=float))
    t.conform_to_edges(ee)
    assert len(t.vertices) == 12, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 19, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 6, "Wrong fixed edge count in triangulation"
    assert triangulation_md5_checksum(t) == '9c87b435e247c1658ec0f04af3340dc7', "Wrong OFF file contents"
    assert_triangles_array_matches_objects(t)


def test_triangles_as_array_before_and_after_erase() -> None:
    """Before erase_super_triangle() the array holds the super-triangle's fan and input vertex i
    is index i + 3; after it, the fan is gone and input vertex i is index i."""
    vv, ee = read_input_file("CDT/visualizer/data/ditch.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.insert_edges(ee)

    before = t.triangles_as_array()["vertices"]
    assert (before < 3).any(axis=1).sum() > 0, "unfinalized array must include triangles touching the super-triangle"
    kept = before[(before >= 3).all(axis=1)] - 3

    t.erase_super_triangle()
    after = t.triangles_as_array()["vertices"]
    assert np.array_equal(np.sort(np.sort(kept, axis=1), axis=0), np.sort(np.sort(after, axis=1), axis=0)), \
        "dropping the super-triangle's fan and shifting by 3 must give the erased mesh"
    assert after.max() < len(vv), "after erase every index addresses an input vertex"


def test_triangles_as_array_empty() -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    arr = t.triangles_as_array()
    assert arr.shape == (0,) and arr["vertices"].shape == (0, 3)


def test_triangles_as_array_is_a_copy() -> None:
    """The array is a copy: writing to it must not alter the triangulation."""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    t.insert_vertices(np.array([[-1, 0], [0, 0.5], [1, 0], [0, -0.5]], dtype=np.float64))
    arr = t.triangles_as_array()
    arr["vertices"][:] = 0
    assert all(list(tri.vertices) != [0, 0, 0] for tri in t.triangles), "triangles_as_array must return a copy"


def test_insert_releases_the_gil() -> None:
    """Concurrent builds on threads must each equal the single-threaded build."""
    vv, ee = read_input_file("CDT/visualizer/data/Constrained Sweden.txt")
    verts = np.array([[v.x, v.y] for v in vv], dtype=np.float64)
    edges = np.array([[e.v1, e.v2] for e in ee], dtype=np.uintc)

    def build() -> str:
        t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
        t.insert_vertices(verts)
        t.insert_edges(edges)
        t.erase_super_triangle()
        assert cdt.verify_topology(t)
        return triangulation_md5_checksum(t)

    expected = build()
    results = [None] * 4
    def worker(i):
        results[i] = build()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results == [expected] * 4, "Threaded builds must match the single-threaded build"
