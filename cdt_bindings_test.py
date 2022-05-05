#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

""" Tests for CDT Python bindings """

import numpy as np
import pytest
import tempfile
import hashlib

import PythonCDT as cdt


def test_constants() -> None:
    """Test that constants have proper values"""
    assert cdt.NO_NEIGHBOR == np.iinfo(np.uint32).max, "NO_NEIGHBOR constant has wrong value"
    assert cdt.NO_VERTEX == np.iinfo(np.uint32).max, "NO_VERTEX constant has wrong value"


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
    e = cdt.Edge(np.array([2, 1], dtype=np.uint32))
    assert e.v1 == 1 and e.v2 == 2, "Constructed wrong edge"

    assert cdt.Edge(1, 2).__repr__() == "Edge(1, 2)", "Wrong __repr__ output for Edge"


def test_Triangulation() -> None:
    """Test Triangulation class"""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.IGNORE, 0.0)
    assert len(t.vertices) == 0, "Wrong vertex count in empty triangulation"
    assert len(t.triangles) == 0, "Wrong triangle count in empty triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in empty triangulation"
    assert len(t.vertices_triangles) == 0, "Wrong vertices triangles count in empty triangulation"

    vv = [cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)]
    t.insert_vertices(vv)
    assert len(t.vertices) == 7, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 9, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in triangulation"
    assert t.vertices_triangles, "Wrong vertices triangles count in triangulation"

    ee = [cdt.Edge(0, 2)]
    t.insert_edges(ee)
    assert len(t.fixed_edges) == 1, "Wrong fixed edge count in triangulation"
    assert cdt.Edge(0 + 3, 2 + 3) in t.fixed_edges, "Constraint edge was not properly added"

    t.erase_super_triangle()
    assert cdt.Edge(0, 2) in t.fixed_edges, "Constraint edge was not properly added"
    assert len(t.vertices) == 4, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 2, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 1, "Wrong fixed edge count in triangulation"
    assert t.vertices_triangles, "Wrong vertices triangles count in triangulation"

    # test retrieving triangulation data using iterators
    assert t.vertices_count() == len(t.vertices), "Wrong vertex count"
    assert t.triangles_count() == len(t.triangles), "Wrong triangle count"
    assert t.fixed_edges_count() == len(t.fixed_edges), "Wrong fixed edge count"
    assert t.vertices_triangles_count() == len(t.vertices_triangles), "Wrong vertices' triangles count"
    assert t.overlap_count_count() == len(t.overlap_count), "Wrong number of overlap-count"
    assert t.piece_to_originals_count() == len(t.piece_to_originals), "Wrong piece-to-originals count"
    for i, v in enumerate(t.vertices_iter()):
        assert v == t.vertices[i], "Wrong vertex from iterable"
    for i, tri in enumerate(t.triangles_iter()):
        assert tri == t.triangles[i], "Wrong triangle from iterable"
    for fe in t.fixed_edges_iter():
        assert fe in t.fixed_edges, "Wrong fixed edges from iterable"
    for i, vt in enumerate(t.vertices_triangles_iter()):
        assert vt == t.vertices_triangles[i], "Wrong vertices' triangles from iterable"
    for key, val in t.overlap_count_iter():
        assert t.overlap_count[key] == val, "Wrong overlap-count from iterable"
    for key, val in t.piece_to_originals_iter():
        assert t.piece_to_originals[key] == val, "Wrong piece-to-originals from iterable"

    #  Test resolving fixed edge intersections
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.RESOLVE, 0.0)
    ee = [cdt.Edge(0, 2), cdt.Edge(1, 3)]
    t.insert_vertices(vv)
    t.insert_edges(ee)
    t.erase_super_triangle()
    assert len(t.vertices) == 5, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 4, "Wrong triangle count in triangulation"


def test_verify_topology() -> None:
    """Test verifying CDT topology"""
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.RESOLVE, 0.0)
    t.insert_vertices([cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)])
    t.insert_edges([cdt.Edge(0, 2), cdt.Edge(1, 3)])
    assert cdt.verify_topology(t), "Verifying topology produced wrong result"


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


def test_triangulate_input_file() -> None:
    vv, ee = read_input_file("CDT/visualizer/data/Constrained Sweden.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.insert_edges(ee)
    t.erase_outer_triangles_and_holes()
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        with open(off_file, 'rb') as f:
            assert hashlib.md5(f.read()).hexdigest() == '609d662d6628942c7cc7558f9d5ee952', "Wrong OFF file contents"


def test_conform_to_edges() -> None:
    vv, ee = read_input_file("CDT/visualizer/data/ditch.txt")
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.RESOLVE, 0.0)
    t.insert_vertices(vv)
    t.conform_to_edges(ee)
    t.erase_outer_triangles_and_holes()
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        with open(off_file, 'rb') as f:
            assert hashlib.md5(f.read()).hexdigest() == 'df2503c614e2f98656038948b355b27e', "Wrong OFF file contents"


@pytest.mark.parametrize("vv", [[cdt.V2d(-1, 0), cdt.V2d(0, 0.5), cdt.V2d(1, 0), cdt.V2d(0, -0.5)],
                                np.array([[-1, 0], [0, 0.5], [1, 0], [0, -0.5]], dtype=np.float64),
                                np.array([-1, 0, 0, 0.5, 1, 0, 0, -0.5], dtype=np.float64)])
def test_insert_vertices(vv) -> None:
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.IGNORE, 0.0)
    t.insert_vertices(vv)
    assert len(t.vertices) == 7, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 9, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 0, "Wrong fixed edge count in triangulation"
    assert t.vertices_triangles, "Wrong vertices triangles count in triangulation"
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        with open(off_file, 'rb') as f:
            assert hashlib.md5(f.read()).hexdigest() == 'c424c4f2691dc3b9aabd39dcf2e17c53', "Wrong OFF file contents"


@pytest.mark.parametrize("ee", [[cdt.Edge(0, 1), cdt.Edge(2, 3), cdt.Edge(3, 4), cdt.Edge(5, 6)],
                                np.array([[0, 1], [2, 3], [3, 4], [5, 6]], dtype=np.uint32),
                                np.array([0, 1, 2, 3, 3, 4, 5, 6], dtype=np.uint32)])
def test_insert_conform_edges(ee) -> None:
    # insert edges
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.IGNORE, 0.0)
    t.insert_vertices(np.array([[0, 0], [4, 0], [5, 1], [2, 1], [-1, 1], [0, 2], [4, 2]], dtype=float))
    t.insert_edges(ee)
    assert len(t.vertices) == 10, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 15, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 4, "Wrong fixed edge count in triangulation"
    assert t.vertices_triangles, "Wrong vertices triangles count in triangulation"
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        with open(off_file, 'rb') as f:
            assert hashlib.md5(f.read()).hexdigest() == '8424ba2c8f8ebabe1bea4141464a347b', "Wrong OFF file contents"
    # conform to edges
    t = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED, cdt.IntersectingConstraintEdges.IGNORE, 0.0)
    t.insert_vertices(np.array([[0, 0], [4, 0], [5, 1], [2, 1], [-1, 1], [0, 2], [4, 2]], dtype=float))
    t.conform_to_edges(ee)
    assert len(t.vertices) == 12, "Wrong vertex count in triangulation"
    assert len(t.triangles) == 19, "Wrong triangle count in triangulation"
    assert len(t.fixed_edges) == 6, "Wrong fixed edge count in triangulation"
    assert t.vertices_triangles, "Wrong vertices triangles count in triangulation"
    save_triangulation_as_off(t, "/tmp/cdt.off")
    with tempfile.TemporaryDirectory() as tmp_dir:
        off_file = f"{tmp_dir}/cdt.off"
        save_triangulation_as_off(t, off_file)
        with open(off_file, 'rb') as f:
            assert hashlib.md5(f.read()).hexdigest() == '9cb9dbaca4943ff0e3aab6c1d31f5a35', "Wrong OFF file contents"
