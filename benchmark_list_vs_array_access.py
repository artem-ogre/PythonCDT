"""Compares reading a triangulation as Python lists vs numpy arrays"""

import timeit

import numpy as np

import PythonCDT as cdt

t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
t.insert_vertices(np.random.default_rng(0).random((200_000, 2)))

print(f"{'':<10}{'count':>9}{'list':>12}{'copy':>12}{'view':>12}")
for name in ["vertices", "triangles"]:
    array = getattr(t, f"{name}_array")
    ways = [lambda: getattr(t, name), array, lambda: array(copy=False)]
    ms = [1e3 * min(timeit.repeat(way, number=1, repeat=5)) for way in ways]
    print(f"{name:<10}{len(array()):>9}" + "".join(f"{m:>9.3f} ms" for m in ms))
