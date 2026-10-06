# PythonCDT

Python bindings for [CDT: C++ library for constrained Delaunay triangulation](https://github.com/artem-ogre/CDT) implemented with [pybind11](https://github.com/pybind/pybind11)

***If PythonCDT helped you please consider adding a star on [GitHub](https://github.com/artem-ogre/PythonCDT). This means a lot to the authors*** 🤩

## Installation

```bash
pip install pythoncdt
```

Wheels are published for Linux, macOS and Windows.

## Building from source

Building needs a C++17 compiler; pip fetches pybind11, and CMake if it is missing.

```bash
git clone --recurse-submodules https://github.com/artem-ogre/PythonCDT.git
cd PythonCDT
pip install . --group test  # needs pip 25.1+
pytest
```

## Usage examples

### Constrained triangulation

```python
import numpy as np
import pythoncdt as cdt

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.4, 0.4]])
edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.uintc)

t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
t.insert_vertices(vertices)
t.insert_edges(edges)
t.erase_outer_triangles_and_holes()

vv = t.vertices_array()   # numpy array with fields 'x' and 'y'
tt = t.triangles_array()  # numpy array with fields 'vertices' and 'neighbors'
tt["vertices"]            # (T, 3) vertex indices into vv
```

#### Notes

- `vertices_array()` and `triangles_array()` return copies. With `copy=False` they return read-only views of the
  triangulation's memory instead; a view is invalidated by any call that modifies the triangulation.

### Conforming triangulation

```python
t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
t.insert_vertices(vertices)
t.conform_to_edges(edges)
t.erase_outer_triangles_and_holes()
```

#### Notes

- `conform_to_edges()` splits the constraint edges as needed, instead of keeping them as they are.

### Refined triangulation

```python
t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
t.insert_vertices(vertices)
t.insert_edges(edges)

to_erase = t.collect_outer_triangles_and_holes()
unrefined = t.refine_triangles(1000, cdt.RefinementCriterion.SMALLEST_ANGLE, cdt.deg_to_rad(20.0), to_erase)
t.finalize_triangulation(to_erase)
```

#### Notes

- `refine_triangles()` improves the shape of the triangles by inserting new points (Steiner points). It must be called
  before the triangulation is finalized: collect the triangles to remove first so that they are not refined, then
  remove them with `finalize_triangulation()` (the set is updated in place).
- Some places can not be refined: e.g., a sharp angle between two constraint edges comes from the input and can not be
  made any larger. `refine_triangles()` returns the counts of such refinements; `find_unrefined_triangles()` and
  `find_encroached_fixed_edges()` locate them in the resulting triangulation.

### Threads

Triangulating releases the GIL, so separate triangulations can be built in parallel on Python threads.
One triangulation can be shared between threads: calls on it wait for each other.
Iterators (`*_iter()`) and `copy=False` views are not protected: don't use them while another thread modifies the triangulation.

## License
[Mozilla Public License, v. 2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)

## Contributors
- [SioulisChris](https://github.com/SioulisChris): fixing the tests on Windows
- [sccolbert](https://github.com/sccolbert): reading the triangulation back as numpy arrays, releasing the GIL
