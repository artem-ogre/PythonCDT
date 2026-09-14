# PythonCDT

Python bindings for [CDT: C++ library for constrained Delaunay triangulation](https://github.com/artem-ogre/CDT) implemented with [pybind11](https://github.com/pybind/pybind11)

***If PythonCDT helped you please consider adding a star on [GitHub](https://github.com/artem-ogre/PythonCDT). This means a lot to the authors*** 🤩
## Building

### Pre-conditions
- Clone with submodules: `git clone --recurse-submodules https://github.com/artem-ogre/PythonCDT.git`
- Make sure packages from requirements.txt are available.

```bash
# build the wheel and install the package with pip
pip3 install .
# run tests
pytest ./cdt_bindings_test.py
```

## Usage

```python
import numpy as np
import PythonCDT as cdt

t = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.TRY_RESOLVE, 0.0)
t.insert_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
t.insert_edges(np.array([[0, 3]], dtype=np.uintc))
t.erase_super_triangle()

vertices = t.vertices_array()    # numpy array with fields 'x' and 'y'
triangles = t.triangles_array()  # numpy array with fields 'vertices' and 'neighbors'
triangles["vertices"]            # (T, 3) vertex indices into vertices
```

`vertices_array()` and `triangles_array()` return copies. With `copy=False` they return read-only views of the
triangulation's memory without copying; a view is invalidated by any call that modifies the triangulation.

### Threads

Triangulating releases the GIL, so separate triangulations can be built in parallel on Python threads.
One triangulation can be shared between threads: calls on it wait for each other.
Iterators (`*_iter()`) and `copy=False` views are not protected: don't use them while another thread modifies the triangulation.

## License
[Mozilla Public License, v. 2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)

## Contributors
- [SioulisChris](https://github.com/SioulisChris): fixing the tests on Windows
