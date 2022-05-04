# PythonCDT

Python bindings for [CDT: C++ library for constrained Delaunay triangulation](https://github.com/artem-ogre/CDT) implemented with [pybind11](https://github.com/pybind/pybind11)

Current status: **work-in-progress, under development.**

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

## License
[Mozilla Public License, v. 2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)
