# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Python bindings for CDT: constrained Delaunay triangulation"""

from ._core import (
    NO_NEIGHBOR,
    NO_VERTEX,
    Edge,
    IntersectingConstraintEdges,
    RefinementCriterion,
    Triangle,
    Triangulation,
    Unrefined,
    V2d,
    VertexInsertionOrder,
    __version__,
    area,
    circumcenter,
    deg_to_rad,
    smallest_angle,
    verify_topology,
    verify_winding,
)

__all__ = [
    "NO_NEIGHBOR",
    "NO_VERTEX",
    "Edge",
    "IntersectingConstraintEdges",
    "RefinementCriterion",
    "Triangle",
    "Triangulation",
    "Unrefined",
    "V2d",
    "VertexInsertionOrder",
    "__version__",
    "area",
    "circumcenter",
    "deg_to_rad",
    "smallest_angle",
    "verify_topology",
    "verify_winding",
]
