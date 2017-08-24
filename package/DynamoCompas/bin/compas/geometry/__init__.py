"""
.. _compas.geometry:

********************************************************************************
Geometry (:mod:`compas.geometry`)
********************************************************************************

.. currentmodule:: compas.geometry

A package defining common geometric functions and objects.

The functions in this package expect input arguments to be structured in a certain
way. This is properly documented in their *docstrings* (or at least, it should be :).
In general the following is assumed.

- **point** -- The xyz coordinates as a sequence of floats.
- **vector** -- The xyz coordinates of the end point. The start is always the origin.
- **line** -- A tuple with two points representing a continuous line (ray).
- **segment** -- A tuple with two points representing a line segment.
- **plane** -- A tuple with a base point and normal vector.
- **circle** -- A tuple with a point, the normal vector of the plane of the circle, and the radius as float.
- **polygon** -- A sequence of points. First and last are not the same. The polygon is assumed closed.
- **polyline** -- A sequence of points. First and last are the same if the polyline is closed. Otherwise, it is assumed open.
- **polyhedron** -- A list of vertices represented by their XYZ coordinates, and a list of faces referencing the vertex list.
- **frame** -- A list of three orthonormal vectors.


.. note::

    Most functions in the geometry package have a 2D equivalent. These equivalent functions
    do exactly the same as the original, but assume the operands lie in the XY plane.
    They thus simply ignore the Z component of the operands and, when relevant, return
    a value with the Z component set to zero. Most of these functions are provided because
    of speed concerns.
    The 

    The planar geometry (compas.geometry.planar) package provides specific
    functionality for 2D geometry. Inputs are expected to be 2D and return values are also
    2D.


.. rubric:: Submodules

.. toctree::
    :maxdepth: 1

    compas.geometry.planar
    compas.geometry.elements


Basics
======

Vector functions
----------------

.. autosummary::
    :toctree: generated/

    sum_vectors
    add_vectors
    subtract_vectors
    scale_vector
    scale_vectors
    normalize_vector
    normalize_vectors
    norm_vector
    norm_vectors
    length_vector
    length_vector_sqrd
    cross_vectors
    vector_component
    dot_vectors
    multiply_matrices
    multiply_matrix_vector

Distance
--------

.. autosummary::
    :toctree: generated/

    distance_point_point
    distance_point_point_sqrd
    distance_point_line
    distance_point_line_sqrd
    distance_point_plane
    distance_line_line
    closest_point_in_cloud
    closest_point_on_line
    closest_point_on_segment
    closest_point_on_polyline
    closest_point_on_plane

Angles
------

.. autosummary::
    :toctree: generated/

    angles_points
    angles_points_degrees
    angles_vectors
    angles_vectors_degrees
    angle_smallest_points
    angle_smallest_points_degrees
    angle_smallest_vectors
    angle_smallest_vectors_degrees

Average
-------

.. autosummary::
    :toctree: generated/

    centroid_points
    center_of_mass_polygon
    center_of_mass_polyhedron
    midpoint_line

Orientation
-----------

.. autosummary::
    :toctree: generated/

    normal_polygon
    normal_triangle

Size
----

.. autosummary::
    :toctree: generated/

    area_polygon
    area_triangle
    volume_polyhedron
    bounding_box


Intersections
=============

.. autosummary::
    :toctree: generated/

    intersection_line_line
    intersection_circle_circle
    intersection_line_triangle
    intersection_line_plane
    intersection_segment_plane
    intersection_plane_plane
    intersection_plane_plane_plane
    intersection_lines
    intersection_planes


Transformations
===============

.. autosummary::
    :toctree: generated/

    translate_points
    translate_lines
    rotate_points
    orient_points
    mirror_point_point
    mirror_points_point
    mirror_point_line
    mirror_points_line
    mirror_point_plane
    mirror_points_plane
    reflect_line_plane
    reflect_line_triangle
    project_point_plane
    project_points_plane
    project_point_line
    project_points_line
    offset_line
    offset_polygon


Queries
=======

.. autosummary::
    :toctree: generated/

    is_colinear
    is_coplanar
    is_polygon_convex
    is_point_on_plane
    is_point_on_line
    is_point_on_segment
    is_closest_point_on_segment
    is_point_on_polyline
    is_point_in_triangle
    is_point_in_circle
    is_intersection_line_line
    is_intersection_line_plane
    is_intersection_segment_plane
    is_intersection_plane_plane
    is_intersection_line_triangle
    is_intersection_box_box


Miscellaneous
=============

Hull
----

.. currentmodule:: compas.geometry.hull

:mod:`compas.geometry.hull`

.. autosummary::
    :toctree: generated/

    convex_hull
    polyhedron_from_node


KD-tree
------

.. currentmodule:: compas.geometry.kdtree

:mod:`compas.geometry.kdtree`

.. autosummary::
    :toctree: generated/

    KDTree


xforms
------

.. currentmodule:: compas.geometry.xforms

:mod:`compas.geometry.xforms`

.. autosummary::
    :toctree: generated/

    translation_matrix
    rotation_matrix
    scale_matrix
    shear_matrix
    projection_matrix

"""


def is_point(point):
    assert len(point) >= 2, 'A point is defined by at least two coordinates.'


def is_vector(vector):
    assert len(vector) >= 2, 'A vector has at least two components.'


def is_line(line):
    assert len(line) == 2, 'A line is specified by two points.'
    a, b = line
    is_point(a)
    is_point(b)


def is_segment(segment):
    assert len(segment) == 2, 'A segment is defined by two points.'
    a, b = segment
    is_point(a)
    is_point(b)


def is_plane(plane):
    assert len(plane) == 2, 'A plane is defined by a base point and a normal vector.'
    base, normal = plane
    is_point(base)
    is_vector(normal)


def is_circle():
    pass


def is_polygon():
    pass


def is_polyline():
    pass


def is_polyhedron():
    pass


def is_frame():
    pass


# level 0

from .basic import *

# level 1

from .distance import *
from .angles import *
from .average import *
from .intersections import *
from .constructors import *

# level 2

from .orientation import *
from .bestfit import *
from .queries import *

# level 3

from .size import *
from .transformations import *


__all__ = [s for s in dir() if not s.startswith('_')]
