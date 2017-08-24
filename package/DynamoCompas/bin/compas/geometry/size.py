""""""

from __future__ import print_function
from __future__ import division

from compas.geometry.basic import subtract_vectors
from compas.geometry.basic import subtract_vectors_2d
from compas.geometry.basic import length_vector
from compas.geometry.basic import length_vector_2d
from compas.geometry.basic import cross_vectors
from compas.geometry.basic import cross_vectors_2d
from compas.geometry.basic import dot_vectors

from compas.geometry.average import centroid_points
from compas.geometry.average import centroid_points_2d

from compas.geometry.orientation import normal_triangle
from compas.geometry.orientation import normal_triangle_2d


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'bounding_box', 'area_polygon', 'area_triangle', 'volume_polyhedron',
    'bounding_box_2d', 'area_polygon_2d', 'area_triangle_2d'
]


def bounding_box(points):
    """Computes the bounding box of a list of points.
    """
    x, y, z = zip(*points)
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    min_z = min(z)
    max_z = max(z)
    return [(min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z)]


def bounding_box_2d(points):
    """Compute the bounding box of a list of points lying in the XY-plane.

    Parameters
    ----------
    points : sequence
        A sequence of XY(Z) coordinates of a 2D or 3D points.

    Returns
    -------
    sequence of float
        XYZ coordinates of four points defining a rectangle (Z components = 0).

    """
    x, y = zip(*points)[:2]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    return [(min_x, min_y, 0.0),
            (max_x, min_y, 0.0),
            (max_x, max_y, 0.0),
            (min_x, max_y, 0.0)]


def area_polygon(polygon):
    """Compute the area of a polygon.

    Parameters
    ----------
    polygon : sequence
        The XYZ coordinates of the vertices/corners of the polygon.
        The vertices are assumed to be in order.
        The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.

    Returns
    -------
    float
        The area of the polygon.

    """
    o = centroid_points(polygon)
    u = subtract_vectors(polygon[-1], o)
    v = subtract_vectors(polygon[0], o)
    a = 0.5 * length_vector(cross_vectors(u, v))
    for i in range(0, len(polygon) - 1):
        u = v
        v = subtract_vectors(polygon[i + 1], o)
        a += 0.5 * length_vector(cross_vectors(u, v))
    return a


def area_polygon_2d(polygon):
    """Compute the area of a polygon lying in the XY-plane.

    Parameters
    ----------
    polygon : sequence
        A sequence of XY(Z) coordinates of 2D or 3D points
        representing the locations of the corners of a polygon.
        The vertices are assumed to be in order. The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.

    Returns
    -------
    float
        The area of the polygon.

    """
    o = centroid_points_2d(polygon)
    u = subtract_vectors_2d(polygon[-1], o)
    v = subtract_vectors_2d(polygon[0], o)
    a = 0.5 * cross_vectors_2d(u, v)[2]
    for i in range(0, len(polygon) - 1):
        u = v
        v = subtract_vectors_2d(polygon[i + 1], o)
        a += 0.5 * cross_vectors_2d(u, v)[2]
    return abs(a)


def area_triangle(triangle):
    """Compute the area of a triangle defined by three points.
    """
    return 0.5 * length_vector(normal_triangle(triangle, False))


def area_triangle_2d(triangle):
    """Compute the area of a triangle defined by three points lying in the XY-plane.

    Parameters
    ----------
    triangle : list of list
        XY(Z) coordinates of the corners of the triangle.

    Returns
    -------
    float
        The area of the triangle.

    """
    return 0.5 * length_vector_2d(normal_triangle_2d(triangle, False))


def volume_polyhedron(polyhedron):
    r"""Compute the volume of a polyhedron represented by a closed mesh.

    This implementation is based on the divergence theorem, the fact that the
    *area vector* is constant for each face, and the fact that the area of each
    face can be computed as half the length of the cross product of two adjacent
    edge vectors.

    .. math::
        :nowrap:

        \begin{align}
            V  = \int_{P} 1
              &= \frac{1}{3} \int_{\partial P} \mathbf{x} \cdot \mathbf{n} \\
              &= \frac{1}{3} \sum_{i=0}^{N-1} \int{A_{i}} a_{i} \cdot n_{i} \\
              &= \frac{1}{6} \sum_{i=0}^{N-1} a_{i} \cdot \hat n_{i}
        \end{align}


    References
    ----------
    http://www.ma.ic.ac.uk/~rn/centroid.pdf

    """
    V = 0
    for fkey in polyhedron.face:
        vertices = polyhedron.face_vertices(fkey, ordered=True)
        if len(vertices) == 3:
            faces = [vertices]
        else:
            faces = []
            for i in range(1, len(vertices) - 1):
                faces.append(vertices[0:1] + vertices[i:i + 2])
        for face in faces:
            a  = polyhedron.vertex_coordinates(face[0])
            b  = polyhedron.vertex_coordinates(face[1])
            c  = polyhedron.vertex_coordinates(face[2])
            ab = subtract_vectors(b, a)
            ac = subtract_vectors(c, a)
            n  = cross_vectors(ab, ac)
            V += dot_vectors(a, n)
    return V / 6.


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
