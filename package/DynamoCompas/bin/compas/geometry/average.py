""""""

from __future__ import print_function
from __future__ import division

from compas.geometry.basic import add_vectors
from compas.geometry.basic import subtract_vectors
from compas.geometry.basic import subtract_vectors_2d
from compas.geometry.basic import length_vector
from compas.geometry.basic import length_vector_2d
from compas.geometry.basic import dot_vectors
from compas.geometry.basic import cross_vectors


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'centroid_points', 'midpoint_point_point', 'midpoint_line', 'center_of_mass_polygon',
    'centroid_points_2d', 'midpoint_point_point_2d', 'midpoint_line_2d', 'center_of_mass_polygon_2d',
    'center_of_mass_polyhedron'
]


def centroid_points(points):
    """Compute the centroid of a set of points.

    Warning
    -------
    Duplicate points are **NOT** removed. If there are duplicates in the
    sequence, they should be there intentionally.

    Parameters
    ----------
    points : sequence
        A sequence of XYZ coordinates.

    Returns
    -------
    list
        XYZ coordinates of the centroid.

    Examples
    --------
    >>> centroid_points()

    """
    p = len(points)
    x, y, z = zip(*points)
    return sum(x) / p, sum(y) / p, sum(z) / p


def centroid_points_2d(points):
    """Compute the centroid of a set of points lying in the XY-plane.

    Warning
    -------
    Duplicate points are **NOT** removed. If there are duplicates in the
    sequence, they should be there intentionally.

    Parameters
    ----------
    points : list of list
        A sequence of points represented by their XY(Z) coordinates.

    Returns
    -------
    list
        XYZ coordinates of the centroid (Z = 0.0).

    Examples
    --------
    >>> centroid_points_2d()

    """
    p = len(points)
    x, y = zip(*points)[:2]
    return [sum(x) / p, sum(y) / p, 0.0]


def midpoint_point_point(a, b):
    """Compute the midpoint of two points lying in the XY-plane.

    Parameters
    ----------
    a : sequence of float
        XYZ coordinates of the first point.
    b : sequence of float
        XYZ coordinates of the second point.

    Returns
    -------
    tuple
        XYZ coordinates of the midpoint.

    """
    return [0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2])]


def midpoint_point_point_2d(a, b):
    """Compute the midpoint of two points lying in the XY-plane.

    Parameters
    ----------
    a : sequence of float
        XY(Z) coordinates of the first 2D or 3D point (Z will be ignored).
    b : sequence of float
        XY(Z) coordinates of the second 2D or 3D point (Z will be ignored).

    Returns
    -------
    tuple
        XYZ coordinates of the midpoint (Z = 0.0).

    """
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0]


def midpoint_line(line):
    """Compute the midpoint of a line defined by two points.

    Parameters
    ----------
    a : sequence of float
        XYZ coordinates of the first point.
    b : sequence of float
        XYZ coordinates of the second point.

    Returns
    -------
    tuple
        XYZ coordinates of the midpoint.

    Examples
    --------
    >>> midpoint_line()

    """
    return midpoint_point_point(*line)


def midpoint_line_2d(line):
    return midpoint_point_point_2d(*line)


def center_of_mass_polygon(polygon):
    """Compute the center of mass of a polygon defined as a sequence of points.

    The center of mass of a polygon is the centroid of the midpoints of the edges,
    each weighted by the length of the corresponding edge.

    Parameters
    ----------
    polygon : sequence
        A sequence of XYZ coordinates representing the locations of the corners of a polygon.

    Returns
    -------
    tuple of floats
        The XYZ coordinates of the center of mass.

    Examples
    --------
    >>> pts = [(0.,0.,0.),(1.,0.,0.),(0.,10.,0.)]
    >>> print("Center of mass: {0}".format(center_of_mass(pts)))
    >>> print("Centroid: {0}".format(centroid(pts)))

    """
    L  = 0
    cx = 0
    cy = 0
    cz = 0
    p  = len(polygon)
    for i in range(-1, p - 1):
        p1  = polygon[i]
        p2  = polygon[i + 1]
        d   = length_vector(subtract_vectors(p2, p1))
        cx += 0.5 * d * (p1[0] + p2[0])
        cy += 0.5 * d * (p1[1] + p2[1])
        cz += 0.5 * d * (p1[2] + p2[2])
        L  += d
    cx = cx / L
    cy = cy / L
    cz = cz / L
    return cx, cy, cz


def center_of_mass_polygon_2d(polygon):
    """Compute the center of mass of a polygon defined as a sequence of points lying in the XY-plane.

    The center of mass of a polygon is the centroid of the midpoints of the edges,
    each weighted by the length of the corresponding edge.

    Parameters
    ----------
    polygon : sequence
        A sequence of XY(Z) coordinates of 2D or 3D points (Z will be ignored)
        representing the locations of the corners of a polygon.

    Returns
    -------
    tuple of floats
        The XYZ coordinates of the center of mass (Z = 0.0).

    Examples
    --------
    >>>

    """
    L  = 0
    cx = 0
    cy = 0
    p  = len(polygon)
    for i in range(-1, p - 1):
        p1  = polygon[i]
        p2  = polygon[i + 1]
        d   = length_vector_2d(subtract_vectors_2d(p2, p1))
        cx += 0.5 * d * (p1[0] + p2[0])
        cy += 0.5 * d * (p1[1] + p2[1])
        L  += d
    cx = cx / L
    cy = cy / L
    return cx, cy, 0.0


def center_of_mass_polyhedron(polyhedron):
    """Compute the center of mass of a polyhedron"""
    vertices, faces = polyhedron

    V  = 0
    x  = 0.0
    y  = 0.0
    z  = 0.0
    ex = [1.0, 0.0, 0.0]
    ey = [0.0, 1.0, 0.0]
    ez = [0.0, 0.0, 1.0]

    for face in faces:
        if len(face) == 3:
            triangles = [face]
        else:
            triangles = []
            # for i in range(1, len(face) - 1):
            #     triangles.append(face[0:1] + vertices[i:i + 2])

        for triangle in triangles:
            a  = vertices[triangle[0]]
            b  = vertices[triangle[1]]
            c  = vertices[triangle[2]]
            ab = subtract_vectors(b, a)
            ac = subtract_vectors(c, a)
            n  = cross_vectors(ab, ac)
            V += dot_vectors(a, n)
            nx = dot_vectors(n, ex)
            ny = dot_vectors(n, ey)
            nz = dot_vectors(n, ez)

            for j in (-1, 0, 1):
                ab = add_vectors(vertices[triangle[j]], vertices[triangle[j + 1]])
                x += nx * dot_vectors(ab, ex) ** 2
                y += ny * dot_vectors(ab, ey) ** 2
                z += nz * dot_vectors(ab, ez) ** 2

    if V < 1e-9:
        V = 0.0
        d = 1.0 / 48.0
    else:
        V = V / 6.0
        d = 1.0 / 48.0 / V

    x *= d
    y *= d
    z *= d

    return x, y, z


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
