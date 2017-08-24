from __future__ import print_function

import random
import itertools

from math import sqrt
from math import acos
from math import pi
from math import sin
from math import cos
from math import radians
from math import degrees

from compas.geometry import multiply_matrix_vector


__author__     = ['Tom Van Mele <vanmelet@ethz.ch>', 'Matthias Rippmann <rippmann@ethz.ch>']
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


# note:
# move 2D version of 3D algorithms to respective modules
# create planar package with modules specifically for planar stuff


__all__ = []


# ------------------------------------------------------------------------------
# miscellaneous
# ------------------------------------------------------------------------------


def scatter_points_2d():
    return [(1.0 * random.randint(0, 100), 1.0 * random.randint(0, 100), 0.0) for i in range(20)]


# ------------------------------------------------------------------------------
# constructors
# ------------------------------------------------------------------------------


def vector_from_points_2d(a, b):
    """
    Create a vector based on a start point a and end point b in the XY-plane.

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        Tuple: Resulting 3D vector in the XY-plane (Z = 0.0)

    Notes:
        The result of this function is equal to subtract_vectors(b, a)

    """
    return b[0] - a[0], b[1] - a[1], 0.0


def circle_from_points_2d(a, b, c):
    """Create a circle from three points lying in the XY-plane.

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        c (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        tuple: XYZ coordinates of center in the XY-plane (Z = 0.0) and radius of the circle.

    References:
        https://en.wikipedia.org/wiki/Circumscribed_circle

    """
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    a = bx - ax
    b = by - ay
    c = cx - ax
    d = cy - ay
    e = a * (ax + bx) + b * (ay + by)
    f = c * (ax + cx) + d * (ay + cy)
    g = 2 * (a * (cy - by) - b * (cx - bx))
    if g == 0:
        return None
    centerx = (d * e - b * f) / g
    centery = (a * f - c * e) / g
    r = sqrt((ax - centerx) ** 2 + (ay - centery) ** 2)
    return (centerx, centery, 0.0), r


# ------------------------------------------------------------------------------
# misc
# ------------------------------------------------------------------------------


def vector_component_2d(u, v):
    """Compute the component of vector u in direction of vector v lying in the XY-plane.
        Also described as the orthogonal projection of vector u on vector v.
        See: https://en.wikipedia.org/wiki/Vector_projection

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        Tuple: Resulting vector in the XY-plane (Z = 0.0)
    """
    x = dot_vectors_2d(u, v) / length_vector_sqrd_2d(v)
    return x * v[0], x * v[1], 0.0


# ------------------------------------------------------------------------------
# operations
# ------------------------------------------------------------------------------


def add_vectors_2d(u, v):
    """Add two vectors lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        Tuple: Resulting vector in the XY-plane (Z = 0.0)
    """
    return u[0] + v[0], u[1] + v[1], 0.0


def subtract_vectors_2d(u, v):
    """Subtract the second vector from the first lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        Tuple: Resulting vector in the XY-plane (Z = 0.0)
    """
    return u[0] - v[0], u[1] - v[1], 0.0


def scale_vector_2d(vector, scale):
    """Scale a vector lying in the XY-plane by a factor.

    Parameters:
        vector (sequence of float): The second 2D or 3D vector (Z will be ignored).
        scale (float): Scale factor

    Returns:
        Tuple: Resulting vector in the XY-plane (Z = 0.0)
    """
    return vector[0] * scale, vector[1] * scale, 0.0


def normalize_vector_2d(vector):
    """Normalize a vector lying in the XY-plane.

    Parameters:
        vector (sequence of float): The 2D or 3D vector (Z will be ignored).

    Returns:
        Tuple: The normalized vector in the XY-plane (Z = 0.0)
    """
    l = float(length_vector_2d(vector))
    if l == 0.0:
        return vector
    return vector[0] / l, vector[1] / l, 0.0


def normalize_vectors_2d(vectors):
    """Normalize a list of vectors lying in the XY-plane.

    Parameters:
        vectors (a list of sequences of floats): The list of 2D or 3D vectors
        (the Z components will be ignored).

    Returns:
        Tuple: The normalized vectors in the XY-plane (Z components = 0.0)
    """
    return [normalize_vector_2d(vector) for vector in vectors]


def dot_vectors_2d(u, v):
    """Compute the dot product of two vectors lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        float: The dot product of the two vectors in the XY-plane (Z = 0.0)

    Examples:

        .. code-block:: python

            print(dot_vectors_2d([1.0, 0], [2.0, 0]))
            # 2.0

            print(dot_vectors_2d([1.0, 0, 0], [2.0, 0, 0]))
            # 2.0

            print(dot_vectors_2d([1.0, 0, 1], [2.0, 0, 1]))
            # 2.0

    """
    return u[0] * v[0] + u[1] * v[1]


def cross_vectors_2d(u, v):
    """Compute the cross product of two vectors lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        list: The cross product of the two vectors.

    Examples:

        .. code-block:: python

            cross([1.0, 0.0], [0.0, 1.0])
            # [0.0, 0.0, 1.0]

            cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            # [0.0, 0.0, 1.0]

            cross([1.0, 0.0, 1.0], [0.0, 1.0, 1.0])
            # [0.0, 0.0, 1.0]

    """
    return [0.0, 0.0, u[0] * v[1] - u[1] * v[0]]


# ------------------------------------------------------------------------------
# length
# ------------------------------------------------------------------------------


def length_vector_2d(vector):
    """Compute the length of a vector lying in the XY-plane.

    Parameters:
        v (sequence of float): The 2D or 3D vector (Z will be ignored).

    Returns:
        float: The length of the vector.

    Examples:
        length([2.0, 0.0])
        #2.0

        length([2.0, 0.0, 0.0])
        #2.0

        length([2.0, 0.0, 2.0])
        #2.0

    """
    return sqrt(dot_vectors_2d(vector, vector))


def length_vector_sqrd_2d(vector):
    """Computes the squared length of a vector lying in the XY-plane.

    Parameters:
        v (sequence of float): The 2D or 3D vector (Z will be ignored).

    Returns:
        float: The squared length of the vector.

    Examples:
        length_sqrd([2.0, 0.0])
        #4.0

        length_sqrd([2.0, 0.0, 0.0])
        #4.0

        length_sqrd([2.0, 0.0, 2.0])
        #4.0

    """
    return dot_vectors_2d(vector, vector)


# ------------------------------------------------------------------------------
# distance
# ------------------------------------------------------------------------------


def distance_point_point_2d(a, b):
    """Compute the distance between points a and b lying in the XY-plane.

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        float: distance between a and b in the XY-plane

    Examples:
        distance([0.0, 0.0], [2.0, 0.0])
        #2.0

        distance([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        #2.0

        distance([0.0, 0.0, 1.0], [2.0, 0.0, 1.0])
        #2.0

    """
    ab = subtract_vectors_2d(b, a)
    return length_vector_2d(ab)


def distance_point_point_sqrd_2d(a, b):
    """Compute the squared distance between points a and b lying in the XY-plane.

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        float: squared distance between a and b in the XY-plane

    Examples:
        distance([0.0, 0.0], [2.0, 0.0])
        #4.0

        distance([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        #4.0

        distance([0.0, 0.0, 1.0], [2.0, 0.0, 1.0])
        #4.0

    """
    ab = subtract_vectors_2d(b, a)
    return length_vector_sqrd_2d(ab)


def distance_point_line_2d(point, line):
    """Compute the distance between a point and a line lying in the XY-plane.

    This implementation computes the orthogonal distance from a point P to a
    line defined by points A and B as twice the area of the triangle ABP divided
    by the length of AB.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        line (list, tuple) : Line defined by two points.

    Returns:
        float : The distance between the point and the line.

    References:
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    """
    a, b = line
    ab   = subtract_vectors_2d(b, a)
    pa   = subtract_vectors_2d(a, point)
    pb   = subtract_vectors_2d(b, point)
    l    = abs(cross_vectors_2d(pa, pb)[2])
    l_ab = length_vector_2d(ab)
    return l / l_ab


def distance_point_line_sqrd_2d(point, line):
    """Compute the squared distance between a point and a line lying in the XY-plane.

    This implementation computes the orthogonal squared distance from a point P to a
    line defined by points A and B as twice the area of the triangle ABP divided
    by the length of AB.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        line (list, tuple) : Line defined by two points.

    Returns:
        float : The squared distance between the point and the line.

    References:
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    """
    a, b = line
    ab   = subtract_vectors_2d(b, a)
    pa   = subtract_vectors_2d(a, point)
    pb   = subtract_vectors_2d(b, point)
    l    = cross_vectors_2d(pa, pb)[2]**2
    l_ab = length_vector_sqrd_2d(ab)
    return l / l_ab

# ------------------------------------------------------------------------------
# angles
# ------------------------------------------------------------------------------


def angles_vectors_2d(u, v):
    """Compute the angles between the XY components of two vectors lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        tuple: The two angles.

        The smallest angle is returned first.

    """
    a = angle_smallest_vectors_2d(u, v)
    return a, 2. * pi - a


def angles_points_2d(a, b, c):
    """Compute the angles defined by the XY components of three points lying in
       the XY-plane where the angles are computed at point A in the triangle ABC

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        c (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        tuple: The two angles.

        The smallest angle is returned first.

    Notes:
        The vectors are defined in the following way

        .. math::

            \mathbf{u} = \mathbf{b} - \mathbf{a} \\
            \mathbf{v} = \mathbf{c} - \mathbf{a}

        Z components may be provided, but are simply ignored.

    """
    u = subtract_vectors_2d(b, a)
    v = subtract_vectors_2d(c, a)
    a = angle_smallest_vectors_2d(u, v)
    return a, 2. * pi - a


def angle_smallest_vectors_2d(u, v):
    """Compute the smallest angle (radians) between the XY components of two vectors lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        float: The smallest angle between the vectors in radians.

        The angle is always positive.

    """
    a = dot_vectors_2d(u, v) / (length_vector_2d(u) * length_vector_2d(v))
    a = max(min(a, 1), -1)
    return acos(a)


def angle_smallest_vectors_degrees_2d(u, v):
    """Compute the smallest angle (degrees) between the XY components of two vectors lying in the XY-plane.

    Parameters:
        u (sequence of float): The first 2D or 3D vector (Z will be ignored).
        v (sequence of float): The second 2D or 3D vector (Z will be ignored).

    Returns:
        float: The smallest angle between the vectors in degrees.

        The angle is always positive.

    """
    return degrees(angle_smallest_vectors_2d)


def angle_smallest_points_2d(a, b, c):
    """Compute the smallest angle defined by the XY components of three points lying in
       the XY-plane where the angle is computed at point A in the triangle ABC

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        c (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        float: The smallest angle between the vectors.

        The angle is always positive.

    Notes:
        The vectors are defined in the following way

        .. math::

            \mathbf{u} = \mathbf{b} - \mathbf{a} \\
            \mathbf{v} = \mathbf{c} - \mathbf{a}

        Z components may be provided, but are simply ignored.

    """
    u = subtract_vectors_2d(b, a)
    v = subtract_vectors_2d(c, a)
    a = angle_smallest_vectors_2d(u, v)
    return a


def angle_smallest_points_degrees_2d(a, b, c):
    """Compute the smallest angle defined by the XY components of three points lying in
       the XY-plane where the angle is computed at point A in the triangle ABC

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        c (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        float: The smallest angle between the vectors.

        The angle is always positive.

    Notes:
        The vectors are defined in the following way

        .. math::

            \mathbf{u} = \mathbf{b} - \mathbf{a} \\
            \mathbf{v} = \mathbf{c} - \mathbf{a}

        Z components may be provided, but are simply ignored.

    """
    a = degrees(angle_smallest_points_2d(a, b, c))
    return a


# ------------------------------------------------------------------------------
# average
# ------------------------------------------------------------------------------


def centroid_points_2d(points):
    """Compute the centroid of a set of points lying in the XY-plane.

    Warning:
        Duplicate points are **NOT** removed. If there are duplicates in the
        sequence, they should be there intentionally.

    Parameters:
        points (sequence): A sequence of XY(Z) coordinates of a 2D or 3D points
        (Z components will be ignored).
    Returns:
        list: XYZ coordinates of the centroid (Z = 0.0).

    Examples:
        >>> centroid()
    """
    p = float(len(points))
    x, y = zip(*points)[:2]
    return sum(x) / p, sum(y) / p, 0.0


def midpoint_point_point_2d(a, b):
    """Compute the midpoint of two points lying in the XY-plane.

    Parameters:
        a (sequence of float): XY(Z) coordinates of the first 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of the second 2D or 3D point (Z will be ignored).

    Returns:
        tuple: XYZ coordinates of the midpoint (Z = 0.0).

    """
    return 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.0


def midpoint_line_2d(line):
    a, b = line
    return midpoint_point_point_2d(a, b)


def center_of_mass_polygon_2d(polygon):
    """Compute the center of mass of a polygon defined as a sequence of points lying in the XY-plane.

    The center of mass of a polygon is the centroid of the midpoints of the edges,
    each weighted by the length of the corresponding edge.

    Parameters:
        polygon (sequence) : A sequence of XY(Z) coordinates of 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a polygon.

    Returns:
        tuple of floats: The XYZ coordinates of the center of mass (Z = 0.0).

    Examples:
        pts = [(0.,0.,0.),(1.,0.,0.),(0.,10.,0.)]
        print "Center of mass: {0}".format(center_of_mass(pts))
        print "Centroid: {0}".format(centroid(pts))

    """
    L  = 0
    cx = 0
    cy = 0
    p  = len(polygon)
    for i in range(-1, p - 1):
        p1  = polygon[i]
        p2  = polygon[i + 1]
        d   = distance_point_point_2d(p1, p2)
        cx += 0.5 * d * (p1[0] + p2[0])
        cy += 0.5 * d * (p1[1] + p2[1])
        L  += d
    cx = cx / L
    cy = cy / L
    return cx, cy, 0.0


# ------------------------------------------------------------------------------
# size
# ------------------------------------------------------------------------------


def area_polygon_2d(polygon):
    """Compute the area of a polygon lying in the XY-plane.

    Parameters:
        polygon (sequence) : A sequence of XY(Z) coordinates of 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a polygon.
        The vertices are assumed to be in order. The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.

    Returns:
        float: The area of the polygon.

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


# shall we use a, b, c or triangle as a special type (kind of polygon)?
# triangle = (a,b,c) would be more consistent with line, segment, polygon
def area_triangle_2d(a, b, c):
    """Compute the area of a triangle defined by three points lying in the XY-plane.

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        c (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns:
        float: The area of the triangle.

    """
    return abs((a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) * 0.5)

# ------------------------------------------------------------------------------
# orientation
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# bounding boxes
# ------------------------------------------------------------------------------


def bounding_box_2d(points):
    """Compute the bounding box of a list of points lying in the XY-plane.

    Parameters:
        points (sequence): A sequence of XY(Z) coordinates of a 2D or 3D points
        (Z components will be ignored).

    Returns:
        (sequence of float): XYZ coordinates of four points defining a rectangle (Z components = 0).

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


# ------------------------------------------------------------------------------
# convex hull
# ------------------------------------------------------------------------------

# #https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
# def convex_hull(points):
#     """Computes the convex hull of a set of 2D points.
# 
#     Input: an iterable sequence of (x, y) pairs representing the points.
#     Output: a list of vertices of the convex hull in counter-clockwise order,
#       starting from the vertex with the lexicographically smallest coordinates.
#     Implements Andrew's monotone chain algorithm. O(n log n) complexity.
#     """
# 
#     # Sort the points lexicographically (tuples are compared lexicographically).
#     # Remove duplicates to detect the case we have just one unique point.
#     points = sorted(set(points))
# 
#     # Boring case: no points or a single point, possibly repeated multiple times.
#     if len(points) <= 1:
#         return points
# 
#     # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
#     # Returns a positive value, if OAB makes a counter-clockwise turn,
#     # negative for clockwise turn, and zero if the points are collinear.
#     def cross(o, a, b):
#         return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
# 
#     # Build lower hull 
#     lower = []
#     for p in points:
#         while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
#             lower.pop()
#         lower.append(p)
# 
#     # Build upper hull
#     upper = []
#     for p in reversed(points):
#         while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
#             upper.pop()
#         upper.append(p)
# 
#     # Concatenation of the lower and upper hulls gives the convex hull.
#     # Last point of each list is omitted because it is repeated at the beginning of the other list. 
#     return lower[:-1] + upper[:-1]


# ------------------------------------------------------------------------------
# proximity
# ------------------------------------------------------------------------------

def sort_points_2d(point, points):
    """Sorts points of a pointcloud to a point in the XY-plane.

    Notes:
        Check kdTree class for an optimized implementation (MR).

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        points (sequence): A sequence of XY(Z) coordinates of a 2D or 3D points
        (Z components will be ignored).

    Returns:
        list (floats): min distances
        list (tuples): sorted points
        list (ints): closest point indices
    """
    minsq = [distance_point_point_sqrd_2d(p, point) for p in points]
    return sorted(zip(minsq, points, range(len(points))), key=lambda x: x[0])


def closest_point_in_cloud_2d(point, points):
    """Calculates the closest point in a list of points in the XY-plane.

    Notes:
        Check kdTree class for an optimized implementation (MR).

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        points (sequence): A sequence of XY(Z) coordinates of a 2D or 3D points
        (Z components will be ignored).

    Returns:
        float: min distance
        tuple: closest point
        int: closest point index

    """
    data = sort_points_2d(point, points)
    return data[0]


def closest_point_on_line_2d(point, line):
    """
    Compute closest point on line (continuous) to a given point lying in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        line (tuple): Two 2D or 3D points defining the line (Z components will be ignored).

    Returns:
        list: XYZ coordinates of closest point (Z = 0.0).

    """
    a, b = line
    ab = subtract_vectors_2d(b, a)
    ap = subtract_vectors_2d(point, a)
    c = vector_component_2d(ap, ab)
    return add_vectors_2d(a, c)


def closest_point_on_segment_2d(point, segment):
    """
    Compute closest point on line segment to a given point lying in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        segment (tuple): Two 2D or 3D points defining the line segment (Z components will be ignored).

    Returns:
        list: XYZ coordinates of closest point (Z = 0.0).

    """
    a, b = segment
    p  = closest_point_on_line_2d(point, segment)
    d  = distance_point_point_sqrd_2d(a, b)
    d1 = distance_point_point_sqrd_2d(a, p)
    d2 = distance_point_point_sqrd_2d(b, p)
    if d1 > d or d2 > d:
        if d1 < d2:
            return a
        return b
    return p


def closest_point_on_polygon_2d(point, polygon):
    """Compute closest point on a polygon to a given point lying in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        polygon (sequence) : A sequence of XY(Z) coordinates of 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a polygon.
        The vertices are assumed to be in order. The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.

    Returns:
        list: XYZ coordinates of closest point (Z = 0.0).

    """
    points = []
    for i in range(len(polygon)):
        segment = polygon[i - 1], polygon[i]
        points.append(closest_point_on_segment_2d(point, segment))

    return closest_point_in_cloud_2d(point, points)[1]


# this function seems to be rather specific.
# where is it used?
# is it needed in the geometry package?
def closest_part_of_triangle(point, triangle):
    """Computes the closest part (edge or point) of a triangle to a test point
    lying in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        triangle (sequence): A sequence of three points representing the triangle.

    Returns:
        The coordinates of the corner point if a corner point is closest. Two corner points defining
        the edge, if an edge is closest to the test point.
    """
    a, b, c = triangle
    ab = subtract_vectors_2d(b, a)
    bc = subtract_vectors_2d(c, b)
    ca = subtract_vectors_2d(a, c)
    # closest to edge ab?
    ab_ = cross_vectors_2d(ab, [0, 0, 1])
    ba_ = add_vectors_2d(ab, ab_)
    if not is_ccw_2d(a, b, point) and not is_ccw_2d(b, ba_, point) and is_ccw_2d(a, ab_, point):
        return a, b
    # closest to edge bc?
    bc_ = cross_vectors_2d(bc, [0, 0, 1])
    cb_ = add_vectors_2d(bc, bc_)
    if not is_ccw_2d(b, c, point) and not is_ccw_2d(c, cb_, point) and is_ccw_2d(b, bc_, point):
        return b, c
    # closest to edge ac?
    ca_ = cross_vectors_2d(ca, [0, 0, 1])
    ac_ = add_vectors_2d(ca, ca_)
    if not is_ccw_2d(c, a, point) and not is_ccw_2d(a, ac_, point) and is_ccw_2d(c, ca_, point):
        return c, a
    # closest to a?
    if not is_ccw_2d(a, ab_, point) and is_ccw_2d(a, ac_, point):
        return a
    # closest to b?
    if not is_ccw_2d(b, bc_, point) and is_ccw_2d(b, ba_, point):
        return b
    # closest to c?
    if not is_ccw_2d(c, ca_, point) and is_ccw_2d(c, cb_, point):
        return c


# ------------------------------------------------------------------------------
# queries
# ------------------------------------------------------------------------------


def is_ccw_2d(a, b, c, colinear=False):
    """Verify if ``c`` is on the left of ``ab`` when looking from ``a`` to ``b``.

    Parameters:
        a (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        b (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        c (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        colinear (bool): Optional. Allow points to be colinear. Default is ``False``.

    Returns:
        bool : ``True`` if ccw, ``False`` otherwise.

    Examples:
        print(is_ccw([0,0,0], [0,1,0], [-1, 0, 0]))
        # True

        print(is_ccw([0,0,0], [0,1,0], [+1, 0, 0]))
        # False

        print(is_ccw([0,0,0], [1,0,0], [2,0,0]))
        # False

        print(is_ccw([0,0,0], [1,0,0], [2,0,0], True))
        # True

    References:
        https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation

    """
    if colinear:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1])  * (c[0] - a[0]) >= 0
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1])  * (c[0] - a[0]) > 0


def is_colinear_2d():
    raise NotImplementedError


def is_polygon_convex_2d(polygon, colinear=False):
    """Verify if the polygon is convex in the XY-plane.

    Parameters:
        polygon (sequence) : A sequence of XY(Z) coordinates of 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a polygon.
        The vertices are assumed to be in order. The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.
        colinear (bool): Are points allowed to be colinear?

    Returns:
        bool: True if the figure is convex, False otherwise.

    """
    a = polygon[-2]
    b = polygon[-1]
    c = polygon[0]
    direction = is_ccw_2d(a, b, c, colinear)
    for i in range(-1, len(polygon) - 2):
        a = b
        b = c
        c = polygon[i + 2]
        if direction != is_ccw_2d(a, b, c, colinear):
            return False
    return True


def is_point_on_line_2d(point, line, tol=0.0):
    """Verify if a point lies on a line in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        line (tuple): Two 2D or 3D points defining a line (Z components will be ignored).
        tol (float): Optional. A tolerance. Default is ``0.0``.

    Returns:
        (bool): True if the point is in on the line, False otherwise.

    """
    d = distance_point_line_2d(point, line)
    return d <= tol


def is_point_on_segment_2d(point, segment, tol=0.0):
    """Verify if a point lies on a given line segment in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        segment (tuple): Two 2D or 3D points defining the start and end points of 
        the line segment (Z components will be ignored).

    Returns:
        (bool): True if the point is on the line segment, False otherwise.

    """
    a, b = segment
    if not is_point_on_line_2d(point, segment, tol=tol):
        return False
    d_ab = distance_point_point_2d(a, b)
    if d_ab == 0:
        return False
    d_pa = distance_point_point_2d(a, point)
    d_pb = distance_point_point_2d(b, point)
    if d_pa + d_pb <= d_ab + tol:
        return True
    return False


def is_point_on_polygon_2d():
    raise NotImplementedError


def is_point_in_convex_polygon_2d(point, polygon):
    """Verify if a point is in the interior of a convex polygon lying in the XY-plane.

    Parameters:
        (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        polygon (sequence) : A sequence of XY(Z) coordinates of 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a polygon.
        The vertices are assumed to be in order. The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.

    Warning:
        Does not work for concave polygons.

    Returns:
        bool: True if the point is in the convex polygon, False otherwise.
    """
    ccw = None
    for i in range(-1, len(polygon) - 1):
        a = polygon[i]
        b = polygon[i + 1]
        if ccw is None:
            ccw = is_ccw_2d(a, b, point, True)
        else:
            if ccw != is_ccw_2d(a, b, point, True):
                return False
    return True


def is_point_in_polygon_2d(point, polygon):
    """Verify if a point is in the interior of a polygon lying in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        polygon (sequence) : A sequence of XY(Z) coordinates of 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a polygon.
        The vertices are assumed to be in order. The polygon is assumed to be closed:
        the first and last vertex in the sequence should not be the same.

    Warning:
        A boundary check is not yet implemented.
        This should include a tolerance value.

    Returns:
        bool: True if the point is in the polygon, False otherwise.
    """
    x, y = point[0], point[1]
    polygon = [(p[0], p[1]) for p in polygon]  # make 2D
    inside = False
    for i in range(-1, len(polygon) - 1):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
    return inside


def is_point_in_triangle_2d(point, triangle):
    """Verify if a point is in the interior of a triangle lying in the XY-plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        triangle (sequence) : A sequence of XY(Z) coordinates of three 2D or 3D points
        (Z will be ignored) representing the locations of the corners of a triangle.

    Returns:
        bool: True if the point is in the convex polygon, False otherwise.
    """
    a, b, c = triangle
    ccw = is_ccw_2d(c, a, point, True)
    if ccw != is_ccw_2d(a, b, point, True):
        return False
    if ccw != is_ccw_2d(b, c, point, True):
        return False
    return True


def is_point_in_circle_2d(point, circle):
    """Verify if a point lies in a circle lying in the XY plane.

    Parameters:
        point (sequence of float): XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
        circle (tuple): center, radius of the circle in the xy plane.

    Returns:
        (bool): True if the point lies in the circle, False otherwise.

    """
    dis = distance_point_point_2d(point, circle[0])
    if dis <= circle[1]:
        return True
    return False


def is_intersection_line_line_2d(l1, l2):
    """Verify if two lines intersect in 2d lying in the XY plane.

    Parameters:
        l1 (tuple):
        l2 (tuple):

    Returns:
        (bool): True if there is a intersection, False otherwise.

    """
    raise NotImplementedError


def is_intersection_segment_segment_2d(ab, cd):
    """Verify if two the segments ab and cd intersect?

    Two segments a-b and c-d intersect, if both of the following conditions are true:

        * c is on the left of ab, and d is on the right, or vice versa
        * d is on the left of ac, and on the right of bc, or vice versa

    Parameters:
        ab: (tuple): A sequence of XY(Z) coordinates of two 2D or 3D points
        (Z will be ignored) representing the start and end points of a line segment.
        cd: (tuple): A sequence of XY(Z) coordinates of two 2D or 3D points
        (Z will be ignored) representing the start and end points of a line segment.

    Returns:
        bool: ``True`` if the segments intersect, ``False`` otherwise.

    """
    a, b = ab
    c, d = cd
    return is_ccw_2d(a, c, d) != is_ccw_2d(b, c, d) and is_ccw_2d(a, b, c) != is_ccw_2d(a, b, d)


# ==============================================================================
# intersections
# ==============================================================================


def intersection_line_line_2d(ab, cd):
    """Compute the intersection of two lines in the XY plane.

    Parameters:
        ab: (tuple): A sequence of XY(Z) coordinates of two 2D or 3D points
        (Z will be ignored) representing two points on the line.
        cd: (tuple): A sequence of XY(Z) coordinates of two 2D or 3D points
        (Z will be ignored) representing two points on the line.

    Returns:
        None: if there is no intersection point (parallel lines).
        list: XY coordinates of intersection point.

    Note:
        If the lines are parallel, there is no intersection point.

    References:
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    """
    x1, y1 = ab[0][0], ab[0][1]
    x2, y2 = ab[1][0], ab[1][1]
    x3, y3 = cd[0][0], cd[0][1]
    x4, y4 = cd[1][0], cd[1][1]

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0.0:
        return None
    a = (x1 * y2 - y1 * x2)
    b = (x3 * y4 - y3 * x4)
    x = (a * (x3 - x4) - (x1 - x2) * b) / d
    y = (a * (y3 - y4) - (y1 - y2) * b) / d
    return x, y, 0.0


def intersection_lines_2d(lines):
    """Compute the intersections of mulitple lines in the XY plane.

    Parameters:
        lines: (sequence): A list of sequences of XY(Z) coordinates of two 2D or 3D points
        (Z will be ignored) representing the lines.

    Returns:
        None: if there is no intersection point (parallel lines).
        list: XY coordinates of intersection point.

    Note:
        If the lines are parallel, there is no intersection point.

    References:
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    """
    pdic = []
    for a, b in itertools.combinations(lines, 2):
        intx = intersection_line_line_2d(a, b)
        if not intx:
            continue
        pdic.append(intx)
    if pdic:
        return pdic
    return None


def intersection_segment_segment_2d(ab, cd, tol=0.):
    
    intx_pt = intersection_line_line_2d(ab, cd)
    if not intx_pt:
        return None
    if not is_point_on_segment_2d(intx_pt, ab, tol):
        return None
    if not is_point_on_segment_2d(intx_pt, cd, tol):
        return None   
    return intx_pt

def intersection_segments_2d(segments):
    raise NotImplementedError


def intersection_circle_circle_2d(circle1, circle2):
    """Calculates the intersection points of two circles in 2d lying in the XY plane.

    Parameters:
        circle1 (tuple): center, radius of the first circle in the xy plane.
        circle2 (tuple): center, radius of the second circle in the xy plane.

    Returns:
        points (list of tuples): the intersection points if there are any
        None: if there are no intersection points

    """
    p1, r1 = circle1[0], circle1[1]
    p2, r2 = circle2[0], circle2[1]
    d = distance_point_point_2d(p1, p2)
    if d > r1 + r2:
        return None
    if d < abs(r1 - r2):
        return None
    if (d == 0) and (r1 == r2):
        return None
    a   = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h   = (r1 * r1 - a * a) ** 0.5
    cx2 = p1[0] + a * (p2[0] - p1[0]) / d
    cy2 = p1[1] + a * (p2[1] - p1[1]) / d
    i1  = ((cx2 + h * (p2[1] - p1[1]) / d), (cy2 - h * (p2[0] - p1[0]) / d), 0)
    i2  = ((cx2 - h * (p2[1] - p1[1]) / d), (cy2 + h * (p2[0] - p1[0]) / d), 0)
    return i1, i2


# ==============================================================================
# transformations
# ==============================================================================


def translate_points_2d(points, vector):
    return [add_vectors_2d(point, vector) for point in points]


def translate_lines_2d(lines, vector):
    sps, eps = zip(*lines)
    sps = translate_points_2d(sps, vector)
    eps = translate_points_2d(eps, vector)
    return zip(sps, eps)


# ------------------------------------------------------------------------------
# rotate
# ------------------------------------------------------------------------------


def rotate_points_2d(points, axis, angle, origin=None):
    """Rotates points around an arbitrary axis in 2D.

    Parameters:
        points (sequence of sequence of float): XY coordinates of the points.
        axis (sequence of float): The rotation axis.
        angle (float): the angle of rotation in radians.
        origin (sequence of float): Optional. The origin of the rotation axis.
            Default is ``[0.0, 0.0, 0.0]``.

    Returns:
        list: the rotated points

    References:
        https://en.wikipedia.org/wiki/Rotation_matrix

    """
    if not origin:
        origin = [0.0, 0.0]
    # rotation matrix
    x, y = normalize_vector_2d(axis)
    cosa = cos(angle)
    sina = sin(angle)
    R = [[cosa, -sina], [sina, cosa]]
    # translate points
    points = translate_points_2d(points, scale_vector_2d(origin, -1.0))
    # rotate points
    points = [multiply_matrix_vector(R, point) for point in points]
    # translate points back
    points = translate_points_2d(points, origin)
    return points


# ------------------------------------------------------------------------------
# mirror
# ------------------------------------------------------------------------------


def mirror_point_point_2d(point, mirror):
    """Mirror a point about a point.

    Parameters:
        point (sequence of float): XY coordinates of the point to mirror.
        mirror (sequence of float): XY coordinates of the mirror point.

    """
    return add_vectors_2d(mirror, subtract_vectors_2d(mirror, point))


def mirror_points_point_2d(points, mirror):
    """Mirror multiple points about a point."""
    return [mirror_point_point_2d(point, mirror) for point in points]


def mirror_point_line_2d(point, line):
    pass


def mirror_points_line_2d(points, line):
    pass


# ------------------------------------------------------------------------------
# project (not the same as pull) => projection direction is required
# ------------------------------------------------------------------------------


def project_point_line_2d(point, line):
    """Project a point onto a line.

    Parameters:
        point (sequence of float): XY coordinates.
        line (tuple): Two points defining a line.

    Returns:
        list: XY coordinates of the projected point.

    References:
        https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line

    """
    a, b = line
    ab = subtract_vectors_2d(b, a)
    ap = subtract_vectors_2d(point, a)
    c = vector_component_2d(ap, ab)
    return add_vectors_2d(a, c)


def project_points_line_2d(points, line):
    """Project multiple points onto a line."""
    return [project_point_line_2d(point, line) for point in points]


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
