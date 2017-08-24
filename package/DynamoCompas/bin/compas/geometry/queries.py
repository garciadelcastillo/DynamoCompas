""""""

from __future__ import print_function
from __future__ import division

from math import pi

from compas.geometry.basic import subtract_vectors
from compas.geometry.basic import cross_vectors
from compas.geometry.basic import dot_vectors

from compas.geometry.distance import distance_point_point
from compas.geometry.distance import distance_point_point_2d
from compas.geometry.distance import distance_point_plane
from compas.geometry.distance import distance_point_line
from compas.geometry.distance import distance_point_line_2d
from compas.geometry.distance import closest_point_on_segment

from compas.geometry.angles import angle_smallest_vectors
from compas.geometry.average import center_of_mass_polygon


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'is_colinear',
    'is_coplanar',
    'is_polygon_convex',
    'is_point_on_plane',
    'is_point_on_line',
    'is_point_on_segment',
    'is_point_on_polyline',
    'is_point_in_triangle',
    'is_point_in_triangle_2d',
    'is_point_in_polygon_2d',
    'is_point_in_convex_polygon_2d',
    'is_point_in_circle_2d',
]


def is_ccw_2d(a, b, c, colinear=False):
    """Verify if ``c`` is on the left of ``ab`` when looking from ``a`` to ``b``,
    and assuming that all points lie in the XY plane.

    Parameters
    ----------
    a : sequence of float
        XY(Z) coordinates of the base point.
    b : sequence of float
        XY(Z) coordinates of the first end point.
    c : sequence of float
        XY(Z) coordinates of the second end point.
    colinear : bool, optional
        Allow points to be colinear.
        Default is ``False``.

    Returns
    -------
    bool
        ``True`` if ccw.
        ``False`` otherwise.

    Examples
    --------
    >>> print(is_ccw_2d([0,0,0], [0,1,0], [-1, 0, 0]))
    True

    >>> print(is_ccw_2d([0,0,0], [0,1,0], [+1, 0, 0]))
    False

    >>> print(is_ccw_2d([0,0,0], [1,0,0], [2,0,0]))
    False

    >>> print(is_ccw_2d([0,0,0], [1,0,0], [2,0,0], True))
    True

    References
    ----------
    https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation

    """
    ab_x = b[0] - a[0]
    ab_y = b[1] - a[1]
    ac_x = c[0] - a[0]
    ac_y = c[1] - a[1]

    if colinear:
        return ab_x * ac_y - ab_y  * ac_x >= 0
    return ab_x * ac_y - ab_y  * ac_x > 0


def is_colinear(a, b, c):
    """Verify if three points are colinear.

    Parameters
    ----------
    a : tuple, list, Point
        Point 1.
    b : tuple, list, Point
        Point 2.
    c : tuple, list, Point
        Point 3.

    Returns
    -------
    bool
        ``True`` if the points are collinear
        ``False`` otherwise.

    """
    raise NotImplementedError


def is_colinear_2d(a, b, c):
    """"""
    ab_x = b[0] - a[0]
    ab_y = b[1] - a[1]
    ac_x = c[0] - a[0]
    ac_y = c[1] - a[1]

    return ab_x * ac_y == ab_y  * ac_x


def is_coplanar(points, tol=0.01):
    """Verify if the points are coplanar.

    Compute the normal vector (cross product) of the vectors formed by the first
    three points. Include one more vector at a time to compute a new normal and
    compare with the original normal. If their cross product is not zero, they
    are not parallel, which means the point are not in the same plane.

    Four points are coplanar if the volume of the tetrahedron defined by them is
    0. Coplanarity is equivalent to the statement that the pair of lines
    determined by the four points are not skew, and can be equivalently stated
    in vector form as (x2 - x0).[(x1 - x0) x (x3 - x2)] = 0.

    Parameters
    ----------
    points : sequence
        A sequence of locations in three-dimensional space.

    Returns
    -------
    bool
        ``True`` if the points are coplanar.
        ``False`` otherwise.

    """
    tol2 = tol ** 2

    if len(points) == 4:
        v01 = subtract_vectors(points[1], points[0])
        v02 = subtract_vectors(points[2], points[0])
        v23 = subtract_vectors(points[3], points[0])
        res = dot_vectors(v02, cross_vectors(v01, v23))
        return res**2 < tol2

    # len(points) > 4
    # compare length of cross product vector to tolerance

    u = subtract_vectors(points[1], points[0])
    v = subtract_vectors(points[2], points[1])
    w = cross_vectors(u, v)

    for i in range(1, len(points) - 2):
        u = v
        v = subtract_vectors(points[i + 2], points[i + 1])
        wuv = cross_vectors(w, cross_vectors(u, v))

        if wuv[0]**2 > tol2 or wuv[1]**2 > tol2 or wuv[2]**2 > tol2:
            return False

    return True


def is_polygon_convex(polygon):
    """Verify if a polygon is convex.

    Parameters
    ----------
    polygon : sequence of sequence of floats
        The XYZ coordinates of the corners of the polygon.

    Note
    ----
    Use this function for *spatial* polygons.
    If the polygon is in a horizontal plane, use :func:`is_polygon_convex_2d` instead.

    See Also
    --------
    is_polygon_convex_2d

    """
    c = center_of_mass_polygon(polygon)

    for i in range(-1, len(polygon) - 1):
        p0 = polygon[i]
        p1 = polygon[i - 1]
        p2 = polygon[i + 1]
        v0 = subtract_vectors(c, p0)
        v1 = subtract_vectors(p1, p0)
        v2 = subtract_vectors(p2, p0)
        a1 = angle_smallest_vectors(v1, v0)
        a2 = angle_smallest_vectors(v0, v2)
        if a1 + a2 > pi:
            return False

    return True


def is_polygon_convex_2d(polygon, colinear=False):
    """Verify if the polygon is convex in the XY-plane.

    Parameters
    ----------
    polygon : list, tuple
        The XY(Z) coordinates of the corners of a polygon.
        The vertices are assumed to be in order.
        The polygon is assumed to be closed: the first and last vertex in the sequence should not be the same.
    colinear : bool
        Are points allowed to be colinear?

    Returns
    -------
    bool
        ``True`` if the figure is convex.
        ``False`` otherwise.

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


def is_point_on_plane(point, plane, tol=0.0):
    """Verify if a point lies in a plane.

    Parameters
    ----------
    point : sequence of float
        XYZ coordinates.
    plane : tuple
        Base point and normal defining a plane.
    tol : float, optional
        A tolerance. Default is ``0.0``.

    Returns
    -------
    bool
        ``True`` if the point is in on the plane.
        ``False`` otherwise.

    """
    return distance_point_plane(point, plane) <= tol


def is_point_on_line(point, line, tol=0.0):
    """Verify if a point lies on a line.

    Parameters
    ----------
    point (sequence of float): XYZ coordinates.
    line (tuple): Two points defining a line.
    tol (float): Optional. A tolerance. Default is ``0.0``.

    Returns
    -------
    bool
        ``True`` if the point is in on the line.
        ``False`` otherwise.

    """
    return distance_point_line(point, line) <= tol


def is_point_on_line_2d(point, line, tol=0.0):
    """Verify if a point lies on a line in the XY-plane.

    Parameters
    ----------
    point : sequence of float
        XY(Z) coordinates of a point.
    line : tuple
        XY(Z) coordinates of two points defining a line.
    tol : float, optional
        A tolerance.
        Default is ``0.0``.

    Returns
    -------
    bool
        ``True`` if the point is in on the line.
        ``False`` otherwise.

    """
    return distance_point_line_2d(point, line) <= tol


def is_point_on_segment(point, segment, tol=0.0):
    """Verify if a point lies on a given line segment.

    Parameters
    ----------
    point : sequence of float
        XYZ coordinates.
    segment : tuple
        Two points defining the line segment.

    Returns
    -------
    bool
        ``True`` if the point is on the line segment.
        ``False`` otherwise.

    """
    a, b = segment

    if not is_point_on_line(point, segment, tol=tol):
        return False

    d_ab = distance_point_point(a, b)

    if d_ab == 0:
        return False

    d_pa = distance_point_point(a, point)
    d_pb = distance_point_point(b, point)

    if d_pa + d_pb <= d_ab + tol:
        return True

    return False


def is_point_on_segment_2d(point, segment, tol=0.0):
    """Verify if a point lies on a given line segment in the XY-plane.

    Parameters
    ----------
    point : sequence of float
        XY(Z) coordinates of a point.
    segment : tuple, list
        XY(Z) coordinates of two points defining a segment.

    Returns
    -------
    bool
        ``True`` if the point is on the line segment.
        ``False`` otherwise.

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


# def is_closest_point_on_segment(point, segment, tol=0.0, return_point=False):
#     """Verify if the closest point on the line of a segment is on the segment.

#     Parameters
#     ----------
#     point : sequence of float
#         XYZ coordinates of the point.
#     segment : tuple
#         Two points defining the line segment.
#     tol : float, optional
#         A tolerance.
#         Default is ``0.0``.
#     return_point : bool, optional
#         If ``True`` return the closest point.
#         Default is ``False``.

#     Returns
#     -------
#     bool, tuple
#         XYZ coordinates of the point on the line.
#     bool
#         True if the point is in on the line, False otherwise.

#     """
#     a, b = segment
#     v = subtract_vectors(b, a)
#     d_ab = distance_point_point_sqrd(a, b)
#     if d_ab == 0:
#         return
#     u = sum((point[i] - a[i]) * v[i] for i in range(3)) / d_ab
#     c = a[0] + u * v[0], a[1] + u * v[1], a[2] + u * v[2]
#     d_ac = distance_point_point_sqrd(a, c)
#     d_bc = distance_point_point_sqrd(b, c)
#     if d_ac + d_bc <= d_ab + tol:
#         if return_point:
#             return c
#         return True
#     return False


def is_point_on_polyline(point, polyline, tol=0.0):
    """Verify if a point is on a polyline.

    Parameters
    ----------
    point : sequence of float
        XYZ coordinates.
    polyline : sequence of sequence of float
        XYZ coordinates of the points of the polyline.
    tol : float, optional
        The tolerance.
        Default is ``0.0``.

    Returns
    -------
    bool
        ``True`` if the point is on the polyline.
        ``False`` otherwise.

    """
    for i in xrange(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        c = closest_point_on_segment(point, (a, b))

        if distance_point_point(point, c) <= tol:
            return True

    return False


def is_point_in_triangle(point, triangle):
    """Verify if a point is in the interior of a triangle.

    Parameters
    ----------
    point : sequence of float
        XYZ coordinates.
    triangle : sequence of sequence of float
        XYZ coordinates of the triangle corners.

    Returns
    -------
    bool
        True if the point is in inside the triangle.
        False otherwise.

    Note
    ----
    Should the point be in the same plane as the triangle?

    See Also
    --------
    is_point_in_triangle_2d

    """
    def is_on_same_side(p1, p2, segment):
        a, b = segment
        v = subtract_vectors(b, a)
        c1 = cross_vectors(v, subtract_vectors(p1, a))
        c2 = cross_vectors(v, subtract_vectors(p2, a))

        if dot_vectors(c1, c2) >= 0:
            return True

        return False

    a, b, c = triangle

    if is_on_same_side(point, a, (b, c)) and \
       is_on_same_side(point, b, (a, c)) and \
       is_on_same_side(point, c, (a, b)):
        return True

    return False


def is_point_in_triangle_2d(point, triangle):
    """Verify if a point is in the interior of a triangle lying in the XY-plane.

    Parameters
    ----------
    point : sequence of float
        XY(Z) coordinates of a point.
    triangle : sequence
        XY(Z) coordinates of the corners of the triangle.

    Returns
    -------
    bool
        True if the point is in the convex polygon
        False otherwise.

    """
    a, b, c = triangle
    ccw = is_ccw_2d(c, a, point, True)

    if ccw != is_ccw_2d(a, b, point, True):
        return False

    if ccw != is_ccw_2d(b, c, point, True):
        return False

    return True


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


def is_point_in_circle(point, circle):
    center, radius, normal = circle
    if is_point_on_plane(point, (center, normal)):
        return distance_point_point(point, center) <= radius
    return False


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


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
