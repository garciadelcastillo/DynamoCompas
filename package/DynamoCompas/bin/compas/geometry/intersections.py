""""""

from __future__ import print_function
from __future__ import division

from math import fabs

from compas.geometry.basic import add_vectors
from compas.geometry.basic import subtract_vectors
from compas.geometry.basic import scale_vector
from compas.geometry.basic import cross_vectors
from compas.geometry.basic import dot_vectors
from compas.geometry.basic import normalize_vector
from compas.geometry.basic import length_vector_2d
from compas.geometry.basic import subtract_vectors_2d


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'intersection_line_line',
    'intersection_line_line_2d',
    'intersection_segment_segment',
    'intersection_segment_segment_2d',
    'intersection_circle_circle',
    'intersection_circle_circle_2d',
    'intersection_plane_plane',
    'intersection_plane_plane_plane',
    'intersection_lines',
    'intersection_lines_2d',
    'intersection_planes',
    'intersection_line_triangle',
    'intersection_line_plane',
    'intersection_segment_plane',
    'is_intersection_line_line',
    'is_intersection_line_plane',
    'is_intersection_segment_plane',
    'is_intersection_plane_plane',
    'is_intersection_line_triangle',
]


# is_intersection_line_line => line_line_intersect
# if is_intersection_line_line ... => if line_line_intersect ...
# => if lines_intersect ...

# ..._2d => ..._xy


def intersection_line_line(l1, l2):
    """Computes the intersection of two lines.

    Parameters
    ----------
    l1 : tuple, list
        XYZ coordinates of two points defining the first line.
    l2 : tuple, list
        XYZ coordinates of two points defining the second line.

    Returns
    -------
    list
        XYZ coordinates of the two points marking the shortest distance between the lines.
        If the lines intersect, these two points are identical.
        If the lines are skewed and thus only have an apparent intersection, the two
        points are different.
        If the lines are parallel, ...

    Examples
    --------
    >>>

    """
    a, b = l1
    c, d = l2

    ab = subtract_vectors(b, a)
    cd = subtract_vectors(d, c)

    n  = cross_vectors(ab, cd)
    n1 = cross_vectors(ab, n)
    n2 = cross_vectors(cd, n)

    plane_1 = (a, n1)
    plane_2 = (c, n2)

    i1 = intersection_line_plane(l1, plane_2)
    i2 = intersection_line_plane(l2, plane_1)

    return [i1, i2]


def is_intersection_line_line(l1, l2, epsilon=1e-6):
    """Verifies if two lines intersection in one point.

    Parameters:
        ab: (tuple): A sequence of XYZ coordinates of two 3D points representing
            two points on the line.
        cd: (tuple): A sequence of XYZ coordinates of two 3D points representing
            two points on the line.

    Returns:
        True (bool): if the lines intersect in one point, False is the lines are
        skew, parallel or lie on top of each other.
    """
    a, b = l1
    c, d = l2

    e1 = normalize_vector(subtract_vectors(b, a))
    e2 = normalize_vector(subtract_vectors(d, c))

    # check for parallel lines
    if abs(dot_vectors(e1, e2)) > 1.0 - epsilon:
        return False

    # check for intersection
    d_vector = cross_vectors(e1, e2)
    if dot_vectors(d_vector, subtract_vectors(c, a)) == 0:
        return True

    return False


def intersection_line_line_2d(l1, l2):
    """Compute the intersection of two lines, assuming they lie in the XY plane.

    Parameters
    ----------
    ab : tuple
        XY(Z) coordinates of two points defining a line.
    cd : tuple
        XY(Z) coordinates of two points defining another line.

    Returns
    -------
    None
        If there is no intersection point (parallel lines).
    list
        XYZ coordinates of intersection point if one exists (Z = 0).

    Note
    ----
    Only if the lines are parallel, there is no intersection point.

    References
    ----------
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    """
    a, b = l1
    c, d = l2

    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if d == 0.0:
        return None

    a = (x1 * y2 - y1 * x2)
    b = (x3 * y4 - y3 * x4)
    x = (a * (x3 - x4) - (x1 - x2) * b) / d
    y = (a * (y3 - y4) - (y1 - y2) * b) / d

    return x, y, 0.0


def is_intersection_line_line_2d(l1, l2):
    """Verify if two lines intersect in 2d lying in the XY plane.

    Parameters:
        l1 (tuple):
        l2 (tuple):

    Returns:
        (bool): True if there is a intersection, False otherwise.

    """
    raise NotImplementedError


def intersection_segment_segment(ab, cd, tol=0.0):
    """"""
    intx_pt = intersection_line_line(ab, cd)

    if not intx_pt:
        return None

    if not is_point_on_segment(intx_pt, ab, tol):
        return None

    if not is_point_on_segment(intx_pt, cd, tol):
        return None   

    return intx_pt


def is_intersection_segment_segment():
    raise NotImplementedError


def intersection_segment_segment_2d(ab, cd, tol=0.):
    """"""
    intx_pt = intersection_line_line_2d(ab, cd)

    if not intx_pt:
        return None

    if not is_point_on_segment_2d(intx_pt, ab, tol):
        return None

    if not is_point_on_segment_2d(intx_pt, cd, tol):
        return None   

    return intx_pt


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


def intersection_circle_circle():
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

    d = length_vector_2d(subtract_vectors_2d(p2, p1))

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


def intersection_line_triangle(line, triangle, epsilon=1e-6):
    """
    Computes the intersection point of a line (ray) and a triangle
    based on the Moeller Trumbore intersection algorithm

    Parameters:
        line (tuple): Two points defining the line.
        triangle (sequence of sequence of float): XYZ coordinates of the triangle corners.

    Returns:
        point (tuple) if the line (ray) intersects with the triangle, None otherwise.

    Note:
        The line is treated as continues, directed ray and not as line segment with a start and end point
    """
    a, b, c = triangle
    v1 = subtract_vectors(line[1], line[0])
    p1 = line[0]
    # Find vectors for two edges sharing V1
    e1 = subtract_vectors(b, a)
    e2 = subtract_vectors(c, a)
    # Begin calculating determinant - also used to calculate u parameter
    p = cross_vectors(v1, e2)
    # if determinant is near zero, ray lies in plane of triangle
    det = dot_vectors(e1, p)
    # NOT CULLING
    if(det > - epsilon and det < epsilon):
        return None
    inv_det = 1.0 / det
    # calculate distance from V1 to ray origin
    t = subtract_vectors(p1, a)
    # Calculate u parameter and make_blocks bound
    u = dot_vectors(t, p) * inv_det
    # The intersection lies outside of the triangle
    if(u < 0.0 or u > 1.0):
        return None
    # Prepare to make_blocks v parameter
    q = cross_vectors(t, e1)
    # Calculate V parameter and make_blocks bound
    v = dot_vectors(v1, q) * inv_det
    # The intersection lies outside of the triangle
    if(v < 0.0 or u + v  > 1.0):
        return None
    t = dot_vectors(e2, q) * inv_det
    if t > epsilon:
        return add_vectors(p1, scale_vector(v1, t))
    # No hit
    return None


def is_intersection_line_triangle(line, triangle, epsilon=1e-6):
    """Verifies if a line (ray) intersects with a triangle.

    Notes
    -----
    Based on the Moeller Trumbore intersection algorithm.
    The line is treated as continues, directed ray and not as line segment with a start and end point

    Parameters
    ----------
    line : tuple
        Two points defining the line.
    triangle : sequence of sequence of float
        XYZ coordinates of the triangle corners.

    Returns
    -------
    bool
        True if the line (ray) intersects with the triangle, False otherwise.

    Examples
    --------
    >>>

    """
    a, b, c = triangle
    # direction vector and base point of line
    v1 = subtract_vectors(line[1], line[0])
    p1 = line[0]
    # Find vectors for two edges sharing V1
    e1 = subtract_vectors(b, a)
    e2 = subtract_vectors(c, a)
    # Begin calculating determinant - also used to calculate u parameter
    p = cross_vectors(v1, e2)
    # if determinant is near zero, ray lies in plane of triangle
    det = dot_vectors(e1, p)

    # NOT CULLING
    if det > - epsilon and det < epsilon:
        return False

    inv_det = 1.0 / det
    # calculate distance from V1 to ray origin
    t = subtract_vectors(p1, a)
    # Calculate u parameter and make_blocks bound
    u = dot_vectors(t, p) * inv_det

    # The intersection lies outside of the triangle
    if u < 0.0 or u > 1.0:
        return False

    # Prepare to make_blocks v parameter
    q = cross_vectors(t, e1)
    # Calculate V parameter and make_blocks bound
    v = dot_vectors(v1, q) * inv_det

    # The intersection lies outside of the triangle
    if v < 0.0 or u + v  > 1.0:
        return False

    t = dot_vectors(e2, q) * inv_det

    if t > epsilon:
        return True

    # No hit
    return False


def intersection_line_plane(line, plane, epsilon=1e-6):
    """Computes the intersection point of a line (ray) and a plane

    Parameters:
        line (tuple): Two points defining the line.
        plane (tuple): The base point and normal defining the plane.

    Returns:
        point (tuple) if the line (ray) intersects with the plane, None otherwise.

    """
    pt1 = line[0]
    pt2 = line[1]
    p_cent = plane[0]
    p_norm = plane[1]

    v1 = subtract_vectors(pt2, pt1)
    dot = dot_vectors(p_norm, v1)

    if fabs(dot) > epsilon:
        v2 = subtract_vectors(pt1, p_cent)
        fac = -dot_vectors(p_norm, v2) / dot
        vec = scale_vector(v1, fac)
        return add_vectors(pt1, vec)

    return None


def is_intersection_line_plane(line, plane, epsilon=1e-6):
    """Verify if a line (continuous ray) intersects with a plane.

    Parameters:
        line (tuple): Two points defining the line.
        plane (tuple): The base point and normal defining the plane.
    Returns:
        (bool): True if the line intersects with the plane, False otherwise.

    """
    pt1 = line[0]
    pt2 = line[1]
    p_norm = plane[1]

    v1 = subtract_vectors(pt2, pt1)
    dot = dot_vectors(p_norm, v1)

    if fabs(dot) > epsilon:
        return True
    return False


def intersection_segment_plane(segment, plane, epsilon=1e-6):
    """Computes the intersection point of a line segment and a plane

    Parameters:
        segment (tuple): Two points defining the line segment.
        plane (tuple): The base point and normal defining the plane.

    Returns:
        point (tuple) if the line segment intersects with the plane, None otherwise.

    """
    pt1 = segment[0]
    pt2 = segment[1]
    p_cent = plane[0]
    p_norm = plane[1]

    v1 = subtract_vectors(pt2, pt1)
    dot = dot_vectors(p_norm, v1)

    if fabs(dot) > epsilon:
        v2 = subtract_vectors(pt1, p_cent)
        fac = - dot_vectors(p_norm, v2) / dot
        if fac > 0. and fac < 1.:
            vec = scale_vector(v1, fac)
            return add_vectors(pt1, vec)
        return None
    else:
        return None


def is_intersection_segment_plane(segment, plane, epsilon=1e-6):
    """Verify if a line segment intersects with a plane.

    Parameters:
        segment (tuple): Two points defining the line segment.
        plane (tuple): The base point and normal defining the plane.
    Returns:
        (bool): True if the line segment intersects with the plane, False otherwise.

    """
    pt1 = segment[0]
    pt2 = segment[1]
    p_cent = plane[0]
    p_norm = plane[1]

    v1 = subtract_vectors(pt2, pt1)
    dot = dot_vectors(p_norm, v1)

    if fabs(dot) > epsilon:
        v2 = subtract_vectors(pt1, p_cent)
        fac = - dot_vectors(p_norm, v2) / dot
        if fac > 0. and fac < 1.:
            return True
        return False
    else:
        return False


def intersection_plane_plane(plane1, plane2, epsilon=1e-6):
    """Computes the intersection of two planes

    Parameters:
        plane1 (tuple): The base point and normal (normalized) defining the 1st plane.
        plane2 (tuple): The base point and normal (normalized) defining the 2nd plane.

    Returns:
        line (tuple): Two points defining the intersection line. None if planes are parallel.

    """
    # check for parallelity of planes
    if abs(dot_vectors(plane1[1], plane2[1])) > 1 - epsilon:
        return None
    vec = cross_vectors(plane1[1], plane2[1])  # direction of intersection line
    p1 = plane1[0]
    vec_inplane = cross_vectors(vec, plane1[1])
    p2 = add_vectors(p1, vec_inplane)
    px1 = intersection_line_plane((p1, p2), plane2)
    px2 = add_vectors(px1, vec)
    return px1, px2


def is_intersection_plane_plane(plane1, plane2, epsilon=1e-6):
    """Computes the intersection of two planes

    Parameters:
        plane1 (tuple): The base point and normal (normalized) defining the 1st plane.
        plane2 (tuple): The base point and normal (normalized) defining the 2nd plane.
    Returns:
        (bool): True if the planes intersect, False otherwise.

    """
    # check for parallelity of planes
    if abs(dot_vectors(plane1[1], plane2[1])) > 1 - epsilon:
        return False
    return True


def intersection_plane_plane_plane(plane1, plane2, plane3, epsilon=1e-6):
    """Computes the intersection of three planes

    Parameters:
        plane1 (tuple): The base point and normal (normalized) defining the 1st plane.
        plane2 (tuple): The base point and normal (normalized) defining the 2nd plane.

    Returns:
        point (tuple): The intersection point. None if two (or all three) planes are parallel.

    Note:
        Currently this only computes the intersection point. E.g.: If two planes
        are parallel the intersection lines are not computed. see:
        http://geomalgorithms.com/Pic_3-planes.gif
    """
    line = intersection_plane_plane(plane1, plane2, epsilon)
    if not line:
        return None
    pt = intersection_line_plane(line, plane3, epsilon)
    if pt:
        return pt
    return None


def intersection_lines():
    raise NotImplementedError


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
    import itertools

    pdic = []
    for a, b in itertools.combinations(lines, 2):
        intx = intersection_line_line_2d(a, b)
        if not intx:
            continue
        pdic.append(intx)
    if pdic:
        return pdic
    return None


def intersection_planes():
    raise NotImplementedError


# def is_intersection_box_box(box_1, box_2):
#     """Checks if two boxes are intersecting in 3D.

#     Parameters:
#         box_1 (list of tuples): list of 8 points (bottom: 0,1,2,3 top: 4,5,6,7)
#         box_2 (list of tuples): list of 8 points (bottom: 0,1,2,3 top: 4,5,6,7)

#     Returns:
#         bool: True if the boxes intersect, False otherwise.

#     Examples:

#         .. code-block:: python

#             x, y, z = 1, 1, 1
#             box_a = [
#                 (0.0, 0.0, 0.0),
#                 (x,   0.0, 0.0),
#                 (x,   y,   0.0),
#                 (0.0, y,   0.0),
#                 (0.0, 0.0, z),
#                 (x,   0.0, z),
#                 (x,   y,   z),
#                 (0.0, y,   z)
#             ]
#             box_b = [
#                 (0.5, 0.5, 0.0),
#                 (1.5, 0.5, 0.0),
#                 (1.5, 1.5, 0.0),
#                 (0.5, 1.5, 0.0),
#                 (0.5, 0.5, 1.0),
#                 (1.5, 0.5, 1.0),
#                 (1.5, 1.5, 1.0),
#                 (0.5, 1.5, 1.0)
#             ]
#             if is_box_intersecting_box(box_a, box_b):
#                 print("intersection found")
#             else:
#                 print("no intersection found")

#     Warning:
#         Does not check if one box is completely enclosed by the other.

#     """
#     # all edges of box one
#     edges = [
#         (box_1[0], box_1[1]),
#         (box_1[1], box_1[2]),
#         (box_1[2], box_1[3]),
#         (box_1[3], box_1[0])
#     ]
#     edges += [
#         (box_1[4], box_1[5]),
#         (box_1[5], box_1[6]),
#         (box_1[6], box_1[7]),
#         (box_1[7], box_1[4])
#     ]
#     edges += [
#         (box_1[0], box_1[4]),
#         (box_1[1], box_1[5]),
#         (box_1[2], box_1[6]),
#         (box_1[3], box_1[7])
#     ]
#     # triangulation of box two
#     tris = [
#         (box_2[0], box_2[1], box_2[2]),
#         (box_2[0], box_2[2], box_2[3])
#     ]  # bottom
#     tris += [
#         (box_2[4], box_2[5], box_2[6]),
#         (box_2[4], box_2[6], box_2[7])
#     ]  # top
#     tris += [
#         (box_2[0], box_2[4], box_2[7]),
#         (box_2[0], box_2[7], box_2[3])
#     ]  # side 1
#     tris += [
#         (box_2[0], box_2[1], box_2[5]),
#         (box_2[0], box_2[5], box_2[4])
#     ]  # side 2
#     tris += [
#         (box_2[1], box_2[2], box_2[6]),
#         (box_2[1], box_2[6], box_2[5])
#     ]  # side 3
#     tris += [
#         (box_2[2], box_2[3], box_2[7]),
#         (box_2[2], box_2[7], box_2[6])
#     ]  # side 4
#     # checks for edge triangle intersections
#     intx = False
#     for pt1, pt2 in edges:
#         for tri in tris:
#             for line in [(pt1, pt2), (pt2, pt1)]:
#                 test_pt = intersection_line_triangle(line, tri)
#                 if test_pt:
#                     if is_point_on_segment(test_pt, line):
#                         # intersection found
#                         intx = True
#                         break
#             else:
#                 continue
#             break
#         else:
#             continue
#         break
#     return intx


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
