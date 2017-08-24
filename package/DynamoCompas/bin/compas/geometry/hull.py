from __future__ import print_function

from compas.geometry import cross_vectors
from compas.geometry import subtract_vectors
from compas.geometry import dot_vectors
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import intersection_plane_plane_plane
from compas.geometry import distance_point_point

from compas.geometry import cross_vectors_2d

from compas.datastructures.mesh import Mesh


__author__     = ['Matthias Rippmann <rippmann@ethz.ch>']
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = '<rippmannt@ethz.ch>'


__all__ = [
    'convex_hull',
    'convex_hull_2d',
    'polyhedron_from_node'
]


def _normal_face(points, face):
    u = subtract_vectors(points[face[1]], points[face[0]])
    v = subtract_vectors(points[face[-1]], points[face[0]])
    return cross_vectors(u, v)


def _seen(points, face, p):
    normal = _normal_face(points, face)
    vec = subtract_vectors(points[p], points[face[0]])
    return (dot_vectors(normal, vec) >= 0)


def _bdry(faces):
    bdry_fw = set([(face[i - 1], face[i]) for face in faces for i in range(len(face))])
    bdry_bk = set([(face[i], face[i - 1]) for face in faces for i in range(len(face))])
    return bdry_fw - bdry_bk


def _add_point(points, hull, p):
    seen_faces = [face for face in hull if _seen(points, face, p)]
    if len(seen_faces) == len(hull):
        # if can see all faces, unsee ones looking "down"
        normal = _normal_face(points, seen_faces[0])
        seen_faces = [face for face in seen_faces if dot_vectors(_normal_face(points, face), normal) > 0]
    for face in seen_faces:
        hull.remove(face)
    for edge in _bdry(seen_faces):
        hull.append([edge[0], edge[1], p])


def convex_hull(points):
    """Construct convex hull for a set of points.

    Parameters:
        points (sequence): A sequence of XYZ coordinates.

    Returns:
        faces (sequence of sequences of integers): the triangular faces of the convex hull

    References:
        https://gist.github.com/anonymous/5184ba0bcab21d3dd19781efd3aae543

    Note:
        The algorithm is not optimized and relatively slow on large sets of points.
        See here for a more optimized version of this algorithm:
        http://thomasdiewald.com/blog/?p=1888

    Examples:

        .. code-block:: python

            import math
            import random

            from compas.geometry import distance_point_point
            from compas.cad.rhino.helpers.mesh import draw_mesh
            from compas.datastructures.mesh import Mesh


            radius = 5
            origin = (0., 0., 0.)
            count = 0
            points = []
            while count < 1000:
                x = (random.random() - 0.5) * radius * 2
                y = (random.random() - 0.5) * radius * 2
                z = (random.random() - 0.5) * radius * 2
                pt = x, y, z
                if distance_point_point(origin, pt) <= radius:
                    points.append(pt)
                    count += 1

            faces =  convex_hull(points)

            mesh = Mesh.from_vertices_and_faces(points, faces)

            draw_mesh(mesh,
                        show_faces = True,
                        show_vertices = False,
                        show_edges = False)

    """
    hull = [[0, 1, 2], [0, 2, 1]]
    for i in range(3, len(points)):
        _add_point(points, hull, i)
    return hull


# https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
def convex_hull_2d(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # # Returns a positive value, if OAB makes a counter-clockwise turn,
    # # negative for clockwise turn, and zero if the points are collinear.
    # def cross(o, a, b):
    #     return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_vectors_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_vectors_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


def polyhedron_from_node(cent, points, scale=1., tolerance=1e-6):
    """Computes polyhedra from a spatial node.

    Parameters:

        cent (sequence of float): XYZ coordinates of the central node vertex
        points (sequence of sequence of float): XYZ coordinates of the neighboring vertices

    Returns:
        points (sequence of sequences of floats):  the XYZ coordinates of the
        vertices of the polyhedra
        faces (sequence of sequences of integers): the faces of the polyhedra

    Examples:

        .. code-block:: python

            from compas.datastructures.network import Network

            import rhinoscriptsyntax as rs


            crvs = rs.GetObjects("Select Edges", 4)
            lines = [[rs.CurveStartPoint(crv), rs.CurveEndPoint(crv)] for crv in crvs]
            network = Network.from_lines(lines)

            keys = []
            for key in network.vertices():
                if not network.is_vertex_leaf(key):
                    keys.append(key)

            for key in keys:
                nbrs = network.neighbours(key)
                cent = network.vertex_coordinates(key)
                points = [network.vertex_coordinates(nbr) for nbr in nbrs]

                points, faces = polyhedron_from_node(cent, points, 1)
                for face in faces:
                    pts = [points[i] for i in face]
                    poly = rs.AddPolyline(pts+[pts[0]])
                    rs.AddPlanarSrf(poly)

    """
    pts_hull = []
    for pt in points:
        vec = subtract_vectors(pt, cent)
        vec = scale_vector(normalize_vector(vec), scale)
        pt = add_vectors(cent, vec)
        pts_hull.append(pt)

    faces = convex_hull(pts_hull)
    mesh = Mesh.from_vertices_and_faces(pts_hull, faces)

    planes = {}
    for key in mesh.vertices():
        vec = subtract_vectors(mesh.vertex_coordinates(key), cent)
        planes[key] = [mesh.vertex_coordinates(key), vec]

    faces = []
    pts = []
    for key in mesh.vertices():
        nbrs = mesh.vertex_neighbours(key, True)
        face = []
        for i in range(len(nbrs)):
            pt = intersection_plane_plane_plane(planes[nbrs[i - 1]], planes[nbrs[i]], planes[key])
            if not pt:
                continue
            dup = False
            n = len(pts)
            for j in range(n):
                if distance_point_point(pt, pts[j]) < tolerance:
                    face.append(j)
                    dup = True
                    break
            if not dup:
                face.append(n)
                pts.append(pt)
        if face:
            faces.append(face)

    return pts, faces


# ==============================================================================
# Debugging
# ==============================================================================


if __name__ == "__main__":

    # todo: distinguish between vertices of hull and internal vertices

    import random

    from compas.visualization.viewers.meshviewer import MeshViewer

    radius = 5
    origin = (0., 0., 0.)
    count = 0
    points = []

    while count < 1000:
        x = (random.random() - 0.5) * radius * 2
        y = (random.random() - 0.5) * radius * 2
        z = (random.random() - 0.5) * radius * 2
        pt = x, y, z
        if distance_point_point(origin, pt) <= radius:
            points.append(pt)
            count += 1

    faces = convex_hull(points)

    mesh = Mesh.from_vertices_and_faces(points, faces)

    viewer = MeshViewer(mesh)

    viewer.setup()
    viewer.show()
