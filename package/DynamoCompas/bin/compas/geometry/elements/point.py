from __future__ import print_function

from compas.geometry.elements.vector import Vector

from compas.geometry import distance_point_point
from compas.geometry import distance_point_line
from compas.geometry import distance_point_plane

from compas.geometry import project_point_plane
from compas.geometry import project_point_line


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'GNU - General Public License'
__email__      = 'vanmelet@ethz.ch'


class PointList(object):
    """"""

    def __init__(self):
        self.points = []

    def __get__(self, obj, objtype=None):
        return self.points

    def __set__(self, obj, points):
        self.points = [Point(xyz) for xyz in points]


class Point(object):
    """A three-dimensional location in space.

    Parameters:
        xyz (list): The xyz coordinates of the point.

    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point, defaults to 0.

    Examples:
        >>> p1 = Point([1, 2, 3])
        >>> p2 = Point([4, 5, 6])

        >>> p1.x
        1.0
        >>> p1[0]
        1.0
        >>> p1[5]
        1.0
        >>> p1[-3]
        1.0
        >>> p1[-6]
        1.0

        >>> p1 + p2
        [5.0, 7.0, 9.0]
        >>> p1 + [4, 5, 6]
        [5.0, 7.0, 9.0]
        >>> p1 * 2
        [2.0, 4.0, 6.0]
        >>> p1 ** 2
        [1.0, 4.0, 9.0]
        >>> p1
        [1.0, 2.0, 3.0]

        >>> p1 += p2
        >>> p1 *= 2
        >>> p1 **= 2
        >>> p1
        [100.0, 196.0, 324.0]

    Note:
        A ``Point`` object supports direct access to its xyz coordinates through
        the dot notation, as well list-style access using indices. Indexed
        access is implemented such that the ``Point`` behaves like a circular
        list.

    References:
        <http://stackoverflow.com/questions/8951020/pythonic-circular-list>

    """
    def __init__(self, xyz):
        self.x = float(xyz[0])
        self.y = float(xyz[1])
        try:
            self.z = float(xyz[2])
        except IndexError:
            self.z = 0.0

    def __repr__(self):
        return '[{0}, {1}, {2}]'.format(self.x, self.y, self.z)

    def __len__(self):
        return 3

    def __getitem__(self, key):
        i = key % 3
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        if i == 2:
            return self.z
        raise KeyError

    def __setitem__(self, key, value):
        i = key % 3
        if i == 0:
            self.x = value
            return
        if i == 1:
            self.y = value
            return
        if i == 2:
            self.z = value
            return
        raise KeyError

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __eq__(self, other):
        """Is this point equal to the other point? Tow points are considered
        equal if their XYZ coordinates are identical.

        Note:
            Perhaps it makes sense to add a `precision` attribute to the point
            class. This would allow comparisons to be made up to a certain
            tolerance.

        Parameters:
            other (sequence, Point): The point to compare.
        """
        return self.x == other[0] and self.y == other[1] and self.z == other[2]

    def __add__(self, other):
        return Point([self.x + other[0], self.y + other[1], self.z + other[2]])

    def __iadd__(self, other):
        self.x += other[0]
        self.y += other[1]
        self.z += other[2]
        return self

    def __sub__(self, other):
        """Create a vector from other to self.

        Parameters:
            other (sequence, Point): The point to subtract.

        Returns:
            Vector: A vector from other to self
        """
        return Vector(self, other)

    def __isub__(self, other):
        self.x -= other[0]
        self.y -= other[1]
        self.z -= other[2]
        return self

    def __mul__(self, n):
        return Point([n * self.x, n * self.y, n * self.z])

    def __imul__(self, n):
        self.x *= n
        self.y *= n
        self.z *= n
        return self

    def __pow__(self, n):
        return Point([self.x ** n, self.y ** n, self.z ** n])

    def __ipow__(self, n):
        self.x **= n
        self.y **= n
        self.z **= n
        return self

    # --------------------------------------------------------------------------
    # transformations
    # --------------------------------------------------------------------------

    def translate(self, vector):
        self.x += vector.x
        self.y += vector.y
        self.z += vector.z
        return self

    def rotate(self, angle, origin=None):
        pass

    def project_to_plane(self, plane):
        plane = (plane.point, plane.normal)
        return project_point_plane(self, plane)

    def project_to_line(self, line):
        pass

    # --------------------------------------------------------------------------
    # distance
    # --------------------------------------------------------------------------

    def distance_to_point(self, point):
        return distance_point_point(self, point)

    def distance_to_line(self, line):
        return distance_point_line(self, line)

    def distance_to_plane(self, plane):
        return distance_point_plane(self, plane)


# ==============================================================================
# Debugging
# ==============================================================================


if __name__ == '__main__':

    # p1 = Point([1, 2, 3])
    # p2 = Point([-10, 0, 0])

    # v1 = Vector([10,0,0])
    # v1.normalize()

    # p2.translate(v1)
    # print(p2)

    # from compas.geometry.elements import Plane
    # plane = Plane.from_points_and_vector([0., 0., 0.], [1., 0., 0.], [0., 0., 1.])

    # point = Point([0., 1., 1.])
    # projection = point.project_to_plane(plane)

    # print(projection)

    import timeit

    t0 = timeit.timeit('points = [Point([i, i, i]) for i in xrange(100000)]', 'from __main__ import Point', number=100)

    t1 = timeit.timeit('points = [[i, i, i] for i in xrange(100000)]', 'from __main__ import Point', number=100)

    print(t0 / t1)
