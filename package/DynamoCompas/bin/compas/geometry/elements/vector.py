from math import sin
from math import cos
from math import sqrt

from compas.geometry import dot_vectors
from compas.geometry import cross_vectors


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


class Vector(object):
    """A ``Vector`` object represents a vector in three-dimensional space.

    The vector is defined as the difference vector between the start and end
    point. The start point is optional and defaults to the origin [0, 0, 0].

    Parameters:
        end (list): The xyz coordinates of the end point.
        start (list): The xyz coordinates of the start point, defaults to [0, 0, 0].

    Attributes:
        x (float): The x-coordinate of the coordinate difference vector.
        y (float): The y-coordinate of the coordinate difference vector.
        z (float): The z-coordinate of the coordinate difference vector.
        length (float): (**read-only**) The length of the vector.

    Examples:
        >>> u = Vector([1, 0, 0])
        >>> v = Vector([0, 2, 0], [0, 1, 0])
        >>> u
        [1.0, 0.0, 0.0]
        >>> v
        [0.0, 1.0, 0.0]
        >>> u.x
        1.0
        >>> u[0]
        1.0
        >>> u.length
        1.0
        >>> u + v
        [1.0, 1.0, 0.0]
        >>> u + [0.0, 1.0, 0.0]
        [1.0, 1.0, 0.0]
        >>> u * 2
        [2.0, 0.0, 0.0]
        >>> u.dot(v)
        0.0
        >>> u.cross(v)
        [0.0, 0.0, 1.0]

    """

    __slots__ = ['x', 'y', 'z']

    def __init__(self, end, start=None):
        if not start:
            start = [0, 0, 0]
        x = end[0] - start[0]
        y = end[1] - start[1]
        z = end[2] - start[2]
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return '[{0}, {1}, {2}]'.format(self.x, self.y, self.z)

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

    def __add__(self, other):
        """Compute the sum of this ``Vector`` and another ``Vector``.

        Parameters:
            other (tuple, list, Vector): The vector to add.

        Returns:
            Vector: The vector sum.
        """
        return Vector([self.x + other[0], self.y + other[1], self.z + other[2]])

    def __iadd__(self, other):
        self.x += other[0]
        self.y += other[1]
        self.z += other[2]
        return self

    def __sub__(self, other):
        """Compute the difference between this ``Vector`` and another ``Vector``.

        Parameters:
            other (tuple, list, Vector): The vector to subtract.

        Returns:
            Vector: The vector difference.
        """
        return Vector([self.x - other[0], self.y - other[1], self.z - other[2]])

    def __isub__(self, other):
        self.x -= other[0]
        self.y -= other[1]
        self.z -= other[2]
        return self

    def __mul__(self, n):
        """Scale this ``Vector`` by a factor.

        Parameters:
            n (int, float): The scaling factor.

        Returns:
            Vector: The scaled vector.

        Examples:
            >>> u = Vector([1, 0, 0])
            >>> v = u * 2
        """
        return Vector([self.x * n, self.y * n, self.z * n])

    def __imul__(self, n):
        self.x *= n
        self.y *= n
        self.z *= n
        return self

    def __pow__(self, n):
        return Vector([self.x ** n, self.y ** n, self.z ** n])

    def __ipow__(self, n):
        self.x **= n
        self.y **= n
        self.z **= n
        return self

    @property
    def length(self):
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dot(self, other):
        """The dot product of this ``Vector`` and another ``Vector``.

        Parameters:
            other (tuple, list, Vector): The vector to dot.

        Returns:
            float: The dot product.
        """
        return dot_vectors(self, other)

    def cross(self, other):
        """The cross product of this ``Vector`` and another ``Vector``.

        Parameters:
            other (tuple, list, Vector): The vector to cross.

        Returns:
            Vector: The cross product.
        """
        return Vector(cross_vectors(self, other))

    def normalize(self):
        l = self.length
        self.x = self.x / l
        self.y = self.y / l
        self.z = self.z / l

    def normalized(self):
        l = self.length
        x = self.x / l
        y = self.y / l
        z = self.z / l
        return Vector(x, y, z)

    def translate(self, vector):
        raise NotImplementedError

    def translated(self, vector):
        raise NotImplementedError

    def rotate(self, angle, axis=None, origin=None):
        """Rotate a vector u over an angle a around an axis k."""
        if axis is None:
            axis = (0, 0, 1.0)
        if origin is None:
            origin = (0, 0, 0)
        sina = sin(angle)
        cosa = cos(angle)
        kxu  = self.cross(axis) * -1
        v    = [sina * x for x in kxu]
        w    = [x * (1 - cosa) for x in cross_vectors(axis, kxu)]
        return [self[_] + v[_] + w[_] + origin[_] for _ in range(3)]

    def rotated(self, angle, axis=None, origin=None):
        raise NotImplementedError

    def scale(self, n):
        """Scale this ``Vector`` by a factor ``n``.

        Parameters:
            n (int, float): The scaling factor.

        Note:
            This is an alias for self \*= n
        """
        self *= n

    def scaled(self, n):
        raise NotImplementedError


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    import time
    import pyximport

    pyximport.install()

    from cvector import Vector as CVector

    tic = time.time()
    for i in range(100000):
        u = [1.0, 0.0, 0.0]
    toc = time.time()
    print(toc - tic)

    tic = time.time()
    for i in range(100000):
        u = CVector([1.0, 0.0, 0.0])
    toc = time.time()
    print(toc - tic)
