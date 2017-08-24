""""""

from __future__ import print_function
from __future__ import division

from math import pi
from math import degrees
from math import acos

from compas.geometry.basic import subtract_vectors
from compas.geometry.basic import subtract_vectors_2d
from compas.geometry.basic import dot_vectors
from compas.geometry.basic import dot_vectors_2d
from compas.geometry.basic import length_vector
from compas.geometry.basic import length_vector_2d


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'angles_vectors', 'angles_vectors_degrees', 'angles_points', 'angles_points_degrees',
    'angle_smallest_vectors', 'angle_smallest_vectors_degrees', 'angle_smallest_points',
    'angle_smallest_points_degrees',
]


def angles_vectors(u, v):
    """Compute the the 2 angles (radians) formed by a pair of vectors.

    Parameters
    ----------
    u : sequence of float
        XYZ components of the first vector.
    v : sequence of float
        XYZ components of the second vector.

    Returns
    -------
    tuple
        The two angles in radians.
        The smallest angle is returned first.

    Examples
    --------
    >>>

    """
    a = angle_smallest_vectors(u, v)
    return a, pi * 2 - a


def angles_vectors_2d(u, v):
    """Compute the angles between the XY components of two vectors lying in the XY-plane.

    Parameters
    ----------
    u : sequence of float
        XY(Z) coordinates of the first vector.
    v : sequence of float
        XY(Z) coordinates of the second vector.

    Returns
    -------
    tuple
        The two angles.
        The smallest angle is returned first.

    Examples
    --------
    >>>

    """
    a = angle_smallest_vectors_2d(u, v)
    return a, 2. * pi - a


def angles_vectors_degrees(u, v):
    """Compute the the 2 angles (degrees) formed by a pair of vectors.

    Parameters
    ----------
    u : sequence of float
        XYZ components of the first vector.
    v : sequence of float
        XYZ components of the second vector.

    Returns
    -------
    tuple
        The two angles in degrees.
        The smallest angle is returned first.

    Examples
    --------
    >>>

    """
    a = angle_smallest_vectors_degrees(u, v)
    return a, 360. - a


def angles_vectors_degrees_2d(u, v):
    a = angle_smallest_vectors_degrees_2d(u, v)
    return a, 360. - a


def angles_points(a, b, c):
    r"""Compute the two angles (radians) define by three points.

    Parameters
    ----------
    a : sequence of float)
        XYZ coordinates.
    b : sequence of float)
        XYZ coordinates.
    c : sequence of float)
        XYZ coordinates.

    Returns
    -------
    tuple
        The two angles in radians.
        The smallest angle is returned first.

    Notes
    -----
    The vectors are defined in the following way

    .. math::

        \mathbf{u} = \mathbf{b} - \mathbf{a} \\
        \mathbf{v} = \mathbf{c} - \mathbf{a}

    Z components may be provided, but are simply ignored.

    Examples
    --------
    >>>

    """
    u = subtract_vectors(b, a)
    v = subtract_vectors(c, a)
    return angles_vectors(u, v)


def angles_points_2d(a, b, c):
    u = subtract_vectors_2d(b, a)
    v = subtract_vectors_2d(c, a)
    return angles_vectors_2d(u, v)


def angles_points_degrees(a, b, c):
    """Compute the two angles (degrees) define by three points.

    Parameters
    ----------
    a : sequence of float)
        XYZ coordinates.
    b : sequence of float)
        XYZ coordinates.
    c : sequence of float)
        XYZ coordinates.

    Returns
    -------
    tuple
        The two angles in degrees.
        The smallest angle is returned first.

    Notes
    -----
    The vectors are defined in the following way

    .. math::

        \mathbf{u} = \mathbf{b} - \mathbf{a} \\
        \mathbf{v} = \mathbf{c} - \mathbf{a}

    Z components may be provided, but are simply ignored.

    Examples
    --------
    >>>

    """
    return degrees(angles_points(a, b, c))


def angles_points_degrees_2d(a, b, c):
    return degrees(angles_points_2d(a, b, c))


def angle_smallest_vectors(u, v):
    """Compute the smallest angle (radians) between two vectors.

    Parameters
    ----------
    u : sequence of float
        XYZ components of the first vector.
    v : sequence of float)
        XYZ components of the second vector.

    Returns
    -------
    float
        The smallest angle in radians.
        The angle is always positive.

    Examples
    --------
    >>> angle_smallest_vectors([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])

    """
    a = dot_vectors(u, v) / (length_vector(u) * length_vector(v))
    a = max(min(a, 1), -1)
    return acos(a)


def angle_smallest_vectors_2d(u, v):
    """Compute the smallest angle (radians) between the XY components of two vectors lying in the XY-plane.

    Parameters
    ----------
    u : sequence of float
        The first 2D or 3D vector (Z will be ignored).
    v : sequence of float)
        The second 2D or 3D vector (Z will be ignored).

    Returns
    -------
    float
        The smallest angle between the vectors in radians.
        The angle is always positive.

    Examples
    --------
    >>>

    """
    a = dot_vectors_2d(u, v) / (length_vector_2d(u) * length_vector_2d(v))
    a = max(min(a, 1), -1)
    return acos(a)


def angle_smallest_vectors_degrees(u, v):
    """Compute the smallest angle (degrees) between two vectors.

    Parameters
    ----------
    u : sequence of float
        XYZ components of the first vector.
    v : sequence of float)
        XYZ components of the second vector.

    Returns
    -------
    float
        The smallest angle in degrees.
        The angle is always positive.

    Examples
    --------
    >>> angle_smallest_vectors_degrees([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])

    """
    return degrees(angle_smallest_vectors(u, v))


def angle_smallest_vectors_degrees_2d(u, v):
    """Compute the smallest angle (degrees) between the XY components of two vectors lying in the XY-plane.

    Parameters
    ----------
    u : sequence of float
        The first 2D or 3D vector (Z will be ignored).
    v : sequence of float)
        The second 2D or 3D vector (Z will be ignored).

    Returns
    -------
    float
        The smallest angle between the vectors in degrees.
        The angle is always positive.

    """
    return degrees(angle_smallest_vectors_2d)


def angle_smallest_points(a, b, c):
    r"""Compute the smallest angle (radians) between the vectors defined by three points.

    Parameters
    ----------
    a : sequence of float
        XYZ coordinates.
    b : sequence of float
        XYZ coordinates.
    c : sequence of float
        XYZ coordinates.

    Returns
    -------
    float
        The smallest angle in radians.
        The angle is always positive.

    Note
    ----
    The vectors are defined in the following way

    .. math::

        \mathbf{u} = \mathbf{b} - \mathbf{a} \\
        \mathbf{v} = \mathbf{c} - \mathbf{a}

    Z components may be provided, but are simply ignored.

    """
    u = subtract_vectors(b, a)
    v = subtract_vectors(c, a)
    return angle_smallest_vectors(u, v)


def angle_smallest_points_2d(a, b, c):
    r"""Compute the smallest angle defined by the XY components of three points lying in
       the XY-plane where the angle is computed at point A in the triangle ABC

    Parameters
    ----------
    a : sequence of float
        XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
    b : sequence of float)
        XY(Z) coordinates of a 2D or 3D point (Z will be ignored).
    c : sequence of float)
        XY(Z) coordinates of a 2D or 3D point (Z will be ignored).

    Returns
    -------
    float
        The smallest angle between the vectors.
        The angle is always positive.

    Notes
    -----
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


def angle_smallest_points_degrees(a, b, c):
    r"""Compute the smallest angle (degrees) between the vectors defined by three points.

    Parameters
    ----------
    a : sequence of float
        XYZ coordinates.
    b : sequence of float
        XYZ coordinates.
    c : sequence of float
        XYZ coordinates.

    Returns
    -------
    float
        The smallest angle in degrees.
        The angle is always positive.

    Note
    ----
    The vectors are defined in the following way

    .. math::

        \mathbf{u} = \mathbf{b} - \mathbf{a} \\
        \mathbf{v} = \mathbf{c} - \mathbf{a}

    Z components may be provided, but are simply ignored.

    """
    return degrees(angle_smallest_points(a, b, c))


def angle_smallest_points_degrees_2d(a, b, c):
    """Compute the smallest angle defined by the XY components of three points lying in
       the XY-plane where the angle is computed at point A in the triangle ABC

    Parameters
    ----------
    a : sequence of float
        XY(Z) coordinates of the origin.
    b : sequence of float)
        XY(Z) coordinates of the first end point.
    c : sequence of float)
        XY(Z) coordinates of the second end point.

    Returns
    -------
    float
        The smallest angle (degrees) between the vectors ``ab`` and ``ac`` in the XY plane.
        The angle is always positive.

    """
    a = degrees(angle_smallest_points_2d(a, b, c))
    return a


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":
    pass
