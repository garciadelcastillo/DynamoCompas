from __future__ import print_function


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'geometric_key', 'geometric_key2'
]


def geometric_key(xyz, precision='3f', tolerance=1e-9, sanitize=True):
    """Convert XYZ coordinates to a string that can be used as a dict key.

    Parameters:
        xyz (sequence of float): The XYZ coordinates.
        precision (str): Optional.
            A formatting option that specifies the precision of the
            individual numbers in the string.
            Supported values are any float precision, or decimal integer (``'d'``).
            Default is ``'3f'``.
        tolerance (float): Optional.
            A tolerance for values that should be considered zero.
            Default is ``1e-9``.
        sanitize (bool): Optional.
            Flag that indicates whether or not the input should be cleaned up
            using the tolerance value.
            Default is ``True``.

    Returns:
        str: the string representation of the given coordinates.

    Example:

        .. code-block:: python

            from math import pi
            from compas.utilities import geometric_key

            print(geometric_key([pi, pi / 2.0, 2.0 * pi], '3f'))

            # 3.142,3.142,3.142

    """
    x, y, z = xyz
    if precision == 'd':
        return '{0},{1},{2}'.format(int(x), int(y), int(z))
    if sanitize:
        tolerance = tolerance ** 2
        if x ** 2 < tolerance:
            x = 0.0
        if y ** 2 < tolerance:
            y = 0.0
        if z ** 2 < tolerance:
            z = 0.0
    return '{0:.{3}},{1:.{3}},{2:.{3}}'.format(x, y, z, precision)


def geometric_key2(xy, precision='3f', tolerance=1e-9, sanitize=True):
    """Convert XY coordinates to a string that can be used as a dict key."""
    x, y = xy
    if precision == 'd':
        return '{0},{1}'.format(int(x), int(y))
    if sanitize:
        tolerance = tolerance ** 2
        if x ** 2 < tolerance:
            x = 0.0
        if y ** 2 < tolerance:
            y = 0.0
    return '{0:.{2}},{1:.{2}}'.format(x, y, precision)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    from math import pi

    print(geometric_key([pi, pi, pi], '3f'))
    print(geometric_key([-0.00001, +0.00001, 0.00001], '3f', tolerance=1e-3))

    print(geometric_key((1.1102230246251565e-16, -1.1102230246251565e-16, -1.7320508075688774), '3f', tolerance=1e-9))
    print(geometric_key((-1.1102230246251565e-16, -1.1102230246251565e-16, -1.7320508075688774), '3f', tolerance=1e-9))
