""""""

from __future__ import division
from __future__ import print_function

from math import sqrt


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = [
    'sum_vectors', 'norm_vector', 'norm_vectors', 'length_vector', 'length_vector_sqrd',
    'length_vector_2d', 'length_vector_sqrd_2d',
    'scale_vector', 'scale_vectors', 'normalize_vector', 'normalize_vectors',
    'scale_vector_2d', 'scale_vectors_2d', 'normalize_vector_2d', 'normalize_vectors_2d',
    'power_vector', 'power_vectors', 'square_vector', 'square_vectors',
    'add_vectors', 'subtract_vectors', 'multiply_vectors', 'divide_vectors',
    'add_vectors_2d', 'subtract_vectors_2d', 'multiply_vectors_2d', 'divide_vectors_2d',
    'cross_vectors', 'dot_vectors', 'vector_component',
    'cross_vectors_2d', 'dot_vectors_2d', 'vector_component_2d',
    'multiply_matrices', 'multiply_matrix_vector',
    'homogenise_vectors', 'dehomogenise_vectors', 'orthonormalise_vectors'
]


# ==============================================================================
# these return something of smaller dimension/length/...
# something_(of)vector/s
# ==============================================================================

def sum_vectors(vectors, axis=0):
    """
    Calculate the sum of a series of vectors along the specified axis.

    Parameters
    ----------
    vectors : list of list
        A list of vectors.
    axis : int, optional
        If ``axis == 0``, the sum is taken across each of the indices of the mesh.
        If ``axis == 1``, the sum is taken across the individual vectors.

    Returns
    -------
    sum : list of float
        The length of the list is ``len(vectors[0])``, if ``axis == 0``.
        The length is ``len(vectors)``, otherwise.

    Examples
    --------
    >>> vectors = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    >>> sum_vectors(vectors)
    [3, 6, 9]
    >>> sum_vectors(vectors, axis=1)
    [6, 6, 6]

    """
    if axis == 0:
        vectors = zip(*vectors)
    return [sum(vector) for vector in vectors]


def norm_vector(vector):
    """
    Calculate the length of a vector.

    Parameters
    ----------
    vector : list of float

    Returns
    -------
    norm : float
        The L2 norm, or *length* of the vector.

    Examples
    --------
    >>>

    """
    return sqrt(sum(axis ** 2 for axis in vector))


def norm_vectors(vectors):
    """
    Calculate the norm of each vector in a list of vectors.

    Parameters
    ----------
    vectors : list of list

    Returns
    -------
    norm : list of float
        A list with the length of each vector.

    Examples
    --------
    >>>

    """
    return [norm_vector(vector) for vector in vectors]


def length_vector(vector):
    """Calculate the length of the vector.

    Parameters
    ----------
    vector : list of float
        The XYZ coordinates of the vector.

    Returns
    -------
    float
        The length of the vector.

    See Also
    --------
    norm_vector

    Examples
    --------
    >>>

    """
    return sqrt(length_vector_sqrd(vector))


def length_vector_2d(vector):
    """Compute the length of a vector, assuming it lies in the XY plane.

    Parameters
    ----------
    vector : sequence of float
        The XY(Z) coordinates of the vector.

    Returns
    -------
    float
        The length of the XY component of the vector.

    Examples
    --------
    >>> length_vector_2d([2.0, 0.0])
    2.0

    >>> length_vector_2d([2.0, 0.0, 0.0])
    2.0

    >>> length_vector_2d([2.0, 0.0, 2.0])
    2.0

    """
    return sqrt(length_vector_sqrd_2d(vector))


def length_vector_sqrd(vector):
    return vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2


def length_vector_sqrd_2d(vector):
    return vector[0] ** 2 + vector[1] ** 2


# ==============================================================================
# these perform an operation on a vector and return a modified vector
# -> elementwise operations on 1 vector
# should this not bet ...ed_vector
# ... or else modify the vector in-place
# ==============================================================================

def scale_vector(vector, factor):
    """Scale a vector by a given factor.

    Parameters
    ----------
    vector : list, tuple
        The XYZ coordinates of the vector.
    factor : float
        The scaling factor.

    Returns
    -------
    list
        The scaled vector.

    Examples
    --------
    >>>

    """
    return [axis * factor for axis in vector]


# does this even make sense?
# should the Z-component not remain the same?

def scale_vector_2d(vector, factor):
    """Scale a vector by a given factor, assuming it lies in the XY plane.

    Parameters
    ----------
    vector : sequence of float
        The XY(Z) coordinates of the vector.
    scale : float
        Scale factor.

    Returns
    -------
    list
        The scaled vector in the XY-plane (Z = 0.0).

    Examples
    --------
    >>>

    """
    return [vector[0] * factor, vector[1] * factor, 0.0]


def scale_vectors(vectors, factor):
    """Scale multiple vectors by a given factor.

    Parameters
    ----------
    vectors : list of list
        A list of vectors represented by XYZ coordinates.
    factor : float
        The scaling factor.

    Returns
    -------
    vectors : list of list
        The scaled vectors.

    Examples
    --------
    >>>

    """
    return [scale_vector(vector, factor) for vector in vectors]


def scale_vectors_2d(vectors, factor):
    return [scale_vector_2d(vector, factor) for vector in vectors]


def normalize_vector(vector):
    """Normalise a given vector.

    Parameters
    ----------
    vector : list, tuple
        A vector, represented by its XYZ coordinates.

    Returns
    -------
    vector : list
        The normalised vector.

    Examples
    --------
    >>>

    """
    return scale_vector(vector, 1.0 / length_vector(vector))


def normalize_vector_2d(vector):
    """Normalize a vector, assuming it lies in the XY-plane.

    Parameters
    ----------
    vector : sequence of float
        The 2D or 3D vector (Z will be ignored).

    Returns
    -------
    tuple
        The normalized vector in the XY-plane (Z = 0.0)

    Examples
    --------
    >>>

    """
    l = length_vector_2d(vector)
    if not l:
        return vector
    return vector[0] / l, vector[1] / l, 0.0


def normalize_vectors(vectors):
    """Normalise multiple vectors.

    Parameters
    ----------
    vectors : list of list
        A list of vectors.

    Returns
    -------
    vectors : list
        The normalised vectors.

    Examples
    --------
    >>>

    """
    return [normalize_vector(vector) for vector in vectors]


def normalize_vectors_2d(vectors):
    return [normalize_vector_2d(vector) for vector in vectors]


def power_vector(vector, power):
    """Raise a vector to the given power.

    Parameters
    ----------
    vector : list, tuple
        A vector, represented by its XYZ coordinates.
    power : int, float
        The power to which to raise the vector.

    Returns
    -------
    vector : list
        The raised vector.

    Examples
    --------
    >>>

    """
    return [axis ** power for axis in vector]


def power_vectors(vectors, power):
    """Raise a list of vectors to the given power.

    Parameters
    ----------
    vectors : list of list
        A list of vectors, represented by their XYZ coordinates.
    power : int, float
        The power to which to raise the vectors.

    Returns
    -------
    vector : list
        The raised vectors.

    Examples
    --------
    >>>

    """
    return [power_vector(vector, power) for vector in vectors]


def square_vector(vector):
    """Raise a vector to the power 2.

    Parameters
    ----------
    vector : list, tuple
        A vector, represented by its XYZ coordinates.

    Returns
    -------
    vector : list
        The squared vector.

    Examples
    --------
    >>>

    """
    return power_vector(vector, 2)


def square_vectors(vectors):
    return [square_vectors(vector) for vector in vectors]


# ==============================================================================
# these perform an operation with corresponding elements of the (2) input vectors as operands
# and return a vector with the results
# -> elementwise operations on two vectors
# ==============================================================================

def add_vectors(u, v):
    return [a + b for (a, b) in zip(u, v)]


def add_vectors_2d(u, v):
    """Add two vectors, assuming they lie in the XY-plane.

    Parameters
    ----------
    u : sequence of float
        The first 2D or 3D vector (Z will be ignored).
    v : sequence of float
        The second 2D or 3D vector (Z will be ignored).

    Returns
    -------
    tuple
        Resulting vector in the XY-plane (Z = 0.0)

    Examples
    --------
    >>>

    """
    return u[0] + v[0], u[1] + v[1], 0.0


def subtract_vectors(u, v):
    return [a - b for (a, b) in zip(u, v)]


def subtract_vectors_2d(u, v):
    """Subtract one vector from another, assuming they lie in the XY plane.

    Parameters
    ----------
    u : sequence of float
        The XY(Z) coordinates of the first vector.
    v : sequence of float
        The XY(Z) coordinates of the second vector.

    Returns
    -------
    tuple
        Resulting vector in the XY-plane (Z = 0.0)

    Examples
    --------
    >>>

    """
    return u[0] - v[0], u[1] - v[1], 0.0


def multiply_vectors(u, v):
    return [a * b for (a, b) in zip(u, v)]


def multiply_vectors_2d(u, v):
    return [u[0] * v[0], u[1] * v[1], 0.0]


def divide_vectors(u, v):
    return [a / b for (a, b) in zip(u, v)]


def divide_vectors_2d(u, v):
    return [u[0] / v[0], u[1] / v[1], 0.0]


# ==============================================================================
# ...
# ==============================================================================

def cross_vectors(u, v):
    r"""Compute the cross product of two vectors.

    The xyz components of the cross product of two vectors :math:`\mathbf{u}`
    and :math:`\mathbf{v}` can be computed as the *minors* of the following matrix:

    .. math::
       :nowrap:

        \begin{bmatrix}
        x & y & z \\
        u_{x} & u_{y} & u_{z} \\
        v_{x} & v_{y} & v_{z}
        \end{bmatrix}

    Therefore, the cross product can be written as:

    .. math::
       :nowrap:

        \mathbf{u} \times \mathbf{v}
        =
        \begin{bmatrix}
        u_{y} * v_{z} - u_{z} * v_{y} \\
        u_{z} * v_{x} - u_{x} * v_{z} \\
        u_{x} * v_{y} - u_{y} * v_{x}
        \end{bmatrix}

    Parameters
    ----------
    u : tuple, list, Vector
        XYZ components of the first vector.
    v : tuple, list, Vector
        XYZ components of the second vector.

    Returns
    -------
    cross : list
        The cross product of the two vectors.

    Exmaples
    --------
    >>> cross_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    [0.0, 0.0, 1.0]

    """
    return [u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]]


def cross_vectors_2d(u, v):
    """Compute the cross product of two vectors, assuming they lie in the XY-plane.

    Parameters
    ----------
    u : sequence of float
        XY(Z) coordinates of the first vector.
    v : sequence of float
        XY(Z) coordinates of the second vector.

    Returns
    -------
    list
        The cross product of the two vectors.
        This vector will be perpendicular to the XY plane.

    Examples
    --------
    >>> cross_vectors_2d([1.0, 0.0], [0.0, 1.0])
    [0.0, 0.0, 1.0]

    >>> cross_vectors_2d([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    [0.0, 0.0, 1.0]

    >>> cross_vectors_2d([1.0, 0.0, 1.0], [0.0, 1.0, 1.0])
    [0.0, 0.0, 1.0]

    """
    return [0.0, 0.0, u[0] * v[1] - u[1] * v[0]]


def dot_vectors(u, v):
    """Compute the dot product of two vectors.

    Parameters
    ----------
    u : tuple, list, Vector
        XYZ components of the first vector.
    v : tuple, list, Vector
        XYZ components of the second vector.

    Returns
    -------
    dot : float
        The dot product of the two vectors.

    Examples
    --------
    >>> dot_vectors([1.0, 0, 0], [2.0, 0, 0])
    2

    """
    return sum(a * b for a, b in zip(u, v))


def dot_vectors_2d(u, v):
    """Compute the dot product of two vectors, assuming they lie in the XY-plane.

    Parameters
    ----------
    u : sequence of float
        XY(Z) coordinates of the first vector.
    v : sequence of float
        XY(Z) coordinates of the second vector.

    Returns
    -------
    float
        The dot product of the XY components of the two vectors.

    Examples
    --------
    >>> dot_vectors_2d([1.0, 0], [2.0, 0])
    2.0

    >>> dot_vectors_2d([1.0, 0, 0], [2.0, 0, 0])
    2.0

    >>> dot_vectors_2d([1.0, 0, 1], [2.0, 0, 1])
    2.0

    """
    return u[0] * v[0] + u[1] * v[1]


def vector_component(u, v):
    """Compute the component of u in the direction of v.

    Note
    ----
    This is similar to computing direction cosines, or to the projection of
    a vector onto another vector. See the respective Wikipedia pages for more
    info:

        - `Direction cosine <https://en.wikipedia.org/wiki/Direction_cosine>`_
        - `Vector projection <https://en.wikipedia.org/wiki/Vector_projection>`_

    Parameters
    ----------
    u : sequence of float
        XYZ components of the vector.
    v : sequence of float
        XYZ components of the direction.

    Returns
    -------
    proj_v(u) : list
        The component of u in the direction of v.

    Examples
    --------
    >>> vector_component([1, 2, 3], [1, 0, 0])
    [1, 0, 0]

    """
    l2 = length_vector_sqrd(v)
    if not l2:
        return [0, 0, 0]
    x = dot_vectors(u, v) / l2
    return scale_vector(v, x)


def vector_component_2d(u, v):
    l2 = length_vector_sqrd_2d(v)
    if not l2:
        return [0, 0, 0]
    x = dot_vectors_2d(u, v) / l2
    return scale_vector_2d(v, x)


# ==============================================================================
# these involve vectors interpreted as matrices (lists of lists)
# -> matrix multiplication
# ==============================================================================


def transpose_matrix(M):
    return zip(*M)


# rename to matmul_...
# rename to dot_...

def multiply_matrices(A, B):
    r"""Mutliply a matrix with a matrix.

    This is a pure Python version of the following linear algebra procedure:

    .. math::

        \mathbf{A} \cdot \mathbf{B} = \mathbf{C}

    with :math:`\mathbf{A}` a *m* by *n* matrix, :math:`\mathbf{B}` a *n* by *o*
    matrix, and :math:`\mathbf{C}` a *m* by *o* matrix.

    Parameters
    ----------
    A : sequence of sequence of float
        The first matrix.
    B : sequence of sequence of float
        The second matrix.

    Returns
    -------
    C : list of list of float
        The result matrix.

    Raises
    ------
    Exception
        If the shapes of the matrices are not compatible.
        If the row length of B is inconsistent.

    Examples
    --------
    >>> A = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    >>> B = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    >>> dot_matrices(A, B)
    [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    """
    n = len(B)  # number of rows in B
    o = len(B[0])  # number of cols in B
    if not all([len(row) == o for row in B]):
        raise Exception('Row length in matrix B is inconsistent.')
    if not all([len(row) == n for row in A]):
        raise Exception('Matrix shapes are not compatible.')
    B = zip(*B)
    return [[dot_vectors(row, col) for col in B] for row in A]


def multiply_matrix_vector(A, b):
    r"""Multiply a matrix with a vector.

    This is a Python version of the following linear algebra procedure:

    .. math::

        \mathbf{A} \cdot \mathbf{x} = \mathbf{b}

    with :math:`\mathbf{A}` a *m* by *n* matrix, :math:`\mathbf{x}` a vector of
    length *n*, and :math:`\mathbf{b}` a vector of length *m*.

    Parameters
    ----------
    A : list of list
        The matrix.
    b : list
        The vector.

    Returns
    -------
    c : list
        The resulting vector.

    Raises
    ------
    Exception
        If not all rows of the matrix have the same length as the vector.

    Examples
    --------
    >>> matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    >>> vector = [1.0, 2.0, 3.0]
    >>> dot_matrix_vector(matrix, vector)
    [2.0, 4.0, 6.0]

    """
    n = len(b)
    if not all([len(row) == n for row in A]):
        raise Exception('Matrix shape is not compatible with vector length.')
    return [dot_vectors(row, b) for row in A]


# ==============================================================================
# linalg
# ==============================================================================


def homogenise_vectors(vectors):
    return [vector + [1.0] for vector in vectors]


def dehomogenise_vectors(vectors):
    return [vector[:-1] for vector in vectors]


def orthonormalise_vectors(vectors):
    """Orthonormalise a set of vectors.

    This creates a basis for the range (column space) of the matrix A.T,
    with A = vectors.

    Orthonormalisation is according to the Gram-Schmidt process.

    Parameters
    ----------
    vectors : list of list
        The set of vectors to othonormalise.

    Returns
    -------
    basis : list of list
        An othonormal basis for the input vectors.

    Examples
    --------
    >>>

    """
    basis = []
    for v in vectors:
        if basis:
            e = subtract_vectors(v, sum_vectors([vector_component(v, b) for b in basis]))
        else:
            e = v
        if any([axis > 1e-10 for axis in e]):
            basis.append(normalize_vector(e))
    return basis


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    # import timeit

    # r = timeit.timeit('length_vector([0.0, 0.0, 0.0])', setup='from __main__ import length_vector', number=100000)
    # print(r)

    # r = timeit.timeit('length_vector_2d([0.0, 0.0, 0.0])', setup='from __main__ import length_vector_2d', number=100000)
    # print(r)

    M = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]

    print(M)
    print(transpose_matrix(M))
