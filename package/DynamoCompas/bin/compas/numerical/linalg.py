from __future__ import print_function

import sys

from functools import wraps

from numpy import argmax
from numpy import array
from numpy import asarray
from numpy import atleast_2d
from numpy import float32
from numpy import nan_to_num
from numpy import nonzero
from numpy import seterr
from numpy import sum
from numpy import where
from numpy import zeros
from numpy import absolute

from numpy.linalg import cond

from scipy import cross

from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import lstsq
from scipy.linalg import lu
from scipy.linalg import qr
from scipy.linalg import solve
from scipy.linalg import svd

from scipy.io import loadmat
from scipy.io import savemat

from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import spsolve

from subprocess import Popen


__author__     = ['Tom Van Mele <vanmelet@ethz.ch>',
                  'Andrew Liew <liew@arch.ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


old_settings = seterr(all='ignore')


__all__ = [
    'nullspace',
    'rank',
    'dof',
    'pivots',
    'nonpivots',
    'rref',
    'normrow',
    'normalizerow',
    'rot90',
    'solve_with_known',
    'spsolve_with_known',
]


# ------------------------------------------------------------------------------
# Fundamentals
# ------------------------------------------------------------------------------


def nullspace(A, tol=0.001):
    r"""Calculates the nullspace of the input matrix A.

    Parameters:
        A (array, list): Matrix A represented as an array or list.
        tol (float): Tolerance.

    Returns:
        array: Null(A).

    The nullspace is the set of vector solutions to the equation

    .. math::

        \mathbf{A} \mathbf{x} = 0

    where 0 is a vector of zeros.

    When determining the nullspace using SVD decomposition (A = U S Vh),
    the right-singular vectors (rows of Vh or columns of V) corresponding to
    vanishing singular values of A, span the nullspace of A.

    Examples:
        >>> nullspace(array([[2, 3, 5], [-4, 2, 3]]))
        [[-0.03273859]
         [-0.85120177]
         [ 0.52381647]]

    """
    A = atleast_2d(asarray(A, dtype=float32))
    u, s, vh = svd(A, compute_uv=True)
    tol = s[0] * tol
    r = (s >= tol).sum()
    # nullspace
    # ---------
    # if A is m x n
    # the last (n - r) columns of v (or the last n - r rows of vh)
    null = vh[r:].conj().T
    return null


def rank(A, tol=0.001):
    r"""Calculates the rank of the input matrix A.

    Parameters:
        A (array, list): Matrix A represented as an array or list.
        tol (float): Tolerance.

    Returns:
        int: rank(A).

    The rank of a matrix is the maximum number of linearly independent rows in
    a matrix. Note that the row rank is equal to the column rank of the matrix.

    Examples:
        >>> rank([[1, 2, 1], [-2, -3, 1], [3, 5, 0]])
        2

    """
    A = atleast_2d(asarray(A, dtype=float32))
    s = svd(A, compute_uv=False)
    tol = s[0] * tol
    r = (s >= tol).sum()
    return r


def dof(A, tol=0.001, condition=False):
    r"""Returns the degrees-of-freedom of the input matrix A.

    Parameters:
        A (array, list): Matrix A represented as an array or list.
        tol (float): Tolerance.
        condition (boolean): Return the condition number of the matrix.

    Returns:
        int: Column degrees-of-freedom.
        int: Row degrees-of-freedom.
        float: (Optional) Condition number.

    The degrees-of-freedom are the number of columns and rows minus the rank.

    Examples:
        >>> dof([[2, -1, 3,], [1, 0, 1], [0, 2, -1], [1, 1, 4]], condition=True)
        (0, 1, 5.073597)

    """
    A = atleast_2d(asarray(A, dtype=float32))
    r = rank(A, tol=tol)
    k = A.shape[1] - r
    m = A.shape[0] - r
    if condition:
        c = cond(A)
        return k, m, c
    return k, m


def pivots(U, tol=None):
    r"""Identify the pivots of input matrix U.

    Parameters:
        U (array, list): Matrix U represented as an array or list.

    Returns:
        list: Pivot column indices.

    The pivots are the non-zero leading coefficients of each row.

    Examples:
        >>> A = [[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]]
        >>> n = rref(A, algo='sympy')
        >>> pivots(n)
        [0, 1]
    """
    if tol is None:
        tol = sys.float_info.epsilon
    U = atleast_2d(array(U, dtype=float32))
    U[absolute(U) < tol] = 0.0
    pivots = []
    for row in U:
        cols = nonzero(row)[0]
        if len(cols):
            pivots.append(cols[0])
    return pivots


def nonpivots(U, tol=None):
    r"""Identify the non-pivots of input matrix U.

    Parameters:
        U (array, list): Matrix U represented as an array or list.

    Returns:
        list: Non-pivot column indices.

    The non-pivots are where there are no non-zero leading coefficients in a row.

    Examples:
        >>> A = [[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]]
        >>> n = rref(A, algo='sympy')
        >>> nonpivots(n)
        [2, 3]
    """
    U = atleast_2d(asarray(U, dtype=float32))
    cols = pivots(U, tol=tol)
    return list(set(range(U.shape[1])) - set(cols))


def rref(A, algo='qr', tol=None, **kwargs):
    r"""Reduced row-echelon form of matrix A.

    Parameters:
        A (array, list): Matrix A represented as an array or list.
        algo (str): Algorithm to use: 'qr', 'sympy', 'matlab'.
        tol (float): Tolerance.

    Returns:
        array/list: RREF of A.

    A matrix is in reduced row-echelon form after Gauss-Jordan elimination, the
    result is independent of the method/algorithm used.

    Examples:
        >>> A = [[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]]
        >>> n = rref(A, algo='sympy')
        >>> array(n)
        [[1.0 0 1.0 3.0]
         [0 1.0 0.667 0.333]
         [0 0 0 0]]

    """
    A = atleast_2d(asarray(A, dtype=float32))
    if algo == 'qr':
        # do qr with column pivoting
        # to have non-decreasing absolute values on the diagonal of R
        # column pivoting ensures that the largest absolute value is used
        # as leading element
        _, U = qr(A)
        lead_pos = 0
        num_rows, num_cols = U.shape
        for r in range(num_rows):
            if lead_pos >= num_cols:
                return
            i = r
            # find a nonzero lead in column lead_pos
            while U[i][lead_pos] == 0:
                i += 1
                if i == num_rows:
                    i = r
                    lead_pos += 1
                    if lead_pos == num_cols:
                        return
            # swap the row with the nonzero lead with the current row
            U[[i, r]] = U[[r, i]]
            # "normalize" the values of the row
            lead_val = U[r][lead_pos]
            U[r] = U[r] / lead_val
            # make sure all other column values are zero
            for i in range(num_rows):
                if i != r:
                    lead_val = U[i][lead_pos]
                    U[i] = U[i] - lead_val * U[r]
            # go to the next column
            lead_pos += 1
        return U
    if algo == 'sympy':
        # return asarray?
        import sympy
        return sympy.Matrix(A).rref()[0].tolist()
    if algo == 'matlab':
        # replace this by the matlab com interface implementation
        import platform
        ifile = kwargs['ifile']
        ofile = kwargs['ofile']
        idict = {'A': A}
        savemat(ifile, idict)
        matlab = ['matlab']
        if platform.system() == 'Windows':
            options = ['-nosplash', '-wait', '-r']
        else:
            options = ['-nosplash', '-r']
        command = ["load('{0}');[R, jb]=rref(A);save('{1}');exit;".format(ifile, ofile)]
        p = Popen(matlab + options + command)
        stdout, stderr = p.communicate()
        odict = loadmat(ofile)
        return odict['R']

# if algo == 'lu':
#     _, _, u = lu(A)
#     if tol is None:
#         eps = sys.float_info.epsilon
#         tol = max(u.shape) * eps * max(sum(abs(u), axis=1))
#     pivots = {}
#     # iterate over the rows to find the non-pivoting elements
#     # a pivoting element in an upper triangular matrix is the first nonzero
#     # element in a row, if all elements below are also zero
#     m, n = u.shape
#     for i in range(m):
#         row = u[i]
#         # find the first element in the row of which the abs value is above
#         # the threshold
#         cols = nonzero(abs(row) > tol)[0]
#         if len(cols) == 0:
#             # in this row, all elements are zero
#             col = None
#         else:
#             # in this row, the first nonzero element is at cols[0]
#             # this is not necessarily a pivot,
#             # since this column might have been identified in a previous
#             # row as the pivot
#             piv = i
#             col = cols[0]
#             # check if the rows below for the highest value at this column
#             # swap rows to put the higher row on top
#             j = argmax(abs(u[i : m, col]))
#             if j > 0:
#                 u[i], u[i + j] = u[i + j], u[i]
#                 row = u[i]
#         # subtract multiples from previously found pivoting rows
#         # until the actual pivot is found
#         # or until the entire row is zero
#         count = 10000
#         while True:
#             if count == 0:
#                 # make sure the loop is not infinite
#                 break
#             count -= 1
#             if col not in pivots:
#                 # if this pivot was not yet found
#                 # update the row in the upper triangular matrix
#                 # this is not a row 'switch'
#                 u[i] = row
#                 break
#             # if there is a row that already had a pivot at this column
#             # update the row by subtracting a multiple of the previously
#             # found row such that the pivot becomes zero
#             piv  = pivots[col]
#             row  = row - (row[col] / u[piv, col]) * u[piv]
#             cols = nonzero(abs(row) > tol)[0]
#             if len(cols) == 0:
#                 # the entire row has become zero
#                 col = None
#             else:
#                 col = cols[0]
#         if col is not None:
#             pivots[col] = piv
#     nonpivots = list(set(range(n)) - set(pivots))


# ------------------------------------------------------------------------------
# Factorisation
# ------------------------------------------------------------------------------


class Memoized:
    """"""
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        key = args[-1]
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = res = self.f(args[0])
        return res


def memoize(f):
    memo = {}
    @wraps(f)
    def wrapper(*args):
        key = args[-1]
        if key in memo:
            return memo[key]
        memo[key] = res = f(args[0])
        return res
    return wrapper


def _chofactor(A):
    """Returns the Cholesky factorisation/decomposition matrix.

    Note:
        Random data inserted in the entries not used by Cholesky factorisation.

    Parameters:
        A (array): Matrix A represented as an (m x m) array.

    Returns:
        array: Matrix (m x m) with upper/lower triangle containing Cholesky
               factor of A.

    The Cholesky factorisation decomposes a Hermitian positive-definite matrix
    A into the product of a lower/upper triangular matrix and its transpose.

    .. math::

        \mathbf{A} = \mathbf{L} \mathbf{L}^{\mathrm{T}}


    Examples:
        >>> _chofactor(array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]]))
        (array([[  5.,   3.,  -1.],
                [ 15.,   3.,   1.],
                [ -5.,   0.,   3.]]), False)
    """
    return cho_factor(A)


def _lufactorized(A):
    """Return a function for solving a sparse linear system (LU decomposition).

    Parameters:
        A (array): Matrix A represented as an (m x n) array.

    Returns:
        function: Function to solve linear system with input matrix (n x 1).

    LU decomposition factors a matrix as the product of a lower triangular and
    an upper triangular matrix L and U.

    .. math::

        \mathbf{A} = \mathbf{L} \mathbf{U}


    Examples:
        >>> fn = _lufactorized(array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]))
        >>> fn(array([1, -2, 0]))
        array([ 1., -2., -2.])

    """
    return factorized(A)


chofactor = memoize(_chofactor)
lufactorized = memoize(_lufactorized)


# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------


def normrow(A):
    """Calculates the 2-norm of each row of matrix A.

    Parameters:
        A (array): Matrix A represented as an (m x n) array.

    Returns:
        array: Column vector (m x 1) of values.

    The calculation is the Euclidean 2-norm, i.e. the square root of the sum
    of the squares of the elements in each row, this equates to the "length" of
    the m row vectors.

    Examples:
        >>> normrow(array([[2, -1, 3,], [1, 0, 1], [0, 2, -1]]))
        [[ 3.74165739]
         [ 1.41421356]
         [ 2.23606798]]

    """
    A = atleast_2d(asarray(A, dtype=float32))
    return (sum(A ** 2, axis=1) ** 0.5).reshape((-1, 1))


def normalizerow(A, do_nan_to_num=True):
    """Normalise the rows of matrix A.

    Note:
        Tiling is not necessary, because of NumPy's broadcasting behaviour.

    Parameters:
        A (array): Matrix A represented as an (m x n) array.
        do_nan_to_num (boolean): Convert NaNs and INF to numbers, default=True.

    Returns:
        array: Matrix of normalised row vectors (m x n).

    Normalises the row vectors of A by the normrows, i.e. creates an array of
    vectors where the row vectors have length of unity.

    Examples:
        >>> normalizerow(array([[2, -1, 3,], [1, 0, 1], [0, 2, -1]]))
        array([[ 0.53452248, -0.26726124,  0.80178373],
               [ 0.70710678,  0.        ,  0.70710678],
               [ 0.        ,  0.89442719, -0.4472136 ]])

    """
    if do_nan_to_num:
        return nan_to_num(A / normrow(A))
    else:
        return A / normrow(A)


def rot90(vectors, axes):
    """Rotate an array of vectors through 90 degrees around an array of axes.

    Parameters:
        vectors (array): An array of row vectors (m x 3).
        axes (array): An array of axes (m x 3).

    Returns:
        array: Matrix of row vectors (m x 3).

    Computes the cross product of each row vector with its corresponding axis,
    and then rescales the resulting normal vectors to match the length of the
    original row vectors.

    Examples:
        >>> vectors = array([[2, 1, 3], [2, 6, 8]])
        >>> axes = array([[7, 0, 1], [4, 4, 2]])
        >>> rot90(vectors, axes)
        [[-0.18456235 -3.50668461  1.29193644]
         [ 5.3748385  -7.5247739   4.2998708 ]]
    """
    return normalizerow(cross(axes, vectors)) * normrow(vectors)


# ------------------------------------------------------------------------------
# Solving
# ------------------------------------------------------------------------------


def solve_with_known(A, b, x, known):
    """Solve a system of linear equations with part of solution known.

    Parameters:
        A (array): Coefficient matrix represented as an (m x n) array.
        b (array): Right-hand-side represented as an (m x 1) array
        x (array): Unknowns/knowns represented as an (n x 1) array.
        known (list): The indices of the known elements of ``x``.

    Returns:
        array: (n x 1) vector solution.

    Computes the solution of the system of linear equations.

    .. math::

        \mathbf{A} \mathbf{x} = \mathbf{b}


    Examples:
        >>> A = array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
        >>> b = array([[1], [3], [5]])
        >>> x = array([[0.3], [0], [0]])
        >>> solve_with_known(A, b, x, [0])
        array([ 0.3, 0.4, 0.0])
    """
    eps = 1 / sys.float_info.epsilon
    unknown = list(set(range(x.shape[0])) - set(known))
    A11 = A[unknown, :][:, unknown]
    A12 = A[unknown, :][:, known]
    b = b[unknown] - A12.dot(x[known])
    if cond(A11) < eps:
        Y = cho_solve(cho_factor(A11), b)
        x[unknown] = Y
        return x
    Y = lstsq(A11, b)
    x[unknown] = Y[0]
    return x


def spsolve_with_known(A, b, x, known):
    """Solve (sparse) a system of linear equations with part of solution known.

    Note:
        Same function as solve_with_known, but for sparse matrix A.

    Parameters:
        A (array): Coefficient matrix (sparse) represented as an (m x n) array.
        b (array): Right-hand-side represented as an (m x 1) array
        x (array): Unknowns/knowns represented as an (n x 1) array.
        known (list): The indices of the known elements of ``x``.

    Returns:
        array: (n x 1) vector solution.

    Computes the solution (using spsolve) of the system of linear equations.

    .. math::

        \mathbf{A} \mathbf{x} = \mathbf{b}


    Examples:
        >>> A = array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
        >>> b = array([[1], [3], [5]])
        >>> x = array([[0.3], [0], [0]])
        >>> solve_with_known(A, b, x, [0])
        array([ 0.3, 0.4, 0.0])

    """
    unknown = list(set(range(x.shape[0])) - set(known))
    A11 = A[unknown, :][:, unknown]
    A12 = A[unknown, :][:, known]
    b = b[unknown] - A12.dot(x[known])
    x[unknown] = spsolve(A11, b)
    return x


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    import numpy as np

    np.set_printoptions(precision=3, threshold=10000, linewidth=1000)

    E = array([[2, 3, 5], [-4, 2, 3]], dtype=float32)

    null = nullspace(E)

    assert np.allclose(zeros((E.shape[0], 1)), E.dot(null), atol=1e-6), 'E.dot(null) not aproximately zero'

    m, n = E.shape
    s, t = null.shape

    print(m, n)
    print(s, t)

    assert n == s, 'num_cols of E should be equal to num_rows of null(E)'

    print(rank(E))
    print(dof(E))

    print(len(pivots(rref(E))))
    print(len(nonpivots(rref(E))))

    # ifile = './data/ifile.mat'
    # ofile = './data/ofile.mat'

    # with open(ifile, 'wb+') as fp: pass
    # with open(ofile, 'wb+') as fp: pass

    # print nonpivots(rref(E, algo='qr'))
    # print nonpivots(rref(E, algo='sympy'))
    # print nonpivots(rref(E, algo='matlab', ifile=ifile, ofile=ofile))
