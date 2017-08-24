from __future__ import print_function

import time

from numpy import abs
from numpy import arccos
from numpy import array
from numpy import cross
from numpy import isnan
from numpy import isinf
from numpy import max
from numpy import mean
from numpy import newaxis
from numpy import ones
from numpy import sin
from numpy import sqrt
from numpy import sum
from numpy import tile
from numpy import zeros

from scipy.linalg import norm

from scipy.sparse import diags
from scipy.sparse import find

from compas.numerical.geometry import lengths
from compas.numerical.matrices import connectivity_matrix
from compas.numerical.matrices import mass_matrix
from compas.numerical.linalg import normrow

try:
    import bpy
    from compas_blender.utilities.layers import layer_clear
    from compas_blender.utilities.objects import objects_layer
    from compas_blender.utilities.drawing import draw_bmesh
    from compas_blender.utilities.drawing import xdraw_pipes
except:
    pass

try:
    from numba import int32
    from numba import int64
    from numba import float64
    from numba import jit
except:
    pass


__author__     = ['Tom Van Mele <vanmelet@ethz.ch>', 'Andrew Liew <liew@arch.ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = [
    'rk1', 'rk2', 'rk3', 'rk4', 'dr'
]


class Coeff():
    def __init__(self, c):
        self.c = c
        self.a = (1 - c * 0.5) / (1 + c * 0.5)
        self.b = 0.5 * (1 + self.a)


def rk1(a, v0, dt):
    return a(dt, v0)


def rk2(a, v0, dt):
    K = [
        [0.0, ],
        [0.5, 0.5, ],
    ]
    B = [0.0, 1.0]
    K0 = dt * a(K[0][0] * dt, v0)
    K1 = dt * a(K[1][0] * dt, v0 + K[1][1] * K0)
    dv = B[0] * K0 + B[1] * K1
    return dv


def rk3(a, v0, dt):
    raise NotImplementedError


def rk4(a, v0, dt):
    K = [
        [0.0, ],
        [0.5, 0.5, ],
        [0.5, 0.0, 0.5, ],
        [1.0, 0.0, 0.0, 1.0, ],
    ]
    B = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
    K0 = dt * a(K[0][0] * dt, v0)
    K1 = dt * a(K[1][0] * dt, v0 + K[1][1] * K0)
    K2 = dt * a(K[2][0] * dt, v0 + K[2][1] * K0 + K[2][2] * K1)
    K3 = dt * a(K[3][0] * dt, v0 + K[3][1] * K0 + K[3][2] * K1 + K[3][3] * K2)
    dv = B[0] * K0 + B[1] * K1 + B[2] * K2 + B[3] * K3
    return dv


# --------------------------------------------------------------------------
# adaptive, explicit RK schemes
# --------------------------------------------------------------------------
# def rk5():
#     K  = [
#         [0.0, ],
#         [0.25, 0.25, ],
#         [3. / 8., 3. / 32., 9. / 32., ],
#         [12. / 13., 1932. / 2197., -7200. / 2197., 7296. / 2197., ],
#         [1.0, 439. / 216., -8., 3680. / 513., -845. / 4104., ],
#         [0.5, -8. / 27., 2.0, -3544. / 2565., 1859. / 4104., -11. / 40., ]
#     ]
#     B5 = [16. / 135., 0.0, 6656. / 12825., 28561. / 56430., -9. / 50., 2. / 55.]
#     B4 = [25. / 216., 0.0, 1408. / 2565., 2197. / 4104., -1. / 5., 0.0]
#     K0 = dt * a(K[0][0] * dt, v0)
#     K1 = dt * a(K[1][0] * dt, v0 + K[1][1] * K0)
#     K2 = dt * a(K[2][0] * dt, v0 + K[2][1] * K0 + K[2][2] * K1)
#     K3 = dt * a(K[3][0] * dt, v0 + K[3][1] * K0 + K[3][2] * K1 + K[3][3] * K2)
#     K4 = dt * a(K[4][0] * dt, v0 + K[4][1] * K0 + K[4][2] * K1 + K[4][3] * K2 + K[4][4] * K3)
#     K5 = dt * a(K[5][0] * dt, v0 + K[5][1] * K0 + K[5][2] * K1 + K[5][3] * K2 + K[5][4] * K3 + K[5][5] * K4)
#     dv = B5[0] * K0 + B5[1] * K1 + B5[2] * K2 + B5[3] * K3 + B5[4] * K4 + B5[5] * K5
#     e  = (B5[0] - B4[0]) * K0 + (B5[1] - B4[1]) * K1 + (B5[2] - B4[2]) * K2 + (B5[3] - B4[3]) * K3 + (B5[4] - B4[4]) * K4 + (B5[5] - B4[5]) * K5
#     e  = sqrt(sum(e ** 2) / num_v)
#     return dv, e
# --------------------------------------------------------------------------
# implicit RK schemes
# --------------------------------------------------------------------------
# def rk2_(K1):
#     K = [
#         [0.0, 0.0, 0.0, ],
#         [1.0, 0.5, 0.5, ],
#     ]
#     B2 = [0.5, 0.5, ]
#     B1 = [1.0, 0.0, ]
#     K0 = dt * a(K[0][0] * dt, v0)
#     K1 = dt * a(K[1][0] * dt, v0 + K[1][1] * K0 + K[1][2] * K1)
#     dv = B2[0] * K0 + B2[1] * K1
#     return dv, K1


def dr(vertices, edges, fixed, loads, qpre, fpre, lpre, linit, E, radius, ufunc=None, **kwargs):
    # --------------------------------------------------------------------------
    # configuration
    # --------------------------------------------------------------------------
    kmax  = kwargs.get('kmax', 10000)
    dt    = kwargs.get('dt', 1.0)
    tol1  = kwargs.get('tol1', 1e-3)
    tol2  = kwargs.get('tol2', 1e-6)
    coeff = Coeff(kwargs.get('c', 0.1))
    ca    = coeff.a
    cb    = coeff.b
    # --------------------------------------------------------------------------
    # attribute lists
    # --------------------------------------------------------------------------
    num_v = len(vertices)
    num_e = len(edges)
    free  = list(set(range(num_v)) - set(fixed))
    # --------------------------------------------------------------------------
    # attribute arrays
    # --------------------------------------------------------------------------
    xyz       = array(vertices, dtype=float).reshape((-1, 3))                   # m
    p         = array(loads, dtype=float).reshape((-1, 3))                      # kN
    qpre      = array(qpre, dtype=float).reshape((-1, 1))
    fpre      = array(fpre, dtype=float).reshape((-1, 1))                       # kN
    lpre      = array(lpre, dtype=float).reshape((-1, 1))                       # m
    linit     = array(linit, dtype=float).reshape((-1, 1))                      # m
    E         = array(E, dtype=float).reshape((-1, 1))                          # kN/mm2 => GPa
    radius    = array(radius, dtype=float).reshape((-1, 1))                     # mm
    # --------------------------------------------------------------------------
    # sectional properties
    # --------------------------------------------------------------------------
    A  = 3.14159 * radius ** 2                                                  # mm2
    EA = E * A                                                                  # kN
    # --------------------------------------------------------------------------
    # create the connectivity matrices
    # after spline edges have been aligned
    # --------------------------------------------------------------------------
    C   = connectivity_matrix(edges, 'csr')
    Ct  = C.transpose()
    Ci  = C[:, free]
    Cit = Ci.transpose()
    Ct2 = Ct.copy()
    Ct2.data **= 2
    # --------------------------------------------------------------------------
    # if none of the initial lengths are set,
    # set the initial lengths to the current lengths
    # --------------------------------------------------------------------------
    if all(linit == 0):
        linit = normrow(C.dot(xyz))
    # --------------------------------------------------------------------------
    # initial values
    # --------------------------------------------------------------------------
    q = ones((num_e, 1), dtype=float)
    l = normrow(C.dot(xyz))
    f = q * l
    v = zeros((num_v, 3), dtype=float)
    r = zeros((num_v, 3), dtype=float)
    # --------------------------------------------------------------------------
    # acceleration
    # --------------------------------------------------------------------------
    def a(t, v):
        dx        = v * t
        xyz[free] = xyz0[free] + dx[free]
        r[free]   = p[free] - D.dot(xyz)
        return cb * r / mass
    # --------------------------------------------------------------------------
    # start iterating
    # --------------------------------------------------------------------------
    for k in xrange(kmax):
        q_fpre = fpre / l
        q_lpre = f / lpre
        q_EA   = EA * (l - linit) / (linit * l)
        q_lpre[isinf(q_lpre)] = 0
        q_lpre[isnan(q_lpre)] = 0
        q_EA[isinf(q_EA)]     = 0
        q_EA[isnan(q_EA)]     = 0
        q      = qpre + q_fpre + q_lpre + q_EA
        Q      = diags([q[:, 0]], [0])
        D      = Cit.dot(Q).dot(C)
        mass   = 0.5 * dt ** 2 * Ct2.dot(qpre + q_fpre + q_lpre + EA / linit)
        xyz0   = xyz.copy()
        # ----------------------------------------------------------------------
        # RK
        # ----------------------------------------------------------------------
        v0        = ca * v.copy()
        dv        = rk2(a, v0, dt)
        v         = v0 + dv
        dx        = v * dt
        xyz[free] = xyz0[free] + dx[free]
        # update
        uvw = C.dot(xyz)
        l   = normrow(uvw)
        f   = q * l
        r   = p - Ct.dot(Q).dot(uvw)
        # crits
        crit1 = norm(r[free])
        crit2 = norm(dx[free])
        # ufunc
        if ufunc:
            ufunc(k, crit1, crit2)
        # convergence
        if crit1 < tol1:
            break
        if crit2 < tol2:
            break
        print(k)
    return xyz, q, f, l, r


def beam_data(beams, network):
    """ Create data for beam element calculations.

    Parameters:
        beams (dic): Dictionary of beam information.
        network (obj): Network to be analysed.

    Returns:
        list: Indices of all beam element start nodes.
        list: Indices of all beam element intermediate nodes.
        list: Indices of all beam element finish nodes beams.
        array: Nodal EIx flexural stiffnesses for all beams.
        array: Nodal EIy flexural stiffnesses for all beams.
    """
    inds, indi, indf = [], [], []
    EIx, EIy = [], []
    for key in beams:
        nodes = beams[key]['nodes']
        inds.extend(nodes[0:-2])
        indi.extend(nodes[1:-1])
        indf.extend(nodes[2:])
        EIx.extend([network.vertex[i]['EIx'] for i in nodes[1:-1]])
        EIy.extend([network.vertex[i]['EIy'] for i in nodes[1:-1]])
    EIx = array(EIx)[:, newaxis]
    EIy = array(EIy)[:, newaxis]
    return inds, indi, indf, EIx, EIy


def beam_shear(S, X, inds, indi, indf, EIx, EIy):
    """ Generate the beam nodal shear forces Sx, Sy and Sz.

    Parameters:
        S (array): Empty or populated beam nodal shear force array.
        X (array): Co-ordinates of nodes.
        inds (list): Indices of all beam element start nodes.
        indi (list): Indices of all beam element intermediate nodes.
        indf (list): Indices of all beam element finish nodes beams.
        EIx (array): Nodal EIx flexural stiffnesses for all beams.
        EIy (array): Nodal EIy flexural stiffnesses for all beams.

    Returns:
        array: Updated beam nodal shears.
    """
    S *= 0
    Xs = X[inds, :]
    Xi = X[indi, :]
    Xf = X[indf, :]
    Qa = Xi - Xs
    Qb = Xf - Xi
    Qc = Xf - Xs
    Qn = cross(Qa, Qb)
    Qnn = normrow(Qn)
    La = normrow(Qa)
    Lb = normrow(Qb)
    Lc = normrow(Qc)
    a = arccos((La**2 + Lb**2 - Lc**2) / (2 * La * Lb))
    k = 2 * sin(a) / Lc
    mu = -0.5 * Xs + 0.5 * Xf
    mun = normrow(mu)
    ex = Qn / tile(Qnn, (1, 3))  # Temporary simplification
    ez = mu / tile(mun, (1, 3))
    ey = cross(ez, ex)
    K = tile(k / Qnn, (1, 3)) * Qn
    Kx = tile(sum(K * ex, 1)[:, newaxis], (1, 3)) * ex
    Ky = tile(sum(K * ey, 1)[:, newaxis], (1, 3)) * ey
    Mc = EIx * Kx + EIy * Ky
    cma = cross(Mc, Qa)
    cmb = cross(Mc, Qb)
    ua = cma / tile(normrow(cma), (1, 3))
    ub = cmb / tile(normrow(cmb), (1, 3))
    c1 = cross(Qa, ua)
    c2 = cross(Qb, ub)
    Lc1 = normrow(c1)
    Lc2 = normrow(c2)
    M = sum(Mc**2, 1)[:, newaxis]
    Sa = ua * tile(M * Lc1 / (La * sum(Mc * c1, 1)[:, newaxis]), (1, 3))
    Sb = ub * tile(M * Lc2 / (Lb * sum(Mc * c2, 1)[:, newaxis]), (1, 3))
    Sa[isnan(Sa)] = 0
    Sb[isnan(Sb)] = 0
    S[inds, :] += Sa
    S[indi, :] += - Sa - Sb
    S[indf, :] += Sb
    # Add node junction duplication for when elements cross each other
    # mu[0, :] = -1.25*x[0, :] + 1.5*x[1, :] - 0.25*x[2, :]
    # mu[-1, :] = 0.25*x[-3, :] - 1.5*x[-2, :] + 1.25*x[-1, :]
    return S


def create_arrays(network):
    """ Create arrays needed for dr_solver.

    Parameters:
        network (obj): Network to analyse.

    Returns:
        array: Constraint conditions.
        array: Nodal loads Px, Py, Pz.
        array: Resultant nodal loads.
        array: Sx, Sy, Sz shear force components.
        array: x, y, z co-ordinates.
        array: Nodal velocities Vx, Vy, Vz.
        array: Edges' initial forces.
        array: Edges' initial lengths.
        list: Compression only edges.
        list: Tension only edges.
        array: Connectivity matrix.
        array: Transposed connectivity matrix.
        array: Axial stiffnesses.
        array: Network edges' start points.
        array: Network edges' end points.
        array: Mass matrix.
        list: Edge adjacencies (rows).
        list: Edge adjacencies (columns).
        list: Edge adjacencies (values).
        array: Young's moduli.
        array: Edge areas.
    """

    # Vertices

    n = len(network.vertices())
    B = zeros((n, 3))
    P = zeros((n, 3))
    X = zeros((n, 3))
    S = zeros((n, 3))
    V = zeros((n, 3))
    k_i = network.key_index()
    for key in network.vertices():
        i = k_i[key]
        vertex = network.vertex[key]
        B[i, :] = vertex['B']
        P[i, :] = vertex['P']
        X[i, :] = [vertex[j] for j in 'xyz']
    Pn = normrow(P)

    # Edges

    m = len(network.edges())
    E = zeros((m, 1))
    A = zeros((m, 1))
    s0 = zeros((m, 1))
    l0 = zeros((m, 1))
    u = []
    v = []
    ind_c = []
    ind_t = []
    edges = []
    uv_i = network.uv_index()
    for ui, vi in network.edges():
        i = uv_i[(ui, vi)]
        edge = network.edge[ui][vi]
        edges.append([k_i[ui], k_i[vi]])
        u.append(k_i[ui])
        v.append(k_i[vi])
        E[i] = edge['E']
        A[i] = edge['A']
        s0[i] = edge['s0']
        if edge['l0']:
            l0[i] = edge['l0']
        else:
            l0[i] = network.edge_length(ui, vi)
        if edge['ct'] == 'c':
            ind_c.append(i)
        elif edge['ct'] == 't':
            ind_t.append(i)
    f0 = s0 * A
    ks = E * A / l0

    # Arrays

    C = connectivity_matrix(edges, 'csr')
    Ct = C.transpose()
    M = mass_matrix(Ct, E, A, l0, f0, c=1, tiled=False)
    rows, cols, vals = find(Ct)

    return B, P, Pn, S, X, V, f0, l0, ind_c, ind_t, C, Ct, ks, array(u), array(v), M, rows, cols, vals, E, A


def dr_solver(tol, steps, C, Ct, V, M, B, S, P, X, f0, ks, l0, ind_c, ind_t, refresh, bmesh=False, factor=1, beams=None,
              inds=None, indi=None, indf=None, EIx=None, EIy=None):
    """ Numpy and SciPy dynamic relaxation solver.

    Parameters:
        tol (float): Tolerance limit.
        steps (int): Maximum number of iteration steps.
        C (array): Connectivity matrix.
        Ct (array): Transposed connectivity matrix.
        V (array): Nodal velocities Vx, Vy, Vz.
        M (array): Mass matrix (untiled).
        B (array): Constraint conditions.
        S (array): Sx, Sy, Sz shear force components.
        P (array): Nodal loads Px, Py, Pz.
        X (array): Nodal co-ordinates.
        f0 (array): Initial edge forces.
        ks (array): Initial edge stiffnesses.
        l0 (array): Initial edge lengths.
        ind_c (list): Compression only edges.
        ind_t (list): Tension only edges.
        refresh (int): Update progress every n steps.
        bmesh (bool): Draw Blender mesh updates on or off.
        factor (float): Convergence factor.
        beams (dic): Dictionary of beam information.
        inds (list): Indices of all beam element start nodes.
        indi (list): Indices of all beam element intermediate nodes.
        indf (list): Indices of all beam element finish nodes beams.
        EIx (array): Nodal EIx flexural stiffnesses for all beams.
        EIy (array): Nodal EIy flexural stiffnesses for all beams.

    Returns:
        array: Updated nodal co-ordinates.
        array: Final forces.
        array: Final lengths.
    """
    res = 1000 * tol
    ts, Uo = 0, 0
    M = factor * tile(M, (1, 3))
    while (ts <= steps) and (res > tol):
        uvw, l = lengths(C, X)
        f = f0 + ks * (l - l0)
        if ind_t:
            f[ind_t] *= f[ind_t] > 0
        if ind_c:
            f[ind_c] *= f[ind_c] < 0
        if beams:
            S = beam_shear(S, X, inds, indi, indf, EIx, EIy)
        q = tile(f / l, (1, 3))
        R = (P - S - Ct.dot(uvw * q)) * B
        Rn = normrow(R)
        res = mean(Rn)
        V += R / M
        Un = sum(0.5 * M * V**2)
        if Un < Uo:
            V *= 0
        Uo = Un
        X += V
        if refresh and (ts % refresh == 0):
            print('ts:{0} res:{1:.3g}'.format(ts, res))
            if bmesh:
                for c, Xi in enumerate(X):
                    bmesh.data.vertices[c].co = Xi
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        ts += 1
    if refresh:
        print('Iterations: {0}'.format(ts - 1))
        print('Residual: {0:.3g}'.format(res))
    return X, f, l


@jit(float64[:](float64[:], float64[:]), nogil=True, nopython=True)
def cross_numba(a, b):
    c = zeros(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


@jit(float64[:,:](float64, int64, int64[:], int64[:], float64[:,:], float64[:], float64[:], float64[:], int32[:],
     int32[:], float64[:], float64[:,:], float64[:,:], float64[:], int64[:], int64[:], float64, int64[:], int64[:],
     int64[:], float64[:], float64[:], int64, float64[:,:], int64), nogil=True, nopython=True)
def dr_solver_numba(tol, steps, u, v, X, f0, ks, l0, rows, cols, vals, P, B, M, ind_c, ind_t, factor, inds, indi, indf,
                    EIx, EIy, beams, S, refresh):
    """ Numba accelerated dynamic relaxation solver.

    Parameters:
        tol (float64): Tolerance limit.
        steps (int64): Maximum number of iteration steps.
        u (int64[:]): Network edges' start points.
        v (int64[:]): Network edges' end points.
        X (float64[:,:]): Nodal co-ordinates.
        f0 (float64[:]): Initial edge forces.
        ks (float64[:]): Initial edge stiffnesses.
        l0 (float64[:]): Initial edge lengths.
        rows (int32[:]): Edge adjacencies (rows).
        cols (int32[:]): Edge adjacencies (columns).
        vals (float64[:]): Edge adjacencies (values).
        P (float64[:,:]): Nodal loads Px, Py, Pz.
        B (float64[:,:]): Constraint conditions.
        M (float64[:]): Mass matrix.
        ind_c (int64[:]): Compression only edges.
        ind_t (int64[:]): Tension only edges.
        factor (float64): Convergence factor.
        inds (int64[:]): Indices of all beam element start nodes.
        indi (int64[:]): Indices of all beam element intermediate nodes.
        indf (int64[:]): Indices of all beam element finish nodes beams.
        EIx (float64[:]): Nodal EIx flexural stiffnesses for all beams.
        EIy (float64[:]): Nodal EIy flexural stiffnesses for all beams.
        beams (int64): Beam analysis on: 1 or off: 0.
        S (float64[:,:]): Empty shear force array.

    Returns:
        float64[:,:]: Updated nodal co-ordinates.
    """
    m = len(u)
    n = X.shape[0]
    f = zeros(m)
    fx = zeros(m)
    fy = zeros(m)
    fz = zeros(m)
    Vx = zeros(n)
    Vy = zeros(n)
    Vz = zeros(n)
    Rx = zeros(n)
    Ry = zeros(n)
    Rz = zeros(n)
    Rn = zeros(n)
    res = 1000 * tol
    ts = 0
    Uo = 0.0
    while (ts <= steps) and (res > tol):
        S *= 0
        if beams:
            for i in range(len(inds)):
                Xs = X[inds[i], :]
                Xi = X[indi[i], :]
                Xf = X[indf[i], :]
                Qa = Xi - Xs
                Qb = Xf - Xi
                Qc = Xf - Xs
                Qn = cross_numba(Qa, Qb)
                Qnn = sqrt(Qn[0]**2 + Qn[1]**2 + Qn[2]**2)
                La = sqrt(Qa[0]**2 + Qa[1]**2 + Qa[2]**2)
                Lb = sqrt(Qb[0]**2 + Qb[1]**2 + Qb[2]**2)
                Lc = sqrt(Qc[0]**2 + Qc[1]**2 + Qc[2]**2)
                a = arccos((La**2 + Lb**2 - Lc**2) / (2 * La * Lb))
                k = 2 * sin(a) / Lc
                mu = -0.5 * Xs + 0.5 * Xf
                mun = sqrt(mu[0]**2 + mu[1]**2 + mu[2]**2)
                ex = Qn / Qnn
                ez = mu / mun
                ey = cross_numba(ez, ex)
                K = k * Qn / Qnn
                Kx = (K[0] * ex[0] + K[1] * ex[1] + K[2] * ex[2]) * ex
                Ky = (K[0] * ey[0] + K[1] * ey[1] + K[2] * ey[2]) * ey
                Mc = EIx[i] * Kx + EIy[i] * Ky
                cma = cross_numba(Mc, Qa)
                cmb = cross_numba(Mc, Qb)
                ua = cma / sqrt(cma[0]**2 + cma[1]**2 + cma[2]**2)
                ub = cmb / sqrt(cmb[0]**2 + cmb[1]**2 + cmb[2]**2)
                c1 = cross_numba(Qa, ua)
                c2 = cross_numba(Qb, ub)
                Lc1 = sqrt(c1[0]**2 + c1[1]**2 + c1[2]**2)
                Lc2 = sqrt(c2[0]**2 + c2[1]**2 + c2[2]**2)
                Ms = Mc[0]**2 + Mc[1]**2 + Mc[2]**2
                Sa = ua * Ms * Lc1 / (La * (Mc[0] * c1[0] + Mc[1] * c1[1] + Mc[2] * c1[2]))
                Sb = ub * Ms * Lc2 / (Lb * (Mc[0] * c2[0] + Mc[1] * c2[1] + Mc[2] * c2[2]))
                S[inds[i], :] += Sa
                S[indi[i], :] += - Sa - Sb
                S[indf[i], :] += Sb
        for i in range(m):
            ui = u[i]
            vi = v[i]
            xd = X[vi, 0] - X[ui, 0]
            yd = X[vi, 1] - X[ui, 1]
            zd = X[vi, 2] - X[ui, 2]
            l = sqrt(xd**2 + yd**2 + zd**2)
            f[i] = f0[i] + ks[i] * (l - l0[i])
            q = f[i] / l
            fx[i] = xd * q
            fy[i] = yd * q
            fz[i] = zd * q
        if ind_t[0] != -1:
            for i in ind_t:
                if f[i] < 0:
                    fx[i] = 0
                    fy[i] = 0
                    fz[i] = 0
        if ind_c[0] != -1:
            for i in ind_c:
                if f[i] > 0:
                    fx[i] = 0
                    fy[i] = 0
                    fz[i] = 0
        Rx *= 0
        Ry *= 0
        Rz *= 0
        for i in range(len(vals)):
            r_i = rows[i]
            c_i = cols[i]
            v_i = vals[i]
            Rx[r_i] += - v_i * fx[c_i]
            Ry[r_i] += - v_i * fy[c_i]
            Rz[r_i] += - v_i * fz[c_i]
        Un = 0.0
        for i in range(n):
            Mi = M[i] * factor
            Rx[i] += P[i, 0] - S[i, 0]
            Ry[i] += P[i, 1] - S[i, 1]
            Rz[i] += P[i, 2] - S[i, 2]
            Rx[i] *= B[i, 0]
            Ry[i] *= B[i, 1]
            Rz[i] *= B[i, 2]
            Rn[i] = sqrt(Rx[i]**2 + Ry[i]**2 + Rz[i]**2)
            Vx[i] += Rx[i] / Mi
            Vy[i] += Ry[i] / Mi
            Vz[i] += Rz[i] / Mi
            Un += Mi * (Vx[i]**2 + Vy[i]**2 + Vz[i]**2)
        res = mean(Rn)
        if Un < Uo:
            Vx *= 0
            Vy *= 0
            Vz *= 0
        Uo = Un
        for i in range(n):
            X[i, 0] += Vx[i]
            X[i, 1] += Vy[i]
            X[i, 2] += Vz[i]
        ts += 1
    if refresh:
        print('Iterations: ', ts - 1)
        print('Residual: ', res)
    return X


def dr_run(network, factor=1.0, tol=1, steps=10000, refresh=0, beams=None, bmesh=False, scale=0, solver='numpy', cad='Blender'):
    """ Run dynamic relaxation analysis.

    Parameters:
        network (obj): Network to analyse.
        factor (float): Convergence factor.
        tol (float): Tolerance value.
        steps (int): Maximum number of iteration steps.
        refresh (int): Update progress every n steps.
        beams (dic): Beam members data.
        bmesh (bool): Draw Blender mesh updates on or off.
        scale (float): Scale on plotting axial forces.
        solver (str): 'numpy', 'numba'.
        cad (str): CAD program to plot final forces: 'Blender'.

    Returns:
        array: Vertex co-ordinates.
        array: Edge forces.
        array: Edge lengths.
    """

    # Setup

    tic = time.time()
    B, P, Pn, S, X, V, f0, l0, ind_c, ind_t, C, Ct, ks, u, v, M, rows, cols, vals, E, A = create_arrays(network)
    if beams:
        inds, indi, indf, EIx, EIy = beam_data(beams, network)
    else:
        inds, indi, indf, EIx, EIy = array([0]), array([0]), array([0]), array([0.0]), array([0.0])
    if refresh:
        print('----------------------------------')
        print('Setup time: {0:.3g}s'.format(time.time() - tic))

    # Solver

    if bmesh:
        edges = [[u[i], v[i]] for i in range(len(u))]
        bmesh = draw_bmesh('network', vertices=X, edges=edges, faces=[], layer=19)

    tic = time.time()
    if solver == 'numpy':
        X, f, l = dr_solver(tol, steps, C, Ct, V, M, B, S, P, X, f0, ks, l0, ind_c, ind_t, refresh, bmesh, factor,
                            beams, inds, indi, indf, EIx, EIy)
    elif solver == 'numba':
        if not ind_c:
            ind_c = [-1]
        if not ind_t:
            ind_t = [-1]
        ind_c = array(ind_c)
        ind_t = array(ind_t)
        if beams:
            EIx = EIx.ravel()
            EIy = EIy.ravel()
            beams = 1
        else:
            beams = 0
        f0 = f0.ravel()
        ks = ks.ravel()
        l0 = l0.ravel()
        M = M.ravel()
        X = dr_solver_numba(tol, steps, u, v, X, f0, ks, l0, rows, cols, vals, P, B, M, ind_c, ind_t, factor,
                            inds, indi, indf, EIx, EIy, beams, S, refresh)
        uvw, l = lengths(C, X)
        f = f0 + ks * (l.ravel() - l0)
    if refresh:
        print('Solver time: {0:.3g}s'.format(time.time() - tic))
        print('----------------------------------')

    if bmesh:
        for c, Xi in enumerate(X):
            bmesh.data.vertices[c].co = Xi

    # Plot results

    if scale:
        fsc = abs(f) / max(abs(f))
        log = (f > 0) * 1
        if cad == 'Blender':
            layer_clear(19)
            pipes = []
            colours = ['blue' if i else 'red' for i in log]
            sp = X[u, :]
            ep = X[v, :]
            for c in range(len(f)):
                r = scale * fsc[c]
                spc = sp[c, :]
                epc = ep[c, :]
                colour = colours[c]
                pipes.append({'radius': r, 'start': spc, 'end': epc, 'colour': colour, 'name': str(f[c])})
            objects = xdraw_pipes(pipes, div=4)
            objects_layer(objects, 19)

    return X, f, l


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    import compas
    from compas.datastructures.network import Network

    import matplotlib.pyplot as plt

    dva = {
        'is_fixed': False,
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'px': 0.0,
        'py': 0.0,
        'pz': 0.0,
        'rx': 0.0,
        'ry': 0.0,
        'rz': 0.0,
    }

    dea = {
        'qpre': 1.0,
        'fpre': 0.0,
        'lpre': 0.0,
        'linit': 0.0,
        'E': 0.0,
        'radius': 0.0,
    }

    lines = compas.get_data('lines.obj')

    network = Network.from_obj(lines)
    network.set_dva(dva)
    network.set_dea(dea)

    for key, attr in network.vertices_iter(True):
        attr['is_fixed'] = network.degree(key) == 1

    count = 1
    for u, v, attr in network.edges_iter(True):
        attr['qpre'] = count
        count += 1

    k2i = dict((key, index) for index, key in network.vertices_enum())

    vertices = [network.vertex_coordinates(key) for key in network.vertex]
    edges    = [(k2i[u], k2i[v]) for u, v in network.edges_iter()]
    fixed    = [k2i[key] for key, attr in network.vertices_iter(True) if attr['is_fixed']]
    loads    = [(attr['px'], attr['py'], attr['pz']) for key, attr in network.vertices_iter(True)]
    qpre     = [attr['qpre'] for u, v, attr in network.edges_iter(True)]
    fpre     = [attr['fpre'] for u, v, attr in network.edges_iter(True)]
    lpre     = [attr['lpre'] for u, v, attr in network.edges_iter(True)]
    linit    = [attr['linit'] for u, v, attr in network.edges_iter(True)]
    E        = [attr['E'] for u, v, attr in network.edges_iter(True)]
    radius   = [attr['radius'] for u, v, attr in network.edges_iter(True)]

    xdata  = []
    ydata1 = []
    ydata2 = []

    plt.show()

    axes   = plt.gca()
    line1, = axes.plot(xdata, ydata1, 'r-')
    line2, = axes.plot(xdata, ydata2, 'b-')
    axes.set_ylim(-10, 60)
    axes.set_xlim(0, 100)

    def plot_iterations(i, crit1, crit2):
        print(i, crit1, crit2)
        xdata.append(i)
        ydata1.append(crit1)
        ydata2.append(crit2)
        line1.set_xdata(xdata)
        line1.set_ydata(ydata1)
        line2.set_xdata(xdata)
        line2.set_ydata(ydata2)
        plt.draw()
        plt.pause(1e-17)

    xyz, q, f, l, r = dr(vertices, edges, fixed, loads, qpre, fpre, lpre, linit, E, radius, ufunc=plot_iterations)

    plt.show()

    for key, attr in network.vertices_iter(True):
        index = k2i[key]
        attr['x'] = xyz[index][0]
        attr['y'] = xyz[index][1]
        attr['z'] = xyz[index][2]

    network.plot()
