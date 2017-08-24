from __future__ import print_function

from time import time

from numpy import abs
from numpy import arccos
from numpy import array
from numpy import cross
from numpy import isnan
from numpy import isinf
from numpy import int64
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

from compas.numerical.geometry import uvw_lengths
from compas.numerical.matrices import connectivity_matrix
from compas.numerical.matrices import mass_matrix
from compas.numerical.linalg import normrow

try:
    from compas.cad.blender.utilities import draw_bmesh
    from compas.cad.blender.utilities import xdraw_pipes
    from compas.cad.blender.geometry import update_bmesh_vertices
except:
    pass


__author__     = ['Tom Van Mele <vanmelet@ethz.ch>',
                  'Andrew Liew <liew@arch.ethz.ch>']
__copyright__  = 'Copyright 2016, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = [
    'rk1',
    'rk2',
    'rk3',
    'rk4',
    'dr'
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


def create_arrays(network):
    """ Create arrays for DR solver.

    Parameters:
        network (obj): Network to analyse.

    Returns:
        array: Nodal co-ordinates x, y, z.
        array: Constraint conditions Bx, By, Bz.
        array: Nodal loads Px, Py, Pz.
        array: Resultant nodal loads.
        array: Shear force components Sx, Sy, Sz.
        array: Nodal velocities Vx, Vy, Vz.
        array: Edge Young's moduli.
        array: Edge areas.
        array: Connectivity matrix.
        array: Transposed connectivity matrix.
        array: Edge initial forces.
        array: Edge initial lengths.
        list: Compression only edges indices.
        list: Tension only edges indices.
        array: Network edges' start points.
        array: Network edges' end points.
        array: Mass matrix.
        array: Edge axial stiffnesses.
    """

    # Vertices

    vertices = list(network.vertices())
    n = len(vertices)
    X = zeros((n, 3))
    B = zeros((n, 3))
    P = zeros((n, 3))
    S = zeros((n, 3))
    V = zeros((n, 3))
    k_i = network.key_index()
    for key in vertices:
        i = k_i[key]
        vertex = network.vertex[key]
        X[i, :] = [vertex[j] for j in 'xyz']
        B[i, :] = vertex.get('B', [1, 1, 1])
        P[i, :] = vertex.get('P', [0, 0, 0])
    Pn = normrow(P)

    # Edges

    edges = list(network.edges())
    m = len(edges)
    E = zeros((m, 1))
    A = zeros((m, 1))
    s0 = zeros((m, 1))
    l0 = zeros((m, 1))
    u = zeros(m, dtype=int64)
    v = zeros(m, dtype=int64)
    ind_c = []
    ind_t = []
    uv_i = network.uv_index()
    for c, uv in enumerate(edges):
        ui, vi = uv
        i = uv_i[(ui, vi)]
        edge = network.edge[ui][vi]
        E[i] = edge.get('E', 0)
        A[i] = edge.get('A', 0)
        l0[i] = edge.get('l0', network.edge_length(ui, vi))
        s0[i] = edge.get('s0', 0)
        ct = edge.get('ct', None)
        if ct == 'c':
            ind_c.append(i)
        elif ct == 't':
            ind_t.append(i)
        u[c] = k_i[ui]
        v[c] = k_i[vi]
    f0 = s0 * A
    ks = E * A / l0

    # Faces (unconfirmed testing formulation)

    faces = list(network.faces())
    if faces:
        for face in faces:
            fdata = network.facedata[face]
            Eh = fdata.get('E', 0)
            Ah = network.face_area(face)
            th = fdata.get('t', 0)
            for ui, vi in network.face_edges(face):
                i = uv_i[(ui, vi)]
                ks[i] += 1.5 * Eh * Ah * th / l0[i]**2

    # Arrays

    C = connectivity_matrix([[k_i[ui], k_i[vi]] for ui, vi in edges], 'csr')
    Ct = C.transpose()
    M = mass_matrix(Ct, ks, f0, c=1, tiled=False)

    return X, B, P, Pn, S, V, E, A, C, Ct, f0, l0, ind_c, ind_t, u, v, M, ks


def beam_data(beams, network):
    """ Create data for beam element calculations.

    Parameters:
        beams (dic): Dictionary of beam information.
        network (obj): Network to be analysed.

    Returns:
        list: Indices of beam element start nodes.
        list: Indices of beam element intermediate nodes.
        list: Indices of beam element finish nodes beams.
        array: Nodal EIx flexural stiffnesses of all beams.
        array: Nodal EIy flexural stiffnesses of all beams.
    """
    inds, indi, indf, EIx, EIy = [], [], [], [], []
    for i, beam in beams.items():
        nodes = beam['nodes']
        inds.extend(nodes[:-2])
        indi.extend(nodes[1:-1])
        indf.extend(nodes[2:])
        EIx.extend([network.vertex[i]['EIx'] for i in nodes[1:-1]])
        EIy.extend([network.vertex[i]['EIy'] for i in nodes[1:-1]])
    EIx = array(EIx)[:, newaxis]
    EIy = array(EIy)[:, newaxis]
    return inds, indi, indf, EIx, EIy


def dr_solver(tol, steps, factor, C, Ct, X, ks, l0, f0, ind_c, ind_t, P, S, B, M, V, refresh, bmesh, beams, inds, indi,
              indf, EIx, EIy):
    """ Numpy and SciPy dynamic relaxation solver.

    Parameters:
        tol (float): Tolerance limit.
        steps (int): Maximum number of iteration steps.
        factor (float): Convergence factor.
        C (array): Connectivity matrix.
        Ct (array): Transposed connectivity matrix.
        X (array): Nodal co-ordinates.
        ks (array): Initial edge axial stiffnesses.
        l0 (array): Initial edge lengths.
        f0 (array): Initial edge forces.
        ind_c (list): Compression only edges.
        ind_t (list): Tension only edges.
        P (array): Nodal loads Px, Py, Pz.
        S (array): Shear forces Sx, Sy, Sz.
        B (array): Constraint conditions.
        M (array): Mass matrix (untiled).
        V (array): Nodal velocities Vx, Vy, Vz.
        refresh (int): Update progress every n steps.
        bmesh (obj): Blender mesh to update.
        beams (bool): Dictionary of beam information.
        inds (list): Indices of beam element start nodes.
        indi (list): Indices of beam element intermediate nodes.
        indf (list): Indices of beam element finish nodes beams.
        EIx (array): Nodal EIx flexural stiffnesses.
        EIy (array): Nodal EIy flexural stiffnesses.

    Returns:
        array: Updated nodal co-ordinates.
        array: Final edge forces.
        array: Final edge lengths.
    """
    res = 1000 * tol
    ts = 0
    Uo = 0
    M = factor * tile(M, (1, 3))
    while (ts <= steps) and (res > tol):
        uvw, l = uvw_lengths(C, X)
        f = f0 + ks * (l - l0)
        if ind_t:
            f[ind_t] *= f[ind_t] > 0
        if ind_c:
            f[ind_c] *= f[ind_c] < 0
        if beams:
            S = beam_shear(S, X, inds, indi, indf, EIx, EIy)
        q = f / l
        qt = tile(q, (1, 3))
        R = (P - S - Ct.dot(uvw * qt)) * B
        res = mean(normrow(R))
        V += R / M
        Un = sum(M * V**2)
        if Un < Uo:
            V *= 0
        Uo = Un
        X += V
        if refresh and (ts % refresh == 0):
            print('Step:{0} Residual:{1:.3g}'.format(ts, res))
            if bmesh:
                update_bmesh_vertices(bmesh, X)
        ts += 1
    if refresh:
        print('Step:{0} Residual:{1:.3g}'.format(ts, res))
    return X, f, l


def dr_run(network, factor=1.0, tol=0.1, steps=10000, refresh=0, bmesh=False, scale=0, update=False):
    """ Run dynamic relaxation analysis.

    Parameters:
        network (obj): Network to analyse.
        factor (float): Convergence factor.
        tol (float): Tolerance value.
        steps (int): Maximum number of iteration steps.
        refresh (int): Update progress every n steps.
        bmesh (bool): Draw Blender mesh updates on or off.
        scale (float): Scale on plotting axial forces.
        update (bool): Update the co-ordinates of the Network.

    Returns:
        array: Vertex co-ordinates.
        array: Edge forces.
        array: Edge lengths.
    """

    # Setup

    tic1 = time()
    X, B, P, Pn, S, V, E, A, C, Ct, f0, l0, ind_c, ind_t, u, v, M, ks = create_arrays(network)
    try:
        beams = network.beams
        inds, indi, indf, EIx, EIy = beam_data(beams, network)
    except AttributeError:
        beams = inds = indi = indf = EIx = EIy = None
    toc1 = time() - tic1
    if bmesh:
        k_i = network.key_index()
        edges = [[u[i], v[i]] for i in range(len(u))]
        faces = []
        for fkey in list(network.faces()):
            faces.append([k_i[key] for key in network.face[fkey]])
        bmesh = draw_bmesh('network', vertices=X, edges=edges, faces=faces, layer=19)

    # Solver

    tic2 = time()
    X, f, l = dr_solver(tol, steps, factor, C, Ct, X, ks, l0, f0, ind_c, ind_t, P, S, B, M, V, refresh, bmesh, beams,
                        inds, indi, indf, EIx, EIy)
    toc2 = time() - tic2

    # Plot results

    if scale:
        fsc = abs(f) / max(abs(f))
        log = (f > 0) * 1
        colours = ['blue' if i else 'red' for i in log]
        pipes = []
        sp = X[u, :]
        ep = X[v, :]
        for c in range(len(f)):
            r = scale * fsc[c]
            pipes.append({'radius': r, 'start': sp[c, :], 'end': ep[c, :], 'colour': colours[c], 'name': str(f[c]), 'layer': 19})
        xdraw_pipes(pipes, div=4)

    # Summary

    if refresh:
        print('\n\nNumPy-SciPy DR -------------------')
        print('Setup time: {0:.3g}s'.format(toc1))
        print('Solver time: {0:.3g}s'.format(toc2))
        print('----------------------------------')

    # Update

    if update:
        k_i = network.key_index()
        for key in list(network.vertices()):
            i = k_i[key]
            x, y, z = X[i, :]
            network.set_vertex_attributes(i, {'x': x, 'y': y, 'z': z})

    return X, f, l


def beam_shear(S, X, inds, indi, indf, EIx, EIy):
    """ Generate the beam nodal shear forces Sx, Sy and Sz.

    Parameters:
        S (array): Nodal shear force array.
        X (array): Co-ordinates of nodes.
        inds (list): Indices of beam element start nodes.
        indi (list): Indices of beam element intermediate nodes.
        indf (list): Indices of beam element finish nodes beams.
        EIx (array): Nodal EIx flexural stiffnesses.
        EIy (array): Nodal EIy flexural stiffnesses.

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
    mu = 0.5 * (Xf - Xs)
    La = normrow(Qa)
    Lb = normrow(Qb)
    Lc = normrow(Qc)
    LQn = normrow(Qn)
    Lmu = normrow(mu)
    a = arccos((La**2 + Lb**2 - Lc**2) / (2 * La * Lb))
    k = 2 * sin(a) / Lc
    ex = Qn / tile(LQn, (1, 3))  # temporary simplification
    ez = mu / tile(Lmu, (1, 3))
    ey = cross(ez, ex)
    K = tile(k / LQn, (1, 3)) * Qn
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
    Ms = sum(Mc**2, 1)[:, newaxis]
    Sa = ua * tile(Ms * Lc1 / (La * sum(Mc * c1, 1)[:, newaxis]), (1, 3))
    Sb = ub * tile(Ms * Lc2 / (Lb * sum(Mc * c2, 1)[:, newaxis]), (1, 3))
    Sa[isnan(Sa)] = 0
    Sb[isnan(Sb)] = 0
    S[inds, :] += Sa
    S[indi, :] -= Sa + Sb
    S[indf, :] += Sb
    # Add node junction duplication for when elements cross each other
    # mu[0, :] = -1.25*x[0, :] + 1.5*x[1, :] - 0.25*x[2, :]
    # mu[-1, :] = 0.25*x[-3, :] - 1.5*x[-2, :] + 1.25*x[-1, :]
    return S


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
