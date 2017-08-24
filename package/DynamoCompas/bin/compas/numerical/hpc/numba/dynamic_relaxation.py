import time

from numpy import arccos
from numpy import array
from numpy import mean
from numpy import sin
from numpy import sqrt
from numpy import zeros

from scipy.sparse import find

from numba import jit

from compas.numerical.methods.dynamic_relaxation import create_arrays
from compas.numerical.methods.dynamic_relaxation import beam_data

from compas.numerical.hpc.numba.spatial import numba_cross
from compas.numerical.hpc.numba.spatial import numba_vdot
from compas.numerical.hpc.numba.spatial import numba_length

from compas.numerical.geometry import uvw_lengths

try:
    from compas.cad.blender.utilities import draw_bmesh
    from compas.cad.blender.utilities import xdraw_pipes
except:
    pass


__author__     = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__  = 'Copyright 2017, BLOCK Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'liew@arch.ethz.ch'


@jit(nogil=True, nopython=True)
def numba_dr_solver(tol, steps, factor, u, v, X, ks, l0, f0, ind_c, ind_t, rows, cols, vals, P, S, B, M, summary,
                    inds, indi, indf, EIx, EIy, beams):
    """ Numba accelerated dynamic relaxation solver.

    Parameters:
        tol (float): Tolerance limit.
        steps (int): Maximum number of iteration steps.
        factor (float): Convergence factor.
        u (array): Network edges' start points.
        v (array): Network edges' end points.
        X (array): Nodal co-ordinates.
        ks (array): Initial edge axial stiffnesses.
        l0 (array) Initial edge lengths.
        f0 (array): Initial edge forces.
        ind_c (array): Compression only edges.
        ind_t (array): Tension only edges.
        rows (array): Edge adjacencies (rows).
        cols (array): Edge adjacencies (columns).
        vals (array): Edge adjacencies (values).
        P (array): Nodal loads Px, Py, Pz.
        S (array): Empty shear force array.
        B (array): Constraint conditions.
        M (array): Mass matrix.
        summary (int): Print summary 1 or 0.
        inds (array): Indices of beam element start nodes.
        indi (array): Indices of beam element intermediate nodes.
        indf (array): Indices of beam element finish nodes beams.
        EIx (array): Nodal EIx flexural stiffnesses.
        EIy (array): Nodal EIy flexural stiffnesses.
        beams (int): Beam analysis on: 1 or off: 0.

    Returns:
        array: Updated nodal co-ordinates.
    """
    m = len(u)
    n = X.shape[0]
    nv = len(vals)
    f = zeros(m)
    fx = zeros(m)
    fy = zeros(m)
    fz = zeros(m)
    Vx = zeros(n)
    Vy = zeros(n)
    Vz = zeros(n)
    frx = zeros(n)
    fry = zeros(n)
    frz = zeros(n)
    Rn = zeros(n)
    res = 1000 * tol
    ts = 0
    Uo = 0
    while (ts <= steps) and (res > tol):
        for i in range(m):
            xd = X[v[i], 0] - X[u[i], 0]
            yd = X[v[i], 1] - X[u[i], 1]
            zd = X[v[i], 2] - X[u[i], 2]
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
        S *= 0
        if beams:
            for i in range(len(inds)):
                Xs = X[inds[i], :]
                Xi = X[indi[i], :]
                Xf = X[indf[i], :]
                Qa = Xi - Xs
                Qb = Xf - Xi
                Qc = Xf - Xs
                Qn = numba_cross(Qa, Qb)
                mu = 0.5 * (Xf - Xs)
                La = numba_length(Qa)
                Lb = numba_length(Qb)
                Lc = numba_length(Qc)
                LQn = numba_length(Qn)
                Lmu = numba_length(mu)
                a = arccos((La**2 + Lb**2 - Lc**2) / (2 * La * Lb))
                k = 2 * sin(a) / Lc
                ex = Qn / LQn
                ez = mu / Lmu
                ey = numba_cross(ez, ex)
                K = k * Qn / LQn
                Kx = numba_vdot(K, ex) * ex
                Ky = numba_vdot(K, ey) * ey
                Mc = EIx[i] * Kx + EIy[i] * Ky
                cma = numba_cross(Mc, Qa)
                cmb = numba_cross(Mc, Qb)
                ua = cma / numba_length(cma)
                ub = cmb / numba_length(cmb)
                c1 = numba_cross(Qa, ua)
                c2 = numba_cross(Qb, ub)
                Lc1 = numba_length(c1)
                Lc2 = numba_length(c2)
                Ms = Mc[0]**2 + Mc[1]**2 + Mc[2]**2
                Sa = ua * Ms * Lc1 / (La * numba_vdot(Mc, c1))
                Sb = ub * Ms * Lc2 / (Lb * numba_vdot(Mc, c2))
                S[inds[i], :] += Sa
                S[indi[i], :] -= Sa + Sb
                S[indf[i], :] += Sb
        frx *= 0
        fry *= 0
        frz *= 0
        for i in range(nv):
            frx[rows[i]] += vals[i] * fx[cols[i]]
            fry[rows[i]] += vals[i] * fy[cols[i]]
            frz[rows[i]] += vals[i] * fz[cols[i]]
        Un = 0.
        for i in range(n):
            Rx = (P[i, 0] - S[i, 0] - frx[i]) * B[i, 0]
            Ry = (P[i, 1] - S[i, 1] - fry[i]) * B[i, 1]
            Rz = (P[i, 2] - S[i, 2] - frz[i]) * B[i, 2]
            Rn[i] = sqrt(Rx**2 + Ry**2 + Rz**2)
            Mi = M[i] * factor
            Vx[i] += Rx / Mi
            Vy[i] += Ry / Mi
            Vz[i] += Rz / Mi
            Un += Mi * (Vx[i]**2 + Vy[i]**2 + Vz[i]**2)
        if Un < Uo:
            Vx *= 0
            Vy *= 0
            Vz *= 0
        Uo = Un
        for i in range(n):
            X[i, 0] += Vx[i]
            X[i, 1] += Vy[i]
            X[i, 2] += Vz[i]
        res = mean(Rn)
        ts += 1
    if summary:
        print('Step:', ts, 'Residual:', res)
    return X


def numba_dr_run(network, factor=1.0, tol=0.1, steps=10000, summary=0, bmesh=False, scale=0, update=False):
    """ Run Numba accelerated dynamic relaxation analysis.

    Parameters:
        network (obj): Network to analyse.
        factor (float): Convergence factor.
        tol (float): Tolerance value.
        steps (int): Maximum number of iteration steps.
        summary (bool): Print summary at end.
        bmesh (bool): Draw Blender mesh on or off.
        scale (float): Scale on plotting axial forces.
        update (bool): Update the co-ordinates of the Network.

    Returns:
        array: Vertex co-ordinates.
        array: Edge forces.
        array: Edge lengths.
    """

    # Setup

    tic1 = time.time()
    X, B, P, Pn, S, V, E, A, C, Ct, f0, l0, ind_c, ind_t, u, v, M, ks = create_arrays(network)
    f0_, ks_, l0_, M_ = f0.ravel(), ks.ravel(), l0.ravel(), M.ravel()
    try:
        beams = 1
        inds, indi, indf, EIx, EIy = beam_data(network.beams, network)
        inds = array(inds)
        indi = array(indi)
        indf = array(indf)
        EIx = EIx.ravel()
        EIy = EIy.ravel()
    except AttributeError:
        beams = 0
        z0 = array([0])
        z1 = array([0.])
        inds, indi, indf, EIx, EIy = z0, z0, z0, z1, z1
    if not ind_c:
        ind_c = [-1]
    if not ind_t:
        ind_t = [-1]
    ind_c = array(ind_c)
    ind_t = array(ind_t)
    rows, cols, vals = find(Ct)
    toc1 = time.time() - tic1

    # Solver

    if summary:
        print('\n\nNumba DR -------------------------')
    tic2 = time.time()
    X = numba_dr_solver(tol, steps, factor, u, v, X, ks_, l0_, f0_, ind_c, ind_t, rows, cols, vals, P, S, B, M_, summary,
                        inds, indi, indf, EIx, EIy, beams)
    uvw, l = uvw_lengths(C, X)
    f = f0 + ks * (l - l0)
    toc2 = time.time() - tic2

    # Plot results

    if bmesh:
        edges = list(zip(u, v))
        bmesh = draw_bmesh('network', vertices=X, edges=edges, layer=19)

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

    if summary:
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


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    pass
