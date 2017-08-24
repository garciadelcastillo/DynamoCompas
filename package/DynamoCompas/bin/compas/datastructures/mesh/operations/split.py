from __future__ import print_function


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = [
    'split_edge_mesh',
    'split_face_mesh',
    'split_edge_trimesh',
]


def _split_halfedge(mesh, fkey, u, v, w):
    pass


def split_edge_mesh(mesh, u, v, t=0.5, allow_boundary=False):
    """Split and edge by inserting a vertex along its length.

    Parameters:
        u (str): The key of the first vertex of the edge.
        v (str): The key of the second vertex of the edge.
        t (float): The position of the inserted vertex.
        allow_boundary (bool): Split boundary edges, if True. Defaults to
            False.

    Returns:
        str: The key of the inserted vertex.

    Raises:
        ValueError: If `u` and `v` are not neighbours.
    """
    if t <= 0.0:
        raise ValueError('t should be greater than 0.0.')
    if t >= 1.0:
        raise ValueError('t should be smaller than 1.0.')

    # check if the split is legal
    # don't split if edge is on boundary
    fkey_uv = mesh.halfedge[u][v]
    fkey_vu = mesh.halfedge[v][u]
    if not allow_boundary:
        if fkey_uv is None or fkey_vu is None:
            return

    # coordinates
    x, y, z = mesh.edge_point(u, v, t)

    # the split vertex
    w = mesh.add_vertex(x=x, y=y, z=z)

    # split half-edge UV
    mesh.halfedge[u][w] = fkey_uv
    mesh.halfedge[w][v] = fkey_uv
    del mesh.halfedge[u][v]

    # update the UV face if it is not the `None` face
    if fkey_uv is not None:
        j = mesh.face[fkey_uv].index(v)
        mesh.face[fkey_uv].insert(j, w)

    # split half-edge VU
    mesh.halfedge[v][w] = fkey_vu
    mesh.halfedge[w][u] = fkey_vu
    del mesh.halfedge[v][u]

    # update the VU face if it is not the `None` face
    if fkey_vu is not None:
        i = mesh.face[fkey_vu].index(u)
        mesh.face[fkey_vu].insert(i, w)

    return w


def split_edge_trimesh(mesh, u, v, t=0.5, allow_boundary=False, interpolate_attr=False):
    """"""
    if t <= 0.0:
        raise ValueError('t should be greater than 0.0.')
    if t >= 1.0:
        raise ValueError('t should be smaller than 1.0.')

    # check if the split is legal
    # don't split if edge is on boundary
    fkey_uv = mesh.halfedge[u][v]
    fkey_vu = mesh.halfedge[v][u]

    if not allow_boundary:
        if fkey_uv is None or fkey_vu is None:
            return

    # coordinates
    x, y, z = mesh.edge_point(u, v, t)

    # the split vertex
    w = mesh.add_vertex(x=x, y=y, z=z)

    # the UV face
    if fkey_uv is None:
        mesh.halfedge[u][w] = None
        mesh.halfedge[w][v] = None
        del mesh.halfedge[u][v]
    else:
        face = mesh.face[fkey_uv]
        o = face[face.index(u) - 1]
        mesh.add_face([u, w, o])
        mesh.add_face([w, v, o])
        del mesh.halfedge[u][v]
        del mesh.face[fkey_uv]

    # the VU face
    if fkey_vu is None:
        mesh.halfedge[v][w] = None
        mesh.halfedge[w][u] = None
        del mesh.halfedge[v][u]
    else:
        face = mesh.face[fkey_vu]
        o = face[face.index(v) - 1]
        mesh.add_face([v, w, o])
        mesh.add_face([w, u, o])
        del mesh.halfedge[v][u]
        del mesh.face[fkey_vu]

    # return the key of the split vertex
    return w


def split_face_mesh(mesh, fkey, u, v):
    """Split a face by inserting an edge between two specified vertices.

    Parameters:
        fkey (str) : The face key.
        u (str) : The key of the first split vertex.
        v (str) : The key of the second split vertex.

    """
    if u not in mesh.face[fkey] or v not in mesh.face[fkey]:
        raise ValueError('The split vertices do not belong to the split face.')

    face = mesh.face[fkey]

    i = face.index(u)
    j = face.index(v)

    if i + 1 == j:
        raise ValueError('The split vertices are neighbours.')

    if j > i:
        f = face[i:j + 1]
        g = face[j:] + face[:i + 1]
    else:
        f = face[i:] + face[:j + 1]
        g = face[j:i + 1]

    f = mesh.add_face(f)
    g = mesh.add_face(g)

    del mesh.face[fkey]

    return f, g


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    import compas
    from compas.datastructures.mesh.mesh import Mesh
    from compas.visualization.plotters.meshplotter import MeshPlotter

    data = compas.get_data('faces.obj')
    mesh = Mesh.from_obj(data)

    split_edge_mesh(mesh, 17, 32)

    print(mesh.face_vertices(11, ordered=True))
    print(mesh.face_vertices(16, ordered=True))

    print(mesh.halfedge[32][36])
    print(mesh.halfedge[36][32])

    print(mesh.halfedge[36][17])
    print(mesh.halfedge[17][36])

    plotter = MeshPlotter(mesh)

    plotter.draw_vertices()
    plotter.draw_faces()

    plotter.show()
