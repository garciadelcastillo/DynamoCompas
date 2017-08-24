from compas.geometry import angle_smallest_vectors
from compas.geometry.queries import is_ccw_2d


__author__     = 'Tom Van Mele'
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = '<vanmelet@ethz.ch>'


__all__ = [
    'construct_dual_network',
    'find_network_faces',
]


def construct_dual_network(network, cls=None):
    """Construct the dual of a network.

    Parameters:
        network (compas.datastructures.network.Network): The network object.
        cls (compas.datastructures.network.Network):
            Optional.
            The class of the dual.
            Default is ``None``.
            If ``None``, the cls is inferred from the type of the provided network
            object.

    Warning:
        A network (or a graph) has a dual if, and only if, it is planar.
        Constructing the dual relies on the information about the faces of the
        network, or, in other words, about the ordering of neighbouring vertices
        around a vertex. To determine the faces of the network (using :func:`find_network_faces`)
        the network should be embedded in the plane, i.e drawn such that it is a
        proper cell decomposition of the plane (it divides the plane in non-overlapping
        spaces).

    Example:

        .. plot::
            :include-source:

            import compas
            from compas.datastructures.network import Network
            from compas.datastructures.network.algorithms import find_network_faces
            from compas.datastructures.network.algorithms import construct_dual_network

            network = Network.from_obj(compas.get_data('grid_irregular.obj'))

            find_network_faces(network, breakpoints=network.leaves())

            dual = construct_dual_network(network, Network)

            points = []
            for key in dual.vertices():
                points.append({
                    'pos': dual.vertex_coordinates(key, 'xy'),
                    'facecolor': '#ffffff',
                    'edgecolor': '#444444',
                    'textcolor': '#000000',
                    'size': 0.15,
                    'text': key
                })

            lines = []
            for u, v in dual.edges():
                lines.append({
                    'start': dual.vertex_coordinates(u, 'xy'),
                    'end': dual.vertex_coordinates(v, 'xy'),
                    'color': '#000000'
                })

            network.plot(
                vertices_on=True,
                vertexsize=0.075,
                vertexcolor={key: '#ff0000' for key in network.leaves()},
                edgecolor={(u, v): '#cccccc' for u, v in network.edges()},
                points=points,
                lines=lines
            )

    """
    if not cls:
        cls = type(network)
    dual = cls()
    for fkey in network.faces():
        x, y, z = network.face_center(fkey)
        dual.add_vertex(fkey, x=x, y=y, z=z)
    for u, v in network.edges():
        f1 = network.halfedge[u][v]
        f2 = network.halfedge[v][u]
        if f1 is not None and f2 is not None:
            dual.add_edge(f1, f2)
    return dual


def find_network_faces(network, breakpoints=None):
    """Find the faces of a network.

    Parameters:
        network (compas.datastructures.network.Network): The network object.
        breakpoints (list): Optional.
            The vertices at which to break the found faces.
            Default is ``None``.

    Note:
        ``breakpoints`` are primarily used to break up the outside face in between
        specific vertices. For example, in structural applications involving dual
        diagrams, any vertices where external forces are applied (loads or reactions)
        should be input as breakpoints.


    Warning:
        This algorithms is essentially a wall follower (a type of maze-solving algorithm).
        It relies on the geometry of the network to be repesented as a planar,
        straight-line embedding. It determines an ordering of the neighbouring vertices
        around each vertex, and then follows the *walls* of the network, always
        taking turns in the same direction.

    Example:

        Compare the faces on the plots of the same network, with and without
        breakpoints at the leaves.

        Note that with the breakpoints, face ``0`` (the outside face) no longer exists.
        Breaking up the face at the breakpoints happens after all faces have been
        found. Therefore, numbering of the faces replacing the outside face starts
        from the highest number of the faces found initially.


        .. plot::
            :include-source:

            # no breakpoints

            import compas
            from compas.datastructures.network import Network
            from compas.datastructures.network.algorithms import find_network_faces

            network = Network.from_obj(compas.get_data('grid_irregular.obj'))

            find_network_faces(network)

            network.plot(
                vertices_on=True,
                vertexsize=0.075,
                vertexcolor={key: '#cccccc' for key in network.leaves()},
                edgecolor={(u, v): '#cccccc' for u, v in network.edges()},
                facelabel={fkey: fkey for fkey in network.faces()}
            )

        .. plot::
            :include-source:

            # leaves as breakpoints

            import compas
            from compas.datastructures.network import Network
            from compas.datastructures.network.algorithms import find_network_faces

            network = Network.from_obj(compas.get_data('grid_irregular.obj'))

            find_network_faces(network, breakpoints=network.leaves())

            network.plot(
                vertices_on=True,
                vertexsize=0.075,
                vertexcolor={key: '#ff0000' for key in network.leaves()},
                edgecolor={(u, v): '#cccccc' for u, v in network.edges()},
                facelabel={fkey: fkey for fkey in network.faces()}
            )

    """
    if not breakpoints:
        breakpoints = []
    network.clear_facedict()
    network.clear_halfedgedict()
    network.halfedge = {key: {} for key in network.vertices()}
    for u, v in network.edges():
        network.halfedge[u][v] = None
        network.halfedge[v][u] = None
    _sort_neighbours(network)
    leaves = network.leaves()
    if leaves:
        u = sorted([(key, network.vertex[key]) for key in leaves], key=lambda x: (x[1]['y'], x[1]['x']))[0][0]
    else:
        u = sorted(network.vertices(True), key=lambda x: (x[1]['y'], x[1]['x']))[0][0]
    v = _find_first_neighbour(u, network)
    _find_edge_face(u, v, network)
    for u, v in network.edges():
        if network.halfedge[u][v] is None:
            _find_edge_face(u, v, network)
        if network.halfedge[v][u] is None:
            _find_edge_face(v, u, network)
    _break_faces(network, breakpoints)
    return network.face


def _find_first_neighbour(key, network):
    angles = []
    nbrs = list(network.halfedge[key].keys())
    if len(nbrs) == 1:
        return nbrs[0]
    vu = [-1, -1, 0]
    for nbr in nbrs:
        w = [network.vertex[nbr][_] for _ in 'xyz']
        v = [network.vertex[key][_] for _ in 'xyz']
        vw = [w[0] - v[0], w[1] - v[1], 0]
        angles.append(angle_smallest_vectors(vu, vw))
    return nbrs[angles.index(min(angles))]


def _sort_neighbours(network, ccw=True):
    sorted_neighbours = {}
    xyz = {key: network.vertex_coordinates(key) for key in network.vertices()}
    for key in network.vertices():
        nbrs = network.vertex_neighbours(key)
        if len(nbrs) == 1:
            sorted_neighbours[key] = nbrs
            continue
        ordered = [nbrs[0]]
        a = xyz[key]
        for i, nbr in enumerate(nbrs[1:]):
            c = xyz[nbr]
            pos = 0
            b = xyz[ordered[pos]]
            while not is_ccw_2d(a, b, c):
                pos += 1
                if pos > i:
                    break
                b = xyz[ordered[pos]]
            if pos == 0:
                pos = -1
                b = xyz[ordered[pos]]
                while is_ccw_2d(a, b, c):
                    pos -= 1
                    if pos < -len(ordered):
                        break
                    b = xyz[ordered[pos]]
                pos += 1
            ordered.insert(pos, nbr)
        if not ccw:
            sorted_neighbours[key] = ordered[::-1]
        sorted_neighbours[key] = ordered
    for key, nbrs in sorted_neighbours.items():
        network.vertex[key]['sorted_neighbours'] = nbrs[::-1]
    return sorted_neighbours


def _find_edge_face(u, v, network):
    cycle = [u]
    while True:
        cycle.append(v)
        nbrs = network.vertex[v]['sorted_neighbours']
        nbr = nbrs[nbrs.index(u) - 1]
        u, v = v, nbr
        if v == cycle[0]:
            # cycle.append(v)
            break
    fkey = network.add_face(cycle)
    return fkey


def _break_faces(network, breakpoints):
    breakpoints = set(breakpoints)

    for fkey in list(network.faces()):
        vertices = network.face_vertices(fkey)

        faces = []
        faces.append([vertices[0]])
        for i in range(1, len(vertices) - 1):
            key = vertices[i]
            faces[-1].append(key)
            if key in breakpoints:
                faces.append([key])

        faces[-1].append(vertices[-1])
        faces[-1].append(vertices[0])

        if len(faces) == 1:
            continue

        if faces[0][0] not in breakpoints and faces[-1][-1] not in breakpoints:
            if faces[0][0] == faces[-1][-1]:
                faces[:] = [faces[-1] + faces[0][1:]] + faces[1:-1]

        if len(faces) == 1:
            continue

        del network.face[fkey]

        if fkey in network.facedata:
            del network.facedata[fkey]

        for vertices in faces:
            network.add_face(vertices)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    import compas
    from compas.datastructures.network import Network
    from compas.visualization.plotters.networkplotter import NetworkPlotter

    network = Network.from_obj(compas.get_data('grid_irregular.obj'))

    find_network_faces(network, breakpoints=network.leaves())

    plotter = NetworkPlotter(network)

    plotter.draw_vertices(radius=0.075, facecolor={key: '#cccccc' for key in network.leaves()})
    plotter.draw_edges(color={(u, v): '#cccccc' for u, v in network.edges()})
    plotter.draw_faces(facecolor={fkey: '#eeeeee' for fkey in network.faces()},
                       text={fkey: fkey for fkey in network.faces()})

    plotter.show()
