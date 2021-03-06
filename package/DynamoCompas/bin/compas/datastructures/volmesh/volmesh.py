import json

from ast import literal_eval as _eval
from math import sqrt

from compas.files.obj import OBJ

from compas.datastructures.mesh import Mesh
from compas.datastructures.volmesh.exceptions import VolMeshError

from compas.geometry import centroid_points


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


def center_of_mass(edges, sqrt=sqrt):
    L  = 0
    cx = 0
    cy = 0
    cz = 0
    for sp, ep in edges:
        l   = sqrt(sum((sp[axis] - ep[axis]) ** 2 for axis in range(3)))
        cx += l * 0.5 * (sp[0] + ep[0])
        cy += l * 0.5 * (sp[1] + ep[1])
        cz += l * 0.5 * (sp[2] + ep[2])
        L  += l
    cx = cx / L
    cy = cy / L
    cz = cz / L
    return cx, cy, cz


class VolMesh(object):
    """Class for working with volumetric meshes.

    Volumetric meshes are 3-mainfold, cellular structures.

    The implementation of ``VolMesh`` is based on the notion of *x-maps* [xmaps]
    and the concepts behind the *OpenVolumeMesh* library [ovm].
    In short, we add an additional entity compared to polygonal meshes,
    the *cell*, and relate cells not through *half-edges*, but *half-planes*.

    References:
        .. [xmaps] xxx
        .. [ovm] `Open Volum Mesh <http://www.openvolumemesh.org>`_
    """

    def __init__(self):
        self._max_int_key  = -1
        self._max_int_fkey = -1
        self._max_int_ckey = -1
        self._key_to_str   = False
        self.vertex   = {}
        self.plane    = {}
        self.halfface = {}
        self.cell     = {}
        self.edge     = {}
        self.attributes = {
            'name'                : 'VolMesh',
            'color.vertex'        : (255, 255, 255),
            'color.edge'          : (0, 0, 0),
            'color.face'          : (200, 200, 200),
            'color.normal:vertex' : (0, 255, 0),
            'color.normal:face'   : (0, 255, 0),
        }
        self.default_vertex_attributes = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        self.default_edge_attributes = {}

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # magic methods
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def __contains__(self, key):
        return key in self.vertex

    def __len__(self):
        return len(self.vertex)

    def __iter__(self):
        return iter(self.vertex)

    def __getitem__(self, key):
        return self.vertex[key]

    def __str__(self):
        """"""
        return """
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
volmesh summary
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

under construction

"""

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # descriptors
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    @property
    def name(self):
        """The name of the mesh."""
        return self.attributes.get('name', None)

    @name.setter
    def name(self, value):
        self.attributes['name'] = value

    @property
    def color(self):
        return dict(
            (key[6:], self.attributes[key])
            for key in self.attributes if key.startswith('color.')
        )

    @color.setter
    def color(self, value):
        try:
            value[0]
            value[1]
            value[1][2]
        except Exception:
            return
        self.attributes['color.{0}'.format(value[0])] = value[1]

    @property
    def data(self):
        """The data representing the mesh."""
        data = {
            'attributes'               : self.attributes,
            'default_vertex_attributes': self.default_vertex_attributes,
            'default_edge_attributes'  : self.default_edge_attributes,
            'vertex'                   : {},
            'cell'                     : {},
            'halfface'                 : {},
            'plane'                    : {},
            'edge'                     : {},
            'max_int_key'              : self._max_int_key,
            'max_int_fkey'             : self._max_int_fkey,
            'max_int_ckey'             : self._max_int_ckey, }

        key_rkey = {}

        for key in self.vertex:
            rkey = repr(key)
            key_rkey[key] = rkey
            data['vertex'][rkey] = self.vertex[key]
            data['plane'][rkey] = {}
            data['edge'][rkey] = {}

        for u in self.edge:
            ru = key_rkey[u]
            for v in self.edge[u]:
                rv = key_rkey[v]
                data['edge'][ru][rv] = self.edge[u][v]

        for f in self.halfface:
            _f = repr(f)
            data['halfface'][_f] = {}
            for u, v in self.halfface[f].iteritems():
                _u = repr(u)  # use the map?
                _v = repr(v)  # use the map?
                data['halfface'][_f][_u] = _v

        for u in self.plane:
            _u = repr(u)
            for v in self.plane[u]:
                _v = repr(v)
                if _v not in data['plane'][_u]:
                    data['plane'][_u][_v] = {}
                for w, c in self.plane[u][v].iteritems():
                    _w = repr(w)
                    _c = repr(c)
                    data['plane'][_u][_v][_w] = _c

        for c in self.cell:
            _c = repr(c)
            data['cell'][_c] = {}
            for u in self.cell[c]:
                _u = repr(u)
                if _u not in data['cell'][_c]:
                    data['cell'][_c][_u] = {}
                for v, f in self.cell[c][u].iteritems():
                    _v = repr(v)
                    _f = repr(f)
                    data['cell'][_c][_u][_v] = _f

        return data

    @data.setter
    def data(self, data):
        """"""
        attributes                = data.get('attributes') or {}
        default_vertex_attributes = data.get('default_vertex_attributes') or {}
        default_edge_attributes   = data.get('default_edge_attributes') or {}
        vertex                    = data.get('vertex') or {}
        cell                      = data.get('cell') or {}
        halfface                  = data.get('halfface') or {}
        plane                     = data.get('plane') or {}
        edge                      = data.get('edge') or {}
        max_int_key               = data.get('max_int_key', - 1)
        max_int_fkey              = data.get('max_int_fkey', - 1)
        max_int_ckey              = data.get('max_int_ckey', - 1)

        if not vertex or not edge or not plane or not halfface or not cell:
            return

        self.clear()

        self.attributes.update(attributes)
        self.default_vertex_attributes.update(default_vertex_attributes)
        self.default_edge_attributes.update(default_edge_attributes)

        for _k, attr in vertex.iteritems():
            k = _eval(_k)
            self.vertex[k] = self.default_vertex_attributes.copy()
            if attr:
                self.vertex[k].update(attr)
            self.plane[k] = {}
            self.edge[k] = {}

        for _u, nbrs in edge.iteritems():
            nbrs = nbrs or {}
            u = _eval(_u)
            for _v, attr in nbrs.iteritems():
                v = _eval(_v)
                self.edge[u][v] = self.default_edge_attributes.copy()
                if attr:
                    self.edge[u][v].update(attr)

        for _f in halfface:
            f = _eval(_f)
            self.halfface[f] = {}
            for _u, _v in halfface[_f].iteritems():
                u = _eval(_u)
                v = _eval(_v)
                self.halfface[f][u] = v

        for _u in plane:
            u = _eval(_u)
            for _v in plane[_u]:
                v = _eval(_v)
                if v not in self.plane[u]:
                    self.plane[u][v] = {}
                for _w, _c in plane[_u][_v].iteritems():
                    w = _eval(_w)
                    c = _eval(_c)
                    self.plane[u][v][w] = c

        for _c in cell:
            c = _eval(_c)
            self.cell[c] = {}
            for _u in cell[_c]:
                u = _eval(_u)
                if u not in self.cell[c]:
                    self.cell[c][u] = {}
                for _v, _f in cell[_c][_u].iteritems():
                    v = _eval(_v)
                    f = _eval(_f)
                    self.cell[c][u][v] = f

        self._max_int_key = max_int_key
        self._max_int_fkey = max_int_fkey
        self._max_int_ckey = max_int_ckey

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # constructors
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    @classmethod
    def from_vertices_and_cells(cls, vertices, cells):
        mesh = cls()
        for x, y, z in vertices:
            mesh.add_vertex(x=x, y=y, z=z)
        for halffaces in cells:
            mesh.add_cell(halffaces)
        return mesh

    @classmethod
    def from_vertices_and_edges(cls, vertices, edges):
        raise NotImplementedError

    @classmethod
    def from_data(cls, data):
        volmesh = cls()
        volmesh.data = data
        return volmesh

    @classmethod
    def from_json(cls, filepath):
        volmesh = cls()
        data = None
        with open(filepath, 'rb') as fp:
            data = json.load(fp)
        if data:
            volmesh.data = data
        return volmesh

    @classmethod
    def from_obj(cls, filepath):
        obj = OBJ(filepath)
        vertices = obj.parser.vertices
        faces = obj.parser.faces
        groups = obj.parser.groups
        cells = []
        for name in groups:
            group = groups[name]
            cell = []
            for item in group:
                if item[0] != 'f':
                    continue
                face = faces[item[1]]
                cell.append(face)
            cells.append(cell)
        return cls.from_vertices_and_cells(vertices, cells)

    # --------------------------------------------------------------------------
    # special
    # --------------------------------------------------------------------------

    # def dual(self, cls):
    #     network = cls()
    #     for ckey in self.cell:
    #         x, y, z = self.cell_center(ckey)
    #         network.add_vertex(key=ckey, x=x, y=y, z=z)
    #         for nbr in self.cell_neighbours(ckey):
    #             if nbr in network.edge[ckey]:
    #                 continue
    #             if nbr in network.edge and ckey in network.edge[nbr]:
    #                 continue
    #             network.add_edge(ckey, nbr)
    #     return network

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # converters
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def to_data(self):
        return self.data

    def to_json(self, filepath):
        with open(filepath, 'wb+') as fp:
            json.dump(self.data, fp)

    def to_obj(self, filepath):
        raise NotImplementedError

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # helpers
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def clear(self):
        del self.vertex
        del self.cell
        del self.halfface
        del self.plane
        del self.edge
        self.vertex = {}
        self.cell = {}
        self.halfface = {}
        self.plane = {}
        self.edge = {}
        self._max_int_key = -1
        self._max_int_fkey = -1
        self._max_int_ckey = -1

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # modifiers
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def _get_vertexkey(self, key):
        if key is None:
            key = self._max_int_key = self._max_int_key + 1
        else:
            if int(key) > self._max_int_key:
                self._max_int_key = int(key)
        return key

    def _get_facekey(self, fkey):
        if fkey is None:
            fkey = self._max_int_fkey = self._max_int_fkey + 1
        else:
            if int(fkey) > self._max_int_fkey:
                self._max_int_fkey = int(fkey)
        return fkey

    def _get_cellkey(self, ckey):
        if ckey is None:
            ckey = self._max_int_ckey = self._max_int_ckey + 1
        else:
            if int(ckey) > self._max_int_ckey:
                self._max_int_ckey = int(ckey)
        return ckey

    def add_vertex(self, vkey=None, attr_dict=None, **kwattr):
        attr = self.default_vertex_attributes.copy()

        if attr_dict:
            attr.update(attr_dict)

        attr.update(kwattr)

        vkey = self._get_vertexkey(vkey)

        if vkey not in self.vertex:
            self.vertex[vkey] = attr
            self.plane[vkey] = {}
            self.edge[vkey] = {}
        # else:
        #     self.vertex[vkey].update(attr)
        return vkey

    def add_halfface(self, vertices, fkey=None):
        if vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        if vertices[-2] == vertices[-1]:
            vertices = vertices[:-1]

        if len(vertices) < 3:
            raise VolMeshError('Corrupt halfface.')

        fkey = self._get_facekey(fkey)

        self.halfface[fkey] = {}

        for i in range(-2, len(vertices) - 2):
            u = vertices[i]
            v = vertices[i + 1]
            w = vertices[i + 2]

            self.add_vertex(vkey=u)
            self.add_vertex(vkey=v)
            self.add_vertex(vkey=w)

            self.halfface[fkey][u] = v
            self.halfface[fkey][v] = w

            if v not in self.plane[u]:
                self.plane[u][v] = {}

            self.plane[u][v][w] = None

            if v not in self.plane[w]:
                self.plane[w][v] = {}
            if u not in self.plane[w][v]:
                self.plane[w][v][u] = None

            if v not in self.edge[u] and u not in self.edge[v]:
                self.edge[u][v] = {}
            if w not in self.edge[v] and v not in self.edge[w]:
                self.edge[v][w] = {}

        u = vertices[-1]
        v = vertices[0]

        if v not in self.edge[u] and u not in self.edge[v]:
            self.edge[u][v] = {}

        return fkey

    def add_cell(self, halffaces, ckey=None):
        ckey = self._get_cellkey(ckey)

        self.cell[ckey] = {}

        for vertices in halffaces:
            fkey = self.add_halfface(vertices)
            for u in self.halfface[fkey]:
                v = self.halfface[fkey][u]
                w = self.halfface[fkey][v]
                if u not in self.cell[ckey]:
                    self.cell[ckey][u] = {}
                self.cell[ckey][u][v] = fkey
                self.plane[u][v][w] = ckey

        return ckey

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # lists and iterators
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def vertices(self, data=False):
        return list(self.vertices_iter(data=data))

    def vertices_iter(self, data=False):
        for key in self.vertex:
            if data:
                yield key, self.vertex[key]
            else:
                yield key

    def vertices_enum(self, data=False):
        return enumerate(self.vertices_iter(data=data))

    def cells(self, data=False):
        return list(self.cells_iter(data=data))

    def cells_iter(self, data=False):
        for ckey in self.cell:
            if data:
                raise NotImplementedError
                # data should be stored on the dual network
                # yield ckey, self.cell[ckey]
            else:
                yield ckey

    def cells_enum(self, data=False):
        return enumerate(self.cells_iter(data=data))

    def planes(self):
        raise NotImplementedError

    def planes_iter(self):
        raise NotImplementedError

    def edges(self, data=False):
        return list(self.edges_iter(data))

    def edges_iter(self, data=False):
        for u in self.edge:
            for v in self.edge[u]:
                if data:
                    yield u, v, self.edge[u][v]
                else:
                    yield u, v

    def edges_enum(self, data=False):
        return enumerate(self.edges_iter(data=data))

    # --------------------------------------------------------------------------
    # special purpose
    # --------------------------------------------------------------------------

    def faces(self):
        faces = []
        seen = set()
        for ckey in self.cell:
            for fkey in self.cell_halffaces(ckey):
                vertices = self.halfface_vertices(fkey, ordered=True)
                vset = frozenset(vertices)
                if vset not in seen:
                    faces.append(vertices)
                seen.add(vset)
        return faces

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # topology
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def vertex_neighbours(self, vkey):
        return self.plane[vkey].keys()

    def cell_neighbours(self, ckey):
        nbrs = []
        for fkey in self.cell_halffaces(ckey):
            u   = self.halfface[fkey].iterkeys().next()
            v   = self.halfface[fkey][u]
            w   = self.halfface[fkey][v]
            nbr = self.plane[w][v][u]
            if nbr is not None:
                nbrs.append(nbr)
        return nbrs

    def cell_vertex_neighbours(self, ckey):
        raise NotImplementedError

    def halfface_cell(self, fkey):
        u = self.halfface[fkey].iterkeys().next()
        v = self.halfface[fkey][u]
        w = self.halfface[fkey][v]
        return self.plane[u][v][w]

    def halfface_vertices(self, fkey, ordered=False):
        if not ordered:
            return self.halfface[fkey].keys()
        u = self.halfface[fkey].iterkeys().next()
        vertices = [u]
        while True:
            u = self.halfface[fkey][u]
            if u == vertices[0]:
                break
            vertices.append(u)
        return vertices

    def halfface_edges(self, fkey):
        vertices = self.halfface_vertices(fkey, ordered=True)
        edges = []
        for i in range(-1, len(vertices) - 1):
            edges.append((vertices[i], vertices[i + 1]))
        return edges

    def halfface_adjacency(self, ckey):
        raise NotImplementedError

    def cell_halffaces(self, ckey):
        halffaces = set()
        for u in self.cell[ckey]:
            for v in self.cell[ckey][u]:
                fkey = self.cell[ckey][u][v]
                halffaces.add(fkey)
        return list(halffaces)

    def cell_vertices(self, ckey):
        return list(set([vkey for fkey in self.cell_halffaces(ckey) for vkey in self.halfface_vertices(fkey)]))

    def cell_edges(self, ckey):
        halfedges = []
        for fkey in self.cell_halffaces(ckey):
            halfedges += self.halfface_edges(fkey)
        edges = set(frozenset(uv) for uv in halfedges)
        return map(list, edges)

    def cell_vertices_and_halffaces(self, ckey):
        vkeys = self.cell_vertices(ckey)
        fkeys = self.cell_halffaces(ckey)
        vkey_vindex = dict((vkey, index) for index, vkey in enumerate(vkeys))
        vertices = [self.vertex_coordinates(vkey) for vkey in vkeys]
        halffaces = [[vkey_vindex[vkey] for vkey in self.halfface_vertices(fkey, ordered=True)] for fkey in fkeys]
        return vertices, halffaces

    def cell_adjacency(self):
        raise NotImplementedError

    def cell_tree(self, root):
        raise NotImplementedError

    def cell_mesh(self, ckey):
        vertices, halffaces = self.cell_vertices_and_halffaces(ckey)
        return Mesh.from_vertices_and_faces(vertices, halffaces)

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # geometry
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    def vertex_coordinates(self, vkey, axes='xyz'):
        attr = self.vertex[vkey]
        return [attr[axis] for axis in axes]

    def edge_coordinates(self, u, v, axes='xyz'):
        return self.vertex_coordinates(u, axes=axes), self.vertex_coordinates(v, axes=axes)

    def face_coordinates(self, fkey, axes='xyz'):
        raise NotImplementedError

    def cell_centroid(self, ckey):
        vkeys = self.cell_vertices(ckey)
        return centroid_points([self.vertex_coordinates(vkey) for vkey in vkeys])

    def cell_center(self, ckey):
        edges = self.cell_edges(ckey)
        return center_of_mass([(self.vertex_coordinates(u), self.vertex_coordinates(v)) for u, v in edges])

    # --------------------------------------------------------------------------
    # geometric operations
    # --------------------------------------------------------------------------

    def scale(self, factor=1.0):
        for key in self.vertex:
            attr = self.vertex[key]
            attr['x'] *= factor
            attr['y'] *= factor
            attr['z'] *= factor

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # attributes
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************

    # --------------------------------------------------------------------------
    # vertex attributes
    # --------------------------------------------------------------------------

    def update_default_vertex_attributes(self, attr_dict=None, **kwattr):
        if not attr_dict:
            attr_dict = {}
        attr_dict.update(kwattr)
        self.default_vertex_attributes.update(attr_dict)
        for key in self.vertex:
            attr = attr_dict.copy()
            attr.update(self.vertex[key])
            self.vertex[key] = attr

    def set_vertex_attribute(self, key, name, value):
        self.vertex[key][name] = value

    def set_vertex_attributes(self, key, attr_dict=None, **kwattr):
        attr_dict = attr_dict or {}
        attr_dict.update(kwattr)
        self.vertex[key].update(attr_dict)

    def set_vertices_attribute(self, name, value, keys=None):
        if not keys:
            for key, attr in self.vertices_iter(True):
                attr[name] = value
        else:
            for key in keys:
                self.vertex[key][name] = value

    def set_vertices_attributes(self, keys=None, attr_dict=None, **kwattr):
        attr_dict = attr_dict or {}
        attr_dict.update(kwattr)
        if not keys:
            for key, attr in self.vertices_iter(True):
                attr.update(attr_dict)
        else:
            for key in keys:
                self.vertex[key].update(attr_dict)

    def get_vertex_attribute(self, key, name, default=None):
        return self.vertex[key].get(name, default)

    def get_vertex_attributes(self, key, names, defaults=None):
        if not defaults:
            defaults = [None] * len(names)
        return [self.vertex[key].get(name, default) for name, default in zip(names, defaults)]

    def get_vertices_attribute(self, name, default=None, keys=None):
        if not keys:
            return [attr.get(name, default) for key, attr in self.vertices_iter(True)]
        return [self.vertex[key].get(name, default) for key in keys]

    def get_vertices_attributes(self, names, defaults=None, keys=None):
        if not defaults:
            defaults = [None] * len(names)
        temp = zip(names, defaults)
        if not keys:
            return [[attr.get(name, default) for name, default in temp] for key, attr in self.vertices_iter(True)]
        return [[self.vertex[key].get(name, default) for name, default in temp] for key in keys]

    # --------------------------------------------------------------------------
    # edge attributes
    # --------------------------------------------------------------------------

    def update_default_edge_attributes(self, attr_dict=None, **kwargs):
        if not attr_dict:
            attr_dict = {}
        attr_dict.update(kwargs)
        self.default_edge_attributes.update(attr_dict)
        for u, v in self.edges_iter():
            attr = attr_dict.copy()
            attr.update(self.edge[u][v])
            self.edge[u][v] = attr

    def set_edge_attribute(self, u, v, name, value):
        self.edge[u][v][name] = value

    def set_edge_attributes(self, u, v, attr_dict=None, **kwattr):
        attr_dict = attr_dict or kwattr
        attr_dict.update(kwattr)
        self.edge[u][v].update(attr_dict)

    def set_edges_attribute(self, name, value, keys=None):
        if not keys:
            for u, v, attr in self.edges_iter(True):
                attr[name] = value
        else:
            for u, v in keys:
                self.edge[u][v][name] = value

    def set_edges_attributes(self, keys=None, attr_dict=None, **kwattr):
        attr_dict = attr_dict or {}
        attr_dict.update(kwattr)
        if not keys:
            for u, v, attr in self.edges_iter(True):
                attr.update(attr_dict)
        else:
            for u, v in keys:
                self.edge[u][v].update(attr_dict)

    def get_edge_attribute(self, u, v, name, default=None):
        if u in self.edge[v]:
            return self.edge[v][u].get(name, default)
        return self.edge[u][v].get(name, default)

    def get_edge_attributes(self, u, v, names, defaults=None):
        if not defaults:
            defaults = [None] * len(names)
        if v in self.edge[u]:
            return [self.edge[u][v].get(name, default) for name, default in zip(names, defaults)]
        return [self.edge[v][u].get(name, default) for name, default in zip(names, defaults)]

    def get_edges_attribute(self, name, default=None, keys=None):
        if not keys:
            return [attr.get(name, default) for u, v, attr in self.edges_iter(True)]
        return [self.edge[u][v].get(name, default) for u, v in keys]

    def get_edges_attributes(self, names, defaults=None, keys=None):
        if not defaults:
            defaults = [None] * len(names)
        temp = zip(names, defaults)
        if not keys:
            return [[attr.get(name, default) for name, default in temp] for u, v, attr in self.edges_iter(True)]
        return [[self.edge[u][v].get(name, default) for name, default in temp] for u, v in keys]

    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # visualisation
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************
    # **************************************************************************


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    import compas
    from viewer import VolMeshViewer

    mesh = VolMesh.from_obj(compas.get_data('boxes.obj'))

    mesh.scale(0.5)

    mesh = VolMesh.from_data(mesh.to_data())

    viewer = VolMeshViewer(mesh, 600, 600, grid_on=False, zoom=5.)

    viewer.grid_on = False
    viewer.axes_on = False

    viewer.axes.x_color = (0.1, 0.1, 0.1)
    viewer.axes.y_color = (0.1, 0.1, 0.1)
    viewer.axes.z_color = (0.1, 0.1, 0.1)

    viewer.setup()
    viewer.show()

    # dual = mesh.dual()

    # viewer = NetworkViewer(dual, 600, 600)
    # viewer.grid_on = True
    # viewer.setup()
    # viewer.show()
