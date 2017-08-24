import os
import compas

from compas.visualization.viewers.viewer import Viewer

from compas.visualization.viewers.core.drawing import xdraw_polygons
from compas.visualization.viewers.core.drawing import xdraw_lines
from compas.visualization.viewers.core.drawing import xdraw_points


__author__     = 'Tom Van Mele'
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT'
__email__      = 'vanmelet@ethz.ch'


class MeshViewer(Viewer):
    """"""

    def __init__(self, mesh, width=1440, height=900):
        super(MeshViewer, self).__init__(width=width, height=height)
        self.mesh = mesh

    # change this to a more flexible system
    # that provides similar possibilities as the network plotter
    def display(self):
        polygons = []
        for fkey in self.mesh.faces():
            points = self.mesh.face_coordinates(fkey)
            color_front = self.mesh.get_face_attribute(fkey, 'color', (0.8, 0.8, 0.8, 1.0))
            color_back  = (0.2, 0.2, 0.2, 1.0)
            polygons.append({'points': points,
                             'color.front': color_front,
                             'color.back' : color_back})

        lines = []
        for u, v in self.mesh.edges():
            lines.append({'start': self.mesh.vertex_coordinates(u),
                          'end'  : self.mesh.vertex_coordinates(v),
                          'color': (0.1, 0.1, 0.1),
                          'width': 1.})

        points = []
        for key in self.mesh.vertices():
            points.append({'pos'   : self.mesh.vertex_coordinates(key),
                           'color' : (0.4, 0.4, 0.4),
                           'size'  : 5.0})

        xdraw_polygons(polygons)
        xdraw_lines(lines)
        xdraw_points(points)

    def keypress(self, key, x, y):
        """
        Assign mesh functionality to keys.

        The following keys have a mesh function assigned to them:
            * u: unify cycle directions
            * f: flip cycle directions
            * s: subdivide using quad subdivision
        """
        if key == 'u':
            self.mesh.unify_cycles()
            return
        if key == 'f':
            self.mesh.flip_cycles()
            return
        if key == 's':
            self.mesh.subdivide('quad')
            return
        if key == 'c':
            self.screenshot(os.path.join(compas.TEMP, 'screenshot.jpg'))
            return

    def special(self, key, x, y):
        """
        Assign mesh functionality to function keys.
        """
        pass


class SubdMeshViewer(Viewer):
    """Viewer for subdivision meshes.

    Parameters:
        mesh (compas.datastructures.mesh.Mesh): The *control* mesh object.
        subdfunc (callable): The subdivision algorithm/scheme.
        width (int): Optional. Width of the viewport. Default is ``1440``.
        height (int): Optional. Height of the viewport. Default is ``900``.

    Warning:
        Not properly tested on meshes with a boundary.

    Example:

        .. code-block:: python

            from compas.datastructures.mesh.mesh import Mesh
            from compas.datastructures.mesh.algorithms import subdivide_mesh_doosabin
            from compas.datastructures.mesh.viewer import SubdMeshViewer

            from compas.geometry.elements.polyhedron import Polyhedron

            poly = Polyhedron.generate(6)

            mesh = Mesh.from_vertices_and_faces(poly.vertices, poly.faces)

            viewer = SubdMeshViewer(mesh, subdfunc=subdivide_mesh_doosabin, width=600, height=600)

            viewer.axes_on = False
            viewer.grid_on = False

            for i in range(10):
                viewer.camera.zoom_in()

            viewer.setup()
            viewer.show()

    """

    def __init__(self, mesh, subdfunc, width=1440, height=900):
        super(SubdMeshViewer, self).__init__(width=width, height=height)
        self.mesh = mesh
        self.subdfunc = subdfunc
        self.subd = None

    def display(self):
        xyz = {key: self.mesh.vertex_coordinates(key) for key in self.mesh.vertices()}

        lines = []
        for u, v in self.mesh.wireframe():
            lines.append({'start' : xyz[u],
                          'end'   : xyz[v],
                          'color' : (0.1, 0.1, 0.1),
                          'width' : 1.})

        points = []
        for key in self.mesh.vertices():
            points.append({'pos'   : xyz[key],
                           'color' : (0.0, 1.0, 0.0),
                           'size'  : 10.0})

        xdraw_lines(lines)
        xdraw_points(points)

        if self.subd:
            xyz   = {key: self.subd.vertex_coordinates(key) for key in self.subd.vertices()}
            front = (0.7, 0.7, 0.7, 1.0)
            back  = (0.2, 0.2, 0.2, 1.0)

            poly  = []
            for fkey in self.subd.faces():
                poly.append({'points': self.subd.face_coordinates(fkey),
                             'color.front': front,
                             'color.back' : back})

            lines = []
            for u, v in self.subd.wireframe():
                lines.append({'start': xyz[u],
                              'end'  : xyz[v],
                              'color': (0.1, 0.1, 0.1),
                              'width': 1.})

            xdraw_polygons(poly)
            xdraw_lines(lines)

    def keypress(self, key, x, y):
        if key == '1':
            self.subd = self.subdfunc(self.mesh, k=1)
        if key == '2':
            self.subd = self.subdfunc(self.mesh, k=2)
        if key == '3':
            self.subd = self.subdfunc(self.mesh, k=3)
        if key == '4':
            self.subd = self.subdfunc(self.mesh, k=4)
        if key == '5':
            self.subd = self.subdfunc(self.mesh, k=5)
        if key == 'c':
            self.screenshot(os.path.join(compas.TEMP, 'screenshot.jpg'))

    def subdivide(self, k=1):
        self.subd = self.subdfunc(self.mesh, k=k)


class MultiMeshViewer(Viewer):
    """"""

    def __init__(self, meshes, colors, width=1440, height=900):
        super(MultiMeshViewer, self).__init__(width=width, height=height)
        self.meshes = meshes
        self.colors = colors

    def display(self):
        for i in range(len(self.meshes)):
            mesh = self.meshes[i]

            polygons = []
            for fkey in mesh.faces():
                color_front = self.colors[i]
                color_back  = (0.2, 0.2, 0.2, 1.0)
                polygons.append({'points': mesh.face_coordinates(fkey),
                                 'color.front': color_front,
                                 'color.back': color_back})

            lines = []
            for u, v in mesh.wireframe():
                lines.append({'start': mesh.vertex_coordinates(u),
                              'end': mesh.vertex_coordinates(v),
                              'color': (0.1, 0.1, 0.1),
                              'width': 1.})

            xdraw_polygons(polygons)
            xdraw_lines(lines)

    def keypress(self, key, x, y):
        pass


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    from compas.datastructures.mesh.mesh import Mesh
    from compas.datastructures.mesh.algorithms import subdivide_mesh_doosabin

    from compas.geometry.elements.polyhedron import Polyhedron

    poly = Polyhedron.generate(6)

    mesh = Mesh.from_vertices_and_faces(poly.vertices, poly.faces)

    viewer = SubdMeshViewer(mesh, subdfunc=subdivide_mesh_doosabin, width=600, height=600)

    viewer.axes_on = False
    viewer.grid_on = False

    for i in range(10):
        viewer.camera.zoom_in()

    viewer.setup()
    viewer.show()
