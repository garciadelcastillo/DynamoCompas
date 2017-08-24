from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from compas.viewers.viewer import Viewer
from compas.viewers.drawing import xdraw_polygons
from compas.viewers.drawing import xdraw_lines
from compas.viewers.drawing import xdraw_points


__author__     = 'Tom Van Mele'
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = '<vanmelet@ethz.ch>'


class VolMeshViewer(Viewer):
    """"""

    def __init__(self, volmesh, width=1440, height=900, **kwargs):
        super(VolMeshViewer, self).__init__(width=width, height=height)
        self.volmesh   = volmesh
        self.num_cells = len(self.volmesh.cells())

    def display(self):
        polygons = []
        for ckey in self.volmesh.cells_iter():
            for fkey in self.volmesh.cell_halffaces(ckey):
                vkeys       = self.volmesh.halfface_vertices(fkey, ordered=True)
                points      = [self.volmesh.vertex_coordinates(vkey) for vkey in vkeys]
                color_front = (0.7, 0.7, 0.7, 1.0)
                color_back  = (0.0, 0.0, 0.0, 1.0)
                polygons.append({
                    'points'      : points,
                    'color.front' : color_front,
                    'color.back'  : color_back
                })
        lines = []
        for u, v in self.volmesh.edges_iter():
            lines.append({
                'start': self.volmesh.vertex_coordinates(u),
                'end'  : self.volmesh.vertex_coordinates(v),
                'color': (0.1, 0.1, 0.1),
                'width': 3.
            })
        points = []
        for u in self.volmesh.vertices_iter():
            points.append({
                'pos'   : self.volmesh.vertex_coordinates(u),
                'size'  : 10.,
                'color' : (0.0, 1.0, 0.0),
            })
        xdraw_polygons(polygons)
        xdraw_lines(lines)
        xdraw_points(points)

    def keypress(self, key, x, y):
        """
        Assign volmesh functionality to keys.

        The following keys have a mesh function assigned to them:
            * u: unify cycle directions
            * f: flip cycle directions
            * s: subdivide using quad subdivision
        """
        if key == 'u':
            self.volmesh.unify_cycle_directions()
            return
        if key == 'f':
            self.volmesh.flip_cycle_directions()
            return

    def special(self, key, x, y):
        """
        Assign volmesh functionality to function keys.
        """
        pass


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    import compas
    from compas.datastructures.volmesh.volmesh import VolMesh

    mesh = VolMesh.from_json(compas.get_data('boxes.json'))

    viewer = VolMeshViewer(mesh)
    viewer.setup()
    viewer.show()
