from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from compas.geometry import centroid_points

from compas.visualization.viewers.viewer import Viewer

from compas.visualization.viewers.core.drawing import xdraw_points
from compas.visualization.viewers.core.drawing import xdraw_lines

from compas.utilities import color_to_colordict
from compas.utilities import color_to_rgb


__author__     = 'Tom Van Mele'
__copyright__  = 'Copyright 2014, Block Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = '<vanmelet@ethz.ch>'


class NetworkViewer(Viewer):
    """An OpenGL viewer for networks.

    Parameters:
        network (compas.datastructures.network.Network): The network object.
        width (int): Optional. The viewport width. Default is ``1280``.
        height (int): Optional. The viewport height. Default is ``800``.

    Example:

        .. code-block:: python

            import compas
            from compas.datastructures.network import Network
            from compas.datastructures.network.viewer import NetworkViewer

            network = Network.from_obj(compas.get_data('lines.obj'))

            network.add_edge(0, 14)
            network.add_edge(15, 10)
            network.add_edge(21, 24)

            viewer = NetworkViewer(network, 600, 600)

            viewer.grid_on = False

            viewer.setup()
            viewer.show()

    """

    def __init__(self, network, width=1280, height=800):
        super(NetworkViewer, self).__init__(width=width, height=height)
        self.default_vertexcolor = (0, 0, 0)
        self.default_edgecolor   = (0, 0, 0)
        self.default_facecolor   = (0, 0, 0)
        self.vertices_on = True
        self.edges_on    = True
        self.faces_on    = False
        self.vertexcolor = None
        self.edgecolor   = None
        self.facecolor   = None
        self.vertexlabel = None
        self.edgelabel   = None
        self.facelabel   = None
        self.vertexsize  = None
        self.edgewidth   = None
        self.network = network
        self.center()

    # --------------------------------------------------------------------------
    # helpers (temp)
    # --------------------------------------------------------------------------

    def center(self):
        xyz = [self.network.vertex_coordinates(key) for key in self.network.vertices()]
        cx, cy, cz = centroid_points(xyz)
        for key, attr in self.network.vertices(True):
            attr['x'] -= cx
            attr['y'] -= cy

    # --------------------------------------------------------------------------
    # main drawing functionality
    # --------------------------------------------------------------------------

    def display(self):
        points = []
        vcolor = self.network.attributes['color.vertex']
        vcolor = vcolor or self.default_vertexcolor
        vcolor = color_to_rgb(vcolor, True)
        for key, attr in self.network.vertices(True):
            points.append({
                'pos'  : (attr['x'], attr['y'], attr['z']),
                'size' : 6.,
                'color': vcolor,
            })
        lines = []
        ecolor = self.network.attributes['color.vertex']
        ecolor = ecolor or self.default_edgecolor
        ecolor = color_to_rgb(ecolor, True)
        for u, v in self.network.edges():
            lines.append({
                'start': self.network.vertex_coordinates(u),
                'end'  : self.network.vertex_coordinates(v),
                'color': ecolor,
                'width': 1.
            })
        xdraw_points(points)
        xdraw_lines(lines)

    # --------------------------------------------------------------------------
    # keyboard functionality
    # --------------------------------------------------------------------------

    def keypress(self, key, x, y):
        """Assign network functionality to keys.
        """
        if key == 'c':
            self.screenshot(os.path.join(compas.TEMP, 'screenshots/network-viewer_screenshot.jpg'))

    def special(self, key, x, y):
        """Define the meaning of pressing function keys.
        """
        pass


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    import compas
    from compas.datastructures.network import Network

    network = Network.from_obj(compas.get_data('saddle.obj'))

    viewer = NetworkViewer(network, 600, 600)

    viewer.setup()
    viewer.show()
