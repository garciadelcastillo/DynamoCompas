""""""

import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from compas.visualization.plotters.core.drawing import create_axes_2d

from compas.visualization.plotters.core.drawing import draw_xpoints_2d
from compas.visualization.plotters.core.drawing import draw_xlines_2d
from compas.visualization.plotters.core.drawing import draw_xpolygons_2d
from compas.visualization.plotters.core.drawing import draw_xarrows_2d


__author__    = ['Tom Van Mele', ]
__copyright__ = 'Copyright 2016 - Block Research Group, ETH Zurich'
__license__   = 'MIT License'
__email__     = 'vanmelet@ethz.ch'


__all__ = []


# https://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode
# https://matplotlib.org/api/pyplot_summary.html
# https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
# https://matplotlib.org/api/axes_api.html
# https://matplotlib.org/api/index.html


class Plotter(object):

    def __init__(self):
        self._interactive = False
        self._axes = None
        # use descriptors for these
        # to help the user set these attributes in the right format
        # figure attributes
        self.figure_size = (10, 7)
        self.figure_bgcolor = '#ffffff'
        # axes attributes
        self.axes_xlabel = None
        self.axes_ylabel = None
        # drawing defaults
        # z-order
        # color
        # size/thickness
        self.default_point_edgecolor = '#000000'
        self.default_point_facecolor = '#ffffff'
        self.default_line_color = '#000000'
        self.default_polygon_edgecolor = '#000000'
        self.default_polygon_facecolor = '#ffffff'

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, value):
        self._interactive = value
        # interactive mode seems to be intended for other things
        # see: https://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode
        # if value:
        #     plt.ion()
        # else:
        #     plt.ioff()

    @property
    def axes(self):
        if self._axes is None:
            # customise the use of this function
            # using attributes of the plotter class
            self._axes = create_axes_2d()
        return self._axes

    @property
    def figure(self):
        return self.axes.get_figure()

    @property
    def bgcolor(self):
        return self.figure.get_facecolor()

    @bgcolor.setter
    def bgcolor(self, value):
        self.figure.set_facecolor(value)

    @property
    def title(self):
        return self.figure.canvas.get_window_title()

    @title.setter
    def title(self, value):
        self.figure.canvas.set_window_title(value)

    def show(self):
        self.axes.autoscale()
        plt.show()

    def save(self, filepath, **kwargs):
        self.axes.autoscale()
        plt.savefig(filepath, **kwargs)

    def update(self, pause=0.0001):
        self.axes.autoscale()
        self.figure.canvas.draw()
        plt.pause(pause)

    def update_pointcollection(self, collection, centers, radius=1.0):
        try:
            len(radius)
        except Exception:
            radius = [radius] * len(centers)
        data = zip(centers, radius)
        circles = [Circle(center, radius) for center, radius in data]
        collection.set_paths(circles)

    def update_linecollection(self, collection, segments):
        pass

    def update_polygoncollection(self, collection, polygons):
        pass

    def draw_points(self, points):
        raise NotImplementedError

    def draw_lines(self, lines):
        raise NotImplementedError

    def draw_polygons(self, polygons):
        raise NotImplementedError

    def draw_xpoints(self, points):
        return draw_xpoints_2d(points, self.axes)

    def draw_xlines(self, lines):
        return draw_xlines_2d(lines, self.axes)

    def draw_xpolygons(self, polygons):
        return draw_xpolygons_2d(polygons, self.axes)

    def draw_xarrows(self, arrows):
        return draw_xarrows_2d(arrows, self.axes)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    import compas

    from compas.datastructures.mesh import Mesh
    from compas.datastructures.mesh.algorithms import smooth_mesh_centroid

    # make a mesh

    mesh = Mesh.from_obj(compas.get_data('faces.obj'))

    # make lists of plotting geometry

    points = []
    for key in mesh.vertices():
        points.append({
            'pos': mesh.vertex_coordinates(key, 'xy'),
            'radius': 0.1,
            'facecolor': '#ff0000' if mesh.vertex_degree(key) == 2 else '#ffffff'
        })

    lines = []
    for u, v in mesh.edges():
        lines.append({
            'start': mesh.vertex_coordinates(u, 'xy'),
            'end': mesh.vertex_coordinates(v, 'xy'),
            'width': 1.0
        })

    polygons = []
    for fkey in mesh.faces():
        polygons.append({
            'points': mesh.face_coordinates(fkey, 'xy'),
            'edgecolor': '#ff0000',
            'linewidth': 3.0
        })

    fixed = [key for key in mesh.vertices() if mesh.vertex_degree(key) == 2]

    # make a plotter
    # provide customisation options here

    plotter = Plotter()

    # extract collections

    points = plotter.draw_xpoints(points)
    lines = plotter.draw_xlines(lines)

    # a callback function for live updates

    def callback(mesh, k, args):
        plotter, (points, lines), pause = args

        plotter.update_pointcollection(points, [mesh.vertex_coordinates(key, 'xy') for key in mesh.vertex], 0.1)

        segments = []
        for u, v in mesh.edges():
            segments.append([mesh.vertex_coordinates(u, 'xy'), mesh.vertex_coordinates(v, 'xy')])

        lines.set_segments(segments)

        plotter.update(pause=pause)

    # live visualisation of smoothing

    smooth_mesh_centroid(mesh,
                         fixed=fixed,
                         kmax=100,
                         callback=callback,
                         callback_args=(plotter, (points, lines), 0.01))

    plotter.show()
