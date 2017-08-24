from __future__ import print_function

try:
    from Rhino.UI import MouseCallback
except ImportError:
    import platform
    if platform.python_implementation() == 'IronPython':
        raise

    class MouseCallback(object):
        pass


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, BLOCK Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = ['Mouse', ]


class Mouse(MouseCallback):
    """"""
    def __init__(self):
        super(Mouse, self).__init__()
        self.x  = None  # x-coordinate of 2D point in the viewport
        self.y  = None  # y-coordinate of 2D point in the viewport
        self.p1 = None  # start of the frustum line in world coordinates
        self.p2 = None  # end of the frustum line in world coordinates

    def OnMouseMove(self, e):
        line    = e.View.ActiveViewport.ClientToWorld(e.ViewportPoint)
        self.x  = e.ViewportPoint.X
        self.y  = e.ViewportPoint.Y
        self.p1 = line.From
        self.p2 = line.To
        e.View.Redraw()

    def OnMouseDown(self, e):
        pass

    def OnMouseUp(self, e):
        pass


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == '__main__':

    from compas.geometry import distance_point_line

    import System
    from System.Drawing import Color

    import Rhino
    from Rhino.Geometry import Point3d

    import rhinoscriptsyntax as rs

    # calculate "right angle distance" from point to frustum line
    # to check if mouse is hovering over that point
    # the distance from a point P to a line defined by points A, B
    # is twice the area of the trianlge ABP divided by the length of AB
    # see: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    class Inspector(Rhino.Display.DisplayConduit):
        """"""

        def __init__(self, points, tol=0.1):
            super(Inspector, self).__init__()
            self.mouse     = Mouse()
            self.points    = points
            self.tol       = tol
            self.dotcolor  = Color.FromArgb(255, 0, 0)
            self.textcolor = Color.FromArgb(0, 0, 0)

        def DrawForeground(self, e):
            p1  = self.mouse.p1
            p2  = self.mouse.p2
            for i, p0 in enumerate(self.points):
                if distance_point_line(p0, (p1, p2)) < self.tol:
                    e.Display.DrawDot(Point3d(*p0), str(i), self.dotcolor, self.textcolor)
                    break

    points = [[i, i, 0] for i in range(10)]

    try:
        inspector = Inspector(points)
        inspector.mouse.Enabled = True
        inspector.Enabled = True

        # this interrupts the script until the user provides a string or escapes
        rs.GetString(message='Do some hovering')

    except Exception as e:
        print(e)

    finally:
        inspector.mouse.Enabled = False
        inspector.Enabled = False
        del inspector.mouse
        del inspector
