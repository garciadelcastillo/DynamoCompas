"""
.. _compas.viewers:

********************************************************************************
viewers
********************************************************************************

.. module:: compas.viewers


Standalone (OpenGL) viewers for visualisation outside of CAD environments.


.. autosummary::
    :toctree: generated/

    Viewer
    App


drawing
=======

.. currentmodule:: compas.viewers.drawing

:mod:`compas.viewers.drawing`

.. autosummary::
    :toctree: generated/

    draw_points
    draw_lines
    draw_faces
    draw_sphere
    xdraw_points
    xdraw_lines
    xdraw_polygons


helpers
=======

.. currentmodule:: compas.viewers.helpers

:mod:`compas.viewers.helpers`

.. autosummary::
    :toctree: generated/

    Axes
    Camera
    Grid
    Mouse


widgets
=======

.. currentmodule:: compas.viewers.widgets

:mod:`compas.viewers.widgets`

.. autosummary::
    :toctree: generated/

    Browser
    GLView

"""

from .viewer import Viewer
from .app import App
