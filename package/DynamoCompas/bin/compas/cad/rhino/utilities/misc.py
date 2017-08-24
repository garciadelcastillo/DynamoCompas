import os
import ast
import inspect

from compas.cad.rhino.forms import TextForm
from compas.cad.rhino.forms import ImageForm

try:
    import System
    import Rhino
    import rhinoscriptsyntax as rs
    from Rhino.UI.Dialogs import ShowPropertyListBox
    from Rhino.UI.Dialogs import ShowMessageBox

except ImportError:
    import platform
    if platform.python_implementation() == 'IronPython':
        raise


__author__     = ['Tom Van Mele', ]
__copyright__  = 'Copyright 2014, BLOCK Research Group - ETH Zurich'
__license__    = 'MIT License'
__email__      = 'vanmelet@ethz.ch'


__all__ = [
    'wait', 'get_tolerance', 'toggle_toolbargroup', 'pick_point',
    'browse_for_folder', 'browse_for_file',
    'print_display_on',
    'display_message', 'display_text', 'display_image', 'display_html',
    'update_settings', 'update_attributes', 'update_named_values',
    'screenshot_current_view',
]


# ==============================================================================
# Truly miscellaneous :)
# ==============================================================================


def screenshot_current_view(path,
                            width=1920,
                            height=1080,
                            scale=1,
                            draw_grid=False,
                            draw_world_axes=False,
                            draw_cplane_axes=False,
                            background=False):
    properties = [draw_grid, draw_world_axes, draw_cplane_axes, background]
    properties = ["Yes" if item else "No" for item in properties]
    scale = max(1, scale)  # the rhino command requires a scale > 1
    rs.EnableRedraw(True)
    rs.Sleep(0)
    result = rs.Command("-_ViewCaptureToFile \"" + path + "\""
                        " Width=" + str(width) +
                        " Height=" + str(height) +
                        " Scale=" + str(scale) +
                        " DrawGrid=" + properties[0] +
                        " DrawWorldAxes=" + properties[1] +
                        " DrawCPlaneAxes=" + properties[2] +
                        " TransparentBackground=" + properties[3] +
                        " _enter", False)
    rs.EnableRedraw(False)
    return result


def add_gui_helpers(helpers, overwrite=False, protected=False):
    def decorate(cls):
        # attr = {}
        for helper in helpers:
            # for name, value in helper.__dict__.items():
            for name, value in inspect.getmembers(helper):
                # magic methods
                if name.startswith('__') and name.endswith('__'):
                    continue
                # protected / private methods
                if not protected and name.startswith('_'):
                    continue
                # existing methods
                if not overwrite:
                    if hasattr(cls, name):
                        continue
                # attr[name] = value
                # try:
                #     setattr(cls, name, value.__func__)
                # except:
                #     setattr(cls, name, value)
                # inspect.ismethoddescriptor
                # inspect.isdatadescriptor
                if inspect.ismethod(value):
                    setattr(cls, name, value.__func__)
                else:
                    setattr(cls, name, value)
        # cls = type(cls.__name__, (cls, ), attr)
        return cls
    return decorate


def wait():
    return Rhino.RhinoApp.Wait()


def get_tolerance():
    return rs.UnitAbsoluteTolerance()


def toggle_toolbargroup(rui, group):
    if not os.path.exists(rui) or not os.path.isfile(rui):
        return
    collection = rs.IsToolbarCollection(rui)
    if not collection:
        collection = rs.OpenToolbarCollection(rui)
        if rs.IsToolbar(collection, group, True):
            rs.ShowToolbar(collection, group)
    else:
        if rs.IsToolbar(collection, group, True):
            if rs.IsToolbarVisible(collection, group):
                rs.HideToolbar(collection, group)
            else:
                rs.ShowToolbar(collection, group)


# pick a location
# get_location
def pick_point(message='Pick a point.'):
    point = rs.GetPoint(message)
    if point:
        return list(point)
    return None


# ==============================================================================
# File system
# ==============================================================================


def browse_for_folder(message=None, default=None):
    return rs.BrowseForFolder(folder=default, message=message, title='compas')


def browse_for_file(title=None, folder=None, filter=None):
    return rs.OpenFileName(title, filter=filter, folder=folder)


# ==============================================================================
# Display
# ==============================================================================


def print_display_on(on=True):
    if on:
        rs.Command('_PrintDisplay State On Color Display Thickness 1 _Enter')
    else:
        rs.Command('_PrintDisplay State Off _Enter')


def display_message(message):
    return ShowMessageBox(message, 'Message')


def display_text(text, title='Text', width=800, height=600):
    if isinstance(text, (list, tuple)):
        text = '{0}'.format(System.Environment.NewLine).join(text)
    form = TextForm(text, title, width, height)
    return form.show()


def display_image(image, title='Image', width=800, height=600):
    form = ImageForm(image, title, width, height)
    return form.show()


def display_html():
    raise NotImplementedError


# ==============================================================================
# Settings and attributes
# ==============================================================================


def update_settings(settings, message='', title='Update settings'):
    names  = sorted(settings.keys())
    values = [str(settings[name]) for name in names]
    values = ShowPropertyListBox(message, title, names, values)
    if values:
        for name, value in list(zip(names, values)):
            try:
                settings[name] = ast.literal_eval(value)
            except (TypeError, ValueError):
                settings[name] = value
        return True
    return False


def update_attributes(names, values, message='', title='Update attributes'):
    return ShowPropertyListBox(message, title, names, values)


def update_named_values(names, values, message='', title='Update named values'):
    return ShowPropertyListBox(message, title, names, values)


# ==============================================================================
# Debugging
# ==============================================================================

if __name__ == "__main__":

    pass
