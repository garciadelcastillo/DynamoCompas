"""
.. _compas.files:

********************************************************************************
files
********************************************************************************

.. module:: compas.files


A package for working with different types of files.


amf
===

* https://en.wikipedia.org/wiki/Additive_Manufacturing_File_Format

.. currentmodule:: compas.files.amf

:mod:`compas.files.amf`


dxf
===

* https://en.wikipedia.org/wiki/AutoCAD_DXF
* http://paulbourke.net/dataformats/dxf/
* http://paulbourke.net/dataformats/dxf/min3d.html

.. currentmodule:: compas.files.dxf

:mod:`compas.files.dxf`

.. autosummary::
    :toctree: generated/

    DXF
    DXFReader
    DXFParser
    DXFComposer
    DXFWriter


las
===

* http://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf

.. currentmodule:: compas.files.las

:mod:`compas.files.las`


obj
===

* http://paulbourke.net/dataformats/obj/

.. currentmodule:: compas.files.obj

:mod:`compas.files.obj`

.. autosummary::
    :toctree: generated/

    OBJ
    OBJReader
    OBJParser
    OBJComposer
    OBJWriter


ply
===

* http://paulbourke.net/dataformats/ply/

.. currentmodule:: compas.files.ply

:mod:`compas.files.ply`

.. autosummary::
    :toctree: generated/

    PLYreader


stl
===

* http://paulbourke.net/dataformats/stl/

.. currentmodule:: compas.files.stl

:mod:`compas.files.stl`

"""

from .amf import *
from .dxf import *
from .las import *
from .obj import *
from .ply import *
from .stl import *
