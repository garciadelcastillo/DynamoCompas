"""
.. _compas:

********************************************************************************
compas
********************************************************************************

.. module:: compas


This is the main package of the core framework. It defines the base functionality
used by all other packages and can be used entirely standalone.


.. rubric:: Submodules

.. toctree::
    :maxdepth: 1

    compas.com
    compas.datastructures
    compas.files
    compas.geometry
    compas.numerical
    compas.plotters
    compas.utilities
    compas.viewers

"""

import os
import sys

PY3 = sys.version_info.major == 3

HERE = os.path.dirname(__file__)

HOME = os.path.abspath(os.path.join(HERE, '../../'))
DATA = os.path.abspath(os.path.join(HOME, 'data'))
DOCS = os.path.abspath(os.path.join(HOME, 'docs'))
LIBS = os.path.abspath(os.path.join(HOME, 'libs'))
TEMP = os.path.abspath(os.path.join(HOME, '__temp'))


def _find_resource(filename):
    filename = filename.strip('/')
    return os.path.abspath(os.path.join(DATA, filename))


def get_data(filename):
    return _find_resource(filename)


def get_license():
    with open(os.path.join(HOME, 'LICENSE')) as fp:
        return fp.read()


def get_requirements(rtype='list'):
    pass


__all__ = []
