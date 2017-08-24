""".. _compas.numerical:

********************************************************************************
numerical
********************************************************************************

.. module:: compas.numerical


A package for numerical computation.


.. rubric:: Submodules

.. toctree::
    :maxdepth: 1

    compas.numerical.euler
    compas.numerical.gpu
    compas.numerical.methods
    compas.numerical.solvers


geometry
========

.. currentmodule:: compas.numerical.geometry

:mod:`compas.numerical.geometry`

.. autosummary::
    :toctree: generated/

    scalarfield_contours
    plot_scalarfield_contours


linalg
======

.. currentmodule:: compas.numerical.linalg

:mod:`compas.numerical.linalg`

.. autosummary::
    :toctree: generated/

    nullspace
    rank
    dof
    pivots
    nonpivots
    rref
    chofactor
    lufactorized
    normrow
    normalizerow
    rot90
    solve_with_known
    spsolve_with_known


matrices
========

.. currentmodule:: compas.numerical.matrices

:mod:`compas.numerical.matrices`

.. autosummary::
    :toctree: generated/

    adjacency_matrix
    degree_matrix
    connectivity_matrix
    laplacian_matrix
    face_matrix
    mass_matrix
    stiffness_matrix
    equilibrium_matrix


operators
=========

.. currentmodule:: compas.numerical.operators

:mod:`compas.numerical.operators`

.. autosummary::
    :toctree: generated/

    grad
    div
    curl


spatial
=======

.. currentmodule:: compas.numerical.spatial

:mod:`compas.numerical.spatial`

.. autosummary::
    :toctree: generated/

    closest_points_points
    project_points_heightfield
    iterative_closest_point
    bounding_box_2d
    bounding_box_3d


statistics
==========

.. currentmodule:: compas.numerical.statistics

:mod:`compas.numerical.statistics`

.. autosummary::
    :toctree: generated/

    principal_components


utilities
=========

.. currentmodule:: compas.numerical.utilities

:mod:`compas.numerical.utilities`

.. autosummary::
    :toctree: generated/

    set_array_print_precision
    unset_array_print_precision


xforms
======

.. currentmodule:: compas.numerical.xforms

:mod:`compas.numerical.xforms`

.. autosummary::
    :toctree: generated/

    translation_matrix
    rotation_matrix
    random_rotation_matrix
    scale_matrix
    projection_matrix

"""
