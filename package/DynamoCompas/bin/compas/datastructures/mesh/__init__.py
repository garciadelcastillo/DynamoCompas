"""
.. _compas.datastructures.mesh:

********************************************************************************
mesh
********************************************************************************

.. module:: compas.datastructures.mesh


Package for working with mesh objects.


.. autosummary::
    :toctree: generated/

    Mesh

.. autosummary::
    :toctree: generated/

    viewer.MeshViewer
    viewer.SubdMeshViewer


operations
==========

.. currentmodule:: compas.datastructures.mesh.operations

:mod:`compas.datastructures.mesh.operations`

.. autosummary::
    :toctree: generated/

    collapse_edge_mesh
    collapse_edge_trimesh
    insert_edge_mesh
    split_edge_mesh
    split_edge_trimesh
    split_face_mesh
    swap_edge_trimesh
    unweld_vertices_mesh


algorithms
==========

.. currentmodule:: compas.datastructures.mesh.algorithms

:mod:`compas.datastructures.mesh.algorithms`

.. autosummary::
    :toctree: generated/

    construct_dual_mesh
    planarize_mesh
    circularize_mesh
    unify_cycles_mesh
    flip_cycles_mesh
    smooth_mesh_centroid
    smooth_mesh_centerofmass
    smooth_mesh_length
    smooth_mesh_area
    smooth_mesh_angle
    subdivide_mesh
    subdivide_mesh_tri
    subdivide_mesh_catmullclark
    subdivide_mesh_doosabin
    subdivide_trimesh_loop
    delaunay_from_points
    voronoi_from_points
    optimise_trimesh_topology


numerical
=========

.. currentmodule:: compas.datastructures.mesh.numerical

:mod:`compas.datastructures.mesh.numerical`

.. autosummary::
    :toctree: generated/

    mesh_adjacency_matrix
    mesh_connectivity_matrix
    mesh_laplacian_matrix
    trimesh_edge_cotangent
    trimesh_edge_cotangents
    trimesh_cotangent_laplacian_matrix
    trimesh_positive_cotangent_laplacian_matrix
    trimesh_descent
    mesh_contours
    mesh_isolines
    plot_mesh_contours
    plot_mesh_isolines
    delaunay_from_mesh
    delaunay_from_points
    delaunay_from_boundary

"""

from .mesh import Mesh

