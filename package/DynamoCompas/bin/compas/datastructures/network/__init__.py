"""
.. _compas.datastructures.network:

********************************************************************************
network
********************************************************************************

.. module:: compas.datastructures.network


Package for working with network objects.


.. autosummary::
    :toctree: generated/

    Network

.. autosummary::
    :toctree: generated/

    viewer.NetworkViewer


operations
==========

.. currentmodule:: compas.datastructures.network.operations

:mod:`compas.datastructures.network.operations`

.. autosummary::
    :toctree: generated/

    split_edge_network


algorithms
==========

.. currentmodule:: compas.datastructures.network.algorithms

:mod:`compas.datastructures.network.algorithms`

.. autosummary::
    :toctree: generated/

    network_vertex_coloring
    construct_dual_network
    find_network_faces
    is_network_crossed
    are_network_edges_crossed
    count_network_crossings
    is_network_2d
    is_network_planar
    is_network_planar_embedding
    embed_network_in_plane
    smooth_network_mixed
    smooth_network_centroid
    smooth_network_area
    smooth_network_mass
    smooth_network_length
    network_dfs
    network_bfs
    network_dfs_paths
    network_bfs_paths
    network_shortest_path
    network_dijkstra_distances
    network_dijkstra_path


numerical
=========

.. currentmodule:: compas.datastructures.network.numerical

:mod:`compas.datastructures.network.numerical`

.. autosummary::
    :toctree: generated/

    network_adjacency_matrix
    network_degree_matrix
    network_connectivity_matrix
    network_laplacian_matrix
    network_face_matrix

"""

from .network import Network
