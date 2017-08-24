the network does not allow the addition of faces
the base graph provides the infrastructure for storing face data (...)
however, its add_face functionality raises a NotImplementedError
the network does not overwrite this
therefore it leaves the inherited face infrastructure empty
if a network has to provide support for faces
for example, in the case of compas_tna (form and force diagrams)
a deriving class should implement an add_face function
that only adds halfedges
if the network has a corresponding edge
this ensures that vertex_neighbours can return the expected/correct result
based on the halfedge dict
the faces of this custom network can then be found using the find_network_faces function
if the network is planar
and if it is embedded as such in the plane

a mesh does not allow the explicit addition of edges
there is a virtual edge for every pair of halfedges
the direction of the edge depends on the (random) ordering of the halfedge dict
this prevents additional housekeeping during the many mesh algorithms
by having to keep track of the edges
while still providing the possibility of storing and retrieving edge data
the first time an edge is accessed it is created in the edge dict
if the requested edge has corresponding halfedges
