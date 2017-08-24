This is a markdown version of the text available [here](http://www.fabbers.com/tech/STL_Format)


# The STL file format

## Format Specifications

An StL file consists of a list of facet data. Each facet is uniquely identified
by a unit normal (a line perpendicular to the triangle and with a length of 1.0)
and by three vertices (corners). The normal and each vertex are specified by
three coordinates each, so there is a total of 12 numbers stored for each facet.

**Facet orientation**. The facets define the surface of a 3-dimensional object. As
such, each facet is part of the boundary between the interior and the exterior
of the object. The orientation of the facets (which way is “out” and which way
is “in”) is specified redundantly in two ways which must be consistent. First,
the direction of the normal is outward. Second, the vertices are listed in
counterclockwise order when looking at the object from the outside (right-hand
rule). These rules are illustrated in Figure 1.

**Vertex-to-vertex rule**. Each triangle must share two vertices with each of its
adjacent triangles. In other words, a vertex of one triangle cannot lie on the
side of another. This is illustrated in Figure 2.

The object represented must be located in the all-positive octant. In other
words, all vertex coordinates must be positive-definite (nonnegative and
nonzero) numbers. The StL file does not contain any scale information; the
coordinates are in arbitrary units.

The official 3D Systems StL specification document states that there is a
provision for inclusion of “special attributes for building parameters,” but
does not give the format for including such attributes. Also, the document
specifies data for the “minimum length of triangle side” and “maximum triangle
size,” but these numbers are of dubious meaning.

Sorting the triangles in ascending z-value order is recommended, but not
required, in order to optimize performance of the slice program.

Typically, an StL file is saved with the extension “StL,” case-insensitive. The
slice program may require this extension or it may allow a different extension
to be specified.

The StL standard includes two data formats, ASCII and binary. These are
described separately below.

## STL ASCII Format

The ASCII format is primarily intended for testing new CAD interfaces. The large
size of its files makes it impractical for general use.

The syntax for an ASCII StL file is as follows:

```
solid name
    facet normal ni nj nk
        outer loop
            vertex v1x v1y v1z
            vertex v2x v2y v2z
            vertex v3x v3y v3z
        endloop
    endfacet
endsolid name
```

Bold face indicates a keyword; these must appear in lower case. Note that there
is a space in “facet normal” and in “outer loop,” while there is no space in any
of the keywords beginning with “end.” Indentation must be with spaces; tabs are
not allowed. The notation, “{…}+,” means that the contents of the brace brackets
can be repeated one or more times. Symbols in italics are variables which are to
be replaced with user-specified values. The numerical data in the facet normal
and vertex lines are single precision floats, for example, 1.23456E+789. A facet
normal coordinate may have a leading minus sign; a vertex coordinate may not.
