# R-tree polytopes: shape check

Two small programs that agglomerate a structured mesh with polyDEAL's own
R-tree agglomeration and write the polytopes as SVG and as VTU (ParaView).
Only deal.II and polyDEAL are used, along the path of polyDEAL's
`examples/poisson.cc`; nothing is solved.

| program | mesh |
|---|---|
| `rtree_45x45.cc` | `subdivided_hyper_cube(45)` refined globally once: 90x90 cells |
| `rstar_tree.cc` | `hyper_cube` refined globally 6 times: 64x64 cells |

Shared code in `test.h` (`agglomerate_and_write`).

First argument:

- `pack`: `pack_rtree<rstar<4>>`, i.e. Boost's bulk loading (polyDEAL's default).
  The `rstar<4>` tag only sets the node capacity here; Boost's packing is a
  top-down split at the object median along the longest box edge
  (`boost/geometry/index/detail/rtree/pack_create.hpp`, comment above `class pack`).
- `insert`: the same tree type filled by inserting cells one at a time, so the
  R* algorithm itself builds it.

The comment above `class pack` in Boost's
`boost/geometry/index/detail/rtree/pack_create.hpp`, verbatim (Boost 1.84 as
bundled with deal.II, lines 115-140):

```cpp
// STR leafs number are calculated as rcount/max
// and the number of splitting planes for each dimension as (count/max)^(1/dimension)
// <-> for dimension==2 -> sqrt(count/max)
//
// The main flaw of this algorithm is that the resulting tree will have bad structure for:
// 1. non-uniformly distributed elements
//      Statistic check could be performed, e.g. based on variance of lengths of elements edges for each dimension
// 2. elements distributed mainly along one axis
//      Calculate bounding box of all elements and then number of dividing planes for a dimension
//      from the length of BB edge for this dimension (more or less assuming that elements are uniformly-distributed squares)
//
// Another thing is that the last node may have less elements than Max or even Min.
// The number of splitting planes must be chosen more carefully than count/max
//
// This algorithm is something between STR and TGS
// it is more similar to the top-down recursive kd-tree creation algorithm
// using object median split and split axis of greatest BB edge
// BB is only used as a hint (assuming objects are distributed uniformly)
//
// Implemented algorithm guarantees that the number of elements in nodes will be between Min and Max
// and that nodes are packed as tightly as possible
// e.g. for 177 values Max = 5 and Min = 2 it will construct the following tree:
// ROOT                 177
// L1          125               52
// L2  25  25  25  25  25   25  17    10
// L3  5x5 5x5 5x5 5x5 5x5  5x5 3x5+2 2x5
```

Optional second argument: the tree level to extract (default: the level whose
mean polytope is closest to 64 cells).


## Build and run

    cp test.h rtree_45x45.cc rstar_tree.cc Polydeal/examples/
    mkdir Polydeal/build && cd Polydeal/build
    cmake -DDEAL_II_DIR=/path/to/deal.II ..
    make rtree_45x45 rstar_tree     
    programs

## Output (`out/<mesh>/`)

- `pack.svg`, `insert.svg`: deal.II `GridOut::write_svg`, cells coloured by polytope.
- `pack.vtu`, `insert.vtu`: open in ParaView and colour by `polytope_colour`.