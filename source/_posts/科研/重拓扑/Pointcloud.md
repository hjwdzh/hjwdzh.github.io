---
title: Retopologize the point cloud
toc: true
categories: [科研,重拓扑]
date: 2023-12-21 10:00:00
tags: [科研,重拓扑]
---

## Background

What if we want to build a nice topology around a pointcloud similar to [**mesh retopology**](/2023/12/20/科研/重拓扑/Review/)? Let's think about it.

## Method

### Tangent field
- **Direction field.** We can surely estimate the 4-rosy field for each vertex based on a KNN graph.
- **Singularity vertex.** For each point and it's tangent plane, we project points, locally triangulate, and count the rosys (if not 4, it's a singularity).
- **Singularity region.** We can possibly merge singular vertices.

### Patch Segmentation
**Path tracing.**
Path tracing is do-able. We build a motor-cycle graph (vertices-direction connection), and dijkstra algorithm to compute loop for any sampled vertex-direction pair.

**Rounds**
Once we have the path tracing kernel, the main algorithm can work including the recursive patch segmentation, loop selection. However, several problem exists.
- How shall we partition points into patches given paths?
- How shall we verify the validity/quality of a patch as a point cloud?

## Core algorithm

### KNN Split
The core challenge for partitioning the patch is that non-manifold KNN topology. Once we have the path, we need to split the graph seriously. For each graph edge, we make a normal plane with certain thickness, and any graph edge passing the plane should be removed. After that, we are able to partition the graph into subgraphs as patches.

We actually can make an annotation to pseudo-split the edge (by recording its splitting path). When a novel path go through the cutted edge, we are able to determine the intersection (orthogonal path) or reject it (tangential path).

### Path elements
**Sharp edges.** Seems that it is viable via [**CGAL**](https://doc.cgal.org/latest/Point_set_processing_3/index.html#title61), section 14. Then, paths can be traced along sharp features to get sharp edge paths through KNN.

**Boundary edges.** If a vertex is really at the boundary, there is no neighbor along one of the 4-rosy direction.

**Corners** Boundary corners can be determined by line segment simplification.

**Intersections** Intersections of orthogonal paths can be detected via KNN split.

### Patch criterion
- Topology, homeomorphic to a disk

- **Valance, 3-6** By maintaining the path elements, counting corners is trivial.
- **Convex, No left/U-turn** By measuring the angle between paths, we are able to determine the convexity.
- **Valance Match** By counting internal singularity regions, we are able to verify the valance match.
- **Length concerns** It can be estimated by projecting paths through directions and accumulate lengths.

## Data structure
### Points
```json
Point {
  "p": "XYZ coordinate",
  "n": "Normal direction",
  "t": "one tangent direction",
  "orient_num": 4,
  "singular_group": "grouped singular region",
  "is_boundary": "boundary mark",
  "is_sharp": "sharp mark",
  "adj": "struct PointAdj"
}
```
### Adjancency
```json
PointAdj {
	"ids": "an array of neighboring point ids",
}
```
```json
OrientPointAdj {
	"ids": "neighboring ids",
	"path_id": "path id if it follows a path",
	"intersecting_paths": [("path_id","edge_id")],
	"is_singular": "singular edges are not valid as paths"
}
```
```json
Path{
	"end_idx": ["end point indices"],
	"arcs": [
		["internal points"]
	]
}
```
### Patch
```json
Patch {
	"ids": "points belonging to the patch",
	"outer_corners": "outer oriented corners",
	"genus": "genus",
	"num_singularities": "xxx"
}
```
## Algorithm

### Tangent Field
```
Input Point.p.
Build adjacency KNN Point.adj.
Normal estimation to get Point.n.
Instant-mesh to get Point.t.
For each point:
   gather its neighbor, triangulate them in the 2d tangent plane.
   gather neighbor vertices, count p.orient_num.
Region growing to group singularities.
For each group, triangulate the neighbors and count the group orient_num, updating them to each point.orient_num/singular_group.
```

### Motor-cycle graph
```
Split each point into 4.
Assign each edge from Point.adj into two of 4x4 pairs of oriented-points.
```

### Sharp-boundary Initialization
1. Detect sharp points using CGAL.
2. OrientPointAdj without forward edges are considered boundaries in original points.
3. Trace motocycle graphs along sharp points to form sharp paths (find longest paths, remove branches.)
4. Trace boundary points in the original graph, and map the path to motor-cycle graph. (might introduce new connections inside node with orthogonal directions.)

(3 and 4) By tracing the route, we need to filter noise to ensure a clean graph. Other tangential paths should be removed. Paths can be traced by iteratively finding the longest paths (which should be robust), and remove overlapped tangential paths. We also remove very thin paths. Now, every path is simply a sequence of oriented point ids.

5. Record path ids to edges that intersect paths, and confirm path-path intersection, potentially add new points into the path.
6. building path structure with intersections as end_idxs that splits arcs.
7. Partition the region by region growing.
8. Update status for all patches.

### Rounds
9. non-convex corners are connected to each other build novel paths. Followed by 5-6-7-8.
10. non-convex corners are connected to edges to build novel paths. Followed by 5-6-7-8. (Great! Now everything is convex!)
11. Random internal sampling. Sort them, and **If valid**, preserve it Followed by 5-6-7-8.
12. Edge sampling. Sort them, and **If valid**, preserve it Followed by 5-6-7-8.
13. Recursively do 9-12 for invalid subpatches.