---
title: Review of Retopology
toc: true
categories: [科研,重拓扑]
date: 2023-11-26 20:00:00
tags: [科研,重拓扑]
---

## Background

Retopologize the 3D data into a quad-like surface mesh is critical for 3D asset generation. Personally, I like the recent paper [**Reliable Feature-Line Driven Quad-Remeshing**](https://www.quadmesh.cloud/ReliableQuad.pdf) and aim to explain it.

## Method

### Overview
Given a triangle mesh $\mathcal{M}$, it aims to partition the mesh into surface patches satisfying several metrics, including
- Topology, homeomorphic to a disk
- Valance, 3-6
- Convex, No left/U-turn
- Valance Match
- Length concerns

In practice, we compute tangent field, generate paths to segment patches, and quadrangulate patches into the quad mesh.

### Tangent field
We compute 4-rosy field for mesh faces, determine singularities for mesh vertices using existing methods, e.g., instant-meshes.

### Patch Segmentation
**Path tracing.**
Build a motor-cycle graph (vertices-direction connection), and dijkstra algorithm to compute loop for any sampled vertex-direction pair.

**Global round.**
We sample a set of loops, sort them according to distance (priority for further paths). Preserve the loop if
- Resolve/improve any criterion
- Split any problematic patch
and then recursively address each problematic sub-patch.

The rounds consist of three steps:
- point-to-point convexity-connection sampling
- internal loop sampling
- border-border

