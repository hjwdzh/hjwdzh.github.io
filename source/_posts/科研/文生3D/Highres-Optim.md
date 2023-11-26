---
title: High Resolution Optimization
toc: true
categories: [科研,文生3D]
date: 2023-11-26 20:30:00
tags: [科研,文生3D]
---

## Background

We aim to generate geometry with a high-resolution appearance. The only way is to zoom in since stable diffusion has limited resolution.
There are several options for zoom-in. The first question is whether to use a multi-resolution geometry model, and the second question is the way to zoom in for renderings.

## Geometry model
We will investigate this problem in the experiments of 3D representations.

## Zoom-in rendering
I conclude several ways to zoom in.
- Progressive zoom-in. Such methods first render in far distance to capture the whole object. Then, it zooms in to the closer viewpoint to optimize details.
**Progressive zoom-in** is apparently a good option when no initial geometry is given -- in which case there is no definition of close/far. **However, a limitation is that by looking closer, global context is unavailable it usually leads to bad results after long optimization.**
- Alternative zoom-in. Once a good initial shape is available, we can sample far/close viewpoints alternatively or randomly. However, the sample bias still exists, for example, it is unclear how to balance global/local viewpoints, leading to unstable optimization.
- Image pyramid. What if we jointly optimize global/local contexts? Yes! It is more stable! And if the memory is limited, we can separate highres into several patches and sample some of the patches.

**Texture Experiment** We find that stable diffusion cannot generate a good human face at low resolution. However, by applying the image pyramid idea with ControlNet: We sample a view, sample eight 512x512 patches at 2048x2048 resolution, and a global 512x512 resolution image, and sum their SDS loss to optimize the texture. As a result, the global human and her face are both decent.