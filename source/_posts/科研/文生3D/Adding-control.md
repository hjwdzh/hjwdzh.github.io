---
title: Adding Controls
toc: true
categories: [科研,文生3D]
date: 2023-11-26 20:00:00
tags: [科研,文生3D]
---

## Background

 We introduce additional geometry controls to better align the appearance with geometry. As from [**introduction**](/2023/11/18/科研/文生3D/Introduction/), geometry alignment can be achieved by incorporating geometry features for score distillation (align geometry with the rendering), or controlling the RGB to align with geometry features.

## Align Color to Geometry

### ControlNet

**Control Conditions.** One of the most popular choices for aligning color to geometry is via controlnet. For example, we can render geometry features to views as edge/normal/depth. [**Avatarverse**](https://arxiv.org/abs/2308.03610) further controls character appearance with dense-pose features.

From my experiments, different geometry features contribute to the control in various ways.
- Canny edge is suitable for precisely controlling the boundary alignment. For example, the generated RGB image actually aligns with the canny edge with pixel-level accuracy. Therefore, it is important to texture the 3D model with edge control. **However, rendering the edges might be problematic given stupid triangle soups in the CAD model. Occlusion boundary/sharp edge features are naturally ambiguous.** My current solution is to render a normal map and directly extract canny features from the normal image, **which seems to work okay.**
- Normal/depth has **similar behavior** that guides the network to understand continuous/discontinuous regions and remove ambiguity inside regions surrounded by canny edges. Thus, such condition is valuable to use. **Since depth can be rendered without a bug, we use depth rather than normal.**
- Other features like coordinate maps need to be retrained. However, they might provide richer information than local features like normal or edge. Therefore, it also deserves to be considered if a decent pre-trained model is available.

**Working with SDS.** ControlNet converges better than the standard Stable Diffusion model with SDS optimization. It is caused by additional edge features that resolve the ambiguity. However, it still loses details for inner surfaces.

On the other hand, it seems that ControlNet changes the physical meaning of $\epsilon(\mathbf{x},t,\emptyset)$ since it introduces additional controls. As a result, **[**NFSD**](https://arxiv.org/abs/2310.17590)) does not apply to ControlNet**. I plan to drop NFSD unless a good open-source version is available since a non-conditioned SD also does not work well in the 3D texturing case. ControlNet is considered in [**LucidDreamer**](https://github.com/EnVision-Research/LucidDreamer), and I would like to try it.

### Prompt-free Stable Diffusion
[**Prompt-free diffuser**](https://arxiv.org/pdf/2305.16223.pdf) can be viewed as replacing the text prompt with an image prompt in controlnet. However, it works pretty bad.

## Align Geometry with Diffusion Prior

### TODO (sweetdreamer etc., will try it if such geometry optimization is important)