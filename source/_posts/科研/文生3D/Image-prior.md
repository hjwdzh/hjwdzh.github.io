---
title: Image Prior Supervision
toc: true
categories: [科研,文生3D]
date: 2023-12-02 22:00:00
tags: [科研,文生3D]
---

## Background
Image prior supervision is a topic under the 3D texturing task. Although as we experimented in [**SDS**](/2023/11/21/科研/文生3D/Single-view-sds/), [**Highres-optim**](/2023/11/26/科研/文生3D/Highres-Optim/) and [**ControlNet**](/2023/11/26/科研/文生3D/Adding-control/), optimizing the color guided by SDS with ControlNet guidance is sufficient to generate good texture, efficiency and editing is hard to address. Therefore, I believe it is an important to study about editing images and using them to guide the texturing.

## Potential solution
There are two different problems to study to achieve the goal. 
1. How shall we generate good images for supervision?
2. How shall we use images as supervision?

### Image generation

The basic logic is to generate the image for a single view and expand that to multiple views with consistent image content.

**Single-view Image** A straightforward solution is to use ControlNet plus text prompts to generate images. Lower resolution preserves high-level semantic requests, and higher resolution preserves detailed structures like canny edges. Therefore, the question is how we combine the merits of both terms, which is not quite trivial. Of course, we can turn this problem back into an SDS optimization problem, which contradicts our purpose to generate fast and controllable images. Another promising direction is to train upsampling methods considering high-res structural information. [**Controlnet-Tile**](https://github.com/lllyasviel/ControlNet-v1-1-nightly) is an example that fits our needs. **I can generate a low-res image covering high-level semantics and upsample the entire image using joint canny edge and tile control to align to structures.**

**Multiview Image** Several works introduce solutions to generate consistent multiview images from a single image given fixed/given camera poses. It worths to try [**MVDream**](https://arxiv.org/pdf/2308.16512.pdf) or [**SyncDreamer**](https://arxiv.org/pdf/2309.03453.pdf), while more recent work called [**Instant3D**](https://instant-3d.github.io) also deserves to try if it is opensourced. **One important question is whether such methods can be seamlessly integrated with ControlNet.**

**Inpainting** Another idea is through inpainting -- Paint image to texture for a single view and progressively inpaint unobserved regions from other perspectives. I would like to try [**Text2Tex**](https://daveredrum.github.io/Text2Tex) as a typical example.

### Image guidance

**Direct Painting** It is not bad to directly paint images to texture as an initialization step, with certain fixes so that they are seamlessly aggregated in 3D. It is probably true that the quality is not as good as that obtained from SDS considering consistency, but the current result is at least good enough to serve as an initialization.

**Local/Global Correction** If the hacking still does not work perfectly, we should think about switching back to SDS optimization based on the texture initialization. Of course, we do not need to optimize the entire model. Instead, we can ask users to specify view frustum and only optimize local regions.