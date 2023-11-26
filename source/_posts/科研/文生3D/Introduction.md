---
title: Introduction to X-to-3D
toc: true
categories: [科研,文生3D]
date: 2023-11-18 11:00:00
tags: [科研,文生3D]
---

## Background
With the recent progress in the field of generative AI, the community observes new opportunities for 3D content generation exploiting novel neural network architectures pre-trained on large-scale 2D/3D datasets. This article aims to derive the picture of major components that hundreds of recently published papers work on and contribute to this field.

## Introduction
Input and output X-to-3D, in a nutshell, aims to convert some input information to 3D content. To expand it, input is mainly considered as text and image, while an image can be a single-view or multi-view of the same physical object. For 3D, the content can be a tiny object, or an indoor/outdoor scene, each of which faces different challenges. While we mainly discuss the object level since it has great value to the industry and could be the easiest one to handle, some approaches can be extended to work on scenes for potential AR/VR applications.

### Input
The input is easy to understand. The input is text/image which can be encoded as latent features. Images can be directly used as supervision for renderings considering the RGB and depth. Note that the image could also be potentially associated with a camera pose.

### 3D representation
While the input is quite clear, 3D content can be represented in different forms:
- <u>Compact mesh</u> is the most popular representation, containing vertices, faces, material, and texture $\mathcal{M}=\<\mathcal{V},\mathcal{F},\mathcal{M},\mathcal{T}\>$. It is always the most important representation since it fits quite well in the standard rendering pipeline for the gaming industry.
- <u>Signed distance field (SDF)</u> analyzes a signed distance to the nearest surface for any 3D location as $\mathcal{D}(\mathbf{p})$ so that a mesh can be extracted via the marching cube algorithm. I would differentiate SDF from the mesh since the extracted mesh is too complex, since edges and vertices are inappropriately distributed around the surface making the model extremely inefficient for industry use. It is still valuable since it is a 3D function representation that can be straightforwardly encoded by neural networks.
- <u>Neural radiance field (NeRF)</u> predicts volume density and color at any 3D location as $(\sigma,\mathbf{c}) = \mathcal{F}(\mathbf{p})$, so that an image can be rendered from such a representation by ray-marching and accumulate transmittance and colors along the query rays. While it is more complex than SDFs, it introduces additional degrees of freedom and is easier to fit multiview images for reconstruction tasks. However, the transfer from such a representation to explicit geometry usually needs additional regularization so that opacity regions are gathered around the real surface.
- <u>Shape templates.</u> The problem with compact mesh is that it is not friendly to neural networks. Therefore, a compromise is to fix its topology as a template mesh and solve a deformation function to change it to the desired geometry.
- <u>Parameterizable templates.</u> Another form of shape template is by further parametrizing the space of deformation by some hyperparameters to maintain the shape in a valid space, e.g. human body by SMPL.
- <u>Procedural shapes.</u> Shapes can be created by programs, where programs can be potentially solved with an autoregressive model/LLM.
- <u>Novel view synthesis.</u> While 3D consistency is hard to achieve, some works solve an easier problem that predicts rendering of the 3D as images given camera poses. Plausible images can be directly generated without considering the real 3D geometry information.

### Main problem formulation
From my point of view, there are three major categories of methods for dealing with this problem.

- <u>3D Generation.</u> The first set of methods aims to generate 3D content given input in a single forward pass at inference time. The advantage is obvious, that such methods are quite efficient. Instead, the quality of such methods is usually inferior, lacking details in terms of geometry or color. However, it is worthwhile mentioning that it has the potential to further improve given a larger dataset, or at least be used as an initial start for other methods.
- <u>Image generation</u> While 3D is hard to directly generate given limited data and architecture, it is possible to finetune 2D models to generate better images related to 3D, commonly known as the novel view synthesis problem.
- <u>3D Optimization</u> Since 2D generation or guidance is well developed, it can be used to guide the optimization of 3D content. Compared with the 3D generation, it is more time-consuming but expects better results.
- <u>Appearance optimization</u> Given 3D geometry, it is worth further enhancing the appearance, including the material and texture. It is a good idea to disentangle such a problem from 3D optimization for practical concerns: Honestly, the current 3D creation method does not meet the gaming standard. However, we can texture an existing gaming object, where appearance is an easier problem to solve.

## Recent Approaches

### 3D Generation
![General pipeline for 3D generation](2023/11/18/科研/文生3D/Introduction/pipeline-3dgen.png)
The main pipeline for 3D generation is simple. The input is passed to an encoder and sent to a generator to produce the 3D content, which is supervised by some losses during training. Variance is possible: Unconditional generation is possible by simply passing Gaussian noises to the generator. 3D representation is compressed as latents using a pre-trained auto-encoder.

Let's review some papers under this framework.
- [**EG3D**](https://arxiv.org/abs/2112.07945) takes pose and a random noise vector as input (1), encode it with a mapping network (2), and passes it to StyleGAN (3) to generate a triplane-nerf (4), supervised by L2 loss.
- [**ClipForge**](https://arxiv.org/abs/2110.02624) takes an image (1), encodes with a CLIP (2), uses it as a condition to train a normalizing flow (3) to predict a latent shape encoding (4) with L2 (5), where the shape latent space is analyzed by a shape autoencoder.
- [**ClipNerf**](https://arxiv.org/abs/2112.05139) takes an image (1), encodes with a CLIP (2), uses it to enrich the MLPs (3) of a disentangled nerf (4) that renders images and trained with GAN (4).
- [**ATT3D**](https://arxiv.org/pdf/2306.07349.pdf) takes text as input(1), encodes it with CLIP/T5(2), maps latent to instant-ngp weights(3), and the instant-ngp as a nerf representation(4) can be rendered and supervised by the SDS loss(5) given text-conditioned diffusion prior.
- [**RODIN**](https://arxiv.org/abs/2212.06135) takes an image (1) encoded by CLIP (2) and learns a triplane diffusion model (3) with an interesting channel pooling (specific for triplane formulation) to get a triplane NeRF (multi-scaled 4), that is supervised with an L2 and perceptual (VGG) loss.
- [**DiffTF**](https://openreview.net/pdf?id=q57JLSE2j5) is quite similar to RODIN but denoises triplane with a transformer.
- [**SSDNeRF**](https://arxiv.org/abs/2304.06714) jointly optimizes scene nerf (triplane) and a nerf diffuser so that the nerf diffuser generates a scene from Gaussian and the scene can be rendered to fit the image for supervision. In inference time, unconditional Gaussian noise is directly denoised by the NeRF diffuser to obtain the scene.
- [**Point-E**](https://arxiv.org/abs/2212.08751) takes either image/text as input(1), again encodes it with CLIP (2), concatenate it with noisy points and time step t into a transformer as a denoiser of a point diffusion generator(3), and predict the shape point cloud (4) supervised by the GT via L2 loss.
- [**GECCO**](https://arxiv.org/abs/2303.05916) is similar to PointE with the major difference that it utilizes image as a projective condition -- image features are concatenated with point locations for a set-transformer-based diffusion to denoise and generate.
- [**Shap-E**](https://arxiv.org/pdf/2305.02463.pdf) encode point clouds and rendering with transformer, map it to an MLP-based NeRF supervised with L2 loss.
- [**SDFusion**](https://arxiv.org/abs/2212.04493) obtain 3D-diffusion to generate latent volumes that can be decoded to SDF shape, where latent codes aim to compress SDF using VQGAN.
- [**Diffusion-SDF**](https://light.princeton.edu/wp-content/uploads/2023/10/diffsdf.pdf) diffuse to generate latent code that can be decoded into SDFs, learned in the 3D data.
- [**Point-UV-Diffusion**](https://github.com/CVMI-Lab/Point-UV-Diffusion) generates colors of a given shape using PVCNN, with post-processing to directly enhance the texture image.
- [**Sketch-a-shape**](https://arxiv.org/pdf/2307.03869.pdf) encodes shapes into tokens, and learns a masked transformer to generate the shape.
- [**3D-GPT**](https://arxiv.org/abs/2310.12945) use ChatGPT to encode and generate programs of given text description, to model the scene with infinigen.

Let's briefly summarize what we learned from these papers. First, image/text is usually encoded with CLIP, and a common usage of the clip feature is to make it as an input of a generator, which is usually modeled as a diffuser. The common choice for 3D can be SDF, point cloud, nerf, or latent features obtained from pre-trained autoencoder, VQGAN, etc. The generator is usually modernized as a multi-step diffusion network that works in latent or Euclidean space, while autoregressive models are also potentially valuable to use. Loss is simply the rendering loss, SDS loss (which we will discuss further), and CLIP/VGG loss. As a result, most aforementioned methods tend to seek variance of 3D representation and architectures of 3D generators to form new ideas, while their basic ideas are simply the same. From an engineering perspective, I find no reason to explore further on the representation/architecture. Instead, a large dataset may still be the key to success. However, it is worth comparing different existing choices of data representations and generator architecture. I would like to summarize several questions here.

1. <u>**Considering training/inference speed, robustness, and quality, what is the best representation? Candidates are NeRF (triplane, gaussian, NGP, etc.), SDF, and point cloud.**</u>
2. <u>**Is it worth making an auto-encoder to compress the 3D? Shall we compress it into a global vector or a volume of a local vector?**</u>
3. <u>**What is the best neural network architecture considering speed, robustness, and quality, and how do we utilize the conditions? (triplane CNN, point transformer, latent MLP denoiser, image feature or direct latent condition?)**</u>

For engineers not directly working in this field, I feel like the direct take-back is that we should follow SOTAs and make them an initial stage for further 3D content refinement before such methods are powerful enough to solve the problem in an end-to-end manner.

### 2D Novel View Synthesis
The logic of novel view synthesis (NVS) is much simpler than that of 3D generation since the output representation is fixed as an image. While NVS is usually modeled as a multi-view reconstruction problem, we are discussing its potential under the generalization setting -- images from novel views are synthesized by hallucinating rather than inferring. Here are some representative methods:
- [**3DiM**](https://arxiv.org/pdf/2210.04628.pdf) incrementally generate novel views by training a image+pose-conditioned diffusion model. For each denoising timestep, it randomly chooses a previous view with pose as the condition.
- [**Zero-123**](https://zero123.cs.columbia.edu) takes the same setting, but bruteforcely does it with a large dataset.
- [**MVDiffusion**](https://mvdiffusion.github.io) adds attention layers that take the corresponding region of multiple views to synchronize the diffusion of multiple images with better consistency.
- [**DreamFace**](https://arxiv.org/abs/2304.03117) train an image diffuser at texture space.
- [**MVDream**](https://arxiv.org/pdf/2308.16512.pdf) jointly diffuse multiple viewpoints given camera conditions using 3D attention.
- [**SyncDreamer**](https://arxiv.org/pdf/2309.03453.pdf) jointly diffuse multiple fixed viewpoints with depth-wise attation from aggregated feature volume to improve consistency from given training data.
- [**Wonder3D**](https://www.xxlong.site/Wonder3D/) further diffuse to generate consistent normal maps.
- [**3D-ImGen**](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_3D-aware_Image_Generation_using_2D_Diffusion_Models_ICCV_2023_paper.pdf) use depth prior to warp existing views to novel views, and warp back to create condition for image generation. Then, novel view synthesis is possible.
- [**GeNVS**](https://nvlabs.github.io/genvs/) aggregate multiple views into a feature volume and render it as a condition to generate images.

It seems to be magic that simple 2D generation achieves 3D consistency. It could be directly used as results for 3D generation, or it can serve as a guidance network that optimizes the 3D. However, I believe there are some important issues to be understood:

1. <u>**Does such a diffusion model have a limited capacity to generate rich 2D results?**</u>
2. <u>**How consistent the model is?**</u>
3. <u>**What is the effect of using such models to guide the 3D optimization?**</u>
4. <u>**How are the optimization results compared with direct reconstruction from synthesized views?**</u>

### 3D Optimization
![General pipeline for 3D optimization](2023/11/18/科研/文生3D/Introduction/pipeline-3doptim.png)
Since recent works take this topic seriously and aim to provide high-quality reconstruction, more challenges are seriously discussed and subproblems are disentangled with solutions to solve. Therefore, I believe it is worth mentioning each subproblem and contribution made by different works, rather than simply enumerating all the papers.

#### Loss
**CLIP.** Due to historical reason, <u>**CLIP loss**</u> is the first to be used in zero-shot 3D optimization. The basic idea is to optimize 3D content so that the rendering aligns with the given text in the CLIP space. Representative approaches include [**DreamField**](https://arxiv.org/abs/2112.01455), [**PureClipNeRF**](https://arxiv.org/abs/2209.15172), [**Dream3D**](https://arxiv.org/abs/2212.14704), [**TextDeformer**](https://arxiv.org/abs/2304.13348), [**ClipMesh**](https://arxiv.org/abs/2203.13333), and [**FaceClipNeRF**](https://arxiv.org/abs/2307.11418).

**SDS.** Since Clip focuses on semantic alignment with text, it does not offer fine-grained control at the pixel level. Instead, diffusion models offer denoising capability at the pixel level (at least on the downsampled version). Therefore, <u>**score distillation sampling (SDS) loss**</u> is proposed by [**DreamFusion**](https://dreamfusion3d.github.io) and followed by most latter works in this field, where concurrent work [**SJC**](https://arxiv.org/abs/2212.00774) proposes similar ideas. **SDS** often leads to the over-smooth/over-saturated problem. <u>**Variational score distillation (VSD)**</u> is proposed by [**ProlificDreamer**](https://arxiv.org/abs/2305.16213), stating that the problem comes from fixed-parameter sampling in 3D
, and thereby jointly finetuning a LORA diffusion to better accept samples around the mean shape in parameter space. Some modifications can be further made to finetune a dreambooth as <u>**BSD Loss**</u> by [**DreamCraft3D**](https://arxiv.org/abs/2310.16818). From another perspective, <u>**Classifier score distillation Loss**</u> from [**CSD**](https://arxiv.org/abs/2310.19415) empirically observes that the fundamental contribution in SDS only comes from the classifier-related denoising, and a final version with theoretical support comes from [**NFSD**](https://arxiv.org/abs/2310.17590) <u>**(noise-free score distillation)**</u>. *From my point of view, **VSD** seems to be a finetuned correction from SDS to NFSD and might be unnecessary, and it deserves to explore whether **NSFD** is sufficient to guide the optimization.*

SDS can not only guide RGB, but also guide other signals including normals, depths, or normalized coordinate maps. Surprisingly, pretrained stable diffusion can be directly used to guide the normal rendering ([**Fantasia3D**](https://fantasia3d.github.io)), and is also adopted by [**TADA**](https://tada.is.tue.mpg.de). A special case is [**GSGEN**](https://arxiv.org/pdf/2309.16585.pdf) which uses Point-E diffusion as an SDS guidance.

**Shape Regularization.** It is possible to incorporate parametrized 3D shapes to regularize the optimization. In avatar creation, SMPL(-XL) is a common use case adopted by [**ZeroAvatar**](https://arxiv.org/abs/2305.16411), [**AvatarCraft**](https://github.com/songrise/AvatarCraft) and [**Avatarverse**](https://arxiv.org/abs/2308.03610), [**DreamHuman**](https://arxiv.org/abs/2306.09329). Another shape regularization is from depth, predicted from rendering ([**HIFA**](https://hifa-team.github.io/HiFA-site/)) or input condition. The DMTet-based method can start from an initial shape, while an initial shape can also be used as a regularization. For example, [**Points-to-3D**](https://arxiv.org/abs/2307.13908) uses [**Point-E**](https://arxiv.org/abs/2212.08751) to obtain the initial shape and regularize the shape optimization.

**Image Regularization.** Since SDS has limitations in generating high-quality images, [**TextMesh**](https://arxiv.org/abs/2304.12439) proposes to adopt an image-to-image pipeline for generating high-quality images conditioned on rendering, and using such images to further refine the texture with <u>**L2 Loss**</u> ([**NerfDiff**](https://arxiv.org/pdf/2302.10109.pdf), [**NerDi**](https://arxiv.org/abs/2212.03267), [**DiffusioNeRF**](https://arxiv.org/pdf/2302.12231.pdf)), with alternatives as perceptual loss ([**RODIN**](https://arxiv.org/abs/2212.06135)) or <u>**GAN loss**</u> ([**IT3D**](https://arxiv.org/pdf/2308.11473.pdf)).

#### 3D representation
Many approaches adopt a multi-stage optimization scheme. While a <u>coarse nerf</u> is pre-optimized at low resolution to explore valid topology, a second-stage optimization on the mesh is adopted at higher resolution. Without volume rendering, the optimization is more efficient. For example, [**Magic3D**](https://research.nvidia.com/labs/dir/magic3d/) adopts such a scheme. DMTet is also a popular choice for 3D representation used by [**Fantasia3D**](https://fantasia3d.github.io). Recently-proposed gaussian splatting architecture supports faster optimization according to [**GSGEN**](https://arxiv.org/pdf/2309.16585.pdf), [**DreamGaussian**](https://github.com/dreamgaussian/dreamgaussian) and [**GaussianDreamer**](https://github.com/hustvl/GaussianDreamer).

There is also a divergence on whether to optimize color, material ([**TANGO**](https://proceedings.neurips.cc/paper_files/paper/2022/file/c7b925e600ae4880f5c5d7557f70a72b-Paper-Conference.pdf), [**Fantasia3D**](https://fantasia3d.github.io), [**Matlaber**](https://sheldontsui.github.io/projects/Matlaber)) or latent ([**LatentNeRF**](https://arxiv.org/abs/2211.07600)). *From my perspective, optimizing the latent seems unnecessary. While the material is quite important for production, I don't see why such optimization is possible to disentangle albedo from other components of the material model.*

#### View consistency
Another issue that SDS suffers is that denoising direction is not explicitly correlated with view information. A naive solution is to append descriptions like "front/back/side view" to the input text to create view-dependent conditions. [**Perpneg**](https://perp-neg.github.io) offers better control for views by interpolating the latent space of text encoding.

A more popular choice comes from SDS based on the pose-conditioned diffusion networks (check the previous section) using [**Zero-123**](https://zero123.cs.columbia.edu) or potentially more recent ones ([**MVDream**](https://arxiv.org/pdf/2308.16512.pdf), [**SyncDreamer**](https://arxiv.org/pdf/2309.03453.pdf), [**Wonder3D**](https://www.xxlong.site/Wonder3D/)). To improve fidelity, a joint SDS loss from standard stable diffusion and such methods can be used ([**Consistent123**](https://arxiv.org/abs/2309.17261), [**DreamCraft3D**](https://arxiv.org/abs/2310.16818)).

Another direction is to utilize rendered geometry signals to guide the consistency. From geometry to color, controlnet offers additional control over the generation based on geometry features. [**Avatarverse**](https://arxiv.org/abs/2308.03610) pre-train a dense-pose controlnet for avatar SDS. All other methods can adopt ControlNet based on normal/depth/edge. From the reverse way, the diffusion network can guide geometry more straightforwardly by directly denoise geometry signals including coordinate maps ([**SweetDreamer**](https://github.com/wyysf-98/SweetDreamer)), or normal/depth maps ([**HumanNorm**](https://arxiv.org/pdf/2310.01406.pdf)). A special case is [**Prompt-free diffuser**](https://arxiv.org/pdf/2305.16223.pdf) which takes structural image conditions and requires no other prompts by replacing CLIP with a semantic context encoder.

#### Condition Hack
Text can be achieved from an image via DreamBooth/Textual Inversion ([**3DFuse**](https://ku-cvlab.github.io/3DFuse/)). [**Perpneg**](https://perp-neg.github.io) can better assign multiple attributes to different objects by encoding interpolation. [**Progressive3D**](https://arxiv.org/abs/2310.11784?ref=ai-bites.net) makes local edits by conditioning on perpendicular text latent directions.

#### Training tricks
**SDS Training Tricks.** Hack on timestep sampling ([**HIFA**](https://hifa-team.github.io/HiFA-site/)), or gradient clip on SDS ([**PCG3D**](https://arxiv.org/abs/2310.12474)).

**Progressive High Resolution.** Coarse-to-fine is an important idea. For 3D representation, it is a common idea to start from a coarse shape (from Point-E) or optimize one (NeRF) followed by mesh refinement ([**Magic3D**](https://research.nvidia.com/labs/dir/magic3d/)) or DMTet ([**Fantasia3D**](https://fantasia3d.github.io)). [**MTN**](https://github.com/Texaser/MTN) directly optimizes multi-scale triplane NeRF. For colors, progressive zoom-in is commonly adopted ([**Avatarverse**](https://arxiv.org/abs/2308.03610), [**DreamCraft3D**](https://arxiv.org/abs/2310.16818))), while [**Avatarverse**](https://arxiv.org/abs/2308.03610) probably use different text condition from different view given known semantic parts.

### Other approaches
There are some other approaches to straightforwardly hacking the content, including direct texture painting ([**TEXTure**](https://arxiv.org/abs/2302.01721), [**Text2Tex**](https://daveredrum.github.io/Text2Tex), [**TexFusion**](https://research.nvidia.com/labs/toronto-ai/publication/2023_iccv_texfusion/)), direct geometry fusion ([**Scenescape**](https://scenescape.github.io), [**Text2Room**](https://github.com/lukasHoel/text2room)), or iteratively update dataset to achieve the editing goal ([**Control-4D**](https://control4darxiv.github.io), [**InstructNerf2Nerf**](https://instruct-nerf2nerf.github.io)).

Besides generating 3D content, a set of methods aim to understand 3D with foundation models that are worth mentioning, *since I believe understanding can reversely help generation, e.g., by fine-grained condition on scene level*. [**LeRF**](https://www.lerf.io/) reconstruct the nerf with language features obtained from 2D fundation models. [**Cap3d**](https://arxiv.org/pdf/2306.07279.pdf) renders multiple views, captions them one by one with BLIP, filters wrong captions with CLIP, and summarizes them with LLM. [**SATR**](https://arxiv.org/abs/2304.04909) aggregate multiview GLIP features into 3D for semantic segmentation. [**3D-LLM**](https://arxiv.org/pdf/2307.12981.pdf) and [**Point-LLM**](https://arxiv.org/abs/2308.16911) extract multiview features for language-vision-models to caption.

## Potential Approach
Given existing ideas, I would like to design a system that incorporates all valuable techniques.
1. **3D Initialization stage.** Given text/image as input, shall we initialize a shape with 3D networks? (Shap-E/Point-E)? If so, it will be the first step.
2. **3D Representation.** Is multi-stage necessary (e.g. NeRF + DMTet + Mesh-Refine)? What is the best architecture considering optimization quality, efficiency, and robustness? (Instant-NGP, Gaussian Splatting, Triplane, DMTet)
3. **Choice of diffuser.** Controlnet/Prompt-free diffuser.
4. **Choice of SDS.** Take NFSD and check whether VSD/BSD is still necessary.
5. **Geometry SDS.** Is coordinate map/normal/depth worth being finetuned by a diffuser to provide SDS that improves 3D consistency? (SweetDreamer, HumanNorm.)
6. **NVS Diffusion SDS.** Check MVDream/SyncDreamer/Wonder3D and figure out how they work, and how they help the consistency of 3D optimization.
7. **Geometry supervision.** Check whether depth/pseudo-depth/shape regularization from 3D initialization can help the geometry.
8. **Image supervision.** Check whether image diffusion is still enhancing the results as stated in Text2Mesh.
9. **Training Scheme.** Check whether gradient clip (PCG3D) and timestep annealing (HiFA) is still necessary.
10. **Progressive zoom-in** Check whether progressive zoom-in helps the quality (DreamCraft3D).
11. **Texture-space image-to-image diffusion** May enhance the details in one post-processing step.

Some ideas are entangled with each other, and I would like to start with easy experiments.
- **Image generation SDS** Check idea 3/4.
- **Texture generation SDS** Check idea 8/10/9.
- **3D Generation** Check idea 2/6/5/7.
- **Pre-post processing** Check idea 1/11.

### Timeline
- 1/2 is **potentially answered by [**LucidDreamer**](https://github.com/EnVision-Research/LucidDreamer) (d)**.
- 3 is experimented with ControlNet. **Will try the prompt-free diffuser. (a)**
- 4 is thoroughly experimented but not yet solved, **probably solved with LucidDreamer (c)**.
- 5/6/7 **still need to be experimented (e)**.
- **8 will be tried in the texture experiment. (b)**
- 9 seems not critical.
- 10 is verified with the texture experiment.
- **11 will be hacked later.**