---
title: Single View SDS
toc: true
categories: [科研,文生3D]
date: 2023-11-21 23:00:00
tags: [科研,文生3D]
---

## Introduction

According to my [**introduction**](/2023/11/18/科研/文生3D/Introduction/), the simplest experiment to start with is the single-view text-to-image experiment using the stochastic optimization method. The reason that we want to optimize rather than denoise is that we extend to 3D, we need to seamlessly generate textures from multiple views and stochastic optimization appears to be the most promising method. In this article, I am going to investigate all the details related to the optimization process.

## Background

### Formulation
Given a view parameterized by $\theta$, we can render it via $\mathcal{R}(\theta)$. For example, $\theta$ can represent a vector of pixel RGB/Latent, and $\mathcal{R}$ as the identical mapping. A more advanced version parameterizes $\mathcal{R}$ as a hash encoding followed by a tiny MLP according to **InstantNGP**. As from [**DreamFusion**](https://dreamfusion3d.github.io), the SDS loss can be represented as:
<center>$\nabla_{\theta}\mathcal{L}_{SDS} = E_{t,\epsilon}\; w(t)(e_\phi(\mathbf{z}_t;\mathbf{y},t)-\epsilon)\frac{\partial \mathbf{x}}{\partial \theta}$</center>
where $e_\phi$ is the predicted noise given $\mathbf{z}_t$ as input and $\mathbf{y}$ as prompt at timestep $t$, and $w(t)$ is the scale of noise added to the data. Practically, we optimize the SDS loss stochastically by randomly sampling a gaussian noise $\epsilon$, a timestep $t$, add the noise to latent version of $\mathcal{R}(\theta)$ (probably via a encoder $\mathcal{E}$) as $\mathbf{z}_t$, and compute $|w(t)(e_\phi(\mathbf{z}_t;\mathbf{y},t)-\epsilon)|^2$ as the loss for each training step. An Adam optimizer is linked to $\theta$ for optimization.

### Experiment Setting
Although the formulation is clear, we derive three different versions of optimizing a single view.
1. $\theta$ is <u>pixels</u> of <u>latents in VAE</u>. (EPL)
2. $\theta$ is <u>hash encoding</u> of <u>latents in VAE</u>. (EHL)
3. $\theta$ is <u>hash encodiing</u> of <u>RGB</u>. (EHC)

We aim to derive a robust optimization scheme that works for all experiment settings.

### Challenges
The classical one to optimize (discussed as a typical experiment in [**NFSD**](https://arxiv.org/abs/2310.17590)) is EPL. By optimizing it with sufficient steps, you will get a smooth image with only rough shapes representing the prompt $\mathbf{y}$. With the same setting to optimize EHL, you get something that is completely out of expectation. For EHC, things get even worse and you find your image stays unchanged. Let's fix these issues one by one.

## Main ideas for EPL-N
I am mainly discussing three ideas includig Variational Score Distillation(VSD) ([**ProlificDreamer**](https://arxiv.org/abs/2305.16213)), Noise-Free Score Distillation ([**NFSD**](https://arxiv.org/abs/2310.17590))) and time-annealing ([**HIFA**](https://hifa-team.github.io/HiFA-site/)).

### NFSD

<center>$e_\phi(\mathbf{z}_t;\mathbf{y},t) = e_\phi(\mathbf{z}_t,\emptyset,t) + s\cdot(e_\phi(\mathbf{z}_t,\textbf{y},t)-e_\phi(\mathbf{z}_t,\emptyset,t))$</center>

NFSD introduces an important idea that disentangles the SDS loss term.
When $s=7.5$ is the normal classifier guidance scale, the result is oversmooth. When $s=100$, the result is oversaturated. Therefore, $\delta_C=(e_\phi(\mathbf{z}_t,\textbf{y},t)-e_\phi(\mathbf{z}_t,\emptyset,t))$ is the classifier term that really contributes to align the image to the text, and the high $s$ downgrade the effect of the term $e_\phi(\mathbf{z}_t,\emptyset,t)$ that smooth the image. It further states that $e_\phi(\mathbf{z}_t,\emptyset,t)$ can be decomposed into a denoising term $\delta_N$ and a correction term $\delta_D$ moving the "out-of-distribution" $\mathbf{z}_0$ into the data distribution, and the reason that SDS fails is that the denoiser unreliably predict $e_\phi(\mathbf{z}_t,\emptyset,t)$ for small timesteps given $\mathbf{z}_t$ as the in-distribution image added with Gaussian noise. **This is a remarkable finding and it really leads to the oversmooth effect**. They claim that the the correction term at small timestep $\delta_D=e_\phi(\mathbf{z}_t,\emptyset,t)$ since $\delta_N$ is negligible, while $\delta_D$ at larger timestep is approximated by

$-\delta_C(\mathbf{y}_{neg})$. Quite makes sense! **However, the correction term is still noisy and leads to a blurry background since multiple answers exist. Therefore, naively optimizing NFSD won't work as experimented me, and the main reason is that the unstable optimization easily trapped to a local minimum.**

### ProlificDreamer
The main idea of ProlificDreamer is to finetune a LORA model that estimates correct noises for the out-of-distribution image during the intermediate state of the optimization stage. Then, by subtracting the LORA prediction from the pre-trained prediction, the correction term can be robustly obtained. **However, it is questionable whether such a LORA model can be robustly optimized, and the results from an unofficial implementation [**ProlificDreamer 2D**](https://github.com/yuanzhi-zhu/prolific_dreamer2d) seem to be unstable. Further, the optimization is quite inefficient.**

### HIFA
Some works find that a larger timestep helps to build a global structure that aligns with the text. Such an idea is consistent with NFSD's explanation that the score is dominated by the classifier term and thus builds the semantic structure. As a result, these works sample larger time steps at the first stage and smaller timesteps when a reasonable structure is initialized correctly. HiFA further schedules the timestep by enforcing it to be sampled in decreasing order. From the NSFD model, we find that such a process means building the initial structure and then moving it to the data distribution. **I believe it brings an additional challenge since it is important to carefully tune the annealing pace so that the classifier term at large timesteps does not make the image oversaturated, and small timesteps do not make the image over smooth. As a result, for different parameterizations of $\theta$ and initial state $\theta_0$, the hyperparameters may need to change accordingly, which makes the optimization untrustworthy.**

## Solution
### Initialization
I find it important to initialize the latent space to be roughly a Gaussian distribution. Zero initialization sometimes makes the optimization hard to escape the **saddle** zero-point. For the color experiment, we should not directly initialize Gaussian noise at RGB space. Rather, we initialize it to fit the decoded Gaussian latent code.

### NFSD + Decreasing-order Scheduling
Given the aforementioned ideas, I finally derive a hybrid optimization trick combining both NFSD and timestep annealing. For decreasing-order timestep scheduling based on HiFA, we adopt their scheduling scheme $t=t_{max}-(t_{max}-t_{min})\cdot \sqrt{r}$ where $r$ is the timestep divided by the total timestep. For the EPL experiment, we find that making $w(t)=1-\Pi_t \alpha_t$ makes sense for Gaussian noise as the initial condition. Basically, by fully optimizing $\theta$ to match the score at each timestep (which is easy since each step is by optimizing $|\mathbf{x}-\mathbf{\hat{x}}|^2$ as a convex function), we actually get quite decent results.

There are lots of limitations to doing this though, as I discussed in HiFA.
- For EHL experiment has a complex $\frac{\partial \mathbf{x}}{\partial \theta}$. A single step by Adam won't fit the sampled score and leads to horrible results. By optimizing the score 5 times as an inner loop for each training step, the result matches the EPL experiment.
- For EHC, things get worse and we find it extremely hard to exactly fit the score (thanks to the stupid encoder!), and the result is always horrible.

Following is my cuisine.
- Initialize Gaussian noise in the latent space.
- Enumerate 1000 iterations, and for each iteration:
- Sample t according to HiFA.
- Compute NFSD score with $w(t)=1-\alpha_{prod}$. (Scale the correction term by an additional coefficient=0.1)
- **Fully optimize** $\theta$ to mach the score.
- To reduce the number of iterations, accordingly, amplify the NFSD score and we will get similar results.

Therefore, although the result can be great, I seek better solutions since EHC is important for the 3D optimization task.

### Robust NFSD + Time annealing
I first optimize the global structure by highlighting the classifier term in NFSD with larger timesteps as a warmup stage. This is consistent with many works. The only uncertain thing is that how many steps do we need to get such a global structure. Fortunately, there is a decent range of choices for the number of warmup iterations. After that, we really want to make the optimization stably converge to a decent answer. If so, even if we fail to fit the exact score at each training step due to the stupid encoder, we still stochastically converge to that answer that the score points to.

That's a cool idea! So can it happen? I propose a robust NFSD that balances the classifier term and the correction term: **For each iteration, I sample a large timestep to compute the classifier term and additionally, a small timestep to compute the correction term. I scale the correction term to make it twice the norm of the classifier term. Two terms are summed as the score so that each step robustly balances the semantics and the realism**. I find that the optimization can continue without changing the result too much, and it works for the EHC experiment.

Following is my cuisine.

- Initialize Gaussian noise in the latent space.
- Enumerate 600+ iterations:
- For the first 300 iterations, only sample timestep between 200 to 980 and optimize the NFSD. The warmup iteration needs to be determined carefully since the 3D experiment may be different.
- Then, compute the robust NFSD by balancing the smaller-step correction score to twice the scale of the larger-step classifier score and optimize it.
- Set learning rate as 3e-3 for EHL/C and 1e-2 for EPL.

### Investigation of the optimization
Sometimes the result still fails. However, I found a clear checklist to make it work.
- Visualize results every 10 steps.
- Ensure that the warmup stage brings reasonable global structure. (Reduce the step if oversaturated or increase it if nothing shows up)
- Ensure that the final result converges. (Increase the weight of the correction term if oversaturated or decrease it if over smooth)

With these principles, hopefully, we always manage to obtain a fixed set of hyperparameters for different experiment settings.

## Limitations
Very unfortunately, when playing with controlnet, the NFSD idea does not really work. I am seeking to make it work with a recent paper called [**LucidDreamer**](https://github.com/EnVision-Research/LucidDreamer).