<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1A2980,50:7F00FF,100:E100FF&height=220&section=header&text=Tiny-diffusion&fontSize=58&fontColor=ffffff&fontAlignY=38&desc=DDPM%20%C2%B7%20Classifier-free%20guidance%20%C2%B7%20DDIM%20%E2%80%94%20from%20scratch&descAlignY=62&descSize=17&animation=fadeIn" width="100%"/>

<a href="https://github.com/Umarfarook1/Tiny-diffusion">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2800&pause=600&color=B388FF&center=true&vCenter=true&width=900&lines=Build+a+diffusion+model+from+the+forward+process+up.;DDPM+%E2%86%92+CFG+%E2%86%92+DDIM+%E2%80%94+with+the+math+derived.;Trained+on+CIFAR-10%2C+CelebA-HQ%2C+and+FFHQ.;Samples%2C+ablations%2C+and+a+writeup+included." alt="Typing SVG" />
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/status-WIP-F5A623?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/tier-Intermediate-7F00FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/dataset-CIFAR--10%20%2F%20CelebA-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/license-MIT-1f6feb?style=for-the-badge"/>
</p>

<sub><i>The smallest repo that takes you from "what is a noise schedule" to a working classifier-free-guided sampler · without skipping the math.</i></sub>

</div>

---

## Why this repo exists

Diffusion is the most consequential generative paradigm of the decade and most engineers know it as a black box behind a `pipe(prompt)` call. This repo derives it from scratch · forward process, score / ε-prediction parameterization, classifier-free guidance, DDIM sampler · and trains a small UNet on CIFAR-10 and CelebA, with sample grids checked in.

> **Status:** scaffolding the noise schedule and forward process. First milestone: produce non-garbage CIFAR-10 samples.

## The math (in one screen)

**Forward process** · gradually destroy the signal:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)
\qquad\Rightarrow\qquad
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon
$$

**Reverse process** · learn ε with a UNet, denoise step by step:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0, \varepsilon, t}
\left[\, \lVert \varepsilon - \varepsilon_\theta(x_t, t) \rVert^2 \,\right]
$$

**Classifier-free guidance** · at sample time, blend conditional and unconditional ε:

$$
\tilde\varepsilon_\theta(x_t, c) = (1+w)\,\varepsilon_\theta(x_t, c) - w\,\varepsilon_\theta(x_t, \emptyset)
$$

That's the whole game. The repo implements each line above with a pointer back to this section.

## Pipeline

```mermaid
flowchart LR
    X0[x_0 image] -->|forward q| XT[x_t noisy]
    T[timestep t] --> N
    XT --> N[ε-pred UNet]
    C[class label c<br/>10% drop] --> N
    N --> EP[predicted ε]
    EP --> L[MSE loss]

    subgraph Sampling
        direction LR
        XS[x_T ~ N(0, I)] --> NS[ε-pred UNet]
        NS --> CFG[CFG blend]
        CFG --> RS[DDIM step]
        RS -.-> NS
        RS --> X0H[x_0]
    end
```

## Stack

| Component | Choice | Why |
|---|---|---|
| Noise schedule | linear + cosine | reproduce DDPM and Nichol & Dhariwal |
| Backbone | UNet (GroupNorm + SiLU + sinusoidal time embed) | canonical |
| Conditioning | class-embedding + drop-prob 0.1 (CFG) | enables guidance scale at sample time |
| Sampler | ancestral DDPM + deterministic DDIM | quality + speed dial |
| Datasets | CIFAR-10, CelebA-HQ 64x64 | small enough to iterate |
| Eval | FID via clean-fid | comparable to literature |

## Quickstart <sub><i>(coming soon)</i></sub>

```bash
# train DDPM on CIFAR-10 (single GPU, ~few hours)
uv run train.py --config configs/cifar10.yaml

# sample a 8x8 grid with CFG scale 3.0
uv run sample.py --ckpt out/cifar10/ema.pt --grid 8 --w 3.0

# DDIM 50-step sampling
uv run sample.py --ckpt out/cifar10/ema.pt --sampler ddim --steps 50

# eval FID against 50k real samples
uv run eval_fid.py --ckpt out/cifar10/ema.pt
```

## Sample gallery <sub><i>(populated as runs complete)</i></sub>

| Dataset | Model | Steps | Sampler | FID ↓ | Grid |
|---|---|---|---|---|---|
| CIFAR-10 | small UNet | 1000 | DDPM | TBD | TBD |
| CIFAR-10 | small UNet | 50 | DDIM | TBD | TBD |
| CelebA 64 | small UNet | 1000 | DDPM | TBD | TBD |

## Roadmap

- [ ] Forward process + cosine/linear schedule + sanity-check forward animation
- [ ] UNet backbone (GroupNorm, sinusoidal time embed, attention at low res)
- [ ] DDPM training loop + EMA weights
- [ ] DDPM ancestral sampler
- [ ] DDIM sampler (deterministic, 50-step)
- [ ] Classifier-free guidance + ablation over guidance scale w
- [ ] FID eval harness
- [ ] CIFAR-10 sample grid + checkpoint on HF
- [ ] CelebA-HQ 64x64 sample grid + checkpoint on HF
- [ ] Companion blog post deriving DDPM from the forward process

## Inspiration & required reading

- [Lilian Weng · *What are diffusion models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) · single best derivation
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) · the reference implementation
- [Ho et al. · DDPM (2020)](https://arxiv.org/abs/2006.11239)
- [Song et al. · DDIM (2021)](https://arxiv.org/abs/2010.02502)
- [Ho & Salimans · Classifier-free guidance (2022)](https://arxiv.org/abs/2207.12598)
- [Karras et al. · *Elucidating the design space of diffusion models* (EDM)](https://arxiv.org/abs/2206.00364)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1A2980,50:7F00FF,100:E100FF&height=110&section=footer&fontColor=ffffff" width="100%"/>
</div>
