# Primary papers and author-code checks

The teaching implementations were written for this course. Upstream training files were inspected to check objectives and conventions; their code is not vendored here. Small networks, synthetic data, and explicit numerical choices make the methods runnable on CPU. Inspection date: 2026-09-13.

## Flow Map Matching

[Paper](https://arxiv.org/abs/2406.07507v2). Class methods: `fmm-lagrangian, fmm-eulerian`.

[Inspected author source](https://github.com/nmboffi/flow-maps/blob/2f115a07fa9073553193e4b265dfc303827af2b0/py/common/losses.py), revision `2f115a07fa90`.

Frozen learned velocity teachers; residual-map JVPs; small MLP and bounded time range.

## How to build a consistency model: Learning flow maps via self-distillation

[Paper](https://arxiv.org/abs/2505.18825v2). Class methods: `self-distill`.

[Inspected author source](https://github.com/nmboffi/flow-maps/blob/2f115a07fa9073553193e4b265dfc303827af2b0/py/common/losses.py), revision `2f115a07fa90`.

Diagonal regression plus an EMA progressive target, normalized by a floored interval length. Learned uncertainty weights are omitted.

## One Step Diffusion via Shortcut Models

[Paper](https://arxiv.org/abs/2410.12557v3). Class methods: `shortcut`.

[Inspected author source](https://github.com/kvfrans/shortcut-models/blob/601004348667094e1b71f30942199759412d4432/targets_shortcut.py), revision `601004348667`.

Two half-step average target and dyadic intervals retained. Continuous start-time sampling and small MLP replace image-specific schedules and clipping.

## Mean Flows for One-step Generative Modeling

[Paper](https://arxiv.org/abs/2505.13447). Class methods: `meanflow`.

[Inspected author source](https://github.com/Gsunshine/meanflow/blob/d70cb55d298ee03c53bf6da67bec281082e4e2d9/meanflow.py), revision `d70cb55d298e`.

Backward clock, conditional-velocity JVP, detached target, adaptive loss, and 75% diagonal proportion retained. Uniform time sampling; no class guidance.

## Flow Map Language Models: One-step Language Modeling via Continuous Denoising

[Paper](https://arxiv.org/abs/2602.16813v3). Class methods: `fmlm`.

[Inspected author source](https://github.com/david3684/flm/blob/a1918d5164e5038e37d0b7a4fb2010ce75b863b3/algo.py), revision `a1918d5164e5`.

PSD denoiser target and Gaussian decoding clock retained. Independent clean-data diagonal supervision; EMA targets; quadrature lookup and MLP replace full language architecture.

## Categorical Flow Maps

[Paper](https://arxiv.org/abs/2602.12233v1). Class methods: `categorical`.

[Inspected author source](https://github.com/olsdavis/semicat/blob/558602a0fa722514e4a6012f5c46a8ae178b3068/semicat/models/semicat.py), revision `558602a0fa72`.

Released ECLD endpoint CE plus sum-of-squares time energy. Detached-target KL has identical student gradient. EMA target replaces the current network target.

## Discrete Flow Maps

[Paper](https://arxiv.org/abs/2604.09784v1). Class methods: `discrete-lsd, discrete-esd`.

Derived from the paper. The linked project page has a placeholder Code link, so no author implementation was verified. Positive log denominators are explicitly clamped and counted; no gradient surgery.

## Diamond Maps: Stochastic Flow Maps

[Paper](https://arxiv.org/abs/2602.05993). Class methods: `diamond`.

[Inspected author source](https://github.com/PeterHolderrieth/diamond_maps/blob/d30f65c75a169a2ed624f146b2770aeef882543a/posterior_diamond_maps/py/common/losses.py), revision `d30f65c75a16`.

GLASS conditional velocity and Lagrangian posterior distillation. Exact Gaussian-mixture denoiser replaces a pretrained image teacher. Weighted posterior recovery uses an explicit Gaussian proposal with a known density.

## Meta Flow Maps enable scalable reward alignment

[Paper](https://arxiv.org/abs/2601.14430v2). Class methods: `meta`.

[Inspected author source](https://github.com/adh1s/mfm/blob/53c0f60db695cad88bbace8fb26469614e9e7d7d/src/mfm/losses/losses.py), revision `53c0f60db695`.

Data training with independent inner/outer noises and a shared endpoint; conditional semigroup objective. Optional Equation 43 fine-tuning also checked against src/mfm/losses/finetune.py.

## Expanding Flow Maps

[Paper](https://arxiv.org/abs/2607.21585v1). Class methods: `expanding`.

[Inspected author source](https://github.com/sophtang/ExpandingFlowMaps/blob/4defc1ff168d12526d85b551bea78150b9aec605/README.md), revision `4defc1ff168d`.

Paper Algorithms 2-4 and local-clock equations. The inspected repository contains README and artwork only. Classroom implementation includes a shared lifted canvas, remaining/interval count heads, explicit local clocks, and budget-capped binomial insertions.

## Strong Stochastic Flow Maps

[Paper](https://arxiv.org/abs/2606.01086v1). Class methods: `ssfm`.

[Inspected author source](https://github.com/sammccallum/ssfm/blob/24b563620a8ee61c683d791fe802b56668f98f9b/ssfm/losses.py), revision `24b563620a8e`.

First two Legendre integrals, Chen composition, known diffusion, small-step matching. The released loss detaches an EMA split target; paper Algorithm 1 detaches the direct target. Both orientations are selectable. OU replaces image/molecular experiments; no learned uncertainty weights.

## Supporting lecture demonstrations

`flow-matching` supplies the local velocity baseline. `consistency` demonstrates endpoint consistency on a learned flow teacher using the forward clock and a residual endpoint parameterization; it is not a reproduction of the original EDM consistency training system. `latent` demonstrates the encoder/map/decoder construction and does not attribute it to a separate paper called Latent Flow Maps. `numerical_examples.py` retains the lecture’s scalar ODE experiment and exact arithmetic checks.

## Version and result scope

Paper links retain the lecture versions when supplied. Author repositories may have advanced after those paper versions. The source revision above records exactly what was inspected. The lecture’s paper figures and benchmark values remain in the slide deck; the files under `verified_examples/` contain newly executed course examples, with their own data, compute settings, and metrics. No reported FID, generative perplexity, or biological benchmark is reproduced by these CPU teaching runs.
