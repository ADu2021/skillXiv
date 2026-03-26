---
name: perceptio-spatial-token-vlm
title: "Perceptio: Perception Enhanced Vision Language Models via Spatial Token Generation"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.18795
keywords: [Vision-Language Models, Spatial Grounding, Segmentation, Depth Reasoning, Fine-Grained Understanding]
description: "Enhance VLM spatial grounding by enforcing explicit intermediate spatial reasoning before text generation. Generate segmentation and depth tokens as perception pathways, anchoring responses in 2D and 3D geometric reasoning for improved referring expressions and spatial tasks."
---

## Component ID
Spatial Token Generation Pipeline for Perception-Enhanced VLMs

## Motivation
Vision-language models excel at semantic understanding but struggle with fine-grained spatial grounding and geometric reasoning. When responding to spatial questions or referring expressions, models generate text without explicitly reasoning about 2D layout or 3D structure, leading to hallucinations about object locations and spatial relationships.

## The Modification
Perceptio enforces a structured output sequence that grounds language generation in explicit spatial reasoning:

**Output Sequence Constraint**: Segmentation tokens → Depth tokens → Text answer

This forces the model to emit spatial information **before** generating language, anchoring subsequent text responses in measurable 2D and 3D cues.

## Perception Enhancement Mechanism
The model operates through three parallel pathways:

**1. Visual Encoding Pathway** - Standard vision backbone for semantic understanding (object identification, scene comprehension)

**2. Segmentation Pathway** - Frozen SAM2 encoder providing pixel-level mask-aware representations. Generates discretized segmentation tokens indicating spatial regions and object boundaries.

**3. Depth Pathway** - Pre-trained monocular depth estimator with VQ-VAE codebook that discretizes continuous depth into compact token sequences. Represents 3D geometry in discrete tokens.

These pathways run in parallel with different architectural specializations, then combine outputs.

## How It Works

**Token Generation Order**:
1. **Segmentation generation phase** - Model emits segmentation tokens representing spatial layout and object masks
2. **Depth generation phase** - Model emits depth tokens encoding 3D geometry
3. **Language generation phase** - Model generates text answer conditioned on both spatial understanding tokens

This ordering ensures spatial reasoning precedes and grounds linguistic output.

**Integration with Autoregressive Decoding** - The segmentation and depth tokens are inserted into the standard autoregressive generation sequence, making the approach compatible with existing VLM training and decoding pipelines.

## Ablation Results
The paper demonstrates substantial gains on spatial tasks:
- **RefCOCO/+/g benchmarks**: State-of-the-art performance on referring expression comprehension
- **HardBLINK spatial reasoning**: +10.3% improvement on challenging spatial QA tasks
- **VQA performance**: Maintains competitive performance on general visual question-answering while strengthening spatial grounding
- Consistent improvements across diverse spatial understanding benchmarks

## Conditions
- Requires pre-trained depth estimation model (monocular depth predictor)
- Requires SAM2 or similar segmentation encoder for mask representation
- Works best when tasks demand explicit spatial reasoning (referring expressions, spatial QA)
- VQ-VAE discretization requires appropriate codebook size for depth representation

## Drop-In Checklist
- [ ] Integrate frozen SAM2 encoder for segmentation pathway
- [ ] Add pre-trained monocular depth estimator with VQ-VAE codebook
- [ ] Implement segmentation token generation (discretize mask outputs)
- [ ] Implement depth token generation (discretize depth via VQ-VAE)
- [ ] Enforce output sequence constraint (seg → depth → text)
- [ ] Modify autoregressive decoding to include spatial tokens
- [ ] Test on RefCOCO/+/g referring expression benchmarks
- [ ] Validate spatial reasoning improvements on HardBLINK and similar tasks
- [ ] Benchmark general VQA to ensure no regression on non-spatial tasks
