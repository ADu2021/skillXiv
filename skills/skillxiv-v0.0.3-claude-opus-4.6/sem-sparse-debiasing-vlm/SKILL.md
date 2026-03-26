---
name: sem-sparse-debiasing-vlm
title: "SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.19028
keywords: [Vision-Language Models, Debiasing, Sparse Autoencoders, Post-Hoc, Fairness]
description: "Achieve post-hoc debiasing of frozen vision-language models by operating in sparse autoencoder latent space. Identify and modulate bias-relevant neurons with neuron-level precision while preserving task-relevant features, requiring no task-specific fine-tuning."
---

## Component ID
Sparse Embedding Modulation (SEM) for Post-Hoc VLM Debiasing

## Motivation
Vision-language models like CLIP exhibit bias in retrieval and zero-shot classification. Traditional debiasing requires full model fine-tuning. Bias and task-relevant information are heavily entangled in dense embedding space, making precise intervention difficult. A method for post-hoc modification of frozen models would enable deployment-time fairness corrections.

## The Modification
SEM decomposes CLIP text embeddings into disentangled features using a Sparse Autoencoder (SAE), enabling neuron-level bias intervention on frozen pretrained models:

1. **Sparse Autoencoder Decomposition** - Use a pre-trained SAE trained on general text data to decompose dense text embeddings into interpretable sparse dimensions, where each dimension corresponds to identifiable semantic or bias-relevant features.

2. **Dual-Score Neuron Evaluation** - Compute two scores per neuron:
   - **Content relevance**: How much a neuron activates on task-relevant content (via LLM-generated paraphrases)
   - **Bias sensitivity**: How much a neuron responds to different bias classes (structured comparisons)

3. **Modulation Coefficients** - Combine scores into per-neuron multipliers that amplify content-relevant neurons and attenuate bias-relevant ones.

4. **Debiased Reconstruction** - Apply modulation coefficients, then reconstruct embeddings through the SAE decoder.

## How It Works

**Scoring stage**: For each neuron, gather activations across diverse prompts:
- Paraphrase-based content estimation (does the neuron track semantic content?)
- Bias-class comparison (does the neuron vary with demographic attributes?)

**Modulation stage**: Combine scores via weighted combination that prioritizes content-relevant neurons, suppresses bias-sensitive ones, and reconstructs via SAE decoder.

Three implementation variants available:
- **SEMi**: LLM-generated paraphrases for content estimation
- **SEMb**: Explicit bias prompts for structured identification
- **SEMbi**: Combined approach (both sources)

## Ablation Results
The paper demonstrates:
- Substantial fairness gains in both zero-shot classification and retrieval tasks
- Improvements over linear projection baselines, especially in worst-group accuracy
- Consistent performance across multiple bias dimensions and model variants
- Post-hoc applicability to frozen models without retraining
- Effectiveness with minimal task-specific data (just bias class definitions)

## Conditions
- Requires pre-trained SAE (authors use SAE trained on general text)
- Works on frozen models—no fine-tuning needed
- Applicable to any vision-language embedding space (text or image)
- Requires definition of bias classes/attributes for the target domain
- More effective when content and bias features are partially separable in SAE space

## Drop-In Checklist
- [ ] Load pre-trained SAE for the embedding model
- [ ] Implement content relevance scoring (paraphrases or task-specific examples)
- [ ] Implement bias sensitivity scoring (bias-class prompt comparisons)
- [ ] Design modulation coefficient combination (prioritize content, suppress bias)
- [ ] Add modulation application and SAE reconstruction step
- [ ] Test on zero-shot classification and retrieval tasks
- [ ] Validate fairness improvements (worst-group accuracy, demographic parity)
- [ ] Benchmark against fine-tuning-based debiasing methods
