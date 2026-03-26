---
name: snap-speaker-nulling-deepfake-detection
title: "SNAP: Speaker Nulling for Artifact Projection in Speech Deepfake Detection"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.20686"
keywords: [Deepfake Detection, Speaker Embeddings, Subspace Projection, Audio Forensics, Domain Generalization]
description: "Detect speech deepfakes by decomposing features into speaker, artifact, and context subspaces—nulling speaker information via PCA projection to isolate artifact patterns. Train minimal logistic regression classifier on orthogonally-projected representations, achieving 56% error reduction while maintaining cross-speaker and cross-TTS generalization."
---

# SNAP: Speaker Nulling for Deepfake Detection

## Core Insight: Speaker Entanglement Problem

Audio deepfake detectors trained on self-supervised learning (SSL) representations exploit speaker-identity patterns instead of learning genuine synthesis artifacts. When models like WavLM encode speaker information strongly, detectors learn speaker-specific correlations rather than the acoustic cues that distinguish real from synthetic speech. This causes catastrophic failure when encountering new speakers or unseen TTS models.

**Why Non-Obvious:** Conventional wisdom assumes SSL representations separate "what is being said" from "who is speaking." In practice, speaker-identity encoding is so dominant that artifact-detection layers become speaker classifiers instead.

## Minimal Recipe: Subspace Projection

Decompose feature space as ℋ = 𝒮 + 𝒜 + 𝒞 where:
- 𝒮 = speaker-dependent subspace
- 𝒜 = artifact-related subspace
- 𝒞 = context (utterance, channel) subspace

**Three-Stage Pipeline:**

1. **Feature Extraction:** Concatenate WavLM-Large layers 8 and 22, apply mean pooling, L2-normalize
2. **Speaker Subspace Identification:** Compute PCA on speaker centroids to find the speaker-dependent subspace
3. **Nulling via Orthogonal Projection:** Project features orthogonally to suppress speaker subspace while preserving artifact and context information
4. **Classification:** Train logistic regression (only 2,049 parameters) on nullified features

```python
# Speaker subspace nulling via PCA projection
# 1. Compute speaker centroids from enrollment utterances
# 2. Center and PCA-decompose: C = U @ S @ V.T
# 3. For test feature x, compute orthogonal projection:
#    x_nulled = x - U @ U.T @ x
# This removes speaker-dependent variance while preserving other patterns
```

## Results & Cross-Domain Robustness

**ASVspoof 2019 LA (benchmark):** 0.35% EER—56% improvement over WavLM-ECAPA baseline

**In-the-Wild Dataset:** 15.39% EER vs. 22.22% for WavLM baseline

**Generalization to Unseen TTS:**
- CosyVoice2 (not in training set): Maintains strong performance
- F5-TTS (new diffusion-based model): Effective detection despite architectural novelty

**Key Finding:** Nulling speaker subspace removes speaker-specific decision boundaries, forcing the classifier to rely on genuine synthesis artifacts that generalize across speakers and TTS models.

## Deployment Considerations

1. **Enrollment Phase:** Collect 3-5 clean utterances from each reference speaker to establish speaker subspace
2. **PCA Dimensionality:** Determine via scree plot or cross-validation; typically 20-50 dimensions
3. **Scaling:** Method is lightweight—PCA + logistic regression runs on CPU in real-time
4. **Generalization:** Works best when training set includes diverse speakers; speaker diversity in training improves nulling robustness
5. **Failure Mode:** Performance degrades if speaker subspace overlaps with artifact subspace (rare but possible for certain TTS models); mitigate via larger enrollment sets

## Practical Impact

- **Simplicity:** Minimal parameters (logistic regression) reduce overfitting to seen speakers/TTS models
- **Interpretability:** Orthogonal projection provides clear subspace decomposition; decisions are traceable
- **Deployment:** Fast inference, no need for large models; suitable for edge deployment
