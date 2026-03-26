---
name: speed-by-simplicity
title: "Speed by Simplicity: Unified Backbone for Audio-Video-Text Streaming"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.21986
keywords: [Architecture Design, Multimodal, Single-Stream, Audio, Video]
description: "Replace multi-stream modality-specific pathways with a unified Transformer backbone processing text, video, and audio tokens in shared sequence via self-attention. Achieves superior visual quality (4.80 vs 4.76), 75% better speech clarity (14.6% WER vs 19.23%), and 80% human preference wins—particularly strong for human-centric scenarios with expressive facial performance and audio-video sync."
---

## Component Identification

**Old Design (Multi-Stream Baseline)**
- Separate pathways for video and audio tokens
- Dedicated fusion blocks for modality alignment
- Modality-specific preprocessing and embedding layers
- Complex cross-attention mechanisms for temporal coordination

**New Design (Single-Stream)**
Unified Transformer backbone with shared self-attention sequence combining text, video, and audio tokens.

## Motivation & Problem Statement

Leading open-source models rely on heavily specialized designs that increase parameter count, training complexity, and latency. The simplified approach tests whether a single shared backbone—treating all modalities as sequential tokens—can achieve competitive or superior performance while reducing architectural overhead.

## The Modification

The core architectural change replaces modality-specific pathways with token-level fusion:

```python
# Single-stream processing: all modalities in shared sequence
# Text tokens: [CLS] w1 w2 ... wN [SEP]
# Video tokens: [VID] v1 v2 ... vM [SEP]
# Audio tokens: [AUD] a1 a2 ... aK [SEP]

# Combined sequence processed by single Transformer backbone
sequence = [text_tokens + video_tokens + audio_tokens]
output = transformer_backbone(sequence)  # Unified self-attention

# Modality-specific decoders applied to output
text_out = text_decoder(output[text_indices])
video_out = video_decoder(output[video_indices])
audio_out = audio_decoder(output[audio_indices])
```

This eliminates dedicated fusion blocks and modality-specific alignment layers, allowing emergent cross-modal interaction through the shared attention mechanism.

## Ablation Results with Exact Numbers

### Visual Quality
- daVinci-MagiHuman: 4.80
- LTX 2.3: 4.76
- Ovi 1.1: 4.73

### Speech Clarity (Word Error Rate)
- daVinci-MagiHuman: 14.60%
- LTX 2.3: 19.23%
- Ovi 1.1: 40.45%

### Human Preference
- 80% preference vs Ovi 1.1
- 60.9% preference vs LTX 2.3

### Domain-Specific Performance
Strong in human-centric scenarios:
- Expressive facial performance
- Natural speech-expression coordination
- Realistic body motion synthesis
- Precise audio-video synchronization

Multilingual capability: Six major languages with consistent quality

## Conditions of Applicability

**Works well when:**
- Modality tokens share temporal alignment (speech-to-video sync is critical)
- Model capacity is sufficient (unified backbone needs adequate parameters)
- Training data includes diverse modality combinations
- Human-centric content is primary (facial expression, gesture, speech coordination critical)

**Less optimal when:**
- Modalities operate at very different timescales
- Extreme modality imbalance in training data
- Real-time processing with hard latency constraints (unified attention still requires full sequence processing)

## Drop-In Replacement Checklist

- [x] Maintains same input/output interface (text + video + audio conditioning)
- [x] Compatible with existing video diffusion decoders
- [x] No architectural changes required to downstream components
- [x] KV-cache compatible for streaming inference
- [x] Can be trained with standard cross-entropy losses
- [x] Inference latency improved over multi-stream baselines
