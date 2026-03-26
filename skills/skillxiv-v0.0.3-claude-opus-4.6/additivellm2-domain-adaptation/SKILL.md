---
name: additivellm2-domain-adaptation
title: "AdditiveLLM2: Multi-Modal Language Models for Additive Manufacturing"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22017"
keywords: [Domain Adaptation, Language Models, Additive Manufacturing, Knowledge Extraction, Visual Understanding]
description: "Adapt general LLMs to specialized manufacturing domains via domain-adaptive pretraining on open-access journals and visual instruction tuning. Extract 50M tokens and 24K images from peer-reviewed papers, achieve >90% accuracy on domain knowledge tasks, and enable real-time defect identification from manufacturing images."
---

# AdditiveLLM2: Domain Adaptation for Manufacturing

## Domain Problem: Knowledge Gap in General LLMs

Additive manufacturing (AM) involves specialized terminology (LPBF, FDM, melt pool dynamics, material properties) and visual patterns (layer defects, porosity) that general LLMs rarely encounter. Manufacturing practitioners need models that understand equipment specifications, material behavior, and quality control—not general knowledge about 3D printing.

**Gap Analysis:** General Gemma-3-12B achieves <50% accuracy on manufacturing knowledge tasks; domain-specific terminology is treated as out-of-vocabulary or incorrectly classified.

## Source Method & Adaptation Recipe

**Foundation Model:** Gemma-3-12B (open-weights baseline)

**Domain Data Curation:** Extract text and images from 1,704 peer-reviewed papers across four open-access journals:
- Journal of Additive Manufacturing
- Rapid Prototyping Journal
- Specialized AM conferences (via arXiv)

**Dataset Composition:**
- 29 million text tokens (50M target after tokenization)
- 24,000 images with captions
- Focus on: process parameters, material properties, defect modes, quality metrics

**Three-Stage Training Pipeline:**

1. **Text Domain-Adaptive Pretraining (DAPT):** Unsupervised MLM on AM corpus; trains terminology and concept associations specific to manufacturing.

```python
# Domain-adaptive pretraining: continuous pretraining on domain data
# Train masked language modeling on AM corpus
# This builds internal representations for:
# - Equipment types (LPBF, FDM, SLM, DMLS)
# - Material properties (viscosity, thermal conductivity, porosity)
# - Defect modes (layer adhesion, warping, spatter)
```

2. **Image Domain-Adaptive Pretraining:** Vision encoder fine-tuning on AM images; learns visual patterns specific to manufacturing artifacts and defects.

3. **Visual Instruction Tuning:** Supervised fine-tuning on (image, question, answer) triples extracted from papers; teaches model to answer AM-specific questions about images.

```python
# Instruction tuning examples:
# Q: "What defects are visible in this LPBF part cross-section?"
# A: "Lack-of-fusion porosity in layers 5-7, surface roughness >5μm"
# Extracted from figure captions and supplementary materials
```

**Training Details:** LoRA rank-16 for efficiency; epochs tuned on validation set; separate loss weights for text and vision modalities.

## Deployment Lessons

**Lesson 1: Data Quality Over Quantity**
- 29M tokens from curated peer-reviewed sources outperforms 100M tokens from web-scraped manufacturing forums
- Academic papers provide accurate causal explanations; forums often contain myths about equipment behavior

**Lesson 2: Image Diversity Matters**
- 24K images sufficient for defect recognition if sourced from multiple equipment types, materials, and process parameters
- Overfitting risk when training data heavily skewed toward single process (e.g., 80% LPBF); require balanced sampling

**Lesson 3: Evaluation Must Be Domain-Aware**
- Standard NLU benchmarks (GLUE) irrelevant; create domain-specific benchmark:
  - General AM knowledge (multiple-choice): 20 questions from textbooks
  - Process parameter prediction: "Given part geometry and material, what laser power range?"
  - Defect identification: Classification from images

**Lesson 4: User Acceptance Requires Transparency**
- Manufacturing teams need explanations: "Layer 5 shows porosity because melt pool temperature likely dropped below X°C (detected via image features)."
- Black-box accuracy isn't enough; trace predictions to training data examples

## Practical Impact

- **Accessible Specialization:** 29M tokens is feasible for domain teams to curate; previous domain-specific models required billions of tokens
- **Real-Time Defect Detection:** Vision instruction tuning enables edge deployment for factory-floor QC
- **Continuous Improvement:** Model can be re-trained quarterly as new process innovations appear in literature
- **Cost Reduction:** Reduces reliance on expert technicians for initial defect triage

## Generalization to Other Domains

Recipe applies to any technical domain with:
- Established peer-reviewed literature (>1K papers)
- Visual patterns (images, diagrams, schematics)
- Specialized terminology and causal reasoning

Tested concept on biomedical imaging, materials science, semiconductor manufacturing.
