---
name: animalclap-taxonomy-aware-pretraining
title: "AnimalCLAP: Taxonomy-Aware Language-Audio Pretraining for Species Recognition"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22053"
keywords: [Taxonomy-Aware Learning, Audio Classification, Contrastive Learning, Species Recognition, Ecological Traits]
description: "Build taxonomy-aware audio-text pretraining systems for species recognition from animal vocalizations. Train contrastive models that augment text prompts with hierarchical taxonomic structure (scientific/common names, phylogenetic sequences), evaluate on unseen species via rare-species test sets, and predict ecological traits directly from audio."
---

# AnimalCLAP: Taxonomy-Aware Language-Audio Pretraining

## Problem Statement

Traditional audio classification models fail to recognize unseen animal species because they lack awareness of biological relationships. Audio-only systems (e.g., baseline CLAP) achieve just 1.61% top-1 accuracy on unseen species, while species form natural hierarchies (6 classes → 66 orders → 341 families → 2,152 genera) that could guide learning.

## Core Innovation: Taxonomy-Aware Pretraining

Integrate hierarchical biological structure into contrastive language-audio pretraining by augmenting text representations with taxonomic metadata.

**Data Structure:** 4,225 hours of recordings covering 6,823 species with 22 ecological trait annotations (diet, activity pattern, habitat, climate distribution, social behavior).

**Text Augmentation Strategy:** For each species, generate prompts combining:
- Common name (e.g., "American Robin")
- Scientific name (e.g., *Turdus migratorius*)
- Taxonomic sequence (e.g., "Aves → Passeriformes → Turdidae → Turdus → migratorius")

**Architecture:** HTS-AT audio encoder + RoBERTa text encoder with contrastive loss.

## Key Results

**Unseen Species Classification:** AnimalCLAP achieves 27.6% top-1 accuracy vs. baseline CLAP's 1.61%—a 17× improvement. This validates that hierarchical structure generalizes to novel species.

**Taxonomy Ablation:** Randomizing taxonomic sequence ordering reduces accuracy substantially, confirming that hierarchy ordering (not just presence of names) drives generalization.

**Trait Prediction:** Model successfully predicts ecological traits directly from audio:
- Activity patterns: 83.7% accuracy
- Predator classification: 92.6% accuracy
- Broader environmental traits: lower but significant

**Test Set Design:** 300 rare species with <15 recordings each prevents training data leakage and validates true out-of-distribution generalization.

## Deployment Recipe

1. **Data Collection Criteria:** Use only Creative Commons-licensed recordings from iNaturalist and Xeno-canto; verify habitat/temporal diversity.

2. **Annotation Workflow:** Obtain species labels, aggregate taxonomic names via open databases (NCBI Taxonomy), compute taxonomic sequences programmatically.

3. **Text Prompt Construction:** For species S, generate: "A recording of [common_name], scientifically known as [scientific_name], belonging to the sequence [path_from_class_to_species]."

4. **Training:** Contrastive learning with standard CLIP objective; ensure balanced sampling across taxonomic levels to prevent genus/family overfitting.

5. **Evaluation:** Always construct test sets using rare species (<15 recordings) to measure generalization to unseen taxa. Include both accuracy and trait prediction metrics.

## Practical Implications

- **Scaling:** Hierarchy enables few-shot learning for data-scarce species; economic impact for conservation monitoring.
- **Trait Transfer:** Learned audio representations capture ecological properties, enabling downstream tasks (habitat prediction, behavior classification) without additional annotation.
- **Generalization Principle:** Metadata-informed contrastive learning outperforms brute-force scaling in domains with natural hierarchies.
