---
name: chanrg-rna-structure-generalization
title: "Paradigm Challenge: RNA Structure Prediction Generalization Fails with Foundation Models"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22330"
keywords: [RNA Structure Prediction, Generalization Failure, Paradigm Challenge, Out-of-Distribution, Foundation Models]
description: "Overturn the assumption that scaling foundation models improves RNA structure prediction by understanding why they fail out-of-distribution. Includes structure-aware deduplication revealing 33-fold residual redundancy in prior benchmarks, out-of-distribution test regimes (GenA, GenC, GenF), and root cause analysis showing coverage and wiring failures. Foundation models achieving 67.3% on held-out test drop to 18.0% OOD (26.7% retention), while structured decoders retain 92.3%, enabling practitioners to recognize when scaling fails and when inductive structure matters."
category: "Paradigm Challenge"
---

## Prior Belief

The deep learning community widely believed that foundation models represent the frontier of RNA structure prediction. The intuition: scaling parameters, training data, and compute should overcome domain challenges through learned representations. Evidence supporting this belief included strong performance on standard benchmarks (e.g., Archive of Structural Proteomic Data) where foundation models achieved the highest held-out test accuracy. Practitioners made resource allocation decisions based on this assumption: invest in larger models, more pretraining, and expect generalization.

## The Falsifying Experiment

CHANRG introduces structure-aware deduplication and out-of-distribution test regimes that overturn this assumption.

**Core Experiment Protocol**: Using Rfam clustering, the authors deduplicated 10+ million sequences by structural similarity, discovering that prior "held-out" test sets contained 33-fold residual structural redundancy. The same RNA structures, represented by different sequences, leaked across train-test boundaries. This violates the generalization premise: held-out accuracy measures interpolation within the training structure distribution, not true generalization.

**Out-of-Distribution Regimes**: Three complementary OOD splits test different forms of structural transfer:
- **GenA**: Architecturally distant RNAs (different secondary structure families)
- **GenC**: Evolutionarily distant organisms (structural conserved function, sequence diverged)
- **GenF**: Phylogenetically distant lineages (maximum distance within family)

**Measurement**: Direct accuracy comparison on held-out (HD) vs each OOD regime, revealing performance cliff for foundation models.

**Sample Size & Robustness**: 170,083 non-redundant structures tested across multiple models, with length-controlled analysis confirming the effect persists independent of sequence length.

## The Performance Inversion

Foundation models exhibit dramatic performance collapse:
- **Held-out accuracy**: 67.3%
- **Out-of-distribution accuracy**: 18.0%
- **Retention ratio**: 26.7% (worst generalization)

Contrast with structured prediction baselines:
- **Structured decoders**: 30.2% held-out → 27.8% OOD (92.3% retention)
- **Direct neural predictors**: 35.0% held-out → 28.8% OOD (82.5% retention)

The inversion is stark: the model achieving highest held-out performance generalizes *least* robustly. This contradicts the scaling paradigm's core promise.

## Root Cause Analysis

The paper rules out obvious explanations:

**Not length**: Length-controlled evaluation on sequences matched for length shows the inversion persists. Scaling doesn't help length generalization.

**Not model scale**: Scaling RiNALMo improves held-out performance substantially more than OOD robustness. Larger models widen the OOD gap.

**Two failure modes identified**:

1. **Coverage failure**: Foundation models predict overly conservative base pairs (recall 14.0%, precision 34.8%), missing correct pairings. The model learns to avoid false positives at the cost of massive false negatives.

2. **Wiring failure**: Foundation models correctly predict some helices but assemble them into wrong global topology. Local predictions are partially correct but fail to respect global constraints (no crossing base pairs, non-interleaving structures).

## Revised Principle

Foundation model scaling alone is insufficient for RNA structure prediction generalization. Inductive structure—constraints embedding domain knowledge about RNA physical properties (no crossing, planarity, thermodynamic stability)—is essential. Models must learn both sequence-to-structure mapping AND global topology constraints. Unstructured foundation models fail because they treat structure prediction as unconstrained sequence-to-sequence learning, losing the discrete combinatorial properties that define valid RNA folds.

**Scope**: This revised principle applies broadly to structured prediction problems where outputs must satisfy hard constraints (graph planarity, thermodynamic feasibility, structural alignment). It may not apply to sequence-level understanding tasks where soft constraints suffice.

## Implications for Practice

**For research on RNA models**: Stop assuming foundation model scaling solves generalization. Explicitly incorporate inductive biases. Structured decoders and physics-informed architectures outperform raw scaling despite lower absolute capacity.

**For benchmark design**: Standard benchmark held-out splits are insufficient for evaluating structure prediction. Use structure-aware deduplication and OOD regimes. Standard benchmarks substantially overestimate real-world robustness by 3-4x.

**For practitioners deploying RNA tools**: Foundation models may achieve impressive lab performance on curated datasets but fail on novel RNA families. Use ensembles of structured models for production systems; don't rely on single large models.

**For intuition updating**: Inductive structure matters as much as scale. The mental model "bigger model = better generalization" needs revision to "scale + structure = generalization."

## When This Principle Applies

Apply this revised understanding when:
- Predicting structured outputs (RNA, protein, 3D geometry)
- Outputs must satisfy hard combinatorial constraints
- Test set structural diversity exceeds training diversity
- Practitioners assume scaling alone solves generalization

## When the Prior Belief Still Holds

Foundation models can still excel when:
- Predicting semantic properties not constrained by structure (functional classification, binding prediction)
- Domain contains few hard constraints (natural language, unconstrained text generation)
- Train and test sets share structural distributions (in-domain evaluation)
- Soft constraints suffice (sequence likelihood, perplexity)

## Validation Checklist

- [ ] Use structure-aware deduplication on your held-out test set to detect redundancy
- [ ] Evaluate on OOD subsets testing architectural, evolutionary, and phylogenetic distance
- [ ] Compare foundation models against structured decoders on OOD accuracy specifically (not just held-out)
- [ ] Analyze failure modes: are they coverage (missing correct pairs) or wiring (wrong global assembly)?
- [ ] Implement reference decoder (padding-free, symmetry-aware) with 6.7x memory savings and 3.3x speedup without accuracy loss
