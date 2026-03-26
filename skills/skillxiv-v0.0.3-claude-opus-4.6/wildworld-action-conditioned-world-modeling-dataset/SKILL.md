---
name: wildworld-action-conditioned-world-modeling-dataset
title: "WildWorld: Large-Scale Dataset for Dynamic World Modeling"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.23497"
keywords: [World Modeling, Action-Conditioned, Video Dataset, State Annotations, Interactive Environment]
description: "Build action-conditioned world models with explicit state tracking using WildWorld's 108M+ frames from Monster Hunter: Wilds. Includes data acquisition protocol with skeleton and world state annotations, quality filtering pipeline removing temporal discontinuities and cutscenes, and WildBench evaluation metrics (video quality, camera control, action following, state alignment) for assessing long-horizon consistency and state-aware predictions."
category: "Evaluation Infrastructure"
---

## Collection Methodology

WildWorld captures gameplay data from Monster Hunter: Wilds using a custom acquisition platform that synchronizes multiple data streams: RGB frames, depth maps, character skeletons, camera poses, and world states. The dataset contains over 108 million frames with 450+ action annotations spanning movement, attacks, and skill casting. Raw footage is processed through an automated gameplay pipeline that uses UI navigation automation and behavior trees controlling NPC companions, enabling large-scale data collection from interactive game environments. This approach directly addresses the gap in existing datasets: most lack meaningful action spaces and conflate visual observations with action signals.

## Data Acquisition Protocol

The collection pipeline operates in distinct stages:

- **Multi-stream recording**: Simultaneously capture RGB, depth, skeletal poses, camera pose (position + rotation), and action vectors at synchronized intervals
- **Automated gameplay**: Drive extended play sessions using behavior trees for NPC control and UI automation to navigate game menus without manual intervention
- **Action annotation**: Rich action space (450+) captures discrete actor movements and semantic skill execution rather than treating actions as frame-level visual changes
- **Hierarchy and metadata**: Establish action-level hierarchy enabling temporal understanding of multi-step skill sequences

## Quality Control & Filtering

Post-capture filtering removes problematic samples across multiple dimensions:

- **Temporal coherence**: Discard clips with duration < 81 frames to ensure sufficient temporal context; remove samples with temporal discontinuities or frame gaps
- **Visual occlusion**: Filter out scenes where critical observations (character pose, state variables) are occluded or unreliable
- **Cutscene removal**: Exclude non-interactive sequences and camera transitions that violate world-state assumptions
- **Continuity validation**: Verify skeletal motion smoothness and camera pose continuity across sequential frames

## WildBench Evaluation Metrics

The benchmark operationalizes world modeling quality across four dimensions:

**Video Quality Metrics**: Measure motion smoothness, dynamic degree (scene variability), aesthetic quality, and pixel-level image fidelity. These capture whether generated content is visually plausible and maintains consistent rendering quality.

**Camera Control Metrics**: Compute absolute and relative pose error (APE/RPE) comparing generated camera trajectories against ground truth, isolating control accuracy from content quality.

**Action Following Metrics**: Use LLM-based consistency judgment to assess whether generated video faithfully executes intended action sequences. This dimension catches semantic failures missed by perceptual metrics.

**State Alignment Metrics**: Track skeletal keypoint accuracy measuring whether predicted character poses remain consistent across time, quantifying error accumulation in autoregressive generation.

## Data Format & Splits

WildWorld provides hierarchical structured data with per-frame annotations enabling multiple experimental setups. Splits partition data across skill types and session boundaries to prevent state leakage. Each frame includes synchronized RGB and depth plus structured action vectors and world state variables, enabling research into how actions drive state transitions versus visual changes.

## When to Use

Use WildWorld when developing world models for interactive environments with rich action spaces and explicit state dynamics. This dataset is particularly suited for training action-conditioned architectures like skeleton-controlled (SkelCtrl) and state-aware (StateCtrl) models that predict future frames conditioned on sequences of discrete actions.

## Key Findings & Implications

Skeleton-conditioned models achieve near-perfect action following but sacrifice aesthetic quality. State-aware models show promise but demonstrate error accumulation in iterative next-state prediction when operating autoregressively. This reveals the fundamental tradeoff: explicit state tracking enables action control but introduces compounding errors without mechanisms to correct state drift across long horizons.
