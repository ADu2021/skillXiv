---
name: rlvr-update-direction-analysis
title: "On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.22117
keywords: [RLVR, LLM Reasoning, Directional Analysis, Token Importance, Interpretability]
description: "Analyze reinforcement learning with verifiable rewards using signed log-probability differences to identify reasoning-critical tokens. Reveals that RLVR updates concentrate on low-probability tokens, enabling test-time amplification and training-time reweighting techniques."
---

## Research Question
How can directional analysis of parameter updates reveal which tokens are critical for reasoning improvements in RLVR-trained models, and how can this insight be exploited for inference and training?

## Analytical Instrument
**Signed Log-Probability Differences (Δlog p)** - Captures the direction and magnitude of probability shifts per token between base and RLVR models. More sensitive to reasoning-critical changes than magnitude-only metrics (entropy, KL divergence).

The paper shows that Δlog p exhibits "clear bimodal patterns" whereas magnitude-based metrics yield "nearly identical histograms for base and RLVR models."

## Controls
1. **Statistical comparison** of directional versus magnitude-based metrics on generated responses
2. **Token replacement interventions** - Selectively substitute base model outputs with RLVR choices based on different selection criteria (Δlog p vs. magnitude-based alternatives)
3. **Gradient analysis** - Explain why updates concentrate on low-probability tokens through theoretical analysis of natural gradient scaling

## Findings
- Δlog p identifies reasoning-critical tokens with ~10% replacement rate versus higher percentages for competing metrics
- RLVR's sparse updates concentrate on low-probability tokens receiving disproportionate gradient updates
- Magnitude-only metrics fail to discriminate critical tokens from noise (nearly identical distributions)
- Natural gradient norm scales directly with reward variance, explaining why low-probability tokens receive more optimization focus

## Practitioner Implications

**1. Test-Time Extrapolation** - Amplify the learned Δlog p direction at inference without retraining. Scale probability shifts by a factor α > 1 to enhance reasoning improvements. Improves accuracy without additional training cost.

**2. Training-Time Reweighting** - Upweight advantages (reward signals) for low-probability tokens during RLVR training. Concentrates gradient flow where it matters most, yielding consistent performance gains across benchmarks and models.

Both techniques translate the directional insight into actionable procedures for practitioners.

## Methodology Generality
The directional analysis framework generalizes beyond RLVR:
- Applicable to any RL setting with verifiable rewards (math, code, planning)
- Works across different model architectures and sizes
- Token-level granularity enables fine-grained optimization
- Δlog p metric is model-agnostic (requires only probability outputs)
