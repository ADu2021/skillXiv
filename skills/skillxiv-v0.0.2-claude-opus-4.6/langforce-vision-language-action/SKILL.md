---
name: langforce-vision-language-action
title: "LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.15197"
keywords: [vision-language-action, bayesian-decomposition, robotic-control, language-alignment, latent-actions]
description: "Align vision-language-action models with natural language instructions using Bayesian decomposition with latent action queries, improving robotic manipulation generalization. Use when building agents that follow language commands while observing visual scenes and producing motor control."
---

# LangForce: Vision-Language-Action Alignment

This skill enables vision-language-action models to better follow language instructions by decomposing the problem using Bayesian inference and latent action queries, improving out-of-distribution generalization for robotic tasks.

## When to Use
- Building robotic agents that follow natural language instructions
- Training vision-language-action models for manipulation tasks
- Systems where language commands should guide visual policy decisions
- Tasks requiring strong generalization to unseen instruction variations

## When NOT to Use
- Pure visual navigation without language guidance
- Simple rule-based control systems (doesn't need Bayesian decomposition)
- Real-time inference with strict latency requirements
- Tasks with limited instruction variation

## Key Concept
Vision-language-action (VLA) models often ignore language instructions, treating them as secondary. LangForce addresses this through Bayesian decomposition that explicitly maximizes alignment between:
- **Language instructions** (what the user wants)
- **Visual observations** (what's in the scene)
- **Actions** (what the robot should do)

The method uses latent action queries to explore the space of possible actions consistent with both vision and language.

## Implementation Pattern

Implement Bayesian decomposition to align vision, language, and action:

```python
# Pseudocode for LangForce alignment
class LangForce:
    def __init__(self, vla_model, latent_action_dim=64):
        self.vla = vla_model
        self.latent_action_dim = latent_action_dim

    def infer_action(self, observation, language_instruction):
        # Encode visual observation
        visual_features = self.vla.encode_vision(observation)

        # Encode language instruction
        language_features = self.vla.encode_language(language_instruction)

        # Bayesian decomposition: find action that maximizes
        # P(action | vision, language)
        latent_actions = self.sample_latent_actions(k=8)

        scores = []
        for latent_action in latent_actions:
            # Decode latent to real action
            action = self.vla.decode_action(latent_action)

            # Score consistency with vision and language
            vision_score = self.score_consistency(action, visual_features)
            language_score = self.score_consistency(action, language_features)

            combined_score = vision_score + language_score
            scores.append(combined_score)

        # Select highest-scoring action
        best_idx = argmax(scores)
        return latent_actions[best_idx]
```

The method ensures language instructions actively guide action selection rather than being ignored.

## Key Results
- Substantial improvements in out-of-distribution robotic manipulation
- Better generalization to novel instruction phrasings
- Addresses failure mode where VLA models ignore language
- Maintains performance on in-distribution tasks

## Research Context
The paper identifies that standard VLA training leads models to ignore language instructions when vision is informative. Bayesian decomposition forces explicit alignment, improving both within-distribution and zero-shot generalization.
