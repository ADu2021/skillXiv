---
name: few-tokens-matter-vlm-attacks
title: "Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.21815"
keywords: [Vision-Language Models, Adversarial Attacks, Security, Model Robustness]
description: "Demonstrate that adversarial attacks on vision-language models need not target all tokens equally. Entropy-guided attacks identify high-entropy tokens (critical decision points) where perturbations have maximum impact, achieving comparable attack success with 80% fewer tokens targeted."
---

## When to Use This Skill
- VLM robustness evaluation and security testing
- Understanding model vulnerabilities to targeted attacks
- Developing defensive mechanisms against token-level attacks
- Analyzing which model components are most critical
- Security audits of vision-language systems

## When NOT to Use This Skill
- Building adversarial attacks against real-world systems (ethical and legal concerns)
- Production systems without proper security review
- Applications without explicit security testing mandate

## Problem Summary
Prior adversarial attack research on vision-language models assumed all tokens contribute equally to generation instability, leading to global attack strategies. This creates computationally expensive attacks requiring perturbations across extensive token sequences. However, VLM generation follows entropy-driven decision-making where only high-entropy tokens—approximately 20% of positions—disproportionately govern output distributions.

## Key Insight: Entropy-Guided Attack Strategy

Rather than distributing attacks globally, concentrate perturbations on high-entropy tokens where model uncertainty is maximal.

```python
class EntropyGuidedAttack:
    def __init__(self, vlm_model):
        self.vlm = vlm_model

    def compute_token_entropy(self, logits):
        """Identify uncertainty critical points"""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy

    def targeted_adversarial_attack(self, image, benign_prompt, target_harm):
        """Attack only high-entropy tokens"""

        # Step 1: Identify high-entropy token positions
        benign_logits = self.vlm.forward_logits(image, benign_prompt)
        entropy_scores = self.compute_token_entropy(benign_logits)

        # Select top 20% by entropy (critical decision points)
        entropy_threshold = torch.quantile(entropy_scores, 0.8)
        high_entropy_positions = entropy_scores > entropy_threshold

        # Step 2: Generate adversarial perturbations
        adversarial_image = image.clone()
        for high_entropy_pos in high_entropy_positions.nonzero():
            # Optimize perturbation for this position
            perturbation = self.compute_perturbation(
                image, benign_prompt, high_entropy_pos, target_harm
            )
            adversarial_image += perturbation * 0.1  # Small perturbation magnitude

        return adversarial_image

    def compute_perturbation(self, image, prompt, target_position, target_harm):
        """Gradient-based perturbation for specific token position"""
        # Use only gradients from target token position
        # Standard adversarial optimization: maximize harm output at position
        return torch.autograd.grad(
            loss=harm_loss(target_position),
            inputs=image,
            retain_graph=True
        )[0]
```

## Attack Success Metrics

**Empirical Results:**
- Attack success rate: 93-95% (comparable to global attacks)
- Token coverage: Only 20% of generation positions
- Computational efficiency: 80% reduction in gradient computations
- Transferability: 17-26% harmful output rate on unseen target models

**Generalization:**
- Entropy-based vulnerability recurs across architecturally diverse VLMs
- Works on Qwen, Gemini, LLaVA, and other model families
- Fundamental weakness in sequential decision-making, not architecture-specific quirk

## Defense Implications

**Understanding Vulnerabilities:**
High-entropy tokens represent uncertain decisions where:
- Multiple plausible continuations exist
- Perturbations easily shift outcomes
- Small input changes have outsized effects

**Defensive Strategies:**
1. **Entropy Regularization**: Flatten entropy distribution (reduce high-entropy peaks)
2. **Robust Token Selection**: Defend high-entropy positions preferentially
3. **Detection**: Monitor entropy for anomalously high scores
4. **Ensemble Voting**: Aggregate across different entropy-weighted experts

## Implementation Considerations

**For Security Researchers:**
- Use for authorized VLM robustness testing only
- Document findings through responsible disclosure
- Work within institutional review processes

**For VLM Developers:**
- Understand entropy hotspots in your models
- Design defenses targeting vulnerable positions
- Implement detection mechanisms

## Benchmark Evaluation

**Models Tested:**
- Qwen-VL, Qwen2-VL
- Gemini Vision models
- LLaVA-NeXT
- Claude-3 Vision

**Attack Success Metrics:**
- Benign→Harmful: 35-49% conversion rate
- Consistency: 93-95% attack success rate
- Efficiency: 5× faster than baseline attacks
