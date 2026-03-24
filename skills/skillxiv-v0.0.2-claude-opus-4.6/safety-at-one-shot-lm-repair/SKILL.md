---
name: safety-at-one-shot-lm-repair
title: "Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.01887"
keywords: [LLM Safety, Model Alignment, Efficient Fine-Tuning, Low-Rank Adaptation]
description: "Recover safety alignment in fine-tuned LLMs using only a single safety example, without sacrificing utility. Leverages low-rank structure of safety gradients to enable minimal-cost correction even when models have been compromised by extensive harmful training data."
---

## When to Use This Skill
- Recovering safety in accidentally fine-tuned models
- Rapid safety patching when new safety issues emerge
- Minimal-resource safety corrections in resource-constrained settings
- Multi-model safety updates across model portfolios
- Scenarios where full retraining is impractical

## When NOT to Use This Skill
- Initial model development (use comprehensive safety training instead)
- Situations requiring extensive safety certification
- Adversarially compromised models (one example insufficient)

## Problem Summary
Fine-tuning safety-aligned LLMs on domain data can substantially compromise safety properties. Prior solutions required extensive safety training data (hundreds of examples) and computational overhead, often degrading model utility. This creates a dilemma: practitioners need practical safety corrections without rebuilding entire models or sacrificing performance on useful tasks.

## Solution: Low-Rank Safety Gradient Exploitation

Safety corrections employ low-rank structure of safety gradients, enabling single-example recovery.

```python
class OneShot SafetyPatch:
    def __init__(self, model, reference_model):
        self.model = model  # Compromised model
        self.reference = reference_model  # Original safe model

    def patch_with_single_example(self, safety_example, learning_rate=0.01):
        """Recover safety with one reference example"""

        # Step 1: Compute safety gradient
        # Forward pass with compromised model
        output = self.model(safety_example)
        safe_target = self.reference(safety_example)

        # Compute alignment loss (KL divergence)
        alignment_loss = kl_divergence(output.logits, safe_target.logits)

        # Compute gradient
        gradients = torch.autograd.grad(
            alignment_loss, self.model.parameters(), retain_graph=True
        )

        # Step 2: Identify low-rank structure
        # Reshape gradients for singular value decomposition
        gradient_matrix = torch.cat([g.flatten() for g in gradients])

        # Perform SVD: G ≈ U @ S @ V^T
        U, S, Vt = torch.linalg.svd(gradient_matrix, full_matrices=False)

        # Retain only top-K singular values (low-rank approximation)
        k_rank = 10  # Effective rank
        U_low = U[:, :k_rank]
        S_low = S[:k_rank]

        # Step 3: Apply low-rank correction
        correction = U_low @ torch.diag(S_low) @ Vt[:k_rank, :]

        # Update model parameters
        for param, grad_correction in zip(self.model.parameters(), correction):
            param.data -= learning_rate * grad_correction

    def validate_safety_recovery(self, test_prompts):
        """Verify safety is restored without utility loss"""
        harmful_outputs = 0
        utility_score = 0

        for prompt in test_prompts:
            output = self.model.generate(prompt)

            # Check safety
            if is_harmful(output):
                harmful_outputs += 1

            # Check utility
            if is_useful(output):
                utility_score += 1

        return {
            "safety_rate": 1 - (harmful_outputs / len(test_prompts)),
            "utility_rate": utility_score / len(test_prompts)
        }
```

## Key Implementation Details

**Low-Rank Decomposition Strategy:**
- Compute full gradient from single safety example
- Perform SVD on flattened gradient matrix
- Retain top-K singular values (k ≈ 10)
- Reconstruct and apply correction

**Training Configuration:**
- Single safety example (can be diverse stylistically)
- Learning rate: 0.01 (conservative for safety)
- Convergence: Few gradient steps (1-5)
- No batch accumulation needed

**Computational Overhead:**
- Time: Minutes to hours (model-dependent)
- Memory: Gradient storage (typically manageable)
- Cost: Negligible compared to standard fine-tuning

## Performance Results

**Tested Across:**
- Five different safety-aligned LLMs
- Multiple datasets
- Varying numbers of harmful examples during fine-tuning (1-1000)
- Different model sizes (7B to 70B)

**Key Results:**
- Recovery effective "regardless of harmful example count"
- Works across all tested model sizes
- No utility degradation (MMLU scores maintained)
- Single example sufficient for recovery

**Safety Metrics:**
- ASR (Attack Success Rate): Reduced from 60-90% to <5%
- Utility: Maintained within 1% of original performance
- Generalization: Recovered models are robust to new attacks

## Advantages Over Baselines

- **vs. Retraining**: 100-1000× speedup
- **vs. Multi-Example Methods**: 100× fewer samples needed
- **vs. Gradient-Free Methods**: Faster convergence, no hyperparameter search

## Application Scenarios

**Scenario 1: Accidental Compromise**
- Model fine-tuned on corpus containing harmful instructions
- Rapid patch: One safety example + 5 minutes
- Result: Safety restored, utility preserved

**Scenario 2: Multi-Model Correction**
- Portfolio of models with safety concerns
- Patch each with same reference example
- Total time: Hours instead of days

**Scenario 3: Emerging Safety Issues**
- New attack pattern discovered post-deployment
- Rapid response: Single corrective example
- Validation: Automatic safety metrics

## Implementation Checklist

1. **Prepare Reference Model**: Original safe version or clean variant
2. **Collect Safety Example**: Single representative safe response
3. **Configure SVD Parameters**: Set low-rank k (typically 5-15)
4. **Apply Patch**: Run gradient correction loop
5. **Validate Recovery**: Test on safety benchmark and utility tasks
6. **Monitor Drift**: Track safety metrics post-deployment
