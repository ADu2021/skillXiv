---
name: dynamic-fine-tuning-sft-rl
title: Dynamic Fine-Tuning - Reward Rectification in SFT
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.05629
keywords: [supervised-fine-tuning, reward-modeling, gradient-rescaling, generalization]
description: "Minimal modification to SFT that dynamically rescales objectives by token probability. Rectifies implicit reward structure to improve generalization comparable to RL while maintaining SFT simplicity."
---

# Dynamic Fine-Tuning: Reward Rectification in SFT

## Core Concept

Dynamic Fine-Tuning (DFT) reveals and corrects a fundamental limitation in standard Supervised Fine-Tuning: the implicit reward structure encoded in SFT loss severely restricts generalization compared to RL approaches. By dynamically rescaling the objective function with token probability, DFT rectifies this underlying reward signal with just a one-line code change, achieving RL-comparable performance while maintaining SFT's simplicity.

## Architecture Overview

- **Probability-Weighted Rescaling**: Scale gradient magnitudes by token generation probability
- **Implicit Reward Correction**: Address the problematic reward structure in standard SFT
- **Minimal Implementation**: Single-line modification to standard training loop
- **Broad Applicability**: Works across math, code, and multimodal domains
- **Theoretical Grounding**: Connects SFT loss to underlying reward structure

## Implementation Steps

### Step 1: Understand the Implicit Reward in Standard SFT

Analyze the problematic reward structure in standard supervised fine-tuning.

```python
import torch
import torch.nn.functional as F

def analyze_standard_sft_reward(model, batch_tokens, target_tokens):
    """
    Analyze implicit reward structure in standard SFT.

    In standard SFT:
    Loss = -log(p_model(target | context))

    This implicitly assumes all tokens have equal importance,
    but tokens with low model probability get larger gradients.

    Args:
        model: Language model
        batch_tokens: Input token IDs [batch, seq_len]
        target_tokens: Target token IDs [batch, seq_len]

    Returns:
        Analysis of reward signals
    """
    # Forward pass
    logits = model(batch_tokens).logits

    # Get model probabilities for target tokens
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

    # Standard SFT loss
    sft_loss = -target_log_probs.mean()

    # Compute gradient magnitudes
    # Gradient w.r.t. loss for each token
    grad_magnitude = -target_log_probs  # Negative because we minimize -log(p)

    # Key insight: tokens with p < 0.5 have higher gradients
    high_gradient_mask = target_log_probs.exp() < 0.5
    low_gradient_mask = target_log_probs.exp() >= 0.5

    print("Standard SFT Reward Analysis:")
    print(f"Avg gradient for p < 0.5: {grad_magnitude[high_gradient_mask].mean():.4f}")
    print(f"Avg gradient for p >= 0.5: {grad_magnitude[low_gradient_mask].mean():.4f}")
    print("Problem: Model learns to ignore likely tokens!")

    return {
        "sft_loss": sft_loss.item(),
        "grad_magnitude": grad_magnitude,
        "target_probs": target_log_probs.exp()
    }
```

### Step 2: Implement Dynamic Fine-Tuning with Probability Weighting

Apply probability-weighted rescaling to rectify the reward structure.

```python
class DynamicFineTuningLoss:
    """
    Dynamically rescaled loss that corrects implicit reward structure.
    """

    def __init__(self, model):
        self.model = model

    def forward(self, batch_tokens, target_tokens, use_dynamic_rescaling=True):
        """
        Compute DFT loss with dynamic probability weighting.

        Args:
            batch_tokens: Input token IDs [batch, seq_len]
            target_tokens: Target token IDs [batch, seq_len]
            use_dynamic_rescaling: Whether to apply probability rescaling

        Returns:
            Loss value and per-token metrics
        """
        # Forward pass
        logits = self.model(batch_tokens).logits

        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather target log probabilities
        target_log_probs = log_probs.gather(
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Standard SFT loss (negative log likelihood)
        sft_loss_per_token = -target_log_probs  # [batch, seq_len]

        if not use_dynamic_rescaling:
            return sft_loss_per_token.mean()

        # Dynamic rescaling: multiply loss by target probability
        # Key insight: p(token) * (-log p(token)) balances gradient signals
        target_probs = target_log_probs.exp()

        # Dynamically rescale: scale by probability
        # This encourages model to learn from difficult (low p) tokens
        # while not over-emphasizing easy (high p) tokens
        dynamic_rescaling_factor = target_probs

        # Rescaled loss
        dft_loss_per_token = sft_loss_per_token * dynamic_rescaling_factor

        return {
            "loss": dft_loss_per_token.mean(),
            "sft_loss": sft_loss_per_token.mean(),
            "per_token_loss": dft_loss_per_token,
            "rescaling_factor": dynamic_rescaling_factor
        }

    def training_step(self, batch):
        """
        Single training step with DFT loss.

        Args:
            batch: Dictionary with 'input_ids' and 'labels' keys

        Returns:
            Loss value
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Forward pass
        loss_dict = self.forward(input_ids, labels, use_dynamic_rescaling=True)

        # Backward
        loss_dict["loss"].backward()

        return loss_dict
```

### Step 3: One-Line Implementation for Existing SFT Code

Show how to integrate DFT into standard training loops with minimal changes.

```python
# STANDARD SFT CODE (before DFT)
def standard_sft_training_loop(model, dataloader, num_epochs=3):
    """Standard supervised fine-tuning."""
    for epoch in range(num_epochs):
        for batch in dataloader:
            logits = model(batch["input_ids"]).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1)
            )
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()


# DYNAMIC FINE-TUNING (after DFT) - Only one line changes!
def dynamic_fine_tuning_loop(model, dataloader, num_epochs=3):
    """SFT with dynamic probability-weighted rescaling."""
    for epoch in range(num_epochs):
        for batch in dataloader:
            logits = model(batch["input_ids"]).logits

            # Get log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Standard cross entropy
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                reduction="none"
            )

            # THIS ONE LINE implements DFT:
            # Rescale by target token probability
            target_probs = log_probs.gather(
                dim=-1,
                index=batch["labels"].unsqueeze(-1)
            ).squeeze(-1).exp()

            # Apply probability weighting
            weighted_loss = (loss * target_probs.view(-1)).mean()

            weighted_loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
```

### Step 4: Theoretical Justification and Empirical Validation

Explain why DFT improves generalization and validate across domains.

```python
def compare_sft_vs_dft_learning_dynamics(
    model_sft,
    model_dft,
    validation_problems
):
    """
    Compare learning dynamics between standard SFT and DFT.

    Args:
        model_sft: Model trained with standard SFT
        model_dft: Model trained with DFT
        validation_problems: Validation dataset

    Returns:
        Comparison metrics
    """
    results = {
        "sft": {"accuracy": 0, "loss": 0, "gradient_variance": 0},
        "dft": {"accuracy": 0, "loss": 0, "gradient_variance": 0}
    }

    for name, model in [("sft", model_sft), ("dft", model_dft)]:
        accuracies = []
        losses = []
        grad_vars = []

        for problem in validation_problems:
            # Forward pass
            output = model.generate(problem["input"], max_length=500)

            # Check correctness
            is_correct = problem["verify_fn"](output)
            accuracies.append(1.0 if is_correct else 0.0)

            # Compute loss
            logits = model(problem["input_ids"]).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                problem["labels"].view(-1)
            )
            losses.append(loss.item())

            # Analyze gradient variance
            loss.backward()
            grad_var = sum(
                (p.grad ** 2).mean().item()
                for p in model.parameters()
                if p.grad is not None
            )
            grad_vars.append(grad_var)
            model.optimizer.zero_grad()

        results[name]["accuracy"] = sum(accuracies) / len(accuracies)
        results[name]["loss"] = sum(losses) / len(losses)
        results[name]["gradient_variance"] = sum(grad_vars) / len(grad_vars)

    return results


def validate_dft_generalization(model, benchmark_suites):
    """
    Validate DFT generalization across different domains.

    Args:
        model: Model trained with DFT
        benchmark_suites: List of domain benchmarks

    Returns:
        Performance metrics per domain
    """
    domain_results = {}

    # Test on math
    math_accuracy = evaluate_on_benchmark(model, benchmark_suites["math"])
    domain_results["math"] = math_accuracy

    # Test on code
    code_accuracy = evaluate_on_benchmark(model, benchmark_suites["code"])
    domain_results["code"] = code_accuracy

    # Test on multimodal
    multimodal_accuracy = evaluate_on_benchmark(model, benchmark_suites["multimodal"])
    domain_results["multimodal"] = multimodal_accuracy

    print("DFT Generalization Results:")
    for domain, acc in domain_results.items():
        print(f"  {domain}: {acc:.2%}")

    return domain_results
```

### Step 5: Integration with Existing Training Infrastructure

Show how to integrate DFT into standard training libraries.

```python
class DFTTrainer:
    """
    Trainer class integrating DFT into standard training loops.
    Compatible with Hugging Face transformers.
    """

    def __init__(self, model, use_dynamic_rescaling=True, rescaling_strategy="probability"):
        self.model = model
        self.use_dynamic_rescaling = use_dynamic_rescaling
        self.rescaling_strategy = rescaling_strategy

    def compute_loss(self, model_output, labels):
        """
        Compute loss with optional DFT.

        Args:
            model_output: Model outputs with logits
            labels: Target token IDs

        Returns:
            Loss value
        """
        logits = model_output.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none"
        )

        if not self.use_dynamic_rescaling:
            return loss.mean()

        if self.rescaling_strategy == "probability":
            # Get target token probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            target_probs = target_log_probs.exp()

            # Apply probability weighting
            weighted_loss = (loss * target_probs.view(-1)).mean()

        elif self.rescaling_strategy == "entropy":
            # Alternative: weight by entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropy_weights = entropy / entropy.max()
            weighted_loss = (loss * entropy_weights.view(-1)).mean()

        return weighted_loss

    def training_step(self, batch):
        """Single training step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"]
        )

        loss = self.compute_loss(outputs, batch["labels"])

        return loss
```

## Practical Guidance

### When to Use DFT

- **Standard SFT training**: Drop-in replacement with minimal code changes
- **Multi-domain fine-tuning**: Math, code, and multimodal tasks
- **Limited computational budget**: Simpler than RL but RL-comparable results
- **Production training**: Already familiar SFT infrastructure, no major changes

### When NOT to Use DFT

- **Already using RL**: DFT targets SFT limitations; RL may be better
- **Simple fine-tuning tasks**: Standard SFT may suffice
- **Extreme domain specialization**: May need task-specific tuning

### Hyperparameter Recommendations

- **Probability rescaling**: Direct scaling by p(token), no additional hyperparameters
- **Loss reduction**: Always use "none" in cross_entropy to enable per-token weighting
- **Learning rate**: Same as standard SFT (no adjustment needed)
- **Batch size**: Same as standard SFT

### Key Insights

The critical insight is recognizing the problematic implicit reward in standard SFT: treating all tokens equally regardless of probability. By weighting loss proportional to token probability, DFT rectifies this reward structure and improves generalization to RL-comparable levels. The one-line implementation makes adoption trivial.

## Reference

**On the Generalization of SFT: A RL Perspective with Reward Rectification** (arXiv:2508.05629)

Reveals how standard SFT encodes a problematic reward structure and proposes dynamic fine-tuning through probability-weighted loss rescaling. Single-line modification achieves RL-comparable generalization across math, code, and multimodal domains.
