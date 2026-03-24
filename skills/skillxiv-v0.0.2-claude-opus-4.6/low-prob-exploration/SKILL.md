---
name: low-prob-exploration
title: "Low-Probability Tokens Sustain Exploration in Reasoning RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.03222
keywords: [reinforcement-learning, exploration, entropy-regularization, reasoning, token-filtering]
description: "Preserve reasoning diversity in RL by protecting low-probability tokens that represent novel thinking paths. Trigger: maintain exploration capability during RL training while avoiding entropy collapse on reasoning tasks."
---

# Low-Probability Tokens Sustain Exploration

## Core Concept

During RLVR (RL with Verifiable Rewards) training on reasoning tasks, models suffer from entropy collapse—the policy converges too tightly to common answer patterns, eliminating valuable exploratory reasoning. Low-probability tokens (rare in pretraining) represent genuine thinking diversity but are systematically eliminated during RL. This skill introduces Low-probability Regularization (Lp-Reg): create a proxy distribution filtering noise while protecting "reasoning sparks"—low-probability tokens that enable novel solution paths.

The key insight: Not all entropy is useful exploration; distinguish signal-carrying low-probability tokens from noise, and protect only the signal.

## Architecture Overview

- **Reasoning Sparks Identification**: Detect low-probability tokens that carry reasoning value
- **Proxy Distribution Creation**: Filter out noise while preserving signal
- **Soft Regularization**: KL divergence-based entropy maintenance targeted to valuable tokens
- **Stable On-Policy RL**: Enable continuous scaling without entropy collapse
- **Token-Level Analysis**: Fine-grained understanding of what enables diverse reasoning

## Implementation Steps

### 1. Identify Reasoning Sparks

Analyze which low-probability tokens contribute to diverse reasoning paths.

```python
class ReasoningSparkDetector:
    """
    Identify low-probability tokens that enable novel reasoning.
    """
    def __init__(self, model, reference_model):
        self.model = model
        self.ref_model = reference_model  # Pretrained baseline

    def identify_reasoning_sparks(self, reasoning_traces, solutions, correctness):
        """
        Determine which low-probability tokens lead to correct solutions.

        Args:
            reasoning_traces: List of reasoning token sequences
            solutions: Corresponding final answers
            correctness: Boolean list of correctness

        Returns:
            Set of token IDs that represent reasoning sparks
        """
        spark_candidates = {}

        for trace, solution, is_correct in zip(reasoning_traces, solutions, correctness):
            tokens = tokenize(trace)

            for token_id in tokens:
                token_prob_pretrain = self.ref_model.get_token_prob(token_id)

                # Low-probability tokens: prob < 0.1
                if token_prob_pretrain < 0.1:
                    if token_id not in spark_candidates:
                        spark_candidates[token_id] = {
                            "correct_count": 0,
                            "total_count": 0,
                            "solutions": []
                        }

                    spark_candidates[token_id]["total_count"] += 1
                    spark_candidates[token_id]["solutions"].append(solution)

                    if is_correct:
                        spark_candidates[token_id]["correct_count"] += 1

        # Filter: keep tokens with high correctness rate
        reasoning_sparks = set()
        for token_id, stats in spark_candidates.items():
            correctness_rate = stats["correct_count"] / (stats["total_count"] + 1e-8)

            # Keep tokens that appear in correct solutions >50% of the time
            if correctness_rate > 0.5 and stats["total_count"] > 3:
                reasoning_sparks.add(token_id)

        return reasoning_sparks
```

### 2. Create Proxy Distribution

Build a filtered probability distribution that removes noise while preserving reasoning sparks.

```python
def create_proxy_distribution(logits, reasoning_sparks, noise_threshold=0.01):
    """
    Create proxy distribution: filter noise, amplify reasoning sparks.

    Args:
        logits: Raw model logits (vocab_size,)
        reasoning_sparks: Set of token IDs to protect
        noise_threshold: Probability below which tokens are filtered

    Returns:
        Filtered probability distribution
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Create mask for tokens to keep
    keep_mask = torch.zeros_like(probs)

    for token_id in reasoning_sparks:
        keep_mask[token_id] = 1.0

    # Remove low-probability noise (except protected sparks)
    for token_id in range(len(probs)):
        if token_id not in reasoning_sparks and probs[token_id] < noise_threshold:
            probs[token_id] = 0.0

    # Re-normalize to valid probability distribution
    proxy_dist = probs / (torch.sum(probs) + 1e-8)

    # Amplify reasoning sparks slightly
    for token_id in reasoning_sparks:
        proxy_dist[token_id] *= 1.2  # 20% boost

    # Final normalization
    proxy_dist = proxy_dist / (torch.sum(proxy_dist) + 1e-8)

    return proxy_dist
```

### 3. Implement Low-Probability Regularization (Lp-Reg)

Apply soft KL regularization using the proxy distribution.

```python
class LowProbabilityRegularization:
    """
    Maintain entropy using low-probability token protection.
    """
    def __init__(self, reference_model):
        self.ref_model = reference_model

    def compute_lp_reg_loss(
        self,
        model_dist,
        proxy_dist,
        reasoning_sparks,
        temperature=1.0,
        reg_weight=0.1
    ):
        """
        Compute regularization loss protecting reasoning sparks.

        Args:
            model_dist: Current model's probability distribution
            proxy_dist: Filtered proxy distribution
            reasoning_sparks: Protected token set
            temperature: Distribution sharpness
            reg_weight: Regularization strength

        Returns:
            Scalar regularization loss
        """
        # Apply temperature to soften distributions
        model_dist_soft = model_dist ** (1 / temperature)
        model_dist_soft = model_dist_soft / torch.sum(model_dist_soft)

        proxy_dist_soft = proxy_dist ** (1 / temperature)
        proxy_dist_soft = proxy_dist_soft / torch.sum(proxy_dist_soft)

        # KL divergence: encourage model to match proxy distribution
        # But only for tokens related to reasoning sparks
        kl_loss = 0

        for token_id in reasoning_sparks:
            kl = model_dist_soft[token_id] * (
                torch.log(model_dist_soft[token_id] + 1e-8) -
                torch.log(proxy_dist_soft[token_id] + 1e-8)
            )
            kl_loss += kl

        # Mild regularization: don't fully constrain, allow some flexibility
        regularization = reg_weight * kl_loss

        return regularization
```

### 4. Full RL Loop with Lp-Reg

Integrate regularization into standard RL training.

```python
def train_with_lp_regularization(
    model,
    reference_model,
    dataset,
    config
):
    """
    RL training with low-probability token protection.
    """
    # Phase 1: Identify reasoning sparks
    print("Phase 1: Identifying reasoning sparks")

    # Collect initial rollouts
    rollouts = []
    for problem in dataset[:1000]:  # Sample for spark detection
        trace = model.generate(problem, max_tokens=500)
        solution = extract_solution(trace)
        is_correct = evaluate_correctness(solution, problem)
        rollouts.append({
            "trace": trace,
            "solution": solution,
            "is_correct": is_correct
        })

    detector = ReasoningSparkDetector(model, reference_model)
    reasoning_sparks = detector.identify_reasoning_sparks(
        [r["trace"] for r in rollouts],
        [r["solution"] for r in rollouts],
        [r["is_correct"] for r in rollouts]
    )
    print(f"Identified {len(reasoning_sparks)} reasoning sparks")

    # Phase 2: RL training with Lp-Reg
    print("\nPhase 2: RL training with low-probability regularization")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lp_regularizer = LowProbabilityRegularization(reference_model)

    for epoch in range(config.num_epochs):
        epoch_loss = 0

        for batch_idx, problem in enumerate(dataset):
            # Generate reasoning with current policy
            trace = model.generate(problem, max_tokens=500)
            solution = extract_solution(trace)

            # Evaluate reward
            is_correct = evaluate_correctness(solution, problem)
            reward = 1.0 if is_correct else -1.0

            # Get model logits and probabilities
            logits = model.get_logits_for_sequence(trace)
            model_dist = torch.softmax(logits, dim=-1)

            # Create proxy distribution for this sequence
            proxy_dist = create_proxy_distribution(
                logits,
                reasoning_sparks,
                noise_threshold=0.01
            )

            # Policy gradient loss
            log_prob = model.compute_log_prob(trace)
            pg_loss = -reward * log_prob

            # Lp-Reg loss: protect reasoning sparks
            reg_loss = lp_regularizer.compute_lp_reg_loss(
                model_dist,
                proxy_dist,
                reasoning_sparks,
                temperature=1.0,
                reg_weight=0.1
            )

            # Total loss: RL + regularization
            total_loss = pg_loss + reg_loss

            # Update
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch {epoch}, Batch {batch_idx}: loss={avg_loss:.4f}")

    return model
```

### 5. Evaluation: Stability and Performance

Assess that model maintains exploration while improving correctness.

```python
def evaluate_lp_reg_training(model, benchmark_dataset, reference_model):
    """
    Evaluate exploration stability and performance.
    """
    results = {
        "accuracy": 0,
        "entropy": 0,
        "diversity": 0
    }

    correct = 0
    entropies = []
    solutions_set = set()

    for problem in benchmark_dataset:
        # Generate multiple samples to assess diversity
        traces = []
        for _ in range(5):
            trace = model.generate(problem, max_tokens=500, temperature=0.8)
            traces.append(trace)
            solution = extract_solution(trace)
            solutions_set.add(solution)

            # Check correctness
            if evaluate_correctness(solution, problem):
                correct += 1

        # Compute entropy of solutions for this problem
        solution_entropy = compute_entropy([extract_solution(t) for t in traces])
        entropies.append(solution_entropy)

    results["accuracy"] = correct / (len(benchmark_dataset) * 5) * 100
    results["entropy"] = np.mean(entropies)
    results["diversity"] = len(solutions_set)

    print(f"Accuracy: {results['accuracy']:.1f}%")
    print(f"Entropy: {results['entropy']:.3f}")
    print(f"Solution diversity: {results['diversity']}")

    return results
```

## Practical Guidance

**Hyperparameters:**
- **Noise threshold**: 0.01 (filter very rare tokens)
- **Spark correctness threshold**: 0.5 (keep tokens in correct solutions >50%)
- **Regularization weight**: 0.1 (soft constraint)
- **Temperature for soft distributions**: 1.0-2.0 (sharpness control)
- **Spark detection sample size**: 1000+ problems

**When to Use:**
- RL training on reasoning tasks (math, code, logic)
- Experiencing entropy collapse during training
- Want to preserve diverse solution paths
- Tasks have multiple valid approaches

**When NOT to Use:**
- Single-solution tasks where diversity not valued
- Simple supervised learning (not RL)
- Streaming inference where entropy control not critical
- Tasks where rare tokens are pure noise

## Reference

[Low-Probability Tokens Sustain Exploration in Reasoning RL](https://arxiv.org/abs/2510.03222) — arXiv:2510.03222
