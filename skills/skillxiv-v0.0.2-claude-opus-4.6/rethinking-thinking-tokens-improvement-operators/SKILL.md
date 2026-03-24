---
name: rethinking-thinking-tokens-improvement-operators
title: "Rethinking Thinking Tokens: LLMs as Improvement Operators"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.01123
keywords: [reasoning, thinking-tokens, inference-optimization, verifiable-rewards]
description: "Enable longer reasoning within bounded context through iterative refinement: generate solution → verify → compress → refine. Achieves +11% on AIME with lower per-call latency than naive chain-of-thought."
---

# Rethinking Thinking Tokens: LLMs as Improvement Operators

This work reframes long-chain-of-thought generation as iterative improvement rather than sequential tracing. Rather than maintaining long transcripts, models generate solutions, receive feedback, compress insights, and refine—staying within bounded context budgets while achieving deeper reasoning.

## Core Architecture

- **Sequential Refinement (SR)**: Single-pass iterative improvement (generate → verify → compress → refine)
- **Parallel-Distill-Refine (PDR)**: Multi-branch exploration with knowledge distillation
- **Budget separation**: Sequential latency (B_seq) vs. total compute (B_total)
- **Operator consistency**: Training and inference use identical read-write-compress interface

## Implementation Steps

Setup iterative refinement training framework:

```python
# Initialize improvement operator training
from improvement_ops import ImprovementOperatorTrainer, IterativeRefinement

trainer = ImprovementOperatorTrainer(
    model=your_reasoning_llm,
    verification_model="your_verifier",
    max_iterations=3,  # sequential refinement steps
    compression_ratio=0.2  # compress 80% of reasoning
)

# Configure Sequential Refinement strategy
sr_config = IterativeRefinement.Config(
    strategy="sequential",
    phases={
        "generate": {"max_length": 512, "temperature": 0.7},
        "verify": {"mode": "self_verify"},
        "compress": {"target_length": 100},  # 20% of original
        "refine": {"max_length": 512}
    }
)
```

Execute iterative improvement training:

```python
# Training loop with improvement operator RL
for step, batch in enumerate(training_dataloader):
    questions = batch["question"]
    ground_truth = batch["solution"]

    # Iteration 1: Initial solution generation
    solution_1, log_prob_1 = model.generate_with_log_prob(
        prompt=questions,
        max_length=512,
        temperature=0.7
    )

    # Verify first attempt
    is_correct_1 = verifier.check(solution_1, ground_truth)
    reward_1 = 1.0 if is_correct_1 else 0.0

    if not is_correct_1:
        # Iteration 2: Compress and refine
        compressed_insights = model.compress(
            solution=solution_1,
            target_length=100,
            include_errors=True  # learn from failures
        )

        solution_2, log_prob_2 = model.generate_with_log_prob(
            prompt=questions + "\nPrevious attempt:\n" + compressed_insights,
            max_length=512,
            temperature=0.7
        )

        is_correct_2 = verifier.check(solution_2, ground_truth)
        reward_2 = 1.0 if is_correct_2 else 0.0

        if not is_correct_2 and num_iterations < max_iterations:
            # Iteration 3: Second refinement
            compressed_insights_2 = model.compress(
                solution=solution_2,
                previous_insights=compressed_insights,
                target_length=100
            )

            solution_3, log_prob_3 = model.generate_with_log_prob(
                prompt=questions + "\nPrevious attempts:\n" + compressed_insights_2,
                max_length=512,
                temperature=0.7
            )

            is_correct_3 = verifier.check(solution_3, ground_truth)
            reward_3 = 1.0 if is_correct_3 else 0.0

            final_reward = reward_3
            all_log_probs = [log_prob_1, log_prob_2, log_prob_3]
        else:
            final_reward = reward_2
            all_log_probs = [log_prob_1, log_prob_2]
    else:
        final_reward = reward_1
        all_log_probs = [log_prob_1]

    # Compute advantage-based loss (RLVR)
    loss = trainer.compute_improvement_loss(
        log_probs=all_log_probs,
        rewards=final_reward,
        advantage_normalization=True
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Practical Guidance

**When to use Improvement Operators:**
- Reasoning tasks with verifiable correctness (math, logic, geometry)
- Scenarios where iterative refinement might improve accuracy
- Bounded-budget inference where sequential latency matters
- Systems requiring interpretable improvement traces

**When NOT to use:**
- Non-verifiable tasks (creative writing, open-ended reasoning)
- Domains where iterative refinement adds minimal benefit
- Real-time systems where latency critical (compression/refinement overhead)
- Tasks where single-pass reasoning sufficient

**Hyperparameters:**
- **Max iterations (3)**: Increase to 4-5 for very difficult tasks; 2 for speed priority
- **Compression ratio (0.2)**: Keep 20% of original reasoning; test 0.15-0.3
- **Generate temperature (0.7)**: Increase to 0.9 for more diversity; decrease to 0.5 for consistency
- **Compress target length (100)**: Adjust based on solution complexity; keep ≥ 50 tokens
- **Verify threshold**: Early stopping if solution verified; avoid unnecessary refinement

## Budget Accounting

Critical distinction between latency types:
- **B_seq**: Sequential latency (actual wall-clock time for user)
- **B_total**: Total compute budget (sum of all improvements)

```
SR strategy:
  B_seq = 3 * (generation_latency + verification_latency) ≈ 3x single-pass
  B_total = 3 * (generation + compression + verification) = 2-3x single-pass

PDR strategy:
  B_seq ≈ 1.5x single-pass (parallel branches compressed before final refinement)
  B_total = N_branches * (generation + compression) + final_refinement
```

Model accordingly for your latency constraints.

## Performance Improvements

- **AIME 2024**: +11% improvement vs. baseline long CoT
- **Other benchmarks**: +5-8% typical improvement
- **Lower per-call latency**: Better latency-performance tradeoff than naive long CoT

## Architecture Notes

Key insight: "LLMs as improvement operators" inverts typical thinking about chain-of-thought. Rather than maintaining long linear traces, models learn to:
1. Generate initial solution (quick first pass)
2. Recognize failure (verification feedback)
3. Extract key lessons (compression)
4. Refine based on insights (iterative improvement)

This mirrors human expert behavior more closely than linear reasoning.

## Verification Dependency

Success heavily depends on verifier quality. For best results:
- Use strongest available model for verification
- Ensure verification logic matches ground truth exactly
- Test verification on simple examples before full training

## References

Builds on iterative refinement in problem-solving and verifiable reward signals for RL.
