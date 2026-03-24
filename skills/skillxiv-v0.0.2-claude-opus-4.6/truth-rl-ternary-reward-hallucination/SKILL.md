---
name: truth-rl-ternary-reward-hallucination
title: "TruthRL: Reducing Hallucinations via Ternary Reward Design"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2509.25760
keywords: [truthfulness, hallucination, RLVR, ternary-reward, verification]
description: "Train LLMs to reduce hallucinations by 28.9% using a ternary reward scheme that explicitly incentivizes abstention (+0) over false claims (-1) while rewarding correct answers (+1). Apply when improving factual reliability is critical and verification signals are available."
---

# TruthRL: Reducing Hallucinations via Ternary Reward Design

TruthRL addresses a fundamental mismatch in accuracy-focused training: optimizing for correct answers inadvertently reinforces hallucinations when models guess incorrectly. By introducing ternary rewards distinguishing correct answers, hallucinations, and abstentions, models learn to admit uncertainty rather than fabricate information.

## Core Architecture

- **Ternary reward design**: +1 (correct), -1 (hallucination/false claim), 0 (abstention)
- **LLM-based verification**: Uses another LLM to verify answer correctness
- **GRPO training framework**: Advantage-based optimization leveraging natural imbalance in rewards
- **Knowledge boundary enforcement**: Models learn what they don't know through reward asymmetry

## Implementation Steps

Setup ternary reward verification system:

```python
# Initialize verification-based reward model
from truth_rl import TruthRLTrainer, TernaryRewardModel

# Create verifier using stronger LLM
verifier = TernaryRewardModel(
    verifier_model="gpt-4o",  # or stronger open model
    correct_reward=1.0,
    hallucination_penalty=-1.0,
    abstention_reward=0.0
)

trainer = TruthRLTrainer(
    model=your_llm,
    verifier=verifier,
    algorithm="GRPO"
)
```

Execute RL training with ternary rewards:

```python
# Training loop with question-answering tasks
for epoch in range(num_epochs):
    # Generate responses (allowing abstention)
    responses = model.generate(
        prompts=questions,
        max_length=512,
        temperature=1.0,
        top_p=0.9
    )

    # Compute ternary rewards
    rewards = verifier.evaluate(
        questions=questions,
        responses=responses,
        allow_abstention=True  # enables "I don't know" responses
    )

    # Update policy using advantage-based GRPO
    loss = trainer.compute_grpo_loss(
        responses=responses,
        rewards=rewards,
        advantage_normalization=True
    )
    loss.backward()
    optimizer.step()
```

## Practical Guidance

**When to use TruthRL:**
- Knowledge-intensive tasks requiring high factual accuracy (QA, retrieval-augmented generation)
- Safety-critical applications where hallucinations pose risks (medical, legal, financial)
- Multi-domain question-answering with diverse knowledge boundaries
- Scenarios where user trust depends on reliability and epistemic honesty

**When NOT to use:**
- Creative tasks where "I don't know" responses reduce utility
- Domains lacking reliable verification signals
- Applications where partial credit matters more than binary correctness
- Real-time systems where verification latency is prohibitive

**Hyperparameter considerations:**
- **Hallucination penalty (-1.0)**: Match in magnitude to correct_reward (+1.0) to enforce symmetry
- **Abstention reward (0.0)**: Keep neutral; higher values encourage excessive abstention
- **Verification model strength**: Use strongest available model; weaker verifiers reduce training signal quality
- **Temperature (1.0)**: Increase to 1.2 if model underdiversifies in responses; decrease to 0.8 for consistency

## Benchmark Performance

TruthRL shows consistent improvements across four benchmarks:
- **CRAG (retrieval)**: 28.9% hallucination reduction
- **NQ, HotpotQA, MuSiQue (QA)**: 21.1% truthfulness improvement
- **Scale invariance**: Gains persist from 3B to 32B parameters

## References

Related work on verification signals and factuality in LLMs builds on prior approaches to supervised correctness optimization.
