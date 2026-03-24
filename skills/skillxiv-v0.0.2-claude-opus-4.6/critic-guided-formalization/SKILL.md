---
name: critic-guided-formalization
title: "CriticLean: Critic-Guided Reinforcement Learning for Mathematical Formalization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06181"
keywords: [Theorem Proving, Formal Verification, Reinforcement Learning, Mathematical Reasoning, Semantic Correctness]
description: "Improve formal theorem proofs by treating criticism—evaluation of semantic correctness—as a learning signal. Train critic models to distinguish correct from incorrect formalizations, then use their feedback to guide RL-based proof generation."
---

# CriticLean: Elevating Critique as a Learning Signal for Formal Verification

Translating informal mathematics into formal, executable code (e.g., Lean 4) requires not just generating syntactically correct proofs but ensuring they capture the original mathematical intent. Prior work focused on generation and compilation; CriticLean shifts focus to the critic phase—the evaluation of whether a formalization is semantically correct. By training critic models to assess semantic accuracy and using their feedback as a reinforcement learning signal, CriticLean improves both the quality of generated proofs and the reliability of the evaluation process itself.

The core problem is that compiling without semantic verification produces proofs that are technically valid but miss the mathematical meaning. A proof might compile and be "correct" in isolation yet fail to capture what the original problem asked for.

## Core Concept

CriticLean operates on three interconnected components:

1. **CriticLeanGPT**: A critic model trained via supervised fine-tuning and RL to assess whether a Lean 4 formalization semantically matches a natural language problem
2. **CriticLeanBench**: A benchmark measuring critic ability to distinguish truly correct from subtly incorrect formalizations
3. **FineLeanCorpus**: A dataset of 285,000+ formalization problems with human evaluation of correctness

The framework elevates criticism from a post-hoc filter to an active learning component that guides proof generation toward semantic fidelity.

## Architecture Overview

- **Semantic critic model**: Classifies whether formal code captures mathematical intent
- **Semantic correctness benchmark**: Tests critic on challenging cases where proofs compile but are semantically wrong
- **RL training loop**: Uses critic feedback as reward signal for proof generation
- **Human evaluation dataset**: 285K problems with domain-diverse and human-validated correctness labels
- **Feedback integration**: Critic signals guide generation toward semantically correct proofs

## Implementation

Build the critic model by fine-tuning on semantic correctness judgments:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lean.parser import LeanCodeParser

# Load base model for critic training
critic_base = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b",
    num_labels=2  # Binary: semantically correct or incorrect
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Load dataset with human-verified semantic correctness labels
from criticlean.data import FineLeanCorpus

corpus = FineLeanCorpus(split="train")

def prepare_criticism_example(problem, formal_proof, label):
    """Prepare input for semantic correctness assessment."""

    # Concatenate problem statement and formal proof
    text = f"""
    Problem: {problem['statement']}

    Formal Proof:
    {formal_proof}

    Is this formalization semantically correct? (captures the problem intent)
    """

    encoding = tokenizer(
        text,
        max_length=2048,
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": torch.tensor(label)  # 1 if correct, 0 if incorrect
    }

# Fine-tune critic on semantic correctness
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./critic_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
)

train_dataset = [
    prepare_criticism_example(p, p["proof"], p["semantic_correctness"])
    for p in corpus
]

trainer = Trainer(
    model=critic_base,
    args=training_args,
    train_dataset=train_dataset,
)

critic_model = trainer.train()
print(f"Critic model trained; validation accuracy: {trainer.evaluate()['eval_accuracy']:.2%}")
```

Create a benchmark to assess critic performance on challenging cases:

```python
from criticlean.benchmarks import CriticLeanBench

bench = CriticLeanBench()

# Load test set with pairs: correct proof, subtle incorrect variants
test_pairs = bench.load_challenge_pairs()

# Examples of subtle incorrectness the critic must distinguish:
# 1. Proof is valid but proves slightly different theorem
# 2. Proof uses different assumptions than problem statement
# 3. Proof omits crucial mathematical constraint
# 4. Proof redefines terms ambiguously

critic_predictions = []
ground_truth = []

for problem, correct_proof, incorrect_proof in test_pairs:
    # Evaluate both proofs
    correct_score = evaluate_semantic_correctness(
        critic=critic_model,
        problem=problem,
        proof=correct_proof
    )

    incorrect_score = evaluate_semantic_correctness(
        critic=critic_model,
        problem=problem,
        proof=incorrect_proof
    )

    # Critic should rank correct > incorrect
    critic_correct = correct_score > incorrect_score
    critic_predictions.append(critic_correct)
    ground_truth.append(True)

accuracy = sum(critic_predictions) / len(ground_truth)
print(f"Critic accuracy on challenge pairs: {accuracy:.2%}")
```

Use critic feedback to guide proof generation via reinforcement learning:

```python
from lean.prover import LeanProver
from criticlean.rl import CriticGuidedRL

prover = LeanProver()
rl_trainer = CriticGuidedRL(
    critic_model=critic_model,
    prover=prover
)

def critic_reward(problem, generated_proof):
    """Score proof generation based on semantic correctness."""

    try:
        # Step 1: Does proof compile?
        compilation_result = prover.compile(generated_proof)
        if not compilation_result["success"]:
            return -1.0  # Failed to compile

        # Step 2: Is it semantically correct?
        semantic_score = evaluate_semantic_correctness(
            critic=critic_model,
            problem=problem,
            proof=generated_proof
        )

        # Combine signals: compilation (hard constraint) + semantics (soft signal)
        # Semantic correctness is normalized to [-1, 1]
        return semantic_score

    except Exception:
        return -1.0  # Failure case

# RL training loop
problems = corpus.load_problems(split="train")

for epoch in range(10):
    total_reward = 0

    for problem in problems:
        # Generate multiple proof candidates
        candidates = prover.generate_proof_candidates(
            problem=problem,
            num_candidates=5
        )

        # Score each candidate
        rewards = [critic_reward(problem, cand) for cand in candidates]

        # Update proof generation model based on rewards
        # High reward: this candidate is semantically sound
        # Low reward: regenerate with different strategy
        rl_trainer.update(
            problem=problem,
            candidates=candidates,
            rewards=rewards
        )

        total_reward += max(rewards)

    avg_reward = total_reward / len(problems)
    print(f"Epoch {epoch} average critic reward: {avg_reward:.3f}")
```

## Practical Guidance

### When to Use CriticLean

Use this approach for:
- Generating formal proofs from informal mathematics
- Theorem proving tasks requiring semantic verification
- Building automated verification systems
- Translating research papers into executable Lean code
- Improving quality of machine-generated formal mathematics

### When NOT to Use

Avoid CriticLean for:
- Simple syntactic verification (compilation checks suffice)
- Tasks where semantic correctness is unambiguous
- Domains lacking ground truth for training critics
- Real-time proof generation (RL is slow)
- Problems where multiple valid formalizations exist equally

### Semantic Correctness Categories

The benchmark distinguishes subtle failure modes:

| Category | Example | Criticality |
|----------|---------|------------|
| Compilation mismatch | Proof uses undefined identifier | Critical |
| Assumption divergence | Proof assumes A≥0 but problem didn't state it | High |
| Constraint omission | Proof ignores uniqueness requirement | High |
| Scope drift | Proof proves different theorem entirely | Critical |
| Subtle interpretation | Proof uses non-standard definition | Medium |

### Dataset Characteristics

FineLeanCorpus contains:

| Metric | Value |
|--------|-------|
| Total problems | 285,000+ |
| Domains covered | Math, CS, Logic, Statistics |
| Human annotations | Correctness labels from domain experts |
| Proof variants | Multiple proofs per problem (some incorrect) |
| Difficulty range | Beginner to research-level |

### Training Hyperparameters

| Parameter | Typical Range | Guidance |
|-----------|---------------|----------|
| Critic learning rate | 1e-5 to 5e-5 | Standard; monitor for divergence |
| RL reward discount | 0.99 | Standard for RL |
| Proof generation temperature | 0.7 | Controls diversity of candidates |
| Num candidates per problem | 3-10 | More candidates improve exploration |

### Common Pitfalls

1. **Conflating correctness types**: Compilation ≠ semantic correctness. Don't skip semantic evaluation.
2. **Insufficient critic training**: Critic must be well-calibrated before using as reward signal. Validate on test set first.
3. **Ignoring human evaluation**: Some ambiguous cases require human judgment. Don't over-automate.
4. **Proof distribution mismatch**: If critic is trained on simple proofs but evaluated on complex ones, performance drops. Use curriculum learning.
5. **Forgetting ablation studies**: Always compare: critic-guided RL vs. baseline generation vs. compilation-only filtering.

### Evaluation Protocol

- [ ] Critic is trained on 80% of corpus, evaluated on held-out 20%
- [ ] Critic distinguishes subtle incorrect variants in challenge set
- [ ] RL training uses different problem distribution than critic training
- [ ] Human evaluation confirms critic decisions on ambiguous cases
- [ ] Proof generation without critic serves as baseline

### Critic Failure Analysis

When critic errors occur, categorize:

1. **False negatives**: Rejects semantically correct proofs
   - Solution: Increase critic training data
   - Impact: Misses valid proofs; lowers generation performance

2. **False positives**: Accepts semantically incorrect proofs
   - Solution: Improve critic on this failure mode
   - Impact: Generates invalid proofs; critical error

False positives are more dangerous; prioritize reducing them.

## Reference

"CriticLean: Critic-Guided Reinforcement Learning for Mathematical Formalization" - [arXiv:2507.06181](https://arxiv.org/abs/2507.06181)
