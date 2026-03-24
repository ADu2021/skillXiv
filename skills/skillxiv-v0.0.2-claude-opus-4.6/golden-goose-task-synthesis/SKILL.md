---
name: golden-goose-task-synthesis
title: "Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.22975"
keywords: [Task Synthesis, RLVR, Data Generation, Multiple-Choice, Training Data]
description: "Synthesize unlimited verifiable training tasks from unverifiable text by converting reasoning passages into multiple-choice problems. Creates higher-quality training data for RLHF systems without requiring new human labels."
---

# Golden Goose: Unlimited Task Synthesis from Text

## Problem
Models trained with Reinforcement Learning from Verifiable Rewards (RLVR) saturate quickly on limited, manually-created datasets. Performance plateaus prevent continued improvement on reasoning tasks like mathematics and coding, where verifiable correct/incorrect answers are expensive to generate at scale.

The core challenge is that RLVR requires ground-truth labels for verification—yet most internet text (textbooks, forum posts, research papers) cannot be automatically verified. Traditional RLHF approaches require expensive human feedback, creating a scaling bottleneck.

## Core Concept
Golden Goose converts unverifiable reasoning text into verifiable multiple-choice training problems through fill-in-the-middle masking. The key insight is that intermediate reasoning steps naturally form correct answers, while other plausible continuations become high-quality distractors.

The process extracts reasoning-rich passages, masks crucial internal steps, generates diverse alternatives, and filters problems by difficulty. This transforms passive text into active verification tasks without human intervention.

## Architecture Overview

- **Passage Identification**: Scan corpora (textbooks, AoPS forums, code repositories) for reasoning-rich content with clear intermediate steps
- **Masking Strategy**: Identify contiguous spans of crucial reasoning and replace with [MASK] token, preserving the underlying problem structure
- **Choice Generation**: Retain masked content as ground-truth answer; generate 8 diverse distractors using language model sampling
- **Difficulty Filtering**: Score by self-consistency; remove problems scoring <0.2 or >0.8 confidence to focus on medium-difficulty problems
- **Scale**: Results in 700K+ verifiable problems suitable for standard RL training

## Implementation

### Step 1: Extract Reasoning Passages
Identify passages containing multi-step reasoning by scanning educational texts for sufficient length and step diversity.

```python
import re

def extract_reasoning_passages(documents, min_steps=3, min_length=500):
    """Extract passages likely to contain multiple reasoning steps."""
    passages = []
    for doc in documents:
        # Look for patterns with numbered steps, clear logic flow
        if len(doc) > min_length and doc.count('\n') >= min_steps:
            passages.append(doc)
    return passages
```

### Step 2: Create Fill-in-the-Middle Format
Select a contiguous span of crucial reasoning, remove it, and treat as the ground-truth answer.

```python
def create_masked_problem(passage, model):
    """Convert reasoning passage to masked problem format."""
    # Identify crucial reasoning spans using language model analysis
    important_spans = model.identify_reasoning_spans(passage)

    # Select one span for masking
    span = important_spans[0]
    start, end = span['start'], span['end']

    # Create problem with [MASK]
    problem = passage[:start] + "[MASK]" + passage[end:]
    correct_answer = passage[start:end]

    return problem, correct_answer
```

### Step 3: Generate Diverse Distractors
Sample plausible alternatives from the language model and filter for diversity and plausibility.

```python
def generate_distractors(problem, correct_answer, model, num_distractors=8):
    """Generate diverse, plausible incorrect alternatives."""
    distractors = []

    # Sample continuations conditioned on the problem
    candidates = model.sample_continuations(
        problem,
        num_samples=50,
        temperature=0.8
    )

    # Filter for diversity and quality
    for candidate in candidates:
        if candidate != correct_answer:
            # Check for similarity to existing choices
            if not is_too_similar(candidate, [correct_answer] + distractors):
                distractors.append(candidate)
                if len(distractors) == num_distractors:
                    break

    return distractors

def is_too_similar(text1, candidates, threshold=0.85):
    """Check if text is too similar to existing candidates."""
    for candidate in candidates:
        similarity = compute_similarity(text1, candidate)
        if similarity > threshold:
            return True
    return False
```

### Step 4: Filter by Difficulty
Remove problems that are too easy or too hard to avoid wasting training capacity.

```python
def filter_by_difficulty(problem, choices, model, target_confidence=0.5, tolerance=0.3):
    """Filter problems by difficulty using self-consistency."""
    # Generate multiple solution attempts
    attempts = []
    for _ in range(5):
        response = model.solve_problem(problem)
        attempts.append(response)

    # Calculate confidence as fraction selecting correct answer
    correct_count = sum(1 for a in attempts if choices[0] in a)
    confidence = correct_count / len(attempts)

    # Keep problems with intermediate difficulty
    return abs(confidence - target_confidence) < tolerance
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Min passage length | 500 chars | Filter out trivial examples |
| Num distractors | 9 choices | Standard multiple-choice format |
| Num distractor samples | 50 | Balance quality with efficiency |
| Confidence threshold | [0.2, 0.8] | Reject trivial and impossible problems |
| Diversity threshold | 0.85 similarity | Remove near-duplicate distractors |

### When to Use

- **Continuous RLVR training**: Overcome saturation on fixed datasets by generating new problems daily
- **Scaling data**: Replace expensive human annotation with automated task synthesis
- **Domain adaptation**: Bootstrap training data for new domains using domain-specific text corpora
- **Benchmark construction**: Create diverse evaluation sets from internet text

### When Not to Use

- When human-verified problems are already abundant and inexpensive
- For tasks requiring ground-truth beyond binary correct/incorrect
- When masked passages don't naturally correspond to meaningful reasoning steps
- For open-ended reasoning where multiple valid paths exist equally

### Common Pitfalls

1. **Insufficient masking diversity**: Selecting only the final step leaves context too obvious. Mask multiple reasoning styles to increase difficulty.
2. **Distractor quality**: Distractors that are obviously wrong or semantically identical reduce learning signal. Require both plausibility and diversity.
3. **Difficulty bias**: Filtering threshold must match your model's capability. Too tight filtering wastes data; too loose includes only trivial problems.
4. **Domain mismatch**: Text corpora from one domain may not generalize. Validate on target domain benchmarks.

## Reference
Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text
https://arxiv.org/abs/2601.22975
