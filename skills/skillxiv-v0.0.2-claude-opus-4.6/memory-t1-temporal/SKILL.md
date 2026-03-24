---
name: memory-t1-temporal
title: "Memory-T1: RL for Temporal Reasoning in Multi-Session Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.20092
keywords: [reinforcement-learning, temporal-reasoning, memory-retrieval, multi-session]
description: "Enable agents to accurately identify temporally relevant information in long multi-session dialogues through RL-based memory retrieval. Combines coarse-to-fine candidate selection with multi-level temporal consistency rewards—providing dense supervision that disambiguates time expressions and maintains coherence across 128k-token contexts."
---

## Overview

Memory-T1 addresses a critical failure mode in long-context language models: inaccurate temporal reasoning when processing multi-session dialogues. As conversation history grows, models conflate events from different time periods. This framework adds explicit temporal reasoning via RL-trained memory retrieval.

## Core Technique

The key insight is that temporal reasoning requires multi-level supervision: accuracy, evidence grounding, and temporal consistency.

**Coarse-to-Fine Retrieval Strategy:**
Two-phase filtering manages computational complexity while preserving accuracy.

```python
# Two-phase candidate selection for temporal reasoning
class TemporalMemoryRetrieval:
    def __init__(self):
        self.temporal_predictor = TemporalFilter()
        self.bm25_ranker = BM25Ranker()
        self.fine_grained_selector = RLPolicyModel()

    def retrieve_with_temporal_filtering(self, query, dialogue_history):
        """
        Phase 1: Coarse filtering narrows candidate pool
        Phase 2: RL-based fine selection with temporal rewards
        """
        # Phase 1a: Temporal filtering
        # Predict time range of query
        query_time_range = self.temporal_predictor(query)
        # Filter dialogue to sessions within time range
        candidate_sessions = filter_by_timerange(
            dialogue_history, query_time_range
        )

        # Phase 1b: Relevance filtering (BM25)
        relevant_candidates = self.bm25_ranker.rank(
            query, candidate_sessions, top_k=50
        )

        # Phase 2: Fine-grained RL selection
        selected_evidence = self.fine_grained_selector.select_with_policy(
            query, relevant_candidates
        )

        return selected_evidence
```

**Multi-Level Temporal Consistency Reward:**
Three complementary reward signals enable dense supervision on temporal dimensions.

```python
def compute_temporal_consistency_rewards(
    selected_evidence, correct_evidence, query_time
):
    """
    Three-level reward structure for temporal reasoning.
    """
    rewards = {}

    # Reward 1: Accuracy (Answer correctness)
    answer_correct = evaluate_answer_correctness(
        selected_evidence, query_time
    )
    rewards['accuracy'] = float(answer_correct)

    # Reward 2: Evidence Grounding (Right sessions cited)
    # Jaccard similarity between predicted and correct sessions
    predicted_sessions = extract_sessions(selected_evidence)
    correct_sessions = extract_sessions(correct_evidence)
    session_overlap = jaccard_similarity(predicted_sessions, correct_sessions)
    rewards['grounding'] = session_overlap

    # Reward 3: Temporal Consistency (Novel - dual level)
    # 3a. Session-level: chronological proximity to query
    session_temporal_distance = compute_temporal_distance(
        predicted_sessions, query_time
    )
    rewards['temporal_proximity'] = 1.0 / (1.0 + session_temporal_distance)

    # 3b. Utterance-level: time expressions align with range
    utterance_temporal_alignment = compute_utterance_alignment(
        selected_evidence, query_time
    )
    rewards['utterance_alignment'] = utterance_temporal_alignment

    # Composite reward
    total_reward = (
        0.4 * rewards['accuracy'] +
        0.3 * rewards['grounding'] +
        0.15 * rewards['temporal_proximity'] +
        0.15 * rewards['utterance_alignment']
    )

    return total_reward, rewards
```

**RL Training with Sparse-to-Dense Reward:**
GRPO training uses the multi-level reward structure for robust learning.

```python
def train_temporal_memory_policy(
    model, dialogue_dataset, num_epochs
):
    """
    RL training with structured temporal rewards.
    """
    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dialogue_dataset:
            query = batch['query']
            dialogue_history = batch['history']
            correct_evidence = batch['correct_evidence']

            # Forward: retrieve evidence via policy
            selected = model.retrieve(query, dialogue_history)

            # Compute multi-level rewards
            reward, reward_breakdown = compute_temporal_consistency_rewards(
                selected, correct_evidence, batch['query_time']
            )

            # GRPO loss: maximize weighted reward
            log_prob = model.log_probability(selected)
            loss = -reward * log_prob

            total_loss += loss
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}: Loss={total_loss:.4f}")
```

## When to Use This Technique

Use Memory-T1 when:
- Processing long multi-session dialogues
- Temporal reasoning is critical
- Distinguishing between similar events across time periods
- Extended context (128k+ tokens) with temporal structure

## When NOT to Use This Technique

Avoid this approach if:
- Single-session conversations (temporal reasoning unnecessary)
- No temporal dimension in data
- Real-time inference requires fast retrieval
- Dialogue lacks clear time markers

## Implementation Notes

The framework requires:
- Temporal filter to predict query time ranges
- BM25 ranker for relevance pre-filtering
- RL policy model for fine-grained selection
- Multi-level reward computation infrastructure
- GRPO training loop with composite rewards

## Key Performance

- Handles 128k-token contexts with temporal coherence
- Disambiguates time expressions robustly
- Maintains consistency across multi-session dialogues
- Baseline model collapse on long contexts prevented

## References

- Coarse-to-fine memory retrieval for efficiency
- Multi-level temporal consistency rewards
- Temporal filtering and relevance ranking
- RL-based fine-grained selection
- Dense supervision from multi-dimensional rewards
