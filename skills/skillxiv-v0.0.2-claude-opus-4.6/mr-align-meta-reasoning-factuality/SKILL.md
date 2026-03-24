---
name: mr-align-meta-reasoning-factuality
title: "MR-Align: Meta-Reasoning Informed Factuality Alignment for Large Reasoning Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24794"
keywords: [Reasoning Alignment, Factuality, Meta-Reasoning, Preference Optimization, LLM Reasoning]
description: "Improve factuality in large reasoning models by analyzing reasoning state transitions and reweighting preference optimization signals, suppressing defective reasoning segments while amplifying patterns that lead to factual outputs."
---

# Title: Align Reasoning Trajectories for Factual Correctness

Large reasoning models sometimes identify correct facts during thinking but fail to incorporate them into final answers. MR-Align detects this "reasoning-answer hit gap" by modeling reasoning as transitions between 15 distinct meta-cognitive states (framing, decomposition, verification, etc.). The framework reweights preference optimization based on state transition patterns, amplifying trajectories that consistently lead to factuality.

The key is treating reasoning as a navigable state space, not monolithic text.

## Core Concept

**State-Transition-Aware Preference Optimization**:
- **Meta-Reasoning States**: 15 cognitive patterns grouped into 4 strategy categories
- **Transition Matrices**: Track probability of moving between states in factual vs. defective outputs
- **Implicit Rewards**: Reweight preference signals based on local-to-global transition probabilities
- **Segment-Level Supervision**: Annotate reasoning with fine-grained cognitive labels
- **KTO Adaptation**: Enhanced Kahneman-Tversky Optimization with transition-aware weighting

## Architecture Overview

- **Reasoning Taxonomy**: 15 meta-reasoning states (framing, decomposition, chaining, verification, etc.)
- **Segment Annotation**: LLM-based coarse-to-fine labeling of reasoning text
- **Transition Analysis**: EM algorithm to estimate state transition matrices from paired examples
- **Reward Computation**: Token-level signals reweighted by transition probabilities
- **Preference Optimization**: KTO with segment-aware reward shaping

## Implementation Steps

**1. Define Meta-Reasoning Taxonomy**

Create a taxonomy of cognitive patterns in reasoning.

```python
class MetaReasoningTaxonomy:
    # 4 macro-strategies, 15 total cognitive states
    STRATEGIES = {
        'Meta-Cognitive Regulation': [
            'framing',          # Problem understanding/context setting
            'backtracking',      # Reconsidering previous steps
            'verification'       # Checking intermediate results
        ],
        'Problem-Solving Operations': [
            'decomposition',     # Breaking down complex problems
            'chaining',          # Linking multiple steps
            'transformation'     # Converting problem form
        ],
        'Knowledge Operations': [
            'retrieval',         # Fetching relevant knowledge
            'synthesis',         # Combining information
            'analogy'            # Drawing parallels
        ],
        'Explanatory & Communication': [
            'explanation',       # Clarifying reasoning
            'summarization',     # Condensing information
            'refinement'         # Improving clarity
        ]
    }

    @staticmethod
    def tag_reasoning_segment(text, llm):
        # Use LLM to annotate text with cognitive state
        prompt = f"""Label this reasoning segment with ONE cognitive state from:
        {', '.join(MetaReasoningTaxonomy.STRATEGIES.keys())}

        Segment: {text}
        State:"""
        state = llm.generate(prompt).strip()
        return state
```

**2. Implement Segment-Level Annotation**

Parse reasoning into segments and annotate with cognitive states.

```python
def annotate_reasoning_trajectory(reasoning_text, llm, chunk_size=100):
    # Split into segments (roughly sentences)
    sentences = reasoning_text.split('.')
    segments = [s.strip() + '.' for s in sentences if s.strip()]

    annotations = []
    for segment in segments:
        if len(segment.split()) > 10:  # Only meaningful segments
            state = MetaReasoningTaxonomy.tag_reasoning_segment(segment, llm)
            annotations.append({
                'text': segment,
                'state': state
            })

    return annotations

def create_positive_negative_pairs(correct_example, incorrect_example, llm):
    # Annotate both correct and incorrect reasoning
    correct_anno = annotate_reasoning_trajectory(correct_example['reasoning'], llm)
    incorrect_anno = annotate_reasoning_trajectory(incorrect_example['reasoning'], llm)

    return {
        'positive': correct_anno,
        'negative': incorrect_anno,
        'correct_answer': correct_example['answer'],
        'incorrect_answer': incorrect_example['answer']
    }
```

**3. Estimate State Transition Matrices**

Learn transition probabilities using EM algorithm.

```python
class TransitionMatrixEstimator:
    def __init__(self, states):
        self.states = states
        self.num_states = len(states)
        # Initialize uniform transitions
        self.P_positive = np.ones((self.num_states, self.num_states)) / self.num_states
        self.P_negative = np.ones((self.num_states, self.num_states)) / self.num_states

    def state_to_idx(self, state):
        return self.states.index(state)

    def e_step(self, positive_examples, negative_examples):
        # Compute expected transitions given current P estimates
        positive_transitions = np.zeros((self.num_states, self.num_states))
        negative_transitions = np.zeros((self.num_states, self.num_states))

        for example in positive_examples:
            for i in range(len(example) - 1):
                curr_state = self.state_to_idx(example[i]['state'])
                next_state = self.state_to_idx(example[i + 1]['state'])
                positive_transitions[curr_state, next_state] += 1

        for example in negative_examples:
            for i in range(len(example) - 1):
                curr_state = self.state_to_idx(example[i]['state'])
                next_state = self.state_to_idx(example[i + 1]['state'])
                negative_transitions[curr_state, next_state] += 1

        return positive_transitions, negative_transitions

    def m_step(self, positive_transitions, negative_transitions):
        # Update transition matrices from counts
        self.P_positive = positive_transitions / positive_transitions.sum(axis=1, keepdims=True)
        self.P_negative = negative_transitions / negative_transitions.sum(axis=1, keepdims=True)

    def fit(self, positive_examples, negative_examples, num_iterations=5):
        for _ in range(num_iterations):
            p_trans, n_trans = self.e_step(positive_examples, negative_examples)
            self.m_step(p_trans, n_trans)
```

**4. Compute Transition-Aware Rewards**

Reweight preference optimization based on transition deviation.

```python
def compute_transition_aware_reward(reasoning_trajectory, P_positive, P_negative):
    # For each token, compute how much its transition deviates from global patterns
    state_sequence = [seg['state'] for seg in reasoning_trajectory]

    rewards = []
    for i in range(len(state_sequence) - 1):
        curr_idx = state_to_idx(state_sequence[i])
        next_idx = state_to_idx(state_sequence[i + 1])

        # Probability of this transition in positive/negative examples
        prob_positive = P_positive[curr_idx, next_idx]
        prob_negative = P_negative[curr_idx, next_idx]

        # Reward: how much this transition favors positive over negative
        # Higher prob in positive, lower in negative = higher reward
        transition_reward = prob_positive / (prob_negative + 1e-8)

        # This reward applies to all tokens in the segment
        segment_tokens = len(reasoning_trajectory[i]['text'].split())
        for _ in range(segment_tokens):
            rewards.append(transition_reward)

    return rewards

def apply_transition_aware_preference_optimization(
    positive_reasoning, negative_reasoning,
    P_positive, P_negative,
    model, tokenizer
):
    # Compute transition-aware rewards
    pos_rewards = compute_transition_aware_reward(positive_reasoning, P_positive, P_negative)
    neg_rewards = compute_transition_aware_reward(negative_reasoning, P_positive, P_negative)

    # Tokenize reasoning
    pos_tokens = tokenizer(positive_reasoning_text)
    neg_tokens = tokenizer(negative_reasoning_text)

    # KTO loss with transition-aware weighting
    pos_loss = -pos_rewards.mean() * cross_entropy_loss(model(pos_tokens), pos_tokens)
    neg_loss = neg_rewards.mean() * cross_entropy_loss(model(neg_tokens), neg_tokens)

    return pos_loss + neg_loss
```

## Practical Guidance

**When to Use**:
- Reasoning models where reasoning quality doesn't directly correlate with final answer
- Factuality-critical applications
- Fine-tuning models post-training to improve trustworthiness

**Hyperparameters**:
- num_em_iterations: 5-10 (usually converges quickly)
- segment_min_tokens: 10 (avoid overly granular segmentation)
- transition_reward_exponent: 2.0 (how much to amplify good transitions)

**When NOT to Use**:
- Models where reasoning text isn't available
- Supervised fine-tuning from scratch (use for alignment only)
- Tasks where reasoning steps don't decompose into meta-cognitive patterns

**Pitfalls**:
- **Sparse transition matrices**: With limited examples, EM may estimate unreliable transitions; use Laplace smoothing
- **Mislabeled states**: Automatic annotation errors propagate to transition matrices; validate manually
- **Negative examples quality**: If "negative" examples aren't truly inferior, reward learning fails

**Integration Strategy**: Apply after initial training convergence. Use on 5-10% of data with careful transition matrix validation.

## Reference

arXiv: https://arxiv.org/abs/2510.24794
