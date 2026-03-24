---
name: entropy-guided-regeneration
title: "ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.14077"
keywords: [multi-turn dialogue, entropy monitoring, context reset, uncertainty detection, prompt consolidation]
description: "Monitor Shannon entropy in LLM token distributions to detect sudden uncertainty spikes, then trigger adaptive prompt consolidation to realign conversation context and improve accuracy by 56.6% and reliability by 35.3%."
---

# Technique: Entropy-Guided Resetting — Uncertainty-Driven Context Consolidation

Multi-turn conversations with LLMs degrade over time as context accumulates and the model loses track of earlier instructions or goals. Rather than treating uncertainty as noise to eliminate, ERGO uses entropy—the model's own uncertainty signal—to detect when the conversation has drifted, then triggers guided context consolidation.

The key insight is that sharp entropy spikes indicate critical moments: when the model suddenly becomes uncertain about what to do next, that's a signal that instructions have become misaligned or context has become incoherent. By monitoring per-token entropy and consolidating context when entropy spikes, ERGO prevents compounding errors while maintaining conversation flow.

## Core Concept

ERGO operates on three principles:
- **Entropy Monitoring**: Track Shannon entropy of next-token probability distributions at each step
- **Spike Detection**: Identify sudden increases in entropy (threshold-based or learned)
- **Prompt Consolidation**: When entropy spikes, synthesize conversation state and reinject clear instructions
- **Adaptive Thresholds**: Learn task-specific entropy baselines during warm-up phase

The method improves multi-turn reasoning tasks where instructions are revealed incrementally by detecting when the model has lost context and automatically recovering.

## Architecture Overview

- **Token-Level Entropy Computation**: Calculate H(P) for each generated token
- **Entropy Baseline Tracker**: Maintain running exponential moving average of baseline entropy
- **Spike Detector**: Compare current entropy to baseline; flag when difference exceeds threshold
- **Context Compressor**: Summarize conversation history to essential information
- **Instruction Injector**: Synthesize high-level task state and goals
- **Prompt Rewriter**: Insert consolidated context + original instructions back into conversation

## Implementation Steps

The core algorithm computes entropy per token and triggers consolidation when entropy spikes. This example shows entropy monitoring and context reset.

```python
import numpy as np
from collections import deque

class EntropyGuidedContextManager:
    """
    Monitor entropy in multi-turn conversations and trigger consolidation
    when the model becomes suddenly uncertain.
    """

    def __init__(
        self,
        entropy_window_size=20,
        baseline_alpha=0.9,
        spike_threshold=0.5,  # Increase in entropy (nats)
        consolidation_interval=10  # Max turns before forced consolidation
    ):
        self.entropy_window = deque(maxlen=entropy_window_size)
        self.baseline_entropy = None
        self.baseline_alpha = baseline_alpha  # EMA decay rate
        self.spike_threshold = spike_threshold
        self.consolidation_interval = consolidation_interval
        self.turn_counter = 0

    def compute_entropy(self, token_logits):
        """
        Compute Shannon entropy over token distribution.
        Args:
            token_logits: shape (vocab_size,) or (batch, vocab_size)
        Returns:
            entropy: scalar or (batch,) in nats
        """
        probs = np.softmax(token_logits, axis=-1)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-10)), axis=-1)
        return entropy

    def should_consolidate(self, token_entropy, conversation_length):
        """
        Detect if context should be consolidated based on entropy spike
        or forced consolidation interval.
        """
        # Track entropy for averaging
        self.entropy_window.append(token_entropy)

        # Initialize baseline on first few tokens
        if self.baseline_entropy is None and len(self.entropy_window) >= 5:
            self.baseline_entropy = np.mean(list(self.entropy_window))

        # Update baseline with EMA
        if len(self.entropy_window) > 5:
            current_avg = np.mean(list(self.entropy_window))
            if self.baseline_entropy is not None:
                self.baseline_entropy = (
                    self.baseline_alpha * self.baseline_entropy +
                    (1 - self.baseline_alpha) * current_avg
                )

        # Detect entropy spike
        if self.baseline_entropy is not None:
            entropy_increase = token_entropy - self.baseline_entropy
            spike_detected = entropy_increase > self.spike_threshold
        else:
            spike_detected = False

        # Force consolidation periodically
        self.turn_counter += 1
        forced_consolidation = (
            self.turn_counter >= self.consolidation_interval
        )

        should_reset = spike_detected or forced_consolidation
        return should_reset, entropy_increase if self.baseline_entropy else 0


def consolidate_conversation_context(
    conversation_history: list,
    original_task_instruction: str,
    summarizer_model
):
    """
    Compress conversation to essential information and reinject task context.
    Args:
        conversation_history: list of (role, text) tuples
        original_task_instruction: original user instruction
        summarizer_model: LLM used to summarize
    Returns:
        consolidated_prompt: string with compressed context
    """
    # Summarize recent conversation
    summary_prompt = f"""
Summarize the following conversation in 2-3 sentences, focusing on:
1. What goal is the user trying to achieve?
2. What progress has been made?
3. What is the next step?

Conversation:
{format_conversation(conversation_history[-10:])}  # Last 10 turns

Summary:
"""
    summary = summarizer_model.generate(summary_prompt, max_tokens=100)

    # Reconstruct high-level state
    state_prompt = f"""
Based on this conversation summary and original task, what is the current state?

Original Task: {original_task_instruction}

Summary: {summary}

Current State (2-3 lines):
"""
    current_state = summarizer_model.generate(state_prompt, max_tokens=80)

    # Inject back into conversation
    consolidated = f"""
<CONTEXT_CONSOLIDATION>
Task: {original_task_instruction}

Progress Summary: {summary}

Current State: {current_state}
</CONTEXT_CONSOLIDATION>

Please continue with the task, keeping the above context in mind.
"""
    return consolidated


def multi_turn_with_entropy_guidance(
    initial_prompt: str,
    user_turns: list,
    model,
    entropy_manager: EntropyGuidedContextManager
):
    """
    Execute multi-turn conversation with entropy-guided context reset.
    """
    conversation = []
    current_context = initial_prompt

    for turn_idx, user_input in enumerate(user_turns):
        # Add user message
        conversation.append(("user", user_input))

        # Generate response with entropy tracking
        response_tokens = []
        for token_idx in range(100):  # Max 100 tokens per turn
            # Get next token with logits
            next_token, logits = model.generate_next(
                current_context,
                return_logits=True
            )

            response_tokens.append(next_token)

            # Compute entropy
            token_entropy = entropy_manager.compute_entropy(logits)

            # Check if consolidation needed
            should_consolidate, entropy_delta = entropy_manager.should_consolidate(
                token_entropy,
                len(conversation)
            )

            if should_consolidate:
                print(f"Entropy spike detected (delta={entropy_delta:.3f}). "
                      f"Consolidating context...")

                # Consolidate and reset
                consolidated = consolidate_conversation_context(
                    conversation,
                    initial_prompt,
                    summarizer_model=model
                )

                current_context = consolidated + " ".join(response_tokens)

        # Add assistant response
        response_text = model.tokenizer.decode(response_tokens)
        conversation.append(("assistant", response_text))
        current_context += response_text + "\n"

    return conversation
```

Entropy thresholds should be calibrated on task-specific validation data. Simple tasks (factual QA) have lower baseline entropy; complex reasoning tasks higher. Start with 0.5 nats and adjust based on false positive rate.

## Practical Guidance

| Task Type | Baseline Entropy | Spike Threshold | Consolidation Interval |
|-----------|-----------------|-----------------|----------------------|
| Factual QA | 1.5-2.0 nats | 0.4 | 15 turns |
| Multi-step reasoning | 2.5-3.5 nats | 0.6 | 10 turns |
| Open-ended dialogue | 3.0-4.0 nats | 0.8 | 12 turns |

**When to Use:**
- Multi-turn conversations with incremental instructions
- Tasks where model context degrades over time (long conversations)
- You need to detect model confusion without explicit verification
- Task has clear entropy baseline you can establish

**When NOT to Use:**
- Single-turn generation (no multi-turn context drift)
- Tasks where entropy naturally spikes (open-ended generation)
- Low-latency requirements (consolidation adds computational cost)
- Conversations where interruptions are disruptive

**Common Pitfalls:**
- Threshold too low → consolidates constantly, adds overhead
- Threshold too high → misses actual confusion spikes
- Not distinguishing between entropy from legitimate diversity vs. actual confusion
- Consolidation that loses important recent context (keep last N turns verbatim)
- Summarizer model introduces errors (use reliable summarization or keep full history)

## Reference

[ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models](https://arxiv.org/abs/2510.14077)
