---
name: pepo-token-level-multimodal-policy
title: "PEPO: Perception-Exploration Policy Optimization for Multimodal Chain-of-Thought"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22847"
keywords: [Token-Level RL, Perception Prior, Entropy Gating, Vision-Language, Chain-of-Thought]
description: "Replace uniform token-level advantages with perception-exploration gating that weights tokens by visual grounding strength. Adds 3.67 points to geometry reasoning and 5.32 to few-shot classification with <1% compute overhead. Works best for multimodal CoT where visual grounding anchors reasoning. Trigger: When using token-level RL on VLMs and want to emphasize visually-grounded reasoning steps."
category: "Component Innovation"
---

## What This Skill Does

Replace uniform token-level policy optimization with perception-exploration gating that computes token advantages using dual signals: (1) visual alignment strength (cosine similarity to vision tokens) and (2) reasoning uncertainty (output entropy). Improves multimodal chain-of-thought reasoning by emphasizing tokens that are both visually grounded and uncertain.

## Problem with Uniform Token-Level RL

Standard token-level RL (GRPO, DAPO) assigns equal advantage weights to all tokens in the response sequence. This is suboptimal for vision-language models because:
- Not all tokens contribute equally to correct reasoning (some are explanatory filler)
- Some high-RL-loss tokens may be ungrounded hallucinations, not reasoning steps
- Entropy alone doesn't distinguish between productive uncertainty (reasoning) and noise
- Model wastes RL signal on tokens that don't anchor reasoning to visual input

The paper's insight: Successful multimodal CoT depends on a compact subset of visually-aligned tokens that anchor the reasoning process. RL should concentrate advantage weight on these grounded, uncertain tokens.

## The Swap: Uniform Advantages → Perception-Exploration Gating

Replace flat token weighting with perception-prior-gated advantages:

```python
# Uniform token-level advantages (baseline)
def uniform_token_advantages(response_logits, reward):
    """Standard token-level RL: all tokens get same advantage scaling"""
    # Response logits: (seq_len, vocab_size)
    # Compute per-token log probabilities
    log_probs = log_softmax(response_logits, dim=-1)

    # Advantage for each token: uniform scaling of reward
    # All tokens equally encouraged/discouraged based on final reward
    advantages = reward * ones(seq_len)  # Broadcast reward to all tokens
    return advantages

# Perception-exploration gating (proposed)
def perception_exploration_gating(response_logits, response_hidden, vision_hidden, reward):
    """
    Weight tokens by (1) visual grounding + (2) reasoning entropy.
    Only reward tokens that are both grounded and uncertain.
    """
    seq_len, vocab_size = response_logits.shape
    batch_size, num_vision_tokens, hidden_dim = vision_hidden.shape

    # Signal 1: Perception prior (visual alignment)
    # Compute cosine similarity between response tokens and vision tokens
    response_normalized = F.normalize(response_hidden, dim=-1)  # (seq_len, hidden_dim)
    vision_normalized = F.normalize(vision_hidden, dim=-1)  # (batch, num_vision, hidden_dim)

    # Max similarity across vision tokens (find most aligned vision token per response token)
    vision_similarity = torch.matmul(response_normalized, vision_normalized.T)  # (seq_len, num_vision)
    perception_score = vision_similarity.max(dim=-1).values  # (seq_len,) ∈ [0, 1]

    # Signal 2: Exploration entropy (reasoning uncertainty)
    log_probs = log_softmax(response_logits, dim=-1)  # (seq_len, vocab_size)
    entropy = -(exp(log_probs) * log_probs).sum(dim=-1)  # (seq_len,)
    entropy_normalized = entropy / log(vocab_size)  # Normalize to [0, 1]

    # Combine: smooth gating function
    # wt(i) = softmax((1 + α·tanh(g_t(i))) * perception_score)
    # where g_t(i) is entropy-based exploration signal
    alpha = 1.0  # Balance parameter
    combined_gate = (1 + alpha * torch.tanh(entropy_normalized)) * perception_score
    weights = F.softmax(combined_gate, dim=0)  # Normalize across tokens

    # Apply weights to token advantages
    advantages = reward * weights  # Weighted advantage: high for grounded + uncertain tokens
    return advantages, weights
```

Key differences:
- Uniform: all tokens → advantage = reward
- Perception-exploration: tokens weighted by (visual alignment × gating(entropy)) → sparse, targeted advantages

## Performance Impact

**Baseline (Uniform Token-Level RL):**
- Geometry reasoning (Qwen2.5-VL-3B): ~X points
- Few-shot classification (FGVC Aircraft): ~Y points
- Visual grounding: baseline IoU@50

**With Perception-Exploration Gating:**
- Geometry reasoning: +3.67 points (20-30% relative improvement)
- Visual grounding: +0.86 IoU@50 (measurable improvement in bounding box precision)
- Few-shot classification (FGVC Aircraft): +5.32 points (~10% relative)
- Compute overhead: <1% (gating is lightweight: similarity + softmax)

**Ablation findings:**
- Perception prior alone (no entropy): moderate gain (+1-2 points)
- Entropy weighting alone (no vision): minor gain (+0.5-1 point)
- Combined perception + entropy: best results (+3-5 points)
- Scaling with model size: gains persist on 3B, larger models expected to improve further

## When to Use

- Vision-language models doing chain-of-thought reasoning with multimodal input
- Token-level RL frameworks (GRPO, DAPO) applied to VLMs
- Tasks where visual grounding matters (geometry, object reasoning, visual question answering)
- When you want to prevent hallucinations (entropy gate discourages high-entropy false tokens)
- Datasets with diverse reasoning complexity (perception prior adapts to image complexity)

## When NOT to Use

- Pure text-only language models (no vision tokens to ground)
- Tasks where all tokens contribute equally (e.g., structured output with fixed format)
- Models where vision and text are processed independently without cross-attention
- Inference-only scenarios (PEPO is integrated into training loop)
- If compute overhead is critical (though <1% overhead is negligible in most cases)

## Implementation Checklist

To integrate perception-exploration gating:

1. **Extract vision hidden states**:
   - Capture vision tower outputs during forward pass (typically before cross-modal fusion)
   - Shape: (batch_size, num_vision_tokens, hidden_dim)

2. **Implement perception-exploration gating**:
   - Compute perception prior: cosine similarity between response token hidden states and vision hidden states
   - Compute exploration signal: output entropy per token
   - Combine via gating: `(1 + α·tanh(entropy)) * perception_score`
   - Softmax to normalize weights across sequence

3. **Integrate with token-level RL**:
   - Replace uniform advantage computation with gated advantages
   - Formula: `token_advantage[t] = reward * weight[t]`
   - Use standard GRPO/DAPO loss with weighted advantages

4. **Train and evaluate**:
   - Benchmarks: Geometry (ScienceQA), visual grounding (RefCOCO), few-shot (FGVC Aircraft)
   - Metrics: accuracy gain (target +3 points geometry), IoU improvement (target +0.8 on grounding)
   - Validation: Verify perception score is high for grounded tokens, low for hallucinations

5. **Optional tuning**:
   - Gating alpha: 0.5-2.0 (controls entropy weight, higher = more emphasis on uncertain tokens)
   - Perception threshold: If too many tokens weighted equally, lower vision similarity threshold
   - Entropy normalization: Use log(vocab_size) or learned normalization

## Known Issues

- **Vision token extraction**: Some architectures don't preserve vision hidden states. May need to hook intermediate layers.
- **Similarity metric**: Cosine similarity in embedding space may be noisy early in training. Consider applying layer normalization to hidden states.
- **Entropy collapse**: If model becomes overconfident (low entropy everywhere), gating loses effect. Monitor entropy distribution and adjust training schedule.
- **Generalization across vision towers**: If training on one vision encoder, gains may not transfer to different encoders. Validate cross-encoder.
- **Small batch effects**: With small batches, entropy estimates are noisier. Use batch_size >= 8 for stable gating signals.

## Related Work

Builds on token-level RL (GRPO, DAPO from concurrent work) and attention-based visual grounding. Relates to multimodal contrastive learning (CLIP) for computing vision-text similarity. Extends prior work on visual question answering by making the implicit perceptual anchor (grounded tokens) explicit in the RL objective. First work to systematically study token-level perception priors in multimodal CoT.
