---
name: videoauto-r1-adaptive-video-reasoning
title: "VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05175"
keywords: [Video Understanding, Adaptive Reasoning, Efficiency, Multi-modal LLMs]
description: "Reduce inference latency and token cost in video question-answering by selectively invoking chain-of-thought reasoning. VideoAuto-R1 uses confidence-based early-exit to determine when explicit reasoning is necessary, achieving 3.3× response length reduction while maintaining state-of-the-art accuracy."
---

## When to Use This Skill
- Video understanding tasks with mixed perception and reasoning needs
- Applications requiring efficient inference (reduced token generation)
- Scenarios where reasoning quality matters more than speed
- Tasks combining perception (MVBench) and reasoning (VideoMMMU)
- Real-time video QA with computational constraints

## When NOT to Use This Skill
- All tasks equally require step-by-step reasoning
- Output length minimization is less important than accuracy
- Applications needing guaranteed reasoning traces for auditability
- Domains where skipping reasoning increases error rates significantly

## Problem Summary
Chain-of-thought reasoning in video understanding produces detailed step-by-step analyses, but often underperforms or merely matches direct answering while incurring 2-3× computational overhead. This creates a fundamental inefficiency: perception-heavy tasks (detecting objects, describing scenes) benefit minimally from reasoning, while reasoning-intensive tasks (causal inference, counting across frames) require detailed intermediate steps. Existing approaches treat reasoning as always-on, wasting computation on tasks that don't need it.

## Solution: Think When Necessary Framework

Train models to generate initial answers, evaluate confidence, and only invoke reasoning when necessary.

```python
class VideoAutoR1:
    def __init__(self, base_model):
        self.model = base_model

    def forward_with_adaptive_reasoning(self, video, question):
        """Generate initial answer, then decide on reasoning"""

        # Step 1: Generate initial answer
        initial_logits = self.model.generate_initial_answer(video, question)
        initial_answer = sample_from_logits(initial_logits)

        # Step 2: Compute confidence from initial response
        initial_logprobs = get_token_logprobs(initial_logits)
        # Use length-normalized mean log probability
        confidence_score = initial_logprobs.mean()
        confidence_normalized = confidence_score / sqrt(len(initial_answer))

        # Step 3: Decide on reasoning via threshold
        threshold = 0.97  # Tuned on validation set
        if confidence_normalized > threshold:
            # High confidence: return initial answer directly
            return initial_answer, reasoning_trace=None
        else:
            # Low confidence: invoke explicit reasoning
            reasoning_trace = self.model.generate_reasoning(
                video, question, initial_answer
            )
            # Generate refined answer using reasoning
            refined_answer = self.model.generate_refined_answer(
                video, question, reasoning_trace
            )
            return refined_answer, reasoning_trace=reasoning_trace

    def train_with_dual_reward(self, dataset):
        """Train using reward on both initial and refined answers"""
        for batch in dataset:
            video, question, gold_answer = batch

            # Forward pass: initial answer + reasoning + refined answer
            initial_answer = self.model.generate_initial_answer(video, question)
            reasoning_trace = self.model.generate_reasoning(
                video, question, initial_answer
            )
            refined_answer = self.model.generate_refined_answer(
                video, question, reasoning_trace
            )

            # Dual-answer reward system (GRPO)
            initial_correctness = (initial_answer == gold_answer)
            refined_correctness = (refined_answer == gold_answer)

            # Weight refined answer higher to encourage refinement
            reward_initial = initial_correctness * 0.3
            reward_refined = refined_correctness * 0.7 + fallback_bonus

            # Backward pass with weighted reward
            loss = compute_policy_gradient_loss(
                [reward_initial, reward_refined],
                weights=[0.3, 0.7]
            )
            loss.backward()
```

## Key Implementation Details

**Training Configuration:**
- Base Models: Qwen2.5-VL-7B, Qwen3-VL-8B
- Training Data: 83K curated samples (text, image, video)
- Training Framework: Direct RL (no cold-start SFT)
- Hardware: 32 H100 GPUs for ~35 hours

**Training Approach:**
Dual-answer reward system within GRPO:
- Generate initial answer → reasoning trace → refined answer
- Both answers receive verifiable rewards
- Higher weight on refined answer (0.7) vs. initial (0.3)
- "Fallback bonus" discourages low-confidence guesses on difficult problems

**Data Preprocessing:**
- Remove overly easy samples (99%+ accuracy—no refinement benefit)
- Remove overly hard samples (0% baseline accuracy—noisy reward signal)
- Balance task difficulty to maintain stable GRPO training

**Confidence-Based Early Exit:**
```python
def compute_confidence_score(logits, tokens):
    """Length-normalized mean log probability"""
    logprobs = torch.log_softmax(logits, dim=-1)
    # Gather log probs of generated tokens
    token_logprobs = logprobs[range(len(tokens)), tokens]
    # Normalize by sequence length to prevent length bias
    confidence = token_logprobs.mean() / sqrt(len(tokens))
    return confidence
```

## Performance Results

**Task-Dependent Activation:**
- **Perception tasks (MVBench)**: 25% think-mode activation rate
- **Reasoning tasks (VideoMMMU)**: 51% think-mode activation rate
- Demonstrates automatic task-type detection via confidence estimation

**State-of-the-Art Performance:**
- VideoMME: 71.7% accuracy
- VideoMMMU: 65.0% accuracy
- Maintains/exceeds baselines while reducing computation

**Efficiency Gains:**
- Response length: 149 tokens → 44 tokens (3.3× reduction)
- Selective reasoning on 25-51% of instances
- Maintains quality through weighted dual-reward training

## Advantages Over Baselines

- **vs. Always-On CoT**: 3.3× token reduction on perception-heavy tasks
- **vs. No Reasoning**: Better accuracy on reasoning-intensive tasks
- **vs. Fixed Reasoning Rate**: Adaptive selection matches task difficulty
- **vs. Heuristic Triggers**: Learned confidence from model outputs vs. hand-crafted rules

## Benchmark Performance

**Perception Tasks:**
- Object detection, scene description, temporal understanding
- Minimal benefit from reasoning (25% activation)
- Token savings with maintained accuracy

**Reasoning Tasks:**
- Causal inference, counting, spatial reasoning
- Significant reasoning benefits (51% activation)
- Better accuracy justifies computational investment

## Implementation Checklist

1. **Model Selection**: Use video-capable VLMs (Qwen-VL, LLaVA-Video)
2. **Dataset Preparation**: Curate 83K+ examples with quality/difficulty filtering
3. **Dual-Reward Training**: Implement GRPO with separate initial/refined rewards
4. **Confidence Calibration**: Tune threshold τ on validation set (0.95-0.99)
5. **Inference Mode**: Implement early-exit logic in generation loop
6. **Evaluation**: Measure task-specific activation rates + accuracy
7. **Deployment**: Profile latency gains from reduced token generation
