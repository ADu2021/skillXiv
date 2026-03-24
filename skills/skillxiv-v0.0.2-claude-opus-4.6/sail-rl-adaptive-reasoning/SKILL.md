---
name: sail-rl-adaptive-reasoning
title: "SAIL-RL: Guiding MLLMs in When and How to Think via Dual-Reward RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.02280"
keywords: [Multimodal RL, Thinking Control, Adaptive Reasoning, Reward Shaping, Post-Training]
description: "Teach multimodal models to determine when deep reasoning is necessary and how to reason effectively through dual-reward reinforcement learning, preventing both overthinking on simple tasks and underthinking on complex ones while reducing hallucinations."
---

# Title: Learn Adaptive Reasoning Allocation Through Dual Reward Signals

Reasoning enhances accuracy but costs tokens and latency. SAIL-RL teaches models to reason selectively: answer simple questions directly, think deeply on complex ones. The framework uses two complementary rewards: (1) thinking reward evaluates reasoning quality (factual grounding, logical coherence), and (2) judging reward determines task complexity. Combined through cascading logic, these rewards create a system that learns when thinking helps and when it hurts.

The approach balances accuracy against efficiency at inference time.

## Core Concept

**Dual-Reward Reasoning Control**:
- **Thinking Reward**: Evaluates reasoning quality (factual, coherent, consistent)
- **Judging Reward**: Determines when to apply reasoning vs. direct answers
- **Cascading Logic**: Combined multiplicatively, nullifying bad components
- **Discrete Rewards**: Binary signals provide sharper gradients than continuous scores
- **Multimodal Integration**: Works with text and images, maintaining visual grounding

## Architecture Overview

- **Judge Module**: Determines task complexity (simple vs. complex)
- **Thinking Module**: Generates reasoning (if judge says necessary)
- **Answer Module**: Produces final output
- **Reward Computation**: Three-component reward (judge × think × answer)
- **DAPO Training**: Discrete preference optimization with binary rewards

## Implementation Steps

**1. Implement Judge-Think-Answer Architecture**

Structure model to output decision → reasoning → answer.

```python
class JudgeThinkerAnswerer(nn.Module):
    def __init__(self, vlm_model):
        self.vlm = vlm_model
        self.judge = nn.Linear(vlm_model.hidden_dim, 2)  # simple or complex
        self.thinker = ThinkingModule(vlm_model)
        self.answerer = AnswerModule(vlm_model)

    def forward(self, image, text):
        # Encode input
        features = self.vlm.encode(image=image, text=text)

        # Judge: determine complexity
        judge_logits = self.judge(features)
        judge_probs = F.softmax(judge_logits, dim=-1)
        is_complex = judge_probs[:, 1] > 0.5  # Complex if prob > 0.5

        # Conditional thinking
        thinking = []
        for i, complex_flag in enumerate(is_complex):
            if complex_flag:
                thought = self.thinker(features[i:i+1])
                thinking.append(thought)
            else:
                thinking.append("")

        # Answer with or without reasoning
        answers = []
        for i, thought in enumerate(thinking):
            if thought:
                # Use reasoning
                answer = self.answerer(features[i:i+1], context=thought)
            else:
                # Direct answer
                answer = self.answerer(features[i:i+1], context=None)
            answers.append(answer)

        return {
            'judge_decision': is_complex,
            'thinking': thinking,
            'answers': answers
        }
```

**2. Implement Multi-Component Reward System**

Create rewards for thinking quality, judgment accuracy, and answer correctness.

```python
class DualRewardComputer:
    def __init__(self, reward_model):
        self.reward_model = reward_model

    def compute_thinking_reward(self, thinking, image, answer, ground_truth):
        """Evaluate reasoning quality"""
        if not thinking:
            return 1.0  # No reasoning needed is valid

        # Check factual grounding
        factual_score = self.reward_model.score_factuality(thinking, image)

        # Check logical coherence
        coherence_score = self.reward_model.score_coherence(thinking)

        # Check answer consistency with reasoning
        consistency_score = self.reward_model.score_consistency(thinking, answer)

        thinking_reward = (factual_score + coherence_score + consistency_score) / 3
        return thinking_reward

    def compute_judging_reward(self, judge_decision, thinking, answer, ground_truth):
        """Reward correct meta-cognition"""
        answer_correct = answer == ground_truth

        # Optimal: complex problems have reasoning, simple don't
        has_reasoning = len(thinking) > 0 if isinstance(thinking, str) else thinking is not None

        # Infer actual problem difficulty from correctness
        actual_difficulty = 1.0 if not answer_correct else 0.0

        # Reward if judgment matched actual difficulty
        judgment_correct = (actual_difficulty > 0.5) == judge_decision

        judging_reward = 1.0 if judgment_correct else 0.0
        return judging_reward

    def compute_answer_reward(self, answer, ground_truth):
        """Reward answer correctness"""
        correct = answer == ground_truth
        return 1.0 if correct else 0.0

    def compute_cascading_reward(self, thinking, judge_decision, answer,
                                 image, ground_truth, alpha=0.9):
        """Combine rewards with cascading logic"""
        thinking_r = self.compute_thinking_reward(thinking, image, answer, ground_truth)
        judge_r = self.compute_judging_reward(judge_decision, thinking, answer, ground_truth)
        answer_r = self.compute_answer_reward(answer, ground_truth)

        # Multiplicative cascade: all must be good
        format_r = 1.0 if self.check_format(thinking, answer) else 0.0

        cascading = (thinking_r * judge_r * answer_r) + (1 - alpha) * format_r

        return {
            'total': cascading,
            'thinking': thinking_r,
            'judging': judge_r,
            'answer': answer_r,
            'format': format_r
        }
```

**3. Implement Supervised Fine-Tuning Stage**

Pre-train with behavioral cloning before RL.

```python
def sft_stage(model, sft_dataset, num_epochs=3):
    """Supervised fine-tuning on judge-think-answer format"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        for batch in sft_dataset:
            images, texts, judge_labels, thinking_texts, answers = batch

            # Forward pass
            output = model(image=images, text=texts)

            # SFT losses
            judge_loss = F.cross_entropy(
                output['judge_logits'], judge_labels
            )

            # Think loss (if complex problem)
            think_loss = 0
            for i, should_think in enumerate(judge_labels):
                if should_think:
                    think_loss += F.cross_entropy(
                        model.thinker.get_logits(output['thinking'][i]),
                        thinking_texts[i]
                    )
            think_loss /= max(sum(judge_labels), 1)

            # Answer loss
            answer_loss = F.cross_entropy(
                output['answer_logits'], answers
            )

            total_loss = judge_loss + think_loss + answer_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

**4. Implement DAPO Training**

Optimize using discrete preference optimization on binary rewards.

```python
def dapo_training(model, rl_dataset, num_steps=10000):
    """Discrete preference optimization with cascading rewards"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    reward_computer = DualRewardComputer(reward_model)

    for step in range(num_steps):
        batch = rl_dataset.sample_batch()
        images, texts, answers_gt = batch

        # Generate outputs from model
        outputs = model(image=images, text=texts)

        # Compute cascading rewards
        rewards_batch = []
        for i, output in enumerate(outputs):
            reward = reward_computer.compute_cascading_reward(
                thinking=output['thinking'][i],
                judge_decision=output['judge_decision'][i],
                answer=output['answers'][i],
                image=images[i],
                ground_truth=answers_gt[i]
            )
            rewards_batch.append(reward['total'])

        # DAPO: convert continuous rewards to binary preferences
        rewards_tensor = torch.tensor(rewards_batch)
        sorted_indices = torch.argsort(rewards_tensor, descending=True)

        # High-reward group (positive) vs low-reward group (negative)
        num_positive = len(rewards_batch) // 2
        positive_indices = sorted_indices[:num_positive]
        negative_indices = sorted_indices[num_positive:]

        # Preference loss: maximize likelihood of positive, minimize negative
        positive_outputs = [outputs[i] for i in positive_indices]
        negative_outputs = [outputs[i] for i in negative_indices]

        positive_log_probs = model.get_log_prob(positive_outputs)
        negative_log_probs = model.get_log_prob(negative_outputs)

        loss = -positive_log_probs.mean() + negative_log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            avg_reward = rewards_tensor.mean().item()
            print(f"Step {step}: Avg Reward {avg_reward:.3f}")
```

## Practical Guidance

**When to Use**:
- Post-training for reasoning models (after SFT)
- Reducing hallucinations by selective reasoning
- Inference-cost optimization (avoid unnecessary reasoning)
- Multimodal tasks mixing simple and complex problems

**Hyperparameters**:
- alpha (format weight): 0.9 (heavy weight on correctness)
- rl_dataset_size: 70K for STEM + 20K general QA (typical)
- learning_rate: 1e-5 (lower than SFT)

**When NOT to Use**:
- Tasks where reasoning is always necessary
- Models without clear judge/think/answer structure
- Domains where ground truth is expensive to annotate

**Pitfalls**:
- **Reward hackingthrough judge**: Model learns to always output "complex" to use reasoning; mitigate with format penalties
- **Underdeveloped thinker**: If thinking module never trains (judge always says "simple"), it remains weak; warm-start with SFT
- **Cascading failure**: If any component fails, total reward becomes zero; make individual rewards more robust

**Integration Strategy**: Apply after basic SFT convergence. Use 10-20% of compute budget for RL fine-tuning. Monitor thinking rate during evaluation (should be 40-70% on typical datasets).

## Reference

arXiv: https://arxiv.org/abs/2511.02280
