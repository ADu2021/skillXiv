---
name: vision-language-reasoning-transfer
title: "Skywork-R1V3 Technical Report: Vision-Language Reasoning Transfer"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06167"
keywords: [Vision-Language Models, Reasoning Transfer, Reinforcement Learning, Multimodal AI, RL Fine-tuning]
description: "Transfer reasoning capabilities from text LLMs to visual domains using reinforcement learning, achieving human-level visual reasoning on complex benchmarks. 38B parameters match closed-source VLMs by optimizing cross-modal connector alignment and entropy-based reasoning signals."
---

# Vision-Language Reasoning Transfer: Teaching Visual Tasks Through RL Knowledge Transfer

Vision-language models typically excel at image understanding but struggle with complex reasoning over visual content—they cannot think step-by-step through challenging spatial, mathematical, or comparative problems shown in images. Skywork-R1V3 solves this by transferring proven reasoning abilities from text-based reasoning models into visual domains through reinforcement learning, enabling models to think methodically about what they see rather than pattern-matching to memorized visual features.

When you need vision-language models to solve MMMU-style problems—college-level mathematics with diagrams, engineering schematics requiring spatial reasoning, or visual logic puzzles—simple scaling or prompt engineering fails. Transferring structured reasoning patterns from text proves far more effective. By treating visual reasoning as a constraint satisfaction problem where intermediate reasoning steps must align with both image content and mathematical rigor, models learn to decompose visual problems systematically rather than guessing.

## Core Concept

Skywork-R1V3 transfers reasoning from text-based LLMs to visual domains through a three-stage RL approach. First, it inherits the reasoning framework—step-by-step decomposition, mathematical proof structures, constraint checking—from a text reasoning model. Second, it uses the connector module (which bridges image encoders to reasoning layers) as the learning focus, allowing the visual backbone to remain stable while improving reasoning-vision alignment. Third, it monitors "entropy of critical reasoning tokens"—positions where the model expresses highest uncertainty about what the image shows—to identify checkpoint quality during training. This entropy signal reveals when the model is learning genuine visual reasoning (high entropy at ambiguous positions) versus memorizing shortcuts (low entropy everywhere).

## Architecture Overview

- **Vision Encoder**: Processes image inputs into visual feature tokens using CLIP or similar architecture
- **Connector Module**: Aligns visual features with reasoning layers, learns cross-modal alignment during RL training
- **Reasoning Backbone**: Text reasoning model adapted for visual inputs, generates step-by-step explanations
- **RL Training Loop**: Uses reward signals from answer correctness, intermediate step validity, and visual grounding
- **Entropy Monitoring**: Tracks uncertainty in critical reasoning tokens to assess learning quality
- **Curriculum Learning**: Progresses from simpler visual reasoning tasks to complex college-level problems

## Implementation

This example demonstrates the RL fine-tuning approach that transfers text reasoning to visual domains. The system monitors entropy of critical reasoning tokens to guide training quality.

```python
# Vision-language model with RL-based reasoning transfer
class VisualReasoningModel:
    def __init__(self, vision_encoder, reasoning_backbone, connector_module):
        self.vision_encoder = vision_encoder  # CLIP or similar
        self.reasoning_backbone = reasoning_backbone  # Inherits from text LLM
        self.connector = connector_module  # Learnable alignment
        self.optimizer = torch.optim.AdamW(self.connector.parameters(), lr=1e-4)

    def forward_with_reasoning(self, image, question):
        """Generate step-by-step reasoning over visual content."""
        # Extract visual features
        visual_features = self.vision_encoder(image)  # Shape: [seq_len, hidden_dim]

        # Align visual features with reasoning space via connector
        aligned_visual = self.connector(visual_features)

        # Generate reasoning tokens incorporating visual grounding
        reasoning_tokens = self.reasoning_backbone(
            input_text=question,
            visual_context=aligned_visual,
            max_length=512
        )

        return reasoning_tokens

    def identify_critical_reasoning_tokens(self, reasoning_tokens, uncertainty_threshold=0.6):
        """Find positions where model is uncertain about image interpretation."""

        # Get token logits and compute entropy
        logits = self.reasoning_backbone.get_logits(reasoning_tokens)
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Filter for tokens at "reasoning decision points"
        decision_mask = torch.zeros_like(entropy, dtype=torch.bool)
        decision_keywords = ['image shows', 'the diagram', 'looking at', 'observe', 'see']

        for i, token in enumerate(reasoning_tokens):
            if any(kw in self.tokenizer.decode([token]) for kw in decision_keywords):
                decision_mask[i] = True

        # Critical positions: high entropy + decision point
        critical_indices = torch.where(
            (entropy > uncertainty_threshold) & decision_mask
        )[0]

        return critical_indices, entropy[critical_indices]
```

This example shows the RL training loop with combined rewards for correctness and visual grounding. The system uses entropy signals to assess checkpoint quality.

```python
def rl_training_step(self, image, question, ground_truth_answer, gold_reasoning_steps):
    """Execute RL fine-tuning with visual grounding rewards."""

    # Generate reasoning trajectory
    reasoning_trajectory = self.forward_with_reasoning(image, question)
    predicted_answer = self._extract_answer(reasoning_trajectory)

    # Reward 1: Answer correctness
    answer_correct = (predicted_answer == ground_truth_answer)
    answer_reward = float(answer_correct)  # 0 or 1

    # Reward 2: Intermediate step validity (does reasoning align with image?)
    step_rewards = []
    for predicted_step, gold_step in zip(reasoning_trajectory, gold_reasoning_steps):
        visual_grounding_score = self._compute_visual_alignment(
            predicted_step, image, question
        )
        step_rewards.append(visual_grounding_score)
    mean_step_reward = sum(step_rewards) / len(step_rewards)

    # Reward 3: Entropy signal for critical reasoning positions
    critical_indices, critical_entropy = self.identify_critical_reasoning_tokens(
        reasoning_trajectory
    )
    entropy_reward = 0.1 * (critical_entropy.mean().item())  # Encourage productive uncertainty

    # Combined reward
    total_reward = (
        0.6 * answer_reward +
        0.3 * mean_step_reward +
        0.1 * entropy_reward
    )

    # Backward pass on connector only (vision encoder stays frozen)
    loss = -total_reward  # Policy gradient: minimize negative reward
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return {
        'answer_reward': answer_reward,
        'step_reward': mean_step_reward,
        'entropy_reward': entropy_reward,
        'total_reward': total_reward
    }
```

This example demonstrates curriculum learning that progressively increases problem difficulty, helping the model transfer reasoning from simple to complex visual domains.

```python
class VisualReasoningCurriculum:
    def __init__(self, model):
        self.model = model
        self.difficulty_stages = [
            'basic_shapes',      # Stage 1: Identify simple shapes
            'spatial_relations', # Stage 2: Relative positions
            'math_diagrams',     # Stage 3: Geometry with calculations
            'complex_scenes',    # Stage 4: Multi-element reasoning
            'college_problems'   # Stage 5: Full MMMU benchmark
        ]
        self.current_stage = 0

    def get_training_batch(self, dataset, batch_size=32):
        """Get batch filtered by current difficulty stage."""
        stage = self.difficulty_stages[self.current_stage]
        filtered_data = dataset.filter_by_difficulty(stage)
        return filtered_data.sample(batch_size)

    def should_advance_stage(self, validation_accuracy):
        """Advance curriculum when performance plateaus at current stage."""
        if validation_accuracy > 0.85:  # 85% threshold
            if self.current_stage < len(self.difficulty_stages) - 1:
                self.current_stage += 1
                print(f"Advancing to stage: {self.difficulty_stages[self.current_stage]}")
                return True
        return False

    def train_epoch(self, train_dataset, val_dataset, num_steps=1000):
        """Train on current curriculum stage until ready to advance."""
        for step in range(num_steps):
            batch = self.get_training_batch(train_dataset)
            for image, question, answer, reasoning in batch:
                reward_dict = self.model.rl_training_step(
                    image, question, answer, reasoning
                )

            if step % 100 == 0:
                val_acc = self._evaluate_on_validation(val_dataset)
                if self.should_advance_stage(val_acc):
                    break
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Connector learning rate | 1e-4 to 5e-5 | Stable cross-modal alignment training |
| Answer reward weight | 0.6 | Primary optimization signal |
| Step validity weight | 0.3 | Encourage interpretable reasoning |
| Entropy reward weight | 0.1 | Monitor learning quality |
| Entropy threshold (critical tokens) | 0.6 | Identifies uncertain reasoning positions |
| Curriculum stage threshold | 0.85 accuracy | Advance when performance plateaus |
| Vision encoder freeze | Yes | Preserve pre-trained visual understanding |

**When to use:** Apply this technique when you have vision-language models that can describe images but struggle with complex reasoning—mathematical diagrams, engineering schematics, spatial logic puzzles, or visual STEM problems. Use when you have paired text reasoning models to transfer from and when reasoning steps can be validated against ground truth.

**When NOT to use:** Don't apply RL transfer if your visual reasoning task is primarily retrieval-based (e.g., "name this object") rather than inference-based. Skip curriculum learning if you have unlimited compute and can train on full-difficulty data from the start. Avoid if you cannot annotate gold reasoning steps for supervision—the method requires step-level labels, not just final answers.

**Common pitfalls:** Training the vision encoder alongside the connector degrades pre-trained visual features—keep it frozen. Using only answer-level rewards without step validity signals produces step-by-step text that doesn't actually ground in images. Entropy thresholds set too high miss important decision points; too low include all tokens. Not starting curriculum learning from simple problems causes the model to give up on complex ones without proper foundation. Ignoring checkpoint quality based on entropy metrics allows training to diverge on low-quality checkpoints.

## Reference

Skywork Team. (2025). Skywork-R1V3 Technical Report: Vision-Language Reasoning Transfer. arXiv preprint arXiv:2507.06167. https://arxiv.org/abs/2507.06167
