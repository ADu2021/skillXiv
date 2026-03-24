---
name: open-vision-reasoner
title: "Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05255"
keywords: [Multimodal Learning, Visual Reasoning, Reinforcement Learning, Transfer Learning]
description: "Transfer reasoning behaviors learned in language models to visual domains through two-stage training: cold-start linguistic fine-tuning followed by multimodal RL. Open-Vision-Reasoner achieves 95.3% on MATH500 and 54.6% on MathVerse by learning visual analogs of backtracking, verification, and subgoal decomposition using rule-based rewards."
---

# Open Vision Reasoner: Transfer Reasoning From Language to Vision

Language models trained with reinforcement learning develop sophisticated reasoning patterns—backtracking when stuck, verifying intermediate steps, decomposing problems into subgoals. These cognitive behaviors are not language-specific; they're general problem-solving strategies. Open Vision Reasoner transfers these patterns to multimodal models by first training extensively on linguistic reasoning, then adapting to visual domains via reinforcement learning with minimal process annotations. The key insight is that linguistic cold-start memorizes diverse reasoning behaviors, while multimodal RL scales up only the patterns effective for vision, achieving 95.3% accuracy on visual math problems with a 7B parameter model.

The approach reveals a fundamental trade-off: linguistic pretraining sometimes initially degrades visual perception (model focuses on text), but multimodal RL recovers this capacity while preserving reasoning sophistication.

## Core Concept

Open Vision Reasoner uses a two-stage training pipeline:

1. **Cold-Start Linguistic Fine-tuning**: Train on 2M text-only reasoning examples (math, code, logic) to embed backtracking, verification, and subgoal decomposition patterns
2. **Multimodal RL Scaling**: Fine-tune on 300K visual reasoning problems using rule-based verifiable rewards (no process annotations needed), scaling up behaviors effective for images

The "Aha Moment" occurs when the model transitions from generic linguistic patterns to visual-specific reasoning (e.g., "visual reflection" = looking at different parts of image, "divide-and-conquer" = solving sub-problems in different image regions).

## Architecture Overview

- **Base Vision-Language Model**: Qwen2.5-VL-7B or similar multimodal backbone
- **Linguistic Reasoning Head**: Attention layer fine-tuned on text reasoning
- **Visual Reasoning Adapter**: LoRA or prompt tuning for vision-specific patterns
- **Rule-Based Reward Model**: Deterministic verifier for outcomes (e.g., comparing predicted answer to ground truth)
- **Process Reward Estimator**: Lightweight classifier scoring reasoning quality per step (learned from outcome labels)
- **Behavior Extraction Module**: Identifies which patterns (backtracking, verification, etc.) activate during reasoning

## Implementation

The following demonstrates the two-stage pipeline and behavior transfer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class ColdStartLinguisticTrainer:
    """Stage 1: Train reasoning patterns on text-only data."""

    @staticmethod
    def create_linguistic_dataset(num_examples: int = 2_000_000) -> List[Dict]:
        """
        Create diverse linguistic reasoning examples.
        In practice, combine datasets: GSM8K, MATH, CodeContests, etc.
        """
        dataset = []
        for i in range(num_examples):
            example = {
                'problem': f"Solve this: ...",  # Text problem
                'reasoning': f"Let me think step by step...",  # Multi-step reasoning
                'answer': f"The answer is ...",
                'reasoning_type': ['backtrack', 'verify', 'decompose'][i % 3]
            }
            dataset.append(example)
        return dataset

    @staticmethod
    def extract_reasoning_pattern(reasoning_text: str) -> str:
        """Identify which cognitive behavior is used: backtrack, verify, decompose."""
        if "actually" in reasoning_text or "wait" in reasoning_text:
            return "backtrack"
        elif "let me check" in reasoning_text or "verify" in reasoning_text:
            return "verify"
        elif "first" in reasoning_text and "then" in reasoning_text:
            return "decompose"
        else:
            return "unknown"

class VisualReasoningAdapter(nn.Module):
    """Lightweight adapter for visual-specific reasoning patterns."""
    def __init__(self, hidden_dim: int = 4096, adapter_dim: int = 256, num_behaviors: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_behaviors = num_behaviors

        # LoRA-style adaptation for visual domains
        self.visual_down = nn.Linear(hidden_dim, adapter_dim)
        self.visual_up = nn.Linear(adapter_dim, hidden_dim)

        # Behavior selector: which reasoning pattern to use
        self.behavior_selector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_behaviors)
        )

        # Behavior-specific transformations
        self.backtrack_head = nn.Linear(hidden_dim, hidden_dim)
        self.verify_head = nn.Linear(hidden_dim, hidden_dim)
        self.decompose_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply visual reasoning adaptation.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) from VLM backbone
            image_features: (batch, num_patches, hidden_dim) visual patch encodings

        Returns:
            adapted_states: (batch, seq_len, hidden_dim)
            behavior_logits: (batch, num_behaviors) behavior probabilities
        """
        # Fuse visual context with text reasoning
        visual_context = image_features.mean(dim=1)  # (batch, hidden_dim)

        # Select which reasoning behavior to apply
        behavior_logits = self.behavior_selector(torch.cat([hidden_states[:, -1], visual_context], dim=-1))

        # Apply selected behavior
        adapted = self.visual_down(hidden_states)

        # Multi-head behavior application (weighted sum)
        behavior_weights = F.softmax(behavior_logits.unsqueeze(1), dim=-1)  # (batch, 1, num_behaviors)
        backtrack_out = self.backtrack_head(hidden_states)
        verify_out = self.verify_head(hidden_states)
        decompose_out = self.decompose_head(hidden_states)

        # Blend behaviors
        blended = (
            behavior_weights[:, :, 0:1] * backtrack_out +
            behavior_weights[:, :, 1:2] * verify_out +
            behavior_weights[:, :, 2:3] * decompose_out
        )

        adapted = self.visual_up(blended)

        return adapted, behavior_logits

class RuleBasedRewardModel(nn.Module):
    """Deterministic reward based on outcome correctness."""
    def __init__(self):
        super().__init__()

    def forward(self, predicted_answer: str, ground_truth: str, reasoning_steps: List[str]) -> Tuple[float, Dict]:
        """
        Compute outcome reward (binary) and estimate process rewards per step.

        Args:
            predicted_answer: Model's final answer
            ground_truth: Correct answer
            reasoning_steps: List of reasoning text for each step

        Returns:
            outcome_reward: 1.0 if correct, 0.0 if incorrect
            process_rewards: Dict mapping step_idx → reward estimate
        """
        # Outcome reward: simple correctness check
        outcome_reward = 1.0 if self._answers_equal(predicted_answer, ground_truth) else 0.0

        # Process rewards: heuristic scoring per step
        process_rewards = {}
        for step_idx, step_text in enumerate(reasoning_steps):
            # Simple heuristics: longer steps, steps with justification get higher scores
            step_length_score = min(len(step_text) / 100, 1.0)  # Normalized length
            has_justification = 1.0 if "because" in step_text.lower() else 0.5
            process_rewards[step_idx] = step_length_score * has_justification

        return outcome_reward, process_rewards

    @staticmethod
    def _answers_equal(pred: str, truth: str) -> bool:
        """Normalize and compare answers."""
        import re
        # Remove punctuation and normalize whitespace
        pred_clean = re.sub(r'[^\w\s]', '', pred.lower().strip())
        truth_clean = re.sub(r'[^\w\s]', '', truth.lower().strip())
        return pred_clean == truth_clean

class OpenVisionReasonerModel(nn.Module):
    """Complete multimodal reasoning model with visual adapter."""
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-VL-7B"):
        super().__init__()

        # Base VLM (frozen backbone, trained in stage 2)
        self.backbone = None  # Load from base_model_name in practice

        # Visual reasoning adapter (trained in stage 2)
        self.visual_adapter = VisualReasoningAdapter(hidden_dim=4096, adapter_dim=256)

        # Reward model (fixed rules, not trained)
        self.reward_model = RuleBasedRewardModel()

    def generate_reasoning_trajectory(self, question: str, image: torch.Tensor,
                                     max_steps: int = 10) -> Tuple[List[str], torch.Tensor]:
        """Generate multi-step reasoning for visual question."""
        # Process image through VLM
        image_features = self.backbone.encode_image(image)

        trajectory = []
        hidden_state = self.backbone.encode_text(question)

        for step in range(max_steps):
            # Adapt to visual domain using behavior selector
            adapted_state, behavior_logits = self.visual_adapter(
                hidden_state.unsqueeze(1), image_features
            )

            # Generate next reasoning step
            next_token = self.backbone.generate_token(adapted_state[:, -1])
            trajectory.append(self._decode_token(next_token))

            # Update hidden state for next iteration
            hidden_state = self.backbone.embed_text(trajectory[-1])

        return trajectory, behavior_logits

    @staticmethod
    def _decode_token(token: torch.Tensor) -> str:
        """Convert token to text."""
        return "placeholder_text"

def train_stage1_linguistic(model, dataset: List[Dict], optimizer, num_epochs: int = 3):
    """Stage 1: Cold-start fine-tuning on linguistic reasoning."""
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, example in enumerate(dataset[:10000]):  # Example: sample 10K per epoch
            optimizer.zero_grad()

            # Forward pass on text
            problem = example['problem']
            reasoning = example['reasoning']
            answer = example['answer']

            # Tokenize and encode
            problem_ids = model.backbone.encode_text(problem)
            target_ids = model.backbone.encode_text(answer)

            # Generate from problem
            logits = model.backbone(problem_ids)

            # Language modeling loss
            loss = criterion(logits.view(-1, model.backbone.vocab_size), target_ids.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Stage 1 Epoch {epoch + 1}: Loss {total_loss / len(dataset[:10000]):.4f}")

def train_stage2_multimodal_rl(model: OpenVisionReasonerModel,
                              dataset: List[Dict],  # Visual reasoning problems
                              optimizer: torch.optim.Optimizer,
                              num_epochs: int = 10,
                              temperature: float = 1.0):
    """Stage 2: Multimodal RL fine-tuning with rule-based rewards."""

    for epoch in range(num_epochs):
        total_policy_loss = 0
        total_reward = 0

        for batch_idx, example in enumerate(dataset[:30000]):  # Sample 30K per epoch
            question = example['question']
            image = example['image']
            ground_truth = example['answer']

            optimizer.zero_grad()

            # Generate reasoning trajectory
            trajectory, behavior_logits = model.generate_reasoning_trajectory(
                question, image, max_steps=10
            )

            # Compute reward signal
            predicted_answer = trajectory[-1]
            outcome_reward, process_rewards = model.reward_model(
                predicted_answer, ground_truth, trajectory
            )

            # Policy loss: maximize reward (REINFORCE-style)
            if outcome_reward > 0:
                # Positive example: increase probability of this trajectory
                log_prob = torch.log(F.softmax(behavior_logits, dim=-1)).sum()
                policy_loss = -log_prob * outcome_reward  # Negative because we maximize
            else:
                # Negative example: decrease probability
                log_prob = torch.log(F.softmax(behavior_logits, dim=-1)).sum()
                policy_loss = -log_prob * 0.1  # Smaller penalty for wrong trajectories

            policy_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_reward += outcome_reward

        avg_reward = total_reward / len(dataset[:30000])
        print(f"Stage 2 Epoch {epoch + 1}: Avg Reward {avg_reward:.4f}, Policy Loss {total_policy_loss:.4f}")

        if avg_reward > 0.5:  # Check for Aha Moment
            print("Aha Moment detected: Model discriminating reasoning patterns!")
```

This implementation demonstrates the core two-stage approach: linguistic pretraining followed by visual RL with behavior-specific adaptation.

## Practical Guidance

| Stage | Key Hyperparameter | Recommendation | Notes |
|-------|-------------------|-----------------|-------|
| **Stage 1 (Linguistic)** | Num examples | 2M | More examples improve pattern diversity |
| **Stage 1** | Epochs | 3-5 | Memorize reasoning patterns well |
| **Stage 2 (Multimodal)** | Visual examples | 300K | Fewer than linguistic; focus on effectiveness |
| **Stage 2** | RL steps | 10-20 | More steps allow complex decomposition |
| **Adapter dimension** | adapter_dim | 256-512 | Balance efficiency and expressiveness |
| **Behavior types** | num_behaviors | 3-5 | Backtrack, Verify, Decompose, (Reflect, Integrate) |

### When to Use Open Vision Reasoner

- **Visual reasoning tasks**: Math problems with diagrams, visual geometry, chart analysis
- **Multi-step image understanding**: Requires decomposition (e.g., "identify all objects, then count")
- **Verification-heavy domains**: Tasks where checking intermediate steps improves accuracy
- **Transfer learning goals**: Leveraging text reasoning to bootstrap vision
- **Interpretability**: Understanding how models approach visual problems
- **Data-efficient learning**: Using RL with rule-based rewards avoids expensive annotations

### When NOT to Use

- **Simple classification tasks**: Single-step recognition doesn't benefit from complex reasoning patterns
- **Real-time inference <100ms**: Reasoning trajectory generation incurs overhead; use single-step models
- **Domains without clear verification**: Rule-based rewards require deterministic answer checking; poorly-defined problems fail
- **Extreme visual complexity** (medical imaging, remote sensing): Rule-based rewards too simplistic; need learned reward models
- **Models <1B parameters**: Overhead of behavior adaptation exceeds model capacity benefits

### Common Pitfalls

1. **Insufficient Stage 1 Data**: Using <1M linguistic examples under-trains reasoning patterns. The model won't know sophisticated backtracking/verification. Use diverse datasets (GSM8K, MATH, CodeContests, CommonsenseQA).
2. **Rule-Based Reward Too Strict**: If reward=0 for any error, model only learns avoiding mistakes, not reasoning. Use soft rewards: deduct 0.5 for arithmetic errors, 0.2 for format errors.
3. **Behavior Selector Collapse**: If model only uses one behavior (e.g., always backtrack), diversity is lost. Add entropy regularization: encourage behavior distribution. Add to loss: 0.1 * entropy(behavior_logits).
4. **Perception-Reasoning Trade-off Ignored**: Stage 1 linguistic training can degrade visual perception initially. Monitor visual accuracy during early Stage 2. If dropping >5%, reduce RL learning rate or balance linguistic + visual loss.
5. **Weak Process Reward Signals**: Simple heuristics (step length, keyword matching) are noisy. Validate with human annotations on 1-2% of data to ensure rewards correlate with quality.

## Reference

Deng, Z., Liu, H., et al. (2025). Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning. *arXiv preprint arXiv:2507.05255*.

Available at: https://arxiv.org/abs/2507.05255
