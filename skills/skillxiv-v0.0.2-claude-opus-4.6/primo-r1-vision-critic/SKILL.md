---
name: primo-r1-vision-critic
title: "From Passive Observer to Active Critic: RL Elicits Process Reasoning for Robotic Manipulation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.15600"
keywords: [Visual Reasoning, Process Supervision, Robotic Control, Reinforcement Learning, Chain-of-Thought Video]
description: "Transform video multimodal models into active process critics for robotic tasks. Use RL to incentivize explicit reasoning about progress toward goals and anchor reasoning temporally between initial and current states."
---

# PRIMO R1: Active Process Criticism for Robotic Manipulation

Video multimodal language models typically function as passive observers, recognizing events but not evaluatively assessing progress. PRIMO R1 transforms these models into active critics through two key mechanisms: (1) using reinforcement learning to incentivize explicit chain-of-thought reasoning about task progress, and (2) anchoring reasoning temporally by explicitly linking initial and current state images. The result is a 7B model that achieves 67.0% accuracy on robotic failure detection—surpassing closed-source 72B models by 6% and matching models like OpenAI o1 with 50% error reduction on manipulation benchmarks.

The approach combines outcome-based RL with structured temporal context, enabling models to develop genuine process supervision capabilities for long-horizon robot tasks.

## Core Concept

PRIMO R1 transforms passive vision-language models through:

1. **RL-Driven Reasoning** — Use outcome rewards to train explicit chain-of-thought generation
2. **Temporal Anchoring** — Provide structured context linking sequence start to current frame
3. **Progress Estimation** — Train model to generate step-by-step assessments of task progress
4. **Failure Detection** — Use process reasoning to identify error conditions early

The key insight: explicit reasoning about progress, when incentivized by RL, enables models to develop genuine understanding of robot action effects rather than pattern matching.

## Architecture Overview

- **Video Encoder** — Process video frames and extract temporal sequences
- **Temporal Anchor Representation** — Explicit structure linking initial and current states
- **Process Reasoner** — Generate intermediate reasoning steps about progress
- **Progress Estimator** — Predict task completion likelihood at each step
- **Error Detector** — Flag failure conditions based on reasoning
- **Outcome Reward Model** — Ground truth signal for RL training
- **RL Optimizer** — Policy gradient training for reasoning improvement

## Implementation Steps

Start by implementing the temporal anchoring mechanism that structures context.

```python
import torch
import torch.nn as nn
from typing import List

class TemporalAnchorRepresentation:
    """Structured representation linking initial and current video states."""

    def __init__(self, embed_dim=768):
        self.embed_dim = embed_dim

        # Embeddings for key temporal markers
        self.initial_anchor = nn.Parameter(torch.randn(embed_dim))
        self.current_anchor = nn.Parameter(torch.randn(embed_dim))
        self.temporal_marker = nn.Parameter(torch.randn(embed_dim))

    def create_anchored_context(self, initial_frame: torch.Tensor,
                               current_frame: torch.Tensor,
                               intermediate_frames: List[torch.Tensor] = None):
        """
        Create structured context with explicit temporal anchors.

        Args:
            initial_frame: [1, embed_dim] initial state encoding
            current_frame: [1, embed_dim] current state encoding
            intermediate_frames: list of intermediate state encodings
        """
        batch_size = initial_frame.size(0)

        # Initial state marker
        anchored = initial_frame + self.initial_anchor.unsqueeze(0)

        # Process intermediate frames if provided
        if intermediate_frames:
            for i, frame in enumerate(intermediate_frames):
                # Add temporal position encoding
                temporal_pos = (i + 1) / len(intermediate_frames)
                pos_embed = torch.sin(torch.tensor([temporal_pos])) * self.temporal_marker
                anchored = torch.cat([anchored, frame + pos_embed], dim=0)

        # Current state marker
        anchored = torch.cat([anchored,
                             current_frame + self.current_anchor.unsqueeze(0)],
                            dim=0)

        return anchored  # [num_frames, embed_dim]

    def forward(self, video_sequence: torch.Tensor):
        """Process video with temporal anchoring."""
        # Extract initial and current frames
        initial = video_sequence[:, 0, :]  # [batch, embed_dim]
        current = video_sequence[:, -1, :]

        intermediate = [video_sequence[:, i, :] for i in range(1, video_sequence.size(1) - 1)]

        # Create anchored representation
        anchored = self.create_anchored_context(initial, current, intermediate)

        return anchored
```

Now implement the process reasoning component that generates explicit thoughts.

```python
class ProcessReasoner(nn.Module):
    """Generate explicit reasoning about task progress."""

    def __init__(self, embed_dim=768, max_reasoning_steps=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_reasoning_steps = max_reasoning_steps

        # Transformer for reasoning generation
        self.reasoning_encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )

        # Reasoning step generator
        self.step_generator = nn.GRUCell(embed_dim, embed_dim)
        self.step_head = nn.Linear(embed_dim, embed_dim)

        # Progress estimation head
        self.progress_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability 0-1
        )

    def generate_reasoning_chain(self, anchored_context: torch.Tensor,
                                max_steps: int = None) -> (str, List[float]):
        """
        Generate reasoning steps explaining progress.

        Args:
            anchored_context: [num_frames, embed_dim] structured video
            max_steps: max reasoning steps (default: self.max_reasoning_steps)
        """
        if max_steps is None:
            max_steps = self.max_reasoning_steps

        # Encode context
        encoded = self.reasoning_encoder(anchored_context.unsqueeze(0))
        context_repr = encoded.mean(dim=1)  # [1, embed_dim]

        # Generate reasoning steps
        reasoning_steps = []
        progress_scores = []

        hidden = context_repr.squeeze(0)

        for step_idx in range(max_steps):
            # Estimate progress at this step
            progress = self.progress_head(hidden)
            progress_scores.append(progress.item())

            # Generate step embedding
            step_embed = self.step_generator(context_repr, hidden.unsqueeze(0))

            # Decode to text tokens (simplified; real implementation uses language model head)
            step_repr = self.step_head(step_embed)
            reasoning_steps.append(step_repr.detach())

            hidden = step_embed

        return reasoning_steps, progress_scores

    def forward(self, anchored_context: torch.Tensor) -> (List[str], List[float]):
        """Full reasoning generation."""
        reasoning_steps, progress_scores = self.generate_reasoning_chain(
            anchored_context)

        return reasoning_steps, progress_scores
```

Implement the RL training loop that optimizes for process reasoning quality.

```python
import torch.optim as optim

class ProcessSupervisionRL:
    """Train PRIMO R1 with outcome-based reinforcement learning."""

    def __init__(self, model, process_reasoner, embed_dim=768):
        self.model = model
        self.reasoner = process_reasoner
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        self.reasoner_optimizer = optim.AdamW(process_reasoner.parameters(),
                                             lr=1e-5)

    def compute_outcome_reward(self, video: torch.Tensor,
                              ground_truth_result: bool) -> float:
        """Evaluate whether final outcome matches ground truth."""
        # Get final frame from video
        final_frame = video[:, -1, :]

        # Predict outcome
        with torch.no_grad():
            outcome_pred = self.model.predict_outcome(final_frame)

        # Reward: 1 if correct, 0 if incorrect
        reward = 1.0 if (outcome_pred > 0.5) == ground_truth_result else 0.0

        return reward

    def compute_process_reward(self, reasoning_steps: List[torch.Tensor],
                              progress_scores: List[float],
                              ground_truth_trajectory: List[bool]) -> float:
        """Evaluate quality of intermediate reasoning."""
        if not reasoning_steps or not ground_truth_trajectory:
            return 0.0

        # Reward smooth progress estimation
        monotonicity_bonus = 0.0

        for i in range(1, len(progress_scores)):
            if ground_truth_trajectory[i] and not ground_truth_trajectory[i-1]:
                # Correctly increasing progress
                if progress_scores[i] > progress_scores[i-1]:
                    monotonicity_bonus += 0.1

        # Penalty for non-smooth transitions
        for i in range(1, len(progress_scores)):
            jump = abs(progress_scores[i] - progress_scores[i-1])
            if jump > 0.3:  # Large sudden change is suspicious
                monotonicity_bonus -= 0.05

        return monotonicity_bonus

    def training_step(self, video: torch.Tensor,
                     ground_truth_result: bool,
                     ground_truth_trajectory: List[bool] = None):
        """One RL training step."""
        # Prepare anchored context
        temporal_anchor = TemporalAnchorRepresentation()
        initial_frame = video[:, 0, :]
        current_frame = video[:, -1, :]
        anchored = temporal_anchor.create_anchored_context(
            initial_frame, current_frame)

        # Generate reasoning
        reasoning_steps, progress_scores = self.reasoner(anchored)

        # Compute rewards
        outcome_reward = self.compute_outcome_reward(video, ground_truth_result)

        process_reward = 0.0
        if ground_truth_trajectory:
            process_reward = self.compute_process_reward(
                reasoning_steps, progress_scores, ground_truth_trajectory)

        total_reward = 0.7 * outcome_reward + 0.3 * process_reward

        # Policy gradient loss
        # Simplification: use reasoning steps as actions
        reasoning_logprobs = [torch.log(step.norm()) for step in reasoning_steps]
        reasoning_logprob = sum(reasoning_logprobs) / len(reasoning_logprobs)

        loss = -total_reward * reasoning_logprob

        # Update
        self.reasoner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reasoner.parameters(), 1.0)
        self.reasoner_optimizer.step()

        return loss.item(), total_reward

    def train(self, training_videos: List[torch.Tensor],
             ground_truth_results: List[bool],
             ground_truth_trajectories: List[List[bool]] = None,
             num_epochs: int = 10):
        """Full training loop."""
        for epoch in range(num_epochs):
            total_loss = 0
            total_reward = 0

            for i, video in enumerate(training_videos):
                gt_result = ground_truth_results[i]
                gt_traj = ground_truth_trajectories[i] if ground_truth_trajectories else None

                loss, reward = self.training_step(video, gt_result, gt_traj)
                total_loss += loss
                total_reward += reward

            avg_loss = total_loss / len(training_videos)
            avg_reward = total_reward / len(training_videos)

            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"Reward={avg_reward:.3f}")


def evaluate_primo_r1(model, reasoner, test_videos, ground_truth):
    """Benchmark PRIMO R1 on robotic task evaluation."""
    correct = 0
    error_reduction = []

    for video, ground_truth_result in zip(test_videos, ground_truth):
        # Generate process reasoning
        temporal_anchor = TemporalAnchorRepresentation()
        initial = video[:, 0, :]
        current = video[:, -1, :]
        anchored = temporal_anchor.create_anchored_context(initial, current)

        reasoning_steps, progress_scores = reasoner(anchored)

        # Predict outcome
        final_frame = video[:, -1, :]
        prediction = model.predict_outcome(final_frame)

        if (prediction > 0.5) == ground_truth_result:
            correct += 1

    accuracy = correct / len(test_videos)
    print(f"PRIMO R1 Accuracy: {accuracy:.1%}")
    print(f"Target: 67% on RoboFail benchmark")

    return accuracy
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Outcome reward weight 0.7, process reward 0.3; increase process weight if reasoning quality matters
- Maximum reasoning steps 3-5; longer chains may diverge, shorter chains miss details
- Temporal anchor embedding dimension same as model (768-1024); keeps consistent scale
- Use for robotic tasks with clear progression and failure states
- Particularly effective for long-horizon manipulation where intermediate supervision is sparse

**When NOT to use:**
- For very short tasks with clear success/failure (outcome alone suffices)
- When ground truth trajectory annotations are unavailable (process reward can't train)
- For open-ended tasks without clear progress metrics

**Common Pitfalls:**
- RL training diverging due to sparse rewards; use reward shaping and intermediate targets
- Model overfitting to reasoning style rather than task semantics; regularize with trajectory diversity
- Temporal anchors not providing sufficient structure; experiment with different anchor configurations
- Progress scores becoming uniformly high/low; normalize rewards and use baseline subtraction

## Reference

Paper: [From Passive Observer to Active Critic: RL Elicits Process Reasoning for Robotic Manipulation](https://arxiv.org/abs/2603.15600)
