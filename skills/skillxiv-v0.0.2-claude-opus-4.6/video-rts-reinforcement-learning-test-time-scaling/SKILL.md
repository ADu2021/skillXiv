---
name: video-rts-reinforcement-learning-test-time-scaling
title: "Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Video Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06485"
keywords: [Video Reasoning, Reinforcement Learning, Test-Time Scaling, Data Efficiency, Sparse-to-Dense Sampling]
description: "Train video QA models on 6K examples using pure RL instead of costly supervised fine-tuning, then adaptively sample video frames at inference by monitoring answer consensus across multiple reasoning trajectories."
---

# Video-RTS: Data-Efficient Video Reasoning with Adaptive Test-Time Computation

Most video reasoning approaches waste resources: they collect massive labeled datasets and ignore compute reuse at inference. Video-RTS flips this paradigm. By using reinforcement learning directly on small datasets and adapting frame sampling density based on output consistency, the method achieves performance comparable to models trained on 169K examples using only 6K examples.

The key insight is two-fold: (1) outcome-based rewards from RL eliminate the need for expensive supervised reasoning annotations, and (2) at inference, you can cheaply determine when you have seen enough frames by checking if multiple model samples agree on the answer.

## Core Concept

Video-RTS operates in two phases. During training, it applies Group Relative Policy Optimization (GRPO) with simple reward signals: accuracy (did you get the right answer?) and format (is your reasoning properly structured?). No need to label full reasoning chains. During inference, it starts with sparse frame sampling (32 frames for an hour-long video) and gradually densifies until multiple reasoning trajectories converge on the same answer. This voting mechanism stops computation when confident, avoiding wasteful processing of redundant frames.

## Architecture Overview

- **Base Model**: Qwen-2.5-VL-7B vision-language model for video understanding
- **GRPO Training**: Group Relative Policy Optimization with dual rewards (accuracy + formatting)
- **Sparse Frame Sampling**: Initially extract 32 uniformly spaced frames from video
- **Trajectory Ensemble**: Generate multiple reasoning paths with varied frame selections
- **Consensus Voting**: Check if model outputs agree; densify frames only if uncertain
- **Adaptive Stopping**: Halt when majority vote is reached or frame budget exhausted

## Implementation

### Step 1: Prepare Video QA Dataset with Outcome Labels

Collect video QA examples with only question, video, and ground truth answer labels. No reasoning annotations needed:

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset

# Load lightweight dataset (6K examples)
dataset = load_dataset("video_qa", split="train")

# Example structure: only question, video, and answer
example = {
    "video": "path/to/video.mp4",
    "question": "What sport is being played?",
    "answer": "basketball"  # Ground truth only
}

# Initialize Qwen-2.5-VL-7B
model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Model outputs chains of thought + final answer automatically
# GRPO will reward correct final answers and well-formatted outputs
```

### Step 2: GRPO Training - Pure RL Approach

Train using Group Relative Policy Optimization, which compares outputs within a group to determine which are better. This eliminates the need for reference models:

```python
from torch.optim import AdamW
import torch.nn.functional as F

def grpo_loss(model, batch_inputs, batch_answers):
    """
    Compute GRPO loss: compare outputs within groups for relative ranking.
    Rewards: accuracy (correct answer?) + format (proper structure?).
    """
    # Generate multiple responses per input (group)
    num_samples = 4
    all_outputs = []

    for _ in range(num_samples):
        outputs = model.generate(
            **batch_inputs,
            max_length=256,
            do_sample=True,
            temperature=0.7
        )
        all_outputs.append(outputs)

    # Compute rewards for each output
    accuracy_rewards = []
    format_rewards = []

    for output in all_outputs:
        decoded = processor.decode(output, skip_special_tokens=True)

        # Reward 1: Does output contain correct answer?
        has_correct = any(
            ans.lower() in decoded.lower()
            for ans in batch_answers
        )
        accuracy = 1.0 if has_correct else 0.0

        # Reward 2: Is output properly formatted (has answer at end)?
        has_answer_section = any(
            phrase in decoded.lower()
            for phrase in ["answer:", "final answer", "the answer is"]
        )
        format_score = 1.0 if has_answer_section else 0.5

        accuracy_rewards.append(accuracy)
        format_rewards.append(format_score)

    # Combine rewards
    all_rewards = [
        0.7 * acc + 0.3 * fmt
        for acc, fmt in zip(accuracy_rewards, format_rewards)
    ]

    # Group Relative Loss: compare within group
    all_rewards = torch.tensor(all_rewards, device=model.device)
    mean_reward = all_rewards.mean()
    advantages = all_rewards - mean_reward

    # Standard policy gradient: log_prob * advantage
    log_probs = F.log_softmax(torch.stack(all_outputs, dim=0), dim=-1)
    loss = -(log_probs * advantages.unsqueeze(-1)).mean()

    return loss

# Training loop
optimizer = AdamW(model.parameters(), lr=1e-6)
for epoch in range(1):
    for batch in dataloader:
        inputs = processor(batch["video"], batch["question"],
                          return_tensors="pt")
        loss = grpo_loss(model, inputs, batch["answer"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Step 3: Sparse-to-Dense Frame Sampling at Inference

At inference, adaptively increase frame density only when model outputs disagree. Use majority voting to determine convergence:

```python
def adaptive_frame_sampling(model, processor, video_path, question,
                           max_frames=128, convergence_threshold=0.8):
    """
    Sparse-to-dense sampling: start with 32 frames, increase until
    multiple reasoning trajectories agree on the answer.
    """
    import cv2

    # Read video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame_count = 32
    trajectory_outputs = []

    while current_frame_count <= max_frames:
        # Sample current_frame_count frames uniformly
        frame_indices = torch.linspace(
            0, total_frames - 1, current_frame_count
        ).long()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        # Generate reasoning trajectory with current frames
        video_tensor = torch.stack([
            torch.from_numpy(f).float() / 255.0 for f in frames
        ])
        inputs = processor(
            video=video_tensor,
            text=question,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                temperature=0.0
            )

        decoded = processor.decode(output[0], skip_special_tokens=True)
        trajectory_outputs.append(decoded)

        # Check consensus: do N trajectories agree?
        if len(trajectory_outputs) >= 5:
            # Extract final answers from all trajectories
            answers = [extract_final_answer(traj)
                      for traj in trajectory_outputs[-5:]]

            # Majority vote
            from collections import Counter
            vote_counts = Counter(answers)
            max_votes = vote_counts.most_common(1)[0][1]
            agreement_ratio = max_votes / len(answers)

            if agreement_ratio >= convergence_threshold:
                cap.release()
                return trajectory_outputs[-1]

        # Double frame count for next iteration
        current_frame_count = min(current_frame_count * 2, max_frames)

    cap.release()
    return trajectory_outputs[-1]

def extract_final_answer(text):
    """Extract the final answer from reasoning trajectory."""
    lines = text.split('\n')
    for line in reversed(lines):
        if any(phrase in line.lower()
               for phrase in ['answer:', 'final answer', 'the answer is']):
            return line.split(':')[-1].strip()
    return text.split()[-1]
```

### Step 4: Multi-Trajectory Voting

Generate multiple trajectories per input and use consensus voting to decide when to stop:

```python
def multi_trajectory_voting(model, processor, video_path, question,
                           num_trajectories=5, max_frames=128):
    """
    Generate multiple reasoning trajectories and vote on final answer.
    Stops early if consensus is reached before max_frames.
    """
    trajectories = []
    answers = []

    for traj_id in range(num_trajectories):
        # Slight variation in frame sampling per trajectory
        output_text = adaptive_frame_sampling(
            model, processor, video_path, question,
            max_frames=max_frames
        )
        trajectories.append(output_text)

        final_answer = extract_final_answer(output_text)
        answers.append(final_answer)

    # Majority vote on final answer
    from collections import Counter
    answer_votes = Counter(answers)
    best_answer = answer_votes.most_common(1)[0][0]
    confidence = answer_votes[best_answer] / num_trajectories

    return {
        "answer": best_answer,
        "confidence": confidence,
        "trajectories": trajectories
    }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| Training Dataset Size | 6K examples | Shows saturation; diminishing returns beyond this |
| LoRA Rank (if used) | 32-64 | For efficient training on single GPU |
| GRPO Learning Rate | 1e-6 | Conservative to avoid destabilizing policy |
| KL Coefficient | 0.04 | Prevents divergence from base model |
| Initial Frame Count | 32 | Sparse sampling for hour-long videos |
| Max Frame Count | 64-128 | Based on task complexity (knowledge vs general) |
| Convergence Threshold | 0.8 | Stop when 4/5 trajectories agree |
| Number of Trajectories | 5 | Balance between voting confidence and latency |
| Accuracy Reward Weight | 0.7 | Prioritize correct answers over format |
| Format Reward Weight | 0.3 | Secondary emphasis on reasoning clarity |

**When to use Video-RTS:**
- Video QA tasks where you have limited labeled data (< 10K examples)
- Applications where inference compute is flexible (can adapt based on uncertainty)
- Scenarios prioritizing data efficiency over latency
- Multi-turn reasoning where showing your work matters

**When NOT to use Video-RTS:**
- Real-time systems requiring strict latency bounds
- Tasks where output must match specific format (medical reports, precise counts)
- Short videos where frame count is already minimal
- Data-rich domains (100K+ labeled examples) where supervised learning dominates

**Common pitfalls:**
- Setting KL coefficient too high, keeping model too close to base version
- Convergence threshold too strict (0.95+), causing excessive frame sampling
- Frame sampling strategy not matching video type (sports vs dialogue scenes)
- Forgetting to normalize video input (0-1 range vs 0-255)
- Not handling videos shorter than initial frame count (32 frames)

## Reference

Li, Y., Liu, J., Su, Y., & Shen, J. (2025). Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Video Reasoning. arXiv:2507.06485. https://arxiv.org/abs/2507.06485
