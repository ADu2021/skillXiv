---
name: watch-and-learn-computer-use-videos
title: "Watch and Learn: Learning to Use Computers from Online Videos"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.04673"
keywords: [inverse dynamics, video-to-trajectory, UI action extraction, computer-using agents, vision-language]
description: "Convert internet tutorial videos into executable UI action trajectories using inverse dynamics models trained on 600K synthetic screen transitions. Generate 53K high-quality demonstrations spanning 69 applications, achieving state-of-the-art 7B agent performance on WindowsAgentArena via both in-context learning and supervised fine-tuning."
---

# Watch and Learn: Learning to Use Computers from Online Videos

## Core Concept

Rather than directly predicting actions from visual observations, an inverse dynamics model (IDM) infers which action likely caused the transition between consecutive video frames. This formulation simplifies trajectory extraction from web-scale video sources, enabling generation of 53,000+ executable UI demonstrations for training computer-using agents.

## Architecture Overview

- **Inverse Dynamics Model**: Predicts action from (screen_t, screen_t+1) pairs rather than direct action regression
- **Vision Backbone**: SigLIP-2 Base encoder producing 1024 visual tokens per 1000×1000 screenshot
- **Multi-Head Prediction**: Separate branches for action classification (6 primitives), coordinate bins (1000×1000 discretization), and text generation (GPT-2 decoder)
- **Video Filtering Pipeline**: Automated screencast detection and quality filtering (avg frame score >0.8)
- **Action Space Composition**: Atomic primitives (click, release, scroll, type, wait, move) compose into environment-specific commands via deterministic mapping

## Implementation Steps

### 1. Training the Inverse Dynamics Model

Collect 600K synthetic (screen_t, action, screen_t+1) transitions from automated web interaction using Playwright. Train a multi-task model to predict action types, coordinates, and text.

```python
import torch
from transformers import AutoModel

class InverseDynamicsModel(torch.nn.Module):
    def __init__(self):
        # Vision encoder: SigLIP-2 Base (1024 tokens per screen)
        self.vision_encoder = AutoModel.from_pretrained("siglip-so400m-patch14-384")

        # Backbone: 4 Transformer layers
        self.backbone = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=768, nhead=12, dim_feedforward=3072, batch_first=True
            ),
            num_layers=4
        )

        # Prediction heads
        self.action_classifier = torch.nn.Linear(768, 6)  # click, release, scroll, type, wait, move
        self.coord_predictor = torch.nn.Linear(768, 2000)  # 1000 bins x/y
        self.text_decoder = torch.nn.LSTM(768, 768, batch_first=True)

    def forward(self, screen_t, screen_t_plus_1):
        # Encode both screens
        tokens_t = self.vision_encoder(screen_t)
        tokens_next = self.vision_encoder(screen_t_plus_1)

        # Concatenate and process
        combined = torch.cat([tokens_t, tokens_next], dim=1)
        features = self.backbone(combined)

        # Predict action, coordinates, text
        action_logits = self.action_classifier(features[:, 0])
        coord_logits = self.coord_predictor(features[:, 0])
        text_tokens = self.text_decoder(features)

        return action_logits, coord_logits, text_tokens
```

Training details: AdamW (lr=3e-4), batch size 256, 15 epochs on 8×A100 GPUs with bfloat16 mixed precision. Multi-task loss combines classification, coordinate regression, and language modeling with equal weighting.

### 2. Collecting Video Data at Scale

Query YouTube API for tutorial videos across 69 applications spanning 7 categories (productivity, programming, design, audio, utilities, system, media). Use Gemini 2.5 Flash to generate specific task-oriented search queries from initial screenshots.

```python
def retrieve_and_filter_videos(task_instruction, initial_screen):
    # Generate specific query from task and screenshot
    query = gemini.generate(
        f"Generate a specific YouTube search query for: {task_instruction}\n"
        f"Initial screen: {initial_screen}"
    )

    # Retrieve candidate videos
    videos = youtube_search_api.search(query, max_results=50)

    # Automatic quality filtering: remove non-screencast segments
    filtered = []
    for video in videos:
        frames = extract_frames(video, fps=1)  # 1 frame/sec

        # Score frames for screencast quality (model-based classifier)
        scores = screencast_classifier(frames)

        # Retain video if avg score > 0.8 (remove tutorials with talking heads, logos)
        if scores.mean() > 0.8:
            filtered.append(video)

    return filtered
```

Video pool spans 69 applications. Curated searches ensure coverage across diverse task categories rather than biasing toward common applications.

### 3. Extracting Trajectories via Inverse Dynamics

For each quality-filtered video, extract frames at 1 fps and apply the trained IDM to every consecutive frame pair to generate executable action sequences.

```python
def extract_trajectory_from_video(video_path, idm_model):
    # Frame extraction
    frames = extract_frames(video_path, fps=1)  # O_0, O_1, ...
    trajectory = [frames[0]]  # Initial observation

    # Apply IDM to each consecutive pair
    actions = []
    for t in range(len(frames) - 1):
        frame_t = frames[t]
        frame_t_plus_1 = frames[t + 1]

        # IDM predicts action
        action_type, coords, text = idm_model(frame_t, frame_t_plus_1)

        # Parse predictions
        action_class = action_type.argmax()
        x_bin, y_bin = coords.argmax(dim=1)
        x, y = x_bin / 1000, y_bin / 1000  # Denormalize

        # Generate environment command (deterministic mapping)
        command = compose_action(action_class, x, y, text)
        actions.append(command)
        trajectory.append(frame_t_plus_1)

    return {
        'observations': trajectory,
        'actions': actions,
        'num_steps': len(actions)
    }

# Compose primitives into environment-compatible commands
def compose_action(action_type, x, y, text):
    if action_type == 0:  # Click
        return f"click({x}, {y})"
    elif action_type == 1:  # Release (drag end)
        return f"release({x}, {y})"
    elif action_type == 2:  # Scroll
        return f"scroll({x}, {y}, {'up' if y < 0.5 else 'down'})"
    elif action_type == 3:  # Type
        return f"type('{text}')"
    elif action_type == 4:  # Wait
        return "wait(1)"
    elif action_type == 5:  # Move
        return f"move({x}, {y})"
```

IDM achieves 95.8% action type accuracy and 91.7% overall action accuracy on test set. 53K trajectories generated across Windows, macOS, Ubuntu.

### 4. Using Trajectories for Agent Training

Deploy extracted trajectories in two modalities: in-context learning (format as examples) or supervised fine-tuning (direct policy training).

```python
def in_context_learning_mode(task_instruction, trajectories):
    # Retrieve relevant trajectories via semantic search
    relevant_trajs = retrieve_similar_trajectories(task_instruction, trajectories)

    # Format as in-context exemplars (observation, action, reasoning)
    exemplars = []
    for traj in relevant_trajs[:3]:  # Use top 3
        exemplars.append({
            'observation': traj['observations'][0],
            'action_sequence': traj['actions'],
            'reasoning': f"Completed task in {len(traj['actions'])} steps"
        })

    # Prepend to agent prompt without modifying downstream logic
    prompt = format_exemplars(exemplars) + "\n" + task_instruction
    return prompt

def supervised_fine_tuning_mode(model, trajectories, lr=1e-5, epochs=3):
    # Standard sequence modeling: predict action_t from observation_t
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for traj in trajectories:
            observations = traj['observations']
            actions = traj['actions']

            # Forward pass: predict actions from observations
            action_logits = model.predict_action(observations[:-1])

            # Loss: cross-entropy on action tokens
            loss = cross_entropy(action_logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

Performance results: In-context mode provides +3.0 to +11.1 point improvements on OSWorld/WindowsAgentArena benchmarks. SFT on 53K trajectories achieves 24.0% success on WindowsAgentArena with 7B model under 15-step constraint (SOTA for scale).

## Practical Guidance

**Video Quality Matters**: Automatic screencast detection filters out tutorials with excessive talking, logos, or visual noise. Threshold (avg frame score >0.8) removes ~60% of candidates but ensures trajectory quality.

**Coordinate Discretization**: Quantize screen coordinates into 1000 uniform bins per axis (convert regression to classification). This simplifies training and improves stability compared to continuous coordinate prediction.

**Scale Advantages**: Largest improvements occur in domains with abundant tutorial content (GIMP: 60K+ tutorials, VLC: 40K+). Domains with sparse tutorials (niche apps) show minimal gains.

**Inference Trade-off**: In-context learning avoids model updates but requires storage/retrieval of exemplars. SFT permanently encodes knowledge but requires compute-intensive training.

## When to Use / When NOT to Use

**Use When**:
- Training computer-using agents across diverse applications
- Tutorial videos exist for target domains (productivity, design, programming software)
- You need multimodal agent training data at scale (1000s of diverse examples)
- Deployment favors semantic similarity to real human workflows

**NOT For**:
- Tasks with no tutorial content available (niche, proprietary applications)
- Safety-critical systems where observed human behavior may include errors
- Low-bandwidth scenarios (53K trajectories require substantial storage)
- Domains where synthetic data generation is feasible and cost-effective

## Reference

This skill extracts findings from "Watch and Learn: Learning to Use Computers from Online Videos" (arXiv:2510.04673, CVPR 2026). The inverse dynamics formulation and large-scale trajectory extraction pipeline enable practical multimodal agent training from real-world video sources.
