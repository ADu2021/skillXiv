---
name: kwai-keye-vl-video-understanding
title: "Kwai Keye-VL Technical Report"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.01949"
keywords: [Video Understanding, Multimodal LLM, Foundation Model, Short-form Video, Visual Reasoning]
description: "Build an 8B multimodal model specializing in short-form video understanding. Combines four-stage pre-training with instruction-tuning and reinforcement learning to enable advanced reasoning about dynamic video content while maintaining general vision-language capabilities."
---

# Kwai Keye-VL: Teaching LLMs to Understand Short-Form Video

Current multimodal models excel at static images but struggle with dynamic video content. Yet short-form video dominates modern platforms—TikTok, Instagram Reels, YouTube Shorts. Understanding these requires reasoning about motion, temporal relationships, and rapid scene changes. Keye-VL addresses this gap with a specialized 8-billion-parameter model trained on 600+ billion tokens with emphasis on video understanding, achieving state-of-the-art performance on video benchmarks while maintaining competitive general vision-language capabilities.

The architecture combines four-stage pre-training (establishing vision-language alignment), two-phase post-training (instruction-following and advanced reasoning), and reinforcement learning alignment to create a model that understands when to reason deeply versus respond directly, and excels specifically at video comprehension.

## Core Concept

Video understanding requires capabilities beyond static image comprehension:

1. **Temporal Reasoning**: Understanding sequences of events, causality, and temporal relationships
2. **Motion Understanding**: Inferring action, speed, and object trajectories from visual patterns
3. **Scene Dynamics**: Recognizing rapid changes, cuts, and transitions
4. **Context from Multiple Frames**: Aggregating information across time to understand scene-level meaning

Keye-VL addresses this through:

- **Massive Video-Centric Dataset**: 600B+ tokens with heavy emphasis on video, not just images with occasional video frames
- **Four-Stage Pre-training**: Progressive alignment from visual tokens to language, with video-specific objectives
- **Instruction-Following Pipeline**: Teaching the model to follow diverse video-understanding instructions
- **Cold-Start Reasoning Framework**: Training with five different reasoning modes (thinking, non-thinking, auto-reasoning) to let the model learn when reasoning helps
- **RL Alignment**: Reinforcement learning to correct behavioral issues (repetition, misalignment) and improve output quality

## Architecture Overview

The Keye-VL system consists of these components:

- **Visual Encoder**: Video-aware vision transformer that processes multiple frames and aggregates temporal information
- **Temporal Aggregation Module**: Fuses information across frames to create frame-level and scene-level representations
- **Multi-Stage Pre-training Pipeline**: Four stages of progressive vision-language alignment
- **Instruction-Tuning System**: Large-scale SFT on diverse video-understanding tasks
- **Reasoning Mode Scheduler**: Five-mode training mixture teaching when to invoke reasoning
- **Reinforcement Learning Stage**: Policy optimization to improve reasoning quality and reduce artifacts
- **Evaluation Framework**: KC-MMBench benchmark specifically designed for real-world short-video scenarios

## Implementation

This section demonstrates how to build a video-understanding model with Keye-VL's approach.

**Step 1: Build video-aware visual encoder**

This code implements temporal aggregation for multi-frame processing:

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class VideoAwareVisualEncoder(nn.Module):
    """
    Vision transformer that processes video frames and aggregates temporal information.
    Maintains frame-level detail while creating scene-level representations.
    """

    def __init__(self, embed_dim=768, num_frames=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Frame-level visual encoder (processes each frame independently)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),  # 14×14 patch grid
            nn.Flatten(1),  # (B*T, 128*196)
            nn.Linear(128 * 196, embed_dim)
        )

        # Temporal aggregation: cross-frame attention
        temporal_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.1
        )
        self.temporal_encoder = TransformerEncoder(temporal_layer, num_layers=4)

        # Scene-level aggregation (collapse temporal dimension)
        self.scene_aggregation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, video_tensor):
        """
        Process video tensor and return both frame-level and scene-level representations.

        video_tensor: (B, T, 3, H, W) - batch of T-frame videos
        """

        B, T, C, H, W = video_tensor.shape

        # Process frames individually
        frames_flat = video_tensor.reshape(B * T, C, H, W)
        frame_features = self.frame_encoder(frames_flat)  # (B*T, D)
        frame_features = frame_features.reshape(B, T, self.embed_dim)  # (B, T, D)

        # Temporal aggregation via cross-frame attention
        temporal_features = self.temporal_encoder(frame_features)  # (B, T, D)

        # Scene-level representation (aggregate across time)
        scene_features = temporal_features.mean(dim=1)  # (B, D)
        scene_features = self.scene_aggregation(scene_features)  # (B, D)

        return {
            'frame_features': frame_features,  # (B, T, D) - per-frame details
            'scene_features': scene_features,   # (B, D) - holistic scene understanding
            'temporal_features': temporal_features  # (B, T, D) - temporally-aware frames
        }

# Test video encoder
encoder = VideoAwareVisualEncoder(embed_dim=768, num_frames=8)
video = torch.randn(2, 8, 3, 224, 224)  # 2 videos, 8 frames each

output = encoder(video)
print(f"Frame features shape: {output['frame_features'].shape}")
print(f"Scene features shape: {output['scene_features'].shape}")
print(f"Temporal features shape: {output['temporal_features'].shape}")
```

This creates a visual encoder aware of temporal structure.

**Step 2: Implement four-stage pre-training pipeline**

This code shows the progressive alignment approach:

```python
class FourStagePretrain:
    """
    Four-stage pre-training pipeline for video-language alignment.
    """

    @staticmethod
    def stage_1_frame_embedding_alignment():
        """
        Stage 1: Align visual frame embeddings with language.
        Objective: Learn to map frame pixels → semantically meaningful embeddings
        """

        print("Stage 1: Frame Embedding Alignment")
        print("  - Visual tokens → semantic embeddings")
        print("  - Contrastive learning: matching frames to descriptions")
        print("  - Loss: frame-text contrastive loss")

    @staticmethod
    def stage_2_temporal_understanding():
        """
        Stage 2: Learn temporal relationships between frames.
        Objective: Understand how visual content changes across time
        """

        print("Stage 2: Temporal Understanding")
        print("  - Temporal coherence: consecutive frames should have correlated embeddings")
        print("  - Action recognition: short clips → action descriptions")
        print("  - Loss: temporal contrastive + action prediction")

    @staticmethod
    def stage_3_scene_understanding():
        """
        Stage 3: Scene-level video understanding.
        Objective: Aggregate frames into scene-level representations
        """

        print("Stage 3: Scene Understanding")
        print("  - Video summarization: full video → semantic summary")
        print("  - Event detection: identify key moments in video")
        print("  - Loss: video-text contrastive + event classification")

    @staticmethod
    def stage_4_language_generation():
        """
        Stage 4: Full video-to-language generation.
        Objective: Generate natural language descriptions of videos
        """

        print("Stage 4: Language Generation")
        print("  - Video captioning: video → descriptive captions")
        print("  - Video question answering: answer questions about video content")
        print("  - Loss: language modeling + QA accuracy")

# Display pre-training stages
pretrain = FourStagePretrain()
pretrain.stage_1_frame_embedding_alignment()
pretrain.stage_2_temporal_understanding()
pretrain.stage_3_scene_understanding()
pretrain.stage_4_language_generation()

print("\nPre-training data distribution:")
print("  - Total tokens: 600+ billion")
print("  - Video content: ~40% of data")
print("  - Image content: ~35% of data")
print("  - Text-only: ~25% of data")
```

This outlines the progressive training approach.

**Step 3: Implement five-mode reasoning mixture**

This code trains the model to select appropriate reasoning depth:

```python
class FiveModesReasoningTraining:
    """
    Train model with five reasoning modes to learn when reasoning helps.
    """

    MODES = {
        'direct': 'Answer directly without reasoning',
        'thinking': 'Show step-by-step thinking before answer',
        'non_thinking': 'Answer without internal reasoning shown',
        'auto_think': 'Model decides whether to show reasoning',
        'video_reasoning': 'Deep reasoning specifically about video dynamics'
    }

    @staticmethod
    def create_training_mixture():
        """Create balanced training mixture of all five modes."""

        mixture = {
            'direct': 0.15,  # 15% pure direct answers
            'thinking': 0.25,  # 25% explicit step-by-step reasoning
            'non_thinking': 0.15,  # 15% hidden reasoning
            'auto_think': 0.25,  # 25% model-decided reasoning
            'video_reasoning': 0.20  # 20% video-specific deep reasoning
        }

        return mixture

    @staticmethod
    def format_training_example(video, question, answer, mode):
        """Format training example in the specified reasoning mode."""

        if mode == 'direct':
            prompt = f"Video: [video]\nQ: {question}\nA: {answer}"

        elif mode == 'thinking':
            # Include reasoning steps
            prompt = f"""Video: [video]
Q: {question}

Let me think about this step by step:
1. [identify key elements in video]
2. [trace temporal sequence]
3. [synthesize into answer]

A: {answer}"""

        elif mode == 'non_thinking':
            prompt = f"Video: [video]\nQ: {question}\n[internal reasoning happens but not shown]\nA: {answer}"

        elif mode == 'auto_think':
            # Model learns to decide based on question complexity
            prompt = f"Video: [video]\nQ: {question}\n[reasoning if needed]\nA: {answer}"

        elif mode == 'video_reasoning':
            # Deep video analysis
            prompt = f"""Video: [video with temporal markers]
Q: {question}

Video Analysis:
- Frame-by-frame dynamics: [analyze motion/changes]
- Temporal relationships: [trace cause and effect]
- Scene context: [global understanding]

A: {answer}"""

        return prompt

    @staticmethod
    def create_five_mode_dataset(base_dataset, num_samples=100000):
        """Create training dataset with all five reasoning modes."""

        five_mode_data = []

        for sample in base_dataset:
            video = sample['video']
            question = sample['question']
            answer = sample['answer']

            # Create version in each reasoning mode
            for mode in FiveModesReasoningTraining.MODES.keys():
                prompt = FiveModesReasoningTraining.format_training_example(
                    video, question, answer, mode
                )

                five_mode_data.append({
                    'prompt': prompt,
                    'mode': mode,
                    'video': video,
                    'answer': answer
                })

        return five_mode_data

# Demonstrate five-mode training
mixture = FiveModesReasoningTraining.create_training_mixture()
print("Five-mode reasoning training distribution:")
for mode, weight in mixture.items():
    print(f"  {mode}: {weight:.0%}")
    print(f"    → {FiveModesReasoningTraining.MODES[mode]}")
```

This trains the model to adaptively select reasoning depth.

**Step 4: Implement instruction-tuning on video tasks**

This code fine-tunes on diverse video-understanding instructions:

```python
class VideoInstructionTuning:
    """
    Instruction-following training on video-understanding tasks.
    Teaches model to follow diverse instructions about video content.
    """

    INSTRUCTION_TYPES = {
        'description': 'Describe what happens in this video',
        'classification': 'What action is performed in this video?',
        'counting': 'How many objects of type X appear in the video?',
        'temporal': 'What happens first, A or B?',
        'causality': 'Why does Y happen after X?',
        'anomaly': 'Is there anything unusual in this video?',
        'emotion': 'What emotion or mood does this video convey?',
        'detail': 'Describe the appearance of object X at frame T',
        'comparison': 'How is scene A different from scene B?',
        'reasoning': 'Based on the video, what is the most likely next event?'
    }

    @staticmethod
    def generate_instructions_for_video(video_data, num_instructions_per_video=10):
        """Generate diverse instructions for a single video."""

        instructions = []

        for _ in range(num_instructions_per_video):
            # Randomly select instruction type
            instruction_type = np.random.choice(list(VideoInstructionTuning.INSTRUCTION_TYPES.keys()))
            template = VideoInstructionTuning.INSTRUCTION_TYPES[instruction_type]

            # Specialize template for this video
            if instruction_type == 'description':
                instruction = template  # Generic template
            elif instruction_type == 'classification':
                instruction = template  # Generic template
            elif instruction_type == 'detail':
                frame_idx = np.random.randint(0, len(video_data['frames']))
                instruction = f"Describe the appearance of objects at frame {frame_idx}"
            elif instruction_type == 'temporal':
                instruction = "What significant event happens?"
            else:
                instruction = template

            instructions.append(instruction)

        return instructions

    @staticmethod
    def create_sft_dataset(videos, annotations, num_instructions_per_video=10):
        """Create supervised fine-tuning dataset with video instructions."""

        sft_dataset = []

        for video_idx, video in enumerate(videos):
            annotation = annotations[video_idx]

            # Generate instructions
            instructions = VideoInstructionTuning.generate_instructions_for_video(
                video,
                num_instructions_per_video
            )

            # Create training example for each instruction
            for instruction in instructions:
                sft_dataset.append({
                    'video': video,
                    'instruction': instruction,
                    'answer': annotation.get('general_description', ''),
                    'reasoning_mode': 'auto_think'  # Let model decide reasoning
                })

        return sft_dataset

# Create instruction-tuned dataset
print("Video Instruction Types:")
for itype, template in VideoInstructionTuning.INSTRUCTION_TYPES.items():
    print(f"  {itype}: {template}")
```

This shows instruction-tuning on video-specific tasks.

**Step 5: Apply reinforcement learning for alignment**

This code uses RL to improve output quality:

```python
class VideoReasoningRL:
    """
    Reinforcement learning alignment for video reasoning models.
    Optimizes for reasoning quality and eliminating artifacts.
    """

    @staticmethod
    def define_reward_function():
        """Define reward signals for video understanding quality."""

        reward_components = {
            'accuracy': {
                'weight': 0.4,
                'description': 'Factual correctness about video content'
            },
            'temporal_consistency': {
                'weight': 0.2,
                'description': 'Consistent temporal understanding across frames'
            },
            'detail_awareness': {
                'weight': 0.15,
                'description': 'Noticing important details in video'
            },
            'reasoning_quality': {
                'weight': 0.15,
                'description': 'Quality of reasoning when invoked'
            },
            'output_quality': {
                'weight': 0.1,
                'description': 'Avoiding repetition and coherence'
            }
        }

        return reward_components

    @staticmethod
    def compute_temporal_consistency_reward(response, ground_truth_frames):
        """Reward if model correctly understands temporal sequence."""

        # Check if response mentions events in correct temporal order
        # Higher reward if sequence is accurate
        reward = 0.0
        # (simplified; real implementation would parse response semantically)
        return reward

    @staticmethod
    def compute_detail_awareness_reward(response, video_frames):
        """Reward if model notices important visual details."""

        # Check if response mentions specific objects, colors, actions
        details_mentioned = 0  # Count of specific details
        reward = min(details_mentioned / 10.0, 1.0)  # Normalized to [0, 1]

        return reward

    @staticmethod
    def collect_rl_trajectories(model, videos, num_trajectories=10000):
        """Collect model outputs for RL optimization."""

        trajectories = []

        for video_idx in range(len(videos)):
            video = videos[video_idx]

            # Generate response from model
            response = model.generate(video, do_sample=True, temperature=0.8)

            # Compute rewards
            rewards = {}
            for component in VideoReasoningRL.define_reward_function().keys():
                if component == 'temporal_consistency':
                    rewards[component] = VideoReasoningRL.compute_temporal_consistency_reward(response, video)
                elif component == 'detail_awareness':
                    rewards[component] = VideoReasoningRL.compute_detail_awareness_reward(response, video)
                else:
                    rewards[component] = np.random.rand()  # Placeholder

            # Compute combined reward
            weights = {c: d['weight'] for c, d in VideoReasoningRL.define_reward_function().items()}
            total_reward = sum(weights[c] * rewards[c] for c in weights)

            trajectories.append({
                'video': video,
                'response': response,
                'rewards': rewards,
                'total_reward': total_reward
            })

        return trajectories

# Show RL reward structure
print("RL Reward Components for Video Understanding:")
rewards = VideoReasoningRL.define_reward_function()
for component, info in rewards.items():
    print(f"  {component} (weight {info['weight']:.0%}): {info['description']}")
```

This implements RL alignment for improved reasoning.

## Practical Guidance

**When to use Keye-VL approach:**
- Building models specialized for video content understanding
- Applications requiring reasoning about temporal dynamics
- Short-form video platforms needing content analysis
- Systems that must understand motion and scene changes
- Multi-modal models where video is significant use case

**When NOT to use:**
- Pure image-based systems (Keye-VL overspecializes for video)
- Text-only applications (unnecessary overhead)
- Real-time constraints on computational budget
- Domains without temporal/dynamic content
- Scenarios where general-purpose models suffice

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Model Size | 8B parameters | Balance between capability and efficiency |
| Video Tokens | 40% of training | Ensure video is well-represented |
| Num Frames per Video | 8-16 frames | Capture temporal structure without bloat |
| Pre-training Stages | 4 stages | Progressive alignment improves efficiency |
| Reasoning Modes | 5 modes | Coverage of direct, thinking, and auto modes |
| RL Training Episodes | 10,000+ trajectories | Sufficient for stable policy learning |
| Learning Rate | 1e-5 (RL phase) | Conservative for stability during alignment |
| Batch Size | 32-64 | Larger batches reduce variance in RL |

**Common Pitfalls:**
- Treating video as "images with temporal labels" (requires true temporal modeling)
- Over-weighting one reasoning mode (train all five for robustness)
- Insufficient video-specific pre-training (general models don't understand motion)
- Ignoring temporal ordering in training (don't shuffle video frames)
- RL reward misspecification (hard to measure temporal understanding)
- Using too few video tokens during pre-training (underfits video understanding)

**Key Design Decisions:**
Keye-VL specializes in video by treating temporal understanding as a core objective across all four pre-training stages. The five-mode reasoning mixture allows the model to learn when deep reasoning helps versus when direct answers suffice. Reinforcement learning addresses the unique challenge of video understanding: temporal consistency and motion coherence are hard to specify in loss functions but critical for quality. Four-stage pre-training ensures progressive alignment rather than trying to learn everything simultaneously.

## Reference

Li, R., Xu, L., Ren, Z., Jiang, J., Zhang, L., Huang, Y., ... & Ma, K. (2025). Kwai Keye-VL Technical Report. arXiv preprint arXiv:2507.01949. https://arxiv.org/abs/2507.01949
