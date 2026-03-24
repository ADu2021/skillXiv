---
name: f1-vla-understanding-generation-actions
title: "F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.06951"
keywords: [vision-language-action, robotic control, foresight generation, inverse dynamics, mixture-of-experts, multimodal learning, embodied AI, trajectory prediction]
description: "F1 is a 4.2B-parameter Vision-Language-Action model that reformulates robotic control as foresight-guided inverse dynamics. Rather than reactive state-to-action mappings, F1 predicts plausible future visual states and derives actions to achieve them, enabling robust planning in dynamic environments with 82.2% real-world task success rate."
---

## Outcome: Build a Foresight-Guided Robot Control System

Implement a robot controller that explicitly predicts future visual goals before deciding actions. F1 transforms reactive visuomotor policies into planning-aware systems by decomposing the control problem into three modular experts: one understanding language and observations, one generating visual foresight, and one predicting inverse dynamics conditioned on predicted futures.

## Problem Context

Traditional vision-language-action (VLA) models map raw observations directly to motor commands without explicit planning. This reactive paradigm struggles in dynamic environments where future states are uncertain and long-horizon tasks require coordination. F1 addresses this by making prediction of visual futures an explicit intermediate step, allowing the model to learn that actions should drive the environment toward anticipated goal states rather than blindly mimicking training trajectories.

## Core Concept

F1 treats robotic control as a three-stage pipeline:

1. **Understanding**: Encode language instructions and visual observations into a shared semantic space using a pretrained vision-language model backbone.
2. **Generation**: Predict a goal-conditioned visual foresight—a plausible future frame that represents the intermediate or final state the robot should achieve.
3. **Action**: Compute motor commands as inverse dynamics, deriving actions that would transition from the current visual state toward the predicted future state.

This formulation decouples planning from execution: the generation expert learns "what good futures look like" while the action expert learns "how to reach those futures." The hierarchical attention scheme ensures information flows correctly without shortcuts.

## Architecture Overview

F1 employs a Mixture-of-Transformer design with three specialized experts:

- **Understanding Expert**: Inherits from a pretrained multimodal large language model (MLLM) backbone. Processes textual instructions and visual observations to produce semantic tokens that establish shared multimodal grounding.

- **Generation Expert**: Synthesizes goal-conditioned visual foresight using next-scale prediction. Decomposes predicted frames into multi-resolution tokens via residual VQ-VAE encoding. Operates at four spatial scales to balance computational cost with planning accuracy.

- **Action Expert**: Predicts action sequences conditioned on both current observations and predicted future visual states. Uses flow matching objectives for training, enabling continuous action space prediction.

- **Hierarchical UGA Attention**: Enforces causal information flow where generation tokens attend to understanding tokens, and action tokens attend to both understanding and generation. Prevents shortcut correlations and ensures proper modular reasoning.

The full model contains 4.2B parameters with a Gemma transformer backbone. During inference, real-time operation is maintained through efficient tokenization and streaming prediction.

## Implementation

### Stage I: Alignment Phase

Align the generation expert with the pretrained understanding expert through supervised learning on image token prediction.

**Explanation**: The understanding expert starts with pretrained weights from a multimodal foundation model. In this stage, we keep those weights frozen and train the generation expert to produce image tokens that match ground-truth image encodings provided by the understanding expert. This initializes the generation expert's ability to produce valid visual continuations.

```python
# Stage I: Teacher-Forced Image Token Generation Alignment

import torch
import torch.nn as nn
from torch.optim import AdamW

class StageITrainer:
    def __init__(self, understanding_expert, generation_expert, device):
        self.understanding_expert = understanding_expert.to(device)
        self.generation_expert = generation_expert.to(device)
        self.device = device

        # Freeze understanding expert
        for param in self.understanding_expert.parameters():
            param.requires_grad = False

        self.optimizer = AdamW(self.generation_expert.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_batch(self, observations, language_tokens, target_future_frames):
        """
        observations: [B, T, C, H, W] - current and historical observations
        language_tokens: [B, seq_len] - tokenized instructions
        target_future_frames: [B, num_scales, num_tokens] - target image tokens at multiple scales
        """
        # Encode observations and language into semantic space
        with torch.no_grad():
            semantic_repr = self.understanding_expert(
                observations=observations,
                language=language_tokens
            )

        # Generate predicted image tokens conditioned on semantic representation
        predicted_tokens = self.generation_expert(
            semantic_repr=semantic_repr,
            teacher_forcing=True
        )

        # Compute cross-entropy loss for each scale
        total_loss = 0
        for scale_idx in range(len(predicted_tokens)):
            scale_loss = self.loss_fn(
                predicted_tokens[scale_idx],
                target_future_frames[:, scale_idx]
            )
            total_loss += scale_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generation_expert.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()
```

### Stage II: Joint Pretraining on 330K+ Trajectories

Train all three experts jointly on diverse robotic datasets using autoregressive generation and flow-matching action losses.

**Explanation**: With the generation expert initialized, we now train the entire system end-to-end. The understanding expert is unfrozen. We feed trajectory data (sequences of observations and actions) through the pipeline. The generation expert learns to predict realistic future frames contextually, and the action expert learns that its predicted actions should approximately match the inverse dynamics needed to reach the predicted future. Flow matching provides a continuous action space formulation that is more stable than discrete action prediction.

```python
# Stage II: Joint Three-Expert Pretraining with Flow Matching

from diffusers.models import FlowMatchingScheduler
import torch.nn.functional as F

class StageIITrainer:
    def __init__(self, understanding_expert, generation_expert, action_expert,
                 vq_vae_encoder, device, num_diffusion_steps=1000):
        self.understanding_expert = understanding_expert.to(device)
        self.generation_expert = generation_expert.to(device)
        self.action_expert = action_expert.to(device)
        self.vq_vae = vq_vae_encoder.to(device)
        self.device = device

        # All experts trainable now
        self.params = list(understanding_expert.parameters()) + \
                      list(generation_expert.parameters()) + \
                      list(action_expert.parameters())

        self.optimizer = AdamW(self.params, lr=2e-5, weight_decay=1e-4)
        self.flow_scheduler = FlowMatchingScheduler(num_steps=num_diffusion_steps)
        self.num_diffusion_steps = num_diffusion_steps

    def encode_image_tokens(self, images):
        """Encode images to multi-scale VQ-VAE tokens"""
        # Residual VQ-VAE: encode at multiple scales
        z = self.vq_vae.encode(images)  # [B, num_tokens_fine]

        # Generate coarser scales through downsampling
        coarse_scales = []
        current_z = z
        for scale in range(3):  # 4 scales total
            coarse_scales.append(current_z)
            # Downsample by factor of 2
            current_z = current_z[:, ::2]

        return [z] + coarse_scales

    def compute_flow_matching_loss(self, model_pred, target_action, t):
        """
        Flow matching loss for continuous action prediction.
        model_pred: predicted action from action expert
        target_action: ground truth action from trajectory
        t: timestep in [0, 1] for flow matching
        """
        # Interpolate between random noise and target action
        noise = torch.randn_like(target_action)
        flow_target = target_action - noise

        # Compute L2 loss
        loss = F.mse_loss(model_pred, flow_target)
        return loss

    def train_batch(self, trajectories, batch_size=32):
        """
        trajectories: dict with keys 'observations', 'actions', 'language'
                     observations: [B, T, C, H, W]
                     actions: [B, T, action_dim]
                     language: [B, seq_len]
        """
        observations = trajectories['observations'].to(self.device)
        actions = trajectories['actions'].to(self.device)
        language = trajectories['language'].to(self.device)

        # Encode current and future observations
        current_obs = observations[:, 0]
        future_obs = observations[:, 1]

        # Stage I: Understanding
        semantic_repr = self.understanding_expert(
            observations=current_obs,
            language=language
        )

        # Stage II: Generation - predict future visual tokens
        predicted_future_tokens = self.generation_expert(
            semantic_repr=semantic_repr,
            current_tokens=self.encode_image_tokens(current_obs)
        )

        # Get ground truth future tokens
        with torch.no_grad():
            target_future_tokens = self.encode_image_tokens(future_obs)

        # Generation loss: cross-entropy across all scales
        gen_loss = 0
        for scale_idx in range(len(predicted_future_tokens)):
            gen_loss += F.cross_entropy(
                predicted_future_tokens[scale_idx].view(-1, self.vq_vae.vocab_size),
                target_future_tokens[scale_idx].view(-1)
            )

        # Stage III: Action - inverse dynamics with flow matching
        # Sample random timestep for flow matching
        t = torch.rand(observations.shape[0], device=self.device)

        predicted_actions = self.action_expert(
            current_tokens=self.encode_image_tokens(current_obs),
            future_tokens=predicted_future_tokens,
            semantic_repr=semantic_repr,
            t=t
        )

        # Flow matching loss
        action_loss = self.compute_flow_matching_loss(predicted_actions, actions[:, 0], t)

        # Combined loss
        total_loss = gen_loss + action_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'generation_loss': gen_loss.item(),
            'action_loss': action_loss.item()
        }
```

### Stage III: Task-Specific Post-Training and Embodiment Adaptation

Fine-tune on target-domain trajectories to adapt the model to specific robot embodiments and refine task-specific manipulation skills.

**Explanation**: The pretrained model is now specialized to particular robots and tasks. We use significantly smaller learning rates and smaller batch sizes. The focus shifts from learning general visuomotor knowledge to adapting the model's action predictions to the specific kinematics, action spaces, and failure modes of a particular embodiment.

```python
# Stage III: Task-Specific Fine-Tuning with Early Stopping

class StageIIITrainer:
    def __init__(self, pretrained_model, device, validation_split=0.1):
        self.model = pretrained_model.to(device)
        self.device = device
        self.validation_split = validation_split

        # Lower learning rate for fine-tuning
        self.optimizer = AdamW(self.model.parameters(), lr=5e-6, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-7
        )

        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0

    def fine_tune_epoch(self, train_dataloader, val_dataloader):
        """
        Fine-tune on task-specific data with early stopping.
        """
        train_loss = 0
        num_batches = 0

        self.model.train()
        for batch in train_dataloader:
            loss_dict = self.model.train_step(batch)
            loss = loss_dict['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation
        val_loss = 0
        val_batches = 0
        self.model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                loss_dict = self.model.train_step(batch)
                val_loss += loss_dict['total_loss'].item()
                val_batches += 1

        val_loss /= val_batches
        self.scheduler.step()

        # Early stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), 'best_model.pt')
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping at validation loss {self.best_val_loss:.4f}")
                return True  # Stop training

        return False  # Continue training

    def prepare_task_dataset(self, raw_trajectories, task_id):
        """
        Prepare dataset for a specific task embodiment.
        raw_trajectories: list of trajectory dicts with obs, actions, language
        """
        processed = []
        for traj in raw_trajectories:
            obs = traj['observations']
            actions = traj['actions']
            lang = traj['instruction']

            # Normalize actions to [-1, 1] for this embodiment
            action_mean = actions.mean(axis=0)
            action_std = actions.std(axis=0) + 1e-8
            normalized_actions = (actions - action_mean) / action_std

            # Augment language with task context if available
            augmented_lang = f"{lang} [Task ID: {task_id}]"

            processed.append({
                'observations': obs,
                'actions': normalized_actions,
                'language': augmented_lang,
                'action_stats': {'mean': action_mean, 'std': action_std}
            })

        return processed
```

### Inference: Closed-Loop Rollout with Foresight

Deploy the trained model for real-time control by repeatedly predicting future frames and deriving actions.

**Explanation**: At test time, we run the model in a closed-loop: at each timestep, the model predicts what the next visual state should look like, then predicts the action needed to reach that state. After executing the action, the new observation is fed back, and the cycle repeats. This allows the robot to continuously replan and adapt to deviations from the predicted trajectory.

```python
# Inference: Closed-Loop Rollout with Adaptive Foresight

class F1InferenceController:
    def __init__(self, model, robot_interface, device, max_steps=500):
        self.model = model.eval().to(device)
        self.robot = robot_interface
        self.device = device
        self.max_steps = max_steps

    def rollout_episode(self, instruction, initial_obs, action_scale=1.0):
        """
        Run a closed-loop episode guided by predicted visual foresight.

        instruction: language goal (str)
        initial_obs: starting observation [C, H, W]
        action_scale: scale factor for action magnitude (safety mechanism)

        Returns: trajectory dict with obs, actions, success flag
        """
        trajectory = {
            'observations': [initial_obs],
            'actions': [],
            'predicted_futures': [],
            'language': instruction
        }

        current_obs = initial_obs.to(self.device).unsqueeze(0)

        with torch.no_grad():
            for step in range(self.max_steps):
                # Encode instruction
                lang_tokens = self.model.tokenize_language(instruction)

                # Stage 1: Understanding - establish semantic context
                semantic_repr = self.model.understanding_expert(
                    observations=current_obs,
                    language=lang_tokens
                )

                # Stage 2: Generation - predict next visual frame
                predicted_future_tokens = self.model.generation_expert(
                    semantic_repr=semantic_repr,
                    current_tokens=self.model.encode_tokens(current_obs)
                )

                # Decode tokens back to image space for monitoring
                predicted_future_frame = self.model.vq_vae.decode(
                    predicted_future_tokens[0]  # Use finest scale for control
                )
                trajectory['predicted_futures'].append(
                    predicted_future_frame.cpu().numpy()
                )

                # Stage 3: Action - compute inverse dynamics toward predicted future
                action_pred = self.model.action_expert(
                    current_tokens=self.model.encode_tokens(current_obs),
                    future_tokens=predicted_future_tokens,
                    semantic_repr=semantic_repr,
                    t=torch.zeros(1, device=self.device)  # Use deterministic flow at t=0
                )

                # Extract action and apply scaling for safety
                action = action_pred[0].cpu().numpy()
                action = action * action_scale

                # Clip to valid range
                action = np.clip(action, self.robot.action_min, self.robot.action_max)
                trajectory['actions'].append(action)

                # Execute action on robot
                self.robot.execute_action(action)

                # Observe new state
                new_obs = self.robot.get_observation()
                trajectory['observations'].append(new_obs)
                current_obs = new_obs.to(self.device).unsqueeze(0)

                # Check for early termination signals
                if self.should_terminate(current_obs, instruction):
                    trajectory['success'] = True
                    break

        trajectory['num_steps'] = len(trajectory['actions'])
        return trajectory

    def should_terminate(self, obs, instruction):
        """
        Heuristic termination check: detect if task is complete.
        Can be customized per task or learned from data.
        """
        # Placeholder: implement task-specific success detection
        # Examples: gripper closed, object at goal location, etc.
        return False
```

## Practical Guidance

### Hyperparameters and Training Configuration

| Component | Parameter | Recommended Value | Notes |
|-----------|-----------|-------------------|-------|
| Understanding Expert | Learning Rate (Stage II) | 2e-5 | Frozen in Stage I, unfrozen in Stage II and beyond |
| Generation Expert | Learning Rate (Stage I) | 1e-4 | Higher initial rate; lower to 2e-5 in Stage II |
| Action Expert | Learning Rate (Stage II) | 2e-5 | Initialized from scratch; same rate as other experts |
| Fine-tuning (Stage III) | Learning Rate | 5e-6 | Much lower for task-specific adaptation |
| Action Loss | Flow Matching Steps | 1000 | Number of diffusion timesteps for flow matching schedule |
| Gradient Clipping | Max Norm | 1.0 | Prevents training instability |
| Fine-tuning Gradient Clipping | Max Norm | 0.5 | Tighter clipping for stability in Stage III |
| VQ-VAE Tokenization | Vocabulary Size | 8192 | Typical for image reconstruction quality |
| Transformer Backbone | Model Size | 4.2B parameters | Gemma-based architecture |
| Training Data (Stage II) | Trajectory Count | 330,000+ | Across 136 diverse manipulation tasks |
| Batch Size (Stage II) | | 32-64 | Depends on GPU memory; 4 A100s tested |
| Batch Size (Stage III) | | 8-16 | Smaller for fine-tuning stability |
| Foresight Prediction Scales | Number of Scales | 4 | Coarse-to-fine hierarchical prediction |

### When to Use F1

Use F1 when you need:

- **Planning in dynamic environments**: Tasks where reactive policies fail because the future is uncertain and action effects compound over time.
- **Long-horizon sequential manipulation**: Multi-step tasks like pick-place-insert where intermediate visual milestones guide decisions.
- **Transfer across embodiments**: Shared understanding and generation experts enable rapid adaptation to new robot morphologies.
- **Interpretable planning**: Predicted future frames provide explicit intermediate goals, making the model's reasoning inspectable.
- **Real-time closed-loop control**: The three-expert architecture enables streaming prediction with minimal latency.

F1 achieved 82.2% average success on real-world manipulation and ranked first on all four LIBERO simulation benchmarks.

### When NOT to Use F1

Do not use F1 when:

- **Task requires discrete symbolic reasoning**: F1 is not designed for high-level planning or symbolic state machines. Use classical planners or hybrid frameworks if you need explicit symbolic reasoning.
- **Visual observations are unreliable or heavily occluded**: The entire approach depends on accurate visual feedback. If sensors are noisy or environments heavily occluded, ensure robust perception preprocessing.
- **Computational resources are severely limited**: 4.2B parameters and multi-scale VQ-VAE encoding require significant GPU memory (tested on 4x A100). Smaller models may be necessary for edge deployment.
- **Actions are purely non-visual**: If the task requires actions driven by non-visual state (e.g., pure force feedback, proprioception-based balance), F1's vision-centric foresight may be insufficient.
- **Training data is scarce**: Stage II pretraining on 330k+ trajectories is crucial. With <10k task-specific trajectories, overfitting is likely; consider data augmentation or smaller models.
- **Real-time safety is critical**: Foresight prediction adds latency. If millisecond-level reaction is needed (e.g., collision avoidance), reactive models may be more appropriate.

### Common Pitfalls

1. **Skipping Stage I Alignment**: Stage I initializes the generation expert to produce valid visual continuations. Training without this warm-start causes the generation expert to output incoherent or static frames, degrading downstream action prediction.

2. **Insufficient VQ-VAE Pretraining**: The residual VQ-VAE must be well-trained before Stage II. Poor tokenization introduces reconstruction error that propagates through the pipeline. Verify reconstruction quality on validation frames before starting joint pretraining.

3. **Misaligned Multi-Scale Tokenization**: The hierarchical prediction across four scales relies on consistent downsampling. Ensure coarse and fine tokens are aligned (e.g., coarse token at (H/4, W/4) corresponds to a 4x4 patch in the fine tokens).

4. **Ignoring Action Space Normalization in Stage III**: Each embodiment has different action ranges and dynamics. Failing to normalize actions per task leads to poor fine-tuning performance. Always compute and apply task-specific action statistics.

5. **Overfitting on Small Task-Specific Datasets**: Stage III uses a lower learning rate and early stopping, but if the task dataset is <5k trajectories, consider freezing the understanding expert and only fine-tuning the generation and action experts to reduce parameter count.

6. **Not Monitoring Foresight Quality**: Predicted frames that drift far from reality will mislead the action expert. During training and validation, visualize predicted futures and check for hallucinations or mode collapse.

7. **Treating Flow Matching Loss as Standard MSE**: Flow matching is not equivalent to predicting a fixed target frame. The loss formulation involves interpolation between noise and target over diffusion steps. Using plain L2 loss will not converge properly.

## Reference

**Paper**: F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions
**Authors**: InternRobotics / Alibaba DAMO Academy
**ArXiv**: https://arxiv.org/abs/2509.06951
**Code Repository**: https://github.com/InternRobotics/F1-VLA
**Model Weights**: https://huggingface.co/InternRobotics/F1-VLA

**Key Results**:
- Real-world manipulation: 82.2% success rate (vs. π0 baseline: 65.2%)
- LIBERO simulation: Ranked 1st across all four benchmark suites
- Strong performance on dynamic environments and long-horizon sequential tasks

For the full technical details, loss function derivations, and experimental ablations, see the full paper at https://arxiv.org/abs/2509.06951.
