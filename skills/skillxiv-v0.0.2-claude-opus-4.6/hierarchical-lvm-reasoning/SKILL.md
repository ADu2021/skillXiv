---
name: hierarchical-lvm-reasoning
title: "Chain of World: World Model Thinking in Latent Motion"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.03195"
keywords: [World Models, Latent Reasoning, Structure-Motion Disentanglement, Video Understanding, Planning]
description: "Reason about long-horizon dynamics by disentangling structure and motion in video VAE latents. Learn continuous latent motion chains that preserve temporal coherence while predicting terminal keyframes, enabling efficient reasoning about multi-step scenarios."
---

# Chain of World: Latent Motion Reasoning for Long-Horizon Planning

Traditional world models either predict dense frame sequences (computationally expensive) or use discrete action sequences (temporally discontinuous). Chain of World introduces a middle ground: learn continuous latent motion representations extracted from video VAE latents, then reason about multi-step trajectories as chains of motion in latent space. This approach preserves temporal structure while dramatically reducing computational requirements.

The core insight is to decompose each video segment into separable structure (static semantic content) and motion (how objects move). Reasoning then operates on compact continuous motion representations rather than full frames or discrete actions.

## Core Concept

Chain of World implements three coordinated mechanisms:

1. **Structure-Motion Disentanglement**: Use pretrained video VAE to factorize scene into static structure latents and dynamic motion latents
2. **Continuous Latent Motion Chains**: Learn to predict chains of motion latents, which implicitly define object trajectories
3. **Terminal Keyframe Prediction**: Predict end-state visuals from motion chains, reconstructing the trajectory from sparse keyframes

## Architecture Overview

- **Input**: Video segments or action sequences with visual context
- **Video VAE Encoder**: Decompose into structure (global semantics) and motion (directional dynamics)
- **Motion Chain Generator**: Predict sequences of motion latents given instruction
- **Keyframe Decoder**: Convert predicted motion chains to visual output
- **Output**: Structured representations of multi-step scenarios

## Implementation Steps

**Step 1: Implement structure-motion decomposition**

Extract separable structure and motion from pretrained video VAE.

```python
class VideoVAEDecomposer:
    """
    Decompose video into structure and motion latents using pretrained VAE.
    """

    def __init__(self, pretrained_vae_path):
        """Load pretrained video VAE (e.g., VideoMAE)."""
        self.vae = load_pretrained_vae(pretrained_vae_path)
        self.structure_dim = 32
        self.motion_dim = 32

    def decompose_segment(self, video_segment):
        """
        Decompose video segment into structure and motion.

        video_segment: shape (T, H, W, 3) — T frames of video
        Returns:
            structure_latent: (1, 32) — global semantic content
            motion_latents: (T-1, 32) — per-frame motion
        """
        # Encode full segment
        segment_latent = self.vae.encode(video_segment)  # Shape: (1, 64)

        # Split into structure and motion components
        structure_latent = segment_latent[:, :self.structure_dim]

        # Motion: computed via optical flow-like differencing in latent space
        motion_latents = []
        for t in range(len(video_segment) - 1):
            # Encode consecutive frames
            frame_t = self.vae.encode(video_segment[t:t+1])
            frame_t1 = self.vae.encode(video_segment[t+1:t+2])

            # Motion is frame-to-frame difference
            motion_t = frame_t1 - frame_t
            motion_latents.append(motion_t[:, self.structure_dim:])

        motion_latents = np.stack(motion_latents, axis=1).squeeze(0)

        return structure_latent, motion_latents

    def extract_motion_direction(self, motion_latent):
        """
        Extract directional components (e.g., velocity) from motion latent.
        """
        # Normalize to unit direction vector
        magnitude = np.linalg.norm(motion_latent)
        if magnitude > 1e-6:
            direction = motion_latent / magnitude
        else:
            direction = np.zeros_like(motion_latent)

        return direction, magnitude
```

**Step 2: Model continuous latent motion chains**

Learn to generate sequences of motion latents that form coherent trajectories.

```python
class LatentMotionChainModel(nn.Module):
    """
    Generate continuous chains of motion latents.
    Input: initial structure + goal/instruction
    Output: sequence of motion latents describing trajectory
    """

    def __init__(self, structure_dim=32, motion_dim=32, max_chain_length=10):
        super().__init__()
        self.structure_dim = structure_dim
        self.motion_dim = motion_dim
        self.max_chain_length = max_chain_length

        # Encoder: instruction/goal → latent goal representation
        self.goal_encoder = nn.Sequential(
            nn.Linear(768, 256),  # Assume goal from text encoder
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Motion chain generator: recurrent model generating motion latents
        self.motion_gru = nn.GRU(
            input_size=structure_dim + 128,  # structure + goal encoding
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        # Output projection: hidden state → motion latent
        self.motion_head = nn.Linear(128, motion_dim)

        # Length predictor: predict chain length
        self.length_predictor = nn.Sequential(
            nn.Linear(128 + structure_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_chain_length)
        )

    def forward(self, structure_latent, goal_embedding, instruction_text=None):
        """
        Generate motion latent chain.

        structure_latent: (batch, 32) — static scene content
        goal_embedding: (batch, 128) — encoded goal representation
        Returns:
            motion_chain: (batch, T, 32) — sequence of motion latents
            predicted_length: (batch, 1) — predicted chain length
        """
        batch_size = structure_latent.shape[0]

        # Predict chain length from goal + structure
        combined = torch.cat([structure_latent, goal_embedding], dim=1)
        length_logits = self.length_predictor(combined)
        predicted_length = torch.argmax(length_logits, dim=1) + 1  # +1 because 0 means length 1

        # Initialize GRU with combined input
        input_sequence = torch.cat(
            [structure_latent.unsqueeze(1).expand(-1, self.max_chain_length, -1),
             goal_embedding.unsqueeze(1).expand(-1, self.max_chain_length, -1)],
            dim=2
        )

        # Generate motion chain via GRU
        gru_output, _ = self.motion_gru(input_sequence)

        # Project to motion latents
        motion_chain = self.motion_head(gru_output)  # (batch, T, 32)

        return motion_chain, predicted_length

    def generate_trajectory_from_chain(self, motion_chain, structure_latent):
        """
        Integrate motion chain into object trajectories.
        """
        # Cumulative sum of motion latents represents cumulative motion
        trajectories = torch.cumsum(motion_chain, dim=1)

        return trajectories
```

**Step 3: Implement motion-aware co-fine-tuning**

Align continuous latent dynamics with discrete action labels.

```python
class MotionActionAligner:
    """
    Co-fine-tune: align learned latent motion chains with discrete actions.
    """

    def __init__(self, motion_model, vae, action_vocabulary_size=10):
        self.motion_model = motion_model
        self.vae = vae
        self.action_vocabulary_size = action_vocabulary_size

        # Action encoder: discrete action → motion latent prototype
        self.action_embedder = nn.Embedding(action_vocabulary_size, 32)

        # Alignment loss
        self.alignment_loss = nn.MSELoss()

    def co_finetune_step(self, structure_latent, goal_embedding, action_sequence):
        """
        Single co-fine-tuning step.
        action_sequence: list of discrete action indices
        """
        # Generate motion chain from goal
        motion_chain, _ = self.motion_model(structure_latent, goal_embedding)

        # Expected motion from action sequence
        action_embeddings = [
            self.action_embedder(torch.tensor(action))
            for action in action_sequence
        ]
        expected_motion = torch.stack(action_embeddings)  # (T, 32)

        # Alignment: minimize distance between predicted and expected motion
        # Truncate motion chain to match action sequence length
        T = len(action_sequence)
        predicted_motion = motion_chain[:, :T, :]  # (batch, T, 32)

        loss = self.alignment_loss(predicted_motion, expected_motion.unsqueeze(0))

        return loss

    def train_alignment(self, train_trajectories, num_epochs=5):
        """
        Train alignment between latent motion and discrete actions.
        train_trajectories: list of (structure, goal, action_sequence) tuples
        """
        optimizer = torch.optim.Adam(self.motion_model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for structure, goal, actions in train_trajectories:
                loss = self.co_finetune_step(structure, goal, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_trajectories)
            print(f"Epoch {epoch + 1}: Alignment loss = {avg_loss:.4f}")
```

**Step 4: Keyframe-to-video reconstruction**

Convert motion chains back to visual predictions.

```python
class KeyframeToVideoDecoder:
    """
    Reconstruct video from sparse keyframes using motion chains.
    """

    def __init__(self, vae, structure_dim=32, motion_dim=32):
        self.vae = vae
        self.structure_dim = structure_dim
        self.motion_dim = motion_dim

        # Interpolation network: given keyframes + motion, predict intermediate
        self.motion_interpolator = nn.Sequential(
            nn.Linear(structure_dim + motion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, motion_dim)
        )

    def reconstruct_from_motion_chain(self, structure_latent, motion_chain,
                                     keyframe_indices=None):
        """
        Reconstruct full video from sparse keyframes and motion chain.

        structure_latent: (1, 32) — scene content
        motion_chain: (T, 32) — predicted motion sequence
        keyframe_indices: indices of keyframes (e.g., [0, T-1])
        """
        T = motion_chain.shape[0]

        if keyframe_indices is None:
            keyframe_indices = [0, T - 1]

        # Cumulative motion integration
        trajectories = torch.cumsum(motion_chain, dim=0)

        # Predict each frame by integrating motion
        frames = []
        for t in range(T):
            # Current motion applied to structure
            cumulative_motion = trajectories[t]

            # Combine structure + cumulative motion
            combined_latent = torch.cat([
                structure_latent.squeeze(0),
                cumulative_motion
            ])

            # Decode to image space
            frame = self.vae.decode(combined_latent.unsqueeze(0))
            frames.append(frame)

        return torch.cat(frames, dim=0)  # (T, H, W, 3)
```

**Step 5: Training and evaluation**

Train end-to-end with motion chain objectives.

```python
def train_chain_of_world(vae, train_videos, instructions, actions,
                        num_epochs=10):
    """
    Train Chain of World model.
    """
    motion_model = LatentMotionChainModel()
    aligner = MotionActionAligner(motion_model, vae)
    decoder = KeyframeToVideoDecoder(vae)

    optimizer = torch.optim.Adam(
        list(motion_model.parameters()) +
        list(aligner.action_embedder.parameters()),
        lr=1e-4
    )

    for epoch in range(num_epochs):
        total_loss = 0.0

        for video, instruction, action_seq in zip(train_videos, instructions, actions):
            # Decompose video
            structure_latent, motion_latents = vae_decomposer.decompose_segment(video)

            # Encode instruction
            goal_embedding = encode_instruction(instruction)

            # Generate motion chain
            motion_chain, pred_length = motion_model(
                torch.tensor(structure_latent),
                torch.tensor(goal_embedding)
            )

            # Reconstruction loss: reconstruct video from motion chain
            reconstructed_frames = decoder.reconstruct_from_motion_chain(
                torch.tensor(structure_latent),
                motion_chain,
                keyframe_indices=[0, len(video) - 1]
            )

            recon_loss = torch.nn.functional.mse_loss(
                reconstructed_frames,
                torch.tensor(video)
            )

            # Alignment loss: motion chain should align with actions
            alignment_loss = aligner.co_finetune_step(
                torch.tensor(structure_latent),
                torch.tensor(goal_embedding),
                action_seq
            )

            loss = recon_loss + 0.5 * alignment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_videos)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    return motion_model

def evaluate_on_planning_tasks(motion_model, planning_tasks):
    """
    Evaluate: can model plan multi-step scenarios?
    """
    success_count = 0

    for task in planning_tasks:
        initial_state = task['initial_state']
        goal_state = task['goal_state']
        expected_actions = task['expected_actions']

        # Encode goal
        goal_embedding = encode_instruction(task['goal_instruction'])

        # Generate motion chain
        structure_latent = vae.encode_state(initial_state)
        motion_chain, _ = motion_model(structure_latent, goal_embedding)

        # Check if motion chain enables reaching goal
        final_state = integrate_motion(initial_state, motion_chain)

        if state_similarity(final_state, goal_state) > 0.8:
            success_count += 1

    success_rate = success_count / len(planning_tasks)
    print(f"Planning task success rate: {success_rate * 100:.1f}%")

    return success_rate
```

## Practical Guidance

**Hyperparameter Selection:**
- **Structure/motion dimension split**: 16/16, 20/12, or 24/8 depending on scene complexity
- **Max chain length**: 5-15 steps; longer = more complex scenarios but harder training
- **Motion GRU layers**: 1-3; more layers capture longer-range dependencies
- **Co-fine-tuning weight**: 0.3-0.7; balance reconstruction vs. action alignment

**When to Use:**
- Long-horizon planning and reasoning tasks
- Scenarios where compact temporal reasoning is beneficial
- Multi-step visual prediction with sparse supervision
- Tasks requiring integration of motion information

**When NOT to Use:**
- Single-frame tasks (no temporal structure to leverage)
- Real-time applications (latent decomposition is expensive)
- Domains with very complex, non-linear motion
- Tasks where discrete actions are insufficient

**Common Pitfalls:**
- **Motion chain explosion**: Cumulative integration can diverge. Clip motion magnitudes or use bounded integration.
- **Structure-motion entanglement**: If decomposition doesn't separate cleanly, add auxiliary loss encouraging orthogonality.
- **Action alignment failure**: If discrete actions don't map well to latent motions, use curriculum learning (start with simple actions).
- **Keyframe reconstruction artifacts**: Sparse keyframe interpolation can be unreliable. Use denser keyframes or flow-based interpolation.

## Reference

arXiv: https://arxiv.org/abs/2603.03195
