---
name: latent-particle-world-models
title: "Latent Particle World Models: Self-Supervised Object-Centric Stochastic Dynamics Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.04553"
keywords: [World Models, Object-Centric Learning, Latent Dynamics, Stochastic Generation, Video Understanding]
description: "Learn world models by decomposing scenes into latent particles with per-particle dynamics. Each particle represents an object with position, scale, and appearance. Learn distributed latent actions governing per-particle transitions, enabling multimodal video generation from identical initial conditions."
---

# Latent Particle World Models: Object-Centric Stochastic Dynamics

Traditional world models predict full video frames (wasteful) or use global latent actions (insufficient for multi-entity scenes). Latent Particle World Models (LPWM) decomposes video into object-centric particles with independent dynamics. The key innovation is per-particle latent actions: instead of a single global action determining the full scene, each particle has its own latent action distribution, enabling rich multi-entity interactions and genuine stochastic behavior.

The core insight is that complex scenes with multiple agents naturally decompose into independent actors, each with their own latent policy. This enables realistic multi-modal generation where identical initial conditions lead to different outcomes (e.g., multiple agents choosing different actions).

## Core Concept

LPWM operates through three coordinated mechanisms:

1. **Object-Centric Decomposition**: Segment video scenes into latent particles, each representing an object or agent
2. **Per-Particle Latent Actions**: Learn distributions over latent actions specific to each particle, not global to the scene
3. **Flexible Conditioning**: Support diverse control modalities (language goals, external actions, user instructions) by mapping to per-particle latent actions

## Architecture Overview

- **Input**: Video sequences or action-annotated trajectories
- **Scene Decomposition**: Latent VAE decomposes scenes into particles (position, scale, appearance)
- **Dynamics Model**: Temporal transitions with per-particle learned policies
- **Latent Action Sampler**: Sample per-particle actions from learned distributions
- **Reconstruction**: Render predicted particle states back to image space
- **Output**: Multimodal video predictions, controllable generation

## Implementation Steps

**Step 1: Design particle representation**

Define compact per-particle state encoding.

```python
class Particle:
    """
    Represents a single object/entity in a scene.
    """

    def __init__(self, position, scale, appearance, opacity=1.0):
        """
        position: (x, y) coordinates (normalized 0-1)
        scale: (h, w) relative to image size
        appearance: learned embedding (e.g., 16-dim vector)
        opacity: alpha channel (0-1)
        """
        self.position = position
        self.scale = scale
        self.appearance = appearance
        self.opacity = opacity

    def to_state_vector(self):
        """Concatenate into single state vector."""
        return np.concatenate([
            self.position,  # 2D
            self.scale,  # 2D
            self.appearance,  # 16D
            [self.opacity]  # 1D
        ])  # Total: 21D per particle

    @classmethod
    def from_state_vector(cls, state_vector):
        """Reconstruct particle from state vector."""
        pos = state_vector[:2]
        scale = state_vector[2:4]
        app = state_vector[4:20]
        opacity = state_vector[20]
        return cls(pos, scale, app, opacity)

class SceneDecomposition:
    """Decompose video frames into particles."""

    def __init__(self, num_particles=4, video_vae=None):
        self.num_particles = num_particles
        self.video_vae = video_vae  # Pretrained VAE encoder

    def decompose_frame(self, frame):
        """
        Extract particles from a single frame.
        """
        # Encode frame to latent space
        frame_latent = self.video_vae.encode(frame)

        # Slot attention: decompose latent into K particles
        # (Implementation uses pretrained slot attention module)
        particle_slots = self._slot_attention(frame_latent, self.num_particles)

        # Decode each slot to particle representation
        particles = []
        for slot in particle_slots:
            # Decode slot to (position, scale, appearance, opacity)
            pos = self._decode_position(slot)
            scale = self._decode_scale(slot)
            appearance = self._decode_appearance(slot)
            opacity = self._decode_opacity(slot)

            particles.append(Particle(pos, scale, appearance, opacity))

        return particles

    def _slot_attention(self, latent, num_slots):
        """Decompose latent into K slot representations."""
        # Use pretrained slot attention (e.g., from SA-1B)
        return np.random.randn(num_slots, 64)  # Placeholder

    def _decode_position(self, slot):
        """Extract position coordinates from slot."""
        return (slot[0] % 1.0, slot[1] % 1.0)

    def _decode_scale(self, slot):
        """Extract scale from slot."""
        return (0.1 + (slot[2] % 1.0) * 0.8, 0.1 + (slot[3] % 1.0) * 0.8)

    def _decode_appearance(self, slot):
        """Extract appearance embedding from slot."""
        return slot[4:20]

    def _decode_opacity(self, slot):
        """Extract opacity from slot."""
        return 0.5 + (slot[20] % 1.0) * 0.5
```

**Step 2: Model per-particle latent action distributions**

Define learned stochastic policies for each particle.

```python
class ParticleLatentPolicy:
    """
    Learn distribution over latent actions for a specific particle type.
    """

    def __init__(self, latent_action_dim=4):
        self.latent_action_dim = latent_action_dim

        # Policy network: state → latent action distribution
        self.policy_net = nn.Sequential(
            nn.Linear(21, 64),  # Particle state
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_action_dim * 2)  # mean + log_std
        )

    def sample_action(self, particle_state):
        """
        Sample latent action for this particle.
        Returns: latent action vector (e.g., 4D)
        """
        # Forward through policy net
        output = self.policy_net(particle_state)
        mean = output[:self.latent_action_dim]
        log_std = output[self.latent_action_dim:]

        # Reparameterization trick for differentiability
        std = torch.exp(log_std)
        epsilon = torch.randn_like(std)
        latent_action = mean + std * epsilon

        return latent_action

    def compute_logprob(self, particle_state, latent_action):
        """Compute log-probability of action under this policy."""
        output = self.policy_net(particle_state)
        mean = output[:self.latent_action_dim]
        log_std = output[self.latent_action_dim:]

        std = torch.exp(log_std)

        # Gaussian log-prob
        logprob = -0.5 * ((latent_action - mean) / std) ** 2 - log_std
        return logprob.sum()

class DynamicsModel:
    """
    Update particle states based on latent actions.
    z_i^{t+1} = f(z_i^t, a_i^t, context)
    """

    def __init__(self, state_dim=21, latent_action_dim=4):
        self.state_dim = state_dim
        self.latent_action_dim = latent_action_dim

        # Transition function: state + action → new state
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + latent_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

    def predict_next_state(self, particle_state, latent_action):
        """Predict next particle state given current state and action."""
        input_vector = torch.cat([particle_state, latent_action], dim=-1)
        state_delta = self.transition_net(input_vector)

        # Residual update: next_state = current + delta
        next_state = particle_state + state_delta

        # Clip to valid ranges (e.g., position in [0, 1])
        next_state = self._apply_constraints(next_state)

        return next_state

    def _apply_constraints(self, state):
        """Enforce valid particle state ranges."""
        # Position should be in [0, 1]
        state[:, :2] = torch.clamp(state[:, :2], 0, 1)

        # Scale should be in [0.05, 0.95]
        state[:, 2:4] = torch.clamp(state[:, 2:4], 0.05, 0.95)

        # Opacity in [0, 1]
        state[:, -1] = torch.clamp(state[:, -1], 0, 1)

        return state
```

**Step 3: Implement scene rendering from particles**

Convert particle states back to image space.

```python
class ParticleRenderer:
    """Render particles back to image space."""

    def __init__(self, image_size=64):
        self.image_size = image_size

    def render_scene(self, particles):
        """
        Compose particles into single image.
        """
        canvas = np.zeros((self.image_size, self.image_size, 3))

        # Sort particles by z-order (appearance encoding determines depth)
        sorted_particles = sorted(particles,
                                 key=lambda p: np.mean(p.appearance))

        # Composite particles with alpha blending
        for particle in sorted_particles:
            canvas = self._composite_particle(canvas, particle)

        return canvas

    def _composite_particle(self, canvas, particle):
        """Composite single particle onto canvas with alpha blending."""
        # Render particle texture (learned appearance → texture)
        texture = self._appearance_to_texture(particle.appearance)

        # Determine bounding box based on position and scale
        center_x = int(particle.position[0] * self.image_size)
        center_y = int(particle.position[1] * self.image_size)
        h = int(particle.scale[0] * self.image_size)
        w = int(particle.scale[1] * self.image_size)

        x_min = max(0, center_x - w // 2)
        x_max = min(self.image_size, center_x + w // 2)
        y_min = max(0, center_y - h // 2)
        y_max = min(self.image_size, center_y + h // 2)

        # Alpha blend
        if x_max > x_min and y_max > y_min:
            texture_region = texture[:h, :w]
            alpha = particle.opacity

            canvas[y_min:y_max, x_min:x_max] = \
                (1 - alpha) * canvas[y_min:y_max, x_min:x_max] + \
                alpha * texture_region[:y_max - y_min, :x_max - x_min]

        return canvas

    def _appearance_to_texture(self, appearance_embedding):
        """Convert appearance embedding to rendered texture."""
        # Simple: use embedding to generate texture via learned decoder
        # (In practice: use conditional GAN or diffusion model)
        np.random.seed(hash(appearance_embedding.tobytes()) % (2 ** 32))
        return np.random.randn(32, 32, 3) * 0.5 + 0.5  # Placeholder
```

**Step 4: Define training objective**

Train to reconstruct videos and match target distributions.

```python
def compute_lpwm_loss(model, video_batch, latent_action_dim=4):
    """
    LPWM training loss combining reconstruction and KL divergence.
    """
    batch_size, seq_len, H, W, C = video_batch.shape

    # Decompose first frame
    first_particles = model.decomposer.decompose_frame(video_batch[:, 0])

    total_loss = 0.0

    for t in range(seq_len - 1):
        current_frame = video_batch[:, t]
        target_frame = video_batch[:, t + 1]

        # Decompose current frame
        particles = model.decomposer.decompose_frame(current_frame)

        # Sample per-particle latent actions
        latent_actions = []
        log_probs = []

        for i, particle in enumerate(particles):
            particle_state = torch.tensor(particle.to_state_vector(), dtype=torch.float32)

            # Sample action from learned policy
            action = model.policies[i].sample_action(particle_state)
            latent_actions.append(action)

            # Track log-prob for policy loss
            logprob = model.policies[i].compute_logprob(particle_state, action)
            log_probs.append(logprob)

        # Predict next particles
        next_particles = []
        for i, (particle, action) in enumerate(zip(particles, latent_actions)):
            particle_state = torch.tensor(particle.to_state_vector(), dtype=torch.float32)
            next_state = model.dynamics.predict_next_state(particle_state, action)

            next_particle = Particle.from_state_vector(next_state.detach().numpy())
            next_particles.append(next_particle)

        # Render predicted next frame
        predicted_frame = model.renderer.render_scene(next_particles)

        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(
            torch.tensor(predicted_frame),
            torch.tensor(target_frame)
        )

        # KL divergence loss (encourage learned policies)
        kl_loss = -torch.mean(torch.stack(log_probs))

        total_loss += recon_loss + 0.01 * kl_loss

    return total_loss / seq_len
```

**Step 5: Generation and evaluation**

Generate multimodal videos and benchmark on control tasks.

```python
def generate_multimodal_video(model, initial_frame, num_steps=10, num_samples=4):
    """
    Generate multiple video trajectories from identical initial frame.
    Demonstrates stochasticity from per-particle latent actions.
    """
    # Decompose initial frame once
    initial_particles = model.decomposer.decompose_frame(initial_frame)

    trajectories = []

    for sample_idx in range(num_samples):
        particles = initial_particles
        frames = [initial_frame]

        for step in range(num_steps):
            # Sample different actions for this trajectory
            latent_actions = [
                model.policies[i].sample_action(
                    torch.tensor(p.to_state_vector(), dtype=torch.float32)
                )
                for i, p in enumerate(particles)
            ]

            # Predict next particles
            next_particles = [
                Particle.from_state_vector(
                    model.dynamics.predict_next_state(
                        torch.tensor(p.to_state_vector(), dtype=torch.float32),
                        action
                    ).detach().numpy()
                )
                for p, action in zip(particles, latent_actions)
            ]

            # Render
            next_frame = model.renderer.render_scene(next_particles)
            frames.append(next_frame)

            particles = next_particles

        trajectories.append(frames)

    return trajectories
```

## Practical Guidance

**Hyperparameter Selection:**
- **Number of particles**: 2-6. More = finer detail; higher computational cost.
- **Latent action dimension**: 2-8. Higher = more expressive per-particle policies.
- **Appearance embedding dimension**: 8-16. Larger = more visual diversity.
- **KL weight**: 0.001-0.1. Higher = stronger stochasticity pressure.

**When to Use:**
- Multi-agent or multi-entity scene modeling
- Stochastic video generation where different outcomes are valid
- Controllable generation with per-entity control
- Settings requiring object-centric representations

**When NOT to Use:**
- Single-object scenes (global latent actions sufficient)
- Deterministic environments where stochasticity is harmful
- Real-time systems requiring fast generation
- Scenes with many small, indistinct objects (decomposition becomes ambiguous)

**Common Pitfalls:**
- **Slot collapse**: Some particle slots become inactive. Add auxiliary loss encouraging slot diversity.
- **Poor appearance learning**: If appearance embeddings are uninformative, increase embedding dimension or add contrastive loss.
- **Policy convergence**: Latent action policies may become deterministic (log_std → -∞). Add entropy regularization.
- **Rendering artifacts**: Alpha compositing can cause color oversaturation. Use pre-multiplied alpha or other blending modes.

## Reference

arXiv: https://arxiv.org/abs/2603.04553
