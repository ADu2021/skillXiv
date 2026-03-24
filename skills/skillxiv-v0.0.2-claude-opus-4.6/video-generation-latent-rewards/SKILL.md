---
name: video-generation-latent-rewards
title: "Video Generation Models Are Good Latent Reward Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.21541"
keywords: [Video Reward Models, Latent Space Evaluation, Process Rewards, Video Quality Assessment, Diffusion Models]
description: "Use pre-trained video generation models (VGMs) as efficient reward models by evaluating video quality directly in latent space at any denoising timestep, enabling process-reward learning across the entire generation trajectory without expensive VAE decoding to RGB."
---

# Video Generation Models as Latent Reward Models

Training reward models for video generation typically requires expensive evaluation—decoding fully to RGB, computing VLM scores, or maintaining separate quality classifiers. This skill demonstrates how to repurpose pre-trained video generation models themselves as efficient reward evaluators by operating directly on latent representations at intermediate denoising steps. This approach is 1.4× faster than traditional methods while providing richer, process-aware supervision signals.

The core insight is that video generation models are inherently designed to process noisy latent representations at arbitrary timesteps, making them naturally suited for evaluating motion and structure formation during generation.

## Core Concept

Process-Aware Video Reward Modeling (PAVRM) evaluates video quality directly in latent space:

1. **Latent Representation Extraction**: Use intermediate features from the VGM's denoising process rather than fully decoded RGB frames
2. **Learnable Compression**: Query vectors compress variable-length spatiotemporal features into compact quality-aware tokens
3. **Timestep-Aware Evaluation**: Evaluate at any denoising step, providing process rewards that guide generation trajectory
4. **Process Reward Feedback Learning (PRFL)**: Distribute learning signals across the entire generation path rather than optimizing only final outputs

## Architecture Overview

- **VGM Feature Extraction**: Access intermediate activations from pre-trained video generation model
- **Latent Feature Aggregation**: Compress spatiotemporal features from any denoising timestep
- **Query-Based Compression**: Learnable vectors extract quality-relevant information
- **Process Reward Head**: Maps compressed features to scalar quality scores
- **Timestep Sampling**: Randomly sample diffusion steps for training, enabling early-stage supervision

## Implementation Steps

The reward model architecture operates on latent representations extracted from any diffusion timestep.

**1. Extract Features from VGM**

Access intermediate representations from the video generation model at any denoising step.

```python
def extract_vgm_features(vgm_model, video_latents, timestep, layer_name='decoder_features'):
    """
    Extract intermediate features from VGM at specified denoising timestep.
    These latent representations retain spatial-temporal structure without RGB decoding.
    """
    # Register hook to capture features
    features_captured = {}

    def capture_features(module, input, output):
        features_captured[layer_name] = output.detach()

    hook = vgm_model.get_layer(layer_name).register_forward_hook(capture_features)

    # Forward pass at specified timestep
    with torch.no_grad():
        vgm_model.denoise_step(video_latents, timestep)

    hook.remove()

    return features_captured[layer_name]
```

**2. Implement Learnable Query-Based Compression**

Use learnable query vectors to compress variable-length spatiotemporal features into compact tokens.

```python
class LatentRewardCompressor(torch.nn.Module):
    """
    Compress variable-length spatiotemporal features into quality-aware tokens.
    Uses learnable queries to extract salient information without full reconstruction.
    """
    def __init__(self, feature_dim=768, num_queries=8, hidden_dim=512):
        super().__init__()

        self.num_queries = num_queries
        self.feature_dim = feature_dim

        # Learnable query vectors
        self.query_vectors = torch.nn.Parameter(
            torch.randn(1, num_queries, feature_dim) / feature_dim ** 0.5
        )

        # Cross-attention to aggregate features
        self.attention = torch.nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )

        # Compression network
        self.compression_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * num_queries, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 128)  # Compact representation
        )

    def forward(self, features):
        """
        Compress features using cross-attention with learnable queries.
        Args:
            features: (batch, time*height*width, feature_dim)
        Returns:
            compressed: (batch, 128) compact quality-aware representation
        """
        batch_size = features.shape[0]

        # Expand queries for batch
        queries = self.query_vectors.expand(batch_size, -1, -1)

        # Cross-attention: compress variable-length features into query responses
        attended, _ = self.attention(queries, features, features)

        # Flatten and compress
        compressed = attended.reshape(batch_size, -1)
        compressed = self.compression_net(compressed)

        return compressed
```

**3. Implement Process Reward Head**

Map compressed features to scalar quality scores.

```python
class ProcessRewardHead(torch.nn.Module):
    """
    Predict video quality scores from compressed latent representations.
    Outputs continuous quality scores and confidence estimates.
    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()

        self.quality_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

        self.confidence_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()  # Confidence in [0, 1]
        )

    def forward(self, compressed_features):
        """
        Predict quality score and confidence from compressed features.
        Returns:
            quality: (batch,) scalar quality scores
            confidence: (batch,) confidence estimates [0, 1]
        """
        quality = self.quality_head(compressed_features).squeeze(-1)
        confidence = self.confidence_head(compressed_features).squeeze(-1)

        return quality, confidence
```

**4. Build Complete Latent Reward Model**

Integrate feature extraction, compression, and reward head into single model.

```python
class LatentVideoRewardModel(torch.nn.Module):
    """
    Complete latent reward model for video generation guidance.
    Operates on VGM intermediate features without RGB decoding.
    """
    def __init__(self, vgm_model, feature_dim=768):
        super().__init__()

        self.vgm_model = vgm_model
        self.compressor = LatentRewardCompressor(feature_dim=feature_dim)
        self.reward_head = ProcessRewardHead(input_dim=128)

    def forward(self, video_latents, timestep):
        """
        Compute quality reward from latent representations at any timestep.
        Args:
            video_latents: (batch, time, channels, height, width)
            timestep: diffusion timestep (int)
        Returns:
            rewards: (batch,) scalar rewards
            confidence: (batch,) confidence scores
        """
        # Extract VGM features at this timestep
        features = extract_vgm_features(self.vgm_model, video_latents, timestep)

        # Reshape for compression (flatten spatial-temporal dimensions)
        batch, time, channels, height, width = features.shape
        features_flat = features.reshape(batch, time * height * width, channels)

        # Compress using learned queries
        compressed = self.compressor(features_flat)

        # Predict quality and confidence
        rewards, confidence = self.reward_head(compressed)

        return rewards, confidence
```

**5. Implement Process Reward Feedback Learning (PRFL)**

Train the reward model by randomly sampling diffusion steps and supervising at intermediate timesteps.

```python
def train_process_reward_model(reward_model, vgm_model, video_batch, gold_videos, optimizer, num_steps=10):
    """
    Train latent reward model using process-aware supervision.
    Randomly samples denoising timesteps to distribute learning across generation trajectory.
    """
    losses = []

    for step in range(num_steps):
        # Sample random timestep for training
        timestep = torch.randint(1, 1000, (1,)).item()

        # Generate noisy versions of both predicted and gold videos
        pred_latents_noisy = vgm_model.add_noise(video_batch['predicted'], timestep)
        gold_latents_noisy = vgm_model.add_noise(gold_videos, timestep)

        # Compute rewards
        pred_rewards, pred_conf = reward_model(pred_latents_noisy, timestep)
        gold_rewards, gold_conf = reward_model(gold_latents_noisy, timestep)

        # Supervision: gold videos should have higher rewards
        # Use margin loss: gold > pred + margin
        margin = 0.5
        loss = torch.nn.functional.relu(pred_rewards - gold_rewards + margin).mean()

        # Regularize confidence (encourage confident predictions on high-quality videos)
        quality_indicator = (gold_rewards - pred_rewards).detach()
        conf_loss = -torch.log(gold_conf + 1e-6) * (quality_indicator > 0).float()
        loss = loss + 0.1 * conf_loss.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {'avg_loss': np.mean(losses), 'timestep_sampled': timestep}
```

**6. Use Reward Model for Video Generation Guidance**

Apply the learned reward model to guide generation through test-time optimization.

```python
def guided_video_generation(vgm_model, reward_model, text_prompt, num_denoise_steps=50):
    """
    Generate video with process rewards guiding each denoising step.
    Uses rewards to adjust generation trajectory toward high-quality outputs.
    """
    # Initialize noise
    video_latents = torch.randn(1, 16, 4, 32, 32)  # Example dimensions

    for denoising_step in range(num_denoise_steps):
        # Compute current timestep
        timestep = int((1 - denoising_step / num_denoise_steps) * 999)

        # Original denoising step
        video_latents = vgm_model.denoise_step(
            video_latents, timestep, text_prompt
        )

        # Compute process reward
        reward, confidence = reward_model(video_latents, timestep)

        # Use reward to adjust denoising (gradient-based adjustment)
        if confidence > 0.7:
            # High-confidence reward signal; apply gradient-based reward guidance
            video_latents.requires_grad = True
            loss = -reward.mean()  # Maximize reward
            loss.backward()

            # Small gradient step
            video_latents = (video_latents + 0.01 * video_latents.grad).detach()
            video_latents.requires_grad = False

    return video_latents
```

## Practical Guidance

**When to Use Latent Reward Models:**
- Training video generation or world models with continuous reward signals
- Guiding generation toward specific quality criteria
- Avoiding expensive VLM-based evaluation during training
- Providing process supervision across the generation trajectory

**When NOT to Use:**
- Need semantic understanding beyond motion/quality (use VLM rewards instead)
- Working with video generation models you cannot access intermediate features from
- Tasks where full RGB-space metrics are more appropriate

**Key Hyperparameters:**
- `num_queries`: Compression level (4-16; higher = more information retained but slower)
- `feature_dim`: VGM feature dimensionality (typically 512-768)
- `timestep_sampling_strategy`: Uniform random vs. weighted by importance (uniform simpler)
- `margin_loss_margin`: Separation between gold and predicted rewards (0.3-1.0 typical)
- `reward_guidance_scale`: Strength of gradient-based guidance (0.01-0.1)

**Performance Tips:**
- Pre-train compressor on unlabeled video pairs (reduces data requirements)
- Use multiple VGM timesteps per batch for faster training
- Cache extracted features to avoid redundant VGM forward passes
- Combine latent rewards with lightweight auxiliary losses (e.g., perceptual metrics)

**Integration Pattern:**
Latent reward models integrate naturally into video diffusion training pipelines. Use as additional supervision alongside standard diffusion loss—minimize diffusion loss + λ × (−latent_reward).

## Reference

Research paper: https://arxiv.org/abs/2511.21541
