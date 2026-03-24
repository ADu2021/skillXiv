---
name: diffusion-multimodal-feature-extraction
title: "Towards Multimodal Understanding via Stable Diffusion as a Task-Aware Feature Extractor"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07106"
keywords: [Diffusion Models, Vision Features, Multimodal Learning, CLIP Alternative, Feature Fusion]
description: "Extract fine-grained visual features from pretrained text-to-image diffusion models to improve multimodal LLMs beyond CLIP, achieving up to 6% gains through strategic fusion of unconditional and conditional diffusion representations."
---

# Diffusion-Enhanced Multimodal Vision: Beyond CLIP Features

CLIP excels at high-level semantic understanding but struggles with fine-grained visual details: small objects, precise spatial relationships, and subtle attributes. Diffusion models, trained to generate pixels from text, learn remarkably detailed spatial and semantic representations. The insight is that intermediate features from stable diffusion models can serve as superior visual encoders for multimodal LLMs when properly extracted and fused with CLIP.

This approach identifies when text conditioning in diffusion cross-attention focuses features on query-relevant regions, and proposes defensive strategies against "leakage" where LLMs recover input captions from features. The result is a practical fusion strategy that boosts vision-centric benchmarks by 5-6%.

## Core Concept

Diffusion models encode rich structural information at multiple levels. Unconditional diffusion features capture fine-grained semantics that CLIP misses. Conditional diffusion features (guided by text prompts) show exceptional vision-language alignment through cross-attention maps. Rather than replacing CLIP, we strategically combine both: CLIP for robust semantic understanding and diffusion features for spatial detail and query relevance. A lightweight projection head learns to fuse these diverse representations.

## Architecture Overview

- **CLIP Encoder**: Frozen, provides semantic understanding (high-level features)
- **Stable Diffusion Backbone**: Frozen v2.1-base, extracts intermediate layer features
- **Feature Extraction Points**: Down-stages and up-stages at configurable timesteps (t=10-500)
- **Cross-Attention Maps**: Text-conditioned spatial alignment through diffusion blocks
- **Projection Head**: Lightweight 2-layer MLP learns to combine CLIP + diffusion features
- **Leakage Mitigation**: Dropout and feature masking prevent LLM from recovering input text

## Implementation

### Step 1: Extract Diffusion Features from Text-Conditioned Denoising

Extract intermediate features from Stable Diffusion at various timesteps as you denoise with text conditioning. This captures query-relevant spatial information:

```python
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from einops import rearrange

def extract_diffusion_features(image, text_prompt, model_name="stabilityai/stable-diffusion-2-1-base"):
    """
    Extract intermediate diffusion features conditioned on text prompt.
    Features are extracted at multiple timesteps and spatial resolutions.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae

    # Encode text prompt to guide denoising
    text_tokens = pipeline.tokenizer(
        text_prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    ).input_ids.to(model.device)
    text_embeddings = text_encoder(text_tokens)[0]

    # Encode image to latent space
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215

    # Denoise with text conditioning, extracting features at each timestep
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(50)

    all_features = {}

    for t in scheduler.timesteps:
        # Add noise appropriate for this timestep
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, t)

        # Forward pass through UNet with text conditioning
        with torch.no_grad():
            output = unet(
                noisy_latents, t,
                encoder_hidden_states=text_embeddings,
                return_dict=True
            )

        # Collect intermediate features from various UNet blocks
        # Extract from down blocks, middle block, and up blocks
        features = {
            "down": [],
            "mid": output.sample,
            "up": []
        }

        all_features[int(t.item())] = features

    return all_features

# Process image with query as text prompt
image = torch.randn(1, 3, 512, 512)  # Placeholder
query = "small blue object in corner"
features = extract_diffusion_features(image, query)
```

### Step 2: Analyze Cross-Attention Maps for Alignment

Examine cross-attention maps to understand how text tokens align with spatial regions. This reveals which parts of the image the model attends to for each word:

```python
def extract_cross_attention_maps(image, text_prompt, model_name="stabilityai/stable-diffusion-2-1-base"):
    """
    Extract cross-attention maps showing alignment between text tokens and image regions.
    Uses these maps to score vision-language correspondence.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae

    # Tokenize prompt and get text embeddings
    text_tokens = pipeline.tokenizer(
        text_prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    )
    text_input_ids = text_tokens.input_ids.to(unet.device)
    text_embeddings = text_encoder(text_input_ids)[0]

    # Store cross-attention maps during denoising
    cross_attention_maps = {}

    def hook_fn(name):
        def _hook(module, input, output):
            # Save attention weights for this module
            if hasattr(module, 'to_out'):
                attn_output = output[0]
                # Reshape to map form
                h, w = 64, 64  # Spatial dimensions
                cross_attention_maps[name] = attn_output.reshape(1, h, w, -1)
        return _hook

    # Register hooks on cross-attention layers
    hooks = []
    for name, module in unet.named_modules():
        if 'attn' in name and 'cross' in name:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Run one denoising step to capture attention
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        noisy_latents = noise  # Start from pure noise

        unet(noisy_latents, 999, encoder_hidden_states=text_embeddings)

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Compute alignment scores using LogSumExp pooling
    def compute_alignment_score(attn_maps, token_idx):
        """Pool attention map for a token to single alignment score."""
        pooled = torch.logsumexp(attn_maps[..., token_idx], dim=(1, 2))
        return pooled

    return cross_attention_maps
```

### Step 3: Fuse CLIP and Diffusion Features

Combine CLIP features (semantic) with diffusion features (fine-grained spatial) through a learned projection:

```python
from transformers import CLIPVisionModel
import torch.nn as nn

class DiffusionCLIPFusion(nn.Module):
    def __init__(self, clip_dim=768, diffusion_dim=768, output_dim=768):
        super().__init__()

        # Frozen CLIP vision encoder
        self.clip_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_encoder.eval()
        for param in self.clip_encoder.parameters():
            param.requires_grad = False

        # Diffusion feature projection (trainable)
        self.diffusion_projector = nn.Sequential(
            nn.Linear(diffusion_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
            nn.Dropout(0.1)  # Dropout mitigates text leakage
        )

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(clip_dim + output_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, image, diffusion_features):
        """
        Fuse CLIP and diffusion features with learned projections.
        """
        # Get CLIP features (frozen)
        with torch.no_grad():
            clip_outputs = self.clip_encoder(image)
            clip_features = clip_outputs.last_hidden_state  # [B, 257, 768]
            clip_pooled = clip_features[:, 0, :]  # Use [CLS] token

        # Project diffusion features
        diffusion_projected = self.diffusion_projector(diffusion_features)

        # Concatenate and fuse
        fused = torch.cat([clip_pooled, diffusion_projected], dim=-1)
        output = self.fusion_head(fused)

        return output
```

### Step 4: Integrate with Multimodal LLM

Connect the fused features to an LLaVA-style multimodal LLM for VQA/captioning:

```python
from transformers import AutoModel, AutoTokenizer

class DiffusionLLaVAAdapter(nn.Module):
    def __init__(self, fusion_model, llm_model_name="meta-llama/Llama-2-7b"):
        super().__init__()
        self.fusion_model = fusion_model
        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.projection_head = nn.Linear(768, self.llm.config.hidden_size)

    def forward(self, image, question, text_prompt=None):
        """
        End-to-end: extract diffusion features, fuse with CLIP, feed to LLM.
        """
        # Extract diffusion features for text prompt
        if text_prompt is None:
            text_prompt = question

        diffusion_feats = extract_diffusion_features(
            image, text_prompt
        )
        # Aggregate across timesteps
        diffusion_feats_agg = torch.stack(
            list(diffusion_feats.values())
        ).mean(dim=0)

        # Fuse with CLIP
        fused_features = self.fusion_model(image, diffusion_feats_agg)
        projected = self.projection_head(fused_features)

        # Tokenize question and get LLM embeddings
        tokens = tokenizer(
            question,
            return_tensors="pt"
        )

        # Inject visual features into LLM
        input_embeds = self.llm.get_input_embeddings()(tokens.input_ids)
        # Prepend fused visual features
        visual_prefix = projected.unsqueeze(1)
        combined_embeds = torch.cat(
            [visual_prefix, input_embeds],
            dim=1
        )

        # Generate answer
        outputs = self.llm(inputs_embeds=combined_embeds)
        return outputs
```

## Practical Guidance

| Component | Recommended Setting | Notes |
|---|---|---|
| Diffusion Model | Stable Diffusion v2.1-base | Proven feature quality |
| Feature Resolution | 16×16 tokens | After resizing regardless of extraction layer |
| Timesteps to Extract | t=10, 50, 100, 250, 500 | Cover noise range from high to low |
| CLIP Model | ViT-L/14 | Strong semantic baseline |
| Fusion Head Dropout | 0.1 | Prevents text leakage from LLM recovery |
| Projection Head | 2 layers, 512 hidden | Lightweight but expressive |
| Training Stage 1 | 4-6 hours on 4 H100 GPUs | Pretraining on LAION-like data |
| Training Stage 2 | 10 hours on 4 H100 GPUs | Fine-tuning on task-specific data |
| Batch Size Stage 1 | 256 | Large batch for stable contrastive learning |
| Batch Size Stage 2 | 128 | Smaller after initial convergence |

**When to use diffusion features:**
- Vision-centric benchmarks (MMVP, BLINK) emphasizing fine details
- Tasks requiring precise object localization or attribute identification
- Multimodal LLM applications where CLIP features underperform
- Scenarios where spatial detail matters more than semantic correctness

**When NOT to use diffusion features:**
- Real-time applications (diffusion feature extraction is slower than CLIP)
- Tasks requiring only semantic understanding (standard CLIP sufficient)
- Memory-constrained environments (requires two encoders)
- Text-heavy understanding where visual details are secondary

**Common pitfalls:**
- Not including dropout in projection head, allowing LLM to recover input captions
- Extracting features at wrong timesteps (too early = too noisy, too late = overfitted)
- Forgetting to freeze CLIP encoder, causing optimization instability
- Using too-deep fusion networks, overfitting to training data
- Not normalizing diffusion features before fusion
- Batch size too small during Stage 1, causing unstable contrastive learning

## Reference

Liu, Z., Wang, C., Cai, H., & Sun, Y. (2025). Towards Multimodal Understanding via Stable Diffusion as a Task-Aware Feature Extractor. arXiv:2507.07106. https://arxiv.org/abs/2507.07106
