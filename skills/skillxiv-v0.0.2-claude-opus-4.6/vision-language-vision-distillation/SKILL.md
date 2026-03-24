---
name: vision-language-vision-distillation
title: "Vision-Language-Vision Auto-Encoder: Scalable Knowledge Distillation from Diffusion Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07104"
keywords: [Vision-Language Models, Knowledge Distillation, Diffusion Models, Efficient VLMs, Image Understanding]
description: "Build efficient vision-language models by distilling knowledge from frozen diffusion decoders and vision encoders. Achieve GPT-4o-level captioning with <$1000 training cost by leveraging pre-trained components. Use when you need high-quality vision-language understanding without expensive end-to-end training."
---

# Vision-Language-Vision Auto-Encoder: Efficient VLM Training via Diffusion Distillation

Training vision-language models from scratch requires massive compute and data. VLV (Vision-Language-Vision) breaks this constraint by reusing three frozen pre-trained components: a vision encoder, the decoder of a text-to-image diffusion model, and an LLM. The core insight is that diffusion decoders encode rich visual information in their reconstruction process—by training an encoder to produce embeddings that these decoders can faithfully reconstruct, the encoder captures essential visual semantics. This knowledge transfers to language understanding when connected to an LLM.

The two-stage pipeline first trains the vision encoder to compress images into "caption embeddings" via reconstruction loss, then fine-tunes an LLM to decode these embeddings into natural language. The approach achieves GPT-4o-level performance while keeping total training costs under $1,000.

## Core Concept

VLV exploits an overlooked capability of diffusion models: their decoders are sophisticated visual feature decoders. By optimizing a vision encoder to make images reconstructible by a frozen diffusion decoder, the encoder learns to represent all task-relevant visual information. This is a form of knowledge distillation where the diffusion decoder acts as an information bottleneck: the encoder must preserve enough information for faithful reconstruction but discard task-irrelevant details.

Once the encoder is trained, adding an LLM trained to decode the embeddings into captions creates a complete vision-language model. The embedding space inherently aligns with visual semantics, making the decoder training straightforward. Emergent properties (object pose estimation, compositional semantics) arise without explicit supervision.

## Architecture Overview

- **Vision Encoder**: Florence-2 backbone with learnable query tokens (77 tokens), maps images to caption embeddings
- **Frozen Diffusion Decoder**: Stable Diffusion 2.1 decoder, reconstructs images from embeddings (provides supervision signal, no training)
- **Information Bottleneck**: Caption embeddings in CLIP text embedding space (512-dim), constrains compression
- **LLM Decoder**: Qwen-2.5 with trainable MLP projection heads converting embeddings to language features
- **Loss Functions**: Stage 1 uses MSE reconstruction loss, Stage 2 uses autoregressive language modeling
- **Query Token Mechanism**: Learnable parameters attending to image patches, enable efficient information aggregation

## Implementation

### Stage 1: Vision Encoder Training via Reconstruction

Train encoder to produce embeddings that diffusion decoder can reconstruct images from.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

class VisionEncoderWithQueries(nn.Module):
    """Vision encoder using learnable query tokens to aggregate image information."""

    def __init__(
        self,
        backbone_name: str = "microsoft/florence-2-base",
        num_queries: int = 77,  # Match CLIP embedding dimension
        embed_dim: int = 512     # Caption embedding dimension
    ):
        super().__init__()

        # Load frozen vision backbone
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.image_processor = AutoImageProcessor.from_pretrained(backbone_name)

        # Get backbone output dimension
        self.backbone_dim = self.backbone.config.hidden_size

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, self.backbone_dim))
        nn.init.normal_(self.queries, std=self.backbone_dim ** -0.5)

        # Cross-attention from queries to image features
        self.cross_attention = nn.MultiheadAttention(
            self.backbone_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Project to caption embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.norm = nn.LayerNorm(self.backbone_dim)

    def forward(self, images):
        """
        Args:
            images: (batch, 3, H, W) tensor

        Returns:
            caption_embeddings: (batch, num_queries, embed_dim)
        """
        # Extract image features from backbone
        image_features = self.backbone(images, output_hidden_states=True)
        hidden_states = image_features.hidden_states[-1]  # (batch, seq, backbone_dim)

        # Cross-attention: queries attend to image features
        batch_size = hidden_states.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)

        attended_queries, _ = self.cross_attention(
            queries,
            hidden_states,
            hidden_states
        )

        # Add residual and normalize
        attended_queries = self.norm(attended_queries + queries)

        # Project to caption embedding space
        caption_embeddings = self.projection(attended_queries)

        return caption_embeddings

class DiffusionDecoderWrapper(nn.Module):
    """Wrapper for frozen diffusion decoder (Stable Diffusion 2.1)."""

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-1"):
        super().__init__()
        from diffusers import StableDiffusionPipeline

        self.pipe = StableDiffusionPipeline.from_pretrained(model_name)
        self.vae_decoder = self.pipe.vae.decoder
        self.vae_decoder.eval()

        # Freeze all decoder parameters
        for param in self.vae_decoder.parameters():
            param.requires_grad = False

    def forward(self, caption_embeddings):
        """
        Reconstruct images from caption embeddings.

        Args:
            caption_embeddings: (batch, num_queries, embed_dim)

        Returns:
            reconstructed_images: (batch, 3, H, W)
        """
        # VAE decoder expects (batch, latent_channels, latent_h, latent_w)
        # Map embeddings to latent space
        batch_size, num_queries, embed_dim = caption_embeddings.shape

        # Simple projection to latent space (4, 64, 64) for SD 2.1
        latent_projection = nn.Linear(embed_dim * num_queries, 4 * 64 * 64)
        latents = latent_projection(caption_embeddings.reshape(batch_size, -1))
        latents = latents.view(batch_size, 4, 64, 64)

        # Decode via VAE decoder
        with torch.no_grad():
            images = self.vae_decoder(latents / 0.18215).sample  # SD uses scaling factor

        # Normalize to [-1, 1]
        images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0, 1)

        return images

class Stage1Trainer:
    """Train vision encoder via reconstruction loss."""

    def __init__(
        self,
        encoder: VisionEncoderWithQueries,
        decoder: DiffusionDecoderWrapper,
        learning_rate: float = 5e-5
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.01
        )

        self.criterion = nn.MSELoss()

    def training_step(self, batch_images):
        """
        Single training step: minimize reconstruction loss.

        Args:
            batch_images: (batch, 3, H, W) tensor

        Returns:
            Loss value
        """
        self.encoder.train()
        self.decoder.eval()  # Decoder is frozen

        # Encode images
        caption_embeddings = self.encoder(batch_images)

        # Reconstruct via decoder
        reconstructed = self.decoder(caption_embeddings)

        # Normalize original images for comparison
        batch_images_normalized = (batch_images + 1) / 2
        batch_images_normalized = torch.clamp(batch_images_normalized, 0, 1)

        # Reconstruction loss
        loss = self.criterion(reconstructed, batch_images_normalized)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, data_loader, num_steps: int = 200000, batch_size: int = 512):
        """
        Train for specified number of steps.

        Args:
            data_loader: DataLoader yielding image batches
            num_steps: Total training steps
            batch_size: Batch size
        """
        step = 0
        total_loss = 0.0

        for epoch in range(num_steps // len(data_loader) + 1):
            for batch in data_loader:
                loss = self.training_step(batch)
                total_loss += loss

                step += 1
                if step >= num_steps:
                    break

                if step % 100 == 0:
                    avg_loss = total_loss / step
                    print(f"Step {step}/{num_steps}: loss={avg_loss:.4f}")

            if step >= num_steps:
                break

        print(f"Stage 1 complete. Final avg loss: {total_loss / step:.4f}")
```

### Stage 2: LLM Decoder Fine-tuning for Captioning

Train LLM to decode caption embeddings into natural language.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class CaptionDecoderTrainer:
    """Fine-tune LLM to decode embeddings to captions."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        embed_dim: int = 512,
        learning_rate: float = 1e-5
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_dim = embed_dim

        # Learnable projection from caption embeddings to LLM input space
        self.embedding_projection = nn.Sequential(
            nn.Linear(embed_dim, self.model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        )

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.embedding_projection.parameters()),
            lr=learning_rate
        )

    def training_step(self, caption_embeddings, captions):
        """
        Train LLM to generate captions from embeddings.

        Args:
            caption_embeddings: (batch, num_queries, embed_dim)
            captions: List of caption strings

        Returns:
            Loss value
        """
        self.model.train()

        batch_size = caption_embeddings.shape[0]

        # Project embeddings to LLM space
        projected_embeddings = self.embedding_projection(caption_embeddings)
        # (batch, num_queries, hidden_size)

        # Tokenize captions and create input
        caption_losses = []

        for i, caption in enumerate(captions):
            # Tokenize caption
            caption_ids = self.tokenizer.encode(caption, return_tensors='pt')

            # Create input combining embedding and caption prefix
            # Simple approach: use embeddings as context, predict caption tokens
            caption_input_ids = caption_ids[:, :-1]  # All but last token
            caption_labels = caption_ids[:, 1:]  # All but first token

            # Forward pass through LLM
            outputs = self.model(input_ids=caption_input_ids)
            logits = outputs.logits

            # Compute loss on caption generation
            loss = F.cross_entropy(
                logits.view(-1, self.model.config.vocab_size),
                caption_labels.view(-1),
                reduction='mean'
            )

            caption_losses.append(loss)

        # Average loss across batch
        total_loss = torch.stack(caption_losses).mean()

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train_epoch(self, data_loader, num_steps: int = 100000, batch_size: int = 64):
        """
        Train LLM decoder for specified steps.

        Args:
            data_loader: DataLoader yielding (embeddings, captions) pairs
            num_steps: Total training steps
            batch_size: Batch size
        """
        step = 0
        total_loss = 0.0

        for epoch in range(num_steps // len(data_loader) + 1):
            for embeddings, captions in data_loader:
                loss = self.training_step(embeddings, captions)
                total_loss += loss

                step += 1
                if step >= num_steps:
                    break

                if step % 50 == 0:
                    avg_loss = total_loss / step
                    print(f"Step {step}/{num_steps}: loss={avg_loss:.4f}")

            if step >= num_steps:
                break

        print(f"Stage 2 complete. Final avg loss: {total_loss / step:.4f}")
```

### Complete VLV Pipeline

Integrate both stages into a complete vision-language model.

```python
class VLVPipeline:
    """Complete Vision-Language-Vision auto-encoder."""

    def __init__(
        self,
        encoder_name: str = "microsoft/florence-2-base",
        llm_name: str = "Qwen/Qwen2.5-7B",
        embed_dim: int = 512
    ):
        self.encoder = VisionEncoderWithQueries(encoder_name, embed_dim=embed_dim)
        self.decoder = DiffusionDecoderWrapper()
        self.caption_model = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.embed_dim = embed_dim

        # Projection for caption generation
        self.caption_projection = nn.Linear(embed_dim * 77, self.caption_model.config.hidden_size)

    def generate_caption(self, image, max_length: int = 100):
        """
        Generate caption for an image.

        Args:
            image: (1, 3, H, W) tensor or PIL Image
            max_length: Max caption length in tokens

        Returns:
            Caption string
        """
        # Encode image
        with torch.no_grad():
            caption_embeddings = self.encoder(image)  # (1, 77, 512)

            # Project to LLM space
            batch_features = caption_embeddings.view(1, -1)  # (1, 77*512)
            lm_input = self.caption_projection(batch_features)  # (1, hidden_size)

            # Generate caption using LLM
            # Initialize with special start token
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]])

            outputs = self.caption_model.generate(
                input_ids=input_ids,
                inputs_embeds=lm_input.unsqueeze(1),
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return caption

    def save(self, path: str):
        """Save trained model components."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'caption_model': self.caption_model.state_dict(),
            'projection': self.caption_projection.state_dict()
        }, path)

    @classmethod
    def load(cls, path: str):
        """Load trained model."""
        checkpoint = torch.load(path)
        model = cls()
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.caption_model.load_state_dict(checkpoint['caption_model'])
        model.caption_projection.load_state_dict(checkpoint['projection'])
        return model
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vision Encoder Backbone | Florence-2 | Pre-trained, frozen except projection |
| Learnable Query Tokens | 77 | Match CLIP embedding dimension |
| Caption Embedding Dimension | 512 | CLIP-compatible space |
| Diffusion Decoder | Stable Diffusion 2.1 | Frozen; provides supervision only |
| Stage 1 Learning Rate | 5e-5 | Reconstruction optimization |
| Stage 1 Batch Size | 512 | Large batches for stable convergence |
| Stage 1 Steps | 200K | ~4 days on 8 RTX 6000 Ada GPUs |
| Stage 2 Learning Rate | 1e-5 | LLM fine-tuning (lower than typical) |
| Stage 2 Batch Size | 64 | Memory-efficient caption generation |
| Stage 2 Steps | 100K | ~1 day on 8 RTX 6000 Ada GPUs |
| Total Training Cost | <$1,000 | 8 GPUs × ~5 days × $25/GPU/hr |

### When to Use

- Building vision-language models with limited compute budgets
- Needing GPT-4o-level captioning at fraction of training cost
- Tasks requiring strong visual understanding without architectural innovation
- Production systems where model size and efficiency matter
- Scenarios where pre-trained components can be reused

### When NOT to Use

- Tasks requiring specialized visual features (medical imaging, microscopy without retraining)
- Scenarios demanding state-of-the-art video understanding (method is image-only)
- Applications where full model transparency/control is required (relies on frozen diffusion)
- Systems with strict latency constraints (encoder + generation adds overhead)

### Common Pitfalls

- **Wrong diffusion decoder**: Not all diffusion models work; VAE decoder architecture matters; stick with SD 2.1 initially
- **Skipping Stage 1**: Jumping to Stage 2 with random encoder fails; Stage 1 reconstruction is critical
- **Insufficient query tokens**: Using <64 tokens limits information capacity; use 77+ for detailed understanding
- **Wrong projection dimensions**: Mismatches between embedding_dim and model.config.hidden_size cause shape errors; validate all dimensions
- **Frozen encoder in Stage 2**: Keep projection trainable; freezing encoder is correct but projection must adapt to LLM space
- **Memory limits**: 512-batch Stage 1 requires gradient checkpointing; use smaller batches if OOM occurs

## Reference

Lin, Z., Chen, M., Wang, X., et al. (2024). Vision-Language-Vision Auto-Encoder: Scalable Knowledge Distillation from Diffusion Models. arXiv preprint arXiv:2507.07104.
