---
name: omni-reward-modeling
title: "Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.23451"
keywords: [Reward Modeling, Multimodal, Preferences, RLHF, Foundation Model]
description: "Builds generalist reward models evaluating outputs across 5 modalities (text, image, video, audio, 3D) using free-form preference data. Combines discriminative and generative reward modeling approaches. Covers 9 tasks with 317K preference examples, enabling preference-based alignment for diverse output modalities."
---

# Omni-Reward: Multimodal Preference Modeling

Current reward models focus narrowly on text and images, missing alignment opportunities across diverse modalities. Omni-Reward builds a single generalist model that understands preferences for any output modality and free-form preference descriptions beyond binary choices.

The unified architecture combines discriminative and generative modeling to capture nuanced, personalized preferences.

## Core Concept

Key innovation: **single model learns preferences across modalities and preference formats**:
- Discriminative component: learns which output is better
- Generative component: generates preference descriptions
- Multimodal inputs: text, image, video, audio, 3D objects
- Free-form preferences: beyond binary pairs (e.g., "I prefer answers that are concise but detailed")

## Architecture Overview

- Shared multimodal encoder (vision + audio + text embeddings)
- Discriminative head: probability of output A > output B
- Generative head: free-form preference description prediction
- Task-specific adaptation layers for domain customization

## Implementation Steps

Build a multimodal encoder that can process any input modality. Use separate sub-encoders with shared projection layer:

```python
class MultimodalPreferenceEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_modalities=5):
        super().__init__()

        # Sub-encoders per modality
        self.text_encoder = TextEncoder(output_dim=hidden_dim)
        self.image_encoder = VisionTransformer(output_dim=hidden_dim)
        self.video_encoder = VideoTransformer(output_dim=hidden_dim)
        self.audio_encoder = AudioTransformer(output_dim=hidden_dim)
        self.object_3d_encoder = Point3DTransformer(output_dim=hidden_dim)

        # Fusion layer
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=12,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, **modalities):
        """Encode multimodal input."""
        embeddings = []

        if 'text' in modalities:
            embeddings.append(self.text_encoder(modalities['text']))
        if 'image' in modalities:
            embeddings.append(self.image_encoder(modalities['image']))
        if 'video' in modalities:
            embeddings.append(self.video_encoder(modalities['video']))
        if 'audio' in modalities:
            embeddings.append(self.audio_encoder(modalities['audio']))
        if 'object_3d' in modalities:
            embeddings.append(self.object_3d_encoder(modalities['object_3d']))

        # Fuse embeddings with attention
        stacked = torch.stack(embeddings, dim=1)
        fused, _ = self.fusion(stacked, stacked, stacked)
        output = self.norm(fused.mean(dim=1))

        return output
```

Implement discriminative and generative heads for different preference tasks:

```python
class OmniRewardModel(nn.Module):
    def __init__(self, encoder, hidden_dim=768, vocab_size=50000):
        super().__init__()

        self.encoder = encoder

        # Discriminative head: output A vs B preference
        self.discriminative_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # logit for preference
        )

        # Generative head: generate preference description
        self.generative_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)  # token logits
        )

    def score_outputs(self, output_a, output_b, modalities_a, modalities_b):
        """Compare two outputs using discriminative head."""
        # Encode both outputs
        emb_a = self.encoder(**modalities_a)
        emb_b = self.encoder(**modalities_b)

        # Concatenate for comparison
        combined = torch.cat([emb_a, emb_b], dim=-1)

        # Preference score (positive = A better)
        score = self.discriminative_head(combined)

        return score

    def generate_preference(self, output, modalities):
        """Generate natural language preference description."""
        emb = self.encoder(**modalities)

        # Generate preference tokens
        logits = self.generative_head(emb)

        return logits

    def forward(self, outputs_a, outputs_b, mod_a, mod_b):
        """Compute both discriminative and generative losses."""
        disc_score = self.score_outputs(outputs_a, outputs_b, mod_a, mod_b)
        gen_logits = self.generate_preference(outputs_a, mod_a)

        return disc_score, gen_logits
```

Train with a combination of preference ranking and generation losses:

```python
def train_omni_reward(model, preference_data, num_epochs=10):
    """Train on multimodal preference data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch in preference_data:
            # Unpack batch
            output_a = batch['output_a']
            output_b = batch['output_b']
            modalities_a = batch['modalities_a']
            modalities_b = batch['modalities_b']
            preference_label = batch['preference']  # 1 if A preferred
            preference_desc = batch['description']

            # Forward pass
            disc_score, gen_logits = model(
                output_a, output_b, modalities_a, modalities_b
            )

            # Discriminative loss: preference ranking
            disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_score.squeeze(), preference_label.float()
            )

            # Generative loss: preference description
            gen_loss = torch.nn.functional.cross_entropy(
                gen_logits, preference_desc
            )

            # Combined loss
            total_loss = 0.7 * disc_loss + 0.3 * gen_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

## Practical Guidance

| Component | Recommendation |
|-----------|-----------------|
| Hidden dimension | 768-1024 (balance capacity and efficiency) |
| Modalities to support | Start with 3 (text, image, video) |
| Preference data sources | 50K+ examples per modality combination |
| Discriminative/Generative weight | 0.7/0.3 (emphasize ranking) |

**When to use:**
- Multimodal RLHF for diverse output types
- Systems generating text, images, video, audio simultaneously
- Applications needing nuanced preference understanding
- Foundation models requiring broad preference alignment

**When NOT to use:**
- Single modality (specialized models better)
- When discriminative ranking alone suffices
- Real-time inference with strict latency (encoder overhead)

**Common pitfalls:**
- Imbalanced modality representation in training data
- Insufficient preference description examples (generation head underfits)
- Not normalizing modality embeddings (attention dominates one modality)
- Overweighting generation loss (discriminative signal diluted)

Reference: [Omni-Reward on arXiv](https://arxiv.org/abs/2510.23451)
