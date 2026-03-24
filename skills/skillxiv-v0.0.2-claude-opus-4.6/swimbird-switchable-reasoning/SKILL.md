---
name: swimbird-switchable-reasoning
title: "SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06040"
keywords: [Multimodal Learning, Adaptive Reasoning, Vision-Language, Flexible Computation, Dynamic Allocation]
description: "Enable multimodal models to dynamically switch between text and vision reasoning modes, allocating computation based on perceived difficulty and image resolution, achieving strong performance on both vision-dense and text-heavy benchmarks."
---

# SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs

## Problem Context

Fixed reasoning patterns in multimodal systems mismatch question types with computation modes. Text-heavy problems forced to use visual thinking degrade logic quality; vision-dense tasks forced into text-only reasoning lose spatial detail. Models need to adapt their reasoning modality per query rather than commit to a single approach.

## Core Concept

SwimBird combines [hybrid autoregressive modeling, dynamic latent budgets, multi-mode dataset curation] to enable query-adaptive mode selection. The model predicts reasoning difficulty and image resolution, then allocates variable continuous tokens (visual thoughts) dynamically, switching seamlessly between text-only and vision-rich paths.

## Architecture Overview

- **Hybrid autoregressive**: Next-token prediction for text, next-embedding prediction for visual tokens
- **Dynamic allocation**: Variable visual-thought budget based on difficulty scores and resolution
- **Mode switching**: Special delimiters enable flexible mode transitions
- **Dataset curation**: SwimBird-SFT-92K categorizes data by reasoning pattern using pass@8 scoring
- **Resolution awareness**: Adapt token allocation to image resolution

## Implementation

### Step 1: Implement hybrid autoregressive model

Create dual-mode generation supporting both discrete tokens and continuous embeddings.

```python
# Hybrid autoregressive architecture
class HybridAutoregressive(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, vision_dim=256, max_vision_tokens=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.vision_dim = vision_dim
        self.max_vision_tokens = max_vision_tokens

        # Token embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)

        # Vision token processor (continuous embeddings)
        self.vision_processor = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer backbone
        from torch.nn import TransformerDecoderLayer, TransformerDecoder
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=6)

        # Output heads
        self.token_head = torch.nn.Linear(hidden_dim, vocab_size)
        self.vision_embedding_head = torch.nn.Linear(hidden_dim, vision_dim)

        # Mode detection head
        self.mode_logits = torch.nn.Linear(hidden_dim, 2)  # text/vision mode

    def forward(self, input_ids, vision_embeddings=None, mode_mask=None):
        """
        Args:
            input_ids: Text token IDs, shape (batch, seq_len)
            vision_embeddings: Continuous vision embeddings, shape (batch, vision_len, vision_dim)
            mode_mask: Which positions are text vs vision, shape (batch, seq_len + vision_len)
        """
        batch_size = input_ids.shape[0]

        # Embed tokens
        token_embs = self.token_embedding(input_ids)

        # Process vision embeddings if provided
        if vision_embeddings is not None:
            vision_embs = self.vision_processor(vision_embeddings)
            # Interleave or concatenate based on mode_mask
            combined_embs = self._combine_modalities(
                token_embs, vision_embs, mode_mask
            )
        else:
            combined_embs = token_embs

        # Transformer decoding
        hidden_states = self.decoder(combined_embs, combined_embs)

        # Output heads
        token_logits = self.token_head(hidden_states)
        vision_embeddings_pred = self.vision_embedding_head(hidden_states)
        mode_logits = self.mode_logits(hidden_states)

        return token_logits, vision_embeddings_pred, mode_logits

    def _combine_modalities(self, token_embs, vision_embs, mode_mask):
        """Interleave text and vision embeddings based on mode_mask."""
        combined = []
        token_idx = 0
        vision_idx = 0

        for i, mode in enumerate(mode_mask[0]):
            if mode == 0:  # Text mode
                combined.append(token_embs[:, token_idx])
                token_idx += 1
            else:  # Vision mode
                combined.append(vision_embs[:, vision_idx])
                vision_idx += 1

        return torch.stack(combined, dim=1)
```

### Step 2: Implement difficulty detection for dynamic budgeting

Predict reasoning difficulty and image complexity to determine visual-thought allocation.

```python
# Difficulty and complexity detection
class DifficultyDetector:
    def __init__(self, model):
        self.model = model

    def detect_difficulty(self, question, image=None):
        """
        Estimate reasoning difficulty from question text and image.
        Returns: difficulty_score in [0, 1]
        """
        # Embed question
        question_emb = self.embed_text(question)

        if image is not None:
            # Embed image
            image_emb = self.embed_image(image)
            # Combine
            combined = torch.cat([question_emb, image_emb], dim=-1)
        else:
            combined = question_emb

        # Difficulty prediction
        difficulty_logits = self.model.difficulty_head(combined)
        difficulty_score = torch.sigmoid(difficulty_logits).item()

        return difficulty_score

    def detect_image_resolution(self, image):
        """Estimate image complexity/resolution."""
        if image is None:
            return 1.0  # Default to low resolution

        # Proxy: entropy of image features
        image_emb = self.embed_image(image)
        entropy = -torch.sum(image_emb * torch.log(image_emb + 1e-8))
        resolution_score = torch.sigmoid(entropy).item()

        return resolution_score

    def allocate_vision_budget(self, difficulty, resolution, max_tokens=128):
        """
        Dynamically allocate vision token budget.
        """
        # Base budget scaled by difficulty and resolution
        base_budget = max_tokens * 0.3  # Start at 30% of max
        difficulty_multiplier = difficulty  # 0-1
        resolution_multiplier = resolution  # 0-1

        allocated_budget = int(
            base_budget * (1.0 + difficulty_multiplier + resolution_multiplier)
        )
        allocated_budget = min(allocated_budget, max_tokens)

        return allocated_budget

    def embed_text(self, text):
        """Embed question text."""
        # Use model's text encoder
        tokens = self.tokenize(text)
        embs = self.model.token_embedding(tokens)
        return embs.mean(dim=1)  # Average pooling

    def embed_image(self, image):
        """Embed image."""
        # Use vision encoder (e.g., CLIP)
        from torchvision.models import resnet50
        encoder = resnet50(pretrained=True)
        features = encoder(image.unsqueeze(0))
        return features.mean(dim=(2, 3)).squeeze()  # Global average pooling

    def tokenize(self, text):
        """Tokenize text."""
        # Simplified tokenization
        tokens = text.split()
        return torch.tensor([hash(t) % self.model.vocab_size for t in tokens])
```

### Step 3: Create multi-mode dataset with pass@8 curation

Curate training data by categorizing examples into reasoning modes.

```python
# Multi-mode dataset curation
class MultiModeDatasetCurator:
    def __init__(self, model, verifier):
        self.model = model
        self.verifier = verifier

    def categorize_example(self, question, image, answer, num_samples=8):
        """
        Generate multiple responses and categorize by success pattern.
        Text-only, Vision-only, or Interleaved.
        """
        text_only_successes = 0
        vision_only_successes = 0
        interleaved_successes = 0

        for _ in range(num_samples):
            # Text-only reasoning
            text_response = self.model.generate_text_only(question)
            text_only_successes += self.verifier(text_response, answer)

            # Vision-only (embed image heavily)
            if image is not None:
                vision_response = self.model.generate_vision_heavy(
                    question, image
                )
                vision_only_successes += self.verifier(vision_response, answer)

                # Interleaved reasoning
                interleaved_response = self.model.generate_interleaved(
                    question, image
                )
                interleaved_successes += self.verifier(
                    interleaved_response, answer
                )

        # Determine best mode (highest success rate)
        success_rates = {
            'text_only': text_only_successes / num_samples,
            'vision_only': vision_only_successes / num_samples if image else 0,
            'interleaved': interleaved_successes / num_samples if image else 0
        }

        best_mode = max(success_rates, key=success_rates.get)
        return best_mode, success_rates

    def curate_dataset(self, raw_examples, output_path):
        """
        Curate entire dataset, categorizing each example.
        """
        text_only = []
        vision_only = []
        interleaved = []

        for idx, example in enumerate(raw_examples):
            question = example['question']
            image = example.get('image')
            answer = example['answer']

            best_mode, scores = self.categorize_example(
                question, image, answer, num_samples=8
            )

            example_with_mode = {**example, 'best_mode': best_mode}

            if best_mode == 'text_only':
                text_only.append(example_with_mode)
            elif best_mode == 'vision_only':
                vision_only.append(example_with_mode)
            else:  # interleaved
                interleaved.append(example_with_mode)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} examples")

        # Save curated datasets
        import json
        with open(f"{output_path}/text_only.json", 'w') as f:
            json.dump(text_only, f)
        with open(f"{output_path}/vision_only.json", 'w') as f:
            json.dump(vision_only, f)
        with open(f"{output_path}/interleaved.json", 'w') as f:
            json.dump(interleaved, f)

        print(f"Text-only: {len(text_only)}, Vision-only: {len(vision_only)}, "
              f"Interleaved: {len(interleaved)}")
```

### Step 4: Train with mode-aware supervision

Implement training that respects the best reasoning mode for each example.

```python
# Mode-aware training
def train_swimbird(
    model, train_loader, optimizer, device='cuda',
    lambda_mode=0.1
):
    """
    Train SwimBird with mode-aware supervision.
    """
    model = model.to(device)
    criterion_token = torch.nn.CrossEntropyLoss()
    criterion_vision = torch.nn.MSELoss()

    for epoch in range(3):
        total_loss = 0.0

        for batch in train_loader:
            question = batch['question']
            image = batch.get('image')
            answer = batch['answer']
            best_mode = batch['best_mode']

            # Forward pass
            token_logits, vision_preds, mode_logits = model(
                input_ids=question,
                vision_embeddings=image,
                mode_mask=None  # Will be determined dynamically
            )

            # Token prediction loss
            answer_ids = model.tokenize(answer)
            token_loss = criterion_token(token_logits, answer_ids)

            # Mode supervision loss
            mode_labels = torch.tensor([
                0 if m == 'text_only' else 1
                for m in best_mode
            ]).to(device)
            mode_loss = torch.nn.functional.cross_entropy(
                mode_logits[:, 0], mode_labels
            )

            # Vision embedding loss (if applicable)
            if image is not None:
                vision_loss = criterion_vision(vision_preds, image)
            else:
                vision_loss = 0.0

            # Combined loss
            total = token_loss + lambda_mode * mode_loss + 0.5 * vision_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            total_loss += total.item()

        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_loader):.4f}")
```

### Step 5: Inference with dynamic mode selection

Run inference with adaptive mode switching.

```python
# Adaptive inference
def generate_with_adaptive_reasoning(
    model, question, image=None, max_tokens=200
):
    """
    Generate response with dynamically selected reasoning mode.
    """
    difficulty_detector = DifficultyDetector(model)

    # Detect difficulty and resolution
    difficulty = difficulty_detector.detect_difficulty(question, image)
    resolution = difficulty_detector.detect_image_resolution(image) if image else 0

    # Allocate vision budget
    vision_budget = difficulty_detector.allocate_vision_budget(
        difficulty, resolution
    )

    # Generate with allocated budget
    response = []
    current_mode = 'text'  # Start with text
    vision_tokens_used = 0

    for step in range(max_tokens):
        # Decide mode at each step
        if vision_tokens_used < vision_budget and image is not None:
            # Can use vision tokens
            mode_logits = model.mode_logits(model_hidden_state)
            mode_probs = torch.softmax(mode_logits, dim=-1)
            current_mode = 'vision' if mode_probs[1] > 0.5 else 'text'
        else:
            current_mode = 'text'

        # Generate next token/embedding based on mode
        if current_mode == 'text':
            next_token_logits = model.token_head(model_hidden_state)
            next_token = next_token_logits.argmax(dim=-1)
            response.append(model.decode_token(next_token))
        else:
            next_vision_emb = model.vision_embedding_head(model_hidden_state)
            response.append(f"[VISION_TOKEN_{vision_tokens_used}]")
            vision_tokens_used += 1

        # Early stopping
        if next_token == model.eos_token_id:
            break

    return ' '.join(response)
```

## Practical Guidance

**When to use**: Multimodal benchmarks with mixed question types (vision-heavy and text-heavy). Most beneficial for question-answering, document understanding, scientific reasoning.

**Hyperparameters**:
- **Max vision tokens**: 64-256 (balance visual detail vs. text length)
- **Difficulty threshold**: 0.4-0.6 (determines mode-switching point)
- **lambda_mode**: 0.05-0.2 (weight of mode supervision)
- **Resolution scaling**: 0.5-1.5x adjustment factor

**Key empirical findings**:
- State-of-the-art on V-Star (85.5%) and WeMath (49.5%)
- Balanced performance across vision-dense and text-heavy tasks
- Dynamic budget allocation reduces wasted computation
- Dataset curation via pass@8 critical for performance

**Common pitfalls**:
- Fixed token budgets → doesn't adapt to image complexity
- Always using both modalities → computational waste on simple tasks
- Not curating training data by reasoning mode → mode switching becomes unreliable
- Aggressive mode switching → incoherent reasoning flow

**Scaling**: Tested on Qwen3-VL 8B. Scaling to larger models uncertain; recommend testing on target size.

## Reference

Paper: https://arxiv.org/abs/2602.06040
Code: Available at author's repository
Dataset: SwimBird-SFT-92K (publicly available)
Benchmarks: V-Star, WeMath, Vision-Heavy and Text-Heavy QA
