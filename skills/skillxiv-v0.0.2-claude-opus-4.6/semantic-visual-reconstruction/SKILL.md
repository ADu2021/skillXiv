---
name: semantic-visual-reconstruction
title: "Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09040"
keywords: [vision-language, semantic reconstruction, visual tokenization, multimodal supervision]
description: "Add explicit visual supervision to VLMs by training models to autoregressively reconstruct semantic image tokens, achieving 2-3% average gains and 10-point improvements on hallucination robustness."
---

# Autoregressive Semantic Visual Reconstruction

## Core Concept

Traditional Vision-Language Models (VLMs) apply supervision only to text outputs while leaving rich visual input unsupervised. ASVR (Autoregressive Semantic Visual Reconstruction) addresses this asymmetry by training models to predict both semantic visual tokens and text tokens within a unified framework. This establishes a "perceptual foundation for image understanding" that improves robustness and reduces hallucinations.

## Architecture Overview

- **Visual tokenizer**: VQ-SigLIP converts images to discrete semantic tokens capturing high-level features
- **Joint training objective**: Unified loss on both visual token and text token prediction
- **Two-stage training**: Pre-training aligns visual representations, instruction tuning refines understanding
- **Semantic > Appearance**: High-level semantic information matters more than pixel-level reconstruction

## Implementation

### Step 1: Build Semantic Visual Tokenizer

Create tokenizer that captures high-level semantic features:

```python
class SemanticVisualTokenizer:
    def __init__(self, model_name: str = "vq-siglip"):
        self.model_name = model_name
        # Load pretrained semantic tokenizer
        self.tokenizer = self._load_semantic_tokenizer(model_name)
        self.vocab_size = self.tokenizer.config.vocab_size

    def _load_semantic_tokenizer(self, model_name: str):
        """Load pretrained semantic visual tokenizer."""
        # VQ-SigLIP: VQ-VAE variant using SigLIP embeddings
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            f"semantic-tokenizers/{model_name}",
            trust_remote_code=True
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image to discrete semantic tokens."""
        # Normalize image
        image = (image - image.mean()) / image.std()

        # Encode via semantic tokenizer
        with torch.no_grad():
            tokens = self.tokenizer.encode(image)

        return tokens  # Shape: (num_tokens,)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from semantic tokens."""
        with torch.no_grad():
            image = self.tokenizer.decode(tokens)

        return image
```

### Step 2: Design Joint Training Objective

Train model to predict both visual and text tokens:

```python
class SemanticVisualVLM(torch.nn.Module):
    def __init__(self, text_model, vision_encoder,
                 semantic_tokenizer: SemanticVisualTokenizer):
        super().__init__()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.semantic_tokenizer = semantic_tokenizer

        # Output heads for both modalities
        self.visual_head = torch.nn.Linear(
            self.text_model.hidden_size,
            semantic_tokenizer.vocab_size
        )
        self.text_head = torch.nn.Linear(
            self.text_model.hidden_size,
            self.text_model.vocab_size
        )

    def forward(self, image: torch.Tensor,
               text_tokens: torch.Tensor,
               visual_tokens: torch.Tensor) -> dict:
        """Forward pass with both visual and text supervision."""

        # Encode image
        image_features = self.vision_encoder(image)

        # Interleave visual and text tokens for joint training
        # [IMG_START] visual_tokens [IMG_END] text_tokens
        combined_input = self._interleave_modalities(
            image_features,
            text_tokens
        )

        # Get hidden states
        hidden_states = self.text_model(
            combined_input,
            output_hidden_states=True
        ).hidden_states

        # Predict both visual and text tokens
        visual_logits = self.visual_head(hidden_states)
        text_logits = self.text_head(hidden_states)

        return {
            "visual_logits": visual_logits,
            "text_logits": text_logits,
            "hidden_states": hidden_states
        }

    def _interleave_modalities(self, image_features, text_tokens):
        """Create combined input with visual and text modality markers."""
        # This is simplified; in practice, use proper sequence construction
        return torch.cat([image_features, text_tokens], dim=0)
```

### Step 3: Implement Two-Stage Training

Stage 1: Pre-training on large-scale image-text data:

```python
class PretrainingTrainer:
    def __init__(self, model: SemanticVisualVLM):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4
        )

    def compute_joint_loss(self, outputs: dict,
                          visual_tokens: torch.Tensor,
                          text_tokens: torch.Tensor) -> torch.Tensor:
        """Combined visual and text reconstruction loss."""

        # Visual token prediction loss
        visual_logits = outputs["visual_logits"]
        visual_loss = torch.nn.functional.cross_entropy(
            visual_logits.reshape(-1, self.model.semantic_tokenizer.vocab_size),
            visual_tokens.reshape(-1)
        )

        # Text token prediction loss
        text_logits = outputs["text_logits"]
        text_loss = torch.nn.functional.cross_entropy(
            text_logits.reshape(-1, self.model.text_model.vocab_size),
            text_tokens.reshape(-1)
        )

        # Equal weighting for balance
        total_loss = visual_loss + text_loss

        return total_loss

    def train_epoch(self, dataloader):
        """Train one epoch on large-scale data."""
        total_loss = 0.0

        for batch in dataloader:
            images = batch["image"]
            text_tokens = batch["text_tokens"]

            # Encode images to semantic tokens
            visual_tokens = self.model.semantic_tokenizer.encode_image(
                images
            )

            # Forward pass
            outputs = self.model(
                images,
                text_tokens,
                visual_tokens
            )

            # Compute loss
            loss = self.compute_joint_loss(
                outputs,
                visual_tokens,
                text_tokens
            )

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
```

Stage 2: Instruction tuning on diverse vision-language tasks:

```python
class InstructionTuner:
    def __init__(self, pretrained_model: SemanticVisualVLM):
        self.model = pretrained_model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-5
        )

    def finetune_on_tasks(self, task_datasets: dict,
                         num_epochs: int = 3):
        """Fine-tune on diverse instruction-following tasks."""

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_samples = 0

            for task_name, dataset in task_datasets.items():
                for batch in dataset:
                    images = batch["image"]
                    instructions = batch["instruction"]
                    target_responses = batch["response"]

                    # Tokenize
                    text_tokens = self.model.text_model.tokenize(
                        instructions
                    )
                    response_tokens = self.model.text_model.tokenize(
                        target_responses
                    )
                    visual_tokens = (
                        self.model.semantic_tokenizer.encode_image(images)
                    )

                    # Forward
                    outputs = self.model(
                        images,
                        text_tokens,
                        visual_tokens
                    )

                    # Loss focuses on response prediction
                    response_logits = outputs["text_logits"][
                        len(text_tokens):
                    ]
                    loss = torch.nn.functional.cross_entropy(
                        response_logits.reshape(
                            -1,
                            self.model.text_model.vocab_size
                        ),
                        response_tokens.reshape(-1)
                    )

                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    total_samples += 1

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

### Step 4: Evaluate on Benchmarks

Test improvements across multimodal understanding tasks:

```python
def evaluate_model(model: SemanticVisualVLM,
                  benchmark_name: str,
                  dataset) -> dict:
    """Evaluate on standard VLM benchmarks."""

    results = {
        "accuracy": 0.0,
        "hallucination_score": 0.0,
        "semantic_consistency": 0.0
    }

    correct = 0
    hallucination_count = 0
    semantic_scores = []

    for sample in dataset:
        image = sample["image"]
        question = sample["question"]
        gold_answer = sample["answer"]

        # Generate response
        response = model.generate_response(
            image,
            question,
            max_tokens=256
        )

        # Check correctness
        if matches_answer(response, gold_answer):
            correct += 1

        # Check for hallucinations
        if contains_hallucination(response, image):
            hallucination_count += 1

        # Semantic consistency score
        semantic_score = compute_semantic_consistency(
            image,
            response,
            model.semantic_tokenizer
        )
        semantic_scores.append(semantic_score)

    results["accuracy"] = correct / len(dataset)
    results["hallucination_score"] = 1.0 - (
        hallucination_count / len(dataset)
    )
    results["semantic_consistency"] = sum(semantic_scores) / len(
        semantic_scores
    )

    return results
```

## Practical Guidance

**Semantic Tokenization**: VQ-SigLIP outperforms appearance-based tokenizers because high-level semantic structure matters more than pixel fidelity for understanding. Use semantic tokenizers rather than pixel reconstruction.

**Joint Objective Balance**: Equal weighting between visual and text losses works well in practice. If one modality dominates, adjust weights based on downstream task importance.

**Training Data**: Large-scale image-text pairs enable strong pre-training. Instruction-tuning datasets should cover diverse vision-language tasks (VQA, captioning, scene understanding).

**Hallucination Reduction**: The semantic visual supervision significantly reduces hallucinations (10-point improvement on HallusionBench). This is the primary benefit over text-only supervision.

**When to Apply**: Use ASVR when reducing hallucinations or improving visual understanding is critical, or when training on multimodal data with dense supervision.

## Reference

ASVR achieves consistent 2-3% gains across 14 benchmarks by establishing explicit perceptual supervision alongside text objectives. The key insight is that semantic-level visual reconstruction (not pixel-level) provides the right inductive bias for understanding, leading to more robust and grounded vision-language models.
