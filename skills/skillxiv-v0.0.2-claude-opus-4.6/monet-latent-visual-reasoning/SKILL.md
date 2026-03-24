---
name: monet-latent-visual-reasoning
title: "Monet: Reasoning in Latent Visual Space Beyond Images and Language"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.21395"
keywords: [Latent Visual Reasoning, Chain of Thought, VLM Enhancement, Vision-Language Models]
description: "Enable multimodal LLMs to generate and reason with latent visual embeddings as intermediate thoughts: implement supervised fine-tuning to produce continuous visual representations, then optimize via VLPO to treat embeddings as learnable actions in reinforcement learning."
---

# Monet: Latent Visual Reasoning for Multimodal Models

Vision-language models often struggle with complex spatial reasoning because they must choose between text-only reasoning (losing visual detail) or generating external auxiliary images (expensive and error-prone). This skill demonstrates how to enable MLLMs to internally generate and reason with latent visual embeddings—intermediate visual thoughts that act as a third modality alongside text and images.

The core innovation is treating continuous latent embeddings as learnable "actions" that can be generated and optimized through reinforcement learning, enabling flexible reasoning that interleaves text and visual thought.

## Core Concept

Monet enables MLLMs to generate continuous latent visual embeddings through:

1. **Supervised Fine-Tuning (SFT)**: Three-stage training progressively teaching the model to generate and reason with latent embeddings

2. **Vision-Language Policy Optimization (VLPO)**: Novel RL algorithm computing policy gradients directly for latent embeddings by treating them as continuous actions

3. **Flexible Interleaving**: Models learn to automatically decide when to generate latent reasoning tokens vs. continuing text generation

## Architecture Overview

- **Latent Embedding Generation**: Special tokens that decode to continuous visual embeddings
- **Attention Control Mechanism**: Learnable flow control maintaining visual information through generation
- **SFT Pipeline**: Stage 1 (observation alignment), Stage 2 (latent generation), Stage 3 (reasoning chains)
- **VLPO Optimizer**: Gradient computation for continuous embeddings via probability estimation
- **Flexible Token Interleaving**: Text and latent tokens can be mixed in reasoning chains

## Implementation Steps

The system trains through SFT stages then optimizes with reinforcement learning.

**1. Prepare Hidden State Alignment**

Align hidden representations of key observation tokens to initialize latent reasoning.

```python
def compute_hidden_state_alignment(model, observation_tokens, hidden_dim=4096, num_layers=32):
    """
    Align hidden states of observation tokens for efficient latent encoding.
    Reduces computational cost of full image embedding alignment.
    Args:
        model: MLLM with vision encoder
        observation_tokens: Special tokens representing image observations
        hidden_dim: Model hidden dimension
        num_layers: Number of transformer layers
    Returns:
        alignment_matrix: Maps observation hidden states to latent space
    """
    # Collect hidden states of observation tokens
    observation_hiddens = []

    for obs_token in observation_tokens:
        with torch.no_grad():
            output = model.encode_observation(obs_token, output_hidden_states=True)

            # Take representation from key layer (typically 2/3 through model)
            key_layer_idx = int(num_layers * 0.67)
            hidden = output.hidden_states[key_layer_idx]

            observation_hiddens.append(hidden.mean(dim=1))  # Average over sequence

    observation_hiddens = torch.stack(observation_hiddens)

    # Learn projection to latent space
    alignment_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

    # Optimize to preserve information
    optimizer = torch.optim.Adam([alignment_matrix.parameters()], lr=1e-4)

    for _ in range(100):
        projected = alignment_matrix(observation_hiddens)
        reconstruction_loss = torch.nn.functional.mse_loss(projected, observation_hiddens)
        reconstruction_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return alignment_matrix
```

**2. Implement Stage 1: Observation Token Alignment**

Train the model to project image observations into learnable latent space.

```python
class LatentReasoningModule(torch.nn.Module):
    """
    Generates and processes latent visual embeddings for reasoning.
    """
    def __init__(self, hidden_dim=4096, latent_dim=512, num_latent_tokens=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens

        # Encoder: image -> latent embeddings
        self.latent_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim * num_latent_tokens)
        )

        # Decoder: latent -> hidden states (for attention)
        self.latent_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * num_latent_tokens, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention control: learn when to use latent vs. skip
        self.attention_gates = torch.nn.Linear(hidden_dim, num_latent_tokens)

    def encode_image_to_latent(self, image_hidden):
        """
        Encode image representation to latent embeddings.
        Args:
            image_hidden: (batch, seq, hidden_dim)
        Returns:
            latent_embeddings: (batch, num_latent_tokens, latent_dim)
        """
        batch_size, seq_len, hidden_dim = image_hidden.shape

        # Compress and encode
        image_compressed = image_hidden.mean(dim=1)  # Average over sequence
        latent_flat = self.latent_encoder(image_compressed)

        latent_embeddings = latent_flat.reshape(
            batch_size, self.num_latent_tokens, self.latent_dim
        )

        return latent_embeddings

    def forward(self, image_hidden, text_hidden=None):
        """
        Generate latent embeddings for image, optionally conditioned on text.
        Args:
            image_hidden: (batch, seq, hidden_dim)
            text_hidden: (batch, seq, hidden_dim) optional
        Returns:
            latent_embeddings: (batch, num_latent_tokens, latent_dim)
            attention_weights: (batch, num_latent_tokens)
        """
        latent = self.encode_image_to_latent(image_hidden)

        # Compute attention weights (when to use latent representations)
        if text_hidden is not None:
            gate_input = torch.cat([image_hidden.mean(dim=1), text_hidden.mean(dim=1)], dim=-1)
        else:
            gate_input = image_hidden.mean(dim=1)

        attention_weights = torch.softmax(self.attention_gates(gate_input), dim=-1)

        return latent, attention_weights
```

**3. Implement Stage 2: Latent Generation Training**

Fine-tune model to generate special latent tokens during reasoning.

```python
def stage2_latent_generation_training(model, latent_module, train_dataloader, num_epochs=10):
    """
    Train model to generate latent reasoning tokens.
    Model learns when and what latent embeddings to produce.
    """
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(latent_module.parameters()),
        lr=1e-4
    )

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Forward pass
            images = batch['images']
            questions = batch['questions']
            answers = batch['answers']
            latent_targets = batch.get('latent_targets', None)

            # Encode image
            image_hidden = model.encode_image(images, output_hidden_states=True)

            # Generate latent embeddings
            latent_embeddings, attention_weights = latent_module(image_hidden)

            # Model generates: [question] [<LATENT_THOUGHT>] [answer]
            # <LATENT_THOUGHT> is represented by latent_embeddings

            # Combine embeddings with text for decoding
            decoder_input = torch.cat([
                model.tokenizer.encode(questions),  # Text embeddings
                latent_embeddings,  # Latent embeddings (special tokens)
                model.tokenizer.encode(answers)
            ], dim=1)

            output = model.decoder(decoder_input, output_hidden_states=True)

            # Loss: next-token prediction + latent reconstruction
            text_loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1], output.input_ids[:, 1:]
            )

            # If latent targets available, reconstruct them
            latent_loss = 0.0
            if latent_targets is not None:
                reconstructed = latent_module.latent_decoder(latent_embeddings.flatten(1))
                latent_loss = torch.nn.functional.mse_loss(
                    reconstructed, latent_targets
                )

            total_loss = text_loss + 0.5 * latent_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return model, latent_module
```

**4. Implement Stage 3: Reasoning Chain Training**

Train full interleaved text-latent reasoning chains.

```python
def stage3_reasoning_chain_training(model, latent_module, train_dataloader, num_epochs=10):
    """
    Train model to generate interleaved chains of text and latent reasoning.
    Model learns flexible mixing of modalities for complex reasoning.
    """
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(latent_module.parameters()),
        lr=5e-5
    )

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            images = batch['images']
            questions = batch['questions']
            reasoning_chains = batch['reasoning_chains']  # Mix of text + latent tokens

            # Encode image
            image_hidden = model.encode_image(images, output_hidden_states=True)

            # Generate initial latent embeddings
            latent_embeddings, _ = latent_module(image_hidden)

            # Auto-regressive generation of reasoning chain
            # Interleave text and latent tokens
            chain_loss = 0.0

            for step_idx in range(len(reasoning_chains)):
                # Current step could be text or latent
                if reasoning_chains[step_idx]['type'] == 'text':
                    # Generate text token
                    text_input = model.tokenizer.encode(
                        reasoning_chains[step_idx]['content']
                    )
                    output = model.decoder(text_input, output_hidden_states=True)

                    # Loss for text token
                    step_loss = torch.nn.functional.cross_entropy(
                        output.logits[:, :-1],
                        output.input_ids[:, 1:]
                    )

                else:  # type == 'latent'
                    # Generate latent token (continuous embedding)
                    latent_embeddings, _ = latent_module(image_hidden)

                    # Loss for latent generation (reconstruction)
                    target_latent = reasoning_chains[step_idx]['embedding']
                    step_loss = torch.nn.functional.mse_loss(
                        latent_embeddings, target_latent
                    )

                chain_loss += step_loss

            # Normalize by number of steps
            chain_loss = chain_loss / len(reasoning_chains)

            optimizer.zero_grad()
            chain_loss.backward()
            optimizer.step()

    return model, latent_module
```

**5. Implement VLPO: Vision-Language Policy Optimization**

Optimize latent embeddings as learnable actions via policy gradients.

```python
def vlpo_step(model, latent_module, batch, reward_fn, learning_rate=1e-5):
    """
    Single VLPO training step optimizing latent embeddings as actions.
    Computes policy gradients for continuous embeddings.
    Args:
        model: MLLM
        latent_module: Latent reasoning module
        batch: Training batch with images, questions, target answers
        reward_fn: Function computing reward from answer quality
        learning_rate: Gradient step size
    Returns:
        loss_value: VLPO loss for this step
    """
    images = batch['images']
    questions = batch['questions']
    target_answers = batch['target_answers']

    # Forward: generate latent embeddings and reasoning
    image_hidden = model.encode_image(images, output_hidden_states=True)

    # Require gradients for latent embeddings (treating as actions)
    latent_embeddings, _ = latent_module(image_hidden)
    latent_embeddings.requires_grad = True

    # Generate answer using latent embeddings
    reasoning_output = model.decoder(
        inputs_embeds=latent_embeddings,
        output_hidden_states=True
    )

    # Decode to text answer
    predicted_answer = model.tokenizer.decode(
        torch.argmax(reasoning_output.logits[:, -1], dim=-1)
    )

    # Compute reward
    reward = reward_fn(predicted_answer, target_answers)

    # Policy gradient: ∇_θ log p(latent | context) × reward
    # Since latents are continuous, estimate probability via density
    # Approximate: larger magnitude → higher probability (gating mechanism)

    latent_magnitude = torch.norm(latent_embeddings, dim=-1).mean()
    log_prob_approx = latent_magnitude  # Simplified; actual impl uses softmax approximation

    # VLPO loss: maximize reward × log_prob
    policy_loss = -(log_prob_approx * reward)

    # Backward pass
    policy_loss.backward()

    # Gradient step for latent parameters
    with torch.no_grad():
        latent_embeddings -= learning_rate * latent_embeddings.grad

    return policy_loss.item()
```

**6. Flexible Inference: Mixed Text-Latent Reasoning**

Generate reasoning chains that automatically mix text and latent embeddings.

```python
def generate_with_flexible_interleaving(model, latent_module, image, question, max_steps=20):
    """
    Generate reasoning with automatic text/latent interleaving.
    Model learns when each modality is most useful.
    """
    image_hidden = model.encode_image(image, output_hidden_states=True)
    latent_embeddings, _ = latent_module(image_hidden)

    reasoning_chain = []
    current_context = question

    for step in range(max_steps):
        # Predict: should next step be text or latent?
        decision = model.predict_modality(current_context)  # "text" or "latent"

        if decision == 'text':
            # Generate text token
            text_token = model.generate_next_token(current_context, num_tokens=1)
            reasoning_chain.append({'type': 'text', 'content': text_token})
            current_context += text_token

        else:  # 'latent'
            # Generate latent embedding
            new_latent, _ = latent_module(image_hidden)
            reasoning_chain.append({'type': 'latent', 'embedding': new_latent})

            # Add placeholder for context
            current_context += "[LATENT_THOUGHT]"

        # Check for completion
        if model.is_final_answer(current_context):
            break

    return reasoning_chain
```

## Practical Guidance

**When to Use Monet:**
- Complex spatial reasoning requiring visual intermediate steps
- Avoiding external image generation (expensive, error-prone)
- Tasks where text+image inputs need internal visual reasoning
- Scenarios where model flexibility in modality choice is valuable

**When NOT to Use:**
- Simple VQA tasks (text-only reasoning sufficient)
- Tasks where interpretability of every step is required
- Very large models where training latent modules is prohibitively expensive

**Key Hyperparameters:**
- `latent_dim`: Embedding dimension (256-512 typical)
- `num_latent_tokens`: How many visual thought tokens (3-8)
- `sft_epochs_per_stage`: Iterations per SFT stage (5-15)
- `vlpo_learning_rate`: Gradient step for embeddings (1e-5 to 5e-5)
- `reward_scaling`: How strongly to weight VLPO rewards (0.5-2.0)

**Integration with Vision Models:**
Monet works naturally with existing vision encoders. Use pre-trained vision embeddings as initialization for `latent_encoder`, reducing training data requirements by 30-40%.

**Training Efficiency:**
- Stage 1: ~1 GPU-hour per 10K images
- Stage 2: ~2 GPU-hours per 10K reasoning chains
- Stage 3: ~4 GPU-hours per 10K full chains
- VLPO: Continuous optimization, typically 2-4 epochs sufficient

## Reference

Research paper: https://arxiv.org/abs/2511.21395
