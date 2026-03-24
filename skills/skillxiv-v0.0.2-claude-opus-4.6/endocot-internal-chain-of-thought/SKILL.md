---
name: endocot-internal-chain-of-thought
title: "EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.12252"
keywords: [Diffusion, Reasoning, Chain-of-Thought, Latent Space, Generation]
description: "Enable step-by-step reasoning in diffusion models through iterative latent state refinement. Condition diffusion on evolving thought states across multiple reasoning steps, grounded with textual supervision to prevent drift."
---

# Technique: Endogenous Chain-of-Thought via Iterative Latent Refinement

Diffusion models generate images by denoising—but they commit to solutions early in this process. EndoCoT enables intermediate reasoning by allowing the conditioning signal to evolve across steps. Rather than static guidance, the framework iteratively refines hidden states representing thoughts, each conditioned on the previous step's reasoning, then uses these refined states to guide generation.

This "endogenous" approach contrasts with external chain-of-thought: reasoning happens *within* the diffusion process via latent dynamics, enabling truly integrated visual-linguistic reasoning.

## Core Concept

EndoCoT combines three mechanisms:

1. **Iterative Thought Guidance**: Refine hidden states across multiple reasoning steps before denoising
2. **Terminal Thought Grounding**: Align final reasoning state with explicit textual reference using semantic loss
3. **Progressive Training**: First supervise intermediate steps, then optimize final output quality

This enables dynamic, evolving conditioning that guides diffusion through a reasoning trajectory.

## Architecture Overview

- **MLLM backbone**: Language model for thought generation
- **Hidden state buffer**: Evolving reasoning representations across steps
- **Thought refinement network**: Updates states based on prior reasoning
- **Text reference encoder**: Grounds final thoughts in language
- **DiT decoder**: Diffusion transformer conditioned on thought states
- **Progressive loss scheduler**: Balance reasoning supervision vs output quality

## Implementation Steps

### Step 1: Iterative Hidden State Refinement

Refine latent reasoning states across multiple steps before using for generation.

```python
import torch
import torch.nn as nn

class IterativeThoughtRefiner(nn.Module):
    def __init__(self, hidden_dim=768, num_reasoning_steps=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps

        # Thought evolution network
        self.refiner_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(num_reasoning_steps)
        ])

        # Attention for reasoning integration
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(num_reasoning_steps)
        ])

    def forward(self, initial_thought, reasoning_context):
        """
        Iteratively refine thought representations.

        initial_thought: (batch, hidden_dim) initial hidden state
        reasoning_context: (batch, seq_len, hidden_dim) context from MLLM
        """
        thought_trajectory = [initial_thought]

        current_thought = initial_thought

        for step in range(self.num_reasoning_steps):
            # Refine using MLP
            refined = self.refiner_layers[step](current_thought)

            # Integrate context via attention
            attended, _ = self.attention_layers[step](
                refined.unsqueeze(1),  # (batch, 1, hidden_dim)
                reasoning_context,  # (batch, seq_len, hidden_dim)
                reasoning_context
            )

            # Update thought
            current_thought = refined + attended.squeeze(1)

            thought_trajectory.append(current_thought)

        return torch.stack(thought_trajectory, dim=1)  # (batch, steps+1, hidden_dim)
```

### Step 2: Terminal Thought Grounding with Text Supervision

Anchor final reasoning state to explicit textual reference to prevent drift.

```python
class TerminalThoughtGrounder(nn.Module):
    def __init__(self, hidden_dim=768, text_encoder=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_encoder = text_encoder

        # Projection for alignment
        self.thought_projector = nn.Linear(hidden_dim, hidden_dim)
        self.text_projector = nn.Linear(hidden_dim, hidden_dim)

    def ground_with_text(self, final_thought, reference_text):
        """
        Align final thought with textual reference using semantic loss.

        final_thought: (batch, hidden_dim)
        reference_text: str or tokenized reference
        """
        # Encode reference text
        if isinstance(reference_text, str):
            reference_embedding = self.text_encoder.encode(reference_text)
        else:
            reference_embedding = self.text_encoder(reference_text)

        # Project both to common space
        thought_proj = self.thought_projector(final_thought)
        text_proj = self.text_projector(reference_embedding)

        # Semantic L2 alignment loss
        alignment_loss = torch.nn.functional.mse_loss(thought_proj, text_proj)

        return alignment_loss

    def forward(self, thought_trajectory, reference_text):
        """
        Compute grounding loss for final thought.

        thought_trajectory: (batch, steps, hidden_dim)
        reference_text: reference for grounding
        """
        final_thought = thought_trajectory[:, -1, :]

        grounding_loss = self.ground_with_text(final_thought, reference_text)

        return grounding_loss
```

### Step 3: Conditional Diffusion with Evolving Thought

Use thought trajectory to condition diffusion generation at each step.

```python
class DiffusionTransformerWithEvolvedThoughts(nn.Module):
    def __init__(self, dit_model, hidden_dim=768):
        super().__init__()
        self.dit = dit_model
        self.hidden_dim = hidden_dim

        # Project thoughts to condition dimension
        self.thought_to_condition = nn.Linear(hidden_dim, dit_model.cond_dim)

    def forward(
        self,
        noise,
        timestep,
        thought_trajectory,
        instruction_tokens=None
    ):
        """
        Run one diffusion step with evolved thought conditioning.

        noise: (batch, channels, height, width)
        timestep: int or (batch,)
        thought_trajectory: (batch, steps, hidden_dim)
        instruction_tokens: optional additional conditioning
        """
        batch_size = noise.shape[0]

        # Select appropriate thought based on diffusion progress
        # (early steps use earlier thoughts, late steps use refined thoughts)
        progress = timestep / 1000  # Normalized to [0, 1]
        step_idx = min(
            int(progress * thought_trajectory.shape[1]),
            thought_trajectory.shape[1] - 1
        )

        current_thought = thought_trajectory[:, step_idx, :]  # (batch, hidden_dim)

        # Project to condition
        conditioning = self.thought_to_condition(current_thought)  # (batch, cond_dim)

        # Optional: concatenate with instruction tokens
        if instruction_tokens is not None:
            conditioning = torch.cat([conditioning, instruction_tokens], dim=-1)

        # Denoising step
        model_output = self.dit(
            noise,
            timestep,
            c=conditioning
        )

        return model_output
```

### Step 4: Progressive Two-Stage Training

First stage: supervise all intermediate reasoning steps. Second stage: optimize only final output.

```python
def train_endocot(
    model,
    data_loader,
    text_encoder,
    num_epochs=10,
    stage1_epochs=5
):
    """
    Two-stage training: reasoning -> output quality.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        is_stage1 = epoch < stage1_epochs

        for batch_idx, (images, prompts, texts) in enumerate(data_loader):
            # Forward pass
            image_noise = torch.randn_like(images)

            # Get initial thought
            prompt_embedding = text_encoder.encode(prompts)

            # Refine thoughts
            thought_refiner = model.thought_refiner
            thought_trajectory = thought_refiner(
                prompt_embedding,
                text_encoder.encode(texts).unsqueeze(1)
            )

            if is_stage1:
                # Stage 1: Supervise all intermediate thoughts
                total_loss = 0

                for step_idx in range(thought_trajectory.shape[1]):
                    thought = thought_trajectory[:, step_idx, :]

                    # Generate with this thought
                    outputs = model.dit_with_thoughts(
                        image_noise,
                        torch.tensor([500]),  # Midpoint diffusion step
                        thought.unsqueeze(1)
                    )

                    # Supervise against ground truth
                    step_loss = torch.nn.functional.mse_loss(outputs, images)
                    total_loss += step_loss

                # Also supervise terminal grounding
                grounding_loss = model.grounder.ground_with_text(
                    thought_trajectory[:, -1, :],
                    texts
                )

                total_loss += 0.1 * grounding_loss

            else:
                # Stage 2: Optimize final output quality only
                # Freeze intermediate supervision
                thought_trajectory = thought_trajectory.detach()

                final_thought = thought_trajectory[:, -1, :]

                # Full generation with final thought
                outputs = model.dit_with_thoughts(
                    image_noise,
                    torch.tensor([0]),  # Final step
                    final_thought.unsqueeze(1)
                )

                total_loss = torch.nn.functional.mse_loss(outputs, images)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                stage = "Stage1" if is_stage1 else "Stage2"
                print(f"Epoch {epoch} [{stage}]: Loss = {total_loss.item():.4f}")

    return model
```

### Step 5: Integration with Prompt Engineering

Demonstrate full workflow with reasoning prompts.

```python
def generate_with_endogenous_reasoning(
    model,
    text_encoder,
    initial_prompt,
    reasoning_steps=4,
    num_diffusion_steps=50
):
    """
    Full generation pipeline with endogenous reasoning.
    """
    # Step 1: Generate reasoning trajectory
    prompt_embedding = text_encoder.encode(initial_prompt)

    thought_refiner = model.thought_refiner
    thought_trajectory = thought_refiner(
        prompt_embedding,
        torch.zeros(1, 1, 768)  # Placeholder context
    )

    # Step 2: Prepare diffusion
    noise = torch.randn(1, 4, 64, 64)  # Latent space noise
    dit_model = model.dit_with_thoughts

    # Step 3: Iterative diffusion with evolving thoughts
    for diffusion_step in range(num_diffusion_steps):
        timestep = (num_diffusion_steps - diffusion_step) / num_diffusion_steps

        # Update thought based on diffusion progress
        if diffusion_step % (num_diffusion_steps // reasoning_steps) == 0:
            # Optionally re-refine thought mid-generation
            pass

        # Denoising step
        noise = dit_model(
            noise,
            int(timestep * 1000),
            thought_trajectory
        )

    # Step 4: Decode from latent space
    image = model.vae.decode(noise)

    return image
```

## Practical Guidance

**When to Use:**
- Multi-step reasoning required for generation (e.g., "draw a car then add wheels")
- Tasks benefiting from intermediate planning before generation
- Scenarios where output quality depends on reasoning coherence
- Fine-grained control over generation trajectory

**When NOT to Use:**
- Simple prompt-to-image tasks without reasoning requirements
- Extreme latency constraints (iterative refinement adds overhead)
- Tasks where early commitment to solution is beneficial

**Hyperparameter Tuning:**
- **num_reasoning_steps**: 2-6; more enables finer reasoning
- **stage1_epochs vs stage2_epochs**: 50-50 split typical; adjust based on reasoning difficulty
- **grounding_loss weight**: 0.05-0.2; stronger grounding prevents drift
- **thought refinement MLP hidden dim**: Match backbone hidden dimension

**Common Pitfalls:**
- Insufficient grounding loss causing thought drift
- Over-weighting intermediate supervision (stage 1), hurting final quality
- Thought trajectory too rigid (insufficient refinement network capacity)
- Misalignment between thought dimensionality and DIT conditioning

## Reference

[EndoCoT paper on arXiv](https://arxiv.org/abs/2603.12252)
