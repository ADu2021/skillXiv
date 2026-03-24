---
name: continue-thinking-token
title: "Learning a Continue-Thinking Token for Enhanced Test-Time Scaling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.11274"
keywords: [test-time-scaling, reasoning, learned-tokens, inference, reinforcement-learning]
description: "Train a specialized continue-thinking token via reinforcement learning to extend reasoning depth during inference, achieving superior performance over fixed-token baselines."
---

# Learning a Continue-Thinking Token for Enhanced Test-Time Scaling

## Core Concept

This work investigates learned tokens as a mechanism for extending reasoning in language models during inference. Rather than using fixed textual triggers like "Wait", a specialized `<|continue-thinking|>` token is trained via reinforcement learning with only its embedding updated. This learned token provides superior control over reasoning extension compared to static approaches, as evidenced by 4.2% improvements on mathematical benchmarks like GSM8K.

## Architecture Overview

- **Learned Embedding**: Single new token whose embedding is trained while model weights remain frozen
- **Token Integration**: Insert learned token at inference time to trigger extended reasoning
- **Reinforcement Learning Training**: Optimize token embedding through standard RL signals (pass/fail on test tasks)
- **Frozen Base Model**: Maintain fixed LLM weights to ensure interpretability and computational efficiency
- **Budget Control**: Learned token provides implicit control over reasoning depth allocation

## Implementation

### Step 1: Initialize and Register the Learned Token

Add a new learnable token to the vocabulary:

```python
import torch
import torch.nn as nn

class ContinueThinkingToken:
    """
    Manages a single learned continue-thinking token.
    """
    def __init__(self, embedding_dim, token_id=None):
        self.embedding_dim = embedding_dim
        self.token_id = token_id or 999999  # Reserve high ID

        # Initialize embedding with small random values
        self.embedding = nn.Parameter(
            torch.randn(1, embedding_dim) * 0.01
        )

    def get_token_id(self):
        return self.token_id

    def get_embedding(self):
        """Returns embedding for injection into forward pass"""
        return self.embedding

    def reset_initialization(self):
        """Reinitialize if needed"""
        self.embedding.data = torch.randn_like(self.embedding) * 0.01
```

### Step 2: Inject Token During Inference

Insert learned token to trigger extended reasoning:

```python
def inject_continue_token(model, input_ids, continue_token_embedding,
                          position='after_answer'):
    """
    Injects continue-thinking token into sequence at strategic position.

    Args:
        model: Language model
        input_ids: [batch, seq_len] token IDs
        continue_token_embedding: [1, embedding_dim] learnable embedding
        position: 'after_answer' or 'at_end'
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Get base embeddings from model
    base_embeddings = model.get_input_embeddings()(input_ids)
    # [batch, seq_len, embedding_dim]

    if position == 'after_answer':
        # Find where model's generation ends
        seq_len = input_ids.shape[1]
        continue_embedding = continue_token_embedding.expand(
            batch_size, -1, -1
        )  # [batch, 1, embedding_dim]

        # Insert at end
        enhanced_embeddings = torch.cat(
            [base_embeddings, continue_embedding],
            dim=1
        )
    else:
        # position == 'at_end'
        continue_embedding = continue_token_embedding.expand(
            batch_size, -1, -1
        )
        enhanced_embeddings = torch.cat(
            [base_embeddings, continue_embedding],
            dim=1
        )

    return enhanced_embeddings
```

### Step 3: Generate with Continued Thinking

Extend generation using the learned token:

```python
def generate_with_continue_thinking(
    model, input_ids, continue_token_embedding,
    max_new_tokens=200, initial_tokens=100, continue_budget=100
):
    """
    Generate with extended reasoning budget controlled by learned token.

    Args:
        model: Frozen language model
        input_ids: [batch, seq_len] initial prompt
        continue_token_embedding: learned embedding
        max_new_tokens: total generation budget
        initial_tokens: tokens before triggering continuation
        continue_budget: additional tokens after trigger
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Phase 1: Generate initial response
    with torch.no_grad():
        initial_output = model.generate(
            input_ids,
            max_new_tokens=initial_tokens,
            do_sample=False
        )

    # Phase 2: Inject continue token and extend generation
    # Get embeddings including continue token
    enhanced_embeddings = inject_continue_token(
        model, initial_output, continue_token_embedding,
        position='after_answer'
    )

    # Generate continuation
    with torch.no_grad():
        final_output = model.generate(
            inputs_embeds=enhanced_embeddings,
            max_new_tokens=continue_budget,
            do_sample=False
        )

    return final_output
```

### Step 4: Compute Reinforcement Learning Loss

Optimize token embedding using answer correctness:

```python
def compute_rl_loss(model, batch_prompts, batch_answers, batch_labels,
                    continue_token_embedding):
    """
    Compute policy gradient loss for continue-thinking token.

    Args:
        model: Language model (frozen weights)
        batch_prompts: [batch] list of prompt strings
        batch_answers: [batch, num_samples] generated answers
        batch_labels: [batch] binary correctness labels
        continue_token_embedding: learnable token embedding
    """
    batch_size = len(batch_prompts)
    num_samples = batch_answers.shape[1]

    # Tokenize prompts
    tokenized = model.tokenizer(
        batch_prompts, return_tensors='pt', padding=True
    )
    input_ids = tokenized['input_ids']  # [batch, prompt_len]

    # Inject continue token
    enhanced_embeddings = inject_continue_token(
        model, input_ids, continue_token_embedding,
        position='after_answer'
    )

    # Forward pass (frozen model)
    with torch.no_grad():
        outputs = model(inputs_embeds=enhanced_embeddings)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Compute log probabilities for generated tokens
    log_probs = torch.log_softmax(logits, dim=-1)

    # For each sample in batch, compute advantage
    losses = []

    for i in range(batch_size):
        # Get correctness of this sample
        is_correct = batch_labels[i].float()

        # Compute advantage: positive if correct, negative if wrong
        advantage = is_correct - 0.5  # Center around 0

        # Log probability of generated answer
        sample_log_prob = 0
        for j, token_id in enumerate(batch_answers[i]):
            sample_log_prob += log_probs[i, j, token_id]

        # Policy gradient: -advantage * log_prob
        loss = -advantage * sample_log_prob

        losses.append(loss)

    return torch.stack(losses).mean()
```

### Step 5: Training Loop

Train the learned token embedding:

```python
def train_continue_token(model, train_dataloader, num_epochs=5):
    """
    Train continue-thinking token via reinforcement learning.

    Args:
        model: Base language model (weights frozen)
        train_dataloader: DataLoader with (prompts, answers, labels)
        num_epochs: training epochs
    """
    # Initialize learned token
    continue_token = ContinueThinkingToken(
        embedding_dim=model.config.hidden_size
    )

    # Optimizer for token embedding only
    optimizer = torch.optim.Adam(
        [continue_token.embedding],
        lr=1e-3
    )

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    device = model.device

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_idx, (prompts, answers, labels) in enumerate(train_dataloader):
            # Move labels to device
            labels = labels.to(device)

            # Compute loss
            loss = compute_rl_loss(
                model, prompts, answers, labels,
                continue_token.embedding
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return continue_token
```

## Practical Guidance

- **Token Initialization**: Start with small random values (σ=0.01); avoid zero initialization
- **Training Data**: Use task where correctness is verifiable (math, coding, logic)
- **Batch Size**: Use reasonable batch sizes (16-32); very small batches increase variance
- **Learning Rate**: Start with 1e-3; adjust based on convergence; may need warmup
- **Evaluation Metric**: Test on held-out tasks; measure both Pass@1 and Pass@K improvements
- **Comparison Baselines**: Compare against fixed tokens ("Wait", "Let me think"), no continuation, and standard test-time scaling
- **Integration**: Token can be added to any frozen LLM; requires only embedding-dimension knowledge

## Reference

Paper: arXiv:2506.11274
Key metrics: 4.2% improvement on GSM8K over base model with budget forcing
Related work: Test-time scaling, inference optimization, verifiable rewards, chain-of-thought
