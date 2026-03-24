---
name: sequential-diffusion-language-models
title: "Sequential Diffusion Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.24007"
keywords: [diffusion models, language generation, parallel decoding, token prediction, inference efficiency, transformers, next-sequence prediction, multi-token generation]
description: "Implement adaptive parallel decoding for language models using diffusion-based next-sequence prediction. Enable dynamic block-based token generation with confidence thresholds to achieve 2x+ speedups while maintaining competitive performance. Retrofit existing autoregressive models with minimal additional training data."
---

## Achieve Adaptive Fast Decoding for Language Models

This skill teaches how to implement Sequential Diffusion Language Models (SDLMs), which generate multiple tokens per inference step by combining diffusion principles with autoregressive language modeling. Instead of generating one token at a time, SDLMs predict variable-length sequences within fixed-size masked blocks, then dynamically select which predictions to keep based on model confidence.

### The Problem

Standard autoregressive language models generate one token per forward pass, making inference inherently sequential and slow. Existing multi-token prediction methods either require architectural modifications, don't support KV caching for efficiency, or use fixed output block sizes that don't adapt to prediction difficulty.

### Core Concept: Next Sequence Prediction (NSP)

NSP generalizes next-token and next-block prediction into a unified framework. The model predicts a full sequence within a masked block during training and inference, then uses confidence scores to determine how many predicted tokens to accept at each generation step. This creates an adaptive length generation strategy: easy tokens get accepted quickly, hard tokens trigger shorter steps that allow the model to refine predictions.

The key insight is that "causality matters for historical context but tokens being generated in the current step can attend to each other bidirectionally." This enables parallel training on next-block predictions while maintaining autoregressive semantics during inference.

### Architecture Overview

- **Training Block Setup**: Partition input sequence into historical context (masked to attend only to previous tokens) and prediction block (full mutual attention allowed). Train the model to predict masked token positions within the prediction block using a custom attention mask.

- **Attention Mechanism**: Apply causal masking for historical tokens u<i. Apply full mutual attention for prediction tokens u,v≥i. This enables parallel computation during training while respecting causality at generation time.

- **Confidence Selection**: After each forward pass within a block, use logit probabilities or entropy-normalized scores to determine how many generated tokens to accept. Tokens below a confidence threshold remain masked for refinement in the next iteration.

- **Inference Methods**: Two strategies—greedy decoding accepts tokens above a threshold per step, and self-speculative decoding validates predictions through consistency checks before committing them.

- **KV Cache Compatibility**: Unlike fixed block diffusion, NSP maintains standard transformer KV caching. Only tokens added to the sequence extend the cache, enabling efficient streaming generation without special logic.

### Implementation Details

#### Step 1: Create Masked Block Attention Pattern

Implement custom attention masking that enforces causality for history and allows full attention within the prediction block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_block_attention_mask(seq_len, history_len, block_size, device):
    """
    Create attention mask for NSP training.

    Args:
        seq_len: Total sequence length (history + block)
        history_len: Number of historical tokens (read-only)
        block_size: Size of prediction block
        device: Torch device

    Returns:
        Mask tensor shape (seq_len, seq_len) where 1 = attend, 0 = mask
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Historical tokens can only attend to previous history
    for i in range(history_len):
        mask[i, :i+1] = True

    # Prediction block tokens (history_len:) can attend to history and each other
    for i in range(history_len, seq_len):
        mask[i, :i+1] = True  # Causal for history
        mask[i, history_len:seq_len] = True  # Full attention within block

    return mask.unsqueeze(0)  # Add batch dimension


def apply_block_mask_to_attention(attn_scores, mask):
    """
    Apply block mask to attention scores before softmax.

    Args:
        attn_scores: Shape (batch, heads, query_len, key_len)
        mask: Shape (1, seq_len, seq_len)

    Returns:
        Masked attention scores
    """
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    return attn_scores
```

#### Step 2: Implement Confidence-Based Token Selection

After each generation step within a block, use confidence scores to decide which tokens to accept and which to keep for refinement.

```python
def select_tokens_by_confidence(logits, confidence_threshold=0.8, method='logit'):
    """
    Select tokens based on model confidence for acceptance.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        confidence_threshold: Float in [0, 1], tokens above this are accepted
        method: 'logit' uses max probability, 'entropy' uses negative entropy

    Returns:
        accepted_mask: Boolean mask (batch, seq_len) indicating accepted tokens
        confidence_scores: Float tensor (batch, seq_len) with confidence values
    """
    batch_size, seq_len, vocab_size = logits.shape

    if method == 'logit':
        # Maximum probability across vocabulary
        probs = F.softmax(logits, dim=-1)
        confidence_scores, _ = torch.max(probs, dim=-1)  # (batch, seq_len)

    elif method == 'entropy':
        # Normalize entropy to [0, 1]: higher entropy = lower confidence
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))
        confidence_scores = 1.0 - (entropy / max_entropy)

    else:
        raise ValueError(f"Unknown method: {method}")

    accepted_mask = confidence_scores >= confidence_threshold
    return accepted_mask, confidence_scores
```

#### Step 3: Implement Greedy Decoding with Dynamic Block Size

Generate sequences by iteratively predicting blocks, accepting high-confidence tokens, and refining low-confidence predictions.

```python
@torch.no_grad()
def greedy_decode_with_nsp(model, input_ids, block_size=16,
                            confidence_threshold=0.8, max_length=256):
    """
    Generate tokens using greedy NSP decoding.

    Args:
        model: Language model with custom attention masking
        input_ids: Starting prompt, shape (batch, prompt_len)
        block_size: Size of prediction block
        confidence_threshold: Confidence level for accepting tokens
        max_length: Maximum generation length

    Returns:
        generated_ids: Full generated sequence
    """
    device = input_ids.device
    current_ids = input_ids.clone()

    while current_ids.shape[1] < max_length:
        batch_size, seq_len = current_ids.shape
        remaining_len = max_length - seq_len

        # Determine actual block size for this step
        current_block_size = min(block_size, remaining_len)

        # Prepare input with masking for prediction block
        history_len = seq_len

        # Forward pass to get logits for the entire padded sequence
        # (in practice, use masked_fill in the model's attention)
        padded_ids = torch.cat([
            current_ids,
            torch.full((batch_size, current_block_size),
                      model.config.pad_token_id, device=device)
        ], dim=1)

        with torch.no_grad():
            outputs = model(padded_ids, output_hidden_states=False)
            logits = outputs.logits

        # Extract logits for the prediction block only
        block_logits = logits[:, history_len:history_len+current_block_size, :]

        # Select tokens by confidence
        accepted_mask, conf_scores = select_tokens_by_confidence(
            block_logits,
            confidence_threshold=confidence_threshold,
            method='logit'
        )

        # Get greedy predictions
        next_token_ids = torch.argmax(block_logits, dim=-1)  # (batch, block_size)

        # Build output: accept high-confidence tokens in order
        num_accepted = 0
        for i in range(current_block_size):
            if accepted_mask[:, i].all():
                num_accepted = i + 1
            else:
                break

        if num_accepted == 0:
            # If no tokens meet threshold, take the single best token
            num_accepted = 1

        # Append accepted tokens
        accepted_tokens = next_token_ids[:, :num_accepted]
        current_ids = torch.cat([current_ids, accepted_tokens], dim=1)

        # Early stop if we hit pad tokens
        if (accepted_tokens == model.config.pad_token_id).all():
            break

    return current_ids
```

#### Step 4: Training with Masked Block Objective

Train the model to predict masked positions within blocks using a standard language modeling loss, but with the custom attention mask applied.

```python
class SDLMTrainer:
    """
    Trainer for Sequential Diffusion Language Models.
    Implements masked block prediction training.
    """

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def prepare_masked_batch(self, input_ids, history_ratio=0.7):
        """
        Prepare a batch with masked prediction blocks.

        Args:
            input_ids: (batch, seq_len)
            history_ratio: Fraction of sequence kept as history

        Returns:
            prepared_ids: Sequence with padding for block
            attention_mask: Custom block attention mask
            target_positions: Which positions to compute loss on
        """
        batch_size, seq_len = input_ids.shape
        history_len = int(seq_len * history_ratio)
        block_size = seq_len - history_len

        # Create attention mask
        full_seq_len = seq_len + block_size  # Double for prediction block
        attn_mask = create_block_attention_mask(
            full_seq_len, history_len, block_size, self.device
        )

        # Prepare input: history + padding for block predictions
        prepared = torch.cat([
            input_ids[:, :history_len],
            torch.full((batch_size, block_size), -100, device=self.device)  # -100 for ignoring in loss
        ], dim=1)

        return prepared, attn_mask, list(range(history_len, history_len + block_size))

    def training_step(self, batch):
        """
        Compute training loss for NSP objective.

        Args:
            batch: Dict with 'input_ids' key

        Returns:
            loss: Scalar loss value
        """
        input_ids = batch['input_ids'].to(self.device)

        # Prepare masked block batch
        prepared_ids, block_mask, target_pos = self.prepare_masked_batch(input_ids)

        # Forward pass with custom attention mask
        # (Model's forward should accept 'block_attention_mask' parameter)
        outputs = self.model(
            prepared_ids,
            block_attention_mask=block_mask
        )

        logits = outputs.logits

        # Compute loss only on target prediction block positions
        batch_size = input_ids.shape[0]
        target_logits = logits[:, target_pos, :]  # (batch, block_size, vocab)
        target_labels = input_ids[:, int(input_ids.shape[1]*0.7):, :]

        loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.shape[-1]),
            target_labels.reshape(-1),
            ignore_index=-100
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Practical Guidance

#### Hyperparameter Table

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| block_size | 4-32 | Larger blocks = more parallelism but harder to predict. Start at 16. |
| confidence_threshold | 0.75-0.95 | Higher = stricter, fewer tokens per step. 0.85 balances speed/quality. |
| history_ratio | 0.6-0.8 | Fraction of sequence used as context. 0.7 typical. |
| training_samples | 3-5M | For retrofitting 3B+ models. Less for smaller models. |
| learning_rate | 1e-5 to 5e-5 | Lower than typical LM training. Tune from 2e-5. |
| confidence_method | 'logit' or 'entropy' | Logit faster, entropy more calibrated. Start with logit. |

#### When to Use SDLM

- Deployment scenarios requiring faster inference without retraining from scratch
- Streaming applications where variable latency tolerance is acceptable
- Resource-constrained environments where compute budgets are fixed
- Production systems running existing autoregressive models (requires minimal retraining)
- Benchmarks valuing throughput-per-watt efficiency

#### When NOT to Use SDLM

- Applications requiring consistent ultra-low latency per token (SDLMs add variable latency per step)
- Scenarios demanding guaranteed exact deterministic output (confidence thresholds add stochasticity)
- Use cases where quality cannot degrade at all (SDLM trades some quality for speed)
- Real-time interactive applications where 2-3 rounds of refinement per block are unacceptable
- Models under 1B parameters (training efficiency gains diminish; use standard autoregressive)
- Systems already optimized with speculative decoding or other multi-token methods (marginal gains)

#### Common Pitfalls

1. **Setting confidence threshold too high**: Results in single-token generation per step, negating speedup. Start conservative (0.8) and gradually increase if quality permits.

2. **Ignoring KV cache invalidation**: When tokens are not accepted and need refinement, ensure KV cache state aligns with current sequence position. Bugs here cause silent correctness errors.

3. **Block size larger than necessary**: Large blocks increase prediction difficulty. Confidence threshold must drop to maintain throughput, harming quality. Tune together.

4. **Insufficient training data**: Retrofitting works with 3.5M tokens, but model quality depends on data diversity. Use data similar to downstream tasks.

5. **Not ablating confidence method**: Logit-based and entropy-based selection have different behaviors. Test both on your quality metrics before production deployment.

6. **Applying to instruction-tuned models without fine-tuning**: SDLM requires training on the specific instruction format. Base model training alone may not generalize to instructions.

7. **Mismatch between training and inference confidence thresholds**: If you train with one threshold but deploy with another, performance degrades. Must match or re-train.

### Reference

- **Paper**: Sequential Diffusion Language Models
- **ArXiv**: https://arxiv.org/abs/2509.24007
- **Code**: OpenGVLab/SDLM (GitHub)
- **Key Authors**: Examine arXiv page for full author list and affiliations
