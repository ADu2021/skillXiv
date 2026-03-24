---
name: set-block-decoding-inference-acceleration
title: "Set Block Decoding is a Language Model Inference Accelerator"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.04185"
keywords: [language model, inference acceleration, token prediction, masked generation, diffusion, transformer, decoding]
description: "Accelerate language model generation 3-5x by combining autoregressive and masked token prediction. Works via fine-tuning—no architectural changes needed. Parallel decode non-consecutive tokens with entropy-bounded sampling."
---

## Accelerate LLM Inference Without Architectural Changes

### Problem Context

Language model inference is memory-bound: generating each token requires a forward pass through the entire transformer, even though computational density is low. Standard autoregressive decoding generates one token per forward pass, creating a bottleneck for latency-critical applications like real-time chat, coding assistants, and search.

Existing acceleration methods have tradeoffs:

- **Speculative decoding** requires training or maintaining separate draft models
- **Auxiliary prediction heads** (Medusa, Eagle) add architectural complexity and inference overhead
- **Pure diffusion models** lose efficient KV-cache compatibility
- **Early exit strategies** sacrifice accuracy for speed

Set Block Decoding (SBD) eliminates these constraints by enabling parallel generation of multiple non-consecutive tokens within a single fine-tuned model, achieving 3-5x reduction in forward passes.

### Core Concept

SBD integrates two objectives into a single transformer during training:

1. **Next Token Prediction (NTP)**: Standard autoregressive training—predict token at position t+1 given positions 0...t
2. **Masked Token Prediction (MATP)**: Given past tokens and a set of future positions with mask tokens, predict the masked tokens using bidirectional attention

During inference, the model samples multiple non-consecutive future tokens in parallel using entropy-bounded sampling, which identifies tokens with low mutual information dependencies and generates them together. This flexibility allows adaptive batch generation without the overhead of speculative decoding or auxiliary heads.

The mathematical foundation: the model learns to decompose the joint conditional distribution P(tokens at positions S | context) into independent or weakly-dependent marginals, enabling parallel sampling.

### Architecture Overview

**Training Architecture:**

- Hybrid attention pattern applied only during SBD training blocks
  - Past tokens: standard causal attention (NTP objective)
  - Future block of size B (sampled 2-16): bidirectional attention with mask tokens (MATP objective)
- Loss function: weighted combination of NTP loss and MATP loss
  - Both components are essential; ablation shows removing either degrades performance significantly
- Block size varies per training example to ensure robustness across generation scenarios

**Inference Pipeline:**

- Compute logits for masked token positions using bidirectional context
- Apply Entropy-Bounded (EB) Sampler to select which tokens to unfreeze
  - Sort masked positions by conditional entropy
  - Greedily unfreeze tokens until cumulative entropy exceeds threshold γ (hyperparameter)
- Generate all unmasked tokens together in parallel
- Update KV-cache and mask remaining positions; repeat

**Memory and Compute:**

- No additional parameters beyond standard LLM
- KV-cache remains unchanged—maintains generation efficiency
- Forward pass cost: proportional to unmasked token count
- Typical speedup: 3-5x when γ tuned for accuracy-speed tradeoff

### Implementation

#### Step 1: Prepare Training Data and Model Checkpoint

Start with a standard next-token prediction model (e.g., Llama-3.1 8B). Create training batches with variable-size future token blocks injected as mask tokens.

```python
# Training data preparation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set mask token (add to vocabulary if necessary)
MASK_TOKEN_ID = tokenizer.eos_token_id + 1

def create_sbd_training_batch(input_ids, block_size=None):
    """
    Create training example with masked future block.

    Args:
        input_ids: token sequence of shape (seq_len,)
        block_size: size of future block to mask. If None, sample from [2, 16]

    Returns:
        sbd_input_ids: input sequence with future block masked
        past_labels: labels for NTP objective (positions t+1 after context)
        masked_labels: labels for MATP objective (positions in masked block)
        mask_positions: boolean mask indicating which positions are masked
    """
    if block_size is None:
        block_size = torch.randint(2, 17, (1,)).item()

    seq_len = input_ids.shape[0]

    # Sample random position for future block start
    max_start = max(1, seq_len - block_size - 1)
    block_start = torch.randint(1, max_start + 1, (1,)).item()
    block_end = block_start + block_size

    # Create masked version
    sbd_input_ids = input_ids.clone()
    sbd_input_ids[block_start:block_end] = MASK_TOKEN_ID

    # Create labels
    past_labels = input_ids.clone()
    past_labels[block_end:] = -100  # Ignore tokens after masked block in NTP

    masked_labels = input_ids[block_start:block_end].clone()
    masked_labels_full = torch.full_like(input_ids, -100)
    masked_labels_full[block_start:block_end] = masked_labels

    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    mask_positions[block_start:block_end] = True

    return {
        'input_ids': sbd_input_ids,
        'ntp_labels': past_labels,
        'matp_labels': masked_labels_full,
        'mask_positions': mask_positions
    }

# Test data preparation
sample_ids = torch.tensor(tokenizer.encode("The quick brown fox jumps over the lazy dog."))
batch = create_sbd_training_batch(sample_ids)
```

#### Step 2: Implement SBD Training Objective

Modify the forward pass to compute both NTP and MATP losses. The dual objective trains the model to perform both tasks simultaneously.

```python
# SBD training loss implementation
import torch.nn.functional as F

class SBDTrainingLoss:
    def __init__(self, ntp_weight=1.0, matp_weight=1.0):
        """
        Weighted combination of NTP and MATP objectives.

        Args:
            ntp_weight: weight for next-token prediction loss
            matp_weight: weight for masked-token prediction loss
        """
        self.ntp_weight = ntp_weight
        self.matp_weight = matp_weight

    def __call__(self, model_output, batch):
        """
        Calculate combined SBD loss.

        Args:
            model_output: model logits of shape (batch_size, seq_len, vocab_size)
            batch: dict with 'ntp_labels', 'matp_labels', 'mask_positions'

        Returns:
            loss: scalar tensor
            loss_dict: dict with individual loss components
        """
        logits = model_output.logits
        vocab_size = logits.shape[-1]

        # NTP Loss: standard causal language modeling before masked block
        ntp_labels = batch['ntp_labels']
        ntp_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            ntp_labels.view(-1),
            ignore_index=-100
        )

        # MATP Loss: predict masked positions using bidirectional attention
        # During training, the model attends bidirectionally to masked block
        matp_labels = batch['matp_labels']
        matp_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            matp_labels.view(-1),
            ignore_index=-100
        )

        # Combined loss
        total_loss = self.ntp_weight * ntp_loss + self.matp_weight * matp_loss

        return total_loss, {
            'ntp_loss': ntp_loss.item(),
            'matp_loss': matp_loss.item(),
            'total_loss': total_loss.item()
        }

# Training setup
sbd_loss_fn = SBDTrainingLoss(ntp_weight=1.0, matp_weight=1.0)

# In training loop:
# for batch in train_dataloader:
#     sbd_batch = create_sbd_training_batch(batch['input_ids'])
#     output = model(**sbd_batch)
#     loss, loss_dict = sbd_loss_fn(output, sbd_batch)
#     loss.backward()
#     optimizer.step()
```

#### Step 3: Implement Entropy-Bounded Sampler for Inference

The EB Sampler identifies which masked tokens to unfreeze by sorting positions by conditional entropy and greedily unmasking until entropy budget is exhausted.

```python
# Entropy-Bounded Sampler
import torch
import torch.nn.functional as F

class EntropyBoundedSampler:
    def __init__(self, entropy_threshold=1.0):
        """
        Sampler that selects tokens to generate based on conditional entropy.

        Args:
            entropy_threshold: gamma parameter controlling speed-accuracy tradeoff
                - Low values (0.5-1.0): fewer tokens per forward pass, closer to autoregressive
                - High values (2.0-4.0): more tokens per forward pass, faster but more approximation
        """
        self.entropy_threshold = entropy_threshold

    def __call__(self, logits, masked_positions):
        """
        Select which masked positions to unfreeze.

        Args:
            logits: model output logits for masked positions, shape (batch, num_masked, vocab_size)
            masked_positions: indices of masked tokens in sequence

        Returns:
            tokens_to_unfreeze: boolean mask of which positions to decode (batch, num_masked)
            entropies: conditional entropies of each position (batch, num_masked)
        """
        # Compute probability distributions
        probs = F.softmax(logits, dim=-1)

        # Compute conditional entropy for each position
        # H(p) = -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)
        entropies = -(probs * log_probs).sum(dim=-1)  # (batch, num_masked)

        # Sort by entropy (ascending: lowest entropy first)
        sorted_entropies, sorted_indices = torch.sort(entropies, dim=-1)

        # Greedily unfreeze tokens until cumulative entropy exceeds threshold
        cumsum_entropy = torch.cumsum(sorted_entropies, dim=-1)
        num_to_unfreeze = (cumsum_entropy <= self.entropy_threshold).sum(dim=-1)
        num_to_unfreeze = torch.clamp(num_to_unfreeze, min=1)  # Always unfreeze at least 1

        # Create selection mask
        tokens_to_unfreeze = torch.zeros_like(entropies, dtype=torch.bool)
        for b in range(entropies.shape[0]):
            tokens_to_unfreeze[b, sorted_indices[b, :num_to_unfreeze[b]]] = True

        return tokens_to_unfreeze, entropies

    def sample_tokens(self, logits, masked_positions, temperature=1.0):
        """
        Sample token IDs for positions to unfreeze.

        Args:
            logits: model logits for masked positions
            masked_positions: indices in original sequence
            temperature: sampling temperature (1.0 = no modification, >1.0 = more entropy)

        Returns:
            sampled_token_ids: sampled tokens for unfrozen positions
            unfrozen_mask: boolean mask of unfrozen positions
        """
        unfrozen_mask, entropies = self(logits, masked_positions)

        # Sample tokens from the distribution
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        sampled_tokens = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]),
            num_samples=1
        ).reshape(probs.shape[:-1])

        # Mask out frozen positions (replace with special token)
        sampled_tokens[~unfrozen_mask] = -1  # Placeholder for frozen tokens

        return sampled_tokens, unfrozen_mask

# Inference usage
sampler = EntropyBoundedSampler(entropy_threshold=1.5)

# In generation loop:
# - Forward pass with masked block
# - Get logits for masked positions
# - sampled_ids, unfrozen = sampler.sample_tokens(logits, masked_positions)
# - Keep unfrozen tokens, refreeze others
# - Update KV cache and repeat until sequence complete
```

#### Step 4: Inference Loop with SBD

Complete generation process using the SBD model and entropy-bounded sampler.

```python
# Complete SBD inference loop
def generate_with_sbd(
    model,
    tokenizer,
    prompt,
    max_length=256,
    entropy_threshold=1.5,
    block_size=8,
    temperature=1.0
):
    """
    Generate text using Set Block Decoding.

    Args:
        model: SBD-trained language model
        tokenizer: tokenizer with MASK_TOKEN_ID configured
        prompt: initial prompt string
        max_length: maximum generation length
        entropy_threshold: gamma parameter for EB sampler
        block_size: number of tokens to mask at each step
        temperature: sampling temperature

    Returns:
        generated_text: completed text including prompt
        speedup_stats: dict with timing and forward pass counts
    """
    import time

    device = next(model.parameters()).device
    sampler = EntropyBoundedSampler(entropy_threshold=entropy_threshold)

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    # Tracking
    forward_passes = 0
    start_time = time.time()

    with torch.no_grad():
        while seq_len < max_length:
            # Determine how many tokens to mask
            remaining = max_length - seq_len
            current_block_size = min(block_size, remaining)

            # Create masked input
            masked_input = input_ids.clone()
            mask_start = seq_len
            mask_end = min(seq_len + current_block_size, max_length)
            masked_input = torch.cat([
                masked_input,
                torch.full((1, current_block_size), MASK_TOKEN_ID, device=device)
            ], dim=1)

            # Forward pass
            output = model(masked_input)
            logits = output.logits[0, mask_start:mask_end, :]
            forward_passes += 1

            # Sample using EB sampler
            sampled_ids, unfrozen = sampler.sample_tokens(
                logits.unsqueeze(0),
                torch.arange(mask_start, mask_end),
                temperature=temperature
            )
            sampled_ids = sampled_ids.squeeze(0)
            unfrozen = unfrozen.squeeze(0)

            # Collect unfrozen tokens
            new_tokens = sampled_ids[unfrozen].unsqueeze(1)
            if new_tokens.shape[0] > 0:
                input_ids = torch.cat([input_ids, new_tokens], dim=1)
                seq_len = input_ids.shape[1]
            else:
                # Fallback: generate at least one token
                next_token = torch.argmax(logits[0])
                input_ids = torch.cat([
                    input_ids,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                seq_len = input_ids.shape[1]

            # Stop if we hit end token
            if input_ids[0, -1].item() == tokenizer.eos_token_id:
                break

    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    elapsed = time.time() - start_time

    # Estimate speedup: SBD forward passes vs. autoregressive forward passes
    autoregressive_passes = seq_len - len(tokenizer.encode(prompt))
    estimated_speedup = autoregressive_passes / forward_passes if forward_passes > 0 else 1.0

    return generated_text, {
        'forward_passes': forward_passes,
        'total_tokens_generated': seq_len - len(tokenizer.encode(prompt)),
        'elapsed_seconds': elapsed,
        'estimated_speedup': estimated_speedup
    }

# Example usage
# prompt = "The future of AI is"
# text, stats = generate_with_sbd(
#     model,
#     tokenizer,
#     prompt,
#     max_length=256,
#     entropy_threshold=1.5,
#     block_size=8
# )
# print(f"Generated: {text}")
# print(f"Speedup: {stats['estimated_speedup']:.1f}x")
```

### Practical Guidance

#### Hyperparameter Configuration

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|-----------------|
| `entropy_threshold` (γ) | 0.5-4.0 | Controls tokens per forward pass; lower = fewer tokens = slower but more accurate | Start at 1.5, tune based on latency SLA |
| `block_size` | 2-16 | Training block size; wider distribution = more robust inference | Use 4-8 for balanced training; sample variably |
| `ntp_weight` | 0.5-2.0 | Relative importance of autoregressive objective | Keep at 1.0 unless NTP performance degrades |
| `matp_weight` | 0.5-2.0 | Relative importance of masked prediction objective | Keep at 1.0; both losses are critical |
| `temperature` | 0.8-1.2 | Sampling temperature at inference | Use 1.0 for faithful generation; lower for determinism |

#### When to Use SBD

**Ideal scenarios:**

- Latency-critical inference (real-time chat, assistants)
- Cost-sensitive deployment (fewer forward passes → lower GPU time)
- Batch generation of similar lengths (entropy threshold can be tuned once)
- Models 7B-40B (tested on Llama-3.1 8B and Qwen-3 8B; scaling to larger models in progress)
- Tasks with moderate context length (AIME, Math, coding benchmarks)

**Strong results observed on:**

- Mathematical reasoning (GSM8K: 2.3-2.9x speedup, 3-4 bit accuracy loss typical)
- Coding generation (LiveCodeBench: 4.5-5.4x speedup)
- General reasoning (AIME25: 3.4-3.5x speedup)

#### When NOT to Use SBD

**Avoid SBD in these scenarios:**

- **Exact output matching required**: SBD introduces sampling variance even at low entropy thresholds. If outputs must be bit-identical to autoregressive generation, SBD adds inherent approximation.
- **Very long sequences (>8K context)**: Bidirectional attention in masked block scales O(n²). KV-cache remains efficient, but training and inference attention computation grows with sequence length.
- **Token-by-token streaming**: SBD generates multiple tokens per forward pass, preventing true token-by-token streaming to clients. Buffering required.
- **Very low-latency targets (<50ms per request)**: Even 3.5x speedup may not achieve sub-50ms on 8B models without additional quantization or distillation.
- **Instruction-following where exact order matters**: Some tasks (structured output, strict JSON) may be sensitive to parallel generation causing dependencies between tokens to be violated.
- **Real-time adjustments to generation**: Once a block is masked and sampling begins, you cannot inject new constraints mid-generation without restarting.

#### Common Pitfalls

1. **Insufficient training iterations**: SBD requires ~34k iterations to close the performance gap with NTP-only training on 3B models. Scale linearly; 8B models need proportionally more steps. Monitor NTP and MATP losses separately—if either plateaus early, increase learning rate or training duration.

2. **Ignoring loss balance**: Removing either NTP or MATP loss causes 7-12% accuracy degradation. Both are essential. Use balanced weighting during training.

3. **Entropy threshold too high**: γ > 3.0 often unfreeze most/all masked tokens, negating speedup. Start at 1.5, measure wall-clock time, then adjust. Profile on target hardware (H100, A100) since overhead is hardware-dependent.

4. **Block size mismatch at inference**: Training with block_size sampled from 2-16 but inferring with fixed size 8 introduces distribution shift. Use variable block sizes at inference too, or validate that fixed inference block size was included in training distribution.

5. **KV-cache not managed correctly**: SBD maintains KV-cache across forward passes. If you don't increment cache positions after sampling, you'll recompute attention. Ensure your generation loop updates past_key_values indices correctly.

6. **Model convergence before reaching target speedup**: If after 50k training steps performance hasn't recovered, try:
   - Reduce entropy_threshold validation metric (e.g., measure speedup on validation set, not training loss)
   - Verify bidirectional attention mask is correctly applied to masked block
   - Check that MASK_TOKEN_ID doesn't conflict with model vocabulary

### Technical Reference

**Key Papers and Concepts:**

- Next-Token Prediction (NTP): Standard transformer language modeling loss (Vaswani et al., 2017)
- Masked Token Prediction (MATP): Inspired by BERT bidirectional masking (Devlin et al., 2019)
- Entropy-Bounded Sampling: Adaptive discrete diffusion solver (related to Gumbel-max trick, Score-based Diffusion)
- Speculative Decoding (comparison point): Leviathan et al., "Fast Transformer Decoding: One Write-Head is All You Need" (ICLR 2023)
- Medusa / Eagle (comparison point): Multi-head auxiliary prediction architectures for parallel decoding

**Roofline Model Analysis:**

For H100 GPUs, forward pass overhead is ~1-1.5% per pass. A 3.5x reduction in forward passes translates to ~3.4x wall-clock speedup on inference-bound workloads (batch size 1-4, sequence length 2K+). Memory-bound phases (loading weights, I/O) benefit less; compute-bound phases (attention, FFN on large batches) benefit most.

**Reproducibility:**

- Models tested: Llama-3.1 8B, Qwen-3 8B
- Training: supervised fine-tuning with SBD objective on public math/coding datasets (GSM8K, AIME, Math500, LiveCodeBench)
- Hardware: NVIDIA H100 GPUs with standard PyTorch + Flash Attention
- No closed-form distribution shift formula; empirical tuning of γ per use case recommended

### Citation

```bibtex
@article{gat2025setblockdecoding,
  title={Set Block Decoding is a Language Model Inference Accelerator},
  author={Gat, Itai and Ben-Hamu, Heli and Havasi, Marton and Haziza, Daniel and Reizenstein, Jeremy and Synnaeve, Gabriel and Lopez-Paz, David and Karrer, Brian and Lipman, Yaron},
  journal={arXiv preprint arXiv:2509.04185},
  year={2025}
}
```

**arXiv:** https://arxiv.org/abs/2509.04185
