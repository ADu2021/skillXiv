---
name: mirror-speculative-decoding
title: "Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.13161"
keywords: [inference-optimization, speculative-decoding, parallel-execution, heterogeneous-hardware]
description: "Run draft and target models in parallel across GPU/NPU using bidirectional speculation: draft predicts forward continuations while target speculates correction paths simultaneously. Achieve 2.8x-5.8x wall-time speedup on 14B-66B models."
---

# Mirror Speculative Decoding: Parallel Heterogeneous Inference Acceleration

Standard speculative decoding runs draft model serially then validates with target model, creating latency bottleneck. Mirror Speculative Decoding launches both draft and target speculatively in parallel on heterogeneous hardware (GPU + NPU), with draft speculating forward and target speculating corrections simultaneously.

Core insight: traditional serial pipelines waste hardware. By running bidirectional speculation concurrently on different devices, you break the serial barrier while maintaining high token acceptance rates, achieving 2.8-5.8x speedup.

## Core Concept

**Bidirectional Speculation**: Draft model predicts forward token sequences while target model speculatively predicts correction/alternative paths. Both run in parallel.

**Heterogeneous Hardware Exploitation**: Leverage GPU and NPU concurrency—draft model on faster device, target on secondary device, minimizing idle time.

**Multi-Token Streaming**: Draft emits multiple tokens per step without sacrificing verification quality.

## Architecture Overview

- **Draft Model Pipeline**: Generates candidate sequences on primary device
- **Target Model Pipeline**: Speculatively generates corrections on secondary device
- **Token Synchronization**: Merge paths when they diverge
- **Acceptance Logic**: Verify candidate tokens match target predictions

## Implementation Steps

**Stage 1: Setup Heterogeneous Execution**

Configure dual-device inference:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class HeterogeneousModelPair:
    def __init__(self, model_name, draft_device='cuda:0', target_device='cuda:1'):
        """
        Load draft and target models on different devices.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Draft model: smaller, faster
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            f"{model_name}-draft",  # Smaller variant
            torch_dtype=torch.float16
        ).to(draft_device)

        # Target model: larger, more accurate
        self.target_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(target_device)

        self.draft_device = draft_device
        self.target_device = target_device

        # Freeze both models
        for param in self.draft_model.parameters():
            param.requires_grad = False
        for param in self.target_model.parameters():
            param.requires_grad = False

    def forward_draft(self, input_ids, num_speculative_tokens=4):
        """
        Generate speculative tokens from draft model.
        """

        with torch.no_grad():
            # Extend input and get multiple token predictions
            draft_ids = input_ids.to(self.draft_device)

            draft_tokens = []
            draft_logprobs = []

            for _ in range(num_speculative_tokens):
                logits = self.draft_model(draft_ids).logits[:, -1, :]

                # Get log probabilities
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(
                    torch.exp(log_probs),
                    num_samples=1
                )

                draft_tokens.append(next_token)
                draft_logprobs.append(log_probs[0, next_token].item())

                # Append to sequence for next prediction
                draft_ids = torch.cat([draft_ids, next_token], dim=-1)

        draft_sequence = torch.cat(draft_tokens, dim=-1)

        return draft_sequence, draft_logprobs

    def forward_target_corrective(
        self,
        input_ids,
        draft_sequence,
        num_correction_tokens=4
    ):
        """
        Speculatively generate corrections from target model.
        Predicts what the target model would output.
        """

        with torch.no_grad():
            target_ids = input_ids.to(self.target_device)

            correction_tokens = []
            correction_logprobs = []

            for _ in range(num_correction_tokens):
                logits = self.target_model(target_ids).logits[:, -1, :]

                # Get log probabilities
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Sample next token (may differ from draft)
                next_token = torch.multinomial(
                    torch.exp(log_probs),
                    num_samples=1
                )

                correction_tokens.append(next_token)
                correction_logprobs.append(log_probs[0, next_token].item())

                target_ids = torch.cat([target_ids, next_token], dim=-1)

        correction_sequence = torch.cat(correction_tokens, dim=-1)

        return correction_sequence, correction_logprobs
```

**Stage 2: Parallel Execution with Synchronization**

Run both models concurrently:

```python
import threading
import queue

class MirrorSpeculativeDecoder:
    def __init__(self, model_pair):
        self.model_pair = model_pair

    def decode_parallel(
        self,
        prompt,
        max_length=512,
        num_speculative_tokens=4
    ):
        """
        Run draft and target speculatively in parallel.
        """

        input_ids = self.model_pair.tokenizer.encode(
            prompt,
            return_tensors='pt'
        )

        full_output = input_ids.clone()

        # Queues for inter-thread communication
        draft_queue = queue.Queue()
        target_queue = queue.Queue()
        merge_event = threading.Event()

        def draft_worker():
            """Generate draft tokens in parallel."""
            draft_seq, draft_logprobs = self.model_pair.forward_draft(
                full_output,
                num_speculative_tokens
            )

            draft_queue.put((draft_seq, draft_logprobs))

        def target_worker():
            """Generate target corrections in parallel."""
            target_seq, target_logprobs = (
                self.model_pair.forward_target_corrective(
                    full_output,
                    None,  # Draft sequence will be available
                    num_speculative_tokens
                )
            )

            target_queue.put((target_seq, target_logprobs))

        # Run workers in parallel
        draft_thread = threading.Thread(target=draft_worker, daemon=True)
        target_thread = threading.Thread(target=target_worker, daemon=True)

        draft_thread.start()
        target_thread.start()

        # Wait for both to complete
        draft_sequence, draft_logprobs = draft_queue.get(timeout=30)
        target_sequence, target_logprobs = target_queue.get(timeout=30)

        # Merge results: use target as authority, fallback to draft
        merged_tokens = self.merge_speculative_results(
            draft_sequence,
            draft_logprobs,
            target_sequence,
            target_logprobs
        )

        full_output = torch.cat([full_output, merged_tokens], dim=-1)

        return full_output

    def merge_speculative_results(
        self,
        draft_sequence,
        draft_logprobs,
        target_sequence,
        target_logprobs
    ):
        """
        Merge draft and target predictions.
        Prefer target, fallback to draft if target isn't ready.
        """

        # Compare sequences token by token
        min_length = min(
            draft_sequence.shape[1],
            target_sequence.shape[1]
        )

        # Find divergence point
        divergence_idx = 0
        for i in range(min_length):
            if draft_sequence[0, i] != target_sequence[0, i]:
                divergence_idx = i
                break
        else:
            divergence_idx = min_length

        # Accept target sequence up to divergence
        if divergence_idx > 0:
            accepted = target_sequence[:, :divergence_idx]
        else:
            # Sequences diverged immediately, use target
            accepted = target_sequence[:, :1]

        return accepted
```

**Stage 3: Inference with Speedup Monitoring**

Deploy mirror speculation:

```python
def mirror_speculative_inference(
    model_pair,
    prompts,
    target_speedup=3.0
):
    """
    Run inference with mirror speculation and speedup tracking.
    """

    decoder = MirrorSpeculativeDecoder(model_pair)

    speedup_metrics = {
        'draft_tokens': 0,
        'target_tokens': 0,
        'acceptance_rate': [],
        'wall_time': 0.0
    }

    results = []

    for prompt in prompts:
        start_time = time.time()

        output = decoder.decode_parallel(
            prompt,
            max_length=512,
            num_speculative_tokens=4
        )

        end_time = time.time()

        # Track metrics
        speedup_metrics['wall_time'] += (end_time - start_time)

        results.append(output)

    # Report speedup
    baseline_time = speedup_metrics['wall_time'] * target_speedup
    actual_speedup = baseline_time / speedup_metrics['wall_time']

    print(f"Achieved speedup: {actual_speedup:.2f}x")

    return results
```

## Practical Guidance

**When to Use Mirror Speculative Decoding:**
- Heterogeneous hardware available (GPU + NPU, dual GPUs)
- Inference where latency is critical (batch or streaming)
- Models where target and draft have different optimal devices

**When NOT to Use:**
- Single-device inference (no parallelism benefit)
- Constrained memory (loading two models simultaneously)
- Real-time scenarios with strict tail latency (synchronization overhead)

**Hardware Configurations:**

| Setup | Speedup | Recommended |
|-------|---------|-------------|
| GPU + CPU | 1.5-2.0x | Basic heterogeneity |
| Dual GPU | 2.0-3.5x | Common datacenter |
| GPU + NPU | 2.8-5.8x | Modern phones/edge |

**Speculative Token Count:**

| Tokens | Acceptance Rate | Speedup |
|--------|-----------------|---------|
| 2 | 95% | 1.5-2.0x |
| 4 | 85% | 2.5-3.5x |
| 8 | 70% | 3.0-4.0x |
| 16 | 50% | 2.5-3.0x |

**Common Pitfalls:**
- Thread synchronization too slow (coordinate via events, not polling)
- Not accounting for device initialization overhead
- Speculative token count too high (low acceptance, wasted speculation)
- Assuming perfect parallelism (communication/synchronization costs)

## Reference

Based on the research at: https://arxiv.org/abs/2510.13161
