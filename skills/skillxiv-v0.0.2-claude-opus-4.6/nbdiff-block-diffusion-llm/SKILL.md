---
name: nbdiff-block-diffusion-llm
title: "From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.06776
keywords: [diffusion language models, block generation, auto-regressive to diffusion, parallel decoding, model adaptation]
description: "Convert auto-regressive language models to efficient diffusion-based generators through gradual block size increments. NBDiff-7B inherits long-context capabilities from AR predecessors while achieving state-of-the-art parallel generation—ideal when you need efficiency without sacrificing reasoning."
---

## Overview

The paper proposes a principled pathway for converting AR models to diffusion language models through the Block-Diffusion paradigm. Rather than ad-hoc modifications, the approach gradually transitions from single-token to block-based generation while maintaining computational efficiency and preserving AR model capabilities like long-context and reasoning.

## When to Use

- Adapting existing autoregressive models to diffusion-based generation
- Scenarios requiring parallel decoding without sequential bottlenecks
- Long-context models needing efficiency improvements
- Applications where reasoning capabilities must be preserved
- Transitioning production AR models to faster inference

## When NOT to Use

- Models already achieving acceptable inference speed
- Sequential dependencies where parallelization fails
- Scenarios where diffusion generation quality lags behind AR
- Applications without engineering resources for model conversion
- Real-time systems where adaptation overhead is unacceptable

## Core Technique

Principled AR-to-DLM adaptation through gradual block size progression:

```python
# Principled adaptation from AR to Diffusion Language Models
class ARToDiffusionAdapter:
    def __init__(self, ar_model):
        """
        Initialize adapter with pre-trained autoregressive model.
        Preserve all existing capabilities during conversion.
        """
        self.ar_model = ar_model
        self.block_size = 1  # Start with single tokens

    def gradual_block_size_increment(self, training_stages=3):
        """
        Progression from block size 1 (AR) to full blocks.
        Maintains training stability and efficiency.
        """
        for stage in range(training_stages):
            # Gradually increase block size
            self.block_size = 2 ** stage  # 1, 2, 4, 8, etc.

            # Train for this block size
            self.train_block_diffusion(self.block_size)

            print(f"Stage {stage}: Trained with block size {self.block_size}")

        return self.ar_model

    def train_block_diffusion(self, block_size):
        """
        Training loop for block-based diffusion at current block size.
        """
        for batch in self.training_loader:
            # Forward: generate block of size `block_size`
            x = batch['input_ids']
            seq_len = x.shape[1]

            # Prepare targets: blocks of tokens
            blocks = self.reshape_to_blocks(x, block_size)

            # Diffusion process: add noise to blocks
            noise = torch.randn_like(blocks)
            t = torch.randint(0, 1000, (blocks.shape[0],))
            noisy_blocks = self.add_noise(blocks, noise, t)

            # Model predicts denoised blocks
            pred_blocks = self.ar_model(noisy_blocks, t)

            # Loss: prediction error
            loss = torch.nn.functional.mse_loss(pred_blocks, blocks)
            loss.backward()

            self.optimizer.step()

    def reshape_to_blocks(self, x, block_size):
        """
        Convert token sequence to block representation.
        """
        batch_size, seq_len = x.shape

        # Pad to multiple of block_size
        pad_len = (block_size - (seq_len % block_size)) % block_size
        x_padded = torch.nn.functional.pad(x, (0, pad_len))

        # Reshape: (batch, seq_len) -> (batch, num_blocks, block_size, embedding_dim)
        num_blocks = x_padded.shape[1] // block_size
        blocks = x_padded.reshape(batch_size, num_blocks, block_size)

        return blocks

    def context_causal_path_preservation(self):
        """
        Preserve causal attention in prefixes (context).
        Blocks are diffused, but context remains autoregressive.
        """
        # Attention mask: context uses causal mask (AR-style)
        # Generation blocks use masked attention
        attention_mask = self.create_hybrid_attention_mask()
        return attention_mask

    def create_hybrid_attention_mask(self):
        """
        Hybrid attention: causal for context, masked for generation.
        """
        # Context tokens: full causal attention
        context_size = self.ar_model.context_window
        context_mask = torch.tril(
            torch.ones(context_size, context_size, dtype=torch.bool)
        )

        # Generation blocks: no inter-block attention initially
        # (blocks generated independently)
        gen_block_size = self.block_size
        gen_mask = torch.eye(gen_block_size, dtype=torch.bool).unsqueeze(0)

        # Combine masks
        full_mask = torch.block_diag(context_mask, gen_mask)
        return full_mask

    def efficient_parallel_adaptation(self):
        """
        Parallel adaptation strategy using AR guidance.
        Blocks generated in parallel with AR teacher guidance.
        """
        # Initialize with AR model outputs
        ar_logits = self.ar_model(input_ids)

        # Diffusion refinement: improve upon AR predictions
        # Use AR logits as guidance for diffusion process
        initial_blocks = torch.argmax(ar_logits, dim=-1)

        # Iterative refinement
        current_blocks = initial_blocks
        for diffusion_step in range(num_diffusion_steps):
            # Denoise blocks
            denoised = self.ar_model.denoise(
                noisy_blocks=current_blocks,
                guidance=ar_logits,
                step=diffusion_step
            )
            current_blocks = denoised

        return current_blocks

    def inference_with_block_generation(self, prompt, max_tokens=512):
        """
        Inference: generate blocks in parallel rather than tokens sequentially.
        """
        input_ids = self.tokenize(prompt)

        # Determine number of blocks to generate
        num_blocks = max_tokens // self.block_size

        # Generate all blocks in parallel
        generated_blocks = []

        for block_idx in range(num_blocks):
            # Prepare input for block generation
            # Include all previously generated blocks as context
            context = torch.cat(
                [input_ids] + generated_blocks,
                dim=1
            )

            # Generate one block (can parallelize across blocks)
            block = self.generate_block(context)
            generated_blocks.append(block)

        # Concatenate all blocks
        output_ids = torch.cat(
            [input_ids] + generated_blocks,
            dim=1
        )

        return self.detokenize(output_ids)

    def preserve_long_context_capability(self):
        """
        Ensure AR model's long-context ability transfers to DLM.
        """
        # Block-based generation maintains all causal information
        # Prefixes (context) use full causal attention
        # This preserves long-context reasoning from AR model

        # Validate: test on long-context benchmarks
        performance = self.evaluate_on_long_context_tasks()
        return performance

    def preserve_reasoning_capability(self):
        """
        Ensure reasoning capabilities transfer from AR to DLM.
        """
        # Chain-of-thought reasoning preserved through:
        # 1. Causal context path (AR-style attention)
        # 2. Block-wise refinement (improves reasoning quality)
        # 3. Parallel computation (no architectural changes to reasoning)

        # Validate: test on reasoning benchmarks
        performance = self.evaluate_on_reasoning_tasks()
        return performance
```

The framework provides context-causal paths preserving causal attention in prefixes, efficient parallel adaptation using AR guidance, and gradual block size increments for smooth training transitions.

## Key Results

- NBDiff-7B inherits long-context and reasoning from AR predecessor
- State-of-the-art parallel generation performance
- Maintains computational efficiency throughout adaptation
- Preserves AR model capabilities with block-diffusion benefits
- Successful adaptation from AR to DLM paradigm

## Implementation Notes

- Gradual block size progression ensures stable training
- Context-causal paths preserve prefix attention patterns
- Parallel block generation improves inference throughput
- AR guidance helps during diffusion training
- Long-context and reasoning capabilities preserved

## References

- Original paper: https://arxiv.org/abs/2512.06776
- Focus: AR to diffusion model adaptation
- Domain: Language models, parallel decoding, model conversion
