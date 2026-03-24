---
name: dice-cuda-generation
title: "DICE: Diffusion LLMs Excel at Generating CUDA Kernels"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.11715"
keywords: [Code Generation, CUDA Kernels, Diffusion Models, Reinforcement Learning, Bi-Phase Training]
description: "Train diffusion language models to generate optimized CUDA kernels using bi-phase reinforcement learning. First phase masks and regenerates core kernel logic with provided scaffolding to prevent PyTorch shortcuts. Second phase enables end-to-end generation. Leverage diffusion's global context awareness for non-sequential code generation."
---

# DICE: Diffusion LLMs Excel at Generating CUDA Kernels

## Problem Context

Generating correct, efficient CUDA kernels requires understanding memory access patterns, thread synchronization, and hardware constraints. Autoregressive models struggle with the global coordination needed for optimal kernels. Diffusion models excel because they can revise entire sequences, refining global properties like memory coalescing and instruction-level parallelism. However, they require specialized RL training to avoid deceptive solutions (e.g., using PyTorch functions instead of custom kernels).

## Core Concept

DICE (Diffusion for Imperative Code Execution) uses bi-phase RL training:
1. **Kernel infilling**: Models learn correct CUDA logic with scaffolding (prefix/suffix provided), preventing shortcuts
2. **End-to-end generation**: Models progress to full kernel implementation with invocation logic

This leverages diffusion's iterative refinement to handle the multi-constraint optimization problem of kernel generation.

## Architecture Overview

- **CuKe dataset**: 6,303 high-performance CUDA kernels with verified 2.0× speedups
- **Kernel infilling stage**: Learn core logic with fixed prefix/suffix via RL
- **End-to-end stage**: Full kernel generation including invocation and bounds checks
- **Diffusion advantage**: Parallel token generation better suited to code's non-sequential structure
- **RL rewards**: Compilation success + runtime speedup measurements

## Implementation

### Step 1: Prepare kernel dataset and scaffolding

```python
from typing import List, Tuple, Dict
import re

class CUDaKernelDataset:
    """Curated CUDA kernel dataset with verified speedups."""

    def __init__(self, dataset_path: str = None):
        self.kernels = []
        self.load_dataset(dataset_path)

    def load_dataset(self, path: str):
        """Load CuKe dataset."""
        # Format: {kernel_code, problem_statement, speedup, test_cases}
        # Load verified kernels (2.0x+ speedup)
        pass

    def create_kernel_scaffold(
        self,
        kernel_code: str,
        scaffold_type: str = 'middle'
    ) -> Tuple[str, str, str]:
        """
        Create prefix-suffix scaffolding for kernel infilling.

        Args:
            kernel_code: Full kernel code
            scaffold_type: 'middle' (mask core logic)

        Returns:
            (prefix, target_middle, suffix) for infilling
        """
        lines = kernel_code.split('\n')

        # Identify kernel function boundaries
        kernel_start = None
        kernel_end = None

        for i, line in enumerate(lines):
            if '__global__' in line and 'kernel' in line.lower():
                kernel_start = i
            if kernel_start is not None and line.strip() == '}':
                kernel_end = i
                break

        if kernel_start is None or kernel_end is None:
            return None, kernel_code, None

        # Scaffold: keep signature and closing brace
        prefix_lines = lines[:kernel_start + 2]  # Include __global__ def and opening brace
        middle_lines = lines[kernel_start + 2:kernel_end]
        suffix_lines = lines[kernel_end:]  # Closing brace

        prefix = '\n'.join(prefix_lines)
        middle = '\n'.join(middle_lines)
        suffix = '\n'.join(suffix_lines)

        return prefix, middle, suffix

    def get_infill_examples(self, num_examples: int = 100) -> List[Dict]:
        """Get kernel infilling training examples."""
        examples = []

        for kernel in self.kernels[:num_examples]:
            prefix, middle, suffix = self.create_kernel_scaffold(kernel['code'])

            if prefix is None:
                continue

            examples.append({
                'prefix': prefix,
                'target': middle,
                'suffix': suffix,
                'problem': kernel['problem'],
                'expected_speedup': kernel['speedup']
            })

        return examples
```

### Step 2: Implement kernel compilation and verification

```python
import subprocess
import tempfile
import time

class KernelVerifier:
    """Compile and execute kernels to verify correctness and performance."""

    def __init__(self, cuda_toolkit_path: str = '/usr/local/cuda'):
        self.cuda_path = cuda_toolkit_path

    def compile_kernel(self, kernel_code: str) -> Tuple[bool, str]:
        """
        Compile CUDA kernel code.

        Args:
            kernel_code: C++/CUDA source code

        Returns:
            (success, error_message)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            temp_file = f.name

        try:
            # Compile with nvcc
            result = subprocess.run(
                [f'{self.cuda_path}/bin/nvcc', '-c', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            success = result.returncode == 0
            error_msg = result.stderr if not success else ""

            return success, error_msg

        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, str(e)

    def measure_speedup(
        self,
        generated_kernel: str,
        baseline_kernel: str,
        test_inputs: List = None
    ) -> float:
        """
        Measure speedup of generated vs baseline kernel.

        Returns:
            speedup_ratio: Generated runtime / baseline runtime
        """
        # Simplified: would run actual CUDA code
        # In practice, use nvprof or custom timing harness

        # Quick heuristic: count operations
        gen_ops = len(re.findall(r'[+\-*/]|\.x|\.y|\.z', generated_kernel))
        base_ops = len(re.findall(r'[+\-*/]|\.x|\.y|\.z', baseline_kernel))

        if base_ops == 0:
            return 1.0

        estimated_speedup = 1.0 + (base_ops - gen_ops) / base_ops

        return max(0.5, min(estimated_speedup, 3.0))

    def compute_reward(
        self,
        generated_kernel: str,
        baseline_kernel: str,
        expected_speedup: float = 2.0,
        problem_context: str = None
    ) -> float:
        """
        Compute RL reward for kernel generation.

        Rewards:
        - Compilation success: +0.5
        - Speedup >= 1.5x: +0.3
        - Matches expected speedup: +0.2
        """
        reward = 0.0

        # Compilation check
        success, _ = self.compile_kernel(generated_kernel)
        if success:
            reward += 0.5
        else:
            return 0.0  # Failed compilation = no reward

        # Performance check
        speedup = self.measure_speedup(generated_kernel, baseline_kernel)

        if speedup >= 1.5:
            reward += 0.3

        if abs(speedup - expected_speedup) < 0.5:
            reward += 0.2

        return reward
```

### Step 3: Kernel infilling with RL

```python
class KernelInfillingRL:
    """GRPO-based training for kernel infilling."""

    def __init__(
        self,
        model,
        optimizer,
        verifier: KernelVerifier,
        group_size: int = 4
    ):
        self.model = model
        self.optimizer = optimizer
        self.verifier = verifier
        self.group_size = group_size

    def generate_kernel_infill(
        self,
        prefix: str,
        suffix: str,
        max_length: int = 200
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate kernel middle section given prefix/suffix.

        Args:
            prefix: Kernel signature and opening brace
            suffix: Closing brace
            max_length: Max tokens for middle

        Returns:
            (generated_middle_code, log_probs)
        """
        # Prompt structure: prefix + [MASK] * max_length + suffix
        prompt = f"{prefix}\n" + "[MASK]\n" + f"{suffix}"

        middle, log_probs = self.model.generate_with_logprobs(
            prompt,
            max_tokens=max_length,
            mask_positions=[1]  # Regenerate masked region
        )

        return middle, log_probs

    def training_step(
        self,
        examples: List[Dict],
        baseline_kernel: str = None
    ) -> float:
        """
        Single training step on infilling examples.

        Args:
            examples: Kernel infilling examples
            baseline_kernel: Reference kernel for speedup comparison

        Returns:
            loss: Training loss
        """
        total_loss = 0.0

        for batch_idx in range(0, len(examples), self.group_size):
            batch = examples[batch_idx:batch_idx + self.group_size]
            batch_size = len(batch)

            # Generate infills
            log_probs_list = []
            rewards = []

            for example in batch:
                prefix = example['prefix']
                suffix = example['suffix']

                # Generate
                middle, log_probs = self.generate_kernel_infill(prefix, suffix)
                log_probs_list.append(log_probs)

                # Reconstruct full kernel
                full_kernel = f"{prefix}\n{middle}\n{suffix}"

                # Compute reward
                reward = self.verifier.compute_reward(
                    full_kernel,
                    baseline_kernel or example.get('baseline'),
                    example['expected_speedup']
                )

                rewards.append(reward)

            log_probs = torch.stack(log_probs_list)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            # GRPO loss
            log_prob_ratio = log_probs - log_probs.detach()
            ratio = torch.exp(log_prob_ratio)

            # Group relative advantage
            group_mean_reward = rewards.mean()
            relative_rewards = rewards - group_mean_reward

            clipped_ratio = torch.clamp(ratio, 0.5, 2.0)
            loss = -torch.min(
                log_prob_ratio * relative_rewards.unsqueeze(-1),
                torch.log(clipped_ratio) * relative_rewards.unsqueeze(-1)
            ).mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(examples) // self.group_size)
        return avg_loss
```

### Step 4: End-to-end kernel generation

```python
class EndToEndKernelGeneration:
    """Generate complete kernels including invocation code."""

    def __init__(self, model, verifier: KernelVerifier):
        self.model = model
        self.verifier = verifier

    def generate_full_kernel(
        self,
        problem_description: str,
        max_length: int = 500,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate complete CUDA kernel from problem statement.

        Args:
            problem_description: What the kernel should compute
            max_length: Maximum code length
            num_samples: Number of samples to generate

        Returns:
            kernels: List of generated kernel codes
        """
        prompt = f"""
Generate an optimized CUDA kernel for the following problem:

{problem_description}

Provide:
1. __global__ kernel function definition
2. Host code to launch the kernel
3. Memory management

CUDA kernel:
"""

        samples = []
        for _ in range(num_samples):
            kernel, _ = self.model.generate_with_logprobs(
                prompt, max_tokens=max_length, temperature=0.7
            )
            samples.append(kernel)

        return samples

    def evaluate_kernels(
        self,
        kernels: List[str],
        problem: str,
        baseline: str = None
    ) -> List[Dict]:
        """
        Evaluate multiple generated kernels.

        Returns:
            metrics: Compilation success, speedup, etc.
        """
        results = []

        for kernel in kernels:
            success, error = self.verifier.compile_kernel(kernel)
            speedup = self.verifier.measure_speedup(kernel, baseline) if success else 0.0
            reward = self.verifier.compute_reward(kernel, baseline)

            results.append({
                'compiles': success,
                'error': error,
                'speedup': speedup,
                'reward': reward
            })

        return results
```

### Step 5: Bi-phase training

```python
def train_dice_bi_phase(
    model,
    dataset: CUDaKernelDataset,
    optimizer,
    verifier: KernelVerifier,
    num_epochs: int = 10,
    phase_transition_epoch: int = 5,
    device: str = 'cuda'
):
    """
    Train DICE in two phases: infilling then end-to-end.

    Args:
        phase_transition_epoch: Switch to end-to-end training at this epoch
    """
    infill_trainer = KernelInfillingRL(model, optimizer, verifier)
    e2e_trainer = EndToEndKernelGeneration(model, verifier)

    for epoch in range(num_epochs):
        if epoch < phase_transition_epoch:
            # Phase 1: Kernel infilling
            print(f"Epoch {epoch + 1}: Infilling phase")

            infill_examples = dataset.get_infill_examples(num_examples=50)
            avg_loss = infill_trainer.training_step(infill_examples)

            print(f"  Loss: {avg_loss:.4f}")

        else:
            # Phase 2: End-to-end generation
            print(f"Epoch {epoch + 1}: End-to-end phase")

            problems = dataset.get_problems()[:10]

            for problem in problems:
                kernels = e2e_trainer.generate_full_kernel(
                    problem['statement'],
                    num_samples=4
                )

                results = e2e_trainer.evaluate_kernels(
                    kernels, problem, baseline=problem['baseline']
                )

                avg_reward = sum(r['reward'] for r in results) / len(results)
                print(f"  Problem: avg_reward={avg_reward:.4f}")

    return model
```

## Practical Guidance

**When to use**: Generating CUDA kernels, optimized system code, or other complex imperative programs where global structure matters more than sequential coherence

**Hyperparameters**:
- **infilling_phase_epochs**: 3-5 (build fundamental skills)
- **e2e_phase_epochs**: 5-10 (refinement)
- **max_kernel_length**: 300-500 tokens
- **num_generation_samples**: 4-8 (ensembling before compilation)
- **group_size**: 4 (GRPO grouping)

**Key advantages**:
- Diffusion naturally handles non-sequential code generation
- Bi-phase training prevents deceptive shortcuts
- Verified rewards via compilation and performance testing
- Handles global properties (memory coalescing) better than autoregressive

**Common pitfalls**:
- Infilling phase too short → models not learning kernel patterns
- Compilation reward not strict enough → deceptive solutions
- Not measuring actual speedup → accepting slow kernels
- Phase transition too abrupt → catastrophic forgetting

**Scaling**: Dataset curation and verification are bottlenecks; consider synthetic kernel generation.

## Reference

Paper: https://arxiv.org/abs/2602.11715
Related work: Code generation, diffusion models, system optimization, program synthesis
Benchmarks: KernelBench, custom CUDA kernel correctness and performance
Dataset: CuKe (6,303 verified kernels)
