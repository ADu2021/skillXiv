---
name: cuda-l2-kernel-optimization
title: "CUDA-L2: Surpassing cuBLAS via Reinforcement Learning for Matrix Multiplication"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02551
keywords: [cuda-optimization, reinforcement-learning, kernel-tuning, matrix-multiplication, systems-ml]
description: "Uses LLMs with RL to automatically optimize HGEMM CUDA kernels across 1,000 configurations, systematically outperforming NVIDIA's cuBLAS and cuBLASLt through continued pretraining, general RL, and specialized HGEMM RL stages."
---

## Summary

CUDA-L2 combines large language models with reinforcement learning to automatically optimize Half-precision General Matrix Multiply (HGEMM) CUDA kernels. The system employs a three-stage training approach: continued pretraining on diverse CUDA code, general kernel RL training, and specialized HGEMM RL training using execution speed as the reward signal.

## Core Technique

**Multi-Stage LLM Training:**
1. **Continued Pretraining:** Fine-tune LLM on high-quality CUDA kernel code to understand optimization patterns
2. **General RL:** Teach the model to generate working kernels and improve via speed rewards
3. **Specialized HGEMM RL:** Focus RL on matrix multiplication variants and configurations

**Kernel Generation:** The LLM generates complete CUDA kernel source code as text. Each generation is compiled, executed, and evaluated on speed.

**Reward Signal:** Execution time on representative workloads:
```
reward = baseline_speed / optimized_speed
```
Higher reward indicates better optimization.

## Implementation

**Continued pretraining data:** Collect CUDA kernel implementations:
```python
# Dataset: [kernel_code, optimization_notes, performance_hints]
pretrain_data = [(kernel1, notes1), (kernel2, notes2), ...]
```

**General RL training:**
```python
for iteration in range(num_iterations):
    # Generate kernel code
    kernel_code = llm.generate(prompt=problem_spec)
    # Compile and execute
    compiled = compile_cuda(kernel_code)
    speed = measure_execution_time(compiled, workload)
    # Compute reward
    reward = baseline_speed / speed if compiled_successfully else -1.0
    # Update LLM
    llm.update_with_rl(trajectory, reward)
```

**HGEMM specialization:** Create specialized prompts:
```
task_prompt = """
Optimize HGEMM for:
- Matrix size: 4096x4096
- Batch size: 16
- Hardware: A100 GPU
Requirements: maximize throughput
"""
```

## When to Use

- Kernel optimization when manual tuning is insufficient
- Scenarios where a large space of CUDA configurations needs exploration
- Tasks where execution speed directly impacts system performance
- Applications with diverse workload patterns requiring adaptive kernels

## When NOT to Use

- Simple kernels where hand-tuning or cuBLAS is sufficient
- Scenarios without access to LLMs or RL infrastructure
- Real-time compilation where LLM generation latency is prohibitive
- Tasks requiring guaranteed mathematical correctness (LLMs may generate incorrect kernels)

## Key References

- CUDA kernel optimization and performance tuning
- Reinforcement learning for code generation
- Language models for system optimization
- Matrix multiplication and GEMM operations
