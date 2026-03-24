---
name: fairy2i-complex-llm-quantization
title: "Fairy2i: Extreme Quantization of LLMs via Complex-Valued Arithmetic and Phase-Aware Projection"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02901
keywords: [quantization, complex-arithmetic, LLM-compression, 1-2bit, phase-aware]
description: "Enable extreme 1-2bit quantization of pre-trained LLMs by leveraging complex-valued arithmetic. Convert real-valued linear layers to complex domain losslessly, quantize to fourth roots of unity {±1, ±i}, and apply recursive residual error quantization for near full-precision performance."
---

## Skill Summary

Fairy2i introduces extreme quantization through complex-valued arithmetic, enabling conversion of pre-trained real-valued LLMs to 1-2bit precision without retraining. The method proves that any real-valued linear layer can be expressed as a complex-valued operation, then quantizes weights to the fourth roots of unity {±1, ±i} using phase-based projection. Recursive residual error quantization further reduces approximation errors, achieving near full-precision performance on LLaMA-2 7B at effective 2-bit precision.

## When To Use

- Compressing large pre-trained LLMs for deployment on memory-constrained hardware
- Projects requiring extreme (1-2bit) quantization from existing full-precision checkpoints
- Scenarios where retraining is computationally infeasible but compression is necessary
- Research exploring complex-valued neural networks and their compression properties

## When NOT To Use

- Scenarios requiring maximum inference speed on standard hardware (complex arithmetic adds overhead)
- Projects where 1-2bit quantization causes unacceptable performance degradation for your domain
- Domains with very limited domain data making quantization-aware fine-tuning impractical
- Hardware platforms not optimized for complex-valued operations

## Core Technique

Three key technical innovations enable extreme compression:

**1. Widely-Linear Transformation**
Prove that any real-valued linear layer can be mathematically expressed as an equivalent complex-valued operation combining "a complex-linear part and a conjugate-linear part." This lossless reparameterization converts pre-trained real models into the complex domain without altering behavior before quantization.

**2. Phase-Aware Complex Quantization**
Rather than using real-valued binary or ternary sets, quantize complex weights to the "fourth roots of unity ({±1,±i})" using phase-based projection. This utilizes the full 2-bit encoding space more efficiently than real alternatives.

**3. Recursive Residual Error Quantization**
Iteratively quantize the remaining error after each stage using the same codebook. At deployment, the final weight is a sum of multiple ultra-low-bit terms, reducing approximation errors with minimal overhead.

## Implementation Notes

Start with pre-trained real-valued checkpoints. Apply widely-linear transformation to convert layers to complex domain. Implement phase-aware quantization to fourth roots of unity. Apply recursive residual error quantization for multi-term weight representation. Validate performance on downstream tasks—method typically preserves functionality within 1-3% accuracy loss.

## References

- Original paper: Fairy2i (Dec 2025)
- Complex-valued neural network literature
- Quantization-aware training for LLMs
