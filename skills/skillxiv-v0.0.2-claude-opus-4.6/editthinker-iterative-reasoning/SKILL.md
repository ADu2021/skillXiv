---
name: editthinker-iterative-reasoning
title: "EditThinker: Unlocking Iterative Reasoning for Any Image Editor"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.05965
keywords: [image editing, iterative reasoning, multimodal language models, instruction refinement, deliberative reasoning]
description: "Enable image editors to handle complex instructions through iterative critique and refinement cycles. A multimodal LLM critiques editing results, reasons about improvements, and refines instructions until satisfactory output—ideal for instruction-following challenges in visual editing."
---

## Overview

EditThinker implements a deliberative "Think-while-Edit" cycle where a multimodal language model critiques results, generates reasoning explanations, and produces refined instructions that guide the next iteration. This addresses inherent stochasticity in generation models through multi-turn reasoning rather than attempting perfection in single outputs.

## When to Use

- Complex image editing instructions require multiple refinement passes
- You need the model to explain its editing decisions and propose improvements
- Instruction-following quality is critical across various image editing tasks
- Models struggle with ambiguous or multi-step editing requirements

## When NOT to Use

- Single-turn editing where one pass is sufficient
- Real-time applications requiring instant results
- Tasks where iteration overhead is unacceptable
- Simple transformations (crop, resize, rotate)

## Core Technique

The EditThinker model jointly produces critique scores, reasoning processes, and enhanced instructions:

```python
# Iterative editing framework
class EditThinker:
    def __init__(self, vllm_model, editor):
        self.vllm = vllm_model  # Vision-language model
        self.editor = editor     # Any compatible image editor

    def refine_edit(self, image, instruction, max_iterations=3):
        current_image = image
        for iteration in range(max_iterations):
            # Generate one-step edit
            current_image = self.editor.apply(current_image, instruction)

            # Get critique and reasoning
            critique = self.vllm.critique(image, current_image)
            reasoning = self.vllm.explain_improvements(critique)

            # Generate enhanced instruction
            instruction = self.vllm.refine_instruction(
                instruction, reasoning, critique
            )

            if critique.satisfactory:
                break

        return current_image, instruction
```

Training uses reinforcement learning to align the model's reasoning with actual editing improvements, generating more targeted instruction modifications.

## Key Results

- Significant improvements across four evaluation benchmarks
- Works with multiple image editing model variants
- Instruction-following capability enhanced through multi-turn interaction
- Addresses inherent generation stochasticity

## Implementation Notes

- Compatible with any image editor backend
- RL training ensures reasoning correlates with actual improvements
- Critique scores guide iteration depth
- Preserves intermediate reasoning for interpretability

## References

- Original paper: https://arxiv.org/abs/2512.05965
- Focus: Instruction-following in image editing
- Domain: Generative modeling, multimodal reasoning
