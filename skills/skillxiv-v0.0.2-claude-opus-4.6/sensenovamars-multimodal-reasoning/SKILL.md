---
name: sensenovamars-multimodal-reasoning
title: "SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.24330"
keywords: [Multimodal Agents, Vision-Language Models, Agentic Reasoning, Tool Use, RL Training]
description: "Build vision-language agents that seamlessly integrate visual reasoning with dynamic tool manipulation (search, cropping) through reinforcement learning, achieving state-of-the-art performance on fine-grained visual understanding tasks—surpassing proprietary models like Gemini and GPT."
---

## Overview

SenseNova-MARS is a multimodal agentic framework enabling Vision-Language Models (VLMs) to perform sophisticated visual reasoning tasks through interleaved tool manipulation and continuous reasoning.

**Core Innovation:** While VLMs excel at text-oriented reasoning and isolated tool calls, they struggle with coordinated tool sequences on knowledge-intensive visual tasks. SenseNova-MARS enables human-like proficiency in dynamically invoking multiple tools (search, image cropping, text search) to solve complex visual understanding problems.

## Architecture

### Tool Integration

Three complementary tools enable sophisticated visual problem-solving:

**Image Search Tool**
- Retrieves semantically related images from knowledge bases
- Supports fine-grained visual understanding through reference images
- Enables comparative analysis across multiple images

**Text Search Tool**
- Queries knowledge bases for text-based information
- Supports factual grounding of visual analysis
- Bridges visual content with textual knowledge

**Image Crop Tool**
- Extracts regions of interest from high-resolution images
- Enables focus on fine-grained visual details
- Supports iterative refinement of understanding

### Agentic Reasoning Loop

The agent operates through repeated cycles of:
1. **Perceive** - Analyze current visual content and available information
2. **Reason** - Determine which tools would advance understanding
3. **Act** - Invoke appropriate tools (search, crop, etc.)
4. **Integrate** - Synthesize results into improved understanding

## RL Training: Batch-Normalized Group Sequence Policy Optimization

To enable reliable tool invocation and reasoning, SenseNova-MARS introduces **BN-GSPO** algorithm:

**Key Components:**
- **Group Sequence Policy Optimization** - Trains on sequences of decisions, not isolated actions
- **Batch Normalization** - Stabilizes training dynamics in high-variance visual environments
- **Reinforcement Signal** - Rewards both intermediate tool selections and final correctness

**Training Stability Features:**
- Handles variable-length action sequences
- Robust to initialization variance in visual encoders
- Supports training with incomplete or noisy rewards

## Benchmark Performance

**HR-MMSearch (proprietary high-resolution benchmark):**
- SenseNova-MARS-32B: 54.4%
- Outperforms Gemini-3-Pro and GPT-5.2
- First benchmark for search-oriented visual understanding

**MMSearch Benchmark:**
- SenseNova-MARS-32B: 74.3%
- State-of-the-art among open-source models
- Validates framework's effectiveness on knowledge-intensive tasks

## Implementation Pattern

**Tool Invocation in Agent Loop:**

```python
def solve_visual_task(image: PIL.Image, question: str) -> str:
    """Solve visual understanding task through tool orchestration."""

    # Initial analysis
    context = vlm_analyze(image, question)

    # Agentic loop: iteratively invoke tools
    reasoning_steps = []
    for _ in range(max_steps):
        # Determine next action
        action = vlm_decide_next_action(context, reasoning_steps)

        if action == "image_search":
            ref_images = search_visual_db(context.search_query)
            context.add_reference_images(ref_images)

        elif action == "crop_region":
            roi = vlm_identify_roi(image)
            cropped = image.crop(roi)
            detail = vlm_analyze_detail(cropped)
            context.add_detail(detail)

        elif action == "text_search":
            facts = search_knowledge_base(context.search_query)
            context.add_knowledge(facts)

        elif action == "answer":
            return vlm_synthesize_answer(context, image)

        reasoning_steps.append(action)

    return vlm_synthesize_answer(context, image)
```

## Advantages Over Single-Tool Approaches

**vs. Image-Only Analysis:**
- Text search grounds visual understanding in factual knowledge
- Comparative image search provides context for classification
- Handles knowledge-intensive visual tasks (e.g., identifying rare objects)

**vs. Text-Only Approaches:**
- Preserves fine-grained visual details through image cropping
- Captures spatial relationships between entities
- Enables visual verification of text-based claims

**vs. Isolated Tool Invocation:**
- Dynamically determines tool sequences based on task progress
- Learns coordination patterns through RL training
- Handles emergent task structures not foreseen during design

## When to Use

**Use SenseNova-MARS when:**
- Solving knowledge-intensive visual tasks (rare object identification, specialized domains)
- Requiring fine-grained visual understanding on high-resolution images
- Needing to ground visual analysis in textual knowledge bases
- Building agents that perform comparative visual analysis
- Requiring robust tool coordination across multiple modalities

**When NOT to use:**
- Simple image classification or captioning (direct VLM sufficient)
- Real-time applications with strict latency constraints
- Scenarios where tool invocation overhead outweighs benefit
- Tasks with limited textual grounding information available

## Related Work

- **VLM Tool Use:** Gorilla, ToolFormer (static tool selection vs. dynamic agent reasoning)
- **Multimodal Reasoning:** Unified vision-language architectures (SenseNova-MARS emphasizes tool orchestration)
- **RL for VLMs:** Limited prior work; SenseNova-MARS demonstrates BN-GSPO's stability

## Benchmark Contributions

**HR-MMSearch Dataset:**
- First search-oriented multimodal benchmark
- High-resolution images with knowledge-intensive questions
- Enables evaluation of tool coordination capabilities
- Publicly available for reproducibility

## Code Availability

All code, models, and datasets will be released for community research.

## References

- SenseNova-MARS achieves 74.3% on MMSearch, 54.4% on HR-MMSearch
- Outperforms proprietary models Gemini-3-Pro and GPT-5.2
- BN-GSPO algorithm ensures training stability in visual domains
- Demonstrates practical multimodal agent deployment
