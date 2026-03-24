---
name: capmagine-visual-reasoning
title: "Imagination Helps Visual Reasoning, But Not Yet in Latent Space"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.22766"
keywords: [Visual Reasoning, Multimodal LLM, Chain-of-Thought, Imagination, Latent Space]
description: "CapImagine teaches models to explicitly imagine through text rather than latent reasoning, significantly improving visual reasoning performance."
---

# Technique: Explicit Imagination for Visual Reasoning

Multimodal language models struggle with visual reasoning tasks because their hidden (latent) reasoning produces negligible impact on final answers. While explicit chain-of-thought helps textual reasoning, simply moving it to latent space doesn't work for vision tasks. The reason: visual perception requires explicit symbolic reasoning that models can't perform implicitly in token embeddings.

CapImagine solves this by encouraging models to explicitly imagine the visual scene using text before answering. Rather than relying on hidden representations to bridge vision and reasoning, the model explicitly describes what it sees, imagines intermediate steps, and uses these concrete representations to reason. This dramatically improves visual reasoning accuracy.

## Core Concept

The core insight comes from causal mediation analysis: when latent tokens are perturbed, they produce minimal impact on final answers (latent-answer disconnect). This reveals a fundamental limitation of implicit reasoning for vision tasks. Explicit imagination—asking the model to describe what it sees and imagines—provides the concrete symbolic grounding needed for robust visual reasoning.

This connects to cognitive science: human visual reasoning isn't latent; it's explicit—we mentally simulate and verbalize what we observe before drawing conclusions.

## Architecture Overview

- **Visual Input**: Standard image encoding (existing vision encoders work fine)
- **Imagination Module**: Prompt design that cues explicit visual description
- **Explicit Reasoning**: Chain-of-thought over imagined descriptions
- **Answer Generation**: Final output based on explicit reasoning, not latent tokens
- **No Special Architecture**: Works with any vision-language model; training-free

## Implementation Steps

CapImagine is a prompting technique—no model modification needed. Here's how to implement it:

Design prompt templates that explicitly cue imagination of the visual scene:

```python
# Standard visual reasoning prompt (baseline)
def standard_visual_qa(image_caption, question):
    prompt = f"""
    Image caption: {image_caption}
    Question: {question}
    Answer: """
    return prompt

# CapImagine prompt: explicit imagination instruction
def capmagine_visual_qa(image_caption, question):
    prompt = f"""
    Image caption: {image_caption}

    Before answering, imagine the scene described above. What do you see?
    Describe the key objects, their positions, colors, and relationships:

    [Imagination]

    Now, using your imagination of the scene, answer the question:
    Question: {question}
    Answer: """
    return prompt
```

Implement an inference wrapper that collects explicit imagination outputs:

```python
def generate_with_imagination(model, tokenizer, image, question):
    """
    Generate visual reasoning with explicit imagination.
    Returns both imagination and final answer.
    """
    # Get image caption or encoding (using existing vision encoder)
    image_caption = get_image_caption(model, tokenizer, image)

    # Create imagination prompt
    imagination_prompt = f"""
    Image caption: {image_caption}

    Describe the visual scene in detail. What objects are present?
    What are their spatial relationships and colors?

    Description: """

    # Generate imagination
    imagination_tokens = model.generate(
        imagination_prompt,
        max_length=150,
        temperature=0.7
    )
    imagination_text = tokenizer.decode(imagination_tokens)

    # Create reasoning prompt incorporating imagination
    reasoning_prompt = f"""
    Image caption: {image_caption}

    Scene description: {imagination_text}

    Now answer the question using the scene description above:
    Question: {question}
    Answer: """

    # Generate final answer
    answer_tokens = model.generate(
        reasoning_prompt,
        max_length=100,
        temperature=0.7
    )
    answer_text = tokenizer.decode(answer_tokens)

    return {
        'imagination': imagination_text,
        'answer': answer_text,
        'full_prompt': reasoning_prompt
    }
```

Implement training data augmentation with explicit reasoning annotations:

```python
def create_vision_qa_with_imagination(
    dataset,
    model,
    tokenizer,
    annotation_batch_size=32
):
    """
    Augment visual QA dataset with explicit imaginations.
    Use these (image, imagination, question, answer) tuples for supervised fine-tuning.
    """
    augmented_data = []

    for image, question, ground_truth_answer in dataset:
        # Generate imagination
        imagination_output = generate_with_imagination(
            model, tokenizer, image, question
        )

        # Create training example with explicit imagination
        example = {
            'image': image,
            'question': question,
            'imagination': imagination_output['imagination'],
            'answer': ground_truth_answer,
            'full_reasoning': imagination_output['full_prompt']
        }
        augmented_data.append(example)

    return augmented_data
```

## Practical Guidance

**When to Use:**
- Vision-language models on visual reasoning tasks
- VQA systems requiring spatial or relational reasoning
- Scene understanding where objects' positions matter
- When model struggles with complex visual concepts

**When NOT to Use:**
- Simple image classification or tagging (overkill)
- Real-time systems with strict latency budgets (imagination generation adds time)
- Tasks where implicit reasoning suffices (simple attribute queries)

**Prompting Strategies:**
- **Minimal imagination**: Just ask "Describe what you see"
- **Structured imagination**: Specify objects, colors, positions, relationships
- **Comparative imagination**: For relational tasks, ask model to compare objects explicitly
- **Procedural imagination**: For sequential reasoning, ask model to imagine step-by-step

**Hyperparameters:**
- `imagination_max_length`: 100–200 tokens typical
- `temperature`: 0.7–0.9 for creative imagination
- `answer_max_length`: 50–150 depending on task

**Fine-tuning for Better Performance:**
If you have QA data with ground truth answers, create (image, imagination_prompt → imagination, question → answer) pairs and fine-tune the model. This teaches it to generate high-quality imaginations that support reasoning.

**Results:**
- Significant improvements on vision-centric benchmarks
- Outperforms baselines using implicit latent reasoning
- Particularly effective on spatial reasoning and object relationship tasks
- Works across different vision-language architectures

---

**Reference:** [Imagination Helps Visual Reasoning, But Not Yet in Latent Space](https://arxiv.org/abs/2602.22766)
