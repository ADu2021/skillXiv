---
name: control-r-controllable-scaling
title: "Control-R: Towards Controllable Test-Time Scaling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.00189"
keywords: [reasoning-control, test-time-scaling, long-chain-of-thought, reasoning-depth, inference-optimization]
description: "Enable dynamic control over reasoning depth during inference using Reasoning Control Fields that guide long chain-of-thought generation based on execution and quality parameters."
---

# Control-R: Towards Controllable Test-Time Scaling

## Core Concept

Large Reasoning Models (LRMs) face a dilemma: fixed-depth reasoning either wastes compute on simple problems or underperforms on complex ones. Control-R introduces Reasoning Control Fields (RCF)—structured signals that guide test-time reasoning depth based on task-specific requirements. Rather than uniform reasoning, the framework treats reasoning as tree search and enables models to dynamically adjust effort through 11 controllable parameters (search depth/breadth, error correction, strategy switching, and quality metrics). This achieves state-of-the-art performance at 32B scale with interpretable, adjustable reasoning traces.

Reaches 70.0% on AIME2024 and 93.2% on MATH500 while enabling real-time reasoning depth adjustment during inference.

## Architecture Overview

- **11 Reasoning Control Fields**: Split into execution control (5) and process quality (6) dimensions
- **Execution Control**: Search depth, breadth, error detection, correction strategies, approach switching
- **Process Quality**: Correctness, efficiency, completeness, coherence, knowledge accuracy, clarity
- **Conditional Generation**: Model learns P(R|q,C) instead of P(R|q), conditioning on control specs
- **Search Tree Perspective**: Conceptualizes reasoning as exploring decision trees with parameter-guided path selection
- **Control-R-4K Dataset**: 4,000+ problems annotated with detailed reasoning and control field scores

## Implementation

1. **Define Reasoning Control Fields**: Structured textual format for control signals

```python
class ReasoningControlFields:
    """11 control fields guiding reasoning behavior"""

    # Execution Control (5 fields, range 0-9)
    SEARCH_DEPTH = 'search_depth'           # How deep to explore (1-9)
    SEARCH_BREADTH = 'search_breadth'       # Paths to consider (1-9)
    ERROR_DETECTION = 'error_detection'     # Check for mistakes (0-9)
    ERROR_CORRECTION = 'error_correction'   # Fix errors found (0-9)
    STRATEGY_SWITCHING = 'strategy_switching'  # Try different approaches (0-9)

    # Process Quality (6 fields, range 0-9)
    CORRECTNESS = 'correctness'             # Solution accuracy target
    EFFICIENCY = 'efficiency'               # Computational efficiency
    COMPLETENESS = 'completeness'           # Cover all aspects
    COHERENCE = 'coherence'                 # Logical flow quality
    KNOWLEDGE_ACCURACY = 'knowledge_accuracy'  # Factual precision
    CLARITY = 'clarity'                     # Explanation quality

def format_control_prompt(question, control_values):
    """
    Convert control field dictionary to textual prompt format.
    Appended to query to condition reasoning.
    """
    control_prompt = f"{question}\n\n<control>\n"

    for field, value in control_values.items():
        control_prompt += f"{field}: {value};\n"

    control_prompt += "<control/>\n"
    return control_prompt
```

2. **Training Data Annotation**: Create Control-R-4K dataset with control scores

```python
def annotate_reasoning_with_controls(question, reasoning_trace, solution):
    """
    Evaluate reasoning trace across 11 control dimensions.
    ChatGPT-4o assigns 0-9 scores per dimension.
    """
    annotations = {
        'question': question,
        'reasoning': reasoning_trace,
        'solution': solution,
        'control_scores': {}
    }

    # Simulate annotation (in practice: use annotation service)
    gpt4_prompt = f"""
    Evaluate this reasoning trace on 11 dimensions (0-9 scale):

    Question: {question}
    Reasoning: {reasoning_trace}
    Solution: {solution}

    Rate execution control:
    - search_depth: How deeply was the solution space explored?
    - search_breadth: How many approaches were considered?
    - error_detection: Were potential mistakes identified?
    - error_correction: Were errors corrected when found?
    - strategy_switching: Were alternative strategies tried?

    Rate process quality:
    - correctness: Final answer correctness
    - efficiency: Solution efficiency
    - completeness: All aspects covered
    - coherence: Logical flow quality
    - knowledge_accuracy: Factual precision
    - clarity: Explanation clarity
    """

    # In practice: call GPT-4o API
    scores = call_annotation_service(gpt4_prompt)
    annotations['control_scores'] = scores

    return annotations
```

3. **Conditional Distillation Fine-tuning**: Train model to respect control fields

```python
def conditional_distillation_finetune(base_model, train_data, learning_rate=5e-5):
    """
    Fine-tune model to generate reasoning conditioned on control fields.
    Uses LoRA for parameter efficiency.
    """
    # Apply LoRA to model
    model = apply_lora(base_model, rank=16, target_modules=['q_proj', 'v_proj'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(3):  # Typically 3 epochs sufficient
        for batch in train_data:
            questions = batch['question']
            control_scores = batch['control_scores']
            target_reasoning = batch['reasoning']

            # Format control prompt
            control_prompts = []
            for q, controls in zip(questions, control_scores):
                # Normalize scores to 0-1 range
                control_dict = {k: v/9.0 for k, v in controls.items()}
                prompt = format_control_prompt(q, control_dict)
                control_prompts.append(prompt)

            # Forward pass: P(R|q,C)
            outputs = model(control_prompts, max_length=2048)

            # Compute distillation loss
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.shape[-1]),
                target_reasoning.view(-1)
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

4. **Inference Control**: Dynamic reasoning adjustment during generation

```python
def generate_with_control(model, question, control_values, max_length=2048):
    """
    Generate reasoning conditioned on control field specifications.
    Dynamically adjusts reasoning depth, breadth, error checking.
    """
    # Format control prompt
    control_prompt = format_control_prompt(question, control_values)

    # Generate reasoning tokens
    with torch.no_grad():
        output = model.generate(
            control_prompt,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95
        )

    reasoning = output.sequences[0]
    return reasoning
```

5. **Control Field Profiles**: Pre-defined configurations for common scenarios

```python
CONTROL_PROFILES = {
    'fast_reasoning': {
        'search_depth': 3,
        'search_breadth': 2,
        'error_detection': 2,
        'error_correction': 1,
        'strategy_switching': 1,
        'efficiency': 9,
        'completeness': 5
    },
    'careful_reasoning': {
        'search_depth': 8,
        'search_breadth': 5,
        'error_detection': 9,
        'error_correction': 9,
        'strategy_switching': 7,
        'correctness': 9,
        'knowledge_accuracy': 9
    },
    'balanced': {
        'search_depth': 5,
        'search_breadth': 3,
        'error_detection': 6,
        'error_correction': 6,
        'strategy_switching': 4,
        'correctness': 8,
        'efficiency': 6,
        'clarity': 7
    }
}

# Usage during inference
fast_result = generate_with_control(
    model, question,
    CONTROL_PROFILES['fast_reasoning']
)

careful_result = generate_with_control(
    model, question,
    CONTROL_PROFILES['careful_reasoning']
)
```

## Practical Guidance

**When to Apply:**
- Need dynamic reasoning depth adjustment without retraining
- Working with heterogeneous problem difficulties
- Inference latency and cost sensitive to reasoning amount
- Want interpretable, controllable reasoning traces

**Setup Requirements:**
- Base model: Qwen2.5-32B-Instruct or similar
- Training data: 4,000+ problems with diverse reasoning patterns
- Annotation service: ChatGPT-4o for control score assignment
- 32 A100 GPUs with DeepSpeed ZeRO-3 for efficient training

**Performance Expectations:**
- AIME2024: 70.0% accuracy
- MATH500: 93.2% accuracy
- Speed improvement: 20-40% vs. full reasoning on easy problems
- Quality control: Interpretable reasoning traces

**Control Field Ranges and Effects:**
- Search depth (1-9): Higher = more thorough exploration, slower generation
- Search breadth (1-9): Higher = consider more approaches, longer tokens
- Error detection (0-9): Higher = more verification steps, catches mistakes
- Error correction (0-9): Higher = more fix attempts, reduces errors
- Strategy switching (0-9): Higher = tries multiple methods, more thorough

**Recommended Workflows:**

*For Performance-Critical Applications:*
```
Set control_profiles['fast_reasoning']
Monitor accuracy, adjust search_depth if needed
Typical usage: 50-70% of full reasoning tokens
```

*For High-Stakes Reasoning:*
```
Set control_profiles['careful_reasoning']
Full token budget allowance
Typical usage: 90-100% of full reasoning tokens
```

**Fine-tuning Strategies:**
- Epoch 1: Full learning rate (5e-5), establish base behavior
- Epoch 2: Reduced LR (2e-5), fine-tune control sensitivity
- Epoch 3: Very low LR (1e-5), stabilize behavior
- LoRA rank 16 typically sufficient (try 8 for smaller updates)

**Common Issues:**
- Model ignores control fields: Increase LoRA rank or extend training
- Uneven control responsiveness: Some fields matter more—weight losses differently
- Generation instability: Reduce temperature, increase top_p threshold
- Control values overconstrain: Use normalized 0-1 ranges instead of raw 0-9

## Reference

Implemented on Qwen2.5-32B-Instruct with Control-R-4K dataset. Training uses 32 A100 GPUs, DeepSpeed ZeRO-3, ~1,660 steps. Achieves SOTA at 32B scale: 70.0% AIME2024, 93.2% MATH500. Human evaluation validates control field interpretability and effectiveness.
