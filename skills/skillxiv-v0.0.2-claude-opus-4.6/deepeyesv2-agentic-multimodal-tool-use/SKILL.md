---
name: deepeyesv2-agentic-multimodal-tool-use
title: "DeepEyesV2: Toward Agentic Multimodal Model"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.05271"
keywords: [Multimodal AI, Agent Reasoning, Tool Use, Code Execution, Web Search Integration]
description: "Train multimodal agents to dynamically invoke tools (code execution, web search) within reasoning loops through a two-stage pipeline combining cold-start supervised learning with reinforcement learning—enabling task-adaptive tool invocation for perception, reasoning, and retrieval tasks."
---

# Train Agentic Multimodal Models with Dynamic Tool Invocation

Multimodal models excel at perception but struggle with complex reasoning requiring external tools. DeepEyesV2 solves this through a two-stage training approach: first establishing robust tool-use foundations via supervised learning on curated data, then refining tool invocation decisions through reinforcement learning. The result is a model that adaptively chooses when and which tools to use based on task requirements.

The key insight is that direct reinforcement learning fails without foundational tool-use competence—models either abandon code generation or engage in reward hacking. By decoupling cold-start training (establishing reliability) from RL refinement (optimizing invocation), the system achieves both robustness and flexibility.

## Core Concept

DeepEyesV2 architecture treats code execution and web search as interchangeable, context-dependent tools. The model generates reasoning plans, decides whether direct reasoning suffices or tool invocation is necessary, and can emit executable Python code or structured search queries. Tool outputs become observations fed back into the reasoning context for continued iteration.

The two-stage training strategy addresses a fundamental challenge: naive RL on tool invocation fails because models lack foundational competence. Cold-start supervised training on high-quality, curated data establishes this foundation before RL refines tool selection.

## Architecture Overview

- **Multimodal Input Handler**: Processes images and text queries simultaneously
- **Tool Invocation Selector**: Decides whether to reason directly or invoke tools (code/search)
- **Code Executor**: Runs Python operations on images (crop, numerical analysis, marking, enhancement)
- **Web Search Module**: Retrieves relevant webpages via SerpAPI for information-seeking tasks
- **Context Manager**: Converts tool outputs to observations and appends to reasoning context
- **Two-Stage Learning Pipeline**: Cold-start SFT followed by RL refinement

## Implementation Steps

**Step 1: Data Curation for Cold-Start Training**

Construct a diverse, curated dataset emphasizing cases where tools improve performance. Filter by difficulty (unsolvable by base models) and tool necessity (tools required for correctness).

```python
def curate_tool_training_data(tasks, base_model, tool_types):
    """
    Curate high-quality tool-use training data.

    Args:
        tasks: Pool of training tasks with images, questions, answers
        base_model: Reference model for difficulty filtering
        tool_types: Available tools (code, search, etc.)

    Returns:
        curated_dataset: Filtered tasks with tool-annotated solutions
    """
    curated = []

    for task in tasks:
        # Generate solution with tools using advanced LLM
        solution = generate_solution_with_tools(
            task, tool_types, model="gpt-4o"
        )

        # Check if base model can solve without tools
        base_solution = base_model.solve(task)
        if base_solution.correct:
            continue  # Skip: too easy for base model

        # Check if solution uses tools effectively
        if solution.correct and solution.uses_tools:
            # Verify no code errors or placeholder generation
            if validate_code_execution(solution.code):
                curated.append((task, solution))

    return curated
```

**Step 2: Supervised Fine-Tuning (Cold-Start)**

Fine-tune the base multimodal model on curated data using standard supervised learning. This establishes tool-use competence before RL optimization.

```python
def cold_start_supervised_training(model, curated_data):
    """
    Supervised fine-tuning on curated tool-use trajectories.

    Args:
        model: Base multimodal model (e.g., Qwen2.5-VL-7B)
        curated_data: [(image, question, correct_solution), ...]

    Returns:
        finetuned_model: Model with foundational tool-use ability
    """
    import torch
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(curated_data) * 3
    )

    batch_size = 128
    num_epochs = 3

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx in range(0, len(curated_data), batch_size):
            batch = curated_data[batch_idx:batch_idx + batch_size]

            # Prepare batch
            images = [item[0] for item in batch]
            questions = [item[1] for item in batch]
            solutions = [item[2] for item in batch]

            # Forward pass: predict next token given image + question
            logits = model(images=images, text=questions)

            # Compute loss against ground truth solutions
            loss = compute_token_loss(logits, solutions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss / len(curated_data):.4f}")

    return model
```

**Step 3: Reinforcement Learning Refinement**

Use on-policy RL (DAPO algorithm) to optimize tool invocation decisions. Define simple outcome-based rewards: accuracy + format compliance.

```python
def reinforcement_learning_refinement(model, eval_tasks, num_rollouts=16):
    """
    On-policy RL to refine tool invocation decisions.

    Args:
        model: Cold-start fine-tuned model
        eval_tasks: Tasks for RL training
        num_rollouts: Trajectories per prompt

    Returns:
        rl_optimized_model: Model with refined tool selection
    """
    from trl import DPOTrainer  # Example RL framework

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    for task in eval_tasks:
        # Generate multiple rollouts
        rollouts = []
        for _ in range(num_rollouts):
            trajectory = model.generate(
                image=task.image,
                prompt=task.question,
                max_tokens=16384,
                temperature=0.7  # Stochasticity for exploration
            )
            rollouts.append(trajectory)

        # Compute rewards
        rewards = []
        for trajectory in rollouts:
            # Accuracy reward: does final answer match ground truth?
            accuracy = 1.0 if trajectory.answer == task.answer else 0.0

            # Format reward: valid code, structured output?
            format_penalty = 0.0
            if trajectory.uses_code and not is_valid_python(trajectory.code):
                format_penalty = 0.1

            total_reward = accuracy - format_penalty
            rewards.append(total_reward)

        # Policy gradient update using rewards
        advantages = torch.tensor(rewards) - torch.tensor(rewards).mean()

        # Update model to increase probability of high-reward trajectories
        for trajectory, advantage in zip(rollouts, advantages):
            if advantage > 0:  # Update toward good trajectories
                loss = -model.log_prob(trajectory) * advantage
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model
```

**Step 4: Tool Invocation Modules**

Implement code execution and web search modules that model can invoke during reasoning.

```python
def execute_image_code(image, python_code):
    """
    Execute Python code on image (crop, analyze, mark, enhance).

    Args:
        image: PIL Image or numpy array
        python_code: Python code string with image operations

    Returns:
        result: Code output (modified image or numerical result)
    """
    import io
    from PIL import Image, ImageDraw
    import numpy as np

    # Safe execution sandbox
    sandbox = {
        'image': image,
        'np': np,
        'Image': Image,
        'ImageDraw': ImageDraw
    }

    try:
        exec(python_code, sandbox)
        return sandbox.get('result', image)
    except Exception as e:
        return f"Code execution error: {e}"

def web_search_query(query):
    """
    Execute web search returning relevant webpages.

    Args:
        query: Search query string

    Returns:
        results: List of (title, snippet, url) tuples
    """
    from serpapi import GoogleSearch

    params = {
        "q": query,
        "num": 5,
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    output = []
    for item in results.get('organic_results', []):
        output.append({
            'title': item['title'],
            'snippet': item.get('snippet', ''),
            'url': item['link']
        })

    return output
```

## Practical Guidance

**When to Use DeepEyesV2 Approach:**
- Multimodal reasoning tasks (image understanding + code/search)
- Domains where external tools improve accuracy (math, coding, research)
- Applications tolerating RL training overhead for improved generalization

**When NOT to Use:**
- Real-time inference systems (cold-start + RL training is time-consuming)
- Tasks where tool use provides minimal benefit (pure perception tasks)
- Scenarios with restricted code execution environments

**Hyperparameters and Configuration:**
- Cold-start batch size: 128 (larger batches stabilize supervised learning)
- RL learning rate: 1×10⁻⁶ (lower than SFT to avoid destabilizing cold-start gains)
- Max response length: 16,384 tokens (accommodates code + reasoning)
- KL coefficient: 0.0 (no KL penalty; focus on reward maximization)

**Pitfalls to Avoid:**
1. **Skipping cold-start** - Direct RL without supervision typically fails; models resort to reward hacking
2. **Insufficient tool diversity** - Curate data across multiple tool types (code, search, reasoning) to enable flexible invocation
3. **Loose reward signals** - Use binary accuracy + format checks; avoid fuzzy reward engineering that obscures learning signals
4. **Ignoring safety** - Restrict code execution to whitelisted operations; don't allow arbitrary system commands

---

Reference: https://arxiv.org/abs/2511.05271
