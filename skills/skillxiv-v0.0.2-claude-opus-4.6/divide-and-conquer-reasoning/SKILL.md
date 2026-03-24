---
name: divide-and-conquer-reasoning
title: "Training LLMs for Divide-and-Conquer Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02477"
keywords: [Divide-and-Conquer, Structured Reasoning, Problem Decomposition, LLM Training]
description: "Train models to decompose complex problems into subproblems via divide-and-conquer reasoning. Structured approach enables systematic solution assembly and improved long-horizon reasoning compared to end-to-end generation."
---

# Divide-and-Conquer Reasoning for LLMs

## Problem
Large language models often attempt end-to-end reasoning on complex problems, leading to solution quality degradation as problem complexity increases. Models lack systematic decomposition strategies.

Structured problem solving with explicit subproblem identification and assembly enables better performance than flat, unstructured approaches.

## Core Concept
The method trains models to explicitly: (1) recognize decomposable structure in problems, (2) identify independent subproblems, (3) solve subproblems with recursive calls, and (4) synthesize subproblem solutions into final answers.

This mirrors human problem-solving strategies and enables scaling to larger problems through compositional reasoning.

## Architecture Overview

- **Problem Analysis**: Identify decomposable structure and dependencies
- **Subproblem Identification**: Break problem into independent components
- **Recursive Solving**: Apply same reasoning strategy to subproblems
- **Solution Assembly**: Integrate subproblem solutions systematically
- **Base Case Handling**: Direct solving for atomic subproblems
- **Training Data**: Synthetic divide-and-conquer decomposition traces

## Implementation

### Step 1: Annotate Training Data with Decomposition
Create training examples showing divide-and-conquer reasoning.

```python
def create_decomposition_trace(problem, solution, model):
    """Generate divide-and-conquer reasoning trace for training."""
    trace = {
        'original_problem': problem,
        'decomposition': None,
        'subproblems': [],
        'subsolutions': [],
        'final_assembly': None,
        'final_solution': solution
    }

    # Step 1: Identify decomposition structure
    decomposition_prompt = f"""Analyze this problem and identify how to decompose it:

Problem: {problem}

Explain:
1. Is this decomposable? Yes/No
2. What are the independent subproblems?
3. What is the structure of dependencies?"""

    decomposition = model.generate(decomposition_prompt)
    trace['decomposition'] = decomposition

    # Step 2: Extract subproblems
    subproblems = extract_subproblems(decomposition)
    trace['subproblems'] = subproblems

    # Step 3: Solve subproblems
    for subproblem in subproblems:
        subproblem_solution = solve_subproblem(subproblem, model)
        trace['subsolutions'].append(subproblem_solution)

    # Step 4: Assembly
    assembly_prompt = f"""Given these subproblem solutions, assemble the final answer:

Original problem: {problem}
Subproblems and solutions:
{format_subsolutions(subproblems, trace['subsolutions'])}

Final answer:"""

    assembly_explanation = model.generate(assembly_prompt)
    trace['final_assembly'] = assembly_explanation

    return trace

def extract_subproblems(decomposition):
    """Parse decomposition to extract subproblems."""
    # In practice, use regex or NLP to extract subproblem descriptions
    lines = decomposition.split('\n')
    subproblems = []
    for line in lines:
        if line.startswith('-') or line.startswith('*'):
            subproblems.append(line.strip())
    return subproblems
```

### Step 2: Supervise Decomposition Policy
Train model to recognize and execute decomposition.

```python
def train_decomposition_policy(model, decomposition_traces, num_epochs=3):
    """Train model to decompose problems effectively."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for trace in decomposition_traces:
            # Supervised learning on decomposition step
            problem = trace['original_problem']
            target_decomposition = trace['decomposition']

            prompt = f"Decompose this problem:\n\n{problem}\n\nDecomposition:"

            # Generate and compute loss
            generated = model.generate(prompt)
            decomposition_loss = compute_similarity_loss(generated, target_decomposition)

            # Supervised learning on subproblem identification
            target_subproblems = trace['subproblems']
            subproblem_loss = compute_classification_loss(model, problem, target_subproblems)

            # Supervised learning on solution assembly
            subsolution_inputs = format_subsolutions(trace['subproblems'], trace['subsolutions'])
            target_assembly = trace['final_assembly']
            assembly_loss = compute_assembly_loss(model, subsolution_inputs, target_assembly)

            # Weighted combination
            total_loss = 0.4 * decomposition_loss + 0.3 * subproblem_loss + 0.3 * assembly_loss
            total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return model
```

### Step 3: Implement Recursive Inference
Execute divide-and-conquer reasoning at inference time.

```python
def divide_and_conquer_inference(problem, model, max_depth=3, base_case_threshold=100):
    """Solve problem via divide-and-conquer reasoning."""

    def solve_recursive(current_problem, depth):
        # Base case: problem small enough to solve directly
        if len(current_problem) < base_case_threshold or depth >= max_depth:
            prompt = f"Solve this problem directly:\n\n{current_problem}"
            return model.generate(prompt)

        # Recursive case: decompose
        decomposition_prompt = f"Decompose into subproblems:\n\n{current_problem}"
        decomposition = model.generate(decomposition_prompt)

        subproblems = extract_subproblems(decomposition)

        # Solve subproblems recursively
        subsolutions = []
        for subproblem in subproblems:
            subsolution = solve_recursive(subproblem, depth + 1)
            subsolutions.append(subsolution)

        # Assemble solutions
        assembly_prompt = f"""Combine these solutions:

Problem: {current_problem}

Subproblems and solutions:
{format_subsolutions(subproblems, subsolutions)}

Final answer:"""

        final_solution = model.generate(assembly_prompt)
        return final_solution

    return solve_recursive(problem, depth=0)
```

### Step 4: Reinforce Successful Decompositions
Use RL to improve decomposition strategy.

```python
def reinforce_decomposition(model, problems, reward_function, num_rl_steps=1000):
    """Improve decomposition via reinforcement learning."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for step in range(num_rl_steps):
        problem = np.random.choice(problems)

        # Generate decomposition
        decomposition_prompt = f"Decompose:\n\n{problem}"
        decomposition = model.generate_with_logprobs(decomposition_prompt)

        # Extract subproblems
        subproblems = extract_subproblems(decomposition['text'])

        # Solve and get final answer
        final_answer = divide_and_conquer_inference(problem, model)

        # Evaluate solution quality
        reward = reward_function(problem, final_answer)

        # Policy gradient
        log_probs = decomposition['log_probs']
        policy_loss = -reward * log_probs.sum()
        policy_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return model
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max recursion depth | 3-5 | Prevent infinite recursion |
| Base case threshold | 100-200 tokens | When to solve directly |
| Decomposition weight | 0.4 | Importance in training loss |
| RL reward discount | 0.99 | Multi-step credit assignment |
| Learning rate (SFT) | 1e-4 | Standard fine-tuning |
| Learning rate (RL) | 1e-5 | Conservative RL tuning |

### When to Use

- Complex multi-part problems (proofs, algorithm design, data analysis)
- Problems with clear decomposable structure
- Hierarchical reasoning tasks (understanding documents with sections)
- Long-horizon problem solving
- Improving robustness and interpretability

### When Not to Use

- Simple, atomic problems (overhead not justified)
- Problems requiring holistic understanding (some domains)
- Real-time systems where recursion adds latency
- Domains where decomposition isn't natural
- Models already strong at end-to-end reasoning

### Common Pitfalls

1. **Poor subproblem independence**: Decomposition creating interdependent subproblems reduces benefit. Validate independence.
2. **Assembly bottleneck**: Good subproblem solving but poor assembly loses gains. Dedicate training to assembly step.
3. **Recursion depth explosion**: Max depth too high or base case threshold too low causes explosion. Monitor recursion trace.
4. **Inconsistent conventions**: Subproblem format differs from training data. Enforce consistency in prompting.

## Reference
Training LLMs for Divide-and-Conquer Reasoning
https://arxiv.org/abs/2602.02477
