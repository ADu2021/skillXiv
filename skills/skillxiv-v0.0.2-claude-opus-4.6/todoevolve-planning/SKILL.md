---
name: todoevolve-planning
title: "TodoEvolve: Learning to Architect Agent Planning Systems"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.07839"
keywords: [Agent Planning, Architecture Synthesis, Impedance Optimization, Preference Learning, Task-Specific Systems]
description: "Autonomously synthesize task-specific agent planning architectures by optimizing for correctness, stability, and efficiency via impedance-guided preference optimization."
---

# TodoEvolve: Learning to Architect Agent Planning Systems

## Problem Context

Traditional agent planning systems rely on fixed, hand-crafted structures (hierarchical decomposition, linear workflows, graph-based planning) that cannot adapt to diverse task requirements. No single planning topology works optimally across all tasks; some benefit from linear structures, others from dynamic graphs.

## Core Concept

**Impedance-Guided Preference Optimization (IGPO)** trains agents to generate customized planning systems by optimizing for three competing objectives: correctness (finding valid solutions), stability (consistent execution), and efficiency (minimizing tokens). A "Cognitive Impedance" metric combines execution cost, error frequency, smoothness, and planning-to-execution ratio into a unified measure.

## Architecture Overview

- **PlanFactory Design Space**: Standardizes planning systems across four dimensions (topology, initialization, adaptation, navigation)
- **Data Construction**: Standardize existing paradigms, generate candidates, validate via execution, construct preference pairs
- **Two-Stage Training**: SFT for syntactic validity → IGPO for efficiency-aware alignment

## Implementation

**Phase 1: PlanFactory Design Space**

```python
class PlanFactory:
    def __init__(self):
        self.design_dimensions = {
            'topology': ['linear', 'tree', 'dag', 'graph'],
            'initialization': ['top-down', 'bottom-up', 'hybrid'],
            'adaptation': ['static', 'incremental', 'reactive'],
            'navigation': ['sequential', 'parallel', 'dynamic']
        }

    def generate_candidate_plan(self, task):
        # Sample from design space
        topology = random.choice(self.design_dimensions['topology'])
        initialization = random.choice(
            self.design_dimensions['initialization'])
        adaptation = random.choice(
            self.design_dimensions['adaptation'])
        navigation = random.choice(
            self.design_dimensions['navigation'])

        # Generate planning code
        plan_code = llm.generate_plan_code(
            task=task,
            topology=topology,
            initialization=initialization,
            adaptation=adaptation,
            navigation=navigation
        )

        return plan_code, {
            'topology': topology,
            'initialization': initialization,
            'adaptation': adaptation,
            'navigation': navigation
        }
```

**Phase 2: Data Construction Pipeline**

```python
def construct_training_data():
    data = []

    for task in training_tasks:
        # Generate diverse candidate plans
        candidates = []
        for _ in range(50):  # Evolutionary sampling
            plan, design = generate_candidate_plan(task)
            candidates.append((plan, design))

        # Validate through execution
        validated = []
        for plan, design in candidates:
            try:
                # Execute plan on task
                results = execute_plan(plan, task)

                # Check correctness
                if results.is_correct:
                    validated.append({
                        'plan': plan,
                        'design': design,
                        'results': results
                    })
            except Exception:
                continue  # Skip execution failures

        # Compute Cognitive Impedance for each valid plan
        for item in validated:
            impedance = compute_impedance(
                item['results'],
                execution_cost=item['results'].tokens_used,
                error_frequency=item['results'].error_count,
                smoothness=item['results'].execution_smoothness,
                ratio=item['results'].planning_tokens / item['results'].total_tokens
            )
            item['impedance'] = impedance

        # Construct preference pairs
        # Criterion 1: Correctness (all should be correct)
        # Criterion 2: Efficiency (prefer lower impedance)
        for i, item_a in enumerate(validated):
            for j, item_b in enumerate(validated[i+1:]):
                if item_a['impedance'] < item_b['impedance']:
                    # item_a is better (lower impedance)
                    data.append({
                        'task': task,
                        'preferred': item_a['plan'],
                        'dispreferred': item_b['plan']
                    })

    return data
```

**Phase 3: Two-Stage Training**

```python
# Stage 1: Supervised Fine-Tuning (SFT)
# Learn syntactic validity of planning code

def sft_training(model, training_data):
    for epoch in range(num_epochs):
        for task, plan in training_data:
            # Forward pass
            logits = model.forward(task)

            # Loss: likelihood of generating valid plan code
            loss = cross_entropy_loss(logits, plan)
            loss.backward()
            optimizer.step()

# Stage 2: Impedance-Guided Preference Optimization (IGPO)

def igpo_training(model, preference_pairs):
    for epoch in range(num_epochs):
        for pair in preference_pairs:
            task = pair['task']
            preferred_plan = pair['preferred']
            dispreferred_plan = pair['dispreferred']

            # Generate plans from model
            logits_preferred = model.forward(task)
            logits_dispreferred = model.forward(task)

            # Log-likelihood ratio
            log_ratio = (log_likelihood(logits_preferred, preferred_plan) -
                        log_likelihood(logits_dispreferred, dispreferred_plan))

            # DPO-style loss
            loss = -log_sigmoid(log_ratio)
            loss.backward()
            optimizer.step()
```

**Impedance Computation**

```python
def compute_impedance(results,
                     execution_cost,
                     error_frequency,
                     smoothness,
                     ratio):
    # Normalize components
    cost_normalized = execution_cost / max_cost
    error_normalized = error_frequency / max_errors
    smoothness_normalized = 1 - smoothness  # Higher smoothness = lower impedance
    ratio_normalized = ratio  # Lower ratio = better

    # Weighted combination
    impedance = (0.4 * cost_normalized +
                 0.3 * error_normalized +
                 0.2 * smoothness_normalized +
                 0.1 * ratio_normalized)

    return impedance
```

## Practical Guidance

**When to use**: Deploy for complex multi-step tasks with heterogeneous sub-problems that benefit from adaptive planning. Less effective for simple, well-defined workflows.

**Design space customization**: Tailor dimensions to your domain. Add dimensions like "memory-handling" for tasks requiring context management.

**Execution environment**: Ensure reproducible task execution for validation. Use deterministic scheduling; avoid non-deterministic environments.

**Impedance tuning**: Weights (0.4, 0.3, 0.2, 0.1) reflect typical priorities (cost > errors > smoothness > planning-to-execution ratio). Adjust based on task criticality.

**Model capacity**: Start with smaller models (1–3B parameters); larger models improve design quality but increase training cost. Scaling law: performance improves ~10% per doubling of model size.

## Reference

TodoEvolve demonstrates that learning planning architecture is feasible with appropriate preference optimization. The framework achieves 72.12% on GAIA benchmarks versus 55.75% baseline, demonstrating that customized architectures substantially outperform fixed templates. The impedance metric provides a principled way to balance multiple competing objectives in planning system design.
