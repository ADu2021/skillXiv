---
name: k-search-kernel-generation-world-models
title: "K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.19128"
keywords: [program synthesis, GPU optimization, search planning, world models, LLM agents]
description: "Generate optimized GPU kernels by treating LLMs as planning engines that co-evolve with a world model. Decouples high-level algorithmic planning from low-level implementation, enabling structured search through optimization strategies. LLM world model estimates priority scores for pending optimizations while iteratively updating understanding based on execution results. Achieves 2.10× improvement over evolutionary baselines with 14.3× gains on complex MoE kernels."
---

# K-Search: Guided Kernel Optimization via LLM World Modeling

GPU kernel optimization is a high-dimensional search problem combining algorithm selection, memory layout, and platform-specific tuning. Existing approaches treat it as direct code generation—search directly in program space—but this struggles with non-monotonic improvements where some optimizations create temporary performance regressions that unlock larger gains downstream.

The challenge is navigating a complex optimization landscape where local greedy search gets trapped in local optima. Traditional evolutionary approaches apply random mutations uniformly, missing the structure of the optimization space.

## Core Concept

K-Search separates high-level algorithmic planning from low-level implementation details. An LLM world model maintains beliefs about which optimization strategies are most promising, estimates their priority scores, and iteratively updates these beliefs based on execution feedback. Rather than random mutation, the system proposes optimizations according to priority estimates, enabling structured exploration of the optimization space.

The approach operates through three iterative phases:

1. **Action Selection**: World model scores pending optimization strategies and selects most promising
2. **Program Instantiation**: Generate concrete implementations through repeated sampling
3. **World Model Co-Evolution**: Analyze results and update world model (insert new hypotheses, update priorities, prune unpromising branches)

## Architecture Overview

- **Strategy Repository**: Maintains list of optimization strategies (loop unrolling, vectorization, shared memory optimization, etc.) with priority scores
- **LLM World Model**: Estimates which strategies will yield improvement for current kernel; can reason about interactions between strategies
- **Implementation Generator**: Instantiate concrete kernel code from selected strategy using sampling
- **Executor & Profiler**: Run kernel and measure actual improvement
- **Update Logic**: Revise world model beliefs (priorities, confidence) based on real vs. predicted performance

## Implementation

Represent optimization strategies and maintain priorities:

```python
class StrategyRepository:
    def __init__(self):
        self.strategies = [
            {'name': 'loop_unroll', 'priority': 0.5, 'applied': False},
            {'name': 'shared_memory', 'priority': 0.7, 'applied': False},
            {'name': 'vectorize', 'priority': 0.4, 'applied': False},
            {'name': 'thread_block_optimize', 'priority': 0.6, 'applied': False},
            {'name': 'memory_coalesce', 'priority': 0.8, 'applied': False}
        ]

    def select_by_priority(self):
        """Select strategy with highest priority that hasn't been applied."""
        available = [s for s in self.strategies if not s['applied']]
        if not available:
            return None

        sorted_strategies = sorted(available, key=lambda x: x['priority'], reverse=True)
        return sorted_strategies[0]

    def update_priority(self, strategy_name, improvement_factor, uncertainty=0.1):
        """Update priority based on actual improvement."""
        for s in self.strategies:
            if s['name'] == strategy_name:
                # Increase priority if improvement was significant
                if improvement_factor > 1.0:
                    s['priority'] = min(1.0, s['priority'] + 0.1 * improvement_factor)
                else:
                    s['priority'] = max(0.0, s['priority'] - 0.2)
                break
```

Implement world model selection and generation:

```python
def world_model_select_action(kernel_code, repository, model):
    """
    Use LLM to estimate which strategy is most promising for this kernel.
    """
    prompt = f"""
Analyze this GPU kernel and identify the single most impactful optimization:

{kernel_code}

Available strategies:
- loop_unroll: Unroll inner loops to reduce branch overhead
- shared_memory: Use shared memory for frequently accessed data
- vectorize: Convert to vector operations
- thread_block_optimize: Adjust thread block dimensions
- memory_coalesce: Ensure memory accesses are coalesced

Respond with ONLY the strategy name and a confidence score (0.0-1.0).
Format: STRATEGY: [name]
CONFIDENCE: [score]
"""

    response = model.generate(prompt, temperature=0.3, max_tokens=100)

    # Parse response
    lines = response.split('\n')
    strategy_name = None
    confidence = 0.5

    for line in lines:
        if 'STRATEGY:' in line:
            strategy_name = line.split(':')[1].strip()
        elif 'CONFIDENCE:' in line:
            confidence = float(line.split(':')[1].strip())

    # Override if strategy already applied
    selected = repository.select_by_priority()
    if selected:
        return selected['name'], confidence
    return strategy_name, confidence

def instantiate_optimization(kernel_code, strategy, model, num_samples=5):
    """
    Generate multiple concrete kernel implementations applying strategy.
    """
    prompt = f"""
Apply the '{strategy}' optimization to this CUDA kernel:

{kernel_code}

Generate valid, executable CUDA code. Return only the optimized kernel.
"""

    implementations = []
    for _ in range(num_samples):
        impl = model.generate(prompt, temperature=0.7, max_tokens=500)
        if is_valid_cuda(impl):  # Check syntax
            implementations.append(impl)

    return implementations if implementations else [kernel_code]
```

Implement execution and world model update:

```python
def execute_and_evaluate(kernel_code, baseline_kernel, device='cuda'):
    """
    Compile and run kernel, measure speedup vs. baseline.
    """
    try:
        # Compile kernel
        compiled = compile_cuda_kernel(kernel_code)

        # Run and benchmark
        baseline_time = benchmark_kernel(baseline_kernel, iterations=100)
        optimized_time = benchmark_kernel(compiled, iterations=100)

        speedup = baseline_time / optimized_time
        return speedup, None

    except Exception as e:
        return 1.0, str(e)

def k_search_optimize(
    baseline_kernel, model, iterations=10
):
    """
    Main K-Search loop: iteratively select, instantiate, and evaluate.
    """
    repository = StrategyRepository()
    current_best = baseline_kernel
    current_speedup = 1.0

    for iteration in range(iterations):
        # Step 1: Select optimization strategy
        strategy, confidence = world_model_select_action(
            current_best, repository, model
        )

        if strategy is None:
            break

        # Step 2: Instantiate concrete implementations
        implementations = instantiate_optimization(
            current_best, strategy, model, num_samples=5
        )

        # Step 3: Evaluate implementations
        best_impl = current_best
        best_impl_speedup = 1.0

        for impl in implementations:
            speedup, error = execute_and_evaluate(impl, baseline_kernel)

            if speedup > best_impl_speedup:
                best_impl = impl
                best_impl_speedup = speedup

        # Step 4: Update world model
        overall_speedup = current_speedup * best_impl_speedup

        repository.update_priority(
            strategy,
            improvement_factor=best_impl_speedup,
            uncertainty=0.1
        )

        if overall_speedup > current_speedup:
            current_best = best_impl
            current_speedup = overall_speedup

            print(f"Iteration {iteration}: {strategy} → {overall_speedup:.2f}× speedup")

        # Mark strategy applied if no improvement expected
        if best_impl_speedup < 1.01:
            for s in repository.strategies:
                if s['name'] == strategy:
                    s['applied'] = True

    return current_best, current_speedup
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Iterations | 10 | Increase for longer search; diminishing returns after 5–7 major optimizations |
| Samples per strategy | 5 | 3–5 balances diversity with cost; >5 often redundant |
| Temperature (generation) | 0.7 | Higher (0.8–0.9) for exploration; lower (0.3–0.5) for conservative refinement |
| Confidence threshold | 0.3 | Skip strategies with lower confidence to avoid dead ends |

**When to use**: For GPU kernel optimization, operator fusion, or complex program synthesis where optimization space is high-dimensional and non-monotonic.

**When not to use**: For simple code generation without specific performance targets; overhead of iteration and evaluation is unjustified.

**Common pitfalls**:
- Infinite loops on redundant strategies; track applied strategies and mark as complete
- Compilation errors from generated code; validate syntax before execution
- Overfitting world model priorities to early successes; add exploration bonus for untested strategies

## Reference

K-Search achieves 2.10× average improvement over OpenEvolve and 2.21× over ShinkaEvolve, with particularly strong results on complex kernels (14.3× improvement on MoE kernels). The approach demonstrates value of treating optimization as structured planning rather than random search.
