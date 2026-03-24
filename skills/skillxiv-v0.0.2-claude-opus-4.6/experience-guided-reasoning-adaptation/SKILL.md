---
name: experience-guided-reasoning-adaptation
title: "Experience-Guided Adaptation of Inference-Time Reasoning Strategies"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.11519"
keywords: [Inference Optimization, Reasoning Strategies, Experience Memory, Adaptive Control, Cost Reduction]
description: "Dynamically adapt LLM reasoning strategies at inference time by curating episodic memory of past problem solutions—generate task-specific prompts, tool configs, and control logic for up to 111× cost reduction and 14% accuracy gains."
---

# Adapt LLM Reasoning Strategies Dynamically Using Past Experience

Most inference-time reasoning optimization fixes prompts and parameters before deployment. Experience-Guided Reasoner (EGuR) treats reasoning strategy as dynamic: it maintains structured memory of past solutions and generates new strategies tailored to each problem's characteristics. Rather than modifying text inputs, EGuR produces complete computational procedures with custom prompts, sampling configs, tool selections, and control flow.

This approach achieves simultaneous improvements in accuracy (up to 14%) and efficiency (up to 111× cost reduction) by matching strategy intensity to problem difficulty—hard problems get more reasoning steps, easy ones get direct inference.

## Core Concept

Reasoning strategies span multiple dimensions: textual prompts, sampling parameters (temperature, top-k), tool choices, and control structures (few-shot examples, chain-of-thought depth). Most systems fix all dimensions at deployment time. EGuR instead generates strategies adaptively using two components:

1. **Guide**: An LLM-based meta-reasoner that conditions on the current problem and retrieves relevant past experiences, generating multiple candidate strategies
2. **Consolidator**: Maintains structured memory (successful strategies, general insights) and updates it based on execution feedback

The key insight is that strategy generation is itself learnable and cacheable—successful strategies for similar problems can be retrieved, reducing synthesis cost on the critical path.

## Architecture Overview

- **Strategy Library**: Indexed store of successful strategies with problem signatures for fast retrieval
- **General Notes**: High-level insights about strategy effectiveness, failure patterns, and tradeoff principles
- **Guide Module**: LLM that generates k candidate strategies per problem conditioned on current context and retrieved experience
- **Compositional Strategy Representation**: Formal operations (sequential, parallel, conditional, recursive) enabling adaptation across all strategy dimensions
- **Memory Consolidation**: Selective retention policies prioritizing recent and reusable experiences

## Implementation Steps

**Step 1: Memory Structures.** Initialize strategy library and notes for experience curation.

```python
class ExperienceMemory:
    def __init__(self, max_strategies=10000):
        self.strategy_library = {}  # {problem_hash: [strategies]}
        self.general_notes = []     # High-level insights
        self.max_strategies = max_strategies
        self.access_count = {}      # Track usage frequency

    def add_strategy(self, problem_sig, strategy, success, cost, accuracy):
        """
        Store successful strategy with problem signature.
        problem_sig: hash/embedding of problem (domain, difficulty, type)
        strategy: {prompt, temperature, tools, control_flow}
        """
        key = problem_sig
        if key not in self.strategy_library:
            self.strategy_library[key] = []

        entry = {
            'strategy': strategy,
            'success': success,
            'cost': cost,
            'accuracy': accuracy,
            'timestamp': time.time()
        }
        self.strategy_library[key].append(entry)
        self.access_count[key] = self.access_count.get(key, 0) + 1

        # Evict oldest low-frequency entries if at capacity
        if sum(len(v) for v in self.strategy_library.values()) > self.max_strategies:
            self._evict_least_useful()

    def retrieve_strategies(self, problem_sig, k=3):
        """Retrieve top-k strategies similar to problem signature."""
        # Simple: exact match first, then fallback to similar signatures
        if problem_sig in self.strategy_library:
            strategies = self.strategy_library[problem_sig]
            # Sort by success + efficiency
            ranked = sorted(
                strategies,
                key=lambda x: x['success'] * (1 - x['cost']/100),
                reverse=True
            )
            return ranked[:k]
        return []
```

**Step 2: Strategy Generation.** Guide module generates candidate strategies conditioned on problem and retrieved experiences.

```python
def generate_strategies(problem, memory, guide_llm, k=5):
    """
    Generate k candidate strategies for given problem.
    """
    # Extract problem signature (domain, complexity, type)
    problem_sig = extract_problem_signature(problem)

    # Retrieve relevant past strategies
    past_strategies = memory.retrieve_strategies(problem_sig, k=3)

    # Construct context for guide
    guide_prompt = f"""
    Problem: {problem}

    Related successful strategies from past:
    {format_strategies(past_strategies)}

    Generate {k} diverse strategies. Each strategy specifies:
    - prompt_template: the reasoning prompt
    - temperature: sampling temperature (0.0-2.0)
    - tools: list of tools to use (search, calculate, etc.)
    - max_tokens: reasoning budget
    - control_flow: chain_of_thought, direct, tree_search, etc.

    Format each as JSON.
    """

    # Call guide LLM
    response = guide_llm(guide_prompt)
    strategies = parse_json_list(response)

    return strategies
```

**Step 3: Strategy Execution and Feedback.** Execute generated strategies and collect results for consolidation.

```python
def execute_and_rank_strategies(problem, strategies, main_llm, verifier, k=5):
    """
    Execute multiple strategies in parallel, rank by quality.
    Returns best strategy and execution metadata for memory update.
    """
    results = []

    for i, strat in enumerate(strategies):
        # Execute strategy
        solution = execute_strategy(
            problem,
            prompt_template=strat['prompt_template'],
            temperature=strat['temperature'],
            tools=strat['tools'],
            max_tokens=strat['max_tokens'],
            control_flow=strat['control_flow'],
            llm=main_llm
        )

        # Verify and score
        score = verifier.score(problem, solution)
        cost = estimate_cost(strat)

        results.append({
            'strategy': strat,
            'solution': solution,
            'accuracy': score,
            'cost': cost,
            'efficiency': score / (cost + 1e-8)
        })

    # Rank and return best
    best = max(results, key=lambda x: x['efficiency'])

    return best['solution'], best['strategy'], best
```

**Step 4: Memory Consolidation.** Update strategy library and extract general insights.

```python
def consolidate_memory(memory, execution_results, problem_sig):
    """
    Update memory: store successful strategy, extract patterns.
    """
    best_strat = execution_results['strategy']
    accuracy = execution_results['accuracy']
    cost = execution_results['cost']

    # Add to library
    memory.add_strategy(
        problem_sig,
        best_strat,
        success=(accuracy > 0.7),
        cost=cost,
        accuracy=accuracy
    )

    # Extract and update general notes if pattern found
    # e.g., "complex math problems benefit from max_tokens >= 500"
    if accuracy > 0.9 and cost < 50:
        pattern = f"Low-cost high-accuracy solution for {problem_sig}: {best_strat}"
        memory.add_insight(pattern)

    # Cleanup: remove redundant entries
    memory.cleanup()
```

## Practical Guidance

**When to Use:** Complex reasoning tasks (math, coding, multi-step QA) where problem difficulty varies; inference-time budgets matter (cost-accuracy tradeoff).

**Hyperparameters:**
- Number of candidate strategies k: 3–7 trades generation cost vs search quality
- Memory size: 1000–100K entries depending on problem diversity and cache budget
- Update frequency: after every 10–100 problems to keep patterns fresh
- Early-exit threshold: stop if single strategy accuracy exceeds 0.95 to skip generation

**Pitfalls:**
- Over-reliance on retrieval can lead to premature convergence on suboptimal strategies; force diversity in generation
- Consolidation policies must handle changing problem distributions; monitor accuracy drift
- Memory growth unbounded; implement principled eviction (LRU, low-frequency, low-utility)
- Generated strategies can be incoherent; add validation layer before execution

**When NOT to Use:** Fixed-task inference where problem distribution is static and small; simpler deterministic problems not benefiting from strategy adaptation.

**Integration:** Works with any LLM; compatible with tool-use frameworks (RAG, APIs). Pairs well with outcome verification for feedback signal.

---
Reference: https://arxiv.org/abs/2511.11519
