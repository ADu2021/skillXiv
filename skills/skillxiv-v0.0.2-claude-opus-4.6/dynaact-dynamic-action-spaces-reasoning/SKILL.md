---
name: dynaact-dynamic-action-spaces-reasoning
title: "DynaAct: LLM Reasoning with Dynamic Action Spaces"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.08043"
keywords: [Action Space Design, Submodular Optimization, Sequential Reasoning, LLM Planning, Greedy Selection]
description: "Automatically construct compact, diverse action spaces for LLM reasoning through corpus-based estimation and submodular optimization—enabling efficient decision-making without manual specification or expensive exhaustive search."
---

# Construct Dynamic Action Spaces for Efficient LLM Reasoning

Language model agents typically use manually-defined action spaces (fixed operations available at each step) that lack scalability, or generate all possible actions exhaustively—an expensive approach. DynaAct solves this through a two-stage method: automatically estimate the space of viable actions from a corpus of reasoning tasks, then greedily select a compact subset optimized for relevance and diversity to the current problem.

The result is efficient, adaptive action spaces that improve performance on six reasoning benchmarks while maintaining fast inference without significant latency overhead.

## Core Concept

DynaAct treats action space construction as a submodular optimization problem. The system learns general action patterns from diverse reasoning examples, then selects task-specific actions balancing two criteria:

1. **Relevance**: How applicable is each action to the current problem state?
2. **Diversity**: Do selected actions cover different solution strategies?

This two-stage approach avoids both manual specification burdens and exhaustive generation costs, enabling scalable, efficient reasoning agents.

## Architecture Overview

- **Corpus Analysis Module**: Scans diverse reasoning tasks to extract general action patterns
- **Action Sketch Extraction**: Uses LLM to identify high-level solution strategies from examples
- **Space Estimation**: Builds compact representation of feasible action space
- **Submodular Objective**: Defines relevance + diversity tradeoff as optimization function
- **Greedy Selection**: Efficiently selects k actions maximizing submodular objective
- **Inference Integration**: Routes selected actions into LLM prompts during reasoning

## Implementation Steps

**Step 1: Extract Action Sketches from Corpus**

Analyze diverse reasoning problems to identify common solution patterns.

```python
def extract_action_sketches(problem_corpus: List[str], llm_api) -> List[str]:
    """
    Extract high-level action patterns from diverse reasoning problems.

    Args:
        problem_corpus: List of reasoning problem descriptions
        llm_api: Language model API for analysis

    Returns:
        action_sketches: List of general solution strategy descriptions
    """
    prompt_template = """Analyze this reasoning problem and identify the key
solution strategy or action type needed:

Problem: {problem}

What high-level action or strategy would solve this?
Examples: 'decompose into subproblems', 'search through possibilities',
'construct a proof', 'optimize a sequence', etc.

Respond with only the action description."""

    action_sketches = set()

    # Sample diverse problems to discover action types
    sampled_problems = random.sample(
        problem_corpus,
        min(1000, len(problem_corpus))
    )

    for problem in sampled_problems:
        prompt = prompt_template.format(problem=problem)
        sketch = llm_api.generate(prompt, max_tokens=50)
        action_sketches.add(sketch.strip())

    return list(action_sketches)

def estimate_action_space(action_sketches: List[str]) -> Dict[str, List[str]]:
    """
    Build structured representation of estimated action space.

    Args:
        action_sketches: Extracted action strategy descriptions

    Returns:
        action_space: Dictionary mapping action types to implementations
    """
    action_categories = {
        'decomposition': [],
        'search': [],
        'construction': [],
        'optimization': [],
        'verification': []
    }

    for sketch in action_sketches:
        sketch_lower = sketch.lower()

        # Categorize sketches
        if any(word in sketch_lower for word in ['decompose', 'split', 'break']):
            action_categories['decomposition'].append(sketch)
        elif any(word in sketch_lower for word in ['search', 'find', 'explore']):
            action_categories['search'].append(sketch)
        elif any(word in sketch_lower for word in ['construct', 'build', 'create']):
            action_categories['construction'].append(sketch)
        elif any(word in sketch_lower for word in ['optimize', 'minimize', 'maximize']):
            action_categories['optimization'].append(sketch)
        elif any(word in sketch_lower for word in ['verify', 'check', 'validate']):
            action_categories['verification'].append(sketch)

    return action_categories
```

**Step 2: Define Submodular Objective Function**

Create an objective that balances relevance and diversity in action selection.

```python
import numpy as np
from typing import Callable

class SubmodularActionObjective:
    """
    Submodular function measuring quality of action subset.
    """

    def __init__(self, all_actions: List[str], llm_relevance_fn: Callable):
        """
        Args:
            all_actions: Complete set of candidate actions
            llm_relevance_fn: Function computing action relevance to problem
        """
        self.all_actions = all_actions
        self.relevance_fn = llm_relevance_fn

        # Precompute action embeddings for diversity
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.encoder.encode(all_actions)

    def compute_relevance(self, problem: str, actions: List[str]) -> np.ndarray:
        """
        Compute relevance scores: how well each action fits the problem.

        Args:
            problem: Current problem description
            actions: Candidate actions

        Returns:
            scores: Relevance scores [0, 1]
        """
        scores = []

        # Could use LLM, but efficient approximation: check keyword overlap
        problem_keywords = set(problem.lower().split())

        for action in actions:
            action_keywords = set(action.lower().split())
            overlap = len(problem_keywords & action_keywords)
            score = overlap / max(len(problem_keywords), 1)
            scores.append(score)

        return np.array(scores)

    def compute_diversity(self, selected_actions_idx: List[int]) -> float:
        """
        Measure diversity of selected actions (average pairwise distance).

        Args:
            selected_actions_idx: Indices of selected actions

        Returns:
            diversity_score: Higher means more diverse [0, 1]
        """
        if len(selected_actions_idx) <= 1:
            return 0.0

        diversity = 0.0
        selected_embeddings = self.embeddings[selected_actions_idx]

        # Average pairwise cosine distance
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(selected_embeddings)
        diversity = distances[np.triu_indices_from(distances, k=1)].mean()

        return diversity

    def evaluate_subset(self, problem: str, subset_indices: List[int],
                        relevance_weight: float = 0.7,
                        diversity_weight: float = 0.3) -> float:
        """
        Evaluate quality of action subset for given problem.

        Args:
            problem: Problem description
            subset_indices: Indices of selected actions
            relevance_weight: Weight for relevance term
            diversity_weight: Weight for diversity term

        Returns:
            objective_value: Combined score [0, 1]
        """
        subset_actions = [self.all_actions[i] for i in subset_indices]

        # Compute relevance of subset (average relevance)
        relevance_scores = self.compute_relevance(problem, subset_actions)
        avg_relevance = relevance_scores.mean()

        # Compute diversity of subset
        diversity = self.compute_diversity(subset_indices)

        # Combined objective (submodular)
        objective = (relevance_weight * avg_relevance +
                    diversity_weight * diversity)

        return objective
```

**Step 3: Greedy Selection Algorithm**

Implement greedy selection to maximize submodular objective.

```python
def greedy_action_selection(problem: str, objective: SubmodularActionObjective,
                           k: int = 5, candidate_actions: List[str] = None) -> List[str]:
    """
    Greedily select k actions maximizing submodular objective.

    Args:
        problem: Current problem description
        objective: SubmodularActionObjective instance
        k: Number of actions to select
        candidate_actions: Actions to consider (or use objective.all_actions)

    Returns:
        selected_actions: Top-k selected action descriptions
    """
    if candidate_actions is None:
        candidate_actions = objective.all_actions

    # Indices of all candidate actions
    candidate_indices = list(range(len(candidate_actions)))

    # Greedy selection
    selected_indices = []

    for step in range(k):
        best_idx = None
        best_gain = -float('inf')

        # Try adding each remaining action
        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            # Evaluate objective with this action added
            candidate_subset = selected_indices + [idx]
            value = objective.evaluate_subset(problem, candidate_subset)

            # Marginal gain (submodular property)
            if selected_indices:
                current_value = objective.evaluate_subset(problem, selected_indices)
                gain = value - current_value
            else:
                gain = value

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)

    # Return selected actions
    selected_actions = [candidate_actions[i] for i in selected_indices]
    return selected_actions
```

**Step 4: Action Space Augmented Reasoning**

Integrate selected actions into LLM prompts during inference.

```python
def construct_reasoning_prompt_with_actions(
        problem: str,
        selected_actions: List[str],
        reasoning_context: str = "") -> str:
    """
    Build prompt with dynamic action space.

    Args:
        problem: Problem to reason about
        selected_actions: Curated actions for this problem
        reasoning_context: Prior reasoning steps (if any)

    Returns:
        augmented_prompt: Prompt with available actions
    """
    prompt = f"""Solve this problem using the following available actions:

Problem: {problem}

Available actions:
"""

    for i, action in enumerate(selected_actions, 1):
        prompt += f"{i}. {action}\n"

    if reasoning_context:
        prompt += f"\nCurrent progress: {reasoning_context}\n"

    prompt += "\nChoose the best action to take next and explain your reasoning."

    return prompt

def run_reasoning_with_dynamic_actions(
        problem: str,
        objective: SubmodularActionObjective,
        llm_api,
        max_steps: int = 10,
        k_actions: int = 5) -> Dict[str, Any]:
    """
    Execute multi-step reasoning with dynamically selected actions.

    Args:
        problem: Problem to solve
        objective: SubmodularActionObjective instance
        llm_api: Language model for reasoning
        max_steps: Maximum reasoning steps
        k_actions: Number of actions to select per step

    Returns:
        reasoning_result: {final_answer, steps, actions_used}
    """
    reasoning_context = ""
    reasoning_steps = []
    actions_used = []

    for step in range(max_steps):
        # Select actions for current state
        selected_actions = greedy_action_selection(
            problem + " " + reasoning_context,
            objective,
            k=k_actions
        )
        actions_used.append(selected_actions)

        # Construct prompt with actions
        prompt = construct_reasoning_prompt_with_actions(
            problem, selected_actions, reasoning_context
        )

        # Get LLM reasoning step
        response = llm_api.generate(prompt, max_tokens=500)
        reasoning_steps.append(response)

        # Update context for next step
        reasoning_context += "\n" + response

        # Check for termination (simple heuristic)
        if "final answer" in response.lower() or step == max_steps - 1:
            break

    return {
        'reasoning_steps': reasoning_steps,
        'actions_used': actions_used,
        'final_reasoning': reasoning_context
    }
```

## Practical Guidance

**When to Use DynaAct:**
- Reasoning tasks with diverse solution strategies (action diversity matters)
- Scenarios with large potential action spaces (computational efficiency needed)
- Applications where actions should adapt to problem characteristics

**When NOT to Use:**
- Tasks with small, well-defined action sets (manual specification sufficient)
- Real-time systems with strict latency constraints (action selection adds overhead)
- Domains without diverse reasoning corpus (corpus-based estimation fails)

**Hyperparameters and Configuration:**
- k (number of actions): 5-10 for most tasks; increase for complex problems
- Relevance weight: 0.6-0.8 (prioritize applicable actions)
- Diversity weight: 0.2-0.4 (balance with relevance)
- Corpus size: 1000+ problems for robust action extraction

**Pitfalls to Avoid:**
1. **Poor corpus coverage** - Action sketches only as good as training corpus; use diverse, representative problems
2. **Relevance function brittleness** - Keyword-based relevance is naive; consider embedding-based similarity
3. **Over-selection** - Selecting too many actions (large k) dilutes prompt; use k ≤ 10
4. **Stale action space** - If problem distribution shifts, retrain action sketches; monitor performance decline

---

Reference: https://arxiv.org/abs/2511.08043
