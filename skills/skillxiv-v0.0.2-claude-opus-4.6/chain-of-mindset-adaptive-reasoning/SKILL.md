---
name: chain-of-mindset-adaptive-reasoning
title: "Chain of Mindset: Reasoning with Adaptive Cognitive Modes"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.10063"
keywords: [Adaptive Reasoning, Multi-Modal Thinking, Step-Level Orchestration, Training-Free, Meta-Agent]
description: "Enable language models to dynamically switch between four cognitive modes (spatial, convergent, divergent, algorithmic) during problem-solving. Meta-agent observes state and selects optimal mode per step, improving reasoning across math, coding, and spatial tasks without requiring model training."
---

# Chain of Mindset: Adaptive Multi-Modal Reasoning

Single problem-solving strategies fail across diverse task types. Mathematical derivations need convergent logic; creative ideation needs divergent exploration; spatial tasks need visualization; calculations need algorithmic precision. Chain of Mindset (CoM) enables training-free dynamic mode switching, where the model selects appropriate cognitive approaches per reasoning step rather than committing upfront to one strategy.

A Meta-Agent observes reasoning progress and decides which of four specialized mindsets to engage next: Spatial (visualization), Convergent (focused analysis), Divergent (parallel exploration), Algorithmic (precise calculation). Context Gates prevent information pollution between modes while maintaining efficiency.

## Core Concept

Standard approach: apply single reasoning strategy throughout problem. Inefficient for multi-faceted problems.

CoM approach: decompose problem into steps, use Meta-Agent to assign optimal cognitive mode per step. Modes collaborate asynchronously: spatial mode generates diagrams, convergent mode analyzes alternatives, divergent mode explores options, algorithmic mode executes calculations.

Key insight: problem-solving naturally decomposes into stages requiring different cognitive approaches. No training needed—just better prompting structure.

## Architecture Overview

- **Four Cognitive Modes**:
  - Spatial: Visualize abstract concepts through diagrams, spatial relationships
  - Convergent: Focused logical analysis on specific questions
  - Divergent: Parallel exploration of multiple solution paths
  - Algorithmic: Precise calculations, symbolic manipulation, code

- **Meta-Agent**: Observes current reasoning state, selects appropriate next mode
- **Context Gates**: Filter information flow between modes, prevent context pollution
- **Step-Level Orchestration**: Dynamically revise reasoning plan when intermediate results suggest better paths
- **Training-Free**: All logic implemented via prompting; no model updates

## Implementation

Define mode-specific prompts and context management:

```python
MODE_PROMPTS = {
    'spatial': """You are a spatial reasoning mode. Visualize the problem using diagrams,
coordinate systems, or visual representations. Describe spatial relationships, positions, and geometric patterns.
Output: ASCII diagram or detailed spatial description.""",

    'convergent': """You are a convergent reasoning mode. Focus on rigorous logical analysis.
Identify constraints, dependencies, and hierarchical relationships. Reason systematically from axioms.
Output: Detailed logical derivation.""",

    'divergent': """You are a divergent reasoning mode. Explore multiple solution paths in parallel.
Generate alternative approaches, creative interpretations, and unconventional methods.
Output: List of distinct solution strategies.""",

    'algorithmic': """You are an algorithmic reasoning mode. Execute precise calculations,
formal procedures, and symbolic manipulations. Use mathematical notation and step-by-step computation.
Output: Numerical result or formal proof."""
}

def select_cognitive_mode(reasoning_state, meta_agent_prompt=None):
    """
    Meta-agent selects appropriate cognitive mode based on reasoning state.
    Args:
        reasoning_state: Current reasoning progress (text)
        meta_agent_prompt: Optional custom selection prompt
    Returns:
        selected_mode: One of ['spatial', 'convergent', 'divergent', 'algorithmic']
    """
    if meta_agent_prompt is None:
        meta_agent_prompt = """Given the current reasoning state below, which cognitive mode would be most helpful?
- Spatial: For visualizing patterns, geometric relationships, or spatial structures
- Convergent: For focused logical derivation and constraint satisfaction
- Divergent: For exploring alternative approaches and creative solutions
- Algorithmic: For precise calculations and symbolic manipulation

Current state:
{state}

Which mode? Answer with exactly one word: spatial, convergent, divergent, or algorithmic."""

    # Query model to select mode
    response = query_model(
        meta_agent_prompt.format(state=reasoning_state)
    )

    # Parse response
    modes = ['spatial', 'convergent', 'divergent', 'algorithmic']
    for mode in modes:
        if mode in response.lower():
            return mode

    return 'convergent'  # Default fallback

class ContextGate:
    """Filters context between cognitive modes."""

    def __init__(self):
        self.mode_contexts = {}

    def filter_for_mode(self, full_context, target_mode):
        """Extract relevant context for specific mode."""
        # Simple filtering: remove incompatible information
        if target_mode == 'spatial':
            # Keep spatial/visual information, remove algorithmic details
            filtered = '\n'.join([
                line for line in full_context.split('\n')
                if not any(x in line.lower() for x in ['formula', 'equation', 'calculate'])
            ])
        elif target_mode == 'algorithmic':
            # Keep symbolic information, remove visual descriptions
            filtered = '\n'.join([
                line for line in full_context.split('\n')
                if not any(x in line.lower() for x in ['visualize', 'diagram', 'imagine'])
            ])
        else:
            filtered = full_context

        return filtered

class AdaptiveMindsetReasoner:
    """Chain of Mindset reasoning orchestrator."""

    def __init__(self, base_model, max_iterations=10):
        self.model = base_model
        self.max_iterations = max_iterations
        self.context_gate = ContextGate()
        self.full_reasoning = ""
        self.mode_history = []

    def reason_adaptively(self, problem):
        """Execute adaptive reasoning with dynamic mode selection."""
        reasoning = f"Problem: {problem}\n\n"

        for iteration in range(self.max_iterations):
            # Meta-agent selects next mode
            selected_mode = select_cognitive_mode(reasoning)
            self.mode_history.append(selected_mode)

            # Filter context for this mode
            filtered_context = self.context_gate.filter_for_mode(reasoning, selected_mode)

            # Generate reasoning step in selected mode
            mode_prompt = MODE_PROMPTS[selected_mode]
            full_prompt = f"{mode_prompt}\n\nCurrent reasoning:\n{filtered_context}\n\nNext step:"

            mode_output = self.model.generate(full_prompt, max_tokens=200)
            reasoning += f"\n[{selected_mode.upper()}]: {mode_output}"

            # Check for completion
            if any(x in mode_output.lower() for x in ['answer:', 'solution:', 'therefore:']):
                break

        self.full_reasoning = reasoning
        return reasoning

    def extract_final_answer(self):
        """Extract final answer from reasoning trace."""
        import re

        # Look for explicit answer statements
        patterns = [
            r'answer:\s*(.+?)(?:\n|$)',
            r'solution:\s*(.+?)(?:\n|$)',
            r'therefore,\s*(.+?)(?:\n|$)',
            r'result:\s*(.+?)(?:\n|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, self.full_reasoning, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return last non-empty line
        lines = [l.strip() for l in self.full_reasoning.split('\n') if l.strip()]
        return lines[-1] if lines else ""
```

Usage example:

```python
# Initialize reasoner
model = load_language_model("gpt-4")
reasoner = AdaptiveMindsetReasoner(model)

# Complex problem requiring multiple cognitive modes
problem = """A farmer has a rectangular field divided into 4 equal quadrants.
He wants to plant crops such that adjacent quadrants have different crop types.
He has 3 crop types. How many distinct planting patterns exist if rotations and
reflections are considered the same?"""

# Adaptive reasoning with dynamic mode selection
reasoning_trace = reasoner.reason_adaptively(problem)
print(reasoning_trace)
print("\nMode sequence:", reasoner.mode_history)

final_answer = reasoner.extract_final_answer()
print("\nFinal Answer:", final_answer)
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Max iterations | 8-15 | More iterations explore thoroughly; cap to avoid repetition. |
| Mode diversity | Encourage all 4 modes | Don't let meta-agent favor one mode; track usage. |
| Context filtering strictness | Moderate | Balance focus with information retention. |
| Integration point | After problem statement | Let model choose modes rather than prescribing strategy. |

**When to Use**
- Complex problems requiring multiple reasoning styles (math, coding, spatial, creative)
- Situations where single strategy hits local optimum
- Benchmarks across diverse domains (mixed reasoning requirements)
- Training-free inference (no retraining of base model)

**When NOT to Use**
- Simple single-type problems (overhead not justified)
- Models that struggle with complex instruction following
- Domains with single dominant reasoning mode

**Common Pitfalls**
- Meta-agent always selects same mode; use diversification prompts
- Context gates too aggressive (removes useful information); test filtering
- Not monitoring mode history; divergent mode should alternate with convergent
- Context explosion; periodically summarize reasoning to avoid token bloat

## Reference

See https://arxiv.org/abs/2602.10063 for full prompt templates, detailed mode descriptions, meta-agent implementation, and benchmarks across mathematics (AIME, MATH), coding (HumanEval), and spatial reasoning (Rotate3D).
