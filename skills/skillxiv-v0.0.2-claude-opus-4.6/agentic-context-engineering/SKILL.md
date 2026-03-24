---
name: agentic-context-engineering
title: "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.04618"
keywords: [Context Evolution, Self-Improvement, Agent Learning, Prompt Engineering, Delta Updates]
description: "Evolve agent behavior through iterative context refinement using delta updates rather than full rewrites, accumulating strategies and insights across execution traces."
---

# Technique: Incremental Context Evolution for Autonomous Agents

Traditional few-shot prompts are static—they capture knowledge at creation time but don't evolve as the agent learns. Agentic Context Engineering treats prompts as evolving playbooks that accumulate and refine strategies through execution. Rather than storing knowledge in model parameters (expensive fine-tuning), knowledge accumulates in detailed, structured contexts that grow and refine over time.

The key innovation is delta-based updates: instead of rewriting contexts from scratch (which causes context collapse), the system makes small, localized edits to itemized bullets representing strategies, concepts, or failure modes. This preserves detailed knowledge while enabling efficient parallel merging of multiple updates.

## Core Concept

ACE employs three specialized roles working iteratively:

1. **Generator**: Produces reasoning trajectories that surface effective strategies and common failure points
2. **Reflector**: Extracts concrete insights from successes and errors across refinement rounds
3. **Curator**: Integrates insights into structured, incremental delta updates

The system balances steady expansion with redundancy control through semantic deduplication, preventing context bloat while maintaining knowledge richness.

## Architecture Overview

- **Initial Context**: Starting prompt with base instructions
- **Generation Phase**: Agent executes tasks using current context, collecting trajectories
- **Reflection Phase**: Analyze trajectories to identify improvements
- **Delta Computation**: Create minimal edits (add/remove/modify bullets) rather than full rewrites
- **Curation Phase**: Merge deltas while removing semantic duplicates
- **Updated Context**: Refined prompt ready for next iteration

## Implementation Steps

Implement the context representation as structured bullets.

```python
class ContextBullet:
    def __init__(self, bullet_type, content, rationale=''):
        self.type = bullet_type  # 'strategy', 'error_pattern', 'tool_tip'
        self.content = content
        self.rationale = rationale
        self.added_at_round = 0
        self.usage_count = 0

    def __repr__(self):
        return f"[{self.type}] {self.content}"


class EvolvingContext:
    def __init__(self, base_instructions):
        self.base_instructions = base_instructions
        self.bullets = []
        self.round = 0

    def to_prompt(self):
        """Convert structured context back to text prompt."""
        prompt = self.base_instructions + "\n\nLearned Strategies:\n"

        grouped = {}
        for bullet in self.bullets:
            if bullet.type not in grouped:
                grouped[bullet.type] = []
            grouped[bullet.type].append(bullet)

        for bullet_type, bullets in grouped.items():
            prompt += f"\n{bullet_type.upper()}:\n"
            for bullet in bullets:
                prompt += f"- {bullet.content}\n"

        return prompt

    def add_bullet(self, bullet_type, content, rationale=''):
        """Add a new bullet to the context."""
        new_bullet = ContextBullet(bullet_type, content, rationale)
        new_bullet.added_at_round = self.round
        self.bullets.append(new_bullet)

    def remove_bullets_by_similarity(self, new_content, threshold=0.9):
        """Remove duplicates when adding similar content."""
        from sentence_transformers import util
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        new_embedding = model.encode(new_content, convert_to_tensor=True)

        to_remove = []
        for bullet in self.bullets:
            existing_embedding = model.encode(bullet.content,
                                             convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(new_embedding,
                                             existing_embedding)[0][0].item()

            if similarity > threshold:
                to_remove.append(bullet)

        for bullet in to_remove:
            self.bullets.remove(bullet)
```

Implement the Reflector that analyzes trajectories.

```python
def reflect_on_trajectories(trajectories, llm_model, context):
    """
    Analyze execution trajectories to extract improvements.

    Args:
        trajectories: List of (actions, observations, success) tuples
        llm_model: Language model for reflection
        context: Current EvolvingContext

    Returns:
        improvements: Dict with 'strategies', 'error_patterns', 'tool_tips'
    """

    # Format trajectories for analysis
    trajectory_text = format_trajectories(trajectories)

    reflection_prompt = f"""Analyze these execution traces and extract improvements:

{trajectory_text}

Current context:
{context.to_prompt()}

What new strategies, error patterns, or tool tips should be added?
Format as:
STRATEGY: [description]
ERROR_PATTERN: [description]
TOOL_TIP: [description]
"""

    reflection = llm_model.generate(reflection_prompt)

    # Parse structured improvements
    improvements = parse_reflection(reflection)

    return improvements


def format_trajectories(trajectories):
    """Format trajectories for LLM analysis."""
    formatted = []

    for i, (actions, observations, success) in enumerate(trajectories):
        formatted.append(f"\nTrace {i+1} (Success: {success}):")
        for action in actions:
            formatted.append(f"  Action: {action}")
        for obs in observations:
            formatted.append(f"  Observation: {obs}")

    return "\n".join(formatted)


def parse_reflection(reflection_text):
    """Extract structured improvements from reflection."""
    improvements = {
        'strategies': [],
        'error_patterns': [],
        'tool_tips': []
    }

    lines = reflection_text.split('\n')
    current_type = None

    for line in lines:
        if 'STRATEGY:' in line:
            current_type = 'strategies'
            content = line.split('STRATEGY:')[1].strip()
            if content:
                improvements['strategies'].append(content)
        elif 'ERROR_PATTERN:' in line:
            current_type = 'error_patterns'
            content = line.split('ERROR_PATTERN:')[1].strip()
            if content:
                improvements['error_patterns'].append(content)
        elif 'TOOL_TIP:' in line:
            current_type = 'tool_tips'
            content = line.split('TOOL_TIP:')[1].strip()
            if content:
                improvements['tool_tips'].append(content)
        elif current_type and line.strip() and not ':' in line:
            # Continuation of previous type
            improvements[current_type][-1] += ' ' + line.strip()

    return improvements
```

Implement the Curator that applies delta updates.

```python
def apply_delta_updates(context, improvements, llm_model):
    """
    Curate improvements and apply as delta updates to context.

    Args:
        context: EvolvingContext to update
        improvements: Dict of strategies, error_patterns, tool_tips
        llm_model: LLM for deduplication decisions

    Returns:
        updated_context: Context with delta updates applied
    """

    for bullet_type, items in improvements.items():
        for content in items:
            # Check for semantic duplicates before adding
            context.remove_bullets_by_similarity(content, threshold=0.85)

            # Add new bullet
            context.add_bullet(bullet_type, content)

    context.round += 1

    return context
```

Implement the full ACE iteration loop.

```python
def ace_iteration(agent, context, tasks, llm_model, num_iterations=3):
    """
    Run full Agentic Context Engineering iteration.

    Args:
        agent: Agent policy
        context: EvolvingContext
        tasks: List of tasks to execute
        llm_model: Language model for reflection
        num_iterations: Number of refinement cycles

    Returns:
        evolved_context: Context after iterations
        metrics: Performance metrics
    """

    metrics = {'round': [], 'success_rate': [], 'context_size': []}

    for iteration in range(num_iterations):
        # GENERATE: Execute tasks with current context
        trajectories = []

        for task in tasks:
            current_prompt = context.to_prompt()
            actions = []
            observations = []
            success = False

            try:
                # Execute task using agent with evolved context
                result = agent.execute_task(task, current_prompt)
                actions = result.get('actions', [])
                observations = result.get('observations', [])
                success = result.get('success', False)

                trajectories.append((actions, observations, success))

            except Exception as e:
                print(f"Task execution failed: {e}")
                trajectories.append(([], [str(e)], False))

        # REFLECT: Analyze trajectories
        improvements = reflect_on_trajectories(trajectories, llm_model, context)

        # MEMORIZE: Apply delta updates
        context = apply_delta_updates(context, improvements, llm_model)

        # Track metrics
        success_rate = sum(1 for _, _, s in trajectories) / len(trajectories)
        metrics['round'].append(iteration)
        metrics['success_rate'].append(success_rate)
        metrics['context_size'].append(len(context.bullets))

        print(f"Round {iteration+1}: Success={success_rate:.2%}, "
              f"Context size={len(context.bullets)} bullets")

    return context, metrics
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Deduplication threshold | 0.85-0.95 | Higher = more aggressive deduplication; lower = preserves nuance |
| Bullet types | 5-10 categories | Strategy, error_pattern, tool_tip, domain_knowledge, etc. |
| Context refresh | Every 10-50 iterations | Prevent unbounded growth; periodic consolidation |
| Reflection LLM | Smaller than agent | Reflection can be cheaper; offload to smaller model |
| When to use | Long-running agents on task sequences | Continuous learning scenarios |
| When NOT to use | One-off tasks or static environments | Context evolution overhead not justified |
| Common pitfall | Context becomes too large or repetitive | Enforce regular consolidation and deduplication |

### When to Use ACE

- Agents operating over task sequences where patterns repeat
- Self-improvement scenarios without fine-tuning
- Multi-turn interactions where knowledge accumulation is valuable
- Research or planning tasks with evolving strategy sets

### When NOT to Use ACE

- Single-task agents where static prompts suffice
- Real-time systems where reflection latency is prohibitive
- Domains with rapidly shifting strategies

### Common Pitfalls

- **Context bloat**: Enforce max context size; periodically consolidate old bullets
- **Generic strategies**: Reflection produces vague improvements; require concrete examples
- **Semantic drift**: Regular validation that bullets remain coherent and non-contradictory
- **Inefficient deduplication**: Use embedding-based similarity; avoid string matching

## Reference

Paper: https://arxiv.org/abs/2510.04618
