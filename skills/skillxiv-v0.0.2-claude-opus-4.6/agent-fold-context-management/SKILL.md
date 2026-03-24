---
name: agent-fold-context-management
title: "AgentFold: Long-Horizon Web Agents with Proactive Context Management"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24699"
keywords: [Web Agent, Context Management, Long-horizon Tasks, Memory, Consolidation]
description: "Enables web agents to handle long-horizon tasks by actively managing context workspace. Implements granular condensations of recent steps and deep consolidations of multi-step sub-tasks, preventing context saturation. Achieves 36.2% on BrowseComp with 30B model, matching larger proprietary agents."
---

# AgentFold: Cognitive Context Management for Web Agents

Long-horizon web tasks accumulate verbose interaction histories, causing context saturation and degraded agent reasoning. AgentFold treats context as a dynamic workspace to be actively sculpted, not passively filled.

By implementing retrospective consolidation inspired by human cognition, agents maintain rich but manageable context across complex multi-step tasks.

## Core Concept

Key insight: **actively compress and consolidate context at multiple scales**:
- Granular condensations: preserve fine-grained details from recent steps
- Deep consolidations: abstract multi-step sub-tasks into summaries
- Dynamic folding: apply consolidation strategically to prevent saturation
- Retrospective processing: summarize after task completion

## Architecture Overview

- Multi-scale context compression (recent details + old abstractions)
- Step-level granular summaries
- Task-level deep consolidations
- Context relevance scoring for selective retention

## Implementation Steps

Implement granular condensation that summarizes recent interactions concisely:

```python
class GranularCondenser:
    def __init__(self, llm):
        self.llm = llm

    def condense_recent_steps(self, recent_interactions, max_steps=5):
        """Create concise summary of recent N steps."""
        if len(recent_interactions) <= max_steps:
            return recent_interactions  # Keep as-is if small

        # Summarize each step briefly
        condensed = []
        for interaction in recent_interactions[-max_steps:]:
            action = interaction['action']
            observation = interaction['observation']

            # Extract key facts (50 token summary)
            summary = self.llm.summarize(
                f"Action: {action}\nObservation: {observation}",
                max_tokens=50
            )

            condensed.append({
                'timestamp': interaction['timestamp'],
                'action_type': self._classify_action(action),
                'summary': summary,
                'key_facts': self._extract_facts(observation)
            })

        return condensed

    def _classify_action(self, action):
        """Classify action type (click, type, scroll, etc)."""
        keywords = {
            'click': ['click', 'submit', 'select'],
            'type': ['type', 'input', 'write'],
            'scroll': ['scroll', 'navigate'],
            'wait': ['wait', 'pause']
        }

        for action_type, keywords_list in keywords.items():
            if any(kw in action.lower() for kw in keywords_list):
                return action_type
        return 'other'

    def _extract_facts(self, observation):
        """Extract salient facts from observation."""
        # Simple extraction: headings, text > 20 chars, form fields
        facts = []
        lines = observation.split('\n')
        for line in lines:
            if len(line) > 20 or any(char.isupper() for char in line):
                facts.append(line)
        return facts[:3]  # Top 3 facts
```

Implement deep consolidation that summarizes completed sub-tasks:

```python
class DeepConsolidator:
    def __init__(self, llm):
        self.llm = llm

    def consolidate_subtask(self, subtask_history, subtask_goal):
        """Create abstract summary of completed subtask."""
        # Collect all interactions in subtask
        all_actions = "\n".join([
            f"{i}. {h['action']}" for i, h in enumerate(subtask_history)
        ])

        # Generate consolidation
        prompt = f"""
Summarize what was accomplished in this subtask in 2-3 sentences.
Goal: {subtask_goal}

Steps taken:
{all_actions}

Consolidation:
"""

        consolidation = self.llm.generate(prompt, max_tokens=100)

        return {
            'goal': subtask_goal,
            'status': self._extract_status(consolidation),
            'summary': consolidation,
            'key_outcome': self._extract_outcome(consolidation)
        }

    def _extract_status(self, consolidation):
        """Extract whether subtask succeeded/failed."""
        if any(word in consolidation.lower() for word in ['success', 'completed', 'achieved']):
            return 'completed'
        elif any(word in consolidation.lower() for word in ['failed', 'unable', 'error']):
            return 'failed'
        return 'partial'

    def _extract_outcome(self, consolidation):
        """Extract main outcome."""
        # Simple: first sentence
        return consolidation.split('.')[0]
```

Implement the folding mechanism that applies consolidation dynamically:

```python
class AgentContextFolder:
    def __init__(self, llm, max_context_length=4096):
        self.llm = llm
        self.max_context_length = max_context_length
        self.granular_condenser = GranularCondenser(llm)
        self.deep_consolidator = DeepConsolidator(llm)

    def fold_context(self, full_history, current_task_progress):
        """Actively manage context workspace."""
        context_tokens = self._estimate_tokens(full_history)

        # Check if folding needed
        if context_tokens < self.max_context_length * 0.7:
            return full_history  # Plenty of space

        # Identify subtasks to consolidate
        subtasks = self._identify_subtasks(full_history)

        # Apply deep consolidation to completed subtasks
        consolidated_subtasks = []
        for subtask in subtasks:
            if subtask['completed']:
                cons = self.deep_consolidator.consolidate_subtask(
                    subtask['history'],
                    subtask['goal']
                )
                consolidated_subtasks.append(cons)
            else:
                # Keep incomplete subtasks detailed
                consolidated_subtasks.append(subtask)

        # Apply granular condensation to recent interactions
        recent_interactions = full_history[-10:]
        condensed_recent = self.granular_condenser.condense_recent_steps(
            recent_interactions, max_steps=5
        )

        # Reconstruct context
        folded_context = {
            'consolidated_subtasks': consolidated_subtasks,
            'recent_interactions': condensed_recent,
            'current_goal': current_task_progress
        }

        return folded_context

    def _estimate_tokens(self, context):
        """Rough token count estimate."""
        if isinstance(context, dict):
            context = str(context)
        return len(context) // 4  # Approximate: 1 token ≈ 4 chars

    def _identify_subtasks(self, history):
        """Parse history into logical subtasks."""
        # Simple: detect when major action types change
        subtasks = []
        current_subtask = {'history': [], 'goal': '', 'completed': False}

        for interaction in history:
            if self._is_subtask_boundary(interaction):
                subtasks.append(current_subtask)
                current_subtask = {'history': [], 'goal': '', 'completed': False}
            current_subtask['history'].append(interaction)

        if current_subtask['history']:
            subtasks.append(current_subtask)

        return subtasks

    def _is_subtask_boundary(self, interaction):
        """Detect subtask completion boundaries."""
        # Simple heuristic: major action changes or explicit goal completion
        action = interaction.get('action', '').lower()
        return any(boundary in action for boundary in ['navigate', 'submit', 'return'])
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Max context length | 4096-8192 tokens |
| Granular condensation depth | Last 5-10 steps |
| Consolidation trigger | 70% context capacity |
| Subtask window | 20-50 interactions |

**When to use:**
- Long-horizon web navigation tasks
- Scenarios with complex multi-step workflows
- Memory-constrained deployments (limited context window)
- Tasks requiring backtracking or revisiting earlier states

**When NOT to use:**
- Short-horizon tasks (<10 steps)
- Tasks requiring verbatim historical information
- Real-time systems (folding adds latency)

**Common pitfalls:**
- Over-aggressive consolidation (losing critical details)
- Consolidation threshold too low (constant folding overhead)
- Not preserving recent details (old decisions matter)
- Subtask boundaries misidentified (logical coherence lost)

Reference: [AgentFold on arXiv](https://arxiv.org/abs/2510.24699)
