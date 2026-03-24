---
name: memento-skills-agent-design
title: "Memento-Skills: Let Agents Design Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.18743"
keywords: [Agent Architecture, Skill Learning, Continual Adaptation, Prompt Engineering]
description: "Enable agents to autonomously design and refine task-specific agents by evolving externalised behavioral skills and prompts without modifying base LLM parameters."
---

# Memento-Skills: Continual Agent Self-Improvement

Production AI agents face a core limitation: they are static. Once deployed, they cannot adapt their decision-making logic without retraining. Memento-Skills solves this through a memory-based architecture where agents evolve reusable behavioral skills—stored as structured markdown—without touching the underlying LLM.

The key innovation is recognizing that agent knowledge should be externalized and malleable. Rather than training new models, agents build and refine a library of skills (mini-prompts encoding behavioral patterns) that guide decision-making. This enables continual improvement: each task failure generates new skills, and each success validates existing ones.

## Core Concept

Memento-Skills implements agent self-design through stateful prompts and evolving skill libraries:

**Read Phase:** When facing a task, the agent uses a trainable skill router to select relevant skills based on context. These skills are appended to the prompt, modifying behavior without parameter updates.

**Write Phase:** After task execution, the agent reflects on outcomes and synthesizes new skills or refines existing ones based on what worked.

**No Parameter Updates:** All adaptation occurs through externalizing knowledge into markdown files and prompt engineering.

The system tracks which skills help on which tasks, enabling a generalist agent to dynamically specialize for each problem.

## Architecture Overview

- **Skill Library**: Markdown files storing behavioral patterns, context requirements, execution logic
- **Stateful Prompt**: Main system prompt that incorporates selected skills dynamically
- **Skill Router**: Trainable selector that chooses appropriate skills based on task context
- **Reflective Learning**: Post-execution analysis that generates new skills or updates weights
- **Agent Generator**: LLM-based tool that designs task-specific agent configurations
- **No Gradient-Based Training**: Learning happens through prompt construction and skill evolution

## Implementation Steps

### Step 1: Define Skill Representation

Structure skills as executable markdown templates.

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import hashlib

@dataclass
class Skill:
    """
    Reusable behavioral skill stored as markdown template.
    Skills encode decision logic without modifying model parameters.
    """
    name: str
    description: str  # What does this skill do?
    context_requirements: List[str]  # When should this skill activate?
    execution_logic: str  # How to use this skill (LLM-executable)
    success_metrics: Dict[str, float]  # Performance on various tasks
    created_timestamp: str
    last_used: Optional[str] = None
    usage_count: int = 0
    success_rate: float = 0.0

    def to_markdown(self) -> str:
        """Convert skill to markdown for inclusion in prompts."""
        return f"""
## Skill: {self.name}

**Description**: {self.description}

**When to use**: {', '.join(self.context_requirements)}

**How to execute**:
```
{self.execution_logic}
```

**Performance**: Success rate {self.success_rate*100:.1f}% ({self.usage_count} uses)
"""

    def to_dict(self) -> dict:
        """Serialize skill for storage."""
        return {
            'name': self.name,
            'description': self.description,
            'context_requirements': self.context_requirements,
            'execution_logic': self.execution_logic,
            'success_metrics': self.success_metrics,
            'created_timestamp': self.created_timestamp,
            'last_used': self.last_used,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        return cls(**data)

class SkillLibrary:
    """Manage collection of reusable skills."""

    def __init__(self, storage_path: str = "./skills"):
        self.skills: Dict[str, Skill] = {}
        self.storage_path = storage_path
        self.load_from_disk()

    def add_skill(self, skill: Skill):
        """Add skill to library."""
        self.skills[skill.name] = skill
        self._save_skill(skill)

    def update_skill_performance(self, skill_name: str, success: bool):
        """Update skill success metrics after execution."""
        if skill_name not in self.skills:
            return

        skill = self.skills[skill_name]
        skill.usage_count += 1
        skill.last_used = datetime.now().isoformat()

        # Update success rate with exponential smoothing
        alpha = 0.3
        new_success = float(success)
        skill.success_rate = alpha * new_success + (1 - alpha) * skill.success_rate

    def get_relevant_skills(self, context: Dict) -> List[Skill]:
        """Retrieve skills matching current task context."""
        relevant = []

        for skill in self.skills.values():
            # Check if skill's context requirements match
            match_count = 0
            for req in skill.context_requirements:
                if req.lower() in str(context).lower():
                    match_count += 1

            # Include skill if at least one requirement matches
            if match_count > 0 and skill.success_rate > 0.1:
                relevant.append(skill)

        # Sort by success rate (descending)
        relevant.sort(key=lambda s: s.success_rate, reverse=True)
        return relevant

    def _save_skill(self, skill: Skill):
        """Persist skill to disk."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        skill_file = os.path.join(self.storage_path, f"{skill.name}.json")
        with open(skill_file, 'w') as f:
            json.dump(skill.to_dict(), f, indent=2)

    def load_from_disk(self):
        """Load all skills from storage."""
        import os
        if not os.path.exists(self.storage_path):
            return

        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                with open(filepath, 'r') as f:
                    skill_data = json.load(f)
                    skill = Skill.from_dict(skill_data)
                    self.skills[skill.name] = skill
```

### Step 2: Implement Skill Router

Build a learnable router that selects appropriate skills.

```python
class SkillRouter:
    """
    Routes to appropriate skills based on task context.
    Uses a trainable similarity model without gradient descent.
    """

    def __init__(self, skill_library: SkillLibrary):
        self.skill_library = skill_library
        self.skill_weights: Dict[str, float] = {}  # Name -> selection weight
        self.context_embeddings: Dict[str, List[float]] = {}  # Name -> embedding

        # Initialize weights uniformly
        for skill_name in skill_library.skills.keys():
            self.skill_weights[skill_name] = 1.0 / max(len(skill_library.skills), 1)

    def route_skills_for_task(self, task_description: str, task_context: Dict,
                              max_skills: int = 3) -> List[str]:
        """
        Select best skills for this task without model training.
        Uses success rate + context matching + weighted sampling.
        """
        candidates = []

        for skill_name, skill in self.skill_library.skills.items():
            # Score 1: Context relevance (keyword matching)
            context_score = 0.0
            for req in skill.context_requirements:
                if req.lower() in task_description.lower():
                    context_score += 1.0
            context_score = min(context_score / max(len(skill.context_requirements), 1), 1.0)

            # Score 2: Success rate (empirical performance)
            success_score = skill.success_rate

            # Score 3: Learned weight (adjusts based on usage patterns)
            weight_score = self.skill_weights.get(skill_name, 0.5)

            # Combined score (context is strongest signal)
            combined_score = (0.5 * context_score +
                             0.3 * success_score +
                             0.2 * weight_score)

            if combined_score > 0.1:  # Threshold
                candidates.append((skill_name, combined_score))

        # Sort by score and select top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_skills = [name for name, score in candidates[:max_skills]]

        return selected_skills

    def update_router_weights(self, selected_skills: List[str], task_success: bool):
        """
        Update router weights based on task outcome.
        Increases weight of skills that contributed to success.
        """
        adjustment = 0.1 if task_success else -0.05

        for skill_name in selected_skills:
            if skill_name in self.skill_weights:
                # Avoid weight explosion/collapse
                new_weight = self.skill_weights[skill_name] + adjustment
                self.skill_weights[skill_name] = max(0.01, min(1.0, new_weight))

        # Re-normalize weights to sum to 1
        total_weight = sum(self.skill_weights.values())
        if total_weight > 0:
            for skill_name in self.skill_weights:
                self.skill_weights[skill_name] /= total_weight
```

### Step 3: Generate Stateful Prompts with Skills

Construct task-specific prompts by incorporating selected skills.

```python
class StatefulPromptBuilder:
    """Builds task-specific prompts with dynamically selected skills."""

    def __init__(self, base_system_prompt: str, skill_library: SkillLibrary,
                 skill_router: SkillRouter):
        self.base_prompt = base_system_prompt
        self.skill_library = skill_library
        self.skill_router = skill_router

    def build_prompt(self, task_description: str, task_context: Dict) - str:
        """Construct full prompt with relevant skills."""

        # Route to appropriate skills
        selected_skill_names = self.skill_router.route_skills_for_task(
            task_description, task_context
        )

        # Build skill section
        skill_section = "## Available Skills\n"
        selected_skills = []
        for skill_name in selected_skill_names:
            skill = self.skill_library.skills.get(skill_name)
            if skill:
                skill_section += skill.to_markdown()
                selected_skills.append(skill)

        # Combine base prompt + skills + task
        full_prompt = f"""{self.base_prompt}

{skill_section}

## Current Task
{task_description}

## Context
{json.dumps(task_context, indent=2)}

You should use the above skills if they apply to this task. Explain your reasoning.
"""

        return full_prompt, selected_skills

    def record_execution(self, selected_skills: List[Skill], task_result: Dict):
        """After task execution, update skill performance metrics."""
        task_success = task_result.get('success', False)

        # Update individual skill metrics
        for skill in selected_skills:
            self.skill_library.update_skill_performance(skill.name, task_success)

        # Update router weights
        skill_names = [s.name for s in selected_skills]
        self.skill_router.update_router_weights(skill_names, task_success)
```

### Step 4: Synthesize New Skills from Failures

Generate new skills when the agent encounters novel failure modes.

```python
class SkillSynthesizer:
    """Generate new skills from failed task attempts."""

    def __init__(self, base_lm):
        self.base_lm = base_lm

    def synthesize_skill_from_failure(self, task_description: str,
                                      failure_analysis: Dict,
                                      base_context: Dict) -> Optional[Skill]:
        """
        Create a new skill to address a discovered failure mode.
        """

        # Construct synthesis prompt
        synthesis_prompt = f"""
You are an AI agent designer. Based on this failure, create a new reusable skill.

**Failed Task**: {task_description}

**Failure Analysis**:
- Error: {failure_analysis.get('error')}
- Root Cause: {failure_analysis.get('root_cause')}
- Failed Approach: {failure_analysis.get('failed_approach')}

**Context**:
{json.dumps(base_context, indent=2)}

Create a new skill that would help avoid this failure. Format your response as:

SKILL NAME: <name in snake_case>
DESCRIPTION: <one sentence description>
CONTEXT REQUIREMENTS: <comma-separated list of when to use>
EXECUTION LOGIC: <step-by-step logic in natural language>

Be specific and actionable.
"""

        # Generate skill definition
        skill_definition = self.base_lm.generate(synthesis_prompt)

        # Parse generated skill
        skill = self._parse_generated_skill(skill_definition)

        return skill

    def _parse_generated_skill(self, generated_text: str) -> Optional[Skill]:
        """Parse LLM-generated skill definition into Skill object."""
        import re
        from datetime import datetime

        try:
            name_match = re.search(r"SKILL NAME:\s*(\w+)", generated_text)
            desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?=CONTEXT|$)", generated_text, re.DOTALL)
            context_match = re.search(r"CONTEXT REQUIREMENTS:\s*(.+?)(?=EXECUTION|$)", generated_text, re.DOTALL)
            logic_match = re.search(r"EXECUTION LOGIC:\s*(.+?)$", generated_text, re.DOTALL)

            if not all([name_match, desc_match, context_match, logic_match]):
                return None

            skill = Skill(
                name=name_match.group(1),
                description=desc_match.group(1).strip(),
                context_requirements=[c.strip() for c in context_match.group(1).split(',')],
                execution_logic=logic_match.group(1).strip(),
                success_metrics={},
                created_timestamp=datetime.now().isoformat(),
                success_rate=0.0  # Start with neutral rating
            )

            return skill

        except Exception as e:
            print(f"Failed to parse skill: {e}")
            return None

    def validate_skill(self, skill: Skill, validation_tasks: List[Dict]) -> float:
        """Test new skill on validation tasks, return success rate."""
        successes = 0

        for task in validation_tasks:
            # Use skill to tackle task
            prompt = f"""Use this skill to solve the task:

Skill: {skill.to_markdown()}

Task: {task['description']}

Solve it step by step."""

            result = self.base_lm.generate(prompt)
            # Simple success check: did we get a complete answer?
            if len(result) > 50 and 'error' not in result.lower():
                successes += 1

        return successes / max(len(validation_tasks), 1)
```

### Step 5: Main Agent Loop

Integrate all components into a self-improving agent.

```python
class MementoAgent:
    """
    Agent that designs itself through evolving skills.
    """

    def __init__(self, base_lm, storage_path: str = "./skills"):
        self.base_lm = base_lm
        self.skill_library = SkillLibrary(storage_path)
        self.skill_router = SkillRouter(self.skill_library)
        self.prompt_builder = StatefulPromptBuilder(
            "You are a helpful AI agent.",
            self.skill_library,
            self.skill_router
        )
        self.skill_synthesizer = SkillSynthesizer(base_lm)

    def execute_task(self, task_description: str, task_context: Dict = None) - Dict:
        """Execute a task, learning and adapting throughout."""
        if task_context is None:
            task_context = {}

        # Build task-specific prompt with skills
        full_prompt, selected_skills = self.prompt_builder.build_prompt(
            task_description, task_context
        )

        # Execute task
        response = self.base_lm.generate(full_prompt)

        # Evaluate result
        task_result = {
            'response': response,
            'success': self._evaluate_response(response, task_description),
            'timestamp': datetime.now().isoformat()
        }

        # Update skill metrics
        self.prompt_builder.record_execution(selected_skills, task_result)

        # If failed, attempt to synthesize recovery skill
        if not task_result['success']:
            failure_analysis = self._analyze_failure(response, task_description)
            new_skill = self.skill_synthesizer.synthesize_skill_from_failure(
                task_description, failure_analysis, task_context
            )

            if new_skill:
                # Validate before adding
                validation_tasks = self._generate_validation_tasks(task_description, 3)
                success_rate = self.skill_synthesizer.validate_skill(
                    new_skill, validation_tasks
                )

                if success_rate > 0.5:
                    new_skill.success_rate = success_rate
                    self.skill_library.add_skill(new_skill)
                    print(f"Added new skill: {new_skill.name}")

        return task_result

    def _evaluate_response(self, response: str, task: str) -> bool:
        """Simple heuristic: response is successful if it's substantive."""
        return len(response) > 100 and 'error' not in response.lower()

    def _analyze_failure(self, response: str, task: str) -> Dict:
        """Extract failure insights from response."""
        return {
            'error': 'Response too brief' if len(response) < 100 else 'Unknown error',
            'root_cause': 'Insufficient context or unclear instructions',
            'failed_approach': 'Used generic approach without task-specific strategies'
        }

    def _generate_validation_tasks(self, base_task: str, num: int) -> List[Dict]:
        """Generate variants of task for validation."""
        return [
            {'description': f"Variant of: {base_task}"} for _ in range(num)
        ]
```

## Practical Guidance

**Hyperparameters:**
- Max skills per task: 3-5 (too many dilutes focus)
- Router weight adjustment: ±0.1 per successful use
- Skill success rate threshold: 0.1-0.2 to activate
- New skill validation tasks: 3-5 for initial confidence

**When to Use:**
- Long-running agent deployments where continuous improvement is valuable
- Domains with learnable patterns from failures
- Scenarios where retraining is expensive or infeasible
- Multi-domain agents that need to specialize per task

**When NOT to Use:**
- Single-pass inference where adaptation overhead isn't justified
- Domains requiring formal guarantees (safety-critical systems)
- Scenarios where parameter updates are needed (skills can't fix fundamental capability gaps)
- Real-time systems where prompt construction is too slow

**Pitfalls:**
- Skill library bloat: periodically prune low-performing skills
- Positive feedback loops: bad skills can reinforce each other; validate rigorously
- Skill interference: contradictory skills can confuse the agent; test interactions
- Context drift: if task distribution changes, skills become stale; monitor performance

## Reference

Paper: [arxiv.org/abs/2603.18743](https://arxiv.org/abs/2603.18743)
