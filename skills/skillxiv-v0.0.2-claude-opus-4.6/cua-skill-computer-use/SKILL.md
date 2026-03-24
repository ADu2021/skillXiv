---
name: cua-skill-computer-use
title: "CUA-Skill: Develop Skills for Computer Using Agent"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.21123"
keywords: [Computer-Use Agents, Skill Abstraction, GUI Interaction, Parameterized Graphs, Skill Composition]
description: "Build desktop agents via reusable, parameterized skills encoding human computer-use knowledge. Skills combine execution graphs (handling UI variations) with composition graphs (chaining strategies). 57.5% success on WindowsAgentArena."
---

# CUA-Skill: Reusable Computer-Use Skills

## Problem
Desktop agents typically model interaction as flat sequences of low-level actions, forcing agents to rediscover common workflows from scratch. There is no reusable abstraction layer for human computer-use knowledge.

Multi-step desktop tasks require recovering common patterns (opening applications, finding UI elements, editing documents) repeatedly.

## Core Concept
CUA-Skill encodes computer-use knowledge as reusable, parameterized skills. Each skill captures a coherent interaction pattern with built-in handling for UI variations and contingencies. Skills compose hierarchically into complete task workflows.

A skill comprises: target application, natural language intent, parameters, and execution graph (handling variations). Composition graphs capture how skills chain into higher-level strategies.

## Architecture Overview

- **Skill Definition**: Application, intent, parameters, execution constraints
- **Execution Graph**: Parameterized paths handling common UI variations
- **Contingency Handling**: Alternative branches for common failures
- **Composition Graph**: Directed graph of skill dependencies and ordering
- **Skill Retrieval**: Dynamic selection based on current UI state
- **Memory Integration**: State tracking across multi-step workflows

## Implementation

### Step 1: Define Parameterized Skill Structure
Create skill representation with execution graphs.

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Skill:
    """Reusable computer-use skill with parameterized execution."""
    skill_id: str
    application: str
    intent: str  # Natural language description
    parameters: Dict[str, str]  # Input parameters
    execution_graph: Dict[str, Any]  # Parameterized execution paths

    def __init__(self, skill_id, application, intent, parameters):
        self.skill_id = skill_id
        self.application = application
        self.intent = intent
        self.parameters = parameters
        self.execution_graph = self._build_execution_graph()

    def _build_execution_graph(self):
        """Build parameterized graph handling UI variations."""
        if self.application == "text_editor":
            if self.intent == "edit_file":
                return {
                    'start': 'find_file',
                    'find_file': {
                        'action': 'search',
                        'params': {'filename': self.parameters['filename']},
                        'on_success': 'open_file',
                        'on_failure': 'create_new_file'
                    },
                    'open_file': {
                        'action': 'open',
                        'next': 'edit_content'
                    },
                    'create_new_file': {
                        'action': 'create',
                        'params': {'name': self.parameters['filename']},
                        'next': 'edit_content'
                    },
                    'edit_content': {
                        'action': 'type',
                        'params': {'text': self.parameters['content']},
                        'next': 'save_file'
                    },
                    'save_file': {
                        'action': 'save',
                        'on_success': 'end',
                        'on_failure': 'retry_save'
                    },
                    'retry_save': {
                        'action': 'save_as',
                        'next': 'end'
                    },
                    'end': None
                }

    def execute(self, ui_agent, current_ui_state):
        """Execute skill handling UI variations via execution graph."""
        current_node = self.execution_graph['start']

        while current_node is not None:
            node = self.execution_graph[current_node]

            # Execute action
            action = node['action']
            params = node.get('params', {}).copy()

            # Fill in parameter values
            for key, value in params.items():
                if isinstance(value, str) and value.startswith('{'):
                    param_name = value.strip('{}')
                    params[key] = self.parameters.get(param_name, value)

            result = ui_agent.execute_action(action, params, current_ui_state)

            # Handle contingencies
            if result.success:
                current_node = node.get('on_success', node.get('next'))
            else:
                current_node = node.get('on_failure', node.get('next'))

        return result
```

### Step 2: Build Skill Composition Graph
Create high-level task workflows by chaining skills.

```python
@dataclass
class CompositionGraph:
    """High-level task workflow via skill composition."""
    task_id: str
    skills: List[Skill]
    dependencies: Dict[str, List[str]]  # skill_id -> list of dependencies
    skill_order: List[str]  # Topological execution order

    def execute(self, ui_agent, initial_state):
        """Execute composition by chaining skills."""
        state = initial_state
        executed_skills = {}

        for skill_id in self.skill_order:
            # Retrieve skill
            skill = self.get_skill(skill_id)

            # Update parameters from previous skill outputs
            for param_key, param_value in skill.parameters.items():
                if param_key.startswith('${'):
                    # Reference to previous skill output
                    source_skill = param_value.split('.')[0].strip('${}')
                    output_key = param_value.split('.')[1]
                    skill.parameters[param_key] = executed_skills[source_skill][output_key]

            # Execute skill
            result = skill.execute(ui_agent, state)
            executed_skills[skill_id] = result.outputs
            state = result.final_state

        return executed_skills, state

    def get_skill(self, skill_id):
        """Retrieve skill by ID."""
        for skill in self.skills:
            if skill.skill_id == skill_id:
                return skill
        raise ValueError(f"Skill {skill_id} not found")
```

### Step 3: Implement Skill Retrieval
Dynamically select appropriate skills based on current UI.

```python
class SkillRetrievalModule:
    """Dynamic skill selection based on UI state and task context."""

    def __init__(self, skill_library):
        self.skill_library = skill_library
        self.retrieval_model = nn.Sequential(
            nn.Linear(512, 256),  # UI embedding size
            nn.ReLU(),
            nn.Linear(256, len(skill_library))
        )

    def retrieve_skills(self, ui_embedding, task_description, k=3):
        """Retrieve top-k relevant skills."""
        # Combine UI and task embeddings
        task_embedding = embed_text(task_description)
        combined = torch.cat([ui_embedding, task_embedding])

        # Compute relevance scores
        scores = self.retrieval_model(combined)

        # Get top-k skills
        top_indices = torch.topk(scores, k).indices
        retrieved_skills = [self.skill_library[i] for i in top_indices]

        return retrieved_skills

    def select_next_skill(self, ui_agent, current_state, available_skills, task_progress):
        """Select next skill for execution."""
        ui_embedding = ui_agent.encode_ui(current_state)
        task_description = task_progress.get_remaining_subtask()

        retrieved = self.retrieve_skills(ui_embedding, task_description)

        # Validate feasibility
        for skill in retrieved:
            if self._is_executable(skill, current_state):
                return skill

        return None

    def _is_executable(self, skill, ui_state):
        """Check if skill can execute in current UI state."""
        target_app = skill.application
        current_app = ui_state['active_application']
        return target_app == current_app or target_app == '*'
```

### Step 4: Training and Refinement
Learn skill parameters and execution strategies from demonstrations.

```python
def learn_skill_from_demonstration(demo_trajectory, skill_template):
    """Extract skill parameters from human demonstration."""
    actions = demo_trajectory['actions']
    ui_states = demo_trajectory['ui_states']

    # Identify consistent action sequences
    action_sequences = identify_patterns(actions)

    # Build execution graph
    execution_steps = []
    for action, pre_state, post_state in zip(actions, ui_states[:-1], ui_states[1:]):
        step = {
            'action': action['type'],
            'params': action.get('params', {}),
            'preconditions': extract_ui_features(pre_state),
            'postconditions': extract_ui_features(post_state)
        }
        execution_steps.append(step)

    # Extract parameterizable values
    parameters = extract_parameters_from_trace(execution_steps)

    return SkillDefinition(
        skill_id=generate_skill_id(),
        execution_graph=build_graph_from_trace(execution_steps),
        parameters=parameters,
        learned_from_demo=True
    )

def refine_skills_via_rl(agent, skill_library, task_set, num_episodes=1000):
    """Improve skill execution via reinforcement learning."""
    skill_policy_optimizer = torch.optim.Adam(agent.skill_selector.parameters(), lr=1e-4)

    for episode in range(num_episodes):
        task = random.choice(task_set)
        state = environment.reset_to_task(task)

        episode_reward = 0
        for step in range(100):
            # Select skill
            skill = agent.skill_selector.select_next_skill(agent, state, task)

            # Execute skill
            result = skill.execute(agent, state)
            state = result.final_state

            # Compute reward
            task_progress = evaluate_task_progress(task, state)
            reward = task_progress - (1.0 if result.success else 0.5)
            episode_reward += reward

            if task_progress >= 1.0:
                break

        # Update skill selector policy
        policy_loss = -episode_reward * agent.skill_selector.log_prob_last_selection
        policy_loss.backward()
        skill_policy_optimizer.step()

    return agent
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Skill retrieval top-k | 3-5 | Balance coverage with efficiency |
| Execution graph branching | 2-4 branches | Handle common variations |
| Parameter embedding dim | 256-512 | Sufficient capacity for diversity |
| Memory buffer size | 100-1000 | State tracking history |
| Skill composition depth | 3-6 levels | Typical task complexity |

### When to Use

- Desktop/GUI automation (Slack, Excel, web forms)
- Multi-application workflows (compose skills across apps)
- Repetitive task automation with human-like strategy
- Environments where action spaces are complex (screenshots + coordinates)
- Scenarios where interpretability of intermediate skills matters

### When Not to Use

- Very simple single-application tasks (direct scripting sufficient)
- Real-time systems with strict latency requirements
- Novel UI patterns not covered by existing skills
- Environments where low-level control is critical
- Tasks requiring deep domain expertise beyond skill library

### Common Pitfalls

1. **Skill granularity mismatch**: Skills too large remove reusability; too small create composition overhead. Validate on diverse tasks.
2. **Parameter binding errors**: Incorrect parameter passing between composed skills. Implement strict type checking.
3. **Incomplete variation handling**: Execution graph missing common UI variations. Audit actual UI patterns.
4. **Retrieval quality**: Skill selector may choose irrelevant skills. Validate retrievals on held-out test tasks.

## Reference
CUA-Skill: Develop Skills for Computer Using Agent
https://arxiv.org/abs/2601.21123
