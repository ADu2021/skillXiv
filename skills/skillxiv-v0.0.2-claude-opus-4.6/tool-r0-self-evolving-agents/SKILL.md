---
name: tool-r0-self-evolving-agents
title: "Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.21320"
keywords: [Self-Play, Reinforcement Learning, Tool Use, Agent Evolution, Curriculum Learning]
description: "Tool-R0 trains tool-use agents through self-play between Generator and Solver, creating curriculum-like progression without manual task curation."
---

# Technique: Self-Play Curriculum for Tool-Use Agent Training

Teaching agents to use tools effectively traditionally requires manually curated task datasets or human feedback. Tool-R0 eliminates this requirement entirely through self-play: two complementary agents (Generator and Solver) co-evolve. The Generator proposes increasingly challenging tasks, while the Solver learns to complete them using real-world tool calls. This creates a natural curriculum where difficulty increases as the Solver improves.

The approach is actionable because (1) it requires no external data or labels, (2) both agents improve simultaneously, and (3) the process naturally converges to useful tool-learning behaviors.

## Core Concept

The core insight is that the best teacher for an agent is an opponent at the agent's competence frontier. Rather than having humans design tasks, let agents design them for each other:

- **Generator**: Proposes tasks that the current Solver fails at (targeted challenges)
- **Solver**: Learns to use tools to solve Generator's tasks (real-world tool calls)
- **Feedback**: Success/failure signals drive both agents to improve
- **Curriculum**: Difficulty naturally increases as Solver improves

This mirrors how humans learn complex skills: practice problems just beyond current ability accelerate learning most effectively.

## Architecture Overview

- **Generator Agent**: Proposes tasks ("write code to X", "search for Y")
- **Solver Agent**: Executes tool calls to complete tasks
- **Tool Environment**: APIs/functions the Solver can invoke (code execution, search, etc.)
- **Reward Signal**: Binary correctness (task completed successfully or not)
- **Self-play Loop**: Alternating agent updates in a curriculum-like progression

## Implementation Steps

Tool-R0 involves setting up two agents that play against each other. Here's how to implement it:

Define the tool environment that the Solver agent can interact with:

```python
import subprocess
import requests
from typing import Dict, List, Any

class ToolEnvironment:
    """Defines available tools the Solver can use."""

    def __init__(self):
        self.tools = {
            'execute_code': self.execute_code,
            'web_search': self.web_search,
            'read_file': self.read_file,
            'write_file': self.write_file,
        }

    def execute_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Execute code and return output."""
        try:
            if language == 'python':
                result = subprocess.run(
                    ['python', '-c', code],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def web_search(self, query: str) -> Dict[str, Any]:
        """Search the web and return results."""
        try:
            # In practice, use an API like Google Search or Bing
            # For this example, return dummy results
            return {
                'success': True,
                'results': [
                    {'title': 'Result 1', 'snippet': '...'},
                    {'title': 'Result 2', 'snippet': '...'},
                ]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def read_file(self, filepath: str) -> Dict[str, Any]:
        """Read a file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            return {'success': True, 'content': content}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Write to a file."""
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool and return results."""
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        return {'success': False, 'error': f'Unknown tool: {tool_name}'}
```

Implement the Generator agent that proposes tasks:

```python
class GeneratorAgent:
    """
    Proposes tasks at the Solver's competence frontier.
    Learns to generate tasks that are challenging but solvable.
    """

    def __init__(self, model, task_difficulty_tracker=None):
        self.model = model
        self.difficulty_tracker = task_difficulty_tracker or {}
        self.task_pool = []

    def propose_task(self, solver_skill_level: float) -> str:
        """
        Generate a task proportional to Solver's current skill level.
        Difficulty increases as Solver improves.
        """
        difficulty_prompt = f"""
        Generate a task that will challenge an agent at skill level {solver_skill_level:.2f}.

        Task categories:
        - Level 1: Simple tool use (1 tool, 1 step)
        - Level 2: Multi-step tool use (2-3 tools, sequence)
        - Level 3: Complex reasoning (decision making, multiple branches)
        - Level 4: Novel problem solving (combine tools creatively)

        Generate a task that is just beyond current capability.
        Format: "TASK: [description]"
        """

        # Generate task using model
        task = self.model.generate(
            difficulty_prompt,
            max_length=200,
            temperature=0.7
        )

        return task.strip()

    def update_from_solver_feedback(self, task: str, solver_succeeded: bool):
        """
        Learn from Solver's performance.
        Adjust future tasks based on success/failure.
        """
        self.task_pool.append({
            'task': task,
            'succeeded': solver_succeeded
        })

        # Track success rate per difficulty level
        # Propose harder tasks if Solver succeeds, easier if fails
```

Implement the Solver agent that learns to use tools:

```python
class SolverAgent:
    def __init__(self, model, tool_env, learning_rate=1e-5):
        self.model = model
        self.tool_env = tool_env
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.success_count = 0
        self.attempt_count = 0

    def solve_task(self, task: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Attempt to solve a task using available tools.
        Returns: success flag and reasoning trace.
        """
        reasoning_trace = []
        current_state = f"Task: {task}"

        for step in range(max_steps):
            # Model plans next action
            action_prompt = f"""
            {current_state}

            Available tools:
            - execute_code(code, language='python')
            - web_search(query)
            - read_file(filepath)
            - write_file(filepath, content)

            What's the next tool to use? Format: TOOL: [tool_name]\nPARAMS: [params]
            """

            action_output = self.model.generate(
                action_prompt,
                max_length=150,
                temperature=0.7
            )

            # Parse action
            tool_name, params = parse_action(action_output)

            if tool_name is None:
                break

            # Execute tool
            result = self.tool_env.execute_tool(tool_name, **params)
            reasoning_trace.append({
                'step': step,
                'tool': tool_name,
                'params': params,
                'result': result
            })

            # Update state
            current_state += f"\nStep {step+1}: {tool_name} -> {result['output'] if result['success'] else result['error']}"

            # Check if task completed
            if result['success'] and self.check_task_completion(task, result):
                return {
                    'success': True,
                    'steps': step + 1,
                    'trace': reasoning_trace
                }

        return {
            'success': False,
            'steps': max_steps,
            'trace': reasoning_trace
        }

    def check_task_completion(self, task: str, result: Dict) -> bool:
        """Determine if task was completed successfully."""
        # Simple heuristic: if tool executed without error, consider it success
        # In practice, validate output against task requirements
        return result.get('success', False)

    def train_on_successful_trajectory(self, trajectory: List[Dict]):
        """Fine-tune model on successful tool-use trajectory."""
        # Create training examples from successful traces
        for i, step in enumerate(trajectory):
            # Simplified: in practice, use full trajectory context
            action_str = f"{step['tool']}({step['params']})"
            # Fine-tune to predict this action given previous state
            # (Omitted for brevity)
        pass

    def update_skill_level(self, success: bool):
        """Track improvement over time."""
        self.attempt_count += 1
        if success:
            self.success_count += 1

    @property
    def skill_level(self) -> float:
        """Current success rate (0 to 1)."""
        if self.attempt_count == 0:
            return 0.0
        return self.success_count / self.attempt_count

def parse_action(output: str) -> tuple:
    """Parse model output into tool name and parameters."""
    import re
    # Format: "TOOL: execute_code\nPARAMS: {'code': '...'}"
    tool_match = re.search(r'TOOL:\s*(\w+)', output)
    params_match = re.search(r'PARAMS:\s*(\{.*\})', output)

    if tool_match:
        tool_name = tool_match.group(1)
        params = {}
        if params_match:
            try:
                params = eval(params_match.group(1))
            except:
                pass
        return tool_name, params
    return None, {}
```

Implement the self-play training loop:

```python
class ToolR0Trainer:
    def __init__(self, model, tool_env, num_iterations=100):
        self.generator = GeneratorAgent(model)
        self.solver = SolverAgent(model, tool_env)
        self.tool_env = tool_env
        self.num_iterations = num_iterations

    def train(self):
        """Self-play training loop."""
        for iteration in range(self.num_iterations):
            # Generator proposes task at current difficulty
            task = self.generator.propose_task(self.solver.skill_level)

            # Solver attempts to complete task
            result = self.solver.solve_task(task)
            success = result['success']

            # Generator learns from result
            self.generator.update_from_solver_feedback(task, success)

            # Solver trains on successful traces
            if success:
                self.solver.train_on_successful_trajectory(result['trace'])

            # Update skill tracking
            self.solver.update_skill_level(success)

            if (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration+1}: "
                    f"Solver skill level: {self.solver.skill_level:.2%}, "
                    f"Task: {task[:50]}..."
                )

        return self.solver
```

## Practical Guidance

**When to Use:**
- Training tool-use agents from scratch
- When labeled task data is unavailable
- For skill development in narrow domains (coding, retrieval, calculation)
- When you want emergent curriculum learning

**When NOT to Use:**
- Real-time systems (self-play takes time)
- General-purpose language modeling (not designed for open-ended generation)
- When you have abundant labeled task data (supervised methods may be faster)

**Hyperparameters:**
- `max_steps_per_task`: 3–10 (allow reasonable exploration)
- `num_iterations`: 50–500 (longer training = better skill development)
- `temperature`: 0.6–0.8 (balance exploration vs. coherent plans)
- Skill level threshold: ~70% success rate before increasing difficulty

**Implementation Notes:**
- Start with simple tool sets (2–3 tools); expand gradually
- Validate tool execution safely (sandbox code execution, limit API calls)
- Use failure signals to guide Generator toward relevant difficulty
- Track task diversity to ensure comprehensive learning

**Results:**
- 92.5% relative improvement over base model on tool-learning benchmarks
- Surpasses supervised baselines without requiring manual task annotation
- Agents learn interpretable tool-use patterns
- Scalable to multiple tool types

---

**Reference:** [Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data](https://arxiv.org/abs/2602.21320)
