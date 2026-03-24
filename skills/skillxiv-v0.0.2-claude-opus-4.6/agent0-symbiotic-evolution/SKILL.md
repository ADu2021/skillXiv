---
name: agent0-symbiotic-evolution
title: "Agent0: Unleashing Self-Evolving Agents from Zero Data via Symbiotic Competition"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.16043"
keywords: [Agent Training, Self-Evolution, Curriculum Learning, Symbiotic Competition, Zero-Data Learning]
description: "Train agents from scratch without human-annotated data via symbiotic competition—curriculum agent proposes progressively harder tasks while executor agent learns to solve them, creating autonomous self-reinforcing loops."
---

# Train Agents from Zero Data via Symbiotic Curriculum Competition

Most agent training requires human-curated task datasets or expert demonstrations. Agent0 breaks this dependency via **symbiotic competition**: two agents create a self-reinforcing loop without external data. A Curriculum Agent proposes increasingly difficult tasks; an Executor Agent learns to solve them. As the executor improves, the curriculum agent escalates difficulty, creating an automatic curriculum.

This achieves significant performance gains (18% on math, 24% on reasoning benchmarks) from a base model with zero human-annotated training data—the only input is the problem domain specification.

## Core Concept

Training agents typically requires:
1. Task datasets (curated by humans)
2. Reward models (trained on human preferences)
3. Expert demonstrations (for imitation learning)

Agent0 eliminates these by creating two complementary agents:

1. **Curriculum Agent**: Proposes novel tasks from the problem space; initially simple, escalates difficulty as executor improves
2. **Executor Agent**: Solves proposed tasks; learns from successes and failures

The feedback loop: Executor improves → Curriculum escalates → harder tasks → Executor gets better signal → loops. This creates high-quality curriculum learning without human intervention.

## Architecture Overview

- **Curriculum Agent**: LLM that generates tasks conditioned on executor capability level; uses tool-aware task generation
- **Executor Agent**: Solves curriculum tasks via RL; builds tool integration capability progressively
- **Capability Assessment**: Track executor performance level (easy/medium/hard); feed to curriculum for difficulty adjustment
- **Tool Integration**: Curriculum proposes tasks requiring specific tools; executor learns which tools to use when
- **Self-Reinforcement**: Executor improvement directly increases curriculum difficulty, creating positive feedback

## Implementation Steps

**Step 1: Define Problem Space and Tool Interface.**

```python
class ProblemDomain:
    """
    Specification of task domain (math, coding, QA, etc.)
    and available tools for solving.
    """
    def __init__(self, domain_name='math', tools=None):
        self.domain_name = domain_name
        self.tools = tools or {}
        self.capability_level = 'easy'  # [easy, medium, hard]

    def register_tool(self, tool_name, fn, description):
        """Register a tool the executor can learn to use."""
        self.tools[tool_name] = {
            'fn': fn,
            'description': description,
            'usage_count': 0
        }

    def get_tool_descriptions(self):
        """Return descriptions for curriculum agent's task generation."""
        descriptions = []
        for name, tool_info in self.tools.items():
            descriptions.append(f"{name}: {tool_info['description']}")
        return "\n".join(descriptions)

    def evaluate_solution(self, task, solution):
        """
        Check if solution solves the task.
        Returns: (is_correct, score)
        """
        # Domain-specific evaluation logic
        if self.domain_name == 'math':
            return self._evaluate_math(task, solution)
        elif self.domain_name == 'coding':
            return self._evaluate_code(task, solution)
        else:
            return self._evaluate_generic(task, solution)

    def _evaluate_math(self, task, solution):
        """Math domain: check numerical correctness."""
        try:
            answer = extract_answer(solution)
            correct = abs(answer - task['answer']) < 1e-6
            return correct, 1.0 if correct else 0.0
        except:
            return False, 0.0
```

**Step 2: Curriculum Agent—Generate Tasks at Appropriate Difficulty.**

```python
class CurriculumAgent:
    """
    Generates tasks calibrated to executor's current capability level.
    """
    def __init__(self, base_llm, problem_domain):
        self.base_llm = base_llm
        self.problem_domain = problem_domain
        self.generated_tasks = []
        self.difficulty_progression = []

    def propose_task(self, executor_level):
        """
        Generate a new task at difficulty matching executor's level.
        executor_level: str in [easy, medium, hard]
        """
        # Construct prompt for curriculum agent
        prompt = f"""
        Problem domain: {self.problem_domain.domain_name}
        Current executor capability level: {executor_level}

        Available tools:
        {self.problem_domain.get_tool_descriptions()}

        Generate a novel task that:
        - Requires {executor_level} reasoning/problem-solving
        - Ideally involves {self._select_tool_for_level(executor_level)} tool
        - Is different from previously generated tasks:
        {self._format_task_history()}

        Format: JSON with keys: task, tool_hint, expected_approach
        """

        # Generate task
        task_json = self.base_llm(prompt)
        task = json.loads(task_json)

        self.generated_tasks.append(task)
        return task

    def escalate_difficulty(self, executor_success_rate):
        """
        Adjust difficulty based on executor's success rate.
        Higher success rate → harder tasks; lower rate → easier tasks.
        """
        if executor_success_rate > 0.8:
            # Executor is proficient at current level
            if self.problem_domain.capability_level == 'easy':
                self.problem_domain.capability_level = 'medium'
            elif self.problem_domain.capability_level == 'medium':
                self.problem_domain.capability_level = 'hard'
        elif executor_success_rate < 0.5:
            # Executor struggling; reduce difficulty
            if self.problem_domain.capability_level == 'hard':
                self.problem_domain.capability_level = 'medium'
            elif self.problem_domain.capability_level == 'medium':
                self.problem_domain.capability_level = 'easy'

    def _select_tool_for_level(self, level):
        """Select tool appropriate for difficulty level."""
        if level == 'easy':
            return 'basic_calculator'
        elif level == 'medium':
            return 'python_interpreter'
        else:
            return 'complex_reasoning_tools'

    def _format_task_history(self):
        """Return recent tasks to encourage diversity."""
        recent = self.generated_tasks[-5:]
        return "\n".join([f"- {t['task']}" for t in recent])
```

**Step 3: Executor Agent—Learn to Solve Curriculum Tasks.**

```python
class ExecutorAgent:
    """
    Solves curriculum-generated tasks via RL.
    Learns to integrate tools and build solving strategies.
    """
    def __init__(self, base_llm, problem_domain, learning_rate=1e-5):
        self.base_llm = base_llm
        self.problem_domain = problem_domain
        self.success_history = []
        self.tool_usage_patterns = {}
        self.optimizer = torch.optim.AdamW(base_llm.parameters(), lr=learning_rate)

    def solve_task(self, task, max_steps=10):
        """
        Attempt to solve task; collect trajectory for RL.
        Returns: (solution, trajectory, reward)
        """
        trajectory = []
        solution_steps = []

        # Initialize context
        context = f"Task: {task['task']}\nAvailable tools: {self.problem_domain.get_tool_descriptions()}"

        for step in range(max_steps):
            # Agent decides next action
            action_prompt = context + f"\nCurrent progress: {solution_steps}\n\nNext action:"

            action = self.base_llm.generate(action_prompt, max_tokens=100)
            solution_steps.append(action)

            # Parse action (is it a tool call or final answer?)
            if self._is_tool_invocation(action):
                tool_name, tool_args = self._parse_tool_call(action)

                # Execute tool
                tool_fn = self.problem_domain.tools[tool_name]['fn']
                tool_result = tool_fn(tool_args)

                # Track tool usage
                self.tool_usage_patterns[tool_name] = self.tool_usage_patterns.get(tool_name, 0) + 1

                # Update context
                context += f"\n[Tool: {tool_name}]\nResult: {tool_result}"

                trajectory.append({
                    'action': action,
                    'tool': tool_name,
                    'result': tool_result,
                    'step': step
                })

            elif self._is_final_answer(action):
                solution = self._extract_answer(action)

                # Evaluate
                is_correct, score = self.problem_domain.evaluate_solution(task, solution)

                trajectory.append({
                    'action': action,
                    'is_final': True,
                    'solution': solution,
                    'reward': 1.0 if is_correct else 0.0
                })

                return solution, trajectory, 1.0 if is_correct else 0.0

        # Max steps reached without solution
        return None, trajectory, 0.0

    def learn_from_trajectory(self, trajectory, reward):
        """
        Update executor via RL (policy gradient).
        """
        # Compute returns (discounted cumulative reward)
        returns = []
        g = 0
        for step in reversed(trajectory):
            g = step.get('reward', 0) + 0.99 * g
            returns.insert(0, g)

        returns = torch.tensor(returns)

        # Policy gradient: maximize log-prob of actions that led to high returns
        policy_loss = 0
        for i, step in enumerate(trajectory):
            # Get log probability of this action
            action_log_prob = self._compute_log_prob(step['action'])

            # Policy gradient
            policy_loss -= action_log_prob * returns[i]

        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def get_success_rate(self, num_eval_tasks=20):
        """Evaluate current capability on curriculum-proposed tasks."""
        successes = 0

        for _ in range(num_eval_tasks):
            task = self.problem_domain.get_evaluation_task()
            _, _, reward = self.solve_task(task)
            successes += reward

        return successes / num_eval_tasks
```

**Step 4: Symbiotic Loop—Agent Co-Evolution.**

```python
def symbiotic_evolution_loop(
    problem_domain, base_llm,
    num_iterations=1000,
    tasks_per_iteration=10
):
    """
    Main training loop: curriculum and executor co-evolve.
    """
    curriculum = CurriculumAgent(base_llm, problem_domain)
    executor = ExecutorAgent(base_llm, problem_domain)

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration} ===")
        print(f"Difficulty level: {problem_domain.capability_level}")

        # Generate tasks at current difficulty
        successes = 0
        trajectories = []

        for task_idx in range(tasks_per_iteration):
            # Curriculum proposes task
            task = curriculum.propose_task(problem_domain.capability_level)

            # Executor attempts task
            solution, trajectory, reward = executor.solve_task(task)
            successes += reward

            # Executor learns from trajectory
            executor.learn_from_trajectory(trajectory, reward)

            trajectories.append({
                'task': task,
                'solution': solution,
                'reward': reward
            })

        # Assess performance and adjust difficulty
        success_rate = successes / tasks_per_iteration
        curriculum.escalate_difficulty(success_rate)

        # Evaluate on harder tasks
        eval_success = executor.get_success_rate(num_eval_tasks=5)

        print(f"Success rate (curriculum): {success_rate:.2%}")
        print(f"Eval success rate: {eval_success:.2%}")
        print(f"Tool usage: {executor.tool_usage_patterns}")

        # Early stopping if performance plateaus
        if iteration > 100 and eval_success < 0.1:
            print("Converged or diverged; stopping")
            break

    return executor, curriculum
```

## Practical Guidance

**When to Use:** Training agents from scratch when task datasets are unavailable; domains where tasks can be procedurally generated or when self-play/curriculum is feasible (math, coding, games).

**Curriculum Design:**
- Start with simple tasks (one-step solutions); escalate gradually
- Introduce tools progressively; hard level introduces complex tool combinations
- Success thresholds: 80%+ for escalation, <50% for de-escalation

**Pitfalls:**
- **Task degeneration**: Curriculum may propose trivial or identical tasks; add diversity metrics and deduplication
- **Executor plateauing**: If executor gets stuck, curriculum tasks may be too hard; implement gradual escalation
- **Tool usage imbalance**: Some tools may be ignored; use tool-hint in curriculum to encourage exploration
- **Evaluation overfitting**: Executor may overfit to curriculum's task distribution; use held-out evaluation set

**When NOT to Use:** Domains with scarce tool sets; problems requiring external knowledge (not learnable from environment); safety-critical applications.

**Integration**: Compatible with any LLM; works best with function-calling APIs for tool use.

---
Reference: https://arxiv.org/abs/2511.16043
