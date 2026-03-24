---
name: wide-seek-multi-agent-scaling
title: "WideSeek: Advancing Wide Research via Multi-Agent Scaling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02636"
keywords: [Multi-Agent Systems, Dynamic Forking, Hierarchical Execution, GRPO Optimization, Breadth Scaling]
description: "Dynamically fork sub-agents at any step based on task requirements instead of using fixed agent counts. Linearize hierarchical trajectories into unified sequences for GRPO training. Demonstrates 6.36x more sub-agents than baselines via learned orchestration."
---

# WideSeek: Breadth Scaling via Dynamic Multi-Agent Forking

Traditional multi-agent systems use fixed agent counts, but task complexity varies: some steps need multiple exploratory agents, others work fine with a single path. WideSeek enables dynamic agent forking—the main agent autonomously creates variable numbers of sub-agents at each step. This is trained end-to-end via GRPO by linearizing hierarchical execution traces into single unified sequences.

The key insight is that agent forking is itself a learnable skill. By treating multi-agent trajectories as extended single sequences, WideSeek applies standard policy optimization to learn when and how many agents to spawn.

## Core Concept

WideSeek operates on two principles:

1. **Dynamic Agent Forking**: Main agent can call `call_subagent(task)` any number of times, creating sub-agents on-demand to parallelize work.

2. **Unified Trajectory Training**: All agent trajectories (main + sub-agents) are linearized into a single sequence, then optimized with GRPO as if from one agent.

This unifies multi-agent coordination with policy optimization.

## Architecture Overview

- **Main Agent**: Orchestrates task decomposition; can fork sub-agents dynamically
- **Sub-Agent Pool**: Execute assigned subtasks in parallel with isolated contexts
- **Trajectory Linearization**: Convert tree of agent decisions into linear sequence
- **GRPO Optimizer**: Standard group relative policy optimization on linearized trajectories
- **Shared Model**: Single LLM backbone for all agents (main and sub)
- **Tool Restrictions**: Sub-agents have subset of main agent's tools (e.g., search, no forking)

## Implementation

### Step 1: Design Agent Forking Interface

Create the tool that allows dynamic sub-agent creation.

```python
# Dynamic agent forking
class SubAgentFork:
    def __init__(self, max_parallel: int = 8, max_depth: int = 2):
        """
        Manage dynamic sub-agent creation.

        Args:
            max_parallel: Maximum sub-agents at one level
            max_depth: Maximum nesting depth
        """
        self.max_parallel = max_parallel
        self.max_depth = max_depth
        self.current_depth = 0
        self.spawned_agents = 0

    def can_fork(self) -> bool:
        """Check if forking is allowed."""
        return (self.current_depth < self.max_depth and
                self.spawned_agents < self.max_parallel)

    def fork_subagent(self, subtask: str, tools: List[str],
                     parent_context: str) -> "SubAgent":
        """
        Create and return a new sub-agent.

        Args:
            subtask: Specific task for sub-agent
            tools: Available tools for sub-agent
            parent_context: Context from parent agent

        Returns:
            Sub-agent instance
        """
        if not self.can_fork():
            return None

        self.spawned_agents += 1
        self.current_depth += 1

        agent = SubAgent(
            task=subtask,
            available_tools=tools,
            context=parent_context,
            depth=self.current_depth
        )

        return agent

class SubAgent:
    def __init__(self, task: str, available_tools: List[str],
                 context: str, depth: int):
        """Lightweight sub-agent for specific subtask."""
        self.task = task
        self.available_tools = available_tools
        self.context = context
        self.depth = depth
        self.trajectory = []
        self.result = None

    def execute(self, model) -> str:
        """Execute subtask and return result."""
        # Construct prompt with restricted tools
        system_prompt = f"""You are a sub-agent working on: {self.task}

Available tools: {', '.join(self.available_tools)}
Note: You cannot fork new agents. Complete this task and report result.

Context: {self.context}"""

        response = model.generate(
            system_prompt,
            temperature=0.7,
            max_tokens=512
        )

        self.trajectory.append({
            "type": "thought",
            "content": response
        })

        self.result = response
        return response
```

### Step 2: Implement Trajectory Linearization

Convert multi-agent tree execution into linear sequence for GRPO.

```python
# Trajectory linearization
class TrajectoryLinearizer:
    def __init__(self):
        """Convert hierarchical agent executions to linear sequences."""
        self.linearized = []

    def linearize_execution(self, main_trajectory: List[dict],
                          subagent_trajectories: Dict[int, List[dict]]) -> List[dict]:
        """
        Convert tree of agent trajectories into single linear sequence.

        Args:
            main_trajectory: Main agent's thought/action sequence
            subagent_trajectories: Dict mapping agent_id to trajectory

        Returns:
            Single linearized trajectory for GRPO
        """
        linearized = []

        # Add main agent thoughts
        for step in main_trajectory:
            if step.get("type") == "thought":
                linearized.append({
                    "agent": "main",
                    "content": step["content"],
                    "type": "thought"
                })

            elif step.get("type") == "fork":
                # Record fork decision
                fork_id = step.get("fork_id")
                task = step.get("task")

                linearized.append({
                    "agent": "main",
                    "content": f"[FORK {fork_id}: {task}]",
                    "type": "fork"
                })

                # Append sub-agent trajectory inline
                if fork_id in subagent_trajectories:
                    for substep in subagent_trajectories[fork_id]:
                        linearized.append({
                            "agent": f"sub_{fork_id}",
                            "content": substep.get("content"),
                            "type": "thought"
                        })

                # Record sub-agent result
                result = step.get("result")
                linearized.append({
                    "agent": f"sub_{fork_id}",
                    "content": f"[RESULT: {result}]",
                    "type": "result"
                })

            elif step.get("type") == "action":
                linearized.append({
                    "agent": "main",
                    "content": step["content"],
                    "type": "action"
                })

        return linearized

    def compute_sequence_loss(self, linearized: List[dict],
                            logits: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for linearized sequence (for gradient updates).

        Args:
            linearized: Linearized trajectory
            logits: Model logits for each token

        Returns:
            Loss tensor
        """
        # Standard language modeling loss on flattened sequence
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            self._tokenize_trajectory(linearized).view(-1)
        )
        return loss

    def _tokenize_trajectory(self, linearized: List[dict]) -> torch.Tensor:
        """Convert trajectory to token IDs."""
        text = " ".join([step["content"] for step in linearized])
        # Pseudo-code: actual implementation uses model's tokenizer
        return torch.tensor([])
```

### Step 3: Implement GRPO Training for Multi-Agent Trajectories

Extend GRPO to optimize linearized multi-agent sequences.

```python
# Multi-agent GRPO training
def train_multi_agent_grpo(
    model: nn.Module,
    dataset: List[dict],
    num_epochs: int = 10,
    group_size: int = 4,
    max_agents: int = 8
):
    """
    Train model with dynamic agent forking via GRPO.

    Args:
        model: Shared backbone for all agents
        dataset: Tasks to solve
        num_epochs: Training epochs
        group_size: GRPO group size
        max_agents: Max sub-agents per task
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        # Group tasks for GRPO
        for group_idx in range(0, len(dataset), group_size):
            group = dataset[group_idx:group_idx + group_size]
            group_rewards = []

            trajectories = []

            for task in group:
                # Execute main agent (with dynamic forking)
                main_agent = Agent(
                    task=task["description"],
                    can_fork=True,
                    max_agents=max_agents
                )

                # Main agent execution: may fork sub-agents
                main_trajectory, subagent_trajectories = main_agent.execute(model)

                # Linearize for GRPO
                linearizer = TrajectoryLinearizer()
                linearized = linearizer.linearize_execution(
                    main_trajectory,
                    subagent_trajectories
                )

                # Evaluate final result
                final_output = main_agent.result
                reward = evaluate_solution(final_output, task["target"])
                group_rewards.append(reward)

                trajectories.append((linearized, reward))

            # GRPO: compute advantages and update
            group_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            group_mean = group_tensor.mean()

            for linearized, reward in trajectories:
                advantage = reward - group_mean

                if advantage != 0:
                    # Compute loss on linearized trajectory
                    logits = model.forward(linearized)
                    loss = linearizer.compute_sequence_loss(linearized, logits)

                    # Scale by advantage
                    scaled_loss = loss * advantage
                    scaled_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}: avg_reward={group_tensor.mean():.4f}")

    return model

class Agent:
    def __init__(self, task: str, can_fork: bool = True,
                 max_agents: int = 8):
        """Main agent with optional forking."""
        self.task = task
        self.can_fork = can_fork
        self.max_agents = max_agents
        self.trajectory = []
        self.subagent_trajectories = {}
        self.result = None
        self.fork_count = 0

    def execute(self, model) -> Tuple[List[dict], Dict]:
        """Execute agent with possible sub-agent forking."""
        system_prompt = f"Solve this task: {self.task}"

        for step in range(10):  # Max reasoning steps
            # Generate thought
            thought = model.generate(system_prompt, max_tokens=256)
            self.trajectory.append({
                "type": "thought",
                "content": thought
            })

            # Decide if forking helps (learned by GRPO)
            if self.can_fork and self.fork_count < self.max_agents:
                fork_decision = should_fork(model, thought, self.task)

                if fork_decision:
                    # Create sub-agent for specific sub-task
                    subtask = extract_subtask(thought)
                    subagent = SubAgent(
                        task=subtask,
                        available_tools=["search", "calculate"],
                        context=self.task,
                        depth=1
                    )

                    result = subagent.execute(model)

                    self.fork_count += 1
                    self.subagent_trajectories[self.fork_count] = subagent.trajectory

                    self.trajectory.append({
                        "type": "fork",
                        "fork_id": self.fork_count,
                        "task": subtask,
                        "result": result
                    })

            # Try to solve (check if done)
            if is_task_complete(thought):
                self.result = extract_answer(thought)
                break

        return self.trajectory, self.subagent_trajectories
```

### Step 4: Evaluation and Metric Computation

Track multi-agent metrics.

```python
# Multi-agent evaluation
def evaluate_multi_agent_solution(
    result: str,
    target: str,
    num_agents_spawned: int = 0,
    total_tokens: int = 0
) -> dict:
    """
    Evaluate solution accounting for multi-agent aspects.

    Returns:
        Dict with correctness, efficiency, and agent metrics
    """
    correct = is_correct(result, target)

    # Reward includes solution quality
    base_reward = 1.0 if correct else 0.0

    # Penalize excessive agent spawning if not needed
    agent_penalty = 0.1 * num_agents_spawned if num_agents_spawned > 3 else 0.0

    # Reward efficient multi-agent use
    if num_agents_spawned > 0 and correct:
        efficiency_bonus = 0.1  # Bonus for parallelization

    return {
        "reward": base_reward - agent_penalty + (efficiency_bonus if num_agents_spawned > 0 else 0),
        "correct": correct,
        "agents_spawned": num_agents_spawned,
        "tokens_total": total_tokens
    }
```

## Practical Guidance

**When to use WideSeek:**
- Complex tasks with variable decomposition needs
- Scenarios where parallelization is beneficial
- Problems where agent depth/breadth tradeoff varies per instance
- End-to-end learning of orchestration is desired

**When not to use:**
- Simple single-path tasks
- Real-time systems where agent coordination overhead matters
- Scenarios requiring consistent agent behavior
- Tasks with fixed decomposition structures (use static multi-agent)

**Common Pitfalls:**
- Too many agents: Linearization becomes unwieldy; cap at 4-8
- Fork scheduling imbalance: Some tasks over-fork, others under-fork
- Communication gaps: Sub-agents can't see sibling results; pass context
- Training instability: Multi-agent GRPO more volatile; use smaller learning rates

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| max_agents | 4-8 | Higher = more parallelism; diminishing returns |
| agent_penalty | 0.05-0.2 | Penalize unnecessary forking |
| group_size | 4-8 | Standard GRPO setting |
| learning_rate | 1e-6 to 5e-6 | Lower than single-agent due to variance |

## Reference

See the full paper at: https://arxiv.org/abs/2602.02636

Key results: 6.36× more sub-agents than baselines via learned scheduling. WideSeekBench with 5,156 tasks. Code and trained 8B models released. Demonstrates breadth scaling as alternative to depth.
