---
name: alphapollo-reasoning
title: "AlphaApollo: Orchestrating Foundation Models and Tools for Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.06261
keywords: [agentic-reasoning, multi-turn-rl, tool-use, verification, iterative-refinement]
description: "Enable LLMs to solve complex problems through multi-turn agentic reasoning with tool-assisted verification and iterative refinement loops. Trigger: improve reasoning reliability on long-horizon tasks by combining RL with verification."
---

# AlphaApollo: Orchestrating Models and Tools for Reliable Reasoning

## Core Concept

AlphaApollo frames complex problem-solving as a multi-turn agentic process with three integrated levels: reasoning (multi-turn interactions), learning (turn-level RL), and evolution (multi-round refinement with verification). By separating actions from tool responses during RL training and implementing a propose-judge-update loop, the system achieves reliable reasoning on tasks requiring tool use, verification, and iterative correction.

The key insight: Decoupling actions from responses and learning from turn-level feedback enables models to build strategies that outlast individual tool failures.

## Architecture Overview

- **Multi-Turn Agentic Reasoning**: Model reasons in turns, calling tools and receiving responses
- **Turn-Level RL Training**: Optimize tool-use decisions at each turn, not just final outcome
- **Tool-Response Separation**: Distinguish between model actions and external tool results
- **Propose-Judge-Update Loop**: Generate solution → verify with tools → refine based on feedback
- **Long-Horizon Memory**: Track prior attempts and insights across refinement rounds

## Implementation Steps

### 1. Design the Agentic Interface

Define how the model interacts with tools and receives feedback.

```python
class AgentAction:
    """Represents a single turn's action."""
    def __init__(self, action_type, content, tool_call=None):
        self.action_type = action_type  # "reason", "call_tool", "output"
        self.content = content  # Text of reasoning or tool name
        self.tool_call = tool_call  # Tool parameters if applicable

    def to_prompt(self):
        if self.action_type == "call_tool":
            return f"TOOL_CALL: {self.tool_call['name']}({self.tool_call['args']})"
        else:
            return f"{self.action_type.upper()}: {self.content}"


class ToolResponse:
    """Represents tool execution result."""
    def __init__(self, tool_name, status, result, error=None):
        self.tool_name = tool_name
        self.status = status  # "success", "error", "timeout"
        self.result = result
        self.error = error

    def to_prompt(self):
        if self.status == "success":
            return f"Tool {self.tool_name} returned: {self.result}"
        else:
            return f"Tool {self.tool_name} error: {self.error}"


class MultiTurnTrajectory:
    """Tracks a complete problem-solving episode."""
    def __init__(self, problem):
        self.problem = problem
        self.turns = []  # List of (action, response) pairs
        self.turn_rewards = []  # Reward per turn
        self.final_reward = None
        self.solution = None

    def add_turn(self, action, response, immediate_reward=0):
        self.turns.append((action, response))
        self.turn_rewards.append(immediate_reward)

    def finalize(self, final_reward, solution):
        self.final_reward = final_reward
        self.solution = solution
```

### 2. Implement Multi-Turn Reasoning Loop

The agent reasons iteratively, calling tools when needed and refining based on responses.

```python
class MultiTurnReasoner:
    def __init__(self, model, tools_registry, max_turns=10):
        self.model = model
        self.tools = tools_registry
        self.max_turns = max_turns

    def reason(self, problem):
        """
        Multi-turn reasoning loop.

        Args:
            problem: Problem statement

        Returns:
            MultiTurnTrajectory with complete episode
        """
        trajectory = MultiTurnTrajectory(problem)
        context = f"Problem: {problem}\n\nReasoning:\n"

        for turn in range(self.max_turns):
            # Model generates reasoning and decides next action
            output = self.model.generate(
                context,
                max_tokens=256,
                stop_tokens=["TOOL_CALL:", "OUTPUT:"]
            )

            # Parse output to determine action type
            if "TOOL_CALL:" in output:
                # Extract tool call
                tool_spec = parse_tool_call(output)
                action = AgentAction(
                    "call_tool",
                    tool_spec["name"],
                    tool_call=tool_spec
                )

                # Execute tool
                try:
                    result = self.tools[tool_spec["name"]](**tool_spec["args"])
                    response = ToolResponse(
                        tool_spec["name"],
                        "success",
                        result
                    )
                    immediate_reward = 0.1  # Reward for attempting tool
                except Exception as e:
                    response = ToolResponse(
                        tool_spec["name"],
                        "error",
                        None,
                        str(e)
                    )
                    immediate_reward = -0.05

            elif "OUTPUT:" in output:
                # Final answer
                action = AgentAction("output", output.split("OUTPUT:")[-1].strip())
                response = None
                immediate_reward = 0  # Evaluated at finalization

                trajectory.add_turn(action, response, immediate_reward)
                break

            else:
                # Continue reasoning
                action = AgentAction("reason", output)
                response = None
                immediate_reward = 0

            trajectory.add_turn(action, response, immediate_reward)

            # Update context for next turn
            context += f"\n{action.to_prompt()}"
            if response:
                context += f"\n{response.to_prompt()}"

        return trajectory
```

### 3. Implement Turn-Level RL Training

Train the model using rewards at each turn, not just the final outcome.

```python
def compute_turn_rewards(trajectory, final_correctness_reward):
    """
    Distribute final reward across turns with discounting.

    Key principle: Successful tool use gets credit;
    poor decisions reduce earlier turn gradients.
    """
    num_turns = len(trajectory.turns)
    turn_rewards = []

    # Backward credit assignment with discount
    discount_factor = 0.99
    accumulated_reward = final_correctness_reward

    for t in reversed(range(num_turns)):
        action, response = trajectory.turns[t]

        # Tool calls get intermediate rewards if they succeeded
        if action.action_type == "call_tool":
            if response.status == "success":
                turn_reward = 0.1 * discount_factor ** (num_turns - t)
            else:
                turn_reward = -0.05 * discount_factor ** (num_turns - t)
        else:
            # Reasoning turns get discounted final reward
            turn_reward = accumulated_reward * (discount_factor ** (num_turns - t))

        turn_rewards.insert(0, turn_reward)
        accumulated_reward = turn_reward

    return turn_rewards


def train_on_trajectory(model, trajectory, optimizer, final_reward):
    """
    Compute RL loss and update model parameters.
    """
    # Compute per-turn rewards
    turn_rewards = compute_turn_rewards(trajectory, final_reward)

    total_loss = 0

    for t, (action, response) in enumerate(trajectory.turns):
        # Get log probability of this action
        # (computed during generation; cached in trajectory)
        log_prob = action.log_prob

        # Policy gradient: higher reward → higher gradient
        loss = -turn_rewards[t] * log_prob

        # Add entropy bonus to encourage exploration
        entropy_bonus = -0.01 * action.entropy

        total_loss += loss + entropy_bonus

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return total_loss.item()
```

### 4. Implement Propose-Judge-Update Loop

For complex problems, iterate: generate solution → verify → refine.

```python
class IterativeRefinement:
    def __init__(self, model, tools_registry, judge_model):
        self.reasoner = MultiTurnReasoner(model, tools_registry)
        self.judge = judge_model  # Separate model for verification

    def propose_judge_update(self, problem, max_rounds=3):
        """
        Iteratively refine solution through propose-judge-update cycles.

        Args:
            problem: Problem statement
            max_rounds: Maximum refinement iterations

        Returns:
            Best solution found across rounds
        """
        best_solution = None
        best_score = -1.0
        refinement_history = []

        for round_num in range(max_rounds):
            # PROPOSE: Generate solution
            trajectory = self.reasoner.reason(problem)
            solution = trajectory.solution

            # JUDGE: Verify solution quality
            verification_result = self.judge.verify(problem, solution)
            score = verification_result["score"]
            critique = verification_result["critique"]

            refinement_history.append({
                "round": round_num,
                "solution": solution,
                "score": score,
                "critique": critique
            })

            if score > best_score:
                best_score = score
                best_solution = solution

            # UPDATE: If not perfect, refine
            if score < 1.0:
                # Add critique to problem context for next round
                problem = f"{problem}\n\nPrior attempt critique: {critique}\nRefined approach:"

            else:
                # Perfect solution found
                break

        return {
            "best_solution": best_solution,
            "best_score": best_score,
            "refinement_history": refinement_history
        }
```

### 5. Full Training Loop with Multi-Round Evolution

Combine multi-turn RL with propose-judge-update refinement.

```python
def train_alphapollo(model, judge_model, dataset, config):
    """
    Full AlphaApollo training pipeline.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    reasoner = MultiTurnReasoner(model, tools_registry=config.tools)
    refiner = IterativeRefinement(model, config.tools, judge_model)

    for epoch in range(config.num_epochs):
        for problem_id, problem in enumerate(dataset):
            # Phase 1: Multi-turn agentic reasoning
            trajectory = reasoner.reason(problem)

            # Phase 2: Multi-round agentic evolution (propose-judge-update)
            evolution_result = refiner.propose_judge_update(problem)

            # Phase 3: Train on trajectories
            final_reward = evolution_result["best_score"]

            loss = train_on_trajectory(
                model,
                trajectory,
                optimizer,
                final_reward
            )

            # Logging
            if problem_id % 100 == 0:
                print(f"Epoch {epoch}, Problem {problem_id}: "
                      f"loss={loss:.4f}, best_score={final_reward:.4f}")

    return model
```

## Practical Guidance

**Hyperparameters:**
- **Max turns per reasoning**: 5-10 (balance quality vs. latency)
- **Max refinement rounds**: 2-3 (diminishing returns)
- **Discount factor**: 0.99 (credit assignment)
- **Entropy bonus weight**: 0.01 (exploration)
- **Tool timeout**: 5-10 seconds

**When to Use:**
- Complex multi-step problems requiring tool use
- Scenarios where verification of solutions is possible
- Tasks where iterative refinement improves quality
- Long-horizon planning with multiple decision points

**When NOT to Use:**
- Single-turn tasks with no tool use
- Domains without reliable verification mechanisms
- Streaming applications with strict latency constraints
- Tasks where intermediate steps don't provide learning signal

## Reference

[AlphaApollo: Orchestrating Foundation Models and Tools for Agentic Reasoning](https://arxiv.org/abs/2510.06261) — arXiv:2510.06261
