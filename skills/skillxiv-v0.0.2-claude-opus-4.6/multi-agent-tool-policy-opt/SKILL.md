---
name: multi-agent-tool-policy-opt
title: "Multi-Agent Tool-Integrated Policy Optimization: Single LLM with Role-Specific RL Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.04678
keywords: [multi-agent, reinforcement-learning, tool-use, credit-assignment, prompt-engineering]
description: "Train planner and worker agent roles within a single LLM via role-specific prompts and RL, avoiding multi-instance overhead while preserving specialization. Trigger: improve tool-use planning robustness to noisy outputs without deploying separate models."
---

# Multi-Agent Tool-Integrated Policy Optimization (MATPO)

## Core Concept

MATPO enables a single LLM instance to embody **both planner and worker agent roles** through specialized prompts and reinforcement learning. Rather than deploying multiple LLM instances (expensive), the framework trains role-specific behaviors within one model using credit assignment across planner and worker rollouts. This maintains specialization benefits while reducing memory and compute overhead.

The key insight: A single model can learn distinct reasoning patterns when prompted appropriately and trained with role-aware reward signals.

## Architecture Overview

- **Dual Roles in Single Model**: Planner (strategic reasoning, tool selection) and Worker (execution, tool invocation)
- **Role-Specific Prompts**: Distinct system prompts guide behavior for each role
- **Unified RL Training**: Both roles optimized simultaneously via shared parameters
- **Credit Assignment Mechanism**: Principled reward flow from final outcomes to planner and worker decisions
- **Tool Response Integration**: Framework handles noisy/partial tool outputs robustly

## Implementation Steps

### 1. Design Role-Specific Prompts

Create distinct system prompts that guide the model toward planner vs. worker behavior. The prompts encode role expectations without requiring separate model instances.

```python
PLANNER_PROMPT = """You are a strategic planner agent. Your role:
1. Analyze the user request and decompose into sub-tasks
2. Decide which tools are needed and in what order
3. Specify parameters for each tool call
4. Handle errors from worker and re-plan if needed

Format your response as: PLAN: <step1> | PLAN: <step2> | ...
Then output: DELEGATE: <tool_call_json>"""

WORKER_PROMPT = """You are a worker agent executing tool calls. Your role:
1. Receive a tool call specification from the planner
2. Execute the tool with exact parameters
3. Process the output and report results
4. Flag errors for planner to handle

Format your response as: EXECUTING: <tool_name> | RESULT: <output>
If error occurs: ERROR: <error_type> with recommendation for replanning."""
```

### 2. Implement Rollout Collection with Role Switching

Collect training data by rolling out the model in both roles. The key is tracking which role is active during each decision for proper credit assignment.

```python
class DualAgentRollout:
    def __init__(self, model, tools_registry):
        self.model = model
        self.tools = tools_registry

    def collect_trajectory(self, task, max_turns=5):
        trajectory = {
            "planner_actions": [],
            "worker_actions": [],
            "rewards": [],
            "dones": []
        }

        context = f"Task: {task}\n"
        planner_history = ""
        worker_history = ""

        for turn in range(max_turns):
            # Planner decides next steps
            planner_output = self.model.generate(
                context + PLANNER_PROMPT + planner_history,
                max_tokens=512
            )

            # Parse planner's tool selection
            tool_calls = parse_tool_calls(planner_output)
            trajectory["planner_actions"].append({
                "turn": turn,
                "output": planner_output,
                "tools_selected": tool_calls
            })

            # Worker executes each tool call
            worker_results = []
            worker_output = ""

            for tool_call in tool_calls:
                try:
                    result = self.tools[tool_call["name"]](
                        **tool_call["args"]
                    )
                    worker_output += f"EXECUTING: {tool_call['name']}\n"
                    worker_output += f"RESULT: {result}\n"
                    worker_results.append(result)
                except Exception as e:
                    worker_output += f"ERROR: {str(e)}\n"

            trajectory["worker_actions"].append({
                "turn": turn,
                "output": worker_output,
                "results": worker_results
            })

            # Update context with worker results
            context += f"\nPlanner output:\n{planner_output}\n"
            context += f"\nWorker executed and got:\n{worker_output}\n"

            # Check if task is solved
            task_reward = evaluate_task_progress(
                worker_results, task
            )
            trajectory["rewards"].append(task_reward)

            if task_reward > COMPLETION_THRESHOLD:
                trajectory["dones"].append(True)
                break
            else:
                trajectory["dones"].append(False)

        return trajectory
```

### 3. Implement Principled Credit Assignment

Distribute rewards fairly between planner and worker based on their contributions. Poor planning decisions should reduce planner gradients; poor execution should reduce worker gradients.

```python
def credit_assignment(trajectory):
    """
    Assign credit to planner and worker based on trajectory outcomes.
    Key principle: planner error if correct tools weren't selected;
    worker error if tools failed to execute properly.
    """
    num_turns = len(trajectory["rewards"])
    final_reward = trajectory["rewards"][-1]

    # Backward credit flow
    planner_credits = []
    worker_credits = []

    for t in reversed(range(num_turns)):
        # Base reward for this turn
        turn_reward = trajectory["rewards"][t]

        # Planner credit: penalize if wrong tools selected
        if trajectory["planner_actions"][t]["tools_selected"]:
            # Check if worker achieved results with these tools
            worker_success = len(trajectory["worker_actions"][t]["results"]) > 0
            planner_credits.insert(0, turn_reward * 0.6 if worker_success else turn_reward * 0.2)
        else:
            planner_credits.insert(0, -0.1)  # Penalty for no action

        # Worker credit: penalize if execution failed
        if trajectory["worker_actions"][t]["results"]:
            # Full credit if results obtained
            worker_credits.insert(0, turn_reward * 0.4)
        else:
            # Reduced credit for errors
            worker_credits.insert(0, turn_reward * 0.1)

    return {
        "planner_credits": planner_credits,
        "worker_credits": worker_credits,
        "final_return": final_reward
    }
```

### 4. Train with Role-Aware RL

Use GRPO or PPO, but apply role-specific policy gradients. The model's parameters are shared, but gradient computation is role-aware.

```python
def train_dual_agent(model, dataset, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(config.num_epochs):
        for task in dataset:
            # Collect rollout with both roles
            trajectory = model.collect_trajectory(task)

            # Assign credit to each role
            credits = credit_assignment(trajectory)

            # Compute policy loss for planner actions
            planner_loss = 0
            for t, action in enumerate(trajectory["planner_actions"]):
                log_prob = compute_log_prob(
                    action["output"],
                    model
                )
                planner_loss += -credits["planner_credits"][t] * log_prob

            # Compute policy loss for worker actions
            worker_loss = 0
            for t, action in enumerate(trajectory["worker_actions"]):
                log_prob = compute_log_prob(
                    action["output"],
                    model
                )
                worker_loss += -credits["worker_credits"][t] * log_prob

            # Combined loss balances both roles
            total_loss = planner_loss + worker_loss
            total_loss += 0.01 * model.entropy_bonus()  # Encourage exploration

            # Update model
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % config.eval_frequency == 0:
                eval_performance = evaluate_on_benchmark(
                    model,
                    ["GAIA", "WebWalker", "FRAMES"]
                )
                print(f"Epoch {epoch}: {eval_performance}")
```

### 5. Handle Noisy Tool Outputs

Implement robust parsing and error recovery. When tools fail, the planner should learn to adjust.

```python
def robust_tool_execution(tool_call, fallback=True):
    """Execute with error handling and fallback strategies."""
    try:
        result = execute_tool(tool_call["name"], tool_call["args"])
        return {"status": "success", "result": result}
    except TimeoutError:
        if fallback:
            # Try with reduced parameters
            return {"status": "timeout", "result": None, "retry_hint": "reduce_scope"}
        else:
            return {"status": "failed", "result": None}
    except ValueError as e:
        # Invalid parameters - provide feedback for replanning
        return {"status": "invalid_params", "error": str(e)}
```

## Practical Guidance

**Hyperparameters:**
- **Planner role weight**: 0.6 (emphasis on planning quality)
- **Worker role weight**: 0.4 (execution is important but guided by planning)
- **Tool timeout**: 5-10 seconds per tool call
- **Credit discount**: 0.99 per turn
- **Learning rate**: 5e-5 for single-model training

**When to Use:**
- Complex multi-step tasks requiring planning + execution
- Noisy tool outputs that need adaptation
- Memory-constrained deployments where multi-agent LLM is prohibitive
- Tasks where role specialization matters

**When NOT to Use:**
- Simple single-tool tasks (overhead of role specialization not justified)
- Scenarios requiring separate fine-tuning for each role
- Tasks where planner/worker roles are not distinct

## Reference

[Multi-Agent Tool-Integrated Policy Optimization](https://arxiv.org/abs/2510.04678) — arXiv:2510.04678
