---
name: tape-tool-guided-adaptive-planning
title: "TAPE: Tool-Guided Adaptive Planning and Constrained Execution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.19633"
keywords: [agent planning, constraint satisfaction, tool use, LLM agents, planning under constraints]
description: "Improve LLM agents operating under strict feasibility constraints (budget limits, tool usage caps) by separating planning from execution. Generate multiple candidate plans, merge into plan graph, then use external solver (ILP) to find optimal feasible path. Constrained decoding forces execution of planned actions, eliminating sampling errors. Adaptive replanning handles observation surprises. Achieves 21+ pp improvements on constrained tasks vs. ReAct."
---

# TAPE: Constraint-Aware Planning for Tool-Using Agents

Language model agents often fail under strict feasibility constraints—budget limits, tool usage quotas, or action count restrictions. Two failure modes dominate:

**Planning Errors**: The agent's internal reasoning suggests non-viable action sequences that violate constraints or become infeasible mid-trajectory.

**Sampling Errors**: Even with correct planning, stochastic token generation produces actions different from what was intended, causing the agent to execute unplanned detours.

Standard frameworks like ReAct treat constraint satisfaction reactively—only checking feasibility after deciding. Better to embed constraint awareness into the planning phase, generating feasible action sequences upfront.

## Core Concept

TAPE separates planning into three phases:

1. **Plan Graph Generation**: Multiple candidate plans from the agent, merged into a directed graph where nodes are states and edges are actions
2. **Solver-Based Path Selection**: External optimizer (ILP) finds the optimal feasible path through the graph, accounting for constraints
3. **Constrained Execution**: Rather than free generation, the agent is constrained to output only the planned next action using controlled decoding

When observations diverge from predictions, the system replans on updated state, maintaining feasibility throughout.

## Architecture Overview

- **Multi-Plan Generator**: Sample K candidate plans from agent without execution
- **Plan Graph Merger**: Combine plans into DAG, deduplicating states and merging equivalent actions
- **Cost Predictor**: Estimate per-action cost (budget consumed, tool calls used, steps taken)
- **Constraint Solver**: ILP formulation to find lowest-cost feasible path given budget constraints
- **Constrained Decoder**: Force agent to output only next planned action using prefix constraints or token masking
- **Observation Tracker**: Monitor actual outcomes vs. predicted; trigger replanning if divergence exceeds threshold
- **Adaptive Replanner**: Regenerate plans from new observation state if replanning triggered

## Implementation

Generate multiple plans and build plan graph:

```python
def generate_candidate_plans(agent, state, num_candidates=5):
    """
    Sample multiple plans from the agent without execution.
    Returns list of (action_sequence, cost_estimate)
    """
    plans = []

    for _ in range(num_candidates):
        # Generate plan using chain-of-thought
        prompt = f"""
State: {state}
Plan the action sequence to reach the goal.
List each action on a new line.

Actions:
"""
        plan_text = agent.generate(prompt, temperature=0.7, max_tokens=200)
        actions = parse_actions(plan_text)

        # Estimate cost for this plan
        cost = estimate_plan_cost(actions, state)

        plans.append((actions, cost))

    return plans

def merge_plans_into_graph(plans):
    """
    Merge multiple plans into a single DAG for efficient pathfinding.
    """
    import networkx as nx

    graph = nx.DiGraph()
    state_nodes = {}
    initial_state = None

    for plan_idx, (actions, _) in enumerate(plans):
        current_state = 'initial'
        if initial_state is None:
            initial_state = current_state

        for action in actions:
            # Add state node if new
            if current_state not in state_nodes:
                state_nodes[current_state] = len(state_nodes)
                graph.add_node(current_state)

            # Execute action to get next state
            next_state = execute_action_simulation(current_state, action)

            # Add edge
            action_cost = estimate_action_cost(action)
            graph.add_edge(current_state, next_state, action=action, cost=action_cost)

            current_state = next_state

    return graph, initial_state

def find_feasible_path_ilp(graph, initial_state, goal_state, budget_constraint):
    """
    Use ILP to find optimal feasible path respecting constraints.
    """
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum

    # Create ILP problem
    prob = LpProblem("PathPlanning", LpMinimize)

    # Decision variables: x[u,v] = 1 if edge (u,v) is in solution
    edge_vars = {}
    for u, v in graph.edges():
        edge_vars[(u, v)] = LpVariable(f"edge_{u}_{v}", cat='Binary')

    # Objective: minimize cost (or maximize success probability)
    objective = lpSum([
        edge_vars[(u, v)] * graph[u][v]['cost']
        for u, v in graph.edges()
    ])
    prob += objective

    # Constraint 1: Flow conservation (path from start to goal)
    for node in graph.nodes():
        if node == initial_state:
            # Source: out-degree = 1
            prob += lpSum([edge_vars[(initial_state, v)] for v in graph.successors(initial_state)]) == 1
        elif node == goal_state:
            # Sink: in-degree = 1
            prob += lpSum([edge_vars[(u, goal_state)] for u in graph.predecessors(goal_state)]) == 1
        else:
            # Intermediate: in-degree = out-degree
            in_edges = lpSum([edge_vars[(u, node)] for u in graph.predecessors(node)])
            out_edges = lpSum([edge_vars[(node, v)] for v in graph.successors(node)])
            prob += in_edges == out_edges

    # Constraint 2: Budget constraint
    total_cost = lpSum([
        edge_vars[(u, v)] * graph[u][v]['cost']
        for u, v in graph.edges()
    ])
    prob += total_cost <= budget_constraint

    # Solve
    prob.solve(solver=None, msg=False)  # Uses default solver

    # Extract path
    path = []
    current = initial_state
    while current != goal_state:
        for v in graph.successors(current):
            if edge_vars[(current, v)].varValue == 1:
                action = graph[current][v]['action']
                path.append(action)
                current = v
                break

    return path
```

Implement constrained execution with forced action generation:

```python
def execute_with_constrained_decoding(agent, state, plan, idx):
    """
    Execute next planned action using constrained decoding.
    Prevents agent from deviating from plan due to sampling randomness.
    """
    planned_action = plan[idx]

    # Convert action to token sequence that agent would produce
    action_tokens = tokenize_action(planned_action)

    # Constrained decoding: force first token to match planned action
    prompt = f"State: {state}. Next action:"

    # Use prefix constraint to force specific action
    generated = agent.generate(
        prompt,
        prefix=action_tokens,  # Force generation to start with these tokens
        temperature=0.0  # Deterministic
    )

    # Validate execution
    actual_outcome = execute_action(state, planned_action)

    return actual_outcome, planned_action
```

Implement adaptive replanning:

```python
def tape_agent_loop(
    agent, initial_state, goal_state, budget, max_replans=3
):
    """
    Main TAPE loop with adaptive replanning.
    """
    current_state = initial_state
    current_cost = 0
    num_replans = 0

    while current_state != goal_state and num_replans < max_replans:
        # Generate candidate plans
        plans = generate_candidate_plans(agent, current_state, num_candidates=5)

        # Build graph and find feasible path
        graph, _ = merge_plans_into_graph(plans)
        plan = find_feasible_path_ilp(
            graph, current_state, goal_state,
            budget - current_cost
        )

        if not plan:
            print("No feasible plan found!")
            break

        # Execute plan with constrained decoding
        for step_idx, planned_action in enumerate(plan):
            outcome, action = execute_with_constrained_decoding(
                agent, current_state, plan, step_idx
            )

            # Check if outcome matches prediction
            predicted_next = simulate_action(current_state, action)
            divergence = compare_outcomes(outcome, predicted_next)

            if divergence > 0.3:  # Threshold for replanning
                print(f"Observation divergence at step {step_idx}; replanning...")
                num_replans += 1
                current_state = outcome['new_state']
                current_cost = outcome['cost_so_far']
                break  # Exit execution, start new plan
            else:
                current_state = outcome['new_state']
                current_cost = outcome['cost_so_far']

    return current_state, current_cost
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Candidate plans | 5 | 3–7 plans; more exploration with higher count |
| Budget constraint | Task-specific | Set to task-specified limit; tighter constraints need more planning |
| Divergence threshold | 0.3 | Replanning triggered when actual vs. predicted diverges by >30% |
| Max replans | 3 | Limit to prevent infinite loops; set per task complexity |
| Solver type | ILP | Use simplified greedy for real-time; ILP for offline planning |

**When to use**: For agent tasks with strict resource constraints (budget limits, API quotas) where infeasible actions are costly or impossible.

**When not to use**: For open-ended tasks without constraints or where planning overhead exceeds benefit.

**Common pitfalls**:
- Planning assumes deterministic outcomes; use robust planning (worst-case costs) when outcomes are stochastic
- Constrained decoding may fail if agent unfamiliar with action format; ensure actions have high model likelihood
- Divergence thresholds too sensitive; calibrate on dev trajectories to find stability point

## Reference

TAPE achieves 21+ percentage point improvements on constrained tasks (Sokoban, ALFWorld) compared to ReAct. The separation of planning and execution enables robust constraint satisfaction and enables safe, predictable agent behavior.
