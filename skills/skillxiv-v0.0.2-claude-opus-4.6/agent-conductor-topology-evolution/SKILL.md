---
name: agent-conductor-topology-evolution
title: "AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.17100"
keywords: [Multi-Agent Coordination, Dynamic Topology, Code Generation, Graph-Based Planning, Adaptive Difficulty]
description: "Optimize multi-agent collaboration by learning task-specific interaction topologies. Use an LLM orchestrator to generate layered DAG topologies that adapt to inferred problem difficulty, treating agent interactions as a learned graph structure rather than fixed patterns."
---

# AgentConductor: Learned Topology Evolution for Multi-Agent Code Generation

Fixed agent collaboration patterns (e.g., always chain A→B→C) waste capacity on simple problems and under-scale complex ones. AgentConductor introduces a learnable orchestrator that dynamically generates task-adapted interaction topologies as directed acyclic graphs (DAGs). For each problem, the system infers difficulty, then constructs a topology balancing agent density with problem complexity—using sparse graphs for simple tasks, denser networks for complex ones.

The core innovation treats multi-agent interaction patterns as learned structures optimized via reinforcement learning. An LLM orchestrator observes problem specifications and task difficulty, then generates topology specifications in YAML format that deterministically describe how agents should interact.

## Core Concept

AgentConductor operates through three stages:

1. **Topology Specification**: Represent multi-agent interaction as a layered DAG with agents as nodes and information flow as edges
2. **Difficulty Inference**: Estimate task complexity from problem statement using heuristics or learned models
3. **Difficulty-Aware Generation**: Use orchestrator to generate topology density proportional to inferred difficulty
4. **Adaptive Refinement**: During execution, evolve topology across multiple turns based on intermediate results

## Architecture Overview

- **Input**: Problem statement (code generation task, requirements, constraints)
- **Difficulty Estimator**: Analyze problem to infer complexity (early/late refinement decision)
- **Orchestrator LLM**: Generate topology specification based on difficulty signal
- **Topology Parser**: Convert YAML to executable agent DAG
- **Multi-Turn Refinement**: Iteratively update topology based on execution feedback
- **Output**: Final code solution with traced agent interactions

## Implementation Steps

**Step 1: Design topology specification format**

Create a structured representation for multi-agent interaction patterns.

```python
# Example YAML topology specification
TOPOLOGY_TEMPLATE = """
agents:
  planner:
    role: high_level_planning
    capability_level: advanced
  coder:
    role: code_generation
    capability_level: standard
  reviewer:
    role: syntax_verification
    capability_level: standard
  debugger:
    role: error_correction
    capability_level: advanced

layers:
  planning:
    - planner
  generation:
    - coder
  validation:
    - reviewer
  refinement:
    - debugger

edges:
  # Layer transitions (information flow)
  planning_to_generation: planner -> coder
  generation_to_validation: coder -> reviewer
  validation_to_refinement: reviewer -> debugger

density: 0.75  # 0-1, higher = more agent cross-communication
"""

class TopologySpec:
    """Topology specification with structured access."""
    def __init__(self, yaml_string):
        self.spec = yaml.safe_load(yaml_string)
        self.agents = self.spec['agents']
        self.layers = self.spec['layers']
        self.edges = self.spec['edges']
        self.density = self.spec.get('density', 0.5)

    def get_execution_order(self):
        """Extract agent execution sequence from topology."""
        return [
            agent
            for layer in self.layers.values()
            for agent in layer
        ]

    def get_predecessors(self, agent):
        """Get agents that feed into this agent."""
        preds = []
        for edge_name, edge_spec in self.edges.items():
            src, dst = edge_spec.split(' -> ')
            if dst.strip() == agent:
                preds.append(src.strip())
        return preds
```

**Step 2: Estimate problem difficulty from specification**

Analyze problem statement to infer complexity and optimal topology density.

```python
def estimate_problem_difficulty(problem_spec):
    """
    Estimate task difficulty from problem characteristics.
    Returns: difficulty score in [0, 1], where 1 = hardest.
    """
    factors = {
        'code_length': 0.0,
        'control_flow_complexity': 0.0,
        'api_usage_count': 0.0,
        'constraint_count': 0.0
    }

    # Factor 1: Code length prediction
    # Longer code typically requires more reasoning
    estimated_lines = len(problem_spec.split('\n'))
    factors['code_length'] = min(estimated_lines / 100, 1.0)

    # Factor 2: Control flow complexity (keywords)
    control_flow_keywords = ['if', 'else', 'for', 'while', 'recursion', 'exception']
    cf_count = sum(problem_spec.lower().count(kw) for kw in control_flow_keywords)
    factors['control_flow_complexity'] = min(cf_count / 5, 1.0)

    # Factor 3: API/library usage (heuristic)
    api_keywords = ['numpy', 'pandas', 'sklearn', 'torch', 'tensorflow']
    api_count = sum(problem_spec.lower().count(api) for api in api_keywords)
    factors['api_usage_count'] = min(api_count / 3, 1.0)

    # Factor 4: Constraint count (heuristic)
    constraint_keywords = ['must', 'required', 'constraint', 'avoid', 'forbidden']
    constraint_count = sum(problem_spec.lower().count(ck) for ck in constraint_keywords)
    factors['constraint_count'] = min(constraint_count / 5, 1.0)

    # Weighted combination
    weights = {
        'code_length': 0.25,
        'control_flow_complexity': 0.35,
        'api_usage_count': 0.25,
        'constraint_count': 0.15
    }

    difficulty = sum(factors[k] * weights[k] for k in factors)
    return min(difficulty, 1.0)

def topology_density_from_difficulty(difficulty):
    """
    Map difficulty score to topology density.
    difficulty=0 (easy) → sparse topology (fewer agent interactions)
    difficulty=1 (hard) → dense topology (rich agent collaboration)
    """
    # Piecewise linear mapping
    if difficulty < 0.3:
        return 0.3  # Easy: sparse, just generation + basic validation
    elif difficulty < 0.7:
        return 0.5 + (difficulty - 0.3) * 0.25  # Medium: moderate density
    else:
        return 0.8  # Hard: dense, maximum collaboration

difficulty = estimate_problem_difficulty(problem_spec)
density = topology_density_from_difficulty(difficulty)
```

**Step 3: Orchestrator generates topology specification**

Use LLM to generate task-adapted topology conditioned on difficulty.

```python
def generate_topology_from_orchestrator(orchestrator_model, problem_spec,
                                       difficulty, density):
    """
    Generate topology specification via LLM orchestrator.
    Orchestrator was trained on 4,500 diverse interaction graphs.
    """
    prompt = f"""
Problem Specification:
{problem_spec[:500]}

Difficulty Level: {difficulty:.2f} (0=easy, 1=hard)
Target Topology Density: {density:.2f} (0=sparse, 1=dense)

Generate an optimal multi-agent interaction topology in YAML format.
Adjust density: lower density = fewer edges, higher density = more cross-agent communication.
Ensure the topology is a valid directed acyclic graph (DAG).

Output YAML topology:
"""

    # Forward pass through orchestrator LLM (trained with GRPO)
    topology_yaml = orchestrator_model.generate(
        prompt,
        max_tokens=300,
        temperature=0.5
    )

    return topology_yaml

# Example orchestrator output for medium-difficulty problem:
generated_topology = """
agents:
  planner:
    role: planning
    capability: advanced
  coder1:
    role: generation
    capability: standard
  coder2:
    role: generation
    capability: standard
  reviewer:
    role: validation
    capability: standard
  debugger:
    role: debugging
    capability: advanced

layers:
  planning:
    - planner
  generation:
    - coder1
    - coder2
  validation:
    - reviewer
  debugging:
    - debugger

edges:
  planner_to_coder1: planner -> coder1
  planner_to_coder2: planner -> coder2
  coder1_to_coder2: coder1 -> coder2
  coder2_to_reviewer: coder2 -> reviewer
  reviewer_to_debugger: reviewer -> debugger

density: 0.55
"""
```

**Step 4: Parse and execute topology**

Convert topology specification to executable agent interaction pattern.

```python
class TopologyExecutor:
    """Execute agents according to topology DAG."""
    def __init__(self, agents_dict, topology_spec):
        """
        agents_dict: {"agent_name": agent_instance}
        topology_spec: TopologySpec object
        """
        self.agents = agents_dict
        self.topology = topology_spec
        self.execution_context = {}

    def execute(self, problem_spec, max_iterations=3):
        """
        Execute agents in topological order.
        Pass intermediate results between agents according to DAG edges.
        """
        execution_order = self.topology.get_execution_order()
        results = {}

        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration + 1} ---")

            for agent_name in execution_order:
                agent = self.agents[agent_name]

                # Get predecessor outputs as input context
                predecessors = self.topology.get_predecessors(agent_name)
                predecessor_outputs = [
                    results.get(pred, "")
                    for pred in predecessors
                ]

                # Agent execution
                agent_input = f"""
Problem: {problem_spec}

Predecessor outputs: {' '.join(predecessor_outputs)}

Your task ({agent_name}):
"""

                agent_output = agent.execute(agent_input)
                results[agent_name] = agent_output

                print(f"{agent_name}: {agent_output[:100]}...")

        return results

# Usage
agents = {
    'planner': PlanningAgent(),
    'coder1': CodeGenerationAgent(),
    'coder2': CodeGenerationAgent(),
    'reviewer': ReviewAgent(),
    'debugger': DebuggingAgent()
}

executor = TopologyExecutor(agents, topology_spec)
final_code = executor.execute(problem_spec)
```

**Step 5: Iterative topology refinement**

Update topology across multiple turns based on intermediate failures.

```python
def refine_topology(executor, intermediate_results, failure_signals):
    """
    Adapt topology based on execution feedback.
    If reviewer finds errors, increase debugger prominence.
    If planner outputs are poor, revise planning layer.
    """
    current_spec = executor.topology.spec

    # Analyze failure patterns
    reviewer_issues = intermediate_results.get('reviewer', '')
    has_errors = 'error' in reviewer_issues.lower()

    if has_errors:
        # Increase debugger layer density
        current_spec['density'] = min(current_spec['density'] + 0.1, 1.0)

        # Add direct reviewer→debugger edge for quick feedback
        current_spec['edges']['direct_debug'] = 'reviewer -> debugger'

    # Reconstruct topology with refinements
    refined_yaml = yaml.dump(current_spec)
    refined_spec = TopologySpec(refined_yaml)

    return refined_spec

# Multi-turn refinement loop
executor = TopologyExecutor(agents, topology_spec)

for turn in range(3):  # 3 refinement turns
    results = executor.execute(problem_spec)
    failure_signals = analyze_failures(results)

    if len(failure_signals) > 0:
        topology_spec = refine_topology(executor, results, failure_signals)
        executor.topology = topology_spec
```

**Step 6: Evaluate with difficulty-aware metrics**

Measure performance improvement relative to problem difficulty.

```python
def evaluate_topology_performance(executor, test_problems):
    """
    Benchmark: solve rate vs. problem difficulty.
    Verify that difficult problems benefit most from denser topologies.
    """
    results = []

    for problem_spec in test_problems:
        difficulty = estimate_problem_difficulty(problem_spec)
        density = topology_density_from_difficulty(difficulty)

        # Generate and execute topology
        topology_yaml = generate_topology_from_orchestrator(
            orchestrator, problem_spec, difficulty, density
        )
        topology_spec = TopologySpec(topology_yaml)
        executor.topology = topology_spec

        solution = executor.execute(problem_spec)

        # Evaluate solution correctness
        is_correct = check_solution_correctness(solution['debugger'], problem_spec)

        results.append({
            'problem': problem_spec[:100],
            'difficulty': difficulty,
            'density': density,
            'correct': is_correct
        })

    # Aggregate by difficulty bins
    print("Performance by Difficulty:")
    for difficulty_bin in [0.25, 0.5, 0.75]:
        bin_results = [r for r in results
                      if abs(r['difficulty'] - difficulty_bin) < 0.1]
        solve_rate = np.mean([r['correct'] for r in bin_results])
        avg_density = np.mean([r['density'] for r in bin_results])
        print(f"  Difficulty {difficulty_bin:.2f}: {solve_rate*100:.0f}% solve rate, "
              f"avg density {avg_density:.2f}")
```

## Practical Guidance

**Hyperparameter Selection:**
- **Difficulty estimation weights**: Adjust [0.25, 0.35, 0.25, 0.15] based on domain (e.g., increase constraint weight for logic problems)
- **Density scaling**: Linear, quadratic, or sigmoid—test which best aligns topology to performance
- **Max refinement iterations**: 1-5. More iterations enable better adaptation; diminishing returns beyond 3.
- **Agent capability levels**: Use 2-3 levels (basic, standard, advanced); more levels inflate agent count

**When to Use:**
- Multi-agent systems with variable problem complexity
- Code generation, planning, or complex reasoning tasks
- Scenarios where different problems benefit from different collaboration patterns
- Settings where inference cost can be amortized across multi-turn reasoning

**When NOT to Use:**
- Single-agent problems (no multi-agent benefit)
- Fixed collaboration patterns proven optimal empirically
- Real-time systems requiring deterministic low latency
- Task settings with unknown difficulty profiles (hard to calibrate density mapping)

**Common Pitfalls:**
- **Topology explosion**: Too many agents or edges can make DAG intractable. Cap agent count at 5-10.
- **Difficulty miscalibration**: If all problems map to mid-range density, refinement provides no benefit. Validate difficulty distribution.
- **Circular dependencies**: If DAG parser creates cycles, execution hangs. Always validate topology as valid DAG.
- **Agent imbalance**: Some agents may become bottlenecks. Monitor per-agent execution time and redistribute work.

## Reference

arXiv: https://arxiv.org/abs/2602.17100
