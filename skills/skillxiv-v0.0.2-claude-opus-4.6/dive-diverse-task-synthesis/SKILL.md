---
name: dive-diverse-task-synthesis
title: "DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.11076"
keywords: [Task Synthesis, Tool Use, Evidence-Based, Agentic Training, Generalization]
description: "Synthesize diverse, verifiable training tasks by executing real tools first, then reverse-deriving tasks from execution traces. Ensure diversity across tools and reasoning patterns while maintaining grounding by construction."
---

# Technique: Evidence-First Task Synthesis via Reverse Derivation

Traditional task synthesis generates queries first, then validates them—brittle and often fails. DIVE inverts this: it executes diverse tools in real environments to generate *evidence*, then reverse-derives tasks strictly entailed by these executions. This ensures tasks are executable by construction and verifiable from observable tool outputs.

The approach guarantees structural diversity (heterogeneous tool-use patterns) and validity without simulation artifacts.

## Core Concept

DIVE operates through three phases:

1. **Diverse Resource Preparation**: Construct decoupled pools of tools, domain concepts, and query exemplars
2. **Evidence-Driven Synthesis Loop**: Execute tools to gather traces, then reverse-derive tasks from evidence
3. **Agentic Training**: Use synthesized tasks for supervised finetuning followed by RL

This ensures every generated task is grounded in real tool behavior with verifiable reference answers.

## Architecture Overview

- **Tool pool**: 373+ validated tools across domains
- **Domain concept database**: Topic-specific seed concepts for diversity
- **Query exemplar library**: Structural priors for task patterns
- **Execution engine**: Real tool invocation
- **Task derivation module**: Reverse-derives tasks from traces
- **Verification system**: Validates task-trace entailment

## Implementation Steps

### Step 1: Prepare Diverse Resource Pools

Construct decoupled resource sets for breadth and coverage.

```python
import json
from collections import defaultdict

class DiverseResourcePool:
    def __init__(self):
        self.tool_pool = {}
        self.domain_concepts = defaultdict(list)
        self.query_exemplars = []

    def load_tool_pool(self, tools_file):
        """Load validated tools from registry."""
        with open(tools_file, 'r') as f:
            tools = json.load(f)

        for tool_name, tool_spec in tools.items():
            self.tool_pool[tool_name] = {
                'category': tool_spec.get('category'),
                'inputs': tool_spec.get('input_schema'),
                'outputs': tool_spec.get('output_schema'),
                'description': tool_spec.get('description'),
                'executor': self._get_tool_executor(tool_name)
            }

    def _get_tool_executor(self, tool_name):
        """Get callable executor for tool."""
        # Return function that invokes tool
        def executor(args):
            # Real tool invocation
            pass
        return executor

    def register_domain_concepts(self, domain, concepts):
        """Register topic-specific concepts for diversity."""
        for concept in concepts:
            self.domain_concepts[domain].append({
                'concept': concept,
                'frequency': 0  # Track usage for diversity balancing
            })

    def add_query_exemplars(self, exemplars):
        """Add structural patterns for task generation."""
        for exemplar in exemplars:
            self.query_exemplars.append({
                'pattern': exemplar['pattern'],
                'tool_sequence': exemplar.get('tools', []),
                'reasoning_type': exemplar.get('reasoning_type')  # e.g., 'retrieval-only', 'analyze'
            })
```

### Step 2: Execute Tools and Gather Evidence

Run diverse tools to generate execution traces as ground truth.

```python
class ToolExecutionEngine:
    def __init__(self, resource_pool):
        self.resource_pool = resource_pool

    def execute_tool_diversity(self, num_executions=1000, domain_balance=True):
        """
        Execute tools to generate diverse evidence traces.

        domain_balance: True to sample evenly across domains
        """
        execution_traces = []

        domains = list(self.resource_pool.domain_concepts.keys())
        domain_counts = {d: 0 for d in domains}

        for exec_idx in range(num_executions):
            # Select tool with balanced domain sampling
            if domain_balance:
                # Pick undersampled domain
                domain = min(domains, key=lambda d: domain_counts[d])
            else:
                domain = random.choice(domains)

            tools_in_domain = [
                t for t, spec in self.resource_pool.tool_pool.items()
                if spec['category'] == domain
            ]

            if not tools_in_domain:
                continue

            # Select tool
            tool_name = random.choice(tools_in_domain)
            tool_spec = self.resource_pool.tool_pool[tool_name]

            # Generate valid arguments for tool
            tool_args = self._generate_valid_arguments(tool_spec)

            # Execute tool
            try:
                result = tool_spec['executor'](tool_args)

                trace = {
                    'tool': tool_name,
                    'arguments': tool_args,
                    'output': result,
                    'domain': domain,
                    'success': result is not None
                }

                execution_traces.append(trace)
                domain_counts[domain] += 1

            except Exception as e:
                print(f"Execution failed for {tool_name}: {e}")

        return execution_traces

    def _generate_valid_arguments(self, tool_spec):
        """Generate arguments matching tool input schema."""
        args = {}
        for param_name, param_schema in tool_spec['inputs'].items():
            if param_schema['type'] == 'string':
                args[param_name] = self._sample_string_value(
                    param_schema.get('description')
                )
            elif param_schema['type'] == 'number':
                args[param_name] = random.uniform(
                    param_schema.get('minimum', 0),
                    param_schema.get('maximum', 100)
                )

        return args

    def _sample_string_value(self, description):
        """Sample realistic string based on parameter description."""
        # Simplified: use description to infer domain
        return f"sample_{hash(description) % 1000}"
```

### Step 3: Reverse-Derive Tasks from Evidence

Generate task descriptions that are strictly entailed by execution traces.

```python
class TaskDeriver:
    def __init__(self, llm_model):
        self.llm = llm_model

    def derive_task_from_trace(self, execution_trace):
        """
        Generate task description from execution trace.

        Ensures task is strictly entailed by observable outputs.
        """
        tool_name = execution_trace['tool']
        tool_args = execution_trace['arguments']
        output = execution_trace['output']

        # Prompt LLM to derive task
        prompt = f"""Given a tool execution trace, derive a task description that is strictly entailed.

Tool: {tool_name}
Arguments: {json.dumps(tool_args)}
Output: {json.dumps(output)}

Generate a task description that:
1. Can be accomplished using {tool_name}
2. Is verifiable from the output
3. Matches the provided arguments as natural parameters

Task:"""

        task_description = self.llm.generate(prompt, max_tokens=150)

        return task_description.strip()

    def synthesize_diverse_task_sequence(
        self,
        execution_traces,
        num_tasks=100,
        ensure_diversity=True
    ):
        """
        Generate task-answer pairs from traces, with diversity guarantees.
        """
        synthesized_tasks = []

        if ensure_diversity:
            # Group traces by tool to ensure coverage
            traces_by_tool = defaultdict(list)
            for trace in execution_traces:
                traces_by_tool[trace['tool']].append(trace)

            # Sample evenly from tools
            tools = list(traces_by_tool.keys())
            num_per_tool = num_tasks // len(tools)

            for tool in tools:
                tool_traces = traces_by_tool[tool]
                sampled = random.sample(
                    tool_traces,
                    min(num_per_tool, len(tool_traces))
                )

                for trace in sampled:
                    task = self.derive_task_from_trace(trace)

                    synthesized_tasks.append({
                        'task_description': task,
                        'tool': trace['tool'],
                        'arguments': trace['arguments'],
                        'reference_answer': trace['output'],
                        'domain': trace['domain']
                    })
        else:
            # Sample without diversity guarantee
            sampled_traces = random.sample(
                execution_traces,
                min(num_tasks, len(execution_traces))
            )

            for trace in sampled_traces:
                task = self.derive_task_from_trace(trace)
                synthesized_tasks.append({
                    'task_description': task,
                    'tool': trace['tool'],
                    'arguments': trace['arguments'],
                    'reference_answer': trace['output'],
                    'domain': trace['domain']
                })

        return synthesized_tasks
```

### Step 4: Verify Task-Trace Entailment

Validate that generated tasks are indeed entailed by evidence.

```python
class TaskVerifier:
    def __init__(self, llm_model):
        self.llm = llm_model

    def verify_task_answer_pair(self, task, reference_answer):
        """
        Verify that reference answer satisfies task requirements.
        """
        prompt = f"""Task: {task['task_description']}

Reference Answer: {json.dumps(reference_answer)}

Does the reference answer correctly accomplish the task? Answer 'Yes' or 'No'."""

        response = self.llm.generate(prompt, max_tokens=10)

        is_valid = 'yes' in response.lower()

        return is_valid

    def verify_task_batch(self, synthesized_tasks, sample_fraction=0.1):
        """
        Spot-check synthesized tasks for quality.
        """
        num_to_check = max(1, int(len(synthesized_tasks) * sample_fraction))
        sampled = random.sample(synthesized_tasks, num_to_check)

        valid_count = 0
        for task in sampled:
            is_valid = self.verify_task_answer_pair(task, task['reference_answer'])
            if is_valid:
                valid_count += 1

        validity_rate = valid_count / len(sampled)

        return {
            'validity_rate': validity_rate,
            'checked': len(sampled),
            'valid': valid_count
        }
```

### Step 5: Training Pipeline with Synthesized Tasks

Fine-tune agent on synthesized tasks, then apply RL.

```python
def train_agent_with_div_tasks(
    model,
    synthesized_tasks,
    num_sft_epochs=3,
    num_rl_steps=10000
):
    """
    Full training pipeline: SFT + RL with synthesized tasks.
    """
    # Phase 1: Supervised Fine-Tuning
    sft_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_sft_epochs):
        total_loss = 0

        for batch in batch_iter(synthesized_tasks, batch_size=32):
            # Forward pass
            outputs = model(batch['task_description'])

            # Loss: cross-entropy with reference answers
            loss = compute_task_loss(outputs, batch['reference_answer'])

            sft_optimizer.zero_grad()
            loss.backward()
            sft_optimizer.step()

            total_loss += loss.item()

        print(f"SFT Epoch {epoch + 1}: Loss={total_loss / len(synthesized_tasks):.4f}")

    # Phase 2: Reinforcement Learning
    rl_optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

    for rl_step in range(num_rl_steps):
        # Sample task batch
        batch = random.sample(synthesized_tasks, batch_size=32)

        # Generate trajectories
        trajectories = [model.sample_trajectory(t['task_description']) for t in batch]

        # Evaluate using reference answers (verifiable)
        rewards = [
            1.0 if trajectory == t['reference_answer'] else 0.0
            for trajectory, t in zip(trajectories, batch)
        ]

        # GRPO-style RL update (simplified)
        loss = compute_rl_loss(trajectories, rewards, model)

        rl_optimizer.zero_grad()
        loss.backward()
        rl_optimizer.step()

        if rl_step % 1000 == 0:
            print(f"RL Step {rl_step}: Avg Reward={sum(rewards) / len(rewards):.2f}")

    return model
```

## Practical Guidance

**When to Use:**
- Generating training data for tool-use agents
- Scenarios where task diversity is critical for generalization
- Domains with well-defined, executable tools
- When task verification from tool outputs is feasible

**When NOT to Use:**
- Tasks requiring human judgment (not verifiable from tool outputs)
- Extremely diverse domains where tool coverage is sparse
- Real-time synthesis requirements (execution-based synthesis is slow)

**Hyperparameter Tuning:**
- **num_executions**: 500-2000; more coverage, diminishing returns
- **num_per_tool**: 10-50; ensure balanced tool representation
- **verify_sample_fraction**: 0.1-0.2; spot-check quality
- **SFT vs RL budget**: 50-50 or 70-30 depending on task difficulty

**Common Pitfalls:**
- Over-sampling from popular tools (neglecting long-tail coverage)
- Insufficient task derivation diversity (same template for all)
- Verification too lenient/strict (affects training quality)
- Tool failures reducing available evidence (fallback to simulation)

## Reference

[DIVE paper on arXiv](https://arxiv.org/abs/2603.11076)
