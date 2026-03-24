---
name: multiverse-parallel-generation
title: "Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09991"
keywords: [parallel generation, MapReduce, non-autoregressive, task decomposition]
description: "Enable native parallel token generation in language models by implementing adaptive task decomposition and merge strategies, achieving 2x speedup with 1.87% performance gains."
---

# Multiverse: Parallel Generation in Language Models

## Core Concept

Multiverse implements a non-autoregressive generative model using MapReduce-style decomposition inside language models. Rather than generating tokens sequentially, the model learns when and how to parallelize subtasks, execute them in parallel, and merge results. A dedicated interpreter manages the transition between sequential and parallel generation modes.

## Architecture Overview

- **Adaptive task decomposition (Map)**: Model decides how to split reasoning into parallel subtasks
- **Parallel execution (Process)**: Subtasks generated independently without dependencies
- **Lossless result synthesis (Reduce)**: Merge subtask outputs back into main sequence
- **Multiverse Attention**: Separate parallel reasoning while maintaining causal compatibility
- **Automated data curation**: LLM-assisted pipeline transforms sequential chains to parallel structure

## Implementation

### Step 1: Design Parallel Attention Mechanism

Create attention that separates parallel vs sequential reasoning:

```python
class MultiverseAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768,
                 num_heads: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        # Mode selector: determines sequential vs parallel
        self.mode_selector = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)  # sequential, parallel
        )

    def forward(self, hidden_states: torch.Tensor,
               attention_mask: torch.Tensor = None,
               task_boundaries: torch.Tensor = None
               ) -> torch.Tensor:
        """Apply attention with mode switching."""

        batch_size, seq_len, _ = hidden_states.shape

        # Predict mode (sequential or parallel) for each position
        mode_logits = self.mode_selector(hidden_states)
        mode_probs = torch.softmax(mode_logits, dim=-1)

        # Sequential attention: standard causal attention
        sequential_attn = self._causal_attention(
            hidden_states,
            attention_mask
        )

        # Parallel attention: allow within-task but not across-task
        parallel_attn = self._parallel_attention(
            hidden_states,
            task_boundaries
        )

        # Mix attentions based on mode
        mode = torch.argmax(mode_probs, dim=-1)
        mode_float = mode_probs[:, :, 1]  # Parallel probability

        output = (sequential_attn * (1 - mode_float.unsqueeze(-1)) +
                 parallel_attn * mode_float.unsqueeze(-1))

        return output

    def _causal_attention(self, hidden_states: torch.Tensor,
                         attention_mask: torch.Tensor
                         ) -> torch.Tensor:
        """Standard causal self-attention."""

        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(
            torch.tensor(self.hidden_dim / self.num_heads)
        )

        # Apply causal mask
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )

        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask

        scores = scores + causal_mask.unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output

    def _parallel_attention(self, hidden_states: torch.Tensor,
                           task_boundaries: torch.Tensor
                           ) -> torch.Tensor:
        """Attention within parallel tasks only."""

        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)

        # Create task-aware mask
        # Only attend to tokens within same task
        seq_len = hidden_states.shape[1]
        task_mask = torch.zeros(seq_len, seq_len)

        if task_boundaries is not None:
            for task_id in task_boundaries.unique():
                task_indices = (task_boundaries == task_id).nonzero()
                for i in task_indices:
                    for j in task_indices:
                        task_mask[i, j] = 1.0

        # Compute attention with task mask
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(
            torch.tensor(self.hidden_dim / self.num_heads)
        )

        task_mask = task_mask.to(scores.device)
        scores = scores * task_mask.unsqueeze(0) + (
            (1 - task_mask) * float('-inf')
        ).unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output
```

### Step 2: Implement Task Decomposition (Map)

Learn to decompose reasoning into parallel subtasks:

```python
class TaskDecomposer(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768,
                 max_tasks: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tasks = max_tasks

        # Generate task descriptions and splits
        self.task_splitter = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, max_tasks)
        )

        # Task description generator
        self.task_describer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, hidden_dim)
        )

    def decompose(self, input_hidden: torch.Tensor,
                 question: str = None) -> dict:
        """Decompose problem into parallel tasks."""

        # Predict number of tasks
        task_logits = self.task_splitter(input_hidden)
        num_tasks = torch.argmax(task_logits, dim=-1) + 1

        # Generate task descriptions
        task_descriptions = self.task_describer(input_hidden)

        # Create task assignments for tokens
        task_boundaries = torch.arange(
            input_hidden.shape[1]
        ) % num_tasks

        return {
            "num_tasks": num_tasks,
            "task_descriptions": task_descriptions,
            "task_boundaries": task_boundaries
        }
```

### Step 3: Implement Result Merging (Reduce)

Synthesize parallel subtask outputs losslessly:

```python
class ResultMerger(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Merge gate: controls which task output to use
        self.merge_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

        # Final merge projection
        self.merge_proj = torch.nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def merge_task_outputs(self, task_outputs: list,
                          task_ids: list) -> torch.Tensor:
        """Combine parallel task outputs losslessly."""

        # Stack task outputs
        stacked = torch.stack(task_outputs)

        # Compute merge weights
        merge_inputs = []
        for i, output in enumerate(task_outputs):
            for j, other in enumerate(task_outputs):
                if i != j:
                    merge_inputs.append(
                        torch.cat([output, other], dim=-1)
                    )

        if merge_inputs:
            merge_scores = torch.stack([
                self.merge_gate(inp)
                for inp in merge_inputs
            ])
            merge_weights = torch.softmax(merge_scores, dim=0)

            merged = torch.sum(
                stacked * merge_weights.squeeze(-1).t().unsqueeze(-1),
                dim=0
            )
        else:
            merged = stacked[0]

        # Project merged output
        merged = self.merge_proj(merged)

        return merged
```

### Step 4: Training with Curator Pipeline

Prepare training data using automated pipeline:

```python
class MultiverseTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

    def curate_training_data(self,
                            sequential_chains: list,
                            curator_model) -> list:
        """Transform sequential reasoning to parallel structure."""

        parallel_data = []

        for chain in sequential_chains:
            # Use curator to identify parallelizable steps
            parallelizable_steps = curator_model.identify_parallel(
                chain
            )

            # Create parallel version
            parallel_version = {
                "original": chain,
                "tasks": [],
                "merge_order": []
            }

            # Group steps into parallel tasks
            for task_id, steps in parallelizable_steps.items():
                task = {
                    "id": task_id,
                    "steps": steps,
                    "output": execute_steps(steps)
                }
                parallel_version["tasks"].append(task)

            # Record merge order for Reduce stage
            parallel_version["merge_order"] = (
                curator_model.get_merge_order(
                    parallel_version["tasks"]
                )
            )

            parallel_data.append(parallel_version)

        return parallel_data

    def train_step(self, data_batch: list) -> float:
        """Train on parallel reasoning data."""

        total_loss = 0.0

        for sample in data_batch:
            # Decompose input
            decomposition = self.model.decomposer(
                sample["input"]
            )

            # Process tasks in parallel
            task_outputs = []
            for task_desc in decomposition["task_descriptions"]:
                output = self.model.process_task(task_desc)
                task_outputs.append(output)

            # Merge results
            merged = self.model.merger.merge_task_outputs(
                task_outputs,
                decomposition["task_boundaries"]
            )

            # Compare with target
            loss = torch.nn.functional.mse_loss(
                merged,
                sample["target_output"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_batch)
```

## Practical Guidance

**Task Decomposition**: Model learns when parallelization helps. Some problems naturally parallelize (independent subproblems), others require sequential reasoning.

**Merge Strategy**: Lossless merging ensures parallel paths don't lose information. The merge gate learns which task output should dominate at each position.

**Scaling Benefits**: 2x speedup achieved on mathematical reasoning with minimal quality loss (1.87% improvement on AIME). Gains improve with larger batch sizes.

**Data Curation**: Automated LLM-assisted pipeline generates parallel training data from sequential chains. Reduces human annotation burden.

**When to Apply**: Use Multiverse for reasoning tasks with inherent parallelism, or when inference latency reduction is critical while maintaining accuracy.

## Reference

Multiverse achieves native parallel generation by learning task decomposition and merge strategies. The MapReduce structure enables models to automatically manage parallelization without human specification. Key innovation: Multiverse Attention allows models to separate parallel reasoning while maintaining autoregressive compatibility for efficient decoding.
