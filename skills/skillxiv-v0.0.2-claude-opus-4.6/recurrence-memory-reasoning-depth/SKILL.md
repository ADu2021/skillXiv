---
name: recurrence-memory-reasoning-depth
title: "Extending Reasoning Depth with Recurrence, Memory and Test-Time Compute"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.16745
keywords: [reasoning-depth, recurrent-computation, external-memory, test-time-scaling, sequential-reasoning]
description: "Extend neural network reasoning capabilities through recurrence (repeated computation cycles), external memory (intermediate state storage), and test-time compute scaling for multi-step reasoning."
---

# Extending Reasoning Depth with Recurrence, Memory, and Compute Scaling

## Core Concept

Multi-step reasoning requires computational depth beyond what fixed-depth architectures provide. This skill combines three mechanisms: recurrence (allowing repeated computation passes), external memory (storing intermediate states), and test-time compute scaling (allocating more cycles during inference) to extend effective reasoning depth. Studies on cellular automata and Boolean functions show that while models can memorize next-step predictions, multi-step reasoning requires explicit recurrent computation.

## Architecture Overview

- **Recurrent Processing**: Iterative computation allowing state evolution
- **External Memory System**: Storage for intermediate reasoning states
- **Test-Time Compute Allocation**: Dynamic cycle budget during inference
- **Depth Extension**: Exceeding architectural layer count
- **Sequential Rule Learning**: Multi-step state transition modeling

## Implementation Steps

### 1. Implement Recurrent Computation Unit

Create iterative computation mechanism:

```python
import torch
import torch.nn as nn
from typing import Tuple, List, Dict

class RecurrentReasoningUnit(nn.Module):
    """Iterative computation for multi-step reasoning."""

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 512,
        num_recurrence_steps: int = 5
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_recurrence_steps = num_recurrence_steps

        # Core recurrent computation
        self.recurrent_cell = nn.GRUCell(state_dim, hidden_dim)

        # State transformation
        self.state_proj = nn.Linear(hidden_dim, state_dim)

        # Gating mechanism to control information flow
        self.input_gate = nn.Linear(state_dim, hidden_dim)
        self.forget_gate = nn.Linear(state_dim, hidden_dim)
        self.output_gate = nn.Linear(state_dim, hidden_dim)

    def forward(
        self,
        initial_state: torch.Tensor,  # (batch, state_dim)
        num_steps: int = None,
        external_memory: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Iterate computation for multiple recurrence steps.
        """
        if num_steps is None:
            num_steps = self.num_recurrence_steps

        batch_size = initial_state.shape[0]
        hidden_state = torch.zeros(batch_size, self.recurrent_cell.hidden_size)
        current_state = initial_state

        state_trajectory = [current_state]

        for step in range(num_steps):
            # Compute gating
            input_g = torch.sigmoid(self.input_gate(current_state))
            forget_g = torch.sigmoid(self.forget_gate(current_state))
            output_g = torch.sigmoid(self.output_gate(current_state))

            # Recurrent update
            hidden_state = self.recurrent_cell(current_state * input_g, hidden_state)

            # Project back to state space
            new_state = self.state_proj(hidden_state * output_g)

            # Residual connection
            current_state = current_state * forget_g + new_state * (1 - forget_g)

            # Store trajectory
            state_trajectory.append(current_state)

        return current_state, state_trajectory

    def compute_reasoning_depth(
        self,
        num_steps: int
    ) -> float:
        """
        Compute effective reasoning depth.
        Recurrent models achieve depth > number of layers.
        """
        num_layers = 1  # Single recurrent unit
        effective_depth = num_layers * num_steps
        return effective_depth
```

### 2. Implement External Memory System

Store and retrieve intermediate states:

```python
class ExternalMemory(nn.Module):
    """External memory for storing reasoning states."""

    def __init__(
        self,
        state_dim: int = 256,
        memory_size: int = 32,
        num_read_heads: int = 4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads

        # Memory storage
        self.register_buffer("memory", torch.randn(memory_size, state_dim) * 0.01)

        # Read/write mechanisms
        self.write_controller = nn.Linear(state_dim, memory_size)
        self.read_query = nn.Linear(state_dim, state_dim)
        self.key_proj = nn.Linear(state_dim, state_dim)

    def write_to_memory(
        self,
        state: torch.Tensor,  # (batch, state_dim)
        write_addr: torch.Tensor = None  # Which locations to write
    ) -> torch.Tensor:
        """
        Write state to memory with soft addressing.
        """
        batch_size = state.shape[0]

        if write_addr is None:
            # Compute write addresses using attention
            write_logits = self.write_controller(state)  # (batch, memory_size)
            write_weights = torch.softmax(write_logits, dim=-1)
        else:
            write_weights = write_addr

        # Soft write: weighted update
        for i in range(batch_size):
            for j in range(self.memory_size):
                # Gated write: preserve old + add new
                self.memory[j] = 0.9 * self.memory[j] + 0.1 * write_weights[i, j] * state[i]

        return write_weights

    def read_from_memory(
        self,
        query: torch.Tensor,  # (batch, state_dim)
        num_read_heads: int = None
    ) -> torch.Tensor:
        """
        Read from memory using content-based addressing.
        """
        if num_read_heads is None:
            num_read_heads = self.num_read_heads

        batch_size = query.shape[0]

        # Compute read weights using similarity
        query_proj = self.read_query(query)  # (batch, state_dim)
        key_proj = self.key_proj(self.memory)  # (memory_size, state_dim)

        # Content-based addressing with multiple heads
        read_output = torch.zeros(batch_size, self.state_dim)

        for head in range(num_read_heads):
            # Partition query and keys for this head
            head_query = query_proj[:, :self.state_dim // num_read_heads]
            head_keys = key_proj[:, :self.state_dim // num_read_heads]

            # Compute attention
            scores = torch.matmul(head_query, head_keys.t())  # (batch, memory_size)
            weights = torch.softmax(scores / (self.state_dim ** 0.5), dim=-1)

            # Read: weighted sum of memory
            head_read = torch.matmul(weights, self.memory)
            read_output[:, head * self.state_dim // num_read_heads:(head + 1) * self.state_dim // num_read_heads] = head_read[:, :self.state_dim // num_read_heads]

        return read_output

    def reset_memory(self):
        """Clear memory for new reasoning episode."""
        self.memory.zero_()
```

### 3. Implement Test-Time Compute Scaling

Allocate variable computation budget during inference:

```python
class TestTimeComputeScaler:
    """Dynamically allocate compute budget at test time."""

    def __init__(
        self,
        base_steps: int = 5,
        max_steps: int = 50,
        budget_per_example: float = 1.0  # Relative budget
    ):
        self.base_steps = base_steps
        self.max_steps = max_steps
        self.budget_per_example = budget_per_example

    def get_step_budget(
        self,
        task_description: str,
        task_complexity: float = 0.5
    ) -> int:
        """
        Compute number of recurrence steps based on task complexity.
        """
        # Complexity-based scaling
        complexity_budget = int(self.base_steps * (1.0 + task_complexity * 5.0))

        # Cap at maximum
        budget = min(complexity_budget, self.max_steps)

        return budget

    def adaptive_compute(
        self,
        model: "ReasoningModel",
        state: torch.Tensor,
        task: str,
        target_accuracy: float = 0.95,
        max_total_steps: int = 50
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Iteratively increase compute until convergence or budget exhausted.
        """
        current_state = state
        step = 0
        previous_output = None
        convergence_reached = False
        metrics = {"total_steps": 0, "convergence_step": None, "final_output_stable": False}

        while step < max_total_steps and not convergence_reached:
            # Run recurrence step
            current_state, trajectory = model.recurrence_unit(current_state, num_steps=1)

            # Check for convergence: output stability
            current_output = model.output_head(current_state)

            if previous_output is not None:
                similarity = torch.nn.functional.cosine_similarity(
                    current_output.unsqueeze(0),
                    previous_output.unsqueeze(0)
                ).item()

                if similarity > target_accuracy:
                    convergence_reached = True
                    metrics["convergence_step"] = step
                    metrics["final_output_stable"] = True

            previous_output = current_output
            step += 1
            metrics["total_steps"] = step

        return current_state, metrics

    def measure_reasoning_quality(
        self,
        trajectories: List[List[torch.Tensor]],
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure quality of multi-step reasoning.
        """
        metrics = {
            "next_step_accuracy": 0.0,
            "multistep_accuracy": 0.0,
            "trajectory_stability": 0.0
        }

        for trajectory in trajectories:
            # Next-step prediction accuracy
            for t in range(len(trajectory) - 1):
                pred_next = trajectory[t + 1]
                actual_next = ground_truth[t + 1] if t + 1 < len(ground_truth) else None

                if actual_next is not None:
                    match = torch.allclose(pred_next, actual_next, atol=1e-5)
                    metrics["next_step_accuracy"] += match

            # Multi-step accuracy
            final_pred = trajectory[-1]
            final_actual = ground_truth[-1] if len(ground_truth) > 0 else None
            if final_actual is not None:
                match = torch.allclose(final_pred, final_actual, atol=1e-5)
                metrics["multistep_accuracy"] += match

            # Trajectory stability: variance across steps
            stacked = torch.stack(trajectory)
            variance = torch.var(stacked, dim=0).mean().item()
            stability = 1.0 / (1.0 + variance)
            metrics["trajectory_stability"] += stability

        # Average
        n = len(trajectories)
        for key in metrics:
            metrics[key] = metrics[key] / n if n > 0 else 0.0

        return metrics
```

### 4. Integrate Recurrence, Memory, and Compute

Combine all components:

```python
class DeepReasoningModel(nn.Module):
    """Complete model with recurrence, memory, and test-time scaling."""

    def __init__(
        self,
        input_dim: int = 256,
        state_dim: int = 256,
        hidden_dim: int = 512,
        memory_size: int = 32
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, state_dim)
        self.recurrence_unit = RecurrentReasoningUnit(state_dim, hidden_dim)
        self.memory = ExternalMemory(state_dim, memory_size)
        self.output_head = nn.Linear(state_dim, input_dim)
        self.compute_scaler = TestTimeComputeScaler()

    def forward(
        self,
        input_state: torch.Tensor,
        num_steps: int = 5,
        use_memory: bool = True
    ) -> torch.Tensor:
        """Forward pass with recurrence and memory."""

        # Project input
        state = self.input_proj(input_state)

        # Recurrent reasoning
        for step in range(num_steps):
            # Update state through recurrence
            state, _ = self.recurrence_unit(state, num_steps=1)

            # Write to memory
            if use_memory:
                self.memory.write_to_memory(state)

            # Read from memory for enrichment
            if use_memory:
                memory_context = self.memory.read_from_memory(state)
                state = state + 0.3 * memory_context  # Blend with memory

        # Output
        output = self.output_head(state)
        return output

    def solve_with_adaptive_compute(
        self,
        input_state: torch.Tensor,
        task_description: str
    ) -> Tuple[torch.Tensor, Dict]:
        """Solve with test-time compute scaling."""

        # Estimate task complexity
        complexity = self._estimate_complexity(task_description)

        # Get adaptive budget
        budget = self.compute_scaler.get_step_budget(task_description, complexity)

        # Solve with adaptive steps
        output, metrics = self.compute_scaler.adaptive_compute(
            self,
            input_state,
            task_description,
            max_total_steps=budget
        )

        return output, metrics

    def _estimate_complexity(self, task: str) -> float:
        """Estimate task complexity from description."""
        complexity_indicators = ["recursive", "nested", "multi-step", "chain"]
        complexity = sum(1 for ind in complexity_indicators if ind in task.lower()) / len(complexity_indicators)
        return min(1.0, complexity)
```

## Practical Guidance

### When to Use Deep Reasoning Models

- Multi-step mathematical reasoning
- Algorithmic problem solving
- Abstract rule learning
- Tasks requiring state accumulation
- Scenarios allowing test-time compute allocation

### When NOT to Use

- Simple single-step generation
- Real-time systems with strict latency (<100ms)
- Tasks without clear sequential structure
- Inference with extremely limited budgets

### Key Hyperparameters

- **num_recurrence_steps**: 5-20 (base steps)
- **memory_size**: 16-64 slots
- **state_dim**: 128-512
- **max_test_time_steps**: 20-100 (dependent on budget)
- **convergence_threshold**: 0.90-0.99

### Performance Expectations

- Next-Step Accuracy: Maintained with recurrence
- Multi-Step Accuracy: Significantly improved
- Effective Depth: Layer count × recurrence steps
- Memory Efficiency: Sub-quadratic vs. attention

## Reference

Researchers. (2024). Beyond Memorization: Extending Reasoning Depth with Recurrence Memory and Test-Time Compute Scaling. arXiv preprint arXiv:2508.16745.
