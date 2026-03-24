---
name: chain-of-trajectories-diffusion-planning
title: "Chain-of-Trajectories: Unlocking Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.14704"
keywords: [Diffusion Models, Trajectory Planning, Graph Algorithms, Computational Allocation, Generative Quality]
description: "Improve diffusion model sampling by planning content-adaptive denoising trajectories. Extract Diffusion DNA signatures quantifying per-stage difficulty, then apply graph planning to allocate computation to challenging generative phases."
---

# Chain-of-Trajectories: Graph-Based Trajectory Planning for Diffusion Models

Diffusion models use fixed sampling schedules regardless of content difficulty. Some generative phases require extensive refinement while others converge quickly, yet standard schedules waste computation by allocating equal effort uniformly. Chain-of-Trajectories (CoT) reformulates diffusion sampling as a graph planning problem by extracting Diffusion DNA—a low-dimensional signature quantifying per-stage denoising difficulty. This enables dynamic allocation of computational resources to challenging phases, improving output quality while reducing total computational cost.

The approach is train-free and applies to existing pre-trained diffusion models, requiring only inference-time trajectory planning.

## Core Concept

Chain-of-Trajectories operates in three phases:

1. **DNA Extraction** — Analyze initial samples to compute per-stage denoising difficulty
2. **Graph Construction** — Build directed acyclic graph (DAG) of possible trajectories
3. **Optimal Planning** — Apply shortest-path algorithms to find computation-efficient sampling paths

The key insight: by treating sampling as a planning problem rather than fixed sequence execution, models can adapt to content characteristics dynamically.

## Architecture Overview

- **Diffusion DNA Extractor** — Compute per-timestep difficulty metrics from model uncertainty
- **Signature Aggregation** — Combine timestep signals into stage-level estimates
- **Trajectory Graph Builder** — Create DAG where nodes represent (stage, refinement_level) pairs
- **Cost Function** — Compute per-edge cost based on DNA and refinement requirements
- **Graph Planner** — Apply Dijkstra or A* to find optimal sampling path
- **Inference Pipeline** — Execute planned trajectory with dynamic step allocation
- **Quality Validator** — Measure output improvements vs standard fixed schedules

## Implementation Steps

Start by implementing Diffusion DNA extraction from model predictions.

```python
import torch
import torch.nn as nn
import numpy as np
from heapq import heappush, heappop

class DiffusionDNA:
    """Extract difficulty signatures from diffusion model."""

    def __init__(self, model, num_stages=10):
        self.model = model
        self.num_stages = num_stages
        self.timesteps = torch.linspace(0, 999, num_stages, dtype=torch.long)

    def extract_dna(self, x_t: torch.Tensor, num_samples=5) -> np.ndarray:
        """Compute difficulty signature for input."""
        difficulties = []

        for t in self.timesteps:
            # Get model prediction at this timestep
            with torch.no_grad():
                noise_pred = self.model(x_t, t.unsqueeze(0))

            # Compute uncertainty (model confidence)
            # Use prediction variance as difficulty proxy
            uncertainty = self._compute_uncertainty(noise_pred, num_samples)

            difficulties.append(uncertainty)

        # Normalize to [0, 1]
        dna = np.array(difficulties)
        dna = (dna - dna.min()) / (dna.max() - dna.min() + 1e-8)

        return dna

    def _compute_uncertainty(self, prediction: torch.Tensor,
                            num_samples: int = 5) -> float:
        """Estimate model uncertainty."""
        # Run multiple forward passes with dropout enabled
        predictions = []

        for _ in range(num_samples):
            pred = self.model.forward_with_dropout(prediction)
            predictions.append(pred)

        predictions = torch.stack(predictions)

        # Uncertainty as prediction variance
        uncertainty = predictions.var(dim=0).mean().item()

        return uncertainty

    def stage_difficulty(self, dna: np.ndarray, stage_idx: int) -> float:
        """Get difficulty for a specific stage."""
        return float(dna[stage_idx])
```

Now implement the trajectory graph and planning algorithm.

```python
class TrajectoryGraph:
    """DAG of possible denoising trajectories."""

    def __init__(self, num_stages=10, max_refinement=3):
        self.num_stages = num_stages
        self.max_refinement = max_refinement

        # Nodes: (stage, refinement_level)
        self.nodes = [(s, r) for s in range(num_stages)
                     for r in range(max_refinement)]
        self.edges = {}
        self.costs = {}

    def add_edge(self, from_node: tuple, to_node: tuple, cost: float):
        """Add directed edge with cost."""
        if from_node not in self.edges:
            self.edges[from_node] = []

        self.edges[from_node].append(to_node)
        self.costs[(from_node, to_node)] = cost

    def build_from_dna(self, dna: np.ndarray, base_cost=1.0):
        """Construct graph using Diffusion DNA."""
        # Edges represent sampling transitions
        for stage in range(self.num_stages - 1):
            for ref_level in range(self.max_refinement):
                from_node = (stage, ref_level)

                # Can transition to next stage (coarse pass)
                to_node = (stage + 1, 0)
                # Cost inversely related to difficulty (hard stages cost more)
                difficulty = dna[stage]
                cost = base_cost / (0.1 + difficulty)  # Normalized
                self.add_edge(from_node, to_node, cost)

                # Can refine current stage (additional passes)
                if ref_level < self.max_refinement - 1:
                    to_node = (stage, ref_level + 1)
                    cost = difficulty * base_cost  # Refine hard stages more
                    self.add_edge(from_node, to_node, cost)

    def find_optimal_path(self, start_node=(0, 0),
                         end_node=None) -> (list, float):
        """Find minimum-cost path using Dijkstra's algorithm."""
        if end_node is None:
            end_node = (self.num_stages - 1, self.max_refinement - 1)

        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        predecessors = {node: None for node in self.nodes}

        pq = [(0, start_node)]

        while pq:
            current_dist, current_node = heappop(pq)

            if current_node == end_node:
                # Reconstruct path
                path = []
                node = end_node
                while node is not None:
                    path.append(node)
                    node = predecessors[node]
                path.reverse()

                return path, current_dist

            if current_dist > distances[current_node]:
                continue

            # Explore neighbors
            if current_node in self.edges:
                for next_node in self.edges[current_node]:
                    edge_cost = self.costs[(current_node, next_node)]
                    new_dist = current_dist + edge_cost

                    if new_dist < distances[next_node]:
                        distances[next_node] = new_dist
                        predecessors[next_node] = current_node
                        heappush(pq, (new_dist, next_node))

        return [], float('inf')  # No path found
```

Implement the sampling executor that follows planned trajectories.

```python
class AdaptiveDiffusionSampler:
    """Sample following computed trajectory."""

    def __init__(self, model, dna_extractor: DiffusionDNA):
        self.model = model
        self.dna_extractor = dna_extractor

    def sample_with_trajectory(self, initial_noise: torch.Tensor,
                              trajectory_graph: TrajectoryGraph,
                              verbose=False) -> torch.Tensor:
        """Execute sampling following optimal trajectory."""
        # Extract DNA
        dna = self.dna_extractor.extract_dna(initial_noise)

        # Build and plan trajectory
        trajectory_graph.build_from_dna(dna)
        path, total_cost = trajectory_graph.find_optimal_path()

        if verbose:
            print(f"Planned trajectory with cost {total_cost:.3f}")
            print(f"Path: {path}")

        # Execute sampling following trajectory
        x_t = initial_noise.clone()
        current_stage = 0

        for node in path:
            stage, refinement = node

            # Skip backwards in trajectory (shouldn't happen with proper planning)
            if stage < current_stage:
                continue

            # Move to this stage with refinement passes
            target_t = self.dna_extractor.timesteps[stage]

            # Multiple denoising steps at this stage if refined
            num_steps = 1 + refinement

            for step in range(num_steps):
                # Denoise one step
                with torch.no_grad():
                    noise_pred = self.model(x_t, target_t.unsqueeze(0))

                # Update based on noise prediction
                x_t = self._denoise_step(x_t, noise_pred, target_t)

            current_stage = stage

        return x_t

    def _denoise_step(self, x_t: torch.Tensor, noise_pred: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        """Single denoising step (simplified)."""
        # Simplified DDIM step
        alpha_t = torch.sqrt(1 - t.float() / 1000)  # Approximate
        x_t_next = alpha_t * x_t + (1 - alpha_t) * noise_pred

        return x_t_next

    def sample_standard(self, initial_noise: torch.Tensor,
                       num_steps=50) -> torch.Tensor:
        """Baseline: standard fixed-schedule sampling."""
        x_t = initial_noise.clone()
        timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long)

        for t in timesteps:
            with torch.no_grad():
                noise_pred = self.model(x_t, t.unsqueeze(0))

            x_t = self._denoise_step(x_t, noise_pred, t)

        return x_t


def benchmark_trajectory_planning(model, test_images, num_trials=10):
    """Compare trajectory-planned vs standard sampling."""
    dna_extractor = DiffusionDNA(model)
    sampler = AdaptiveDiffusionSampler(model, dna_extractor)

    trajectory_graph = TrajectoryGraph(num_stages=10, max_refinement=3)

    results = {'trajectory': [], 'standard': []}

    for img in test_images[:num_trials]:
        # Trajectory-planned sampling
        x_t = torch.randn_like(img)
        output_trajectory = sampler.sample_with_trajectory(x_t, trajectory_graph)

        # Standard sampling
        output_standard = sampler.sample_standard(x_t)

        # Evaluate quality (using proxy metrics)
        quality_trajectory = compute_sample_quality(output_trajectory, img)
        quality_standard = compute_sample_quality(output_standard, img)

        results['trajectory'].append(quality_trajectory)
        results['standard'].append(quality_standard)

    avg_trajectory = np.mean(results['trajectory'])
    avg_standard = np.mean(results['standard'])

    print(f"Trajectory-planned: {avg_trajectory:.3f}")
    print(f"Standard: {avg_standard:.3f}")
    print(f"Improvement: {(avg_trajectory - avg_standard) / avg_standard:.1%}")

    return results
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Number of stages 8-15; more stages provide finer control but slower planning
- Maximum refinement levels 2-4; higher refinement allows concentrated computation
- Base cost 1.0 works as default; adjust based on desired output quality vs latency tradeoff
- Use when content diversity is high (different images have different difficulty profiles)
- Particularly effective for high-resolution generation where efficiency matters

**When NOT to use:**
- For real-time applications where planning overhead exceeds benefits
- For batch processing with homogeneous content (fixed schedules are simpler)
- With models that don't expose uncertainty estimates (DNA extraction won't work)

**Common Pitfalls:**
- DNA extraction being noisy or unreliable; use multiple samples to average uncertainty
- Planning becoming too aggressive, skipping important denoising steps; include constraints
- Trajectories becoming too complex, adding overhead; limit graph size
- Difficulty metrics not correlating with actual sample quality; calibrate metrics on validation set

## Reference

Paper: [Chain-of-Trajectories: Unlocking Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning](https://arxiv.org/abs/2603.14704)
