---
name: crinn-contrastive-rl-ann-search
title: CRINN - Contrastive RL for HNSW Optimization
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.02091
keywords: [reinforcement-learning, nearest-neighbor-search, hnsw, code-optimization]
description: "Optimize approximate nearest neighbor search via contrastive RL, learning to generate efficient code for HNSW graph construction, search, and refinement."
---

## CRINN: Contrastive RL for ANN Search Optimization

CRINN applies contrastive reinforcement learning to optimize Hierarchical Navigable Small World (HNSW) nearest neighbor search. Rather than hand-tuning parameters, it trains LLMs to generate optimized code by comparing implementations with different performance characteristics, learning which optimizations matter most.

### Core Concept

ANN search is performance-critical yet parameter-sensitive. Traditional approaches optimize via grid search, but this doesn't generalize to new datasets or hardware. CRINN treats optimization as an RL problem: LLM generates code variants, measures their query speed (QPS), and learns patterns of what makes code fast. Contrastive learning—comparing fast vs. slow implementations—teaches the model efficiency principles more effectively than absolute rewards.

### Architecture Overview

- **Contrastive Reward System**: Compare code variants, score based on QPS at fixed recall levels
- **Sequential Module Optimization**: Optimize HNSW modules (graph construction, search, refinement) independently
- **Speed-Based Rewards**: nDCG-style metric: area under QPS-recall curve for comparing implementations
- **GRPO Training**: Group Relative Policy Optimization for efficient RL updates
- **Multi-Benchmark Evaluation**: Validate on 6 standard ANN datasets (GIST, MNIST, GloVe, etc.)

### Implementation Steps

**Step 1: Set Up HNSW Benchmarking**

```python
import time
import numpy as np
from typing import Tuple, Dict, List

class HNSWBenchmark:
    """Benchmark HNSW implementations on standard datasets."""

    def __init__(self, dataset_name: str, dataset: np.ndarray, queries: np.ndarray):
        self.dataset_name = dataset_name
        self.data = dataset  # (n, d)
        self.queries = queries  # (m, d)
        self.ground_truth = self._compute_ground_truth()

    def _compute_ground_truth(self):
        """Compute exact nearest neighbors via brute force."""
        distances = np.linalg.norm(
            self.queries[:, np.newaxis, :] - self.data[np.newaxis, :, :],
            axis=2
        )
        # Top-k indices for each query
        return np.argsort(distances, axis=1)[:, :100]  # Top 100

    def evaluate_hnsw(self, hnsw_impl) -> Dict[str, float]:
        """
        Evaluate HNSW implementation on recall and QPS.
        """
        # Time queries
        start = time.time()
        results = hnsw_impl.search_batch(self.queries, k=100)
        qps = len(self.queries) / (time.time() - start)

        # Compute recall
        recall_sum = 0
        for i, result_ids in enumerate(results):
            hits = len(set(result_ids) & set(self.ground_truth[i]))
            recall_sum += hits / len(self.ground_truth[i])

        recall = recall_sum / len(self.queries)

        return {
            'qps': qps,
            'recall': recall,
            'time_per_query': 1000 / qps  # ms
        }

    def compute_reward(self, results: Dict[str, float], ef_sweep: List[int] = [10, 20, 50, 100, 200]) -> float:
        """
        Compute reward as area under QPS-recall curve.
        Sweep ef parameter to get QPS at different recall targets.
        """
        qps_at_recall = []

        for ef in ef_sweep:
            result = self.evaluate_hnsw(...)  # Would eval with this ef
            if result['recall'] >= 0.85:  # Filter to valid recall range
                qps_at_recall.append(result['qps'])

        # Reward = area under curve (integral approximation)
        if not qps_at_recall:
            return 0.0

        reward = np.trapz(qps_at_recall)
        return reward / 1000  # Normalize
```

**Step 2: Implement Contrastive Reward**

```python
def compute_contrastive_reward(impl_a: str, impl_b: str,
                              benchmark: HNSWBenchmark) -> Tuple[float, float]:
    """
    Compare two HNSW implementations.
    Returns rewards for comparing them (preference learning).
    """
    # Build and evaluate both
    hnsw_a = build_hnsw_from_code(impl_a)
    hnsw_b = build_hnsw_from_code(impl_b)

    results_a = benchmark.evaluate_hnsw(hnsw_a)
    results_b = benchmark.evaluate_hnsw(hnsw_b)

    # Reward is comparative: which is better?
    qps_a = results_a['qps']
    qps_b = results_b['qps']

    # Normalize by maximum
    max_qps = max(qps_a, qps_b)
    reward_a = qps_a / max_qps
    reward_b = qps_b / max_qps

    return reward_a, reward_b

class ContrastiveRL:
    """
    Train LLM to generate HNSW code via contrastive learning.
    """
    def __init__(self, model, benchmark: HNSWBenchmark):
        self.model = model
        self.benchmark = benchmark

    def generate_code_variants(self, module_type: str = 'search', num_variants: int = 4) -> List[str]:
        """
        Generate multiple HNSW code variants using LLM.
        """
        prompt = f"""Generate {num_variants} different optimized implementations of HNSW {module_type} module.
Focus on efficiency: prefetching, vectorization, cache locality, etc.
Provide complete C++ code for the {module_type} operation.

Variant 1:
{self._get_template(module_type)}"""

        # Generate multiple variants with different temperatures
        variants = []
        for temp in np.linspace(0.5, 1.5, num_variants):
            code = self.model.generate(prompt, temperature=temp, max_tokens=1000)
            variants.append(code)

        return variants

    def train_step(self, module_type: str = 'search'):
        """
        One RL training step: generate, evaluate, learn from comparisons.
        """
        # Generate variants
        variants = self.generate_code_variants(module_type, num_variants=4)

        # Evaluate all
        rewards = []
        for variant in variants:
            try:
                hnsw = build_hnsw_from_code(variant)
                reward = self.benchmark.compute_reward(
                    self.benchmark.evaluate_hnsw(hnsw)
                )
                rewards.append(reward)
            except:
                rewards.append(0.0)

        # Learn from comparisons
        # Pair-wise comparisons: which variant is better?
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                if rewards[i] > rewards[j]:
                    # Variant i is better
                    better_code = variants[i]
                    worse_code = variants[j]
                else:
                    better_code = variants[j]
                    worse_code = variants[i]

                # Update model: prefer better_code over worse_code
                # (DPO-style: learn from preferences)
                self._update_model(better_code, worse_code)

    def _update_model(self, preferred: str, dispreferred: str):
        """Update model to prefer one implementation over another."""
        # DPO-style loss
        preferred_logp = self.model.compute_logp(preferred)
        dispreferred_logp = self.model.compute_logp(dispreferred)

        # Loss: maximize probability gap
        loss = -torch.log(torch.sigmoid(preferred_logp - dispreferred_logp))
        loss.backward()
        self.model.optimizer.step()

    def _get_template(self, module_type: str) -> str:
        """Return template for code generation."""
        templates = {
            'construction': 'void buildGraph(const vector<vector<float>>& data, ...) {',
            'search': 'vector<int> search(const vector<float>& query, int ef) {',
            'refinement': 'void refineResults(vector<int>& candidates, ...) {'
        }
        return templates.get(module_type, '')
```

**Step 3: Sequential Module Optimization**

```python
class SequentialHNSWOptimizer:
    """Optimize HNSW modules one at a time."""

    def __init__(self, model, benchmarks: Dict[str, HNSWBenchmark]):
        self.model = model
        self.benchmarks = benchmarks
        self.optimized_modules = {}

    def optimize_construction(self, num_iterations: int = 10):
        """Optimize graph construction module."""
        print("Optimizing graph construction...")
        rl = ContrastiveRL(self.model, self.benchmarks['construction'])

        for i in range(num_iterations):
            rl.train_step(module_type='construction')
            print(f"  Iteration {i}: Training contrastive RL")

        # Extract best variant
        best_code = rl.generate_code_variants('construction', num_variants=1)[0]
        self.optimized_modules['construction'] = best_code

    def optimize_search(self, num_iterations: int = 10):
        """Optimize search module."""
        print("Optimizing search module...")
        rl = ContrastiveRL(self.model, self.benchmarks['search'])

        for i in range(num_iterations):
            rl.train_step(module_type='search')

        best_code = rl.generate_code_variants('search', num_variants=1)[0]
        self.optimized_modules['search'] = best_code

    def optimize_refinement(self, num_iterations: int = 10):
        """Optimize refinement module."""
        print("Optimizing refinement...")
        rl = ContrastiveRL(self.model, self.benchmarks['refinement'])

        for i in range(num_iterations):
            rl.train_step(module_type='refinement')

        best_code = rl.generate_code_variants('refinement', num_variants=1)[0]
        self.optimized_modules['refinement'] = best_code

    def optimize_full_pipeline(self):
        """Optimize all modules sequentially."""
        self.optimize_construction(num_iterations=15)
        self.optimize_search(num_iterations=15)
        self.optimize_refinement(num_iterations=10)

        # Compile final HNSW
        return self.compile_optimized_hnsw()

    def compile_optimized_hnsw(self):
        """Assemble optimized modules into final HNSW."""
        full_code = f"""
{self.optimized_modules['construction']}

{self.optimized_modules['search']}

{self.optimized_modules['refinement']}
"""
        return full_code
```

**Step 4: Evaluate on Benchmark Suite**

```python
def evaluate_crinn(optimized_code: str, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """
    Evaluate CRINN-optimized code on multiple benchmarks.
    """
    results = {}

    for dataset_name, (data, queries) in datasets.items():
        benchmark = HNSWBenchmark(dataset_name, data, queries)
        hnsw = build_hnsw_from_code(optimized_code)
        metrics = benchmark.evaluate_hnsw(hnsw)

        results[dataset_name] = metrics
        print(f"{dataset_name}: {metrics['qps']:.1f} QPS @ {metrics['recall']:.3f} recall")

    return results
```

### Practical Guidance

**When to Use:**
- Optimizing ANN search for specific hardware/datasets
- Scenarios where baseline HNSW doesn't meet latency targets
- Applications with domain-specific distance metrics
- Cases where code generation + evaluation is feasible

**When NOT To Use:**
- Standard HNSW parameters are sufficient
- Real-time online optimization (training is slow)
- Proprietary hardware without benchmarking access
- Scenarios requiring formal correctness guarantees

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `num_code_variants` | 4 | More variants = better exploration, higher eval cost |
| `rl_iterations_per_module` | 15 | Training iterations; balance quality vs. time |
| `ef_sweep_points` | [10,20,50,100,200] | Parameter ranges for reward calculation |
| `recall_threshold` | 0.85 | Minimum recall for valid configurations |

### Reference

**Paper**: CRINN: Contrastive RL for Approximate Nearest Neighbor Search (2508.02091)
- Best-in-class on 3/6 benchmarks, ties on 2 others
- 3-85% QPS improvements at fixed recall
- Contrastive learning more effective than absolute rewards
