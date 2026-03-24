---
name: moe-sparsity-reasoning
title: Optimal Sparsity of MoE Language Models for Reasoning
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.18672
keywords: [mixture-of-experts, sparsity, reasoning, memorization, compute-optimal]
description: "Determine optimal MoE sparsity by separating memorization and reasoning trade-offs: active FLOPs improve reasoning while total parameters improve memorization, requiring joint optimization"
---

# Optimal Sparsity of MoE Language Models for Reasoning

## Core Concept

This work reveals that optimal MoE sparsity differs from traditional dense model scaling. The key insight: memorization and reasoning have opposing sparsity preferences. Memorization improves with more total parameters (dense experts better), while reasoning improves with more active FLOPs (sparse routing better). The paper proposes joint optimization of active FLOPs and tokens-per-parameter (TPP) as the path to compute-optimal reasoning models, revising classical scaling laws.

## Architecture Overview

- **Two Capability Dimensions**: Separate memorization from reasoning evaluation
- **Active FLOPs Principle**: More compute helps reasoning independent of model size
- **TPP Principle**: Total tokens per parameter correlates with memorization efficiency
- **MoE Configuration Space**: Vary experts, top-k routing, and total parameters
- **Joint Optimization**: Balance both principles for overall performance

## Implementation Steps

### Stage 1: Establish Evaluation Framework

Separate memorization and reasoning in evaluation.

```python
# Separate memorization from reasoning evaluation
from typing import Dict, List, Tuple
import numpy as np

class CapabilityEvaluator:
    """Evaluate memorization vs reasoning separately"""

    def __init__(self):
        self.benchmarks = {
            "memorization": [
                "fact_recall",
                "knowledge_qa",
                "entity_extraction"
            ],
            "reasoning": [
                "math_reasoning",
                "logical_deduction",
                "reading_comprehension"
            ]
        }

    def evaluate_model(
        self,
        model,
        test_datasets: Dict[str, List]
    ) -> Dict:
        """
        Evaluate model on both memorization and reasoning tasks.
        """
        results = {
            "memorization": {},
            "reasoning": {},
            "pre_training_loss": 0.0
        }

        # Memorization benchmarks
        for benchmark in self.benchmarks["memorization"]:
            dataset = test_datasets.get(benchmark, [])
            accuracy = self.evaluate_benchmark(model, dataset)
            results["memorization"][benchmark] = accuracy

        # Reasoning benchmarks
        for benchmark in self.benchmarks["reasoning"]:
            dataset = test_datasets.get(benchmark, [])
            accuracy = self.evaluate_benchmark(model, dataset)
            results["reasoning"][benchmark] = accuracy

        # Pre-training loss (proxy for memorization efficiency)
        results["pre_training_loss"] = self.get_pretrain_loss(model)

        return results

    def evaluate_benchmark(self, model, dataset: List) -> float:
        """Evaluate accuracy on benchmark"""
        correct = 0
        for example in dataset:
            prediction = model.generate(example["input"])
            if prediction == example["target"]:
                correct += 1
        return correct / len(dataset) if dataset else 0.0

    def get_pretrain_loss(self, model) -> float:
        """Get pre-training loss (approximates memorization capacity)"""
        return 0.0  # Placeholder


class CapabilityAnalysis:
    """Analyze memorization vs reasoning trade-offs"""

    def __init__(self):
        self.results = []

    def add_model_result(
        self,
        model_config: Dict,
        eval_result: Dict
    ):
        """Record model evaluation"""
        self.results.append({
            "config": model_config,
            "evaluation": eval_result
        })

    def analyze_trade_offs(self):
        """Analyze memorization-reasoning trade-off"""
        memo_scores = []
        reasoning_scores = []
        model_sizes = []

        for result in self.results:
            config = result["config"]
            eval_res = result["evaluation"]

            # Aggregate scores
            memo_score = np.mean(list(eval_res["memorization"].values()))
            reasoning_score = np.mean(list(eval_res["reasoning"].values()))
            model_size = config["num_params"]

            memo_scores.append(memo_score)
            reasoning_scores.append(reasoning_score)
            model_sizes.append(model_size)

        return {
            "memorization": memo_scores,
            "reasoning": reasoning_scores,
            "model_sizes": model_sizes
        }
```

### Stage 2: Design MoE Configuration Space

Create different MoE architectures with varying sparsity.

```python
# MoE sparsity configuration
from dataclasses import dataclass

@dataclass
class MoEConfig:
    """MoE model configuration"""
    num_experts: int
    expert_size: int  # Params per expert
    top_k: int  # Top-k routing
    num_layers: int
    hidden_dim: int


class MoESparseFamily:
    """Generate family of MoE models with different sparsity"""

    def __init__(self, target_compute_budget: float = 1.0):
        self.target_compute = target_compute_budget
        self.models = []

    def generate_sparse_variants(self) -> List[MoEConfig]:
        """
        Generate MoE variants with different sparsity levels.

        Vary:
        - Number of experts (E)
        - Top-k routing
        - Expert size
        - Keep compute budget constant
        """
        variants = []

        # Base configuration
        base_hidden_dim = 4096
        base_num_layers = 32

        # Sparse configurations
        for num_experts in [8, 16, 32, 64, 128]:
            for top_k in [1, 2, 4, 8]:
                # Keep total compute roughly constant
                # Compute = (hidden_dim^2 * layers * top_k) / num_experts

                # Adjust hidden_dim to maintain constant compute
                adjusted_dim = int(
                    base_hidden_dim * np.sqrt(num_experts / (top_k + 1e-6))
                )

                expert_size = (adjusted_dim * adjusted_dim) // num_experts

                config = MoEConfig(
                    num_experts=num_experts,
                    expert_size=expert_size,
                    top_k=top_k,
                    num_layers=base_num_layers,
                    hidden_dim=adjusted_dim
                )

                variants.append(config)
                self.models.append(config)

        return variants

    def compute_statistics(self) -> Dict:
        """Compute statistics for each model"""
        stats = []

        for config in self.models:
            # Total parameters
            total_params = (
                config.num_experts * config.expert_size +  # Expert params
                config.num_layers * config.hidden_dim ** 2  # Attention/norm
            )

            # Active parameters (only top-k experts active)
            active_params = (
                config.top_k * config.expert_size +
                config.num_layers * config.hidden_dim ** 2
            )

            # Active FLOPs
            active_flops = (
                config.num_layers * config.hidden_dim * config.top_k * config.expert_size
            )

            stats.append({
                "config": config,
                "total_params": total_params,
                "active_params": active_params,
                "active_flops": active_flops,
                "sparsity": 1 - (active_params / total_params) if total_params > 0 else 0
            })

        return stats
```

### Stage 3: Extract Core Principles

Derive the two scaling principles for MoE optimization.

```python
# Core MoE scaling principles
class MoEPrinciples:
    """Extract active FLOPs and TPP principles"""

    @staticmethod
    def analyze_active_flops_principle(results: List[Dict]) -> Dict:
        """
        Active FLOPs Principle:
        'Models with identical training loss but greater active compute
         achieve higher reasoning accuracy'

        This means: sparse (higher k) > dense (lower k) for reasoning.
        """
        # Group by pre-training loss
        loss_groups = {}
        for result in results:
            loss = result["pretrain_loss"]
            if loss not in loss_groups:
                loss_groups[loss] = []
            loss_groups[loss].append(result)

        # Within each loss group, compare reasoning accuracy vs active FLOPs
        analysis = {}
        for loss, models in loss_groups.items():
            # Sort by active FLOPs
            models.sort(key=lambda x: x["active_flops"])

            reasoning_by_flops = [m["reasoning_accuracy"] for m in models]
            active_flops = [m["active_flops"] for m in models]

            # Check correlation: more FLOPs -> better reasoning
            correlation = np.corrcoef(active_flops, reasoning_by_flops)[0, 1]

            analysis[loss] = {
                "correlation": correlation,
                "reasoning_scores": reasoning_by_flops,
                "active_flops": active_flops
            }

        return analysis

    @staticmethod
    def analyze_tpp_principle(results: List[Dict]) -> Dict:
        """
        TPP Principle:
        'Memorization improves with increased parameters,
         but there exists an optimal TPP for reasoning'

        TPP = total_tokens_seen / total_parameters
        """
        # Analyze memorization vs total params
        total_params_list = [r["total_params"] for r in results]
        memo_accuracy_list = [r["memorization_accuracy"] for r in results]

        memo_param_corr = np.corrcoef(total_params_list, memo_accuracy_list)[0, 1]

        # Analyze reasoning vs TPP
        tpp_list = [r["training_tokens"] / r["total_params"] for r in results]
        reasoning_list = [r["reasoning_accuracy"] for r in results]

        reasoning_tpp_corr = np.corrcoef(tpp_list, reasoning_list)[0, 1]

        return {
            "memorization_param_correlation": memo_param_corr,
            "reasoning_tpp_correlation": reasoning_tpp_corr,
            "interpretation": {
                "memo": "Higher params -> better memorization",
                "reasoning": "Optimal TPP exists; too high or low hurts reasoning"
            }
        }
```

### Stage 4: Joint Optimization Framework

Optimize both active FLOPs and TPP together.

```python
# Joint MoE optimization
class MoEOptimizer:
    """Find optimal sparsity via joint optimization"""

    def __init__(self):
        self.pareto_frontier = []

    def compute_objective(
        self,
        reasoning_accuracy: float,
        memorization_accuracy: float,
        weight_reasoning: float = 0.7
    ) -> float:
        """
        Combined objective balancing reasoning and memorization.

        Objective = w * reasoning_acc + (1-w) * memo_acc
        """
        return (weight_reasoning * reasoning_accuracy +
                (1 - weight_reasoning) * memorization_accuracy)

    def find_optimal_config(
        self,
        results: List[Dict],
        compute_budget: float = 1.0
    ) -> Dict:
        """
        Find optimal MoE configuration.

        Constraints:
        - Maintain compute budget
        - Optimize both reasoning and memorization

        Returns:
        - Recommended config
        - Trade-off curve
        """
        # Filter by compute budget
        valid_results = [
            r for r in results
            if r.get("active_flops", 1.0) <= compute_budget
        ]

        if not valid_results:
            return {}

        # Evaluate each configuration
        objectives = []
        for result in valid_results:
            obj = self.compute_objective(
                result["reasoning_accuracy"],
                result["memorization_accuracy"]
            )
            objectives.append(obj)
            result["objective_score"] = obj

        # Find Pareto frontier
        self.pareto_frontier = self._compute_pareto_frontier(valid_results)

        # Select best overall
        best_result = max(valid_results, key=lambda r: r["objective_score"])

        return {
            "optimal_config": best_result["config"],
            "reasoning_accuracy": best_result["reasoning_accuracy"],
            "memorization_accuracy": best_result["memorization_accuracy"],
            "total_params": best_result["total_params"],
            "active_flops": best_result["active_flops"],
            "pareto_frontier": self.pareto_frontier
        }

    def _compute_pareto_frontier(self, results: List[Dict]) -> List[Dict]:
        """Compute Pareto frontier (reasoning vs memorization)"""
        frontier = []

        for result in results:
            # Check if dominated by any other result
            dominated = False
            for other in results:
                if (other["reasoning_accuracy"] >= result["reasoning_accuracy"] and
                    other["memorization_accuracy"] >= result["memorization_accuracy"] and
                    (other["reasoning_accuracy"] > result["reasoning_accuracy"] or
                     other["memorization_accuracy"] > result["memorization_accuracy"])):
                    dominated = True
                    break

            if not dominated:
                frontier.append(result)

        return frontier
```

### Stage 5: Evaluate and Validate

Test optimal configurations on benchmarks.

```python
# Comprehensive evaluation
class MoEEvaluator:
    """Full evaluation of MoE sparsity findings"""

    def __init__(self):
        self.optimizer = MoEOptimizer()
        self.evaluator = CapabilityEvaluator()

    def run_full_study(
        self,
        compute_budget: float = 1.0
    ) -> Dict:
        """
        Run complete MoE sparsity study.

        1. Generate MoE variants
        2. Train each variant
        3. Evaluate on memorization and reasoning
        4. Optimize for best configuration
        """
        # Step 1: Generate variants
        family = MoESparseFamily(compute_budget)
        configs = family.generate_sparse_variants()

        print(f"Generated {len(configs)} MoE configurations")

        # Step 2: Train variants (placeholder - in reality this is expensive)
        results = []
        for config in configs:
            # Train model
            model = self.train_model(config)

            # Evaluate
            eval_result = self.evaluator.evaluate_model(
                model,
                test_datasets=self.get_test_datasets()
            )

            # Compute statistics
            stats = family.compute_statistics()
            config_stats = next(
                (s for s in stats if s["config"] == config),
                {}
            )

            result = {
                "config": config,
                "reasoning_accuracy": np.mean(list(eval_result["reasoning"].values())),
                "memorization_accuracy": np.mean(list(eval_result["memorization"].values())),
                "pretrain_loss": eval_result["pre_training_loss"],
                **config_stats
            }

            results.append(result)

        # Step 3: Analyze trade-offs
        principles = MoEPrinciples()
        flops_analysis = principles.analyze_active_flops_principle(results)
        tpp_analysis = principles.analyze_tpp_principle(results)

        # Step 4: Optimize
        optimal = self.optimizer.find_optimal_config(results, compute_budget)

        return {
            "total_configs_evaluated": len(results),
            "active_flops_principle": flops_analysis,
            "tpp_principle": tpp_analysis,
            "optimal_configuration": optimal,
            "pareto_frontier": self.optimizer.pareto_frontier
        }

    def train_model(self, config: MoEConfig):
        """Train single MoE model"""
        return None  # Placeholder

    def get_test_datasets(self) -> Dict:
        """Get evaluation datasets"""
        return {}

    def generate_report(self, study_results: Dict) -> str:
        """Generate findings report"""
        report = []
        report.append("## MoE Sparsity Study Results\n")

        optimal = study_results["optimal_configuration"]
        report.append(f"**Optimal Configuration:**")
        report.append(f"- Experts: {optimal['optimal_config'].num_experts}")
        report.append(f"- Top-K: {optimal['optimal_config'].top_k}")
        report.append(f"- Reasoning Accuracy: {optimal['reasoning_accuracy']:.3f}")
        report.append(f"- Memorization Accuracy: {optimal['memorization_accuracy']:.3f}\n")

        report.append("**Key Findings:**")
        report.append("1. Active FLOPs improve reasoning regardless of model size")
        report.append("2. Optimal TPP varies by reasoning task complexity")
        report.append("3. RL/test-time compute don't change fundamental trade-offs\n")

        return "\n".join(report)
```

## Practical Guidance

### Recommended MoE Configurations

**For Reasoning-Heavy Workloads:**
- High sparsity: 32+ experts, top-k 2-4
- Benefit: More active FLOPs per token
- Trade-off: Lower memorization capacity

**For Balanced Workloads:**
- Medium sparsity: 16 experts, top-k 4-8
- Benefit: Reasonable reasoning + memorization
- Trade-off: Middle-ground performance

**For Knowledge-Heavy Tasks:**
- Low sparsity: 8 experts, top-k 8+
- Benefit: Better memorization
- Trade-off: Lower reasoning capability

### Joint Optimization Strategy

1. Define weight between reasoning and memorization (0.7 reasoning is typical)
2. Evaluate candidate configurations on both task types
3. Identify Pareto frontier
4. Select configuration maximizing weighted objective

### When to Use This Framework

- Training custom LLMs for specific use cases
- Allocating compute budget between capabilities
- Understanding model trade-offs
- Planning large-scale training runs

### When NOT to Use

- Fine-tuning existing models (pre-training decisions already made)
- Deployment-stage optimization (affects training)
- Scenarios without clear reasoning vs memorization distinction

### Key Insights

The paper revises compute-optimal scaling laws by showing that more parameters (memorization) and more active FLOPs (reasoning) are orthogonal optimization axes. Classical dense models conflate these: adding parameters increases both FLOPs and capacity. MoE decouples them, enabling expert optimization for reasoning while controlling memorization budget.

## Reference

Optimal Sparsity of MoE Language Models for Reasoning. arXiv:2508.18672
- https://arxiv.org/abs/2508.18672
