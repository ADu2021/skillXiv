---
name: paceevolve-evolution-search
title: "PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.10657"
keywords: [evolutionary-search, LLM-optimization, long-horizon, progress-aware, context-management]
description: "Improves LLM-based evolutionary search by addressing context pollution, mode collapse, and weak collaboration through hierarchical context management, momentum-based backtracking, and adaptive sampling policies."
---

## Overview

Enhance LLM-driven evolutionary search for long-horizon optimization tasks. Address three key failure modes: context pollution from accumulated experiment data, mode collapse from imbalanced exploration-exploitation, and weak collaboration between parallel search trajectories.

## When to Use

- For autonomous optimization and search tasks requiring many candidate evaluations
- When LLMs guide evolutionary algorithms over extended search spaces
- For hyperparameter tuning, architecture search, or program synthesis
- When you need sustained self-improvement over multiple generations

## When NOT to Use

- For simple single-pass optimization
- When evolutionary search already works well without LLM guidance
- For tasks with very limited evaluation budget
- In low-latency applications

## Key Technical Components

### Hierarchical Context Management (HCM)

Prevent context pollution by pruning irrelevant historical data.

```python
# Hierarchical context management
class ContextManager:
    def __init__(self, max_context_tokens=2000):
        self.max_tokens = max_context_tokens
        self.experiment_history = []
        self.context_cache = {}

    def maintain_context_hierarchy(self):
        """Organize context by relevance tiers"""
        tiers = {
            "recent": [],      # Last N experiments
            "best": [],        # Best-performing experiments
            "diverse": []      # Diverse solution approaches
        }

        # Tier 1: Most recent (lexical recency)
        tiers["recent"] = self.experiment_history[-10:]

        # Tier 2: Best performance
        best_experiments = sorted(
            self.experiment_history,
            key=lambda x: x["fitness"],
            reverse=True
        )[:10]
        tiers["best"] = best_experiments

        # Tier 3: Diverse solutions (cluster representatives)
        tiers["diverse"] = self.select_diverse_representatives(
            self.experiment_history,
            k=10
        )

        return tiers

    def select_diverse_representatives(self, experiments, k=10):
        """Select diverse experiments by clustering"""
        if len(experiments) <= k:
            return experiments

        # Cluster by solution structure
        clusters = self.cluster_by_structure(experiments)

        # Select representative from each cluster
        representatives = []
        for cluster in clusters:
            # Pick best from each cluster
            best_in_cluster = max(cluster, key=lambda x: x["fitness"])
            representatives.append(best_in_cluster)

        return representatives[:k]

    def prune_context(self, current_problem):
        """Remove irrelevant historical data"""
        pruned_history = []

        for experiment in self.experiment_history:
            # Keep if: recent, high-performing, or relevant to current problem
            recency_score = self.compute_recency(experiment)
            performance_score = self.compute_performance(experiment)
            relevance_score = self.compute_problem_relevance(
                experiment,
                current_problem
            )

            overall_score = (
                0.3 * recency_score +
                0.3 * performance_score +
                0.4 * relevance_score
            )

            if overall_score > RETENTION_THRESHOLD:
                pruned_history.append(experiment)

        self.experiment_history = pruned_history
        return pruned_history

    def compute_recency(self, experiment):
        """Score based on how recent"""
        age = len(self.experiment_history) - self.experiment_history.index(experiment)
        return 1.0 / (1.0 + age)

    def compute_performance(self, experiment):
        """Normalize fitness score"""
        all_fitness = [e["fitness"] for e in self.experiment_history]
        min_f, max_f = min(all_fitness), max(all_fitness)
        if min_f == max_f:
            return 0.5
        return (experiment["fitness"] - min_f) / (max_f - min_f)

    def compute_problem_relevance(self, experiment, problem):
        """Semantic relevance to current problem"""
        # Simple approximation: use problem_id similarity
        if experiment.get("problem_id") == problem.get("id"):
            return 1.0
        # Could use more sophisticated similarity
        return 0.1
```

### Momentum-Based Backtracking (MBB)

Escape local optima by reverting to diverse solutions.

```python
# Momentum-based backtracking
class MomentumBacktracker:
    def __init__(self, backtrack_window=5):
        self.backtrack_window = backtrack_window
        self.trajectory_history = []
        self.fitness_trend = []

    def detect_local_optimum(self):
        """Check if stuck in local optimum"""
        # Analyze recent fitness trend
        if len(self.fitness_trend) < self.backtrack_window:
            return False

        recent_trend = self.fitness_trend[-self.backtrack_window:]

        # Stagnation: fitness plateau
        improvement = max(recent_trend) - min(recent_trend)
        stagnation = improvement < STAGNATION_THRESHOLD

        # Diversity loss: all solutions similar
        solution_similarity = self.compute_avg_similarity()
        low_diversity = solution_similarity > SIMILARITY_THRESHOLD

        is_stuck = stagnation and low_diversity

        return {
            "is_stuck": is_stuck,
            "stagnation_score": improvement,
            "diversity_score": 1.0 - solution_similarity
        }

    def backtrack_with_momentum(self, current_fitness):
        """Revert to promising past solution with momentum"""
        # Find checkpoint with good fitness and different structure
        checkpoint = self.select_backtrack_checkpoint()

        if checkpoint is None:
            return None

        # Combine checkpoint with momentum from current trajectory
        current_solution = self.trajectory_history[-1]["solution"]
        checkpoint_solution = checkpoint["solution"]

        # Blending: weighted combination
        momentum = self.compute_momentum_vector(current_solution, checkpoint_solution)

        # Backtrack solution = checkpoint + momentum adjustment
        backtrack_solution = self.blend_solutions(
            checkpoint_solution,
            momentum,
            alpha=0.7
        )

        return {
            "solution": backtrack_solution,
            "checkpoint_fitness": checkpoint["fitness"],
            "momentum_direction": momentum
        }

    def select_backtrack_checkpoint(self):
        """Select good past solution with different structure"""
        # Look back in history for high-fitness, diverse solutions
        candidates = []

        for i in range(max(0, len(self.trajectory_history) - 20), len(self.trajectory_history)):
            solution = self.trajectory_history[i]

            # Must have reasonable fitness
            if solution["fitness"] > BACKTRACK_FITNESS_THRESHOLD:
                # And be structurally different from recent solutions
                diversity = self.compute_diversity_from_recent(solution)
                if diversity > DIVERSITY_THRESHOLD:
                    candidates.append(solution)

        if not candidates:
            return None

        # Select best candidate
        return max(candidates, key=lambda x: x["fitness"])

    def blend_solutions(self, base_solution, momentum, alpha=0.7):
        """Blend checkpoint with momentum direction"""
        # Linear combination in solution space
        blended = alpha * base_solution + (1 - alpha) * momentum
        return blended

    def compute_momentum_vector(self, current, checkpoint):
        """Direction of progress from checkpoint to current"""
        return current - checkpoint

    def compute_diversity_from_recent(self, solution):
        """How different from recent solutions"""
        recent_solutions = [s["solution"] for s in self.trajectory_history[-5:]]
        similarities = [
            self.compute_solution_similarity(solution["solution"], recent)
            for recent in recent_solutions
        ]
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
```

### Self-Adaptive Sampling Policy

Dynamically balance exploration and exploitation.

```python
# Self-adaptive sampling
class AdaptiveSamplingPolicy:
    def __init__(self):
        self.exploration_rate = 0.5
        self.sampling_history = []

    def compute_adaptive_rate(self, recent_progress):
        """Adjust exploration based on progress"""
        # High progress -> exploit (lower exploration)
        # Stagnation -> explore (higher exploration)

        improvement = np.mean(recent_progress)

        if improvement > HIGH_PROGRESS_THRESHOLD:
            # Good progress, shift to exploitation
            self.exploration_rate = max(0.1, self.exploration_rate - 0.1)
        elif improvement < LOW_PROGRESS_THRESHOLD:
            # Poor progress, increase exploration
            self.exploration_rate = min(0.9, self.exploration_rate + 0.1)

        return self.exploration_rate

    def sample_next_candidate(self, best_candidates, random_candidates, exploration_rate):
        """Choose between exploiting best or exploring random"""
        if np.random.random() < exploration_rate:
            # Exploration: sample from random pool
            return np.random.choice(random_candidates)
        else:
            # Exploitation: sample from best pool
            return np.random.choice(best_candidates)

    def integrate_backtracking(self, backtrack_solution, exploration_rate):
        """Incorporate backtracking into sampling"""
        # When backtracking, temporarily increase exploration
        # to escape local basin
        backtrack_exploration = min(
            exploration_rate + 0.3,
            0.9
        )

        return backtrack_exploration

    def compute_sampling_efficiency(self):
        """Track whether sampling strategy is effective"""
        # Ratio of improvements to samples
        total_samples = len(self.sampling_history)
        improvements = sum(
            1 for i in range(1, len(self.sampling_history))
            if self.sampling_history[i]["fitness"] > self.sampling_history[i-1]["fitness"]
        )

        efficiency = improvements / total_samples

        return efficiency
```

### Integration: Complete Loop

Combine all components into complete search loop.

```python
# Complete evolutionary search loop
class PACEevolveSearch:
    def __init__(self, llm_generator):
        self.generator = llm_generator
        self.context_manager = ContextManager()
        self.backtracker = MomentumBacktracker()
        self.sampler = AdaptiveSamplingPolicy()

    def search_iteration(self, problem, budget=100):
        """Single generation of evolutionary search"""
        for step in range(budget):
            # 1. Maintain context hierarchy, prune pollution
            context = self.context_manager.maintain_context_hierarchy()
            self.context_manager.prune_context(problem)

            # 2. Generate candidates using LLM
            candidates = self.generator.generate_candidates(
                problem,
                context,
                num_candidates=10
            )

            # 3. Evaluate candidates
            candidates = self.evaluate_candidates(candidates)

            # 4. Check for local optimum
            stuck = self.backtracker.detect_local_optimum()

            if stuck["is_stuck"]:
                # Apply momentum-based backtracking
                backtrack = self.backtracker.backtrack_with_momentum(
                    max(e["fitness"] for e in candidates)
                )
                if backtrack:
                    candidates.append(backtrack["solution"])

            # 5. Adaptive sampling for next generation
            best_candidates = sorted(candidates, key=lambda x: x["fitness"], reverse=True)[:5]
            self.sampler.compute_adaptive_rate(
                [e["fitness"] for e in candidates]
            )

            # 6. Store best and continue
            if best_candidates:
                self.context_manager.experiment_history.append(best_candidates[0])

        return self.context_manager.experiment_history
```

## Performance Characteristics

- State-of-the-art on LLM-SR and KernelBench benchmarks
- Discovers solutions exceeding previous records
- Sustained improvement over 100+ iterations
- Effective on Modded NanoGPT and similar tasks

## Recommendations

- Initialize exploration rate at 0.5; let it adapt
- Backtrack when stagnation detected; don't backtrack too frequently
- Prune context every 10-20 iterations to prevent pollution
- Monitor sampling efficiency; adjust thresholds if needed

## References

- Context pollution prevents long-horizon optimization
- Mode collapse causes premature convergence
- Momentum-based backtracking enables escape from local optima
- Adaptive sampling maintains balance without manual tuning
