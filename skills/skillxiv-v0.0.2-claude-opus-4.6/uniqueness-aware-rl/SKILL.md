---
name: uniqueness-aware-rl
title: "Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.08763"
keywords: [exploration, RL, creative-problem-solving, diversity-reward, reasoning]
description: "Improves LLM reasoning by rewarding correct solutions that exhibit rare high-level strategies, preventing exploration collapse and discovering more diverse solution approaches across mathematics, physics, and medical reasoning."
---

## Overview

Enhance exploration in LLM reasoning by rewarding correct solutions based on their rarity of approach. Rather than treating all correct solutions equally, provide higher rewards for solutions using novel high-level strategies, preventing models from converging to single solution patterns.

## When to Use

- For reasoning tasks where multiple solution approaches exist
- When you want to maximize the diversity of reasoning strategies learned
- For improving pass@k metrics by exploring different solution paths
- For creative problem-solving where approach diversity is valuable

## When NOT to Use

- For tasks with unique optimal solutions
- When solution diversity doesn't matter for the application
- For single-solution problems
- When computational overhead of strategy analysis is unacceptable

## Key Technical Components

### Solution Strategy Clustering

Group solutions by high-level reasoning approach, ignoring surface-level variation.

```python
# Strategy clustering for exploration diversity
class SolutionStrategyClusterer:
    def __init__(self, strategy_classifier_model):
        self.classifier = strategy_classifier_model

    def extract_strategy(self, solution_text, problem):
        """Identify high-level strategy from solution"""
        prompt = f"""
        Analyze this {problem['domain']} solution:
        {solution_text}

        What high-level strategy does it use? (e.g., algebraic, graphical, numerical, recursive)
        Ignore surface-level variations in expression.
        """
        strategy = self.classifier.classify(prompt)
        return strategy

    def cluster_solutions(self, solutions, problem):
        """Group solutions by strategy"""
        strategy_groups = {}

        for solution in solutions:
            strategy = self.extract_strategy(solution["text"], problem)

            if strategy not in strategy_groups:
                strategy_groups[strategy] = []

            strategy_groups[strategy].append(solution)

        return strategy_groups

    def get_strategy_rarity(self, strategy, all_strategies):
        """Compute rarity score for a strategy"""
        strategy_counts = {}
        for s in all_strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        # Rarity = inverse of frequency
        frequency = strategy_counts.get(strategy, 1) / len(all_strategies)
        rarity = 1.0 / (frequency + 1e-6)

        # Normalize to [0, 1]
        normalized_rarity = 1.0 / (rarity + 1.0)  # Inverse sigmoid

        return normalized_rarity
```

### Uniqueness-Aware Reward Assignment

Assign higher rewards to rare but correct solutions.

```python
# Uniqueness-aware reward computation
class UniquenessReward:
    def compute_reward(self, solution, is_correct, strategy, all_strategies, base_reward=1.0):
        """Compute reward based on correctness and strategy rarity"""
        if not is_correct:
            return 0.0

        # Get rarity score for this strategy
        rarity_score = SolutionStrategyClusterer().get_strategy_rarity(
            strategy,
            all_strategies
        )

        # Reward formula: base * (1 + rarity_bonus)
        rarity_bonus = 0.5 * rarity_score  # Up to 50% bonus for rare strategies

        final_reward = base_reward * (1.0 + rarity_bonus)

        return final_reward

    def batch_compute_rewards(self, solutions, problem):
        """Compute rewards for all solutions in a rollout"""
        # Determine correctness for each
        correctness = [self.evaluate_correctness(sol, problem) for sol in solutions]

        # Extract strategies
        clusterer = SolutionStrategyClusterer(self.get_classifier())
        all_strategies = [clusterer.extract_strategy(sol["text"], problem) for sol in solutions]

        # Compute rewards
        rewards = []
        for solution, is_correct, strategy in zip(solutions, correctness, all_strategies):
            reward = self.compute_reward(
                solution,
                is_correct,
                strategy,
                all_strategies
            )
            rewards.append(reward)

        return rewards

    def evaluate_correctness(self, solution, problem):
        """Check if solution is correct"""
        # Domain-specific evaluation
        if problem["domain"] == "math":
            return self.verify_math_solution(solution, problem)
        elif problem["domain"] == "physics":
            return self.verify_physics_solution(solution, problem)
        elif problem["domain"] == "medical":
            return self.verify_medical_diagnosis(solution, problem)
        return False

    def get_classifier(self):
        """Initialize strategy classifier model"""
        # Would load actual model in practice
        pass
```

### Rollout Analysis and Re-weighting

Analyze rollout diversity and re-weight samples.

```python
# Rollout analysis for diversity
class RolloutAnalyzer:
    def analyze_rollout(self, solutions, problem):
        """Assess diversity and correctness of rollout"""
        clusterer = SolutionStrategyClusterer(self.get_classifier())

        # Cluster solutions by strategy
        strategy_groups = clusterer.cluster_solutions(solutions, problem)

        # Compute diversity metrics
        num_strategies = len(strategy_groups)
        total_correct = sum(
            1 for sol in solutions
            if clusterer.extract_strategy(sol, problem) and sol["is_correct"]
        )

        diversity_score = num_strategies / len(solutions)
        correctness_score = total_correct / len(solutions)

        return {
            "diversity_score": diversity_score,
            "correctness_score": correctness_score,
            "num_strategies": num_strategies,
            "strategy_distribution": {
                strategy: len(sols) for strategy, sols in strategy_groups.items()
            }
        }

    def should_reweight_rollout(self, analysis):
        """Determine if rollout needs re-weighting"""
        # Re-weight if collapse detected
        if analysis["diversity_score"] < 0.5:
            return True
        # Don't re-weight if good diversity
        return False

    def reweight_samples(self, solutions, analysis):
        """Re-weight solutions to encourage diversity"""
        reweighted = []
        clusterer = SolutionStrategyClusterer(self.get_classifier())

        for solution in solutions:
            strategy = clusterer.extract_strategy(solution, None)

            # Higher weight for underrepresented strategies
            base_weight = 1.0 / analysis["strategy_distribution"].get(strategy, 1)

            # Multiply by correctness score
            weight = base_weight * (1.0 if solution["is_correct"] else 0.5)

            reweighted.append({
                "solution": solution,
                "weight": weight
            })

        # Normalize weights
        total_weight = sum(w["weight"] for w in reweighted)
        for item in reweighted:
            item["weight"] /= total_weight

        return reweighted
```

### Training Loop with Exploration Tracking

Integrate uniqueness rewards into RL training.

```python
# RL training with uniqueness rewards
class UniquenessRL:
    def __init__(self, policy_model):
        self.policy = policy_model
        self.exploration_history = []

    def train_step(self, problem, num_rollouts=10):
        """Training step with uniqueness-aware rewards"""
        all_solutions = []
        all_rewards = []

        # Generate multiple solutions
        for _ in range(num_rollouts):
            solution = self.policy.generate(problem)
            all_solutions.append(solution)

        # Compute uniqueness-aware rewards
        reward_computer = UniquenessReward()
        rewards = reward_computer.batch_compute_rewards(all_solutions, problem)
        all_rewards.extend(rewards)

        # Analyze rollout diversity
        analyzer = RolloutAnalyzer()
        analysis = analyzer.analyze_rollout(all_solutions, problem)
        self.exploration_history.append(analysis)

        # If collapse detected, apply re-weighting
        if analyzer.should_reweight_rollout(analysis):
            solutions = analyzer.reweight_samples(all_solutions, analysis)
            rewards = [item["weight"] * reward for item, reward in zip(solutions, all_rewards)]

        # Policy gradient update
        total_loss = 0.0
        for solution, reward in zip(all_solutions, all_rewards):
            log_prob = self.policy.get_log_prob(solution)
            loss = -log_prob * reward
            total_loss += loss

        self.policy.backward(total_loss / len(all_solutions))
        self.policy.optimize()

        return {
            "loss": (total_loss / len(all_solutions)).item(),
            "diversity": analysis["diversity_score"],
            "avg_reward": sum(all_rewards) / len(all_rewards)
        }

    def evaluate_exploration(self, test_problems, num_rollouts=10):
        """Measure exploration quality across test set"""
        exploration_metrics = {}

        for problem in test_problems:
            solutions = [self.policy.generate(problem) for _ in range(num_rollouts)]

            analyzer = RolloutAnalyzer()
            analysis = analyzer.analyze_rollout(solutions, problem)

            exploration_metrics[problem["id"]] = analysis

        # Aggregate metrics
        avg_diversity = np.mean([a["diversity_score"] for a in exploration_metrics.values()])
        avg_correctness = np.mean([a["correctness_score"] for a in exploration_metrics.values()])

        return {
            "avg_diversity": avg_diversity,
            "avg_correctness": avg_correctness,
            "per_problem": exploration_metrics
        }
```

## Performance Characteristics

- Consistent improvements on pass@k metrics across domains
- Maintains or improves pass@1 (single attempt) performance
- Increases solution strategy diversity by 2-3x
- Works across mathematics, physics, and medical reasoning

## Integration Pattern

1. Generate multiple solutions for each problem (10-20 samples)
2. Evaluate correctness for each solution
3. Extract high-level strategy for each correct solution
4. Cluster solutions by strategy, compute rarity
5. Assign rewards based on correctness × rarity
6. Perform policy gradient update with uniqueness-aware rewards
7. Monitor exploration diversity; if collapse detected, re-weight

## Key Insights

- Surface-level variation hides true strategy diversity
- Rarity weighting prevents convergence to single approaches
- Curriculum learning helps: first maximize correctness, then optimize diversity

## References

- Exploration collapse in LLM RL causes loss of solution diversity
- Uniqueness rewards maintain diverse strategy exploration
- Strategy-level clustering enables rarity computation
- Re-weighting prevents collapse when detected
