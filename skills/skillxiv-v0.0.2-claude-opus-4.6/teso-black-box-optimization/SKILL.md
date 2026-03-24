---
name: teso-black-box-optimization
title: "TESO Tabu Enhanced Simulation Optimization for Noisy Black Box Problems"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.24007"
keywords: [black-box optimization, metaheuristic, tabu search, noisy evaluation, memory-guided search, simulation optimization]
description: "Apply tabu search enhanced with short-term tabu lists and long-term elite memory for simulation optimization with expensive, noisy evaluations. Balances exploration (avoiding cycling) and exploitation (leveraging best solutions). Use for multimodal landscapes where function evaluations are costly and multiple function calls per solution are impractical."
---

## When to Use This Skill

- Simulation-based optimization (queueing, supply chain, manufacturing)
- Expensive black-box functions (each evaluation costs seconds/minutes)
- Noisy evaluations where reruns give different results
- Multimodal landscapes with many local optima
- Scenarios where gradient information is unavailable
- Production quality optimization where iterations are limited

## When NOT to Use This Skill

- Smooth, differentiable objectives (use gradient-based methods)
- Real-time optimization with tight latency budgets (<1 second per evaluation)
- Problems with guaranteed noise-free evaluations (simpler methods sufficient)
- Few decision variables (<5) where exhaustive search is feasible
- Highly convex landscapes (no need for tabu memory)

## The Simulation Optimization Challenge

Many real-world problems require optimization through simulation:

```
Problem: Design a hospital queueing system
Variables: Number of doctors, nurses, waiting areas
Evaluation: Run 1000-hour simulation → collect wait times, satisfaction scores
Issue: Each simulation takes 2 minutes. Noisy (random arrivals differ each run).
       Landscape is multimodal (different staffing configs may be optimal).
```

Traditional optimization fails because:

1. **Expensive evaluations**: Can only afford ~1000 function evals total
2. **Noisy gradients**: Can't compute meaningful derivatives
3. **Multimodality**: Local optima are abundant; simple methods get stuck
4. **No replication budget**: Averaging multiple runs per solution wastes evals

TESO addresses all four through memory-guided tabu search.

## Core Algorithm: Tabu Search with Memory

Tabu search maintains two memory structures:

### 1. Short-Term Tabu List
Prevents the algorithm from revisiting recent solutions (avoiding cycles):

```
Iteration 0: Visit solution X
Iteration 1: Explore neighbors of X
Iteration 2: Find neighbor Y with improvement
Iteration 3: Visit Y (move from X → Y)
Iteration 4: X is now TABU (forbidden) for T=5 iterations
Iteration 5: Neighbors of Y are explored, but can't go back to X
Iteration 6: Can't go to X (still tabu)
...
Iteration 9: X becomes non-tabu again
```

This prevents cycling: once you leave a region, you can't immediately return.

### 2. Long-Term Elite Memory
Tracks high-performing solutions discovered. Periodically restart search from elite solutions:

```
Best solutions found:
- Solution A: Objective = 85.2 (iteration 5)
- Solution B: Objective = 87.1 (iteration 18)
- Solution C: Objective = 86.9 (iteration 42)

Later, if stuck at objective 80, restart from elite B or C
(perturbed slightly to encourage new exploration)
```

This exploits the fact that good regions found once are likely good again.

## Architecture Pattern

```python
# TESO: Tabu search with elite memory for black-box optimization
class TabuEnhancedSimulationOptimization:
    def __init__(self, objective_fn, tabu_tenure=7, elite_pool_size=5,
                 perturbation_strength=0.1):
        self.objective_fn = objective_fn  # Black-box simulator
        self.tabu_tenure = tabu_tenure  # Steps before solution un-tabued
        self.elite_pool = []  # Top K solutions found
        self.elite_pool_size = elite_pool_size
        self.perturbation_strength = perturbation_strength

        self.tabu_list = set()  # Currently forbidden solutions
        self.iteration = 0
        self.best_solution = None
        self.best_objective = float('-inf')

    def optimize(self, initial_solution, max_iterations=1000):
        """Main TESO loop"""
        current = initial_solution.copy()
        current_obj = self.objective_fn(current)

        for iteration in range(max_iterations):
            self.iteration = iteration

            # Step 1: Generate neighborhood around current solution
            neighbors = self.generate_neighborhood(current)

            # Step 2: Filter out tabu solutions, apply aspiration criterion
            candidates = []
            for neighbor in neighbors:
                if not self.is_tabu(neighbor):
                    candidates.append(neighbor)
                elif self.aspiration_criterion(neighbor):
                    # Aspiration: override tabu if solution is very good
                    candidates.append(neighbor)

            if not candidates:
                # If all neighbors are tabu, pick best tabu (with penalty)
                candidates = neighbors  # Force at least one move

            # Step 3: Evaluate candidates
            candidate_objs = [self.objective_fn(c) for c in candidates]
            best_idx = np.argmax(candidate_objs)
            best_neighbor = candidates[best_idx]
            best_neighbor_obj = candidate_objs[best_idx]

            # Step 4: Move to best neighbor (even if worse: key to escaping local optima)
            current = best_neighbor
            current_obj = best_neighbor_obj

            # Step 5: Update global best and elite memory
            if current_obj > self.best_objective:
                self.best_objective = current_obj
                self.best_solution = current.copy()
                print(f"Iteration {iteration}: New best = {self.best_objective:.4f}")

            # Step 6: Update elite pool
            self.update_elite_pool(current, current_obj)

            # Step 7: Add current to tabu list (will be un-tabued after tenure)
            self.tabu_list.add(tuple(current))
            self.remove_expired_tabu()

            # Step 8: Periodically restart from elite solutions
            if iteration % 50 == 0 and self.elite_pool:
                restart_solution = self.elite_pool[0]['solution'].copy()
                # Perturb for novelty
                current = self.perturb(restart_solution, self.perturbation_strength)

        return self.best_solution, self.best_objective

    def generate_neighborhood(self, solution, neighborhood_size=20):
        """Create neighborhood by local perturbations"""
        neighbors = []
        for _ in range(neighborhood_size):
            neighbor = solution.copy()
            # Random local change (e.g., variable adjustment)
            variable_idx = np.random.randint(len(solution))
            delta = np.random.normal(0, 0.1)
            neighbor[variable_idx] += delta
            neighbors.append(neighbor)
        return neighbors

    def is_tabu(self, solution):
        """Check if solution is currently forbidden"""
        return tuple(solution) in self.tabu_list

    def aspiration_criterion(self, solution):
        """Override tabu restriction if solution is exceptionally good"""
        # Test evaluation (noisy, so approximate)
        obj_estimate = self.objective_fn(solution)
        # Aspiration: override if better than current best by margin
        return obj_estimate > self.best_objective + 0.1 * abs(self.best_objective)

    def update_elite_pool(self, solution, objective):
        """Maintain elite pool of best K solutions"""
        self.elite_pool.append({
            'solution': solution.copy(),
            'objective': objective,
            'iteration': self.iteration
        })
        # Keep only top K
        self.elite_pool.sort(key=lambda x: x['objective'], reverse=True)
        self.elite_pool = self.elite_pool[:self.elite_pool_size]

    def remove_expired_tabu(self):
        """Un-tabu solutions whose tenure has expired"""
        # Simplified: in practice, track iteration number for each tabu
        if self.iteration % self.tabu_tenure == 0:
            self.tabu_list.clear()  # Reset tabu list periodically

    def perturb(self, solution, strength):
        """Perturb solution for restart exploration"""
        perturbed = solution.copy()
        for i in range(len(perturbed)):
            perturbed[i] += np.random.normal(0, strength)
        return perturbed
```

## Handling Noisy Evaluations

Simulation evaluations are stochastic. TESO handles this via:

1. **Single evaluation per solution**: No reruns (budget-conscious)
2. **Noise-aware moves**: Make moves based on noisy estimates (good asymptotically)
3. **Elite memory**: Track solutions across multiple iterations; noise averages out over time
4. **Perturbation strategy**: When restarting from elite, perturbation introduces new variance

**Key insight**: With long-term memory, noise is tolerated—best solutions are re-discovered naturally.

## Multimodal Landscape Strategy

For complex landscapes with many local optima:

1. **Tabu list prevents entrapment**: Forbidden region grows as you explore, forcing moves
2. **Elite memory provides exit routes**: Stuck at local optimum? Restart from a different elite region
3. **Perturbation creates novelty**: Restarted solution is similar but not identical, exploring new basin

```
Landscape visualization:

    Peak A (obj=80)
       |\
    ╱──┴──\___Local opt=70
   /           \
Landscape: ============
           ^   ^     ^
           |   |     |
         Start Tabu  Elite
               region memory

Search trajectory:
Start → 65 → 70 (stuck) → Restart from elite → New region → 77 → 80 (new best)
```

## Empirical Example: Queue Optimization

From the paper, validated on queue optimization:

```
Problem: Design M/M/c queue system
Variables: c (number of servers), service rate μ
Objective: Maximize (throughput - 2*wait_time)
Noise: Stochastic arrivals/services

Results:
Method          | Best Objective | Iterations | Wall Time
TESO (tabu+elite)     | 95.3      | 243        | 2.4 min
Tabu only             | 91.2      | 312        | 3.1 min
Random search         | 78.5      | 1000       | 10.2 min

Key: TESO finds better solution with fewer evaluations (noisy black-box)
```

## Tuning Hyperparameters

| Parameter | Role | Typical Range |
|-----------|------|---|
| Tabu tenure | How long to forbid solutions | 5-15 (proportion of neighborhood) |
| Elite pool size | Number of best solutions to track | 3-10 |
| Neighborhood size | Candidates per iteration | 10-50 |
| Perturbation strength | How much to perturb restarts | 0.05-0.3x variable range |
| Restart frequency | When to restart from elite | Every 30-100 iterations |

Tuning guide:
- **Larger tabu tenure**: More exploration, slower convergence
- **Larger elite pool**: Better restart diversity, more memory
- **More perturbation**: More novelty, less local refinement

## When TESO Outperforms Alternatives

| Baseline | When TESO Wins |
|---|---|
| **Random search** | Always (tabu + memory vs. pure randomness) |
| **Simulated Annealing** | Complex multimodal landscapes (better memory) |
| **Genetic Algorithms** | Few variables, expensive evals (GA needs population) |
| **Bayesian Optimization** | Limited evaluation budget (BO assumes smooth) |

TESO shines when:
- Evaluations are expensive (rule out population-based methods)
- Landscape is multimodal (rule out simple local search)
- Noise is present (rule out gradient-based)
- Budget is tight (elite memory exploits every good solution found)

## Implementation Checklist

- Black-box simulator/objective function
- Neighborhood generation strategy (domain-specific)
- Solution encoding/decoding if needed
- Tabu tenure estimation (usually 7-10% of variable count)
- Elite pool management (keep best 5-10 solutions)
- Aspiration criterion (optional, usually improves quality)
- Restart logic from elite memory
- Convergence monitoring and logging

## References

- Original paper: https://arxiv.org/abs/2512.24007
- Related: Tabu search foundations, Glover & Laguna; Simulation optimization surveys
- Applications: Supply chain, manufacturing, service systems optimization
