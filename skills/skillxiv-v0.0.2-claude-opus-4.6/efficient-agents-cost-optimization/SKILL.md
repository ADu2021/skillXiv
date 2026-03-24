---
name: efficient-agents-cost-optimization
title: Efficient Agents - Cost-Optimized LLM Agent Design
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.02694
keywords: [cost-optimization, agents, efficiency, inference]
description: "Systematically optimize agent system costs via empirical analysis of LLM, planning, memory, and search components achieving 28.4% cost reduction."
---

## Efficient Agents: Optimizing for Cost-Performance Trade-Off

This work conducts systematic empirical analysis of LLM agent cost-effectiveness. Using the "cost-of-pass" metric (expected cost for correct solution), it identifies which architectural choices matter most. Key finding: moderate planning complexity, simple memory, and multiple search sources outperform complex designs at fraction of the cost.

### Core Concept

LLM agents are expensive—solving tasks requires many LLM calls. The paper's key insight: not all complexity is worth it. By isolating architectural components (LLM backbone, planning, memory, search) and evaluating their cost-performance trade-off, it designs "Efficient Agents" achieving 96.7% of SOTA performance at 28.4% lower cost.

### Architecture Overview

- **Cost-of-Pass Metric**: Expected monetary cost to generate correct solution
- **Component-Level Analysis**: Isolate LLM, planning, memory, search costs
- **Efficient Agent Framework**: Optimized architecture for cost-effectiveness
- **Empirical Benchmark**: GAIA dataset with difficulty tiers
- **Multi-Search Strategy**: Multiple free/cheap sources (Google, Wikipedia, Bing, Baidu, DuckDuckGo)

### Implementation Steps

**Step 1: Define Cost Metrics**

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class CostMetrics:
    """Comprehensive cost tracking for agent execution."""
    llm_cost: float  # $ spent on LLM calls
    api_cost: float  # $ spent on external APIs
    total_cost: float
    num_llm_calls: int
    num_api_calls: int
    success: bool
    solution_correct: bool

class CostAnalyzer:
    """Compute cost-of-pass and related metrics."""

    def __init__(self, gpt4_cost_per_1k=0.03, gpt35_cost_per_1k=0.0005,
                 api_costs: Dict[str, float] = None):
        self.gpt4_rate = gpt4_cost_per_1k / 1000
        self.gpt35_rate = gpt35_cost_per_1k / 1000
        self.api_costs = api_costs or {'web_search': 0.001, 'calculator': 0.0001}

    def compute_cost_of_pass(self, results: List[CostMetrics]) -> float:
        """
        Cost-of-pass: expected cost to get one correct answer.
        = sum(cost_i * 1[failed_i]) / num_successes
        """
        failed_costs = [m.total_cost for m in results if not m.solution_correct]
        num_correct = sum(1 for m in results if m.solution_correct)

        if num_correct == 0:
            return float('inf')

        cost_of_pass = sum(failed_costs) / num_correct
        return cost_of_pass

    def compute_efficiency_score(self, results: List[CostMetrics]) -> Dict:
        """Compute efficiency metrics."""
        costs = [m.total_cost for m in results]
        successes = [m.solution_correct for m in results]

        return {
            'mean_cost': np.mean(costs),
            'success_rate': sum(successes) / len(results),
            'cost_per_success': np.mean([c for c, s in zip(costs, successes) if s]),
            'cost_of_pass': self.compute_cost_of_pass(results),
            'efficiency_ratio': sum(successes) / (sum(costs) + 0.01)  # Successes per $
        }
```

**Step 2: Analyze LLM Backbone Trade-Off**

```python
class LLMBackboneAnalysis:
    """Compare different LLM backbones on cost-performance."""

    def __init__(self, cost_analyzer: CostAnalyzer):
        self.analyzer = cost_analyzer

    def evaluate_backbone(self, model_name: str, test_tasks: List[str],
                         num_runs: int = 3) -> Dict:
        """Evaluate a specific LLM backbone."""
        results = []

        for task in test_tasks:
            for _ in range(num_runs):
                # Run agent with this backbone
                agent = Agent(model=model_name)
                success, cost, steps = agent.solve(task)

                results.append(CostMetrics(
                    llm_cost=cost,
                    api_cost=0,
                    total_cost=cost,
                    num_llm_calls=steps,
                    num_api_calls=0,
                    success=success,
                    solution_correct=success
                ))

        metrics = self.analyzer.compute_efficiency_score(results)
        return metrics

    def compare_backbones(self, backbones: List[str], test_tasks: List[str]) -> Dict:
        """Compare multiple LLM backbones."""
        comparison = {}

        for backbone in backbones:
            metrics = self.evaluate_backbone(backbone, test_tasks)
            comparison[backbone] = metrics

            print(f"{backbone}: Cost=${metrics['cost_of_pass']:.2f}, "
                  f"Success={metrics['success_rate']:.1%}")

        return comparison

# Finding: GPT-4.1 (fastest reasoner) best for complex tasks
# Finding: GPT-3.5 (cheap) good for simple tasks
# Recommendation: GPT-4.1 for consistent cost-effectiveness
```

**Step 3: Analyze Planning Complexity**

```python
class PlanningAnalysis:
    """Compare different planning strategies."""

    def __init__(self):
        self.planning_strategies = {
            'direct': self._direct_solving,
            'simple_planning': self._simple_planning,
            'detailed_planning': self._detailed_planning,
            'adaptive_planning': self._adaptive_planning
        }

    def _direct_solving(self, task: str, model) -> float:
        """Solve task directly without planning."""
        prompt = f"Solve this: {task}"
        return model.estimate_cost(prompt)

    def _simple_planning(self, task: str, model) -> float:
        """Simple: break into 2-3 steps."""
        prompt = f"""Task: {task}
Step 1: What information is needed?
Step 2: Find that information
Step 3: Solve"""
        return model.estimate_cost(prompt)

    def _detailed_planning(self, task: str, model) -> float:
        """Detailed: break into many steps, verify each."""
        prompt = f"""Task: {task}
1. Analyze the question carefully
2. Identify required information
3. Search for each piece
4. Verify accuracy
5. Reason through solution
6. Check answer
7. Format result"""
        return model.estimate_cost(prompt)

    def _adaptive_planning(self, task: str, model) -> float:
        """Adaptive: complexity based on task difficulty."""
        difficulty = estimate_difficulty(task)
        if difficulty < 3:
            return self._direct_solving(task, model)
        elif difficulty < 6:
            return self._simple_planning(task, model)
        else:
            return self._detailed_planning(task, model)

    def evaluate_planning(self, tasks: List[str], model):
        """Compare planning strategies."""
        comparison = {}

        for strategy_name, strategy_fn in self.planning_strategies.items():
            costs = []
            for task in tasks:
                cost = strategy_fn(task, model)
                costs.append(cost)

            avg_cost = np.mean(costs)
            comparison[strategy_name] = avg_cost

            print(f"{strategy_name}: ${avg_cost:.3f}")

        # Finding: Moderate planning (simple_planning) is sweet spot
        best = min(comparison, key=comparison.get)
        return best
```

**Step 4: Memory Optimization**

```python
class MemoryOptimization:
    """Compare different memory strategies."""

    def compare_memory_strategies(self, tasks: List[str], model) -> Dict:
        """
        1. No memory: forget everything (wastes computation)
        2. Simple memory: keep observations + actions (efficient)
        3. Full history: keep everything (expensive, may hurt)
        4. Smart memory: keep only useful facts (requires filtering)
        """
        strategies = {
            'no_memory': self._no_memory_agent,
            'simple_memory': self._simple_memory_agent,
            'full_history': self._full_history_agent,
            'smart_memory': self._smart_memory_agent
        }

        results = {}
        for strategy_name, strategy_fn in strategies.items():
            costs = []
            for task in tasks:
                cost = strategy_fn(task, model)
                costs.append(cost)

            results[strategy_name] = np.mean(costs)

        return results

    def _simple_memory_agent(self, task: str, model):
        """Keep only essential: task, observations, actions."""
        memory = {
            'task': task,
            'observations': [],  # Facts learned
            'actions': []  # Actions taken
        }

        # LLM called with minimal context
        prompt = f"Task: {memory['task']}\nFacts known: {len(memory['observations'])}"
        return model.estimate_cost(prompt)

    def _full_history_agent(self, task: str, model):
        """Keep every LLM call, search result, etc."""
        # Grows unboundedly, expensive, may confuse model
        return float('inf')

# Finding: Simple memory (obs + actions) is optimal
# No memory: redundant computation
# Full history: too expensive, diminishing returns
# Recommendation: simple_memory strategy
```

**Step 5: Multi-Search Source Strategy**

```python
class SearchStrategy:
    """Optimize web search strategy."""

    def __init__(self):
        self.search_sources = {
            'google': {'cost': 0.001, 'quality': 0.9},
            'wikipedia': {'cost': 0.0, 'quality': 0.7},
            'bing': {'cost': 0.001, 'quality': 0.85},
            'baidu': {'cost': 0.0005, 'quality': 0.75},
            'duckduckgo': {'cost': 0.0, 'quality': 0.6}
        }

    def evaluate_search_combination(self, task: str, required_facts: int = 3) -> float:
        """
        Multi-source search: query multiple engines, aggregate results.
        Cheaper sources + one quality source.
        """
        total_cost = 0

        # Free sources (Wikipedia, DuckDuckGo)
        free_results = 2 * (0.0 + 0.0)

        # One paid source (Google) for accuracy
        paid_results = 1 * 0.001

        # Total for this task
        task_search_cost = free_results + paid_results

        return task_search_cost

    def compare_search_strategies(self, tasks: List[str]) -> Dict:
        """Single vs. multi-source."""
        strategies = {
            'single_google': 0.001 * len(tasks),
            'single_wikipedia': 0.0 * len(tasks),
            'multi_source': sum(self.evaluate_search_combination(t) for t in tasks)
        }

        return strategies

# Finding: Multi-source approach gets similar quality at lower cost
# Free sources (Wikipedia, DuckDuckGo) are surprisingly effective
# One paid search (Google) for verification
# Recommendation: Use multiple sources
```

**Step 6: Efficient Agent Architecture**

```python
class EfficientAgent:
    """
    Optimized agent combining findings from component analysis.
    """
    def __init__(self, model_name: str = 'gpt-4.1'):
        self.model_name = model_name
        self.model = load_model(model_name)

        # Optimized configuration based on empirical analysis
        self.config = {
            'backbone': model_name,
            'planning_complexity': 'simple',  # 8 max reasoning steps
            'memory_strategy': 'simple',  # Only observations + actions
            'max_steps': 8,
            'search_sources': ['wikipedia', 'google', 'bing', 'baidu', 'duckduckgo']
        }

    def solve(self, task: str) -> Tuple[str, float]:
        """Solve task with optimized efficiency."""
        # Simple planning
        plan_prompt = f"""Task: {task}

Steps:
1. Identify key information needed
2. Search for it
3. Solve"""

        # Execute with simple memory
        memory = {'task': task, 'facts': []}
        total_cost = 0.0

        for step in range(self.config['max_steps']):
            # LLM decision (minimal context)
            prompt = f"Task: {task}\nKnown facts: {len(memory['facts'])}\nNext action?"
            response = self.model.generate(prompt)
            total_cost += estimate_llm_cost(response)

            # Multi-source search if needed
            if 'search' in response.lower():
                search_cost = self._multi_search(response, memory)
                total_cost += search_cost

            # Check if solved
            if 'answer' in response.lower():
                return response, total_cost

        return "Unable to solve", total_cost

    def _multi_search(self, query: str, memory: Dict) -> float:
        """Search across multiple sources."""
        cost = 0.0
        for source in self.config['search_sources']:
            if source == 'wikipedia':
                results = search_wikipedia(query)
                cost += 0.0
            else:
                results = search_engine(query, source)
                cost += 0.001

            memory['facts'].extend(results)

        return cost

def evaluate_efficient_agent(agent: EfficientAgent, test_tasks: List[str]) -> Dict:
    """Evaluate optimized agent."""
    results = []

    for task in test_tasks:
        answer, cost = agent.solve(task)
        success = is_correct(answer, task)

        results.append(CostMetrics(
            llm_cost=cost,
            api_cost=0,
            total_cost=cost,
            num_llm_calls=1,
            num_api_calls=0,
            success=success,
            solution_correct=success
        ))

    analyzer = CostAnalyzer()
    metrics = analyzer.compute_efficiency_score(results)

    return metrics
```

### Practical Guidance

**When to Use:**
- Cost-sensitive agent deployments
- Large-scale agent systems (cost compounds)
- Scenarios with budget constraints
- Production inference

**When NOT to Use:**
- Research/exploration (full complexity may discover better methods)
- Single-run tasks (cost irrelevant for one-offs)
- Scenarios requiring maximum accuracy regardless of cost

**Key Findings:**
- GPT-4.1 provides best cost-performance
- Simple planning (8 steps) beats both direct and elaborate planning
- Simple memory (observations + actions) better than full history
- Multi-source search achieves quality at lower cost (28.4% savings)

### Reference

**Paper**: Efficient Agents: Building Effective Agents While Reducing Cost (2508.02694)
- 96.7% of SOTA performance at 28.4% lower cost
- Cost-of-pass metric for evaluation
- Systematic component-level analysis
