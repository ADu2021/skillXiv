---
name: agentic-context-engineering-evolution
title: "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.04618"
keywords: [context optimization, dynamic prompts, in-context learning, agent memory, self-improvement]
description: "Enable language models to improve via context adaptation rather than weight updates. Use ACE (Agentic Context Engineering) to treat contexts as evolving playbooks that accumulate, refine, and organize strategies through modular generation, reflection, and curation processes. Achieve +10.6% agent benchmark gains and +8.6% on finance tasks using small open-source models matching production-level performance."
---

# Agentic Context Engineering: Evolving Contexts

## Core Concept

Context optimization approaches suffer from brevity bias (eliminating domain insights) and context collapse (iterative rewrites eroding details). ACE treats contexts as evolving playbooks maintained through modular processes of generation, reflection, and curation. Contexts accumulate task-relevant strategies without size explosion, enabling self-improvement without weight updates.

## Architecture Overview

- **Modular Context Evolution**: Separate generation (create strategies), reflection (evaluate effectiveness), and curation (organize playbook)
- **Dynamic Memory**: Evolving system prompts and agent memories that grow and refine through interactions
- **Incremental Updates**: Strategic additions to context preserve previous insights while adding new ones
- **No Labeled Supervision**: Learn directly from natural execution feedback
- **Scalability with Long Context**: Leverage long-context models (128K tokens) for rich playbooks

## Implementation Steps

### 1. Context Generation and Reflection

Iteratively generate strategies and reflect on their effectiveness.

```python
class AgenticContextEvolver:
    def __init__(self, model='gpt-4.1', context_window_size=32000):
        self.model = model
        self.context_window = context_window_size
        self.playbook = {}  # Dict of strategies

    def generate_strategy(self, task_description, execution_history):
        """
        Generate new strategy based on task and prior execution.
        """

        prompt = f"""Task: {task_description}

Execution history (recent failures/successes):
{execution_history[-500:]}  # Recent context window

Based on the task and execution history, generate a new strategy:
- Clear objective
- Key steps
- Success criteria
- Failure recovery

Strategy:"""

        strategy = self.model.generate(prompt, max_tokens=500)
        return strategy

    def reflect_on_effectiveness(self, strategy, execution_results):
        """
        Evaluate whether strategy worked; propose refinements.
        """

        prompt = f"""Strategy being tested:
{strategy}

Execution results:
Success: {execution_results['success']}
Outcome: {execution_results['outcome']}
Time taken: {execution_results['duration']}

Reflection questions:
1. Did the strategy succeed? Why/why not?
2. What worked well?
3. What could be improved?
4. Should we refine, keep, or discard this strategy?

Reflection:"""

        reflection = self.model.generate(prompt, max_tokens=300)
        return reflection

    def curate_context(self, strategies, reflections, context_size_limit=8000):
        """
        Organize strategies into concise, useful playbook.
        Remove redundant or ineffective strategies.
        """

        prompt = f"""Current strategies and reflections:
{chr(10).join([f'Strategy: {s}\nReflection: {r}' for s, r in zip(strategies, reflections)])}

Organize these into a concise playbook:
1. Group related strategies
2. Remove duplicates
3. Keep highest-value strategies
4. Create decision tree for which strategy to use when
5. Total token budget: {context_size_limit} tokens

Curated playbook:"""

        curated = self.model.generate(prompt, max_tokens=context_size_limit)
        return curated

    def evolve_context(self, task_description, execution_log, iterations=5):
        """
        Main loop: generate → reflect → curate → repeat
        """

        current_context = ""  # Start empty

        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")

            # Step 1: Generate new strategy
            strategy = self.generate_strategy(task_description, execution_log)
            print(f"Generated strategy:\n{strategy[:200]}...\n")

            # Step 2: Test strategy
            test_results = self.test_strategy(strategy, task_description)

            # Step 3: Reflect
            reflection = self.reflect_on_effectiveness(strategy, test_results)
            print(f"Reflection:\n{reflection[:200]}...\n")

            # Step 4: Add to playbook
            self.playbook[f'strategy_{iteration}'] = {
                'strategy': strategy,
                'reflection': reflection,
                'success': test_results['success']
            }

            # Step 5: Curate playbook
            strategies_list = [s['strategy'] for s in self.playbook.values()]
            reflections_list = [s['reflection'] for s in self.playbook.values()]

            current_context = self.curate_context(
                strategies_list,
                reflections_list,
                context_size_limit=8000
            )

            # Update execution log
            execution_log += f"\n\n[Iteration {iteration+1}]\nStrategy: {strategy[:100]}...\nResult: {test_results['outcome']}"

        return current_context, self.playbook

    def test_strategy(self, strategy, task):
        """
        Test strategy on task; return success/failure results.
        """
        # Simplified: apply strategy and measure outcome
        result = self.model.generate(
            f"Task: {task}\n\nApply this strategy: {strategy}\n\nResult:",
            max_tokens=200
        )

        return {
            'outcome': result,
            'success': 'success' in result.lower() or 'completed' in result.lower(),
            'duration': 0.5  # Placeholder
        }
```

### 2. Integration with Agent Loop

Use evolved context as dynamic system prompt for agents.

```python
class ContextEvolvedAgent:
    def __init__(self, base_model, context_evolver):
        self.model = base_model
        self.evolver = context_evolver
        self.context_playbook = ""  # Current evolved context

    def step(self, task_instruction, current_observation):
        """
        Execute agent step using evolved context as system prompt.
        """

        # Compose prompt with evolved context
        system_prompt = f"""You are an agent solving tasks efficiently.

Playbook of successful strategies:
{self.context_playbook}

Current task: {task_instruction}

Observation: {current_observation}

Based on the playbook, select and execute the best strategy.
Action:"""

        # Generate action
        action = self.model.generate(system_prompt, max_tokens=500)

        return action

    def improve_context_from_episode(self, episode_trajectory):
        """
        After episode completion, improve context via ACE.
        """

        task_description = episode_trajectory['task']
        execution_log = episode_trajectory['log']
        success = episode_trajectory['success']

        # Generate improvement iteration
        strategy = self.evolver.generate_strategy(task_description, execution_log)

        # Reflect on episode
        reflection = self.evolver.reflect_on_effectiveness(
            strategy,
            {'success': success, 'outcome': execution_log[-500:], 'duration': 0.5}
        )

        # Update playbook
        self.evolver.playbook[f'episode_{id(episode_trajectory)}'] = {
            'strategy': strategy,
            'reflection': reflection,
            'success': success
        }

        # Curate updated context
        strategies = [s['strategy'] for s in self.evolver.playbook.values()]
        reflections = [s['reflection'] for s in self.evolver.playbook.values()]
        self.context_playbook = self.evolver.curate_context(strategies, reflections)

        print(f"Context improved. Playbook size: {len(self.context_playbook)} tokens")
```

### 3. Benchmark Evaluation

Test on agent benchmarks and finance tasks.

```python
def evaluate_ace_agent(
    agent,
    benchmark_tasks,
    num_episodes=100,
    context_update_frequency=10
):
    """
    Evaluate agent with evolving context.
    """

    results = {
        'success_rate': 0,
        'improvement_trajectory': [],
        'context_evolution': []
    }

    for episode in range(num_episodes):
        # Run episode
        task = benchmark_tasks[episode % len(benchmark_tasks)]
        success = agent.execute_task(task)

        results['success_rate'] += (1.0 if success else 0.0)

        # Periodic context improvement
        if (episode + 1) % context_update_frequency == 0:
            episode_trajectory = agent.get_episode_log()
            agent.improve_context_from_episode(episode_trajectory)

            # Track improvement
            avg_success = results['success_rate'] / (episode + 1)
            results['improvement_trajectory'].append(avg_success)
            results['context_evolution'].append({
                'episode': episode,
                'context_tokens': len(agent.context_playbook),
                'success_rate': avg_success
            })

    results['success_rate'] /= num_episodes

    return results

# Benchmark results
results = {
    'appworld_leaderboard': {
        'production_agent': '45% success (baseline)',
        'small_model_with_ace': {
            'success_rate': '45%',
            'improvement': 'Matches production agent',
            'model': 'Qwen-7B',
            'improvement_on_hard_splits': '+10.6%'
        }
    },
    'finance_benchmarks': {
        'improvement': '+8.6%',
        'task_types': [
            'Portfolio optimization',
            'Risk assessment',
            'Market analysis'
        ]
    }
}
```

## Practical Guidance

**Context Size Management**: Monitor context token count; curate regularly to avoid explosion. 8K-16K tokens sufficient for most playbooks.

**Reflection Quality**: Use capable model for reflection (GPT-4.1) even if base agent is smaller. Reflection drives improvement.

**Update Frequency**: Update context every 10-50 episodes. More frequent updates add overhead; less frequent miss improvement opportunities.

**Strategy Diversity**: Keep diverse strategies even if some work better. Diversity handles task distribution shifts.

## When to Use / When NOT to Use

**Use When**:
- Long-running agents that adapt to task distributions
- You want to improve without retraining
- Rich context representations are feasible (long context models)
- Continuous improvement is important

**NOT For**:
- One-off inference without feedback loops
- Severely compute-constrained environments
- Tasks requiring guaranteed, unchanged behavior

## Reference

This skill synthesizes findings from "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618). Context as playbook enables adaptation without weights.
