---
name: gigaevo-llm-evolution
title: "GigaEvo: Open Source Optimization Framework Powered By LLMs And Evolution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.17592"
keywords: [Evolutionary Algorithms, LLM Mutation, Algorithm Optimization, Program Synthesis, Quality-Diversity]
description: "Evolve Python algorithms and programs using LLMs as mutation operators combined with MAP-Elites quality-diversity search, achieving competitive results on geometric optimization and algorithmic problems by iteratively mutating code informed by historical performance and lineage context."
---

# GigaEvo: LLM-Driven Evolutionary Optimization

Rather than optimizing neural network weights, this skill demonstrates how to evolve entire programs and algorithms using large language models as intelligent mutation operators. GigaEvo combines evolutionary computation (MAP-Elites quality-diversity algorithm) with LLM-based code generation, enabling discovery of novel algorithms for geometric optimization, combinatorial problems, and other domains where explicit algorithms may outperform learned models.

The core innovation is using LLMs not for inference, but as mutation operators that generate improved algorithm variants by analyzing parent code, performance metrics, and historical context about what changes succeeded.

## Core Concept

GigaEvo implements an evolutionary framework where:

1. **Population Management**: Python programs stored with their metrics and lineage information in a database
2. **Evolutionary Engine**: MAP-Elites algorithm maps solutions to a behavior space based on fitness and behavioral features
3. **Mutation via LLM**: LLMs generate offspring code by analyzing parent implementations, metrics, and historical mutations
4. **Fitness Evaluation**: Programs are executed and evaluated on benchmark tasks (geometric optimization, bin packing, etc.)

The system tracks bidirectional lineage (parent→offspring and offspring→parent) to enable context-aware code generation.

## Architecture Overview

- **Redis Database**: Stores evolutionary units with code, metrics, fitness scores, and lineage pointers
- **DAG Execution Engine**: Asynchronously processes programs through validation, complexity analysis, and evaluation stages
- **MAP-Elites Quality-Diversity**: Maintains diverse, high-performing solutions across behavior space dimensions
- **LangGraph Mutation Agent**: Constructs rich context from parent code and generates improved variants
- **Metrics Tracking**: Historical performance data to guide evolution direction

## Implementation Steps

The evolutionary process cycles through population initialization, evaluation, and mutation stages.

**1. Initialize Population and Behavior Space**

Create initial population of random programs and define the behavior space dimensions.

```python
def initialize_giga_evo(task_domain, population_size=100):
    """
    Initialize the evolutionary population with random programs.
    Sets up the behavior space for quality-diversity optimization.
    """
    population = []

    # Generate random initial programs
    for idx in range(population_size):
        if task_domain == "geometric":
            program = generate_random_geometric_algorithm()
        elif task_domain == "bin_packing":
            program = generate_random_packing_algorithm()

        individual = {
            'id': f'ind_{idx}',
            'code': program,
            'fitness': None,
            'behavior': None,
            'lineage': {'ancestors': [], 'descendants': []},
            'metrics': {}
        }

        population.append(individual)

    # Define behavior space dimensions (e.g., runtime, solution quality)
    behavior_space = {
        'fitness_dim': (0, 100),      # Quality score
        'efficiency_dim': (0, 1),     # Normalized runtime
        'complexity_dim': (0, 50)     # Code complexity
    }

    return population, behavior_space
```

**2. Execute and Evaluate Programs**

Run each program on benchmark tasks and collect fitness, behavior, and metric data.

```python
def evaluate_program(program_code, task_domain, task_instances):
    """
    Execute program on benchmark tasks and collect metrics.
    Handles crashes gracefully; invalid programs receive low fitness.
    """
    try:
        # Execute the program
        executor = ProgramExecutor()
        results = executor.run(program_code, task_instances, timeout=5.0)

        # Compute fitness from results
        fitness = compute_fitness_score(results, task_domain)

        # Extract behavior coordinates
        behavior = {
            'fitness': fitness,
            'efficiency': np.mean(results['runtimes']),
            'complexity': measure_code_complexity(program_code)
        }

        metrics = {
            'success_rate': results.get('success_rate', 0),
            'mean_quality': np.mean(results.get('qualities', [0])),
            'worst_quality': np.min(results.get('qualities', [0])),
            'runtime': np.mean(results['runtimes'])
        }

        return {
            'fitness': fitness,
            'behavior': behavior,
            'metrics': metrics,
            'valid': True
        }

    except Exception as e:
        return {
            'fitness': -1000,
            'behavior': None,
            'metrics': {'error': str(e)},
            'valid': False
        }
```

**3. Build Mutation Context from Lineage**

Assemble rich context about the parent program, its performance, and historical mutations to guide mutation prompts.

```python
def build_mutation_context(parent_individual, population_db):
    """
    Construct detailed context for LLM mutation prompts.
    Includes parent code, metrics, and successful historical changes.
    """
    context = {
        'parent_code': parent_individual['code'],
        'parent_fitness': parent_individual['fitness'],
        'parent_metrics': parent_individual['metrics'],
        'successful_ancestors': [],
        'descendant_performance': []
    }

    # Retrieve successful ancestors (parents with high fitness)
    for ancestor_id in parent_individual['lineage']['ancestors']:
        ancestor = population_db[ancestor_id]
        if ancestor['fitness'] > parent_individual['fitness'] * 0.8:
            context['successful_ancestors'].append({
                'id': ancestor_id,
                'code_snippet': ancestor['code'][:500],
                'fitness': ancestor['fitness']
            })

    # Retrieve how descendants performed (learning from mutations)
    for descendant_id in parent_individual['lineage']['descendants']:
        descendant = population_db[descendant_id]
        improvement = descendant['fitness'] - parent_individual['fitness']
        context['descendant_performance'].append({
            'improvement': improvement,
            'type': 'positive' if improvement > 0 else 'negative'
        })

    return context
```

**4. Generate Mutant via LLM**

Use LLM as mutation operator, providing context about what worked in the past.

```python
def mutate_via_llm(parent_individual, population_context, llm_agent):
    """
    Generate offspring code using LLM mutation operator.
    LLM receives parent code, metrics, and successful history.
    """
    mutation_prompt = f"""
    You are optimizing a Python algorithm for {parent_individual['domain']}.

    Parent Algorithm:
    {parent_individual['code']}

    Parent Performance:
    - Fitness Score: {parent_individual['fitness']:.2f}
    - Mean Quality: {parent_individual['metrics'].get('mean_quality', 'unknown')}
    - Runtime: {parent_individual['metrics'].get('runtime', 'unknown')}ms

    Successful Historical Changes:
    {format_successful_mutations(population_context['successful_ancestors'])}

    Generate an improved version by:
    1. Analyzing inefficiencies in the parent algorithm
    2. Applying one concrete improvement (e.g., better search strategy, data structure change, pruning rule)
    3. Keeping the core algorithm structure recognizable

    Return ONLY the improved Python code, no explanations.
    """

    mutant_code = llm_agent.generate_code(mutation_prompt)

    # Validate syntax
    try:
        compile(mutant_code, '<string>', 'exec')
        return {'code': mutant_code, 'valid': True}
    except SyntaxError:
        # Request repair if syntax errors
        return {'code': mutant_code, 'valid': False, 'error': 'syntax'}
```

**5. Maintain MAP-Elites Archive**

Update the quality-diversity archive with new individuals, replacing dominated solutions.

```python
def update_map_elites_archive(archive, individual, behavior_space):
    """
    Update MAP-Elites archive with new individual.
    Archives maintain diverse, high-performing solutions.
    """
    # Discretize behavior into grid cell
    cell = discretize_behavior(individual['behavior'], behavior_space)

    if cell not in archive:
        # Empty cell, add the individual
        archive[cell] = individual
        return True

    else:
        # Cell occupied, replace if individual is more fit
        current_best = archive[cell]

        if individual['fitness'] > current_best['fitness']:
            archive[cell] = individual
            return True

        return False
```

**6. Evolutionary Loop with Selection Pressure**

Complete cycle: select parents, mutate, evaluate, update archive.

```python
def evolutionary_loop(population, behavior_space, num_generations=50):
    """
    Main evolutionary loop integrating selection, mutation, evaluation, and archiving.
    Continues for specified generations or until convergence.
    """
    archive = {}  # MAP-Elites archive
    history = []

    for generation in range(num_generations):
        # Select parent (quality-diversity biased)
        parent = select_parent_from_population(population)

        # Mutate via LLM
        context = build_mutation_context(parent, population)
        mutant_result = mutate_via_llm(parent, context, llm_agent)

        if not mutant_result['valid']:
            continue  # Skip invalid mutations

        # Evaluate mutant
        eval_result = evaluate_program(mutant_result['code'], parent['domain'])

        mutant = {
            'code': mutant_result['code'],
            'fitness': eval_result['fitness'],
            'behavior': eval_result['behavior'],
            'lineage': {
                'ancestors': parent['lineage']['ancestors'] + [parent['id']],
                'descendants': []
            },
            'metrics': eval_result['metrics']
        }

        # Update archive
        added = update_map_elites_archive(archive, mutant, behavior_space)

        # Track best fitness
        best_fitness = max([ind['fitness'] for ind in archive.values()])
        history.append(best_fitness)

        population.append(mutant)

    return archive, history
```

## Practical Guidance

**When to Use GigaEvo:**
- Discovering explicit algorithms for geometric or combinatorial optimization
- Evolving program synthesis solutions with measurable, clear objectives
- Benchmarking against learned models to understand algorithm vs. learning tradeoffs
- Generating competitive baselines for algorithm research

**When NOT to Use:**
- Tasks requiring continuous function optimization (use gradient-based evolution)
- Real-time applications requiring tight latency bounds
- Domains where interpretability is not a goal (gradient descent may be simpler)

**Key Hyperparameters:**
- `mutation_prompt_quality`: Richer context → better mutations, but higher latency
- `population_size`: Larger population enables better quality-diversity tradeoff (100-500 typical)
- `num_generations`: Usually 30-100 generations sufficient for convergence
- `task_timeout`: Prevent infinite loops; 5-10 seconds typical for small instances

**Cost Optimization:**
- Use smaller LLM models for mutations (70-80% of large model quality at 1/10 cost)
- Batch-evaluate programs in parallel using executor DAG
- Cache successfully mutated patterns to reduce redundant LLM calls

**Integration with RL:**
Evolved algorithms can be evaluated in multi-agent benchmarks or used as baseline policies in environments. Combine with imitation learning to distill evolved algorithms into neural policies.

## Reference

Research paper: https://arxiv.org/abs/2511.17592
