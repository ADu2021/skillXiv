---
name: cot-dataset-cold-start
title: "One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.02338"
keywords: [long-chain-of-thought, dataset, open-source, reinforcement-learning, cold-start]
description: "Enable open-source reasoning model development with a 100K-instance Long CoT Collection, scaling from 1K o1 seed samples through guided synthesis with GPT-4o, achieving 2-3× RL performance gains."
---

# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL

## Core Concept

Open-source reasoning model development faces a critical bottleneck: the "cold-start problem." High-quality long chain-of-thought data typically comes from proprietary models like OpenAI's o1, but researchers can't train models on private outputs. This paper demonstrates that short-CoT LLMs can generate high-quality long reasoning when guided by learned patterns. The approach creates a 100K-instance Long CoT Collection by (1) manually collecting 1K seed instances from o1, (2) scaling to 100K through GPT-4o synthesis guided by reasoning flows, and (3) filtering for correctness. Models initialized on this dataset achieve 2-3× greater RL performance gains compared to baselines, enabling independent reasoning model development without reliance on proprietary models.

## Architecture Overview

- **Three-Phase Scaling Pipeline**: Seed collection → guided synthesis → quality filtering
- **Reasoning Flow Extraction**: Step-by-step outlines capturing solution structure
- **Thought Budget Control**: Manages reasoning token allocation to prevent overthinking
- **Open-Source Synthesis**: Uses GPT-4o for guidance but not as data source
- **100K Scale Dataset**: Sufficient for RL initialization with strong performance gains
- **Domain Generalization**: Works beyond math (demonstrated on GPQA, multimodal tasks)

## Implementation

1. **Seed Dataset Creation**: Manual collection of high-quality long CoT from o1

```python
def create_seed_dataset(num_seed_samples=1000):
    """
    Manually collect high-quality long CoT demonstrations.
    Use o1 outputs as seed reasoning patterns.
    """
    seed_dataset = []

    for i in range(num_seed_samples):
        # Sample from diverse problem domains
        problem = select_problem_from_o1(i)

        # Get o1 reasoning (via API or cached results)
        o1_response = query_o1_api(problem)

        # Extract reasoning flow (high-level structure)
        reasoning_flow = extract_reasoning_flow(o1_response)

        # Extract thought budget (token allocation info)
        thought_budget = estimate_thought_budget(o1_response)

        # Store seed instance
        seed_instance = {
            'id': f'seed_{i}',
            'problem': problem,
            'o1_reasoning': o1_response['thinking'],
            'solution': o1_response['answer'],
            'reasoning_flow': reasoning_flow,
            'thought_budget': thought_budget
        }

        seed_dataset.append(seed_instance)

    return seed_dataset

def extract_reasoning_flow(o1_response):
    """
    Extract high-level reasoning structure from o1 output.
    Captures step-by-step outline without detailed explanations.
    """
    flow_prompt = f"""
    Summarize the reasoning approach in 5-10 steps:

    Thinking: {o1_response['thinking'][:2000]}

    Return only the step outline.
    """

    # Use GPT-4o to extract structure
    flow = call_gpt4o(flow_prompt)
    return flow
```

2. **Scaling Phase: Guided Synthesis**: Generate 100K instances using retrieved demonstrations

```python
def scale_to_100k_samples(seed_dataset, target_problems, scale_factor=100):
    """
    Scale seed dataset to 100K by generating reasoning for new problems.
    Use seed reasoning flows as guides, synthesize with GPT-4o.
    """
    full_dataset = seed_dataset.copy()

    for problem_id, problem in enumerate(target_problems):
        if len(full_dataset) >= 100000:
            break

        # Step 1: Retrieve similar seed instances by embedding similarity
        similar_seeds = retrieve_similar_seeds(problem, seed_dataset, k=3)

        # Step 2: Extract reasoning flows from similar problems
        reference_flows = [seed['reasoning_flow'] for seed in similar_seeds]
        reference_budgets = [seed['thought_budget'] for seed in similar_seeds]

        # Step 3: Prompt GPT-4o to generate reasoning for new problem
        synthesis_prompt = f"""
        Problem: {problem}

        Similar problem reasoning flows (for guidance only):
        {chr(10).join(reference_flows)}

        Recommended thought budget (tokens): {max(reference_budgets)}

        Generate step-by-step reasoning for this problem.
        Maintain similar structure to reference flows.
        Stay within thought budget.
        """

        gpt4o_reasoning = call_gpt4o(synthesis_prompt, max_tokens=max(reference_budgets))

        # Step 4: Extract solution from reasoning
        solution = extract_solution(gpt4o_reasoning)

        # Step 5: Filter by correctness
        if verify_solution(solution, problem):
            full_dataset.append({
                'id': f'gen_{problem_id}',
                'problem': problem,
                'reasoning': gpt4o_reasoning,
                'solution': solution,
                'reasoning_flow': extract_reasoning_flow_simple(gpt4o_reasoning),
                'thought_budget': len(gpt4o_reasoning.split()),
                'source': 'gpt4o_guided'
            })

    return full_dataset
```

3. **Quality Filtering**: Remove low-quality or incorrect solutions

```python
def filter_generated_dataset(full_dataset, correctness_threshold=0.76):
    """
    Filter dataset to keep only high-quality, correct solutions.
    Target 76% accuracy after filtering (validated on validation set).
    """
    filtered = []

    for instance in full_dataset:
        # Verify solution correctness
        is_correct = verify_solution(instance['solution'], instance['problem'])

        # Check reasoning quality metrics
        reasoning_quality = evaluate_reasoning_quality(instance['reasoning'])

        # Accept if correct and quality passes threshold
        if is_correct:
            filtered.append(instance)
        elif reasoning_quality > 0.8 and reasoning_quality > correctness_threshold:
            # Keep near-correct with high quality reasoning (borderline cases)
            filtered.append(instance)

    # Calculate accuracy statistics
    accuracy = sum(1 for x in filtered if verify_solution(x['solution'], x['problem'])) / len(filtered)

    print(f"Filtered dataset size: {len(filtered)}")
    print(f"Accuracy: {accuracy:.1%}")

    return filtered

def evaluate_reasoning_quality(reasoning):
    """
    Heuristic evaluation of reasoning quality.
    Checks for logical flow, completeness, clarity.
    """
    quality_score = 0.0

    # Length heuristic: too short or too long is worse
    token_count = len(reasoning.split())
    if 100 < token_count < 1000:
        quality_score += 0.3

    # Logical markers: presence of reasoning keywords
    reasoning_keywords = ['therefore', 'because', 'thus', 'step', 'first', 'next', 'finally']
    keyword_count = sum(1 for kw in reasoning_keywords if kw in reasoning.lower())
    quality_score += min(0.3, keyword_count * 0.05)

    # Math notation: equations and symbols in math problems
    if '$' in reasoning or '=' in reasoning:
        quality_score += 0.2

    # No placeholder tokens
    if '<unk>' not in reasoning and '[...]' not in reasoning:
        quality_score += 0.2

    return quality_score
```

4. **Thought Budget Control**: Manage reasoning token allocation

```python
def apply_thought_budget_control(problem, reasoning_type='balanced'):
    """
    Control maximum tokens for reasoning based on problem characteristics.
    Prevents overthinking and excessive token generation.
    """
    base_budget = {
        'simple': 200,      # For straightforward problems
        'balanced': 500,    # For typical problems
        'complex': 1000,    # For challenging problems
        'competition': 2000 # For olympiad-level problems
    }

    if reasoning_type not in base_budget:
        reasoning_type = 'balanced'

    max_tokens = base_budget[reasoning_type]

    # Adjust by problem difficulty heuristic
    if 'difficult' in problem.lower() or 'olympiad' in problem.lower():
        max_tokens = min(2000, max_tokens * 2)

    return max_tokens

def generate_with_budget(problem, model, thought_budget):
    """Generate reasoning respecting thought budget."""
    reasoning = model.generate(
        problem,
        max_tokens=thought_budget,
        temperature=0.7,
        top_p=0.95
    )

    return reasoning
```

5. **RL Fine-tuning Initialization**: Use dataset for GRPO warm-start

```python
def init_rl_with_long_cot_dataset(base_model, dataset, num_steps=100):
    """
    Pre-train model on Long CoT Collection before RL fine-tuning.
    Provides strong warm-start for GRPO optimization.
    """
    # Supervised fine-tuning on full dataset
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5)

    for step in range(num_steps):
        batch = sample_batch(dataset, batch_size=32)

        for example in batch:
            # Forward pass
            output = base_model(example['problem'], max_length=2048)

            # Supervised loss
            loss = torch.nn.functional.cross_entropy(
                output.logits.view(-1, output.logits.shape[-1]),
                example['reasoning_tokens'].view(-1)
            )

            # Backward
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return base_model

def compare_rl_with_without_warmstart(base_model_no_warmstart, model_with_warmstart,
                                       rl_steps=1000):
    """
    Compare RL performance: cold-start vs. Long CoT warm-start.
    Expect 2-3× performance gain with initialization.
    """
    results = {}

    # RL without warm-start (cold-start)
    cold_start_model = train_with_rl(base_model_no_warmstart, steps=rl_steps)
    results['cold_start'] = eval_model(cold_start_model)

    # RL with Long CoT warm-start
    warm_start_model = train_with_rl(model_with_warmstart, steps=rl_steps)
    results['warm_start'] = eval_model(warm_start_model)

    # Calculate improvement
    improvement = results['warm_start']['accuracy'] / results['cold_start']['accuracy']
    print(f"Warm-start improvement: {improvement:.2f}×")

    return results
```

## Practical Guidance

**When to Apply:**
- Developing open-source reasoning models without access to o1
- Limited budget for data annotation
- Need strong baseline performance before RL fine-tuning
- Want to reduce cold-start problem in reasoning model training

**Dataset Acquisition:**
1. Collect 1K seed samples manually from o1 (or use provided magpie-reasoning seed)
2. Extract reasoning flows and thought budgets from seeds
3. Use provided scripts to synthesize 100K samples via GPT-4o
4. Filter for correctness (expect ~76% accuracy after filtering)

**Training with Long CoT Collection:**
- Pre-train 50-100 steps on full dataset (SFT phase)
- Then apply RL fine-tuning (GRPO) for 1-2K steps
- Expect 2-3× performance gains vs. cold-start RL alone

**Performance Expectations:**
- Without warm-start (RL only): 22.7% on GPQA Diamond baseline
- With Long CoT warm-start: 36.4%+ on GPQA Diamond
- Generalization: Works on MATH, AIME, MMLU-Pro, multimodal tasks
- RL convergence: Faster (2-3× improvement per RL step)

**Thought Budget Guidelines:**
- Simple problems: 100-200 tokens
- Standard problems: 300-600 tokens
- Complex problems: 800-1500 tokens
- Olympiad-level: 1500-2500 tokens

**Configuration for Different Model Sizes:**
- 7B models: 100K dataset sufficient, 50 SFT steps
- 14B models: 100K dataset, 100 SFT steps
- 70B+ models: Consider augmenting to 200K samples, 200 SFT steps

**Common Issues:**
- Low correctness after filtering: Check synthesis prompts, may need better reference selection
- Overthinking (excessive tokens): Reduce thought budget, apply earlier filtering
- Poor RL convergence: Ensure SFT warm-start covers diverse problem types
- Domain mismatch: Ensure seed dataset represents target domain well

## Reference

100K Long CoT Collection available for community use. Base implementation on Llama-3.1-8B-Instruct and Qwen-2.5-7B-Instruct. Evaluated on MATH-500, AIME24, GPQA Diamond, MMLU-Pro benchmarks with RL training on 4-16 A100 GPUs using GRPO algorithm. Demonstrates strong generalization beyond seed domains and 2-3× RL performance improvement.
