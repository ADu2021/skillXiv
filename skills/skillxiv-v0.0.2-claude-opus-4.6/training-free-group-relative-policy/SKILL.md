---
name: training-free-group-relative-policy
title: "Training-Free Group Relative Policy Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.08191"
keywords: [Policy Optimization, In-Context Learning, Experience Library, Inference-Only, Cost-Efficient]
description: "Improve LLM performance at test time through in-context learning and experience libraries, eliminating the need for parameter updates while maintaining competitive results."
---

# Technique: Inference-Only Policy Optimization via Experience Libraries

Traditional RL fine-tuning requires parameter updates, which is costly and requires model deployment control. Training-Free GRPO achieves comparable results without updating a single parameter. Instead, it maintains an external experience library that evolves during inference through natural language lessons extracted from successful and failed outputs.

The key insight is that policy improvement doesn't require gradient updates. By analyzing why certain outputs succeeded or failed and injecting those lessons into prompts, frozen LLMs can dynamically adapt to task distributions without retraining.

## Core Concept

Training-Free GRPO operates through five steps:

1. **Multi-Output Generation**: Generate multiple completions from frozen LLM
2. **Scoring**: Evaluate each with reward model
3. **Semantic Advantage Extraction**: LLM analyzes why certain outputs were better
4. **Experience Library Update**: Store lessons (add/delete/modify) in natural language
5. **Conditional Inference**: Inject updated experiences into prompts for subsequent queries

## Architecture Overview

- **Frozen LLM**: Policy remains unchanged (no gradient updates)
- **External Experience Library**: Evolving knowledge base of learned lessons
- **Reward Model**: Scores outputs for advantage computation
- **Semantic Analyzer**: Extracts language-level insights from comparisons
- **Dynamic Prompting**: Conditions future outputs on experience library

## Implementation Steps

Create the experience library structure.

```python
class ExperienceLibrary:
    def __init__(self):
        self.experiences = []  # List of experience entries

    class Experience:
        def __init__(self, content, operation='add', domain='general'):
            self.content = content  # Natural language lesson
            self.operation = operation  # 'add', 'modify', 'delete'
            self.domain = domain
            self.utility = 0.0  # Track usefulness
            self.usage_count = 0

        def __repr__(self):
            return f"{self.operation.upper()}: {self.content}"

    def add_experience(self, lesson, operation='add', domain='general'):
        """Add a new experience to the library."""
        exp = self.Experience(lesson, operation, domain)
        self.experiences.append(exp)

    def get_experiences_for_domain(self, domain, top_k=5):
        """Retrieve top experiences for a given domain."""
        domain_exps = [e for e in self.experiences if e.domain == domain]
        # Sort by utility
        domain_exps.sort(key=lambda e: e.utility, reverse=True)
        return domain_exps[:top_k]

    def update_utility(self, experience, reward_delta):
        """Update experience utility based on outcome."""
        experience.utility += reward_delta
        experience.usage_count += 1
```

Implement multi-output generation with scoring.

```python
def generate_and_score_outputs(model, query, reward_model, num_outputs=5):
    """
    Generate multiple outputs and score them with reward model.

    Args:
        model: Frozen language model
        query: Input query
        reward_model: Scoring function
        num_outputs: Number of outputs to generate

    Returns:
        outputs: List of (text, score) tuples
    """

    outputs = []

    for _ in range(num_outputs):
        # Generate output
        text = model.generate(query, max_length=512, temperature=0.9)

        # Score with reward model
        score = reward_model.score(query, text)

        outputs.append((text, score))

    # Sort by score
    outputs.sort(key=lambda x: x[1], reverse=True)

    return outputs
```

Implement semantic advantage extraction using LLM analysis.

```python
def extract_semantic_advantages(model, query, outputs, top_k=3):
    """
    Use LLM to analyze why top outputs succeeded.

    Args:
        model: Language model
        query: Original query
        outputs: List of (text, score) tuples
        top_k: Top outputs to analyze

    Returns:
        lessons: List of natural language lessons
    """

    top_outputs = outputs[:top_k]

    analysis_prompt = f"""Query: {query}

Successful outputs (ranked by quality):
"""

    for i, (text, score) in enumerate(top_outputs):
        analysis_prompt += f"{i+1}. (Score: {score:.2f}) {text}\n"

    analysis_prompt += """
What made these outputs effective? Extract general principles that could help with similar queries.
Provide 2-3 concrete lessons in the format:
LESSON: [specific advice]
"""

    lessons_text = model.generate(analysis_prompt)

    # Parse lessons
    lessons = []
    for line in lessons_text.split('\n'):
        if 'LESSON:' in line:
            lesson = line.split('LESSON:')[1].strip()
            if lesson:
                lessons.append(lesson)

    return lessons
```

Implement experience library updates based on outcomes.

```python
def update_experience_library(library, lessons, query_domain, outcomes):
    """
    Update experience library with new lessons.

    Args:
        library: ExperienceLibrary instance
        lessons: List of natural language lessons
        query_domain: Domain of current query
        outcomes: (best_score, average_score) from reward model
    """

    for lesson in lessons:
        # Add new experience
        library.add_experience(lesson, operation='add', domain=query_domain)

        # Later: could modify existing experiences if similar
        # Could delete experiences that contradict new lessons

    # Update utility of all experiences based on outcomes
    best_score, avg_score = outcomes
    reward_delta = best_score - avg_score

    for exp in library.experiences:
        if exp.domain == query_domain:
            library.update_utility(exp, reward_delta)
```

Implement dynamic prompting that conditions on experience library.

```python
def generate_with_experience_context(model, query, experience_library, domain):
    """
    Generate output with experience library providing context.

    Args:
        model: Frozen language model
        query: Input query
        experience_library: ExperienceLibrary instance
        domain: Query domain for experience retrieval

    Returns:
        output: Generated text informed by experiences
    """

    # Retrieve relevant experiences
    top_experiences = experience_library.get_experiences_for_domain(domain, top_k=5)

    # Format experiences as context
    experience_context = "Learned principles:\n"
    for exp in top_experiences:
        experience_context += f"- {exp.content}\n"

    # Augment prompt with experience context
    augmented_prompt = f"""{experience_context}

Query: {query}

Generate a response that applies the above principles:"""

    # Generate with frozen model
    output = model.generate(augmented_prompt, max_length=512)

    return output
```

Implement the full Training-Free GRPO loop.

```python
def training_free_grpo_loop(model, reward_model, experience_library,
                           queries, num_iterations=5):
    """
    Main loop for inference-only policy optimization.

    Args:
        model: Frozen language model
        reward_model: Scoring function
        experience_library: ExperienceLibrary instance
        queries: List of evaluation queries
        num_iterations: Number of iteration rounds

    Returns:
        results: Performance metrics
    """

    results = {
        'iteration': [],
        'avg_score': [],
        'best_score': [],
        'library_size': []
    }

    for iteration in range(num_iterations):
        iteration_scores = []

        for query in queries:
            # Determine query domain (simplified)
            domain = extract_domain(query)

            # Generate outputs with experience context
            outputs = []
            for _ in range(5):
                output = generate_with_experience_context(
                    model, query, experience_library, domain
                )
                score = reward_model.score(query, output)
                outputs.append((output, score))

            # Extract lessons from successful outputs
            lessons = extract_semantic_advantages(model, query, outputs, top_k=3)

            # Update experience library
            best_score = max(outputs, key=lambda x: x[1])[1]
            avg_score = sum(x[1] for x in outputs) / len(outputs)
            update_experience_library(
                experience_library, lessons, domain,
                (best_score, avg_score)
            )

            iteration_scores.append(best_score)

        # Track metrics
        avg_iter_score = sum(iteration_scores) / len(iteration_scores)
        results['iteration'].append(iteration)
        results['avg_score'].append(avg_iter_score)
        results['best_score'].append(max(iteration_scores))
        results['library_size'].append(len(experience_library.experiences))

        print(f"Iteration {iteration+1}: Avg score={avg_iter_score:.3f}, "
              f"Library size={len(experience_library.experiences)}")

    return results


def extract_domain(query):
    """Extract or infer domain from query."""
    # Simplified: could use more sophisticated classification
    if 'code' in query.lower() or 'python' in query.lower():
        return 'coding'
    elif 'math' in query.lower():
        return 'math'
    else:
        return 'general'
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Experience library size | 50-500 entries | More experiences improve coverage; manage memory |
| Domain count | 3-10 categories | Balance specificity with generalization |
| Experience refresh | Remove bottom-10% by utility | Prevent library staleness |
| Prompt augmentation | Keep at 500-1000 tokens | Balance context size with latency |
| When to use | Evaluation/production deployment | Cost-sensitive inference scenarios |
| When NOT to use | Training where fine-tuning is available | Fine-tuning often more sample-efficient |
| Common pitfall | Experience library becomes incoherent | Regular quality checks on stored lessons |

### When to Use Training-Free GRPO

- Deployment scenarios where fine-tuning is impractical
- Cost-sensitive inference (API-based LLMs)
- Multi-task evaluation where quick adaptation is needed
- Scenarios without model update permissions

### When NOT to Use Training-Free GRPO

- Tasks where fine-tuning budget permits
- Single-task optimization where in-context learning overhead is high
- Real-time systems where latency is critical

### Common Pitfalls

- **Experience incoherence**: Conflicting lessons in library; enforce consistency checks
- **Utility estimation**: Reward model quality directly impacts experience utility; validate reward model
- **Domain mismatch**: Experience retrieval relies on good domain identification; improve classification
- **Prompt bloat**: Experience context can grow; use ablation to prune low-utility experiences

## Reference

Paper: https://arxiv.org/abs/2510.08191
