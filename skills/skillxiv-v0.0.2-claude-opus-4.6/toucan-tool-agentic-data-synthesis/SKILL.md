---
name: toucan-tool-agentic-data-synthesis
title: "TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.01179"
keywords: [synthetic-data, agent-training, tool-use, MCP, data-generation]
description: "Synthesize large-scale training data for LLM agents by generating diverse tool-use trajectories from real Model Context Protocol (MCP) environments. A 1.5M-example dataset created via multi-stage pipeline: query generation, quality filtering, trajectory creation with real tool execution, validation, and diversification."
---

# TOUCAN: Synthesizing Tool-Agentic Training Data at Scale

The scarcity of high-quality, permissively licensed tool-use training data limits agent research. Existing datasets are small, domain-specific, or closed-source. TOUCAN addresses this by automatically generating 1.5 million trajectories from nearly 500 real-world Model Context Protocol (MCP) environments, creating a diverse, realistic foundation for agent training.

The challenge is that synthetic data generation for agents requires more than prompting LLMs to invent examples. Agents interact with real tools where failures matter. TOUCAN uses actual tool execution to ensure generated trajectories are grounded in reality, not fantasies.

## Core Concept

TOUCAN's pipeline generates tool-use trajectories through six stages:

1. **Query Generation**: Five different models produce diverse tool-use queries (what users ask agents to do)
2. **Quality Filtering**: Model-based filtering removes low-quality, ambiguous, or infeasible queries
3. **Trajectory Creation**: Three teacher models generate agent trajectories using two agentic frameworks with real tool execution
4. **Validation**: Rule-based and model-based validation ensures outputs are correct and follow expected formats
5. **Diversification**: Three extension mechanisms expand dataset variety (multi-turn conversations, paraphrasing, task variants)
6. **Curation**: Manual sampling and review to maintain quality across 500 MCPs

## Architecture Overview

- **Query generator**: Prompt ensemble creating diverse tool-use scenarios
- **Filter module**: Removes infeasible or low-quality queries via model classification
- **Tool executor**: Real MCP runtime executing trajectories (not simulation)
- **Teacher ensemble**: Multiple models generating alternative solution paths
- **Validator**: Step-level and trajectory-level correctness checks
- **Extender**: Multi-turn dialog simulator, paraphraser, task variant generator

## Implementation Steps

Start by generating diverse queries that exercise different tools in an MCP environment:

```python
from toucan import QueryGenerator, MCP_Registry

# Initialize query generation with model ensemble
generators = [
    "gpt-4o",
    "claude-3-sonnet",
    "meta-llama/llama-2-70b",
    "deepseek-coder-33b-instruct",
    "mistral-large",
]

def generate_diverse_queries(mcp_list, num_queries=1000):
    """
    Generate tool-use queries from MCP environment list.

    Args:
        mcp_list: List of available MCPs (e.g., filesystem, web_search, code_exec)
        num_queries: Target number of queries

    Returns:
        queries: List of (query_text, relevant_tools) tuples
    """
    queries = []

    # Use different generators for diversity
    for i, generator_name in enumerate(generators):
        prompt = f"""Given these available tools: {', '.join(mcp_list)}

Generate a realistic user query that requires using one or more of these tools.
The query should be:
- Natural and conversational
- Require at least one tool call
- Completable (not asking for something impossible)
- Diverse from previous queries

Return ONLY the query text, no explanation."""

        num_per_generator = num_queries // len(generators)
        generator_queries = [
            llm_call(generator_name, prompt)
            for _ in range(num_per_generator)
        ]
        queries.extend(generator_queries)

    return queries
```

Next, filter out infeasible or low-quality queries using a trained classifier:

```python
from toucan import QualityFilter

def filter_queries(queries, quality_threshold=0.7):
    """
    Remove low-quality or infeasible queries.

    Args:
        queries: Raw generated queries
        quality_threshold: Minimum quality score (0-1)

    Returns:
        filtered_queries: Queries meeting quality threshold
        quality_scores: Confidence scores for each query
    """
    filter_model = QualityFilter.load_pretrained("toucan-v1.0")

    filtered = []
    scores = []

    for query in queries:
        # Score query on multiple dimensions
        features = {
            "clarity": clarity_score(query),          # Is query understandable?
            "feasibility": feasibility_score(query),  # Can tools accomplish this?
            "complexity": complexity_score(query),    # Not too simple, not too hard
            "diversity": diversity_from_cache(query)  # Different from prior queries
        }

        quality = filter_model.predict(features)
        scores.append(quality)

        if quality >= quality_threshold:
            filtered.append(query)

    return filtered, scores
```

Now execute trajectories using real MCPs and multiple teacher models:

```python
from toucan import TrajectoryGenerator, MCPExecutor

def generate_trajectories(query, mcp_environment, num_teachers=3):
    """
    Generate agent trajectories with real tool execution.

    Args:
        query: User request
        mcp_environment: MCP runtime with available tools
        num_teachers: Number of different agent models to try

    Returns:
        trajectories: List of (action_sequence, reward) tuples
    """
    trajectories = []
    teacher_models = [
        "gpt-4o",
        "claude-3.5-sonnet",
        "meta-llama/llama-3-70b-instruct",
    ]

    for teacher in teacher_models:
        # Create agent with this teacher model
        agent = Agent(model_name=teacher, tools=mcp_environment)

        # Execute trajectory with real tool calls
        trajectory = []
        done = False
        step = 0
        max_steps = 10

        while not done and step < max_steps:
            # Get next action from teacher
            action = agent.decide(query, trajectory)
            trajectory.append({"action": action, "step": step})

            # Execute in real MCP environment
            result = mcp_environment.execute(action)
            trajectory[-1]["result"] = result

            # Check for completion
            done = is_task_complete(query, trajectory)
            step += 1

        trajectories.append({
            "query": query,
            "actions": trajectory,
            "success": done,
            "teacher": teacher
        })

    return trajectories
```

Validate trajectories to ensure correctness before including in dataset:

```python
def validate_trajectories(trajectories, query):
    """
    Multi-level validation: format, correctness, safety.

    Args:
        trajectories: Generated trajectories
        query: Original user query

    Returns:
        valid_trajectories: Filtered trajectories meeting all checks
    """
    valid = []

    for traj in trajectories:
        # Rule-based checks
        if not has_valid_format(traj):
            continue  # Malformed trajectory

        if not matches_query_intent(traj, query):
            continue  # Doesn't solve the user's problem

        if has_unsafe_actions(traj):
            continue  # Contains dangerous operations

        # Model-based correctness check
        correctness_score = correctness_model.score(traj, query)
        if correctness_score > 0.8:
            valid.append(traj)

    return valid
```

Finally, extend dataset with diversification to prevent overfitting:

```python
def diversify_trajectories(trajectory, query):
    """
    Expand single trajectory into multiple variants.

    Args:
        trajectory: Original successful trajectory
        query: Original query

    Returns:
        variants: List of (query, trajectory) pairs
    """
    variants = []

    # 1. Multi-turn variant: break into conversation
    multiturn = convert_to_dialogue(trajectory, query)
    variants.append(multiturn)

    # 2. Paraphrase: rewrite query in different words
    paraphrased_query = paraphrase(query)
    variants.append((paraphrased_query, trajectory))

    # 3. Task variant: modify query slightly (same intent, different specifics)
    variant_query = create_task_variant(query)
    # Re-execute trajectory with variant query to validate
    if trajectories_compatible(trajectory, variant_query):
        variants.append((variant_query, trajectory))

    return variants
```

## Practical Guidance

**When to use TOUCAN:**
- Training agents from scratch with limited real data
- Scaling agent training without human labeling
- Building agents that work across diverse tools/MCPs
- Creating benchmarks for agent evaluation
- Bootstrapping agent training before fine-tuning on real data

**When NOT to use:**
- Tasks requiring domain expertise (medical, legal advice)
- High-precision settings where hallucinations are costly
- Proprietary tool environments (TOUCAN needs tool access)
- Single-task agents (full-dataset synthesis is overkill)

**Dataset composition:**

| Component | Count | Details |
|-----------|-------|---------|
| Unique queries | ~500K | From 5-model ensemble |
| Passed quality filter | ~300K | >0.7 quality score |
| Trajectories (multi-teacher) | ~1.5M | 3-5 trajectories per query |
| Tools/MCPs covered | 500 | Real-world MCP implementations |
| Validation accuracy | >95% | Rule + model-based checks |

**Quality metrics to track:**

- **Diversity**: Unique tool combinations, query varieties, solution approaches
- **Success rate**: % of trajectories that achieve task goal
- **Tool coverage**: How many unique tools are used across dataset
- **Length distribution**: Avoid biasing toward short or long trajectories

**Common pitfalls:**
- **Teacher overfitting**: Trajectories from same model tend to be similar. Use diverse teacher ensemble (different orgs, sizes).
- **Distribution shift**: MCPs may not represent production use. Validate on real-world queries; add reweighting if needed.
- **Validation errors**: Weak validators pass bad trajectories. Maintain ground-truth validation set; monitor false-positive rate.
- **Tool hallucination**: Some teachers invent tools that don't exist. Validate all tools exist before execution.

**Integration checklist:**
- [ ] Inventory available MCPs/tools (aim for 20+ for diversity)
- [ ] Curate seed queries manually (50-100 examples) to guide generation
- [ ] Test quality filter on seed queries (target >90% keep rate)
- [ ] Sample 100 trajectories, have human review (target >95% quality)
- [ ] Monitor tool invocation success rate (target >90%)
- [ ] Create validation set (5% of final dataset) for agent evaluation

Reference: https://arxiv.org/abs/2510.01179
