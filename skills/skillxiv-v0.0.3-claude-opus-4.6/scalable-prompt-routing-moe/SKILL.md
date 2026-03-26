---
name: scalable-prompt-routing-moe
title: "Scalable Prompt Routing for Frontier LLMs"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.19415"
keywords: [Prompt Routing, Mixture of Experts, Graph Clustering, Cost Reduction, Multi-Model Inference]
description: "Route queries across frontier models using two-stage system: graph-based task discovery identifies ~332 latent task types via semantic similarity + preference patterns; MoE with task-specific adapters estimates quality for candidate models. Achieves <50% inference cost of strongest single model while exceeding its performance; applies when managing pools of frontier models with narrow capability gaps."
---

## Component ID
Two-stage LLM prompt routing system combining task discovery and quality estimation.

## Motivation
Frontier model pools show narrow performance gaps where subtle capability differences determine task suitability. Routing queries to diverse specialized models can achieve better cost-performance than over-relying on a single strongest model. However, discovering latent task structure and estimating per-model quality requires principled methods beyond heuristics.

## The Modification

### Stage 1: Graph-Based Task Discovery
Discover latent task types by combining semantic similarity of task descriptions with model preference patterns using Rank Biased Overlap scoring.

```python
# Task discovery via graph clustering on semantic space
def discover_task_types(task_descriptions, model_responses, n_tasks=332):
    """
    Identify latent task types by clustering tasks and models simultaneously.
    Uses semantic embeddings of task descriptions and preference agreement patterns.
    """
    # Embed task descriptions
    task_embeddings = encode_descriptions(task_descriptions)

    # Compute semantic similarity between tasks
    semantic_similarity = cosine_similarity(task_embeddings)

    # Compute preference agreement: which models agree on relative rankings
    model_preferences = []
    for model in available_models:
        rankings = rank_responses(model_responses[model])
        model_preferences.append(rankings)

    # Fuse semantic + preference signals via graph clustering
    combined_graph = semantic_similarity * agreement_matrix(model_preferences)
    task_clusters = spectral_clustering(combined_graph, n_clusters=n_tasks)

    return task_clusters
```

### Stage 2: Task-Specific Quality Estimation
A mixture-of-experts architecture with task-specific prediction heads provides specialized quality estimates alongside general adapters.

```python
# MoE quality estimation with task-specific routing
class TaskAwareQualityMoE:
    """
    Predicts per-model quality for a given task using task-specific adapters.
    Reduces model consideration to ~32% per task (from full 11 models).
    """
    def __init__(self, n_tasks=332, n_models=11):
        self.task_classifier = TaskClassifier(n_tasks)
        self.general_adapters = nn.ModuleList([
            Adapter() for _ in range(n_models)
        ])
        self.task_specific_heads = nn.ModuleList([
            nn.ModuleList([Adapter() for _ in range(n_models)])
            for _ in range(n_tasks)
        ])

    def estimate_quality(self, query_embedding):
        """
        Single forward pass: classify task, activate task-specific heads.
        Maintains constant effective model size via adaptive activation.
        """
        task_id = self.task_classifier(query_embedding)

        # General quality scores across all models
        general_scores = [adapter(query_embedding) for adapter in self.general_adapters]

        # Task-specific refinements
        task_specific_scores = [
            self.task_specific_heads[task_id][i](query_embedding)
            for i in range(len(self.general_adapters))
        ]

        # Combine: task-specific provides calibration for task
        final_scores = general_scores + task_specific_scores
        return final_scores
```

## Ablation Results

**Cost-Performance Tradeoff**:
- **FineRouter performance**: Exceeds Claude-Sonnet-4.5 baseline on 10 benchmarks
- **Cost**: <50% of single strongest model inference cost
- **Routing distribution** (example): Claude-Sonnet-4.5 (28%), DeepSeek-R1 (27%), Llama-4-Maverick (23%), Qwen3-235B (13%)

**Task Discovery**:
- Discovers ~332 fine-grained task types from training data
- Reduces model consideration pool: ~32% of models per task (from full pool of 11)
- Graph clustering identifies meaningful task boundaries beyond semantic similarity alone

**Efficiency**:
- Single forward pass through task classifier; adaptive head activation
- Constant effective model size maintained across routing decisions
- No per-model batching overhead

## Conditions
- **Model pool**: Works with frontier models showing narrow capability gaps (e.g., Claude-Sonnet, DeepSeek-R1, Llama-4, Qwen3)
- **Task discovery requirements**: Access to responses from all candidate models on training data for preference agreement estimation
- **Task pool size**: ~300–400 latent tasks typical; scales with training data diversity
- **Training data**: Balanced coverage of task types to avoid underrepresented clusters
- **Inference constraints**: Single forward pass required; compatible with streaming/batched deployment

## Drop-In Checklist
- [ ] Collect responses from all candidate models on diverse training queries
- [ ] Run graph clustering on task embeddings + preference agreement matrix
- [ ] Build task classifier (encoder + classifier head)
- [ ] Create MoE with general adapters for each model + task-specific heads
- [ ] Validate discovered task types: inspect clusters for semantic coherence
- [ ] Benchmark cost vs. single-model baseline—target <50% relative cost
- [ ] Measure performance vs. single strongest model—target >= performance
- [ ] Profile latency: confirm single forward pass enables streaming
- [ ] Calibrate routing distribution across your model pool
