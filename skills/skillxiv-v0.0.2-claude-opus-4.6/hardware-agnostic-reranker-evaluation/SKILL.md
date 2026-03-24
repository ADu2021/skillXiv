---
name: hardware-agnostic-reranker-evaluation
title: "Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06223"
keywords: [Information Retrieval, LLM Reranking, Efficiency Metrics, FLOPs, Hardware-Agnostic]
description: "Evaluate LLM-based document rerankers using hardware-agnostic FLOPs metrics instead of latency, enabling fair comparison of ranking quality per unit of computation across different models and deployment scenarios."
---

# Hardware-Agnostic Reranker Evaluation: FLOPs-Based Efficiency Metrics

LLM-based document rerankers improve retrieval quality but add significant computational cost. Traditional efficiency metrics—latency, throughput, token counts—depend on hardware, runtime choices (batching, quantization, serving framework), and deployment specifics. This makes fair comparison impossible: the same model shows different efficiency profiles on different hardware. Hardware-agnostic metrics are needed to evaluate the intrinsic computational cost of reranking approaches.

This work proposes Ranking metrics Per PetaFLOP (RPP) and Queries Per PetaFLOP (QPP)—efficiency-effectiveness metrics based on floating-point operations rather than hardware-dependent measurements. These metrics enable reproducible comparison across models, architectures, and deployment scenarios, revealing that pointwise reranking methods dominate efficiency rankings while larger models sacrifice efficiency disproportionately.

## Core Concept

FLOPs (floating-point operations) quantify the intrinsic computational work required by a model regardless of hardware. Unlike latency—which varies across GPUs, TPUs, batch sizes, and optimization strategies—FLOPs directly measure algorithmic work. A closed-form estimator computes FLOPs for different LLM architectures (decoder-only, encoder-decoder, grouped-query attention, mixture-of-experts), enabling reproducible efficiency comparisons.

The key insight is that ranking quality metrics (MRR, NDCG) should be reported per unit of computation (FLOPs) rather than in isolation. This separates model quality from deployment specifics and reveals true efficiency gains from architectural choices.

## Architecture Overview

- **FLOPs Estimator**: Closed-form formulas computing floating-point operations for various LLM architectures covering forward and backward passes
- **Decoder-Only Reranker Support**: Pointwise and listwise reranking with GPT-style models, handling attention patterns and FFN operations
- **Encoder-Decoder Reranker Support**: Sequence-to-sequence models optimized for ranking tasks, with separate FLOPs accounting for encoder and decoder
- **Advanced Architecture Support**: Grouped-query attention (GQA), mixture-of-experts (MoE) modules, and other efficiency techniques with corresponding FLOPs adjustments
- **Ranking Quality Metrics**: MRR (Mean Reciprocal Rank), NDCG (Normalized Discounted Cumulative Gain), Hits@K computed per PetaFLOP
- **Comparative Analysis Framework**: Benchmarks across reranking strategies, model sizes, and architectural choices

## Implementation

The following implements hardware-agnostic FLOPs estimation and efficiency-effectiveness metrics for reranker evaluation.

**Step 1: FLOPs Estimation for Transformer Architectures**

This computes floating-point operations for various LLM architectures.

```python
import math
from typing import Dict, Tuple

class TransformerFLOPsEstimator:
    """Compute FLOPs for Transformer-based LLM rerankers."""

    def __init__(self, model_config: Dict):
        self.vocab_size = model_config.get("vocab_size", 50000)
        self.hidden_dim = model_config.get("hidden_dim", 768)
        self.num_layers = model_config.get("num_layers", 12)
        self.num_heads = model_config.get("num_heads", 12)
        self.ffn_dim = model_config.get("ffn_dim", 3072)
        self.use_gqa = model_config.get("grouped_query_attention", False)
        self.num_kv_heads = model_config.get("num_kv_heads", 1) if self.use_gqa else self.num_heads

    def estimate_single_pass(self, seq_len: int, batch_size: int = 1) -> float:
        """
        Estimate FLOPs for single forward pass through model.
        Args:
            seq_len: sequence length (input + output tokens)
            batch_size: batch size
        Returns:
            estimated FLOPs
        """
        total_flops = 0

        # Embedding projection: (batch * seq_len * hidden_dim) multiplications
        embedding_flops = batch_size * seq_len * self.vocab_size * self.hidden_dim
        total_flops += embedding_flops

        # Per-layer operations
        for _ in range(self.num_layers):
            # Self-attention
            # Q projection: batch * seq_len * hidden_dim -> batch * seq_len * hidden_dim
            q_proj_flops = batch_size * seq_len * self.hidden_dim * self.hidden_dim
            total_flops += 2 * q_proj_flops  # forward + backward

            # K, V projections
            kv_proj_flops = batch_size * seq_len * self.hidden_dim * self.hidden_dim
            total_flops += 2 * 2 * kv_proj_flops  # K and V, forward + backward

            # Attention scores: Q @ K.T
            # Shape: (batch, num_heads, seq_len, hidden_dim/num_heads)
            head_dim = self.hidden_dim // self.num_heads
            attention_flops = batch_size * self.num_heads * seq_len * seq_len * head_dim
            total_flops += 2 * attention_flops  # forward + backward

            # Attention output: attention @ V
            attn_output_flops = batch_size * self.num_heads * seq_len * seq_len * head_dim
            total_flops += 2 * attn_output_flops

            # Output projection
            output_proj_flops = batch_size * seq_len * self.hidden_dim * self.hidden_dim
            total_flops += 2 * output_proj_flops

            # FFN: first layer (hidden_dim -> ffn_dim)
            ffn1_flops = batch_size * seq_len * self.hidden_dim * self.ffn_dim
            total_flops += 2 * ffn1_flops

            # FFN: second layer (ffn_dim -> hidden_dim)
            ffn2_flops = batch_size * seq_len * self.ffn_dim * self.hidden_dim
            total_flops += 2 * ffn2_flops

        # Output projection to vocab
        output_vocab_flops = batch_size * seq_len * self.hidden_dim * self.vocab_size
        total_flops += 2 * output_vocab_flops

        return total_flops

    def estimate_reranking_pass(
        self,
        query_len: int,
        document_len: int,
        num_documents: int,
        batch_size: int = 1,
        reranking_mode: str = "pointwise"
    ) -> float:
        """
        Estimate FLOPs for reranking operation.
        Args:
            query_len: query token length
            document_len: document token length
            num_documents: number of documents to rank
            batch_size: batch size
            reranking_mode: "pointwise", "listwise", or "pairwise"
        Returns:
            estimated FLOPs
        """
        if reranking_mode == "pointwise":
            # Process each query-document pair independently
            total_seq_len = query_len + document_len
            total_pairs = num_documents
            total_flops = 0

            for _ in range(total_pairs):
                total_flops += self.estimate_single_pass(total_seq_len, batch_size)

            return total_flops

        elif reranking_mode == "listwise":
            # Process all documents together: [query] [doc1] [doc2] ... [docN]
            total_seq_len = query_len + (document_len * num_documents)
            return self.estimate_single_pass(total_seq_len, batch_size)

        elif reranking_mode == "pairwise":
            # Process pairs: [query] [doc1] vs [query] [doc2]
            total_seq_len = query_len + document_len
            num_pairs = (num_documents * (num_documents - 1)) // 2
            return num_pairs * self.estimate_single_pass(total_seq_len, batch_size)

        else:
            raise ValueError(f"Unknown reranking mode: {reranking_mode}")

    def estimate_with_gqa(self, seq_len: int, batch_size: int = 1) -> float:
        """
        Estimate FLOPs with grouped-query attention (more efficient than MHA).
        Reduces KV computation by factor of (num_heads / num_kv_heads).
        """
        gqa_factor = self.num_heads / self.num_kv_heads
        standard_flops = self.estimate_single_pass(seq_len, batch_size)
        # GQA reduces K/V projection and attention computation costs
        reduction = standard_flops * (1 - 1/gqa_factor) * 0.3  # Rough estimate
        return standard_flops - reduction
```

**Step 2: Ranking Quality Metrics Per PetaFLOP**

This computes efficiency-effectiveness metrics for reranking.

```python
import numpy as np
from typing import List, Tuple

class RankingQualityMetrics:
    """Compute ranking metrics per unit of computation."""

    @staticmethod
    def compute_mrr(rankings: List[List[int]], true_relevant: List[int]) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first relevant item.
        Args:
            rankings: list of ranked document indices per query
            true_relevant: relevant document indices per query
        Returns:
            MRR score
        """
        mrr_scores = []
        for rank, doc_idx in enumerate(rankings, 1):
            if doc_idx in true_relevant:
                mrr_scores.append(1.0 / rank)
                break
        return np.mean(mrr_scores) if mrr_scores else 0.0

    @staticmethod
    def compute_ndcg(rankings: List[List[int]], true_relevant: List[int], k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain.
        Args:
            rankings: ranked document indices
            true_relevant: relevant document indices
            k: truncate at rank k
        Returns:
            NDCG@k score
        """
        dcg = 0.0
        for rank, doc_idx in enumerate(rankings[:k], 1):
            if doc_idx in true_relevant:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG: all relevant items ranked first
        ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(min(len(true_relevant), k)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def compute_hits_at_k(rankings: List[int], true_relevant: List[int], k: int = 10) -> float:
        """Hits@k: fraction of queries with at least one relevant document in top-k."""
        return float(any(doc in true_relevant for doc in rankings[:k]))

class EfficiencyEffectivenessEvaluator:
    def __init__(self, flops_estimator: TransformerFLOPsEstimator):
        self.flops_estimator = flops_estimator
        self.metrics = RankingQualityMetrics()

    def evaluate_reranker(
        self,
        query_len: int,
        document_len: int,
        num_documents: int,
        rankings: List[List[int]],
        true_relevant: List[List[int]],
        reranking_mode: str = "pointwise"
    ) -> Dict[str, float]:
        """
        Compute efficiency-effectiveness metrics for reranker.
        Returns metrics per PetaFLOP (10^15 FLOPs).
        """
        # Estimate FLOPs
        flops = self.flops_estimator.estimate_reranking_pass(
            query_len, document_len, num_documents, batch_size=1,
            reranking_mode=reranking_mode
        )
        petaflops = flops / 1e15

        # Compute ranking quality
        mrr_scores = []
        ndcg_scores = []
        hits_at_10 = []

        for i, ranking in enumerate(rankings):
            relevant = true_relevant[i]
            mrr = self.metrics.compute_mrr([ranking], relevant)
            ndcg = self.metrics.compute_ndcg(ranking, relevant, k=10)
            hits = self.metrics.compute_hits_at_k(ranking, relevant, k=10)

            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            hits_at_10.append(hits)

        avg_mrr = np.mean(mrr_scores)
        avg_ndcg = np.mean(ndcg_scores)
        avg_hits = np.mean(hits_at_10)

        # Per-PetaFLOP metrics
        return {
            "flops": flops,
            "petaflops": petaflops,
            "mrr": avg_mrr,
            "ndcg@10": avg_ndcg,
            "hits@10": avg_hits,
            "rpp_mrr": avg_mrr / petaflops,  # Ranking metrics per PetaFLOP (MRR)
            "rpp_ndcg": avg_ndcg / petaflops,  # Ranking metrics per PetaFLOP (NDCG)
            "queries_per_petaflop": 1.0 / petaflops,  # Query throughput metric
        }

    def compare_reranking_strategies(
        self,
        query_len: int,
        document_len: int,
        num_documents: int,
        rankings_dict: Dict[str, List[List[int]]],
        true_relevant: List[List[int]]
    ) -> Dict[str, Dict]:
        """Compare multiple reranking strategies (pointwise, listwise, pairwise)."""
        results = {}

        for strategy in ["pointwise", "listwise", "pairwise"]:
            results[strategy] = self.evaluate_reranker(
                query_len, document_len, num_documents,
                rankings_dict[strategy], true_relevant,
                reranking_mode=strategy
            )

        return results
```

**Step 3: Comparative Analysis and Benchmarking**

This framework benchmarks rerankers across different configurations.

```python
class RerankerBenchmark:
    def __init__(self):
        self.results = []

    def benchmark_model_variants(
        self,
        base_config: Dict,
        model_sizes: List[int],
        reranking_modes: List[str],
        query_len: int = 10,
        document_len: int = 100,
        num_documents: int = 100
    ) -> Dict:
        """Benchmark different model sizes and reranking strategies."""
        benchmark_results = {}

        for size in model_sizes:
            config = base_config.copy()
            config["hidden_dim"] = size
            config["ffn_dim"] = size * 4

            estimator = TransformerFLOPsEstimator(config)
            evaluator = EfficiencyEffectivenessEvaluator(estimator)

            for mode in reranking_modes:
                key = f"model_{size}_mode_{mode}"
                flops = estimator.estimate_reranking_pass(
                    query_len, document_len, num_documents,
                    reranking_mode=mode
                )

                # Simulate ranking (in practice, would use actual model)
                rankings = [[i for i in range(num_documents)]]  # Placeholder
                true_relevant = [[0, 1, 2]]  # Placeholder

                metrics = evaluator.evaluate_reranker(
                    query_len, document_len, num_documents,
                    rankings, true_relevant, mode
                )

                benchmark_results[key] = metrics

        return benchmark_results

    def summarize_efficiency_dominance(self, results: Dict) -> str:
        """Analyze which strategies are most efficient."""
        summary = []

        # Group by strategy
        strategy_results = {}
        for key, metrics in results.items():
            strategy = key.split("_mode_")[1]
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append((key, metrics))

        # Rank by efficiency
        for strategy, items in strategy_results.items():
            avg_rpp = np.mean([m["rpp_mrr"] for _, m in items])
            summary.append(f"{strategy}: avg RPP = {avg_rpp:.4f}")

        return "\n".join(sorted(summary, reverse=True))
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| Hidden Dimension | 768-1024 | 256-2048 | Larger dimensions = more FLOPs but better quality |
| FFN Expansion Ratio | 4x hidden | 2-8x | Controls MLP layer size; 4x is standard |
| Number of Layers | 12-24 | 6-48 | More layers improve quality but increase FLOPs |
| Reranking Mode | Pointwise | Pointwise/Listwise/Pairwise | Pointwise is most efficient |
| Query Length | 10-20 tokens | 1-50 | Longer queries add computation |
| Document Length | 100-256 tokens | 50-512 | Longer documents dominate FLOPs cost |
| Number of Documents | 100 | 10-1000 | Scales FLOPs linearly for pointwise |

**When to Use**

- Comparing LLM rerankers across different hardware and deployment scenarios
- Making architectural decisions for reranking (pointwise vs. listwise trade-offs)
- Evaluating efficiency improvements from techniques like grouped-query attention
- Publishing reproducible reranker benchmarks independent of specific hardware
- Cost-benefit analysis for reranking infrastructure decisions
- Selecting reranking strategies for production systems with latency constraints

**When NOT to Use**

- Measuring actual deployment latency (FLOPs ≠ wall-clock time without hardware context)
- Real-time systems where actual observed latency matters more than theoretical FLOPs
- Microarchitecture-specific optimizations (FLOPs is too coarse-grained)
- Scenarios where model accuracy is paramount and efficiency is secondary
- Fine-grained performance tuning (requires actual profiling on target hardware)

**Common Pitfalls**

- **Confusing FLOPs with latency**: FLOPs are intrinsic computational work; latency depends on hardware. Use FLOPs for fair comparisons, latency for deployment decisions.
- **Ignoring memory bandwidth**: FLOPs estimation doesn't account for memory transfers which can dominate on modern GPUs. Consider memory-bound vs. compute-bound operations.
- **Under-estimating overhead**: FLOPs formula assumes perfect efficiency. Real implementations have overhead from attention masking, padding, and framework inefficiencies.
- **Comparing across fundamentally different architectures**: Decoder-only vs. encoder-decoder vs. state-space models have different efficiency characteristics. Account for architectural differences.
- **Ignoring batching effects**: FLOPs per single sample differ from batched operation. Report metrics for both single-sample and batched inference.

## Reference

Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers. https://arxiv.org/abs/2507.06223
