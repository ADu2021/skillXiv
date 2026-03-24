---
name: group-rank-reranking-rl
title: "GroupRank: Groupwise Reranking Paradigm Driven by RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.11653"
keywords: [Information Retrieval, Reranking, GRPO, Ranking Optimization, Heterogeneous Rewards]
description: "Improve ranking quality via groupwise reranking with RL—process document groups jointly for within-group comparisons using GRPO with composite rewards (recall, ranking metrics, distribution alignment)."
---

# Improve Ranking with Groupwise Comparisons and RL-Driven Optimization

Traditional rankers score documents independently (pointwise) or globally (listwise). Groupwise ranking balances both: the model receives multiple documents simultaneously and assigns scores through within-group comparisons, avoiding the computational burden of full listwise scoring while maintaining ranking awareness.

GroupRank combines this groupwise mechanism with GRPO (reinforcement learning), using a composite reward function that optimizes recall, ranking quality, and score distribution alignment. This approach enables more efficient reranking while achieving state-of-the-art ranking performance.

## Core Concept

Information retrieval systems retrieve candidate documents, then rerank them. Standard approaches:

- **Pointwise**: Score each document independently; ignores comparative relationships
- **Listwise**: Process all documents jointly; most accurate but computationally expensive

GroupRank introduces a **groupwise** middle ground: score a batch of documents (e.g., top-20) jointly in a single forward pass. The model performs within-group comparisons, assigning scores that reflect relative relevance while avoiding full-list overhead.

Combined with GRPO training, GroupRank optimizes a heterogeneous reward signal combining three objectives: ensuring relevant documents rank high (recall), optimizing ranking metrics (NDCG, RBO), and aligning score distributions with ground-truth labels.

## Architecture Overview

- **Groupwise Scoring Head**: Takes group of documents and query, outputs relative scores via within-group comparisons
- **SFT Pretraining**: Supervised fine-tuning on ranking data for instruction following and format control
- **GRPO Training**: Reinforcement learning with group-wise comparative advantage and reward aggregation
- **Heterogeneous Reward Function**: Composite signal (recall + ranking metrics + distribution) guiding optimization
- **Data Synthesis Pipeline**: Auto-generate high-quality training data from retrieval and ranking datasets

## Implementation Steps

**Step 1: Groupwise Scoring Mechanism.** Process multiple documents jointly for relative scoring.

```python
class GroupwiseReranker(nn.Module):
    def __init__(self, model_size='base', num_docs_per_group=20):
        super().__init__()
        self.base_model = load_base_model(model_size)  # e.g., BERT-large
        self.num_docs_per_group = num_docs_per_group
        self.group_scorer = nn.Linear(768, 1)  # Project to relevance score

    def forward(self, query, documents, return_logits=False):
        """
        Score documents through within-group comparisons.
        query: (batch_size, query_tokens)
        documents: (batch_size, num_docs, doc_tokens)
        """
        batch_size, num_docs, _ = documents.shape

        # Encode query once
        query_reps = self.base_model(query)  # (batch_size, hidden_dim)

        # Encode documents
        doc_flat = documents.reshape(batch_size * num_docs, -1)
        doc_reps = self.base_model(doc_flat)  # (batch_size * num_docs, hidden_dim)
        doc_reps = doc_reps.reshape(batch_size, num_docs, -1)

        # Score each document relative to query
        # This is the groupwise component: document scores depend on query context
        scores = []
        for i in range(batch_size):
            query_rep = query_reps[i:i+1]  # (1, hidden_dim)
            doc_group = doc_reps[i]  # (num_docs, hidden_dim)

            # Compute similarity and apply groupwise scoring
            similarities = torch.matmul(doc_group, query_rep.t())  # (num_docs, 1)
            group_score = self.group_scorer(doc_group)  # (num_docs, 1)

            # Within-group normalization: softmax so scores are comparative
            combined = similarities + group_score
            combined = torch.softmax(combined, dim=0)

            scores.append(combined.squeeze(-1))

        scores = torch.stack(scores)  # (batch_size, num_docs)

        if return_logits:
            return scores
        return torch.argsort(scores, dim=1, descending=True)  # (batch_size, num_docs)
```

**Step 2: Composite Reward Function.** Combine recall, ranking metrics, and distribution alignment.

```python
class HeterogeneousRewardFunction:
    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3):
        self.alpha = alpha    # recall weight
        self.beta = beta      # ranking weight
        self.gamma = gamma    # distribution weight

    def compute_reward(self, predicted_scores, ground_truth_labels, rank_cutoff=10):
        """
        Compute composite reward combining multiple objectives.
        predicted_scores: model output (batch_size, num_docs)
        ground_truth_labels: relevance labels (batch_size, num_docs)
        """
        # Recall Reward: ensure relevant docs rank high
        recall_reward = self.compute_recall_reward(
            predicted_scores, ground_truth_labels, rank_cutoff
        )

        # Ranking Reward: optimize NDCG and RBO
        ranking_reward = self.compute_ranking_reward(
            predicted_scores, ground_truth_labels
        )

        # Distribution Reward: align score distributions with ground truth
        distribution_reward = self.compute_distribution_reward(
            predicted_scores, ground_truth_labels
        )

        # Composite
        total_reward = (
            self.alpha * recall_reward +
            self.beta * ranking_reward +
            self.gamma * distribution_reward
        )

        return total_reward

    def compute_recall_reward(self, predicted_scores, labels, k=10):
        """
        Recall@k: proportion of relevant docs in top-k predictions.
        """
        batch_size = predicted_scores.shape[0]
        rewards = []

        for i in range(batch_size):
            scores = predicted_scores[i]
            true_labels = labels[i]

            # Top-k predictions
            top_k_indices = torch.topk(scores, min(k, len(scores)))[1]
            top_k_labels = true_labels[top_k_indices]

            # Recall: fraction of relevant docs retrieved
            num_relevant = (true_labels > 0).sum().float()
            num_retrieved = (top_k_labels > 0).sum().float()

            recall = num_retrieved / (num_relevant + 1e-8)
            rewards.append(recall.item())

        return torch.tensor(rewards).mean()

    def compute_ranking_reward(self, predicted_scores, labels):
        """
        Ranking reward: NDCG and RBO metrics.
        """
        # NDCG@10
        ndcg_10 = self.compute_ndcg(predicted_scores, labels, k=10)

        # RBO: rank-biased overlap (gives more weight to top ranks)
        rbo = self.compute_rbo(predicted_scores, labels)

        return 0.6 * ndcg_10 + 0.4 * rbo

    def compute_distribution_reward(self, predicted_scores, labels):
        """
        Distribution alignment: KL divergence between predicted and ground-truth
        score distributions. Preserves magnitude information.
        """
        # Normalize scores to probability distributions
        pred_dist = torch.softmax(predicted_scores, dim=1)
        true_dist = torch.softmax(labels.float(), dim=1)

        # KL divergence (symmetric version: JS divergence)
        kl_div = torch.nn.functional.kl_div(
            torch.log(pred_dist + 1e-8),
            true_dist,
            reduction='batchmean'
        )

        # Reward is negative KL (lower KL = higher reward)
        return -kl_div
```

**Step 3: GRPO Training.** Optimize groupwise ranker with RL using composite rewards.

```python
def train_grouprank_with_grpo(
    model, train_loader, reward_fn, num_iterations=10000,
    lr=1e-5, gamma=0.99
):
    """
    Train groupwise ranker using Group Relative Policy Optimization.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for iteration in range(num_iterations):
        for batch in train_loader:
            queries = batch['queries']
            documents = batch['documents']
            labels = batch['labels']

            # Forward pass
            predicted_scores = model(queries, documents, return_logits=True)

            # Compute reward
            rewards = reward_fn.compute_reward(predicted_scores, labels)

            # GRPO: advantage estimation and policy gradient
            # Compute advantage (how much better than baseline)
            baseline = torch.mean(rewards)
            advantages = rewards - baseline

            # Policy gradient loss: maximize expected reward
            log_probs = torch.log_softmax(predicted_scores, dim=1)
            selected_log_probs = log_probs.gather(
                1,
                torch.argsort(predicted_scores, dim=1, descending=True)
            )

            policy_loss = -(selected_log_probs.sum(dim=1) * advantages).mean()

            # Entropy regularization (encourage exploration)
            entropy = -(predicted_scores.softmax(dim=1) * log_probs).sum(dim=1).mean()
            loss = policy_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: loss={loss.item():.4f}, reward={rewards.mean():.4f}")
```

## Practical Guidance

**When to Use:** Re-ranking in information retrieval systems (search engines, question-answering, recommendation). Use when you have ranked candidate sets and want to optimize ranking quality with RL.

**Hyperparameters:**
- Group size: 20–100 documents; larger groups provide more context but increase computation
- Reward weights (α, β, γ): start equal (1/3 each), then adjust based on task focus (e.g., favor ranking metrics for search)
- GRPO learning rate: 1e-5 to 1e-4; use lower rates to stabilize RL training

**Pitfalls:**
- **Reward hacking**: Composite rewards can lead to optimization of one component at others' expense; monitor all three components during training
- **Group size sensitivity**: Too-small groups lose context; too-large groups inflate computation; profile on your dataset
- **Distribution reward instability**: KL divergence can be noisy; consider smoothing or using symmetric JS divergence
- **Data scarcity**: Ensure you have sufficient labeled ranking data for GRPO training; use data synthesis if needed

**When NOT to Use:** Small retrieval sets (<10 documents) where listwise scoring is already efficient; real-time serving with strict latency budgets.

**Integration:** Replace existing pointwise rankers directly; pairs well with dense retrieval for candidate generation.

---
Reference: https://arxiv.org/abs/2511.11653
