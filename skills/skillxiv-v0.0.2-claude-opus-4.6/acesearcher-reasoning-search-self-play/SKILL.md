---
name: acesearcher-reasoning-search-self-play
title: "AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.24193"
keywords: [reasoning, question-decomposition, self-play, reinforcement-learning, RAG, DPO, multi-hop-reasoning, parameter-efficiency]
description: "Train a single LLM to decompose complex queries into subquestions and integrate retrieved contexts through two-stage supervised and preference-based reinforcement fine-tuning, achieving 7.6% average improvement and matching 685B models with 32B parameters."
---

## AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play

### Outcome

Implement a cooperative self-play framework that trains a single LLM to alternate between decomposer and solver roles, enhancing multi-hop reasoning and retrieval-augmented generation without requiring intermediate answer annotations. Achieve parameter-efficient performance matching much larger models through iterative preference optimization.

### Problem Context

Complex question answering demands multi-step reasoning and precise information retrieval. Traditional approaches struggle with:

- Decomposing ambiguous queries into retrievable subquestions
- Integrating multiple retrieved passages without losing coherence
- Requiring expensive intermediate supervision for each reasoning step
- Scaling linearly with model size for competitive performance

AceSearcher addresses these by training a single LLM to handle both question decomposition and answer synthesis through self-play, using only final-answer signals for optimization.

### Core Concept

AceSearcher treats reasoning as a cooperative process between two roles played by the same model:

**Decomposer (ρ)**: Transforms an input question into a sequence of subquestions designed to guide retrieval. Given question q, produces subquestions z = (z₁, z₂, ..., zₙ).

**Solver (π)**: Generates intermediate answers for each subquestion by consulting retrieved documents, then synthesizes all intermediate answers into a final response. Takes subquestion zᵢ, prior answers w<ᵢ, and retrieved context Dᵢ to produce intermediate answer wᵢ, then generates final answer a' from all components.

The key insight is that better decompositions enable the solver to retrieve more relevant context and produce higher-quality answers. This mutual dependency justifies joint optimization without separate roles or models.

### Architecture Overview

- **Unified Model**: Single LLM controlled by templates and prompts, no separate role-specific parameters
- **Two-Stage Training**: Supervised fine-tuning on diverse reasoning tasks, followed by preference-based reinforcement optimization
- **Preference Generation**: Multipath rollouts sample multiple decompositions and solution paths, creating preference pairs from max/min trajectories
- **Iterative DPO**: Direct Preference Optimization applied across three preference datasets (decomposition quality, intermediate answers, final answers) with convergence guarantees
- **Retrieval Integration**: Sparse retrieval allocates documents per subquestion (typically 15 total documents distributed across decomposed steps)

### Implementation

#### Stage 1: Supervised Fine-Tuning on Diverse Tasks

The foundation requires exposure to search, reasoning, and decomposition patterns across 180,600 examples:

```python
# sft_data_preparation.py
from dataclasses import dataclass
from typing import List

@dataclass
class SFTExample:
    instruction: str
    input: str
    output: str
    task_category: str  # "search", "reasoning", "decomposition"

# Data composition
sft_datasets = {
    "context_rich_qa": {
        "sources": ["NQ", "SQuAD", "DROP", "NarrativeQA", "Quoref", "ROPES", "FEVER", "TAT-QA"],
        "count": 132000,
        "purpose": "Extract answers from retrieved contexts"
    },
    "question_decomposition": {
        "sources": ["GSM8K", "ConvFinQA", "StrategyQA"],
        "count": 9600,
        "purpose": "Break complex questions into subquestions"
    },
    "chain_of_thought": {
        "sources": ["GSM8K", "TabMWP", "IfQA", "MathInstruct"],
        "count": 39000,
        "purpose": "Multi-step reasoning patterns"
    }
}

total_sft_examples = sum(v["count"] for v in sft_datasets.values())
# Result: 180,600 examples

# Hyperparameters for SFT
sft_config = {
    "batch_size": 64,
    "max_tokens": 2560,
    "epochs": 1,
    "warmup_steps_ratio": 0.05,
    "learning_rates": {
        "1.5B": 5e-6,
        "8B": 1e-6,
        "14B": 1e-6,
        "32B_LoRA": 1e-5
    },
    "optimizer": "AdamW",
    "lr_scheduler": "cosine"
}
```

Use standard next-token prediction loss. Mix tasks equally to ensure the model learns all three competencies.

#### Stage 2: Preference-Based Reinforcement Fine-Tuning

Generate preference pairs via multipath rollouts and optimize with iterative DPO:

```python
# preference_generation.py
import numpy as np
from typing import Tuple, List

class PreferenceDataGenerator:
    def __init__(self, model, retriever, num_decompositions=3, num_solutions=4):
        self.model = model
        self.retriever = retriever
        self.m = num_decompositions      # candidate decompositions per question
        self.m_prime = num_solutions      # candidate solutions per decomposition

    def compute_reward(self, question: str, answer: str, gold_answer: str) -> float:
        """
        Reward function: exact match with format validation
        r(q, a', a) = EM(a', a) × 𝕀(f(q, a') = 1)
        """
        em_score = 1.0 if self._normalize_answer(answer) == self._normalize_answer(gold_answer) else 0.0
        format_valid = self._check_format(question, answer)
        return em_score * float(format_valid)

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answers for EM computation"""
        answer = answer.lower().strip()
        answer = ' '.join(answer.split())
        return answer

    def _check_format(self, question: str, answer: str) -> bool:
        """Validate format: proper structure for subquestions, answers, reasoning"""
        required_markers = ['###', 'Answer:'] if '##' in question else []
        return all(marker in answer for marker in required_markers) if required_markers else True

    def generate_rollouts(self, question: str, gold_answer: str) -> Tuple[List, List]:
        """
        Generate m decompositions × m' solutions per decomposition
        """
        decompositions = []
        rewards_per_decomp = []

        # Rollout m candidate decompositions
        for i in range(self.m):
            z_i = self.model.generate_decomposition(question, temperature=1.0)

            # Retrieve documents for this decomposition
            subquestions = self._parse_subquestions(z_i)
            all_docs = []
            for sq in subquestions:
                docs = self.retriever.retrieve(sq, k=10//len(subquestions))
                all_docs.extend(docs)

            # Rollout m' solutions for this decomposition
            solutions = []
            solution_rewards = []
            for j in range(self.m_prime):
                a_j = self.model.generate_answer(
                    question, z_i, all_docs, temperature=1.0
                )
                reward = self.compute_reward(question, a_j, gold_answer)
                solutions.append((z_i, a_j))
                solution_rewards.append(reward)

            decompositions.append((z_i, solutions, solution_rewards))
            avg_reward = np.mean(solution_rewards)
            rewards_per_decomp.append(avg_reward)

        return decompositions, rewards_per_decomp

    def create_preference_pairs(self,
                               question: str,
                               decompositions: List,
                               rewards_per_decomp: List) -> dict:
        """
        Create three preference pair datasets: D_decompose, D_subq, D_final
        """
        preferences = {"decompose": [], "subquestion": [], "final": []}

        # D_decompose: compare decompositions by expected reward
        best_decomp_idx = np.argmax(rewards_per_decomp)
        worst_decomp_idx = np.argmin(rewards_per_decomp)

        if rewards_per_decomp[best_decomp_idx] != rewards_per_decomp[worst_decomp_idx]:
            z_plus = decompositions[best_decomp_idx][0]
            z_minus = decompositions[worst_decomp_idx][0]
            preferences["decompose"].append({
                "question": question,
                "preferred": z_plus,
                "dispreferred": z_minus
            })

        # D_subq: compare intermediate answers
        best_decomp_idx = np.argmax(rewards_per_decomp)
        best_z, best_solutions, best_rewards = decompositions[best_decomp_idx]
        subquestions = self._parse_subquestions(best_z)

        for i, sq in enumerate(subquestions):
            best_sol_idx = np.argmax(best_rewards)
            worst_sol_idx = np.argmin(best_rewards)
            if best_rewards[best_sol_idx] != best_rewards[worst_sol_idx]:
                z_i, w_i_best = best_solutions[best_sol_idx]
                z_i, w_i_worst = best_solutions[worst_sol_idx]
                preferences["subquestion"].append({
                    "subquestion": sq,
                    "preferred_answer": w_i_best,
                    "dispreferred_answer": w_i_worst
                })

        # D_final: compare final answers using best trajectory
        best_sol_idx = np.argmax(best_rewards)
        _, a_best = best_solutions[best_sol_idx]
        worst_sol_idx = np.argmin(best_rewards)
        _, a_worst = best_solutions[worst_sol_idx]

        if best_rewards[best_sol_idx] != best_rewards[worst_sol_idx]:
            preferences["final"].append({
                "question": question,
                "context": best_z,
                "preferred_answer": a_best,
                "dispreferred_answer": a_worst
            })

        return preferences

    def _parse_subquestions(self, decomposition: str) -> List[str]:
        """Extract subquestions from decomposition text"""
        lines = decomposition.split('\n')
        subqs = [line.strip() for line in lines if line.startswith('###')]
        return [sq.replace('###', '').strip() for sq in subqs]
```

#### Stage 2 Continued: Direct Preference Optimization

Apply iterative DPO across the unified preference dataset:

```python
# dpo_training.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AceSearcherDPO:
    def __init__(self, model, reference_model, beta=0.5, learning_rate=5e-7):
        self.model = model
        self.reference_model = reference_model
        self.beta = beta  # KL penalty coefficient
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def dpo_loss(self,
                 prompt: str,
                 preferred_completion: str,
                 dispreferred_completion: str,
                 batch_size: int = 16) -> torch.Tensor:
        """
        Direct Preference Optimization loss
        ℒ_DPO := -log σ(β[log(p_θ(g^+|x)/p_ref(g^+|x)) - log(p_θ(g^-|x)/p_ref(g^-|x))])
        """
        # Tokenize inputs
        prompt_ids = self.model.tokenize(prompt)
        preferred_ids = self.model.tokenize(preferred_completion)
        dispreferred_ids = self.model.tokenize(dispreferred_completion)

        # Compute log probabilities with current model
        with torch.no_grad():
            ref_logits_preferred = self.reference_model(
                torch.cat([prompt_ids, preferred_ids], dim=-1)
            ).logits
            ref_logits_dispreferred = self.reference_model(
                torch.cat([prompt_ids, dispreferred_ids], dim=-1)
            ).logits

        # Get current model logits
        logits_preferred = self.model(
            torch.cat([prompt_ids, preferred_ids], dim=-1)
        ).logits
        logits_dispreferred = self.model(
            torch.cat([prompt_ids, dispreferred_ids], dim=-1)
        ).logits

        # Compute log-probabilities
        def get_log_prob(logits, token_ids):
            """Extract log probability of each token"""
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather log probs for each position
            token_log_probs = torch.gather(
                log_probs[:, :-1, :], 2, token_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            return token_log_probs.sum(dim=-1)

        log_prob_theta_preferred = get_log_prob(logits_preferred, preferred_ids)
        log_prob_theta_dispreferred = get_log_prob(logits_dispreferred, dispreferred_ids)

        log_prob_ref_preferred = get_log_prob(ref_logits_preferred, preferred_ids)
        log_prob_ref_dispreferred = get_log_prob(ref_logits_dispreferred, dispreferred_ids)

        # DPO objective
        logit_diff = (
            (log_prob_theta_preferred - log_prob_ref_preferred) -
            (log_prob_theta_dispreferred - log_prob_ref_dispreferred)
        )

        loss = -F.logsigmoid(self.beta * logit_diff).mean()
        return loss

    def train_iteration(self, preference_dataset, num_epochs=1):
        """One complete DPO training pass"""
        dataloader = DataLoader(preference_dataset, batch_size=32, shuffle=True)

        self.model.train()
        total_loss = 0

        for epoch in range(num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()

                # Compute loss across all preference types
                losses = []
                for pref_type in ["decompose", "subquestion", "final"]:
                    if batch[pref_type]:
                        for pair in batch[pref_type]:
                            loss = self.dpo_loss(
                                pair["prompt"],
                                pair["preferred"],
                                pair["dispreferred"]
                            )
                            losses.append(loss)

                if losses:
                    batch_loss = torch.stack(losses).mean()
                    batch_loss.backward()
                    self.optimizer.step()
                    total_loss += batch_loss.item()

        return total_loss / len(dataloader)
```

#### Multi-Turn DPO with Convergence

Run iterative DPO with model updates:

```python
# iterative_training.py
class IterativeDPOTrainer:
    def __init__(self, model_path, num_iterations=2):
        self.num_iterations = num_iterations
        self.model_path = model_path

    def run_multi_turn_dpo(self, train_questions, gold_answers, retriever):
        """
        Multi-turn DPO: θ^(0) → θ^(1) → θ^(2)
        At each iteration, use current model to generate new preference data
        """
        # Load or initialize model
        model = self._load_model()
        preference_gen = PreferenceDataGenerator(model, retriever)
        dpo_trainer = AceSearcherDPO(model, model.copy())  # ref model = initial model

        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")

            # Generate preference pairs using current model θ^(t)
            all_preferences = {"decompose": [], "subquestion": [], "final": []}

            for question, gold_answer in zip(train_questions, gold_answers):
                decompositions, rewards = preference_gen.generate_rollouts(
                    question, gold_answer
                )
                prefs = preference_gen.create_preference_pairs(
                    question, decompositions, rewards
                )

                # Accumulate preferences
                for pref_type in ["decompose", "subquestion", "final"]:
                    all_preferences[pref_type].extend(prefs[pref_type])

            # Filter out equivalent pairs (reward difference = 0)
            filtered_prefs = self._filter_equivalent_pairs(all_preferences)

            # Train for one epoch with DPO
            loss = dpo_trainer.train_iteration(filtered_prefs, num_epochs=1)
            print(f"  Iteration {iteration + 1} loss: {loss:.4f}")

            # Save checkpoint
            self._save_checkpoint(model, iteration)

            # Update reference model for next iteration
            dpo_trainer.reference_model = model.copy()

        return model

    def _filter_equivalent_pairs(self, preferences):
        """Remove pairs where preferred and dispreferred have equal rewards"""
        filtered = {"decompose": [], "subquestion": [], "final": []}
        for pref_type, pairs in preferences.items():
            for pair in pairs:
                # Pairs already filtered during generation, but validate
                if pair.get("preferred") != pair.get("dispreferred"):
                    filtered[pref_type].append(pair)
        return filtered

    def _load_model(self):
        return torch.load(self.model_path)

    def _save_checkpoint(self, model, iteration):
        torch.save(model.state_dict(), f"{self.model_path}_iter{iteration}")
```

#### Prompt Templates for QA Tasks

```python
# prompt_templates.py

DECOMPOSITION_TEMPLATE = """Please break down the question "{question}" into multiple specific
sub-questions that address individual components of the original question.
Mark each sub-question with ### at the beginning. If you need to refer to
answers from earlier sub-questions, use #1, #2, etc. to reference them.

Provide the sub-questions only, without any additional explanation."""

ANSWER_GENERATION_TEMPLATE = """You have the following passages:
{passages}

You are also given some subquestions and their answers:
{subquestion_answers}

Please answer the question "{original_question}" with a short span
using the documents and subquestions as reference. Be concise and direct."""

# Example output
decomposition_example = """### What is the capital of France?
### What is the population of Paris?
### When did Paris become the capital?"""

answer_example = """The capital of France is Paris. Paris has a metropolitan population
of approximately 2.1 million people. Paris became the capital in 987 AD."""
```

### Practical Guidance

**Hyperparameter Selection**

| Parameter | Recommended Range | Impact | Notes |
|-----------|-------------------|--------|-------|
| m (decompositions) | 2-5 | Quality vs. speed | 3 offers good balance; diminishing returns beyond 4 |
| m' (solutions per decomposition) | 3-8 | Preference diversity | 4 typical; increase if reward signal is noisy |
| beta (KL penalty) | 0.5-1.0 | Regularization strength | Higher beta = stronger ref model adherence |
| k (retrieved passages) | 8-15 | Coverage vs. noise | Gains plateau at k=10; allocate per subquestion |
| num_iterations | 1-3 | Convergence | 2 iterations balances performance and training cost |
| learning_rate SFT | 5e-7 to 5e-6 | Stability | Scale with model size; smaller LR for larger models |
| learning_rate RFT | 5e-7 to 1e-6 | Convergence | Typically lower than SFT to avoid divergence |
| batch_size | 32-64 | GPU memory | Larger batches enable better gradient estimation |

**When to Use AceSearcher**

- Complex multi-hop reasoning requiring decomposition (HotpotQA, 2WikiMHQA)
- Document-level reasoning over tables and finance (TAT-QA, FinQA)
- Fact verification requiring evidence aggregation (FEVER, HOVER)
- Scenarios where parameter efficiency is critical (edge devices, cost-sensitive)
- When intermediate reasoning steps improve interpretability

**When NOT to Use AceSearcher**

- Single-turn factual retrieval (decomposition adds unnecessary latency)
- Closed-domain QA where a single passage answers the question
- Real-time inference with strict latency constraints (2-3x longer than single-turn RAG)
- Tasks requiring real-time knowledge updates (training cycle is multiday)
- Scenarios where explainability of decomposition is not valued

**Common Pitfalls**

- **Insufficient SFT diversity**: Training only on search or only on reasoning hurts both components. Ensure balanced exposure to all three task categories.
- **Subquestion ambiguity**: Decompositions that are too similar or overlapping waste retrieval budget. Add examples of diverse decomposition patterns to SFT data.
- **Reward signal weakness**: If EM-only rewards are sparse (e.g., <10% positive examples), add format validation or intermediate correctness checks.
- **Document allocation mismatches**: Distributing more documents to some subquestions than others can create learning bias. Use uniform or learned allocation.
- **Convergence instability in RFT**: Starting with too high a learning rate causes preference divergence. Use 0.5-1.0x the SFT learning rate for RFT.
- **Preference pair bottleneck**: Generating m × m' rollouts per question scales poorly. Use sampling or importance weighting for large-scale training.

**Efficiency Considerations**

- SFT on 180K examples at batch size 64 takes ~25 GPU hours (A100)
- RFT with 2 iterations on 49K QA examples: ~12 GPU hours per iteration
- Inference is 2-3× slower than single-turn retrieval due to decomposition
- Parameter efficiency: AceSearcher-32B matches DeepSeek-V3 (685B) on document reasoning using <5% parameters

### Reference

For full technical details including theoretical convergence proofs, ablation studies, and complete experimental results, see the paper:

[AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](https://arxiv.org/abs/2509.24193)

Official implementation: https://github.com/ritaranx/AceSearcher
Model weights: https://huggingface.co/AceSearcher
