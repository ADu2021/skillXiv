---
name: sequential-edge-inverse-entropy-voting
title: "Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.02309"
keywords: [Test-Time Scaling, Inference Optimization, Reasoning Chains, Entropy, Voting]
description: "Replace parallel self-consistency with sequential reasoning where chains iteratively build on previous attempts, weighted by inverse entropy to prioritize confident solutions, achieving 46.7 pp accuracy gains over parallel approaches."
---

# Title: Build Reasoning Chains Sequentially for Superior Accuracy

The dominant paradigm for improving language model reasoning is parallel self-consistency: run N independent reasoning chains, then vote. Sequential Edge shows this is suboptimal. By running chains sequentially where each explicitly builds on previous attempts, models achieve 95.6% superiority rates over parallel approaches. Further, weighting solutions by **inverse entropy** (favoring low-entropy/confident chains) outperforms majority voting universally.

This is a pure inference-time technique requiring no model retraining.

## Core Concept

**Sequential Reasoning With Entropy-Weighted Aggregation**:
- **Sequential Scaling**: Each chain reads previous attempts and tries to improve them
- **Iterative Refinement**: Three mechanisms available: error correction, context accumulation, verification
- **Inverse-Entropy Voting**: Weight each answer by how certain the model was (low entropy = high confidence)
- **No Training Required**: Works with base models or fine-tuned variants
- **Universal Superiority**: Better than parallel approaches across all tested models and benchmarks

## Architecture Overview

- **Sequential Chain Generation**: Causal generation where chain k reads output of chain k-1
- **Confidence Estimation**: Compute Shannon entropy from token-level probability distributions
- **Entropy-Weighted Aggregation**: Answers weighted inversely to their generation entropy
- **Stopping Criterion**: Continue sequencing until convergence or budget exhausted
- **Hybrid Architecture**: Can mix sequential and parallel chains for different problem types

## Implementation Steps

**1. Implement Sequential Chain Generation**

Generate reasoning chains that explicitly reference previous attempts.

```python
class SequentialReasoningGenerator:
    def __init__(self, model, tokenizer, max_chains=6):
        self.model = model
        self.tokenizer = tokenizer
        self.max_chains = max_chains

    def generate_sequential_chains(self, question, num_chains=6):
        """Generate chains sequentially with cross-chain reference"""
        chains = []
        entropies = []

        # First chain: independent reasoning
        prompt_1 = f"""Solve this problem step by step:
        {question}

        Solution:"""
        chain_1, entropy_1 = self.generate_chain_with_entropy(prompt_1)
        chains.append(chain_1)
        entropies.append(entropy_1)

        # Subsequent chains: build on previous
        for i in range(1, num_chains):
            # Acknowledge previous attempt
            previous_solution = chains[-1]

            prompt_n = f"""Previous attempt at this problem:
            {previous_solution}

            Reconsider the problem {question}
            Try a different approach to check if the previous solution is correct.

            New solution:"""

            chain_n, entropy_n = self.generate_chain_with_entropy(prompt_n)
            chains.append(chain_n)
            entropies.append(entropy_n)

        return chains, entropies

    def generate_chain_with_entropy(self, prompt):
        """Generate reasoning chain and compute entropy"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate with return_dict_in_generate to get log probabilities
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=300,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode generated tokens
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute entropy over generated tokens
        # outputs.scores: tuple of tensors [batch_size, vocab_size] for each position
        entropy = self.compute_sequence_entropy(outputs.scores)

        return generated_text, entropy

    def compute_sequence_entropy(self, scores):
        """Compute entropy of token probability distribution"""
        entropies = []

        for token_scores in scores:
            # Get probabilities from logits
            probs = F.softmax(token_scores, dim=-1)

            # Shannon entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropies.append(entropy.item())

        # Return average entropy over sequence
        return np.mean(entropies) if entropies else 0.0
```

**2. Implement Inverse-Entropy Voting**

Weight answers by confidence (inverse of entropy).

```python
class InverseEntropyVoter:
    def __init__(self, extraction_fn):
        """extraction_fn: function that extracts answer from solution text"""
        self.extract_answer = extraction_fn

    def extract_answers_and_weights(self, chains, entropies):
        """Extract final answers and compute entropy-based weights"""
        answers = []
        weights = []

        for chain, entropy in zip(chains, entropies):
            # Extract answer from chain
            answer = self.extract_answer(chain)
            answers.append(answer)

            # Weight inversely to entropy: low entropy -> high weight
            # Use exponential scaling for sharper differences
            weight = np.exp(-entropy)
            weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return answers, weights

    def weighted_vote(self, chains, entropies, answer_similarity_fn=None):
        """Aggregate answers weighted by inverse entropy"""
        answers, weights = self.extract_answers_and_weights(chains, entropies)

        # Group similar answers
        if answer_similarity_fn:
            answer_groups = self.group_similar_answers(answers, answer_similarity_fn)
        else:
            answer_groups = {ans: [i for i, a in enumerate(answers) if a == ans]
                             for ans in set(answers)}

        # Compute weighted score for each answer group
        group_scores = {}
        for answer, indices in answer_groups.items():
            score = sum(weights[i] for i in indices)
            group_scores[answer] = score

        # Return highest-scoring answer
        final_answer = max(group_scores, key=group_scores.get)
        confidence = group_scores[final_answer]

        return final_answer, confidence, group_scores

    def group_similar_answers(self, answers, similarity_fn):
        """Group answers that are semantically similar"""
        groups = {}
        for i, answer in enumerate(answers):
            # Find most similar existing group
            best_group = None
            best_similarity = 0

            for group_answer in groups:
                sim = similarity_fn(answer, group_answer)
                if sim > best_similarity:
                    best_similarity = sim
                    best_group = group_answer

            if best_group and best_similarity > 0.8:
                groups[best_group].append(i)
            else:
                groups[answer] = [i]

        return groups
```

**3. Implement Hybrid Sequential-Parallel Strategy**

Combine approaches for complex problems.

```python
class HybridReasoningStrategy:
    def __init__(self, model, tokenizer, num_sequential=3, num_parallel=3):
        self.sequential_gen = SequentialReasoningGenerator(model, tokenizer)
        self.voter = InverseEntropyVoter(self.extract_final_answer)
        self.num_seq = num_sequential
        self.num_par = num_parallel

    def solve_with_budget(self, question, token_budget=10000):
        """Solve using hybrid strategy within token budget"""
        # Phase 1: Sequential reasoning (focus on quality)
        seq_chains, seq_entropies = self.sequential_gen.generate_sequential_chains(
            question, self.num_seq
        )
        tokens_used = sum(len(c.split()) for c in seq_chains) * 1.3  # Estimate

        # Phase 2: Parallel chains if budget remains
        if tokens_used < token_budget * 0.7:
            parallel_chains = []
            for _ in range(self.num_par):
                prompt = f"Solve: {question}\nSolution:"
                chain, _ = self.sequential_gen.generate_chain_with_entropy(prompt)
                parallel_chains.append(chain)

            # Combine all chains
            all_chains = seq_chains + parallel_chains
            all_entropies = seq_entropies + [0.0] * len(parallel_chains)  # Parallel have no reference entropy
        else:
            all_chains = seq_chains
            all_entropies = seq_entropies

        # Aggregate
        answer, confidence, scores = self.voter.weighted_vote(all_chains, all_entropies)
        return answer, confidence

    def extract_final_answer(self, solution_text):
        """Extract final answer from solution"""
        # Simple strategy: last numeric value or final sentence
        import re
        numbers = re.findall(r'-?\d+\.?\d*', solution_text)
        if numbers:
            return float(numbers[-1])

        # Fallback: last sentence
        sentences = solution_text.split('.')
        return sentences[-2] if len(sentences) > 1 else solution_text
```

**4. Evaluate Sequential vs Parallel**

Compare efficiency and accuracy.

```python
def compare_strategies(model, tokenizer, test_questions):
    """Benchmark sequential vs parallel approaches"""
    results = {
        'sequential': {'accuracy': [], 'tokens': []},
        'parallel': {'accuracy': [], 'tokens': []},
        'hybrid': {'accuracy': [], 'tokens': []}
    }

    seq_gen = SequentialReasoningGenerator(model, tokenizer)
    voter = InverseEntropyVoter(extract_answer_fn)

    for question, ground_truth in test_questions:
        # Sequential approach
        seq_chains, seq_entropies = seq_gen.generate_sequential_chains(question, num_chains=6)
        seq_answer, _ = voter.weighted_vote(seq_chains, seq_entropies)
        seq_correct = seq_answer == ground_truth
        seq_tokens = sum(len(c.split()) * 1.3 for c in seq_chains)

        results['sequential']['accuracy'].append(seq_correct)
        results['sequential']['tokens'].append(seq_tokens)

        # Parallel approach (independent chains)
        par_chains = []
        for _ in range(6):
            chain, entropy = seq_gen.generate_chain_with_entropy(
                f"Solve: {question}\nSolution:"
            )
            par_chains.append(chain)

        par_entropies = [0.0] * len(par_chains)
        par_answer, _ = voter.weighted_vote(par_chains, par_entropies)
        par_correct = par_answer == ground_truth
        par_tokens = sum(len(c.split()) * 1.3 for c in par_chains)

        results['parallel']['accuracy'].append(par_correct)
        results['parallel']['tokens'].append(par_tokens)

    # Compute statistics
    for strategy in results:
        acc = np.mean(results[strategy]['accuracy'])
        tokens = np.mean(results[strategy]['tokens'])
        print(f"{strategy}: {acc:.1%} accuracy, {tokens:.0f} tokens/problem")

    return results
```

## Practical Guidance

**When to Use**:
- Any test-time reasoning optimization (no retraining needed)
- Mathematical problems, STEM benchmarks
- Settings where inference cost matters more than model size

**Hyperparameters**:
- num_sequential_chains: 4-8 (returns diminish beyond 6)
- entropy_temperature: 1.0 (controls sharpness of entropy-based weighting)
- token_budget: Adjust based on latency constraints

**When NOT to Use**:
- Real-time applications (sequential generation is slower than parallel)
- Streaming scenarios where latency is critical
- Models without reliable probability distributions

**Pitfalls**:
- **Answer extraction brittleness**: Regex/heuristic extraction fails on varied formats; use semantic matching
- **Entropy unreliability**: Some models produce low-entropy nonsense; validate with baseline
- **Sequential dependency**: If first chain is very wrong, subsequent chains may not recover; use restart strategies

**Key Insight**: This is one of the few inference-time tricks that consistently beats the dominant approach. The mechanism is simple: sequential reasoning enables error correction that parallel approaches miss.

## Reference

arXiv: https://arxiv.org/abs/2511.02309
