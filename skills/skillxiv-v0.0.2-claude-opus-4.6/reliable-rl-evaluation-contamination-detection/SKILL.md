---
name: reliable-rl-evaluation-contamination-detection
title: "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.10532"
keywords: [Data Contamination, Reinforcement Learning, Model Evaluation, Memorization Detection, Benchmark Integrity]
description: "Detect and eliminate data contamination that invalidates RL benchmarks by measuring benchmark reconstruction ability. Implement clean evaluation datasets to distinguish genuine reasoning improvements from memorization. Use when validating RL training results or ensuring benchmark integrity for mathematical reasoning."
---

# Detecting Contamination in RL Benchmarks: Distinguishing Reasoning from Memorization

RL training on mathematical reasoning benchmarks can show spurious improvements with incorrect reward signals, not from genuine reasoning capability but from memorization of benchmark problems in pre-training data. The Qwen2.5 model family exhibited unexpected performance gains even with random or inverted rewards on standard benchmarks (MATH-500, AIME), while Llama models did not. Investigation reveals that Qwen's pre-training corpus contains these benchmark problems, enabling models to memorize rather than reason.

This contamination invalidates benchmark-based evaluation: a model can appear to improve through RL when it's actually learning to retrieve memorized solutions more reliably. The solution is detecting contamination via activation-based metrics and building clean, procedurally-generated benchmarks that guarantee no data overlap.

## Core Concept

The method detects contamination through two key metrics: partial-prompt completion rate (how well models reconstruct full problems from 60% of the prompt using ROUGE-L) and partial-prompt answer accuracy (whether models recover correct answers). Models with high reconstruction ability are memorizing, not reasoning.

The RandomCalculation dataset provides a clean alternative: procedurally-generated arithmetic expressions with random operands and operators guarantee no pre-training overlap. By comparing RL behavior on contaminated vs. clean benchmarks, the approach isolates genuine reasoning improvements from exploitation of memorization.

## Architecture Overview

- **Contamination Detection Module**: ROUGE-L scoring on partial-prompt completions to measure benchmark reconstruction ability
- **Partial-Prompt Test Framework**: Truncate prompts to 60%, measure reconstruction and answer accuracy separately
- **RL Evaluation Pipeline**: GRPO algorithm with three reward variants (correct, random, inverted) to probe memorization exploitation
- **Clean Benchmark Generator**: Procedurally-generated arithmetic expressions ensuring zero data contamination
- **Activation Analysis**: Similarity metrics comparing responses between contaminated and clean benchmarks using ROUGE-L
- **Signal Detection Method**: Exploit-bias mechanism analysis via gradient tracing on high-probability tokens

## Implementation

### Contamination Detection via Partial-Prompt Reconstruction

Measure how well models reconstruct full benchmarks from incomplete prompts.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

class ContaminationDetector:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.model.eval()

    def truncate_prompt(self, problem_text, truncation_ratio=0.6):
        """
        Truncate problem to truncation_ratio of original length.
        E.g., 60% means first 60% of text shown, model reconstructs rest.

        Args:
            problem_text: Full problem text
            truncation_ratio: Fraction to keep (default 0.6 = 60%)

        Returns:
            truncated: Partial problem
            full: Full problem for comparison
        """
        tokens = problem_text.split()
        cutoff = int(len(tokens) * truncation_ratio)
        truncated = ' '.join(tokens[:cutoff])
        return truncated, problem_text

    def completion_reconstruction_rate(self, problems, truncation_ratio=0.6):
        """
        Measure ROUGE-L between model completions and ground truth.
        High scores indicate memorization; low scores indicate reasoning.

        Args:
            problems: List of problem texts
            truncation_ratio: How much of prompt to show

        Returns:
            mean_rouge: Average ROUGE-L score (0-1)
            per_problem_scores: Individual ROUGE-L for each problem
        """
        scores = []

        for problem in problems:
            truncated, full = self.truncate_prompt(problem, truncation_ratio)

            # Generate completion from truncated prompt
            input_ids = self.tokenizer.encode(truncated, return_tensors='pt')

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=len(self.tokenizer.encode(full)) + 50,
                    temperature=0.7,
                    top_p=0.9
                )

            completion = self.tokenizer.decode(output_ids[0])

            # Score reconstruction quality
            rouge_score = self.rouge.score(full, completion)['rougeL'].fmeasure

            scores.append(rouge_score)

        return sum(scores) / len(scores), scores

    def partial_prompt_answer_accuracy(self, qa_pairs, truncation_ratio=0.6):
        """
        Test if models recover correct answers from partial prompts.

        Args:
            qa_pairs: List of {'question': ..., 'answer': ...} dicts
            truncation_ratio: How much of question to show

        Returns:
            accuracy: Fraction of correct answers recovered
        """
        correct = 0

        for pair in qa_pairs:
            question = pair['question']
            ground_truth = pair['answer']

            truncated, _ = self.truncate_prompt(question, truncation_ratio)

            input_ids = self.tokenizer.encode(truncated, return_tensors='pt')

            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.0  # Greedy for answer extraction
                )

            generated = self.tokenizer.decode(output[0])

            # Simple answer matching (for math: check if final number appears)
            if ground_truth in generated:
                correct += 1

        return correct / len(qa_pairs)

# Usage: detect contamination in different models
detector_qwen = ContaminationDetector('Qwen/Qwen2.5-Math-7B')
detector_llama = ContaminationDetector('meta-llama/Llama-3.1-8B')

math_500_problems = [...]  # Load MATH-500 benchmark

# Qwen likely shows high reconstruction
qwen_rouge, _ = detector_qwen.completion_reconstruction_rate(math_500_problems)
print(f"Qwen reconstruction rate: {qwen_rouge:.2%}")  # ~54.6%

# Llama should show low reconstruction
llama_rouge, _ = detector_llama.completion_reconstruction_rate(math_500_problems)
print(f"Llama reconstruction rate: {llama_rouge:.2%}")  # ~3.8%
```

High ROUGE-L scores (>50%) indicate memorization; low scores (<10%) indicate reasoning.

### Clean Benchmark Generation: RandomCalculation Dataset

Generate arithmetic expressions with random operands, guaranteeing no pre-training contamination.

```python
import random
from typing import List, Tuple

class RandomCalculationDataset:
    def __init__(self, seed=42, min_steps=1, max_steps=20):
        random.seed(seed)
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.operators = ['+', '-', '*', '//']  # Integer operations

    def generate_expression(self, num_steps: int) -> Tuple[str, float]:
        """
        Generate random arithmetic expression with num_steps operations.

        Args:
            num_steps: Number of computation steps (1-20)

        Returns:
            expression: String like "5 + 3 * 2 - 1"
            answer: Computed result following operator precedence
        """
        # Start with random operand
        operands = [random.randint(1, 100) for _ in range(num_steps + 1)]
        operators = [random.choice(self.operators) for _ in range(num_steps)]

        # Construct expression
        expression = str(operands[0])
        for op, operand in zip(operators, operands[1:]):
            expression += f" {op} {operand}"

        # Evaluate with proper precedence
        try:
            answer = eval(expression)
        except ZeroDivisionError:
            # Retry if division by zero
            return self.generate_expression(num_steps)

        return expression, answer

    def create_qa_pairs(self, num_pairs: int = 1000) -> List[dict]:
        """
        Create QA dataset with guaranteed no pre-training contamination.

        Args:
            num_pairs: Number of QA pairs to generate

        Returns:
            qa_pairs: List of {'question': ..., 'answer': ...} dicts
        """
        qa_pairs = []

        for _ in range(num_pairs):
            num_steps = random.randint(self.min_steps, self.max_steps)
            expression, answer = self.generate_expression(num_steps)

            qa_pairs.append({
                'question': f"Calculate: {expression}",
                'answer': str(int(answer)),
                'steps': num_steps
            })

        return qa_pairs

# Create clean dataset
generator = RandomCalculationDataset(min_steps=1, max_steps=20)
clean_benchmark = generator.create_qa_pairs(num_pairs=500)

print(f"Generated {len(clean_benchmark)} clean problems")
print(f"Example: {clean_benchmark[0]}")
# Example: {'question': 'Calculate: 47 + 12 * 3 - 5', 'answer': '78', 'steps': 3}
```

### RL Training with Contamination Detection

Compare RL behavior on contaminated vs. clean benchmarks to isolate memorization exploitation.

```python
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.optim as optim

class RLBenchmarkComparison:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_reward(self, generated_answer, ground_truth, reward_type='correct'):
        """
        Three reward variants to probe memorization exploitation.

        Args:
            generated_answer: Model's generated answer
            ground_truth: Correct answer
            reward_type: 'correct' (ground truth), 'random' (50% prob), 'inverted' (wrong reward)

        Returns:
            reward: Scalar reward value
        """
        if reward_type == 'correct':
            # Standard reward: 1 if correct, 0 otherwise
            return 1.0 if generated_answer == ground_truth else 0.0

        elif reward_type == 'random':
            # Random reward: 50% chance of 1
            return 1.0 if random.random() < 0.5 else 0.0

        elif reward_type == 'inverted':
            # Inverted reward: 1 if WRONG
            return 0.0 if generated_answer == ground_truth else 1.0

    def evaluate_with_rewards(self, benchmark, reward_type='correct', num_samples=100):
        """
        Train model briefly on benchmark with specified reward type.
        Compare performance and answer similarity.

        Args:
            benchmark: QA dataset
            reward_type: Which reward variant to use
            num_samples: How many examples to use

        Returns:
            accuracy: Final accuracy on benchmark
            response_similarity: ROUGE-L of generated responses
        """
        self.model.train()

        optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=5e-7)

        responses = []
        total_loss = 0.0

        for i, qa_pair in enumerate(benchmark[:num_samples]):
            question = qa_pair['question']
            answer = qa_pair['answer']

            # Generate response
            input_ids = self.tokenizer.encode(question, return_tensors='pt')
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=20,
                    temperature=1.0
                )

            generated_text = self.tokenizer.decode(output_ids[0])
            responses.append(generated_text)

            # Compute reward
            reward = self.compute_reward(generated_text, answer, reward_type)

            # Simple reward-weighted loss (maximize reward)
            # In practice, use GRPO (Group Relative Policy Optimization)
            if reward > 0:
                loss = -torch.tensor(reward, dtype=torch.float32)
            else:
                loss = torch.tensor(1.0, dtype=torch.float32)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return {
            'average_loss': total_loss / num_samples,
            'responses': responses
        }

# Compare on contaminated vs. clean benchmarks
contaminated = math_500_problems[:100]
clean = generator.create_qa_pairs(100)

print("Testing with CORRECT rewards:")
contaminated_result = model_eval.evaluate_with_rewards(contaminated, reward_type='correct')
clean_result = model_eval.evaluate_with_rewards(clean, reward_type='correct')
print(f"  Contaminated: loss={contaminated_result['average_loss']:.4f}")
print(f"  Clean: loss={clean_result['average_loss']:.4f}")

print("\nTesting with RANDOM rewards (should fail on clean):")
contaminated_result = model_eval.evaluate_with_rewards(contaminated, reward_type='random')
clean_result = model_eval.evaluate_with_rewards(clean, reward_type='random')
print(f"  Contaminated: loss={contaminated_result['average_loss']:.4f}")
print(f"  Clean: loss={clean_result['average_loss']:.4f}")
```

If model benefits from random/inverted rewards on contaminated but not clean, memorization is the issue.

### Response Similarity Analysis

Measure ROUGE-L of model responses to detect memory vs. reasoning patterns.

```python
from rouge_score import rouge_scorer

def analyze_response_similarity(responses_contaminated, responses_clean):
    """
    Compare response patterns: high ROUGE on contaminated + low on clean = memorization.

    Args:
        responses_contaminated: Model outputs on contaminated benchmark
        responses_clean: Model outputs on clean benchmark

    Returns:
        contaminated_similarity: Average ROUGE-L between model responses
        clean_similarity: Average ROUGE-L between model responses
    """
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute_similarity(responses):
        if len(responses) < 2:
            return 0.0

        similarities = []
        for i in range(len(responses) - 1):
            score = rouge.score(responses[i], responses[i+1])['rougeL'].fmeasure
            similarities.append(score)

        return sum(similarities) / len(similarities)

    contam_sim = compute_similarity(responses_contaminated)
    clean_sim = compute_similarity(responses_clean)

    print(f"Response similarity on contaminated: {contam_sim:.3f}")
    print(f"Response similarity on clean: {clean_sim:.3f}")

    if contam_sim > 2 * clean_sim:
        print("MEMORIZATION DETECTED: Contaminated responses are too similar (retrieved, not generated)")
    else:
        print("Responses show genuine diversity (likely reasoning-based)")

    return contam_sim, clean_sim
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| ROUGE Truncation | 60% of prompt | Standard partial-prompt test; 40-60% range works |
| RL Learning Rate | 5e-7 | Critical; prevents reward exploitation without learning |
| Group Size | 16 samples per prompt | For GRPO algorithm |
| Batch Size | 128 | Standard for RL fine-tuning |
| Temperature | 1.0 | Exploration; 0 during greedy answer extraction |
| Random Reward Probability | 50% | γ = 0.5 in paper |
| Clean Dataset Size | 500-1000 problems | Adequate for detection; more data = higher confidence |
| Epsilon (GRPO clipping) | 0.20 | Controls exploitation bias |

### When to Use

- Validating RL training results on mathematical benchmarks before publication
- Checking if language model improvements are genuine reasoning or memorization
- Building evaluation frameworks that guarantee benchmark integrity
- Designing datasets for models that may have seen training data origins
- Comparing models where one might have contaminated pre-training

### When NOT to Use

- Real-time inference; contamination detection is offline validation
- Evaluating models on completely unseen domains (clean dataset is necessary)
- Scenarios where pre-training data is rigorously documented and available
- Quick prototyping; this adds validation overhead

### Common Pitfalls

- **Using single-prompt truncation**: One truncation point is not representative; test at 40%, 50%, 60% to get robust signal
- **Ignoring answer extraction quality**: Simple string matching misses near-correct answers; use semantic similarity for math
- **Conflating low ROUGE with reasoning**: Low ROUGE-L (<10%) is necessary but not sufficient; verify with clean benchmark comparison
- **Insufficient clean dataset size**: <100 problems is too small; generate at least 500 with varied difficulty for reliable signal
- **Applying uniform RL parameters**: Models that memorize respond differently to learning rates; tune per-model and measure reward responsiveness
- **Forgetting control: models**: Must test BOTH incorrect (random/inverted) AND correct rewards; correct rewards should improve on clean data

## Reference

Qian, L., Chen, X., Zhang, Y., et al. (2024). Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination. arXiv preprint arXiv:2507.10532.
