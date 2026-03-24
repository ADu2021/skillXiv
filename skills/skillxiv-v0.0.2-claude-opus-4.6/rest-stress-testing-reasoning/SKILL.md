---
name: rest-stress-testing-reasoning
title: "REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.10541"
keywords: [Model Evaluation, Stress Testing, Reasoning Models, Benchmark Design, Multi-Problem Prompts]
description: "Evaluate large reasoning models under stress by asking multiple problems simultaneously, revealing true multi-tasking capacity and context management. Use to identify robustness gaps invisible in single-problem evaluations and discriminate model capabilities beyond traditional benchmarks."
---

# REST: Stress Testing Reasoning Models Through Multi-Problem Evaluation

State-of-the-art reasoning models like DeepSeek-R1 achieve near-perfect scores on standard benchmarks, yet this masks a critical weakness: performance degrades significantly when answering multiple questions simultaneously. REST (Reasoning Extended Stress Testing) transforms existing benchmarks by concatenating multiple problems into single prompts, measuring how models handle reasoning under load. Results reveal that even top models experience 20-40% accuracy drops under stress, challenging assumptions that LLMs inherently function as multi-problem solvers.

The key insight is that benchmark scores measure single-problem performance; real-world applications require simultaneous reasoning over multiple tasks. By systematically increasing stress level (number of concurrent problems), REST discriminates models that truly maintain reasoning clarity from those that degrade under cognitive load.

## Core Concept

REST transforms any benchmark by creating stress-level-s prompts containing s consecutive questions concatenated into a single prompt. Each original problem appears exactly s times across the full prompt set but at different positions to prevent positional bias. This reveals three critical capabilities: context management (handling multiple reasoning threads), answer localization (retrieving answers to specific questions), and focus maintenance (preventing interference between problems).

The evaluation framework captures three phenomena: global performance degradation with stress level, position effects (earlier problems tend to score higher), and enhanced discrimination between models that appear similar on single-problem benchmarks.

## Architecture Overview

- **Benchmark Transformation Engine**: Converts standard benchmarks into multi-problem stress variants while controlling positional bias
- **Stress Level Controller**: Parametrizes number of concurrent problems (s=1 to s=12 depending on task difficulty)
- **Answer Extraction Module**: Handles both rule-based extraction (for marked formats like LaTeX \boxed{}) and LLM-based extraction (for unstructured answers)
- **Position Bias Mitigation**: Rotates question positions across prompt variants to isolate position effects from model capability
- **Multi-Model Evaluation Framework**: Tests 34 models across 7 benchmarks with consistent hyperparameter application

## Implementation

### Benchmark Transformation: Creating Stress-Level Prompts

Convert standard single-problem benchmarks into multi-problem stress variants.

```python
from typing import List, Tuple
import random

class RESTBenchmarkTransformer:
    """Transform single-problem benchmarks into multi-problem stress variants."""

    def __init__(self, problems: List[dict], seed: int = 42):
        """
        Args:
            problems: List of {'question': ..., 'answer': ...} dicts
            seed: For reproducible rotations
        """
        self.problems = problems
        self.num_problems = len(problems)
        random.seed(seed)

    def create_stress_prompts(self, stress_level: int) -> List[Tuple[str, List[str]]]:
        """
        Create stress_level-s prompt by concatenating s consecutive questions.

        Each problem appears exactly stress_level times across all prompts,
        at different positions to mitigate positional bias.

        Args:
            stress_level: Number of concurrent problems (s)

        Returns:
            List of (prompt_text, ground_truths) tuples
        """
        stress_prompts = []

        # Circular rotation: each set of s consecutive problems
        for i in range(self.num_problems):
            # Get s consecutive problems starting at index i
            problem_indices = [
                (i + j) % self.num_problems
                for j in range(stress_level)
            ]

            # Construct prompt with question labels
            prompt_parts = []
            ground_truths = []

            for q_num, prob_idx in enumerate(problem_indices):
                problem = self.problems[prob_idx]

                # Format: Q1: [question text], Q2: [question text], etc.
                question_text = f"Q{q_num + 1}: {problem['question']}"
                prompt_parts.append(question_text)
                ground_truths.append(problem['answer'])

            # Concatenate into single prompt with newlines for clarity
            prompt = "\n".join(prompt_parts)

            stress_prompts.append((prompt, ground_truths))

        return stress_prompts

    def analyze_position_effect(self, accuracy_by_position: dict) -> dict:
        """
        Analyze position bias: do earlier questions score higher?

        Args:
            accuracy_by_position: {position: accuracy} dict (0-indexed)

        Returns:
            Position effect analysis with trend
        """
        positions = sorted(accuracy_by_position.keys())
        accuracies = [accuracy_by_position[p] for p in positions]

        # Compute linear trend
        position_coeffs = list(range(len(positions)))
        mean_x = sum(position_coeffs) / len(position_coeffs)
        mean_y = sum(accuracies) / len(accuracies)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(position_coeffs, accuracies))
        denominator = sum((x - mean_x) ** 2 for x in position_coeffs)

        slope = numerator / denominator if denominator > 0 else 0

        return {
            'position_scores': accuracy_by_position,
            'trend_slope': slope,
            'first_position_accuracy': accuracies[0],
            'last_position_accuracy': accuracies[-1],
            'positional_bias_detected': slope < -0.05  # Negative slope = degradation
        }

# Example: transform MATH-500 benchmark
math_problems = [
    {'question': 'Calculate the area of a circle with radius 5.', 'answer': '78.54'},
    {'question': 'Solve: 2x + 3 = 11', 'answer': '4'},
    # ... more problems
]

transformer = RESTBenchmarkTransformer(math_problems)

# Create stress variants
stress_1_prompts = transformer.create_stress_prompts(stress_level=1)  # Single problems
stress_3_prompts = transformer.create_stress_prompts(stress_level=3)  # Three concurrent
stress_5_prompts = transformer.create_stress_prompts(stress_level=5)  # Five concurrent

print(f"Stress-1: {len(stress_1_prompts)} prompts")
print(f"Stress-3: {len(stress_3_prompts)} prompts")
print(f"Stress-5: {len(stress_5_prompts)} prompts")
print(f"\nExample stress-3 prompt:")
print(stress_3_prompts[0][0])
```

### Answer Extraction: Rule-Based and LLM-Based Methods

Extract answers from model outputs that may be structured or unstructured.

```python
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class AnswerExtractor:
    """Extract answers from model outputs using rule-based or LLM-based methods."""

    def __init__(self, use_llm_extraction: bool = False, model_name: str = 'Qwen/Qwen2.5-7B'):
        self.use_llm = use_llm_extraction

        if use_llm_extraction:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_rule_based(self, model_output: str, num_questions: int) -> List[str]:
        """
        Extract answers using regex patterns for marked formats.

        Supports:
        - LaTeX: \\boxed{answer}
        - Markdown: **answer**
        - Plain markers: Answer: ..., Final answer: ...

        Args:
            model_output: Raw model generation
            num_questions: Expected number of answers to extract

        Returns:
            List of extracted answers
        """
        answers = []

        # Pattern 1: LaTeX boxed format (most reliable)
        boxed_pattern = r'\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, model_output)

        if len(boxed_matches) >= num_questions:
            return boxed_matches[:num_questions]

        # Pattern 2: "Answer: X" or "Final answer: X"
        answer_pattern = r'(?:Answer|Final\s+answer|The\s+answer\s+is):\s*([^\n,]+)'
        answer_matches = re.findall(answer_pattern, model_output, re.IGNORECASE)

        if len(answer_matches) >= num_questions:
            return answer_matches[:num_questions]

        # Pattern 3: Q1: ... \n Answer: ..., Q2: ... \n Answer: ...
        q_pattern = r'Q\d+:.*?(?:Answer|Result):\s*([^\n]+)'
        q_matches = re.findall(q_pattern, model_output, re.DOTALL | re.IGNORECASE)

        if len(q_matches) >= num_questions:
            return q_matches[:num_questions]

        # Pattern 4: Numeric extraction for math problems
        # If all expected answers are numeric, extract all numbers
        numeric_pattern = r'-?\d+\.?\d*'
        numeric_matches = re.findall(numeric_pattern, model_output)

        if numeric_matches:
            return numeric_matches[:num_questions]

        # Fallback: couldn't extract
        return ['EXTRACTION_FAILED'] * num_questions

    def extract_llm_based(self, model_output: str, num_questions: int) -> List[str]:
        """
        Use LLM to extract answers from unstructured output.
        More robust but slower; use for edge cases.

        Args:
            model_output: Raw model generation
            num_questions: Expected number of answers

        Returns:
            List of extracted answers
        """
        extraction_prompt = f"""Extract the answers to the {num_questions} questions from this text:

{model_output}

Format your response as a list:
Answer 1: <answer>
Answer 2: <answer>
..."""

        input_ids = self.tokenizer.encode(extraction_prompt, return_tensors='pt')

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.0  # Greedy for consistency
            )

        extraction_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Parse extracted answers
        answer_pattern = r'Answer\s+\d+:\s*([^\n]+)'
        extracted = re.findall(answer_pattern, extraction_output)

        return extracted if extracted else ['EXTRACTION_FAILED'] * num_questions

    def extract_answers(self, model_output: str, num_questions: int) -> List[str]:
        """Choose extraction method based on configuration."""
        if self.use_llm:
            return self.extract_llm_based(model_output, num_questions)
        else:
            return self.extract_rule_based(model_output, num_questions)
```

### Stress Testing Evaluation Loop

Evaluate models across stress levels and analyze performance degradation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RESTEvaluator:
    """Evaluate models on REST (multi-problem stress testing)."""

    def __init__(self, model_name: str, benchmark_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.benchmark_name = benchmark_name
        self.extractor = AnswerExtractor(use_llm_extraction=False)

    def evaluate_stress_level(
        self,
        stress_prompts: List[Tuple[str, List[str]]],
        stress_level: int,
        max_tokens: int = 32000,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> dict:
        """
        Evaluate model on all prompts at given stress level.

        Args:
            stress_prompts: List of (prompt, ground_truths) from transformer
            stress_level: Current stress level (s)
            max_tokens: Max generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            results: {
                'accuracy': overall accuracy,
                'accuracy_by_position': {pos: acc},
                'outputs': list of model outputs
            }
        """
        self.model.eval()

        total_correct = 0
        total_predictions = 0
        accuracy_by_position = {}
        all_outputs = []

        for prompt, ground_truths in stress_prompts:
            # Generate response
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0)
                )

            model_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            all_outputs.append(model_output)

            # Extract answers
            extracted_answers = self.extractor.extract_answers(model_output, stress_level)

            # Compare with ground truths
            for position, (extracted, gt) in enumerate(zip(extracted_answers, ground_truths)):
                is_correct = extracted.strip() == gt.strip()

                if position not in accuracy_by_position:
                    accuracy_by_position[position] = {'correct': 0, 'total': 0}

                accuracy_by_position[position]['correct'] += int(is_correct)
                accuracy_by_position[position]['total'] += 1

                total_correct += int(is_correct)
                total_predictions += 1

        # Compute accuracies
        overall_accuracy = total_correct / total_predictions
        pos_accuracy = {
            pos: stats['correct'] / stats['total']
            for pos, stats in accuracy_by_position.items()
        }

        return {
            'stress_level': stress_level,
            'accuracy': overall_accuracy,
            'accuracy_by_position': pos_accuracy,
            'outputs': all_outputs
        }

    def run_stress_test(
        self,
        transformer: RESTBenchmarkTransformer,
        stress_levels: List[int]
    ) -> dict:
        """
        Run full stress test across multiple stress levels.

        Args:
            transformer: Benchmark transformer
            stress_levels: List of stress levels to evaluate (e.g., [1, 2, 3, 5])

        Returns:
            Full evaluation results
        """
        all_results = {}

        for stress_level in stress_levels:
            print(f"Evaluating stress level {stress_level}...")

            stress_prompts = transformer.create_stress_prompts(stress_level)
            results = self.evaluate_stress_level(stress_prompts, stress_level)

            all_results[stress_level] = results

            print(f"  Accuracy: {results['accuracy']:.3%}")
            print(f"  Position effect: ", end="")
            pos_accs = [results['accuracy_by_position'][i] for i in range(stress_level)]
            print(f"{[f'{a:.1%}' for a in pos_accs]}")

        return all_results

# Example usage
evaluator = RESTEvaluator('meta-llama/Llama-3.1-70B', 'MATH-500')

results = evaluator.run_stress_test(
    transformer=transformer,
    stress_levels=[1, 3, 5, 9, 12]
)
```

### Comparative Analysis Across Models

Compare performance degradation patterns between models.

```python
import matplotlib.pyplot as plt

def compare_stress_resilience(model_results: dict) -> dict:
    """
    Compare how different models handle stress levels.

    Args:
        model_results: {model_name: results_from_run_stress_test}

    Returns:
        Comparative analysis
    """
    stress_levels = sorted(list(model_results.values())[0].keys())
    model_names = list(model_results.keys())

    # Extract accuracies per stress level
    accuracy_curves = {}
    for model in model_names:
        accuracies = [
            model_results[model][s]['accuracy']
            for s in stress_levels
        ]
        accuracy_curves[model] = accuracies

    # Compute degradation metrics
    degradation_analysis = {}
    for model in model_names:
        baseline = accuracy_curves[model][0]  # Single-problem accuracy
        max_stress = accuracy_curves[model][-1]  # Highest-stress accuracy
        drop = baseline - max_stress
        drop_percent = drop / baseline

        degradation_analysis[model] = {
            'baseline': baseline,
            'max_stress': max_stress,
            'absolute_drop': drop,
            'percent_drop': drop_percent
        }

    # Identify resilient models (low degradation)
    sorted_by_resilience = sorted(
        degradation_analysis.items(),
        key=lambda x: x[1]['percent_drop']
    )

    return {
        'accuracy_curves': accuracy_curves,
        'degradation_analysis': degradation_analysis,
        'resilience_ranking': sorted_by_resilience,
        'stress_levels': stress_levels
    }

# Evaluate multiple models
results_deepseek = evaluator_deepseek.run_stress_test(transformer, [1, 2, 3, 5])
results_gpt4 = evaluator_gpt4.run_stress_test(transformer, [1, 2, 3, 5])
results_claude = evaluator_claude.run_stress_test(transformer, [1, 2, 3, 5])

comparison = compare_stress_resilience({
    'DeepSeek-R1': results_deepseek,
    'GPT-4': results_gpt4,
    'Claude-3.5': results_claude
})
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| GSM8K Stress Levels | 1, 3, 6, 9, 12 | Arithmetic problems allow high stress |
| MATH500/AMC23 Stress | 1, 3, 5, 7, 9 | Harder math; lower max stress |
| AIME/GPQA/CodeBench | 1, 2, 3, 4, 5 | Hardest problems; minimal stress levels |
| Max Output Tokens | 8K (standard), 32K (reasoning) | DeepSeek-R1, QwQ generate long solutions |
| Temperature | 0.7 (generation), 0.0 (extraction) | Higher for diverse outputs; lower for answer localization |
| Top_p | 0.95 | Typical nucleus sampling threshold |
| Position Control | Circular rotation | Every problem appears at every position |

### When to Use

- Evaluating reasoning model robustness beyond single-problem benchmarks
- Identifying models that fail under realistic multi-task scenarios
- Selecting models for production where simultaneous queries are expected
- Revealing performance gaps invisible in traditional benchmarks
- Analyzing position bias in long-context models

### When NOT to Use

- Benchmarks already designed for multi-problem evaluation
- Models that naturally handle streaming (where problems don't compete for context)
- Scenarios where single-problem accuracy is the only relevant metric
- Quick model selection where comprehensive evaluation is infeasible

### Common Pitfalls

- **Using inconsistent answer formats**: Regex extraction fails if model format changes per stress level; enforce consistent format in prompt instructions
- **Ignoring position effects**: Don't aggregate accuracy across positions without analyzing bias; report per-position results
- **Choosing inappropriate stress levels**: Using stress=12 for AIME causes majority extraction failures; match stress to task difficulty
- **Incomplete prompt rotation**: If not all circular rotations are tested, position bias won't be properly mitigated; test full rotation set
- **Over-tuning extraction patterns**: Regex patterns learned on one model may fail on another; validate extraction on multiple models and use LLM-based fallback
- **Neglecting token budget**: Reasoning models generate 10K+ tokens per problem; stress-5 can exceed context limits; monitor total output length

## Reference

Tian, Z., Chen, X., Li, Y., et al. (2024). REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once. arXiv preprint arXiv:2507.10541.
