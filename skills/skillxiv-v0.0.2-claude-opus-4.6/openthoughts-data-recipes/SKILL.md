---
name: openthoughts-data-recipes
title: "OpenThoughts: Data Recipes for Reasoning Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.04178"
keywords: [data-curation, reasoning, distillation, mathematics, code-generation]
description: "Design data generation pipelines for reasoning models through systematic experimentation with answer sampling, teacher selection, and source quality optimization."
---

# OpenThoughts: Data Recipes for Reasoning Models

## Core Concept

OpenThoughts demonstrates that achieving state-of-the-art reasoning performance depends critically on data curation strategy rather than raw model capacity. Through over 1,000 controlled experiments, the researchers identified that answer diversity from a single high-quality teacher model outweighs source diversity, quality source selection beats mixing, and LLM-based filtering exceeds traditional methods.

## Architecture Overview

- **Question Sourcing**: Evaluate 27 code, 21 math, and 14 science benchmark sources
- **Mixing Strategy**: Select top 1-2 sources per domain based on performance correlation
- **Question Filtering**: Apply LLM-based difficulty and response-length scoring
- **Deduplication**: Exact deduplication with 16× answer multiplicity sampling
- **Answer Filtering**: Counterintuitively, keeping all teacher answers outperforms selective filtering
- **Teacher Model Selection**: QwQ-32B yields best-distilled 7B performance despite lower standalone scores

## Implementation

### Step 1: Source Evaluation Framework

```python
import numpy as np
from collections import defaultdict

class SourceEvaluator:
    def __init__(self, num_sources, domains=['math', 'code', 'science']):
        self.sources = {domain: [] for domain in domains}
        self.domain_configs = {
            'math': 27,      # 27 math sources tested
            'code': 27,      # 27 code sources tested
            'science': 14    # 14 science sources tested
        }

    def evaluate_source_quality(self, source_name, domain, sample_size=100):
        """Measure correlation with downstream task performance"""
        samples = self.sample_from_source(source_name, sample_size)

        # Train small model on samples from this source
        model = train_student_on_source(samples)

        # Evaluate on held-out reasoning benchmarks
        aime_score = evaluate_on_benchmark(model, 'AIME2025')
        livecodebench_score = evaluate_on_benchmark(model, 'LiveCodeBench')

        source_quality = {
            'name': source_name,
            'domain': domain,
            'aime_correlation': aime_score,
            'code_correlation': livecodebench_score,
            'combined_score': (aime_score + livecodebench_score) / 2
        }

        return source_quality

    def select_top_sources(self, domain, num_sources=2):
        """Keep only top 1-2 sources per domain"""
        evaluated = [self.evaluate_source_quality(src, domain)
                     for src in self.domain_configs[domain]]
        ranked = sorted(evaluated,
                       key=lambda x: x['combined_score'],
                       reverse=True)
        return [s['name'] for s in ranked[:num_sources]]
```

### Step 2: Answer Sampling Multiplicity

```python
class AnswerDiversityGenerator:
    def __init__(self, teacher_model_name, multiplicity=16):
        self.teacher = load_model(teacher_model_name)  # e.g., QwQ-32B
        self.multiplicity = multiplicity
        self.sampling_config = {
            'temperature': 1.0,
            'top_p': 0.95,
            'max_new_tokens': 2048,
        }

    def generate_multiple_answers(self, question, num_samples=16):
        """Generate diverse answers from same question via stochastic sampling"""
        answers = []

        for i in range(num_samples):
            # Use stochastic decoding for diversity
            answer = self.teacher.generate(
                question,
                temperature=self.sampling_config['temperature'],
                top_p=self.sampling_config['top_p'],
                do_sample=True,  # Critical for diversity
            )
            answers.append({
                'answer': answer,
                'sample_id': i,
                'question': question
            })

        return answers

    def create_dataset_via_multiplication(self, questions, multiplicity=16):
        """Scale dataset by generating multiple answers per question"""
        dataset = []

        for question in questions:
            answer_samples = self.generate_multiple_answers(
                question,
                num_samples=multiplicity
            )
            dataset.extend(answer_samples)

        print(f"Dataset scaled from {len(questions)} to {len(dataset)} samples")
        print(f"Scaling factor: {len(dataset) / len(questions)}×")

        return dataset
```

### Step 3: LLM-Based Question Filtering

```python
class LLMQuestionFilter:
    def __init__(self, scoring_model_name='GPT-4'):
        self.scorer = load_model(scoring_model_name)

    def compute_difficulty_score(self, question, ground_truth):
        """Use LLM to assess question difficulty"""
        prompt = f"""Rate the difficulty of this question on scale 1-10:
Question: {question}
Answer: {ground_truth}

Consider: problem complexity, reasoning depth, required knowledge."""

        difficulty_response = self.scorer.generate(prompt)
        difficulty = extract_numeric_score(difficulty_response)

        return difficulty  # 1=trivial, 10=hard

    def compute_response_length(self, answer):
        """Estimate expected solution complexity"""
        token_count = len(answer.split())
        return token_count

    def filter_by_lvm_criteria(self, questions_with_answers,
                               min_difficulty=3,
                               target_length_range=(100, 2000)):
        """Keep questions matching quality criteria"""
        filtered = []

        for qa_pair in questions_with_answers:
            difficulty = self.compute_difficulty_score(
                qa_pair['question'],
                qa_pair['answer']
            )
            length = self.compute_response_length(qa_pair['answer'])

            # LLM-based filtering outperforms embedding-based approaches
            if (min_difficulty <= difficulty and
                target_length_range[0] <= length <= target_length_range[1]):
                filtered.append(qa_pair)

        print(f"Filtered from {len(questions_with_answers)} to {len(filtered)} samples")
        return filtered
```

### Step 4: Teacher Model Selection Strategy

```python
class TeacherSelectionFramework:
    def __init__(self):
        self.candidate_teachers = [
            'QwQ-32B',           # Best distillation performance
            'DeepSeek-R1-Distill',
            'LLaMA-3.1-70B',
            'Mistral-Large',
        ]

    def evaluate_teacher_distillation(self, teacher_name, student_size='7B'):
        """Measure how well a teacher distills into 7B student"""

        # Generate data with teacher
        dataset = self.generate_reasoning_dataset_with_teacher(teacher_name)

        # Train student model on teacher data
        student = train_student_model(dataset, student_size)

        # Evaluate on multiple benchmarks
        aime_2025 = evaluate_on_dataset(student, 'AIME2025')
        livecodebench = evaluate_on_dataset(student, 'LiveCodeBench')

        distillation_quality = {
            'teacher': teacher_name,
            'aime_2025_score': aime_2025,
            'livecodebench_score': livecodebench,
            'avg_downstream_performance': (aime_2025 + livecodebench) / 2
        }

        return distillation_quality

    def select_best_teacher(self):
        """Key finding: teacher selection outweighs raw model performance"""
        results = []

        for teacher_name in self.candidate_teachers:
            result = self.evaluate_teacher_distillation(teacher_name)
            results.append(result)

        # QwQ-32B often outperforms stronger models like Claude-3-Opus
        best_teacher = max(results,
                          key=lambda x: x['avg_downstream_performance'])

        print(f"Best teacher for distillation: {best_teacher['teacher']}")
        return best_teacher['teacher']
```

### Step 5: Answer Filtering (Optional but Counterintuitive)

```python
class AnswerFilteringStrategy:
    def __init__(self):
        self.filter_config = {
            'apply_filtering': False,  # Key finding: no filtering > filtering
            'reason': 'Keeping all answers provides better training signal'
        }

    def analyze_filtering_impact(self, dataset_with_answers, dataset_no_filter):
        """Compare distilled model performance with and without answer filtering"""

        # Train student without filtering
        student_no_filter = train_student(dataset_with_answers)
        perf_no_filter = evaluate(student_no_filter)

        # Train student with filtering
        filtered_dataset = self.filter_low_quality_answers(dataset_with_answers)
        student_filtered = train_student(filtered_dataset)
        perf_filtered = evaluate(student_filtered)

        print(f"Performance without filtering: {perf_no_filter}")
        print(f"Performance with filtering: {perf_filtered}")

        # Empirical finding: no filtering wins
        return perf_no_filter >= perf_filtered
```

## Practical Guidance

1. **Source Concentration Strategy**: Focus on 1-2 highest-quality sources per domain rather than mixing 16 sources. Quality correlation with downstream tasks matters more than source diversity.

2. **Multiplicity Over Diversity**: Generate 16× answers from the same teacher and question distribution instead of seeking diverse question sources. Answer diversity at the token level provides better training signal.

3. **Teacher Selection Matters**: Benchmark multiple candidate teachers via distillation performance on downstream tasks, not just their standalone benchmark scores. QwQ-32B proved superior despite lower intrinsic performance on some metrics.

4. **LLM-Based Filtering**: Use LLM judges to score question difficulty and expected response length rather than embedding-based or statistical methods. This outperforms traditional filtering approaches.

5. **Keep All Answers**: Don't filter based on answer correctness or quality thresholds. The surprising finding is that keeping all teacher outputs (including wrong answers) provides a richer learning signal than keeping only high-confidence answers.

6. **Data Scaling Pattern**: Expect consistent improvements with data scaling across math, code, and science domains. The pipeline enables training 7B reasoning models to 53% AIME and 51% LiveCodeBench scores.

## Reference

- Paper: OpenThoughts (2506.04178)
- Result: OpenThinker3-7B with 1.2M training examples
- Key Metric: 53% on AIME 2025, 51% on LiveCodeBench
- Open Source: Released on Hugging Face (OpenThoughts3-1.2M dataset)
