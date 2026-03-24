---
name: compass-judger-reward-based-evaluation
title: "CompassJudger-2: Towards Generalist Judge Model via Verifiable Rewards"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.09104"
keywords: [LLM Evaluation, Judge Models, Verifiable Rewards, Policy Gradient, Model Comparison]
description: "Build generalist judge models for evaluating LLM outputs using verifiable rewards and policy gradient training. Create a 7B model competitive with much larger judges through reward-guided optimization and critical thinking decomposition. Use when you need reliable automated evaluation of model outputs across diverse tasks."
---

# CompassJudger-2: Generalist Judge Models via Verifiable Rewards

Evaluating LLM outputs at scale requires automated judge models that can compare generations and rank quality. Existing approaches rely purely on supervised fine-tuning, limiting learning signal to labeled examples. CompassJudger-2 introduces verifiable rewards—binary signals indicating correctness at designated positions—combined with policy gradient optimization to train more capable judges. The 7B variant achieves competitive performance with significantly larger judge models through this hybrid training approach.

The key insight is that judge models benefit from two signals: supervision through correct answer/judgment examples, and reinforcement through explicitly-marked rewards at decision points. By decomposing judgment into five reasoning steps (demand analysis, strength/weakness identification, reasoning, prediction) and using rejection sampling for hard negatives, the model learns robust evaluation capabilities with minimal data.

## Core Concept

CompassJudger-2 trains judges through a three-phase pipeline: (1) curate task-driven evaluation data with ground truth answers, (2) decompose judgment into structured reasoning steps via chain-of-thought supervision, (3) apply policy gradient optimization to reinforce correct predictions at designated positions. The verifiable reward signal is crucial: rather than comparing full outputs subjectively, rewards mark specific positions where correctness is verifiable (e.g., "is this the correct answer?").

This allows the model to learn what patterns correlate with correctness, not just memorize example judgments. The approach is agnostic to base model, requiring only supervised fine-tuning followed by policy gradient training without architectural changes.

## Architecture Overview

- **Data Curation Pipeline**: Reconstructs outdated judge data using stronger models, enhances diversity via prompt templates, synthesizes knowledge-based evaluations
- **Critical Thinking Framework**: Decomposes judgment into five steps: user demand analysis, identification of response strengths/weaknesses, reasoning about quality, and final prediction
- **Verifiable Reward System**: Rule-based binary signals (1 if correct at designated position, 0 otherwise) enabling clear optimization targets
- **Policy Gradient Optimization**: Combines SFT loss with margin policy loss using rejection sampling for contrastive negatives
- **Rejection Sampling Module**: Generates multiple candidate responses and filters for ground truth correctness as training negatives

## Implementation

### Data Curation and Synthesis

Build diverse evaluation datasets with verifiable ground truths.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class JudgeCurationPipeline:
    """Curate and synthesize judge training data."""

    def __init__(self, curator_model_name: str = "Qwen/Qwen2.5-72B"):
        self.curator = AutoModelForCausalLM.from_pretrained(curator_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(curator_model_name)
        self.curator.eval()

    def reconstruct_outdated_data(self, old_examples: list) -> list:
        """
        Reconstruct outdated evaluation examples using a stronger model.

        Args:
            old_examples: List of {'input': ..., 'answer': ...} pairs

        Returns:
            Reconstructed examples with modern quality standards
        """
        reconstructed = []

        for example in old_examples:
            # Use curator model to re-evaluate old examples
            prompt = f"""Given this task:
{example['input']}

Provide a comprehensive evaluation. Consider:
1. Whether the previous answer was correct
2. What a better answer would be
3. Key quality metrics

Your evaluation:"""

            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                output_ids = self.curator.generate(
                    input_ids,
                    max_new_tokens=300,
                    temperature=0.7
                )

            evaluation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            reconstructed.append({
                'original_input': example['input'],
                'original_answer': example.get('answer'),
                'modern_evaluation': evaluation
            })

        return reconstructed

    def enhance_diversity_with_templates(self, base_examples: list) -> list:
        """
        Create diverse evaluation variants using prompt templates.

        Args:
            base_examples: Original evaluation examples

        Returns:
            Expanded dataset with template variations
        """
        templates = [
            "Evaluate this response for correctness: {response}",
            "Is this answer accurate? {response}",
            "Rate the quality of this response: {response}",
            "What is wrong with this answer? {response}",
            "How would you improve this response? {response}"
        ]

        expanded = []

        for example in base_examples:
            for template in templates:
                variant = example.copy()
                variant['instruction'] = template.format(response=example.get('response', ''))
                expanded.append(variant)

        return expanded

    def synthesize_knowledge_evaluations(self, domains: list, num_per_domain: int = 50) -> list:
        """
        Synthesize evaluation examples with ground truth answers.

        Args:
            domains: List of domains (math, science, coding, etc.)
            num_per_domain: Examples per domain

        Returns:
            Synthetic evaluation dataset with verifiable answers
        """
        synthetic = []

        for domain in domains:
            prompt = f"""Generate {num_per_domain} evaluation examples for the {domain} domain.
For each example, provide:
1. A question or task
2. A response to evaluate
3. Whether it's correct (yes/no)
4. Why it's correct or incorrect

Format:
Example 1:
Question: [question]
Response: [response]
Correct: [yes/no]
Reasoning: [reasoning]"""

            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                output_ids = self.curator.generate(
                    input_ids,
                    max_new_tokens=2000,
                    temperature=0.8
                )

            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Parse generated examples
            # (simplified: in practice use robust parsing)
            synthetic.append({
                'domain': domain,
                'examples': generated
            })

        return synthetic

class VerifiableRewardSystem:
    """Generate verifiable reward signals for judge training."""

    @staticmethod
    def generate_reward_signal(
        response: str,
        ground_truth: str,
        task_type: str = 'qa'
    ) -> int:
        """
        Create binary reward signal (1 if correct, 0 if wrong).

        Args:
            response: Model's response
            ground_truth: Correct answer/judgment
            task_type: Task category (qa, classification, etc.)

        Returns:
            Reward: 1 if correct, 0 if incorrect
        """
        if task_type == 'qa':
            # Exact match or semantic similarity for QA
            return 1 if response.strip() == ground_truth.strip() else 0

        elif task_type == 'classification':
            # Extract predicted class and compare
            predicted_class = response.split()[-1] if response else ""
            return 1 if predicted_class == ground_truth else 0

        elif task_type == 'scoring':
            # Numeric scoring: reward if within threshold
            try:
                pred_score = float(response.strip())
                true_score = float(ground_truth.strip())
                return 1 if abs(pred_score - true_score) < 0.5 else 0
            except:
                return 0

        return 0

    @staticmethod
    def mark_verifiable_positions(full_text: str, answer_position: int) -> str:
        """
        Mark specific positions in text where rewards are verifiable.

        Args:
            full_text: Complete judge response
            answer_position: Character offset of final answer

        Returns:
            Marked text with reward positions indicated
        """
        # Insert marker before answer
        marked = full_text[:answer_position] + "[REWARD_POSITION]" + full_text[answer_position:]
        return marked
```

### Critical Thinking Decomposition Framework

Structure judgment into five interpretable reasoning steps.

```python
class CriticalThinkingFramework:
    """Decompose judgment into structured reasoning steps."""

    @staticmethod
    def construct_judgment_prompt(
        task: str,
        response: str
    ) -> str:
        """
        Create prompt that decomposes judgment into five steps.

        Args:
            task: Original task/question
            response: Response to evaluate

        Returns:
            Structured prompt guiding judgment decomposition
        """
        prompt = f"""Evaluate this response using critical thinking:

Task: {task}
Response: {response}

Please analyze in five steps:

1. USER DEMAND ANALYSIS: What is the user really asking for? What are the requirements?
   [Your analysis]

2. RESPONSE STRENGTHS: What aspects of this response are good or correct?
   [Your analysis]

3. RESPONSE WEAKNESSES: What aspects are lacking, incorrect, or could be improved?
   [Your analysis]

4. REASONING: Weighing strengths vs. weaknesses, what is the overall quality?
   [Your analysis]

5. FINAL JUDGMENT: Is this response satisfactory? Yes or No.
   [Your judgment]"""

        return prompt

    @staticmethod
    def extract_final_judgment(model_output: str) -> str:
        """
        Extract final judgment (Yes/No) from model output.

        Args:
            model_output: Full model generation with five reasoning steps

        Returns:
            Final judgment ("Yes" or "No")
        """
        lines = model_output.split('\n')

        # Find line starting with "5. FINAL JUDGMENT"
        for i, line in enumerate(lines):
            if '5. FINAL JUDGMENT' in line or 'FINAL JUDGMENT' in line:
                # Next non-empty line is the judgment
                for j in range(i + 1, len(lines)):
                    answer_line = lines[j].strip()
                    if answer_line:
                        # Extract yes/no
                        if 'yes' in answer_line.lower():
                            return 'Yes'
                        elif 'no' in answer_line.lower():
                            return 'No'

        return 'Unknown'
```

### Policy Gradient Optimization with Rejection Sampling

Train with both supervised loss and reinforcement signals.

```python
import torch.nn.functional as F
from torch.optim import AdamW

class JudgeTrainer:
    """Train judge models with SFT + policy gradient."""

    def __init__(self, model_name: str, base_lr: float = 6e-5):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=base_lr)
        self.base_lr = base_lr

    def rejection_sample_negatives(
        self,
        prompt: str,
        num_candidates: int = 8,
        ground_truth_label: str = 'Yes'
    ) -> list:
        """
        Generate multiple responses and filter for correctness diversity.

        Args:
            prompt: Judge prompt
            num_candidates: Number of responses to generate
            ground_truth_label: Correct answer to filter for

        Returns:
            List of (response, is_correct) tuples
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        responses = []
        for _ in range(num_candidates):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True
                )

            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            judgment = CriticalThinkingFramework.extract_final_judgment(response)
            is_correct = (judgment == ground_truth_label)

            responses.append((response, is_correct))

        return responses

    def compute_margin_loss(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        gamma: float = 10.0
    ) -> torch.Tensor:
        """
        Compute margin policy loss pushing positive away from negative.

        Loss = max(0, γ - (logits_pos - logits_neg))

        Args:
            positive_logits: Log probabilities of correct responses
            negative_logits: Log probabilities of incorrect responses
            gamma: Margin parameter

        Returns:
            Margin loss value
        """
        margin = positive_logits - negative_logits
        loss = F.relu(gamma - margin)
        return loss.mean()

    def training_step(
        self,
        prompt: str,
        ground_truth_label: str,
        correct_response: str = None,
        lambda_sft: float = 0.5,
        lambda_rl: float = 0.5,
        margin_gamma: float = 10.0
    ) -> dict:
        """
        Single training step combining SFT and policy gradient.

        Args:
            prompt: Judge prompt
            ground_truth_label: Correct judgment
            correct_response: Reference correct response (optional)
            lambda_sft: Weight for SFT loss
            lambda_rl: Weight for RL loss
            margin_gamma: Margin parameter

        Returns:
            Loss dictionary with individual components
        """
        self.model.train()

        # SFT Loss: standard language modeling on correct response
        sft_loss = None
        if correct_response:
            input_ids = self.tokenizer.encode(
                prompt + correct_response,
                return_tensors='pt'
            )
            labels = input_ids.clone()
            # Only compute loss on response part
            prompt_len = len(self.tokenizer.encode(prompt))
            labels[:, :prompt_len] = -100  # Ignore prompt tokens

            outputs = self.model(input_ids, labels=labels)
            sft_loss = outputs.loss

        # Rejection Sampling: generate correct and incorrect responses
        sampled = self.rejection_sample_negatives(
            prompt,
            num_candidates=8,
            ground_truth_label=ground_truth_label
        )

        positive_responses = [r for r, correct in sampled if correct]
        negative_responses = [r for r, correct in sampled if not correct]

        # Policy Gradient Loss: margin between positive and negative
        rl_loss = None
        if positive_responses and negative_responses:
            pos_response = positive_responses[0]
            neg_response = negative_responses[0]

            # Compute log probabilities
            pos_ids = self.tokenizer.encode(prompt + pos_response, return_tensors='pt')
            neg_ids = self.tokenizer.encode(prompt + neg_response, return_tensors='pt')

            with torch.no_grad():
                pos_logits = self.model(pos_ids).logits
                neg_logits = self.model(neg_ids).logits

            # Margin loss
            pos_score = pos_logits.sum()  # Simplified
            neg_score = neg_logits.sum()

            rl_loss = self.compute_margin_loss(pos_score, neg_score, margin_gamma)

        # Combined loss
        total_loss = 0.0
        if sft_loss is not None:
            total_loss += lambda_sft * sft_loss
        if rl_loss is not None:
            total_loss += lambda_rl * rl_loss

        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'sft_loss': sft_loss.item() if sft_loss else None,
            'rl_loss': rl_loss.item() if rl_loss else None
        }

    def train_epoch(self, training_data: list, epochs: int = 1):
        """Train for multiple epochs."""
        for epoch in range(epochs):
            total_loss = 0.0

            for item in training_data:
                loss_dict = self.training_step(
                    prompt=item['prompt'],
                    ground_truth_label=item['label'],
                    correct_response=item.get('correct_response')
                )

                total_loss += loss_dict['total_loss']

            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
```

### Evaluation of Judge Model

Test judge accuracy across diverse benchmarks.

```python
class JudgeEvaluator:
    """Evaluate judge model accuracy on evaluation benchmarks."""

    def __init__(self, judge_model_name: str):
        self.judge = AutoModelForCausalLM.from_pretrained(judge_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        self.judge.eval()

    def evaluate_on_benchmark(
        self,
        benchmark: list,
        num_samples: int = None
    ) -> dict:
        """
        Evaluate judge on benchmark dataset.

        Args:
            benchmark: List of {'task': ..., 'response': ..., 'label': ...}
            num_samples: Limit evaluation to N samples

        Returns:
            Accuracy and detailed results
        """
        if num_samples:
            benchmark = benchmark[:num_samples]

        correct = 0
        total = len(benchmark)

        for item in benchmark:
            prompt = CriticalThinkingFramework.construct_judgment_prompt(
                item['task'],
                item['response']
            )

            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                output_ids = self.judge.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.0
                )

            judgment = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            final_answer = CriticalThinkingFramework.extract_final_judgment(judgment)

            is_correct = (final_answer == item['label'])
            correct += int(is_correct)

        accuracy = correct / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Learning Rate | 6e-5 | Standard for fine-tuning 7B models |
| Batch Size | 512 | During SFT phase |
| Rejection Sampling Candidates | 8 | Balance diversity and computation |
| Margin Parameter γ | 10.0 | Controls positive/negative separation |
| SFT Loss Weight λ_sft | 0.5 | Balance with RL component |
| RL Loss Weight λ_rl | 0.5 | Equal weight to SFT initially |
| Training Epochs | 1 | Single pass sufficient with hard negatives |
| Temperature (inference) | 0.0 | Greedy decoding for consistency |

### When to Use

- Building automated judges for LLM output evaluation
- Comparing generations when human evaluation is infeasible
- Training reward models for RLHF pipelines
- Creating task-specific evaluation models without fine-tuning from scratch
- Building multi-domain evaluation systems

### When NOT to Use

- Tasks requiring nuanced human judgment (no clear ground truth)
- Fine-grained quality assessment (judge learns binary correctness)
- Domains where ground truth is ambiguous or subjective
- Real-time inference under strict latency constraints

### Common Pitfalls

- **Skipping data curation**: Using only old or narrow training data limits generalization; invest in diverse curation
- **Weak ground truth labels**: Incorrect labels in training data degrade judge performance; validate label quality
- **Ignoring rejection sampling size**: Too few candidates (M=4) limits diversity; use M≥8 for robust negatives
- **Static margin parameter**: Margin γ=10 works broadly but may need adjustment per domain; try 5-20 range
- **Insufficient reasoning decomposition**: Using only final answer without intermediate steps limits learning; maintain five-step structure
- **Forgetting base model choice**: Judge quality depends heavily on base model (7B vs 32B); larger base = better judge performance

## Reference

Chen, X., Liu, S., Wang, Y., et al. (2024). CompassJudger-2: Towards Generalist Judge Model via Verifiable Rewards. arXiv preprint arXiv:2507.09104.
