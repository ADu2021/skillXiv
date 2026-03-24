---
name: humanomni-multimodal-reasoning
title: "HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.21277"
keywords: [Multimodal Reasoning, Context Understanding, Omni-modal Analysis, GRPO Training, Emotional Intelligence]
description: "Improve multimodal reasoning by requiring explicit context understanding before reasoning. Use specialized reward mechanisms and context-aware training to prevent information-skipping shortcuts."
---

# HumanOmniV2: Grounding Multimodal Reasoning Through Forced Context Understanding

Vision-language models struggle with complex multimodal understanding tasks. They can identify objects in images but fail at reasoning about context, emotions, and intentions—the subtleties that require genuinely understanding what they see rather than pattern-matching surface features. A common failure mode is "shortcutting": the model ignores crucial visual information and answers based on generic priors.

HumanOmniV2 tackles this by requiring the model to explicitly articulate its understanding of multimodal context before generating answers. Instead of directly answering questions, the model must first summarize what it observes in the images/video/audio, forcing genuine scene understanding. This prevents shortcutting and enables better downstream reasoning about human intentions and emotions.

## Core Concept

The key insight is that **explicit context verbalization prevents shortcutting**. Models that can skip context understanding and jump directly to answering often produce plausible-sounding but incorrect responses. By requiring:

1. `<context>` tag: Explicit summary of what the model observes
2. `<think>` tag: Reasoning process connecting context to answer
3. `<answer>` tag: Final response grounded in the reasoning

The model is forced to:
- Process all relevant visual information
- Connect observations to reasoning steps
- Justify answers with grounded logic

This three-stage output format, combined with specialized reward models that evaluate each stage, creates accountability at every step. Incorrect context gets penalized by the context reward, even if the final answer is plausible.

## Architecture Overview

HumanOmniV2 builds on multimodal large language models with:

- **Vision and Audio Encoders**: Process images, videos, and audio simultaneously
- **Three-Stage Output Format**: Context summary → Reasoning chain → Answer
- **Multi-reward System**: Separate rewards for context accuracy, reasoning quality, format compliance, and answer correctness
- **GRPO Training**: Group relative policy optimization with specialized reward masking for each output stage
- **IntentBench Benchmark**: New dataset emphasizing human intention and emotion understanding (24K training, 633 videos)

## Implementation

**Step 1: Create the three-stage output format framework**

Modify model generation to enforce the context-reason-answer structure.

```python
import torch
import torch.nn as nn
from typing import Tuple

class ThreeStageOutputFormatter:
    """
    Enforces and parses three-stage output format:
    <context>...context...</context>
    <think>...reasoning...</think>
    <answer>...final answer...</answer>
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Add special tokens if not present
        special_tokens = ['<context>', '</context>', '<think>', '</think>',
                         '<answer>', '</answer>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        self.context_start_id = self.tokenizer.encode('<context>')[1]
        self.context_end_id = self.tokenizer.encode('</context>')[1]
        self.think_start_id = self.tokenizer.encode('<think>')[1]
        self.think_end_id = self.tokenizer.encode('</think>')[1]
        self.answer_start_id = self.tokenizer.encode('<answer>')[1]
        self.answer_end_id = self.tokenizer.encode('</answer>')[1]

    def create_format_mask(self, seq_length, stage='context'):
        """
        Create a mask that enforces proper stage ordering.
        At step t, which tokens are valid?
        """
        mask = torch.zeros(seq_length, self.tokenizer.vocab_size, dtype=torch.bool)

        # Logic: enforce ordering of stages
        # This is a simplified version; real implementation uses constraints during generation
        return mask

    def parse_output(self, generated_text):
        """
        Extract context, reasoning, and answer from generated output.
        """
        try:
            # Extract context
            context_start = generated_text.find('<context>') + len('<context>')
            context_end = generated_text.find('</context>')
            context = generated_text[context_start:context_end].strip()

            # Extract thinking
            think_start = generated_text.find('<think>') + len('<think>')
            think_end = generated_text.find('</think>')
            thinking = generated_text[think_start:think_end].strip()

            # Extract answer
            answer_start = generated_text.find('<answer>') + len('<answer>')
            answer_end = generated_text.find('</answer>')
            answer = generated_text[answer_start:answer_end].strip()

            return {
                'context': context,
                'thinking': thinking,
                'answer': answer,
                'is_valid': all([context, thinking, answer])
            }
        except:
            return {
                'context': '',
                'thinking': '',
                'answer': '',
                'is_valid': False
            }

    def enforce_format_during_generation(self, model, prompt, max_length=512):
        """
        Generate with format enforcement: force proper stage transitions.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        current_stage = 'context'
        generated = []

        for step in range(max_length):
            # Get model prediction
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]

            # Apply format constraints
            # If in context stage, forbid think/answer tokens
            if current_stage == 'context':
                logits[0, self.think_start_id] = float('-inf')
                logits[0, self.answer_start_id] = float('-inf')

                # If we see </context>, transition
                if logits[0].argmax() == self.context_end_id:
                    current_stage = 'think'

            elif current_stage == 'think':
                logits[0, self.context_start_id] = float('-inf')
                logits[0, self.answer_start_id] = float('-inf')

                if logits[0].argmax() == self.think_end_id:
                    current_stage = 'answer'

            elif current_stage == 'answer':
                logits[0, self.context_start_id] = float('-inf')
                logits[0, self.think_start_id] = float('-inf')

            # Sample from constrained distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at </answer>
            if next_token.item() == self.answer_end_id:
                break

        return self.tokenizer.decode(generated)
```

**Step 2: Build multi-reward system for each output stage**

Create separate reward models that evaluate context, reasoning, and correctness independently.

```python
class MultiStageRewardSystem:
    """
    Evaluates each stage of the three-part output separately.
    Prevents any one stage from being ignored.
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Separate reward models for each stage
        self.context_reward_model = self._build_context_reward_model()
        self.reasoning_reward_model = self._build_reasoning_reward_model()
        self.answer_reward_model = self._build_answer_reward_model()
        self.format_reward_model = self._build_format_reward_model()

    def _build_context_reward_model(self):
        """
        Reward model that scores how well the context captures visual information.
        Trained on examples where models correctly vs incorrectly summarize images.
        """
        # In practice, this would be a fine-tuned RoBERTa or similar
        # For demonstration, we show the structure
        return ContextRewardClassifier()

    def _build_reasoning_reward_model(self):
        """Scores whether reasoning logically follows from context."""
        return ReasoningRewardClassifier()

    def _build_answer_reward_model(self):
        """Scores answer correctness against ground truth."""
        return AnswerRewardClassifier()

    def _build_format_reward_model(self):
        """Scores whether output follows proper format."""
        return FormatRewardClassifier()

    def compute_stage_rewards(self, parsed_output, ground_truth, image):
        """
        Compute rewards for each stage.
        Returns: {'context': score, 'reasoning': score, 'answer': score, 'format': score}
        """
        rewards = {}

        # Context reward: does the context accurately describe the image?
        # Compare against other valid context descriptions
        context_score = self.context_reward_model.score(
            parsed_output['context'],
            image,
            ground_truth.get('reference_context', '')
        )
        rewards['context'] = context_score

        # Reasoning reward: does reasoning logically follow?
        # Score based on consistency and completeness
        reasoning_score = self.reasoning_reward_model.score(
            parsed_output['context'],
            parsed_output['thinking'],
            ground_truth.get('reference_reasoning', '')
        )
        rewards['reasoning'] = reasoning_score

        # Answer reward: is the answer correct?
        answer_score = self.answer_reward_model.score(
            parsed_output['answer'],
            ground_truth['answer']
        )
        rewards['answer'] = answer_score

        # Format reward: does output follow format?
        format_score = float(parsed_output['is_valid'])
        rewards['format'] = format_score

        return rewards

    def compute_combined_reward(self, stage_rewards, weights=None):
        """
        Combine stage rewards with learnable weights.
        Default: equal weight to all stages.
        """
        if weights is None:
            weights = {'context': 0.25, 'reasoning': 0.25, 'answer': 0.35, 'format': 0.15}

        combined = sum(stage_rewards[stage] * weights[stage]
                      for stage in stage_rewards)
        return combined
```

**Step 3: Implement GRPO with causal masking for stage-specific training**

Train using GRPO but mask gradients so each stage is optimized for its corresponding reward.

```python
def grpo_training_with_stage_masking(model, tokenizer, training_data,
                                    reward_system, learning_rate=1e-5,
                                    num_steps=5000):
    """
    Train with GRPO where each output stage gets its own reward signal.
    Causal masking ensures context doesn't receive answer reward,
    and thinking doesn't receive format reward.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    formatter = ThreeStageOutputFormatter(tokenizer)

    for step in range(num_steps):
        # Sample batch
        batch = random.sample(training_data, batch_size=16)

        stage_losses = {'context': [], 'reasoning': [], 'answer': []}

        for example in batch:
            image = example['image']
            prompt = example['question']
            ground_truth = example['answer_data']  # context, reasoning, answer

            # Generate response
            input_ids = tokenizer.encode(prompt, return_tensors='pt')

            # Generate with format enforcement
            generated_text = formatter.enforce_format_during_generation(
                model, prompt, max_length=512
            )

            # Parse output
            parsed = formatter.parse_output(generated_text)

            # Skip if format is invalid
            if not parsed['is_valid']:
                continue

            # Compute rewards for each stage
            stage_rewards = reward_system.compute_stage_rewards(
                parsed, ground_truth, image
            )

            # Convert to advantage: reward - mean baseline
            baseline_reward = sum(stage_rewards.values()) / len(stage_rewards)

            # Compute log probability for each stage separately
            for stage_name in ['context', 'reasoning', 'answer']:
                # Extract the specific stage tokens
                start_token = formatter.__dict__[f'{stage_name}_start_id']
                end_token = formatter.__dict__[f'{stage_name}_end_id']

                # Find token positions for this stage
                start_pos = (input_ids == start_token).nonzero()[0, 0].item()
                end_pos = (input_ids == end_token).nonzero()[0, 0].item()

                # Compute log prob only for this stage
                # (don't apply reward from other stages to this stage)
                stage_ids = input_ids[:, start_pos:end_pos + 1]

                with torch.no_grad():
                    outputs = model(stage_ids)
                    logits = outputs.logits

                log_probs = F.log_softmax(logits, dim=-1)
                # Extract log probs for actual tokens
                actual_log_probs = log_probs[0, range(stage_ids.shape[1]), stage_ids[0]]

                # GRPO: advantage-weighted log probability
                advantage = stage_rewards[stage_name] - baseline_reward
                stage_loss = -(actual_log_probs.sum() * advantage)

                stage_losses[stage_name].append(stage_loss)

        # Combine losses
        total_loss = (sum(stage_losses['context']) +
                     sum(stage_losses['reasoning']) +
                     sum(stage_losses['answer'])) / (3 * len(batch))

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Loss {total_loss:.4f}")

    return model
```

**Step 4: Evaluate on IntentBench and general multimodal benchmarks**

Test on the IntentBench dataset (human intention/emotion understanding) and traditional benchmarks.

```python
def evaluate_omni_modal_reasoning(model, tokenizer, test_set_intentbench,
                                 test_set_general):
    """
    Evaluate on both IntentBench (human intention focus) and general benchmarks.
    Metrics: accuracy, context quality, reasoning quality.
    """
    formatter = ThreeStageOutputFormatter(tokenizer)

    results = {
        'intentbench': evaluate_on_benchmark(model, formatter, tokenizer,
                                            test_set_intentbench),
        'general': evaluate_on_benchmark(model, formatter, tokenizer,
                                        test_set_general)
    }

    return results

def evaluate_on_benchmark(model, formatter, tokenizer, test_set):
    """
    Evaluate on a specific benchmark.
    Metrics: answer accuracy, context quality, reasoning coherence.
    """
    correct_answers = 0
    context_quality_scores = []
    reasoning_quality_scores = []
    total = 0

    model.eval()

    for example in test_set:
        with torch.no_grad():
            prompt = example['prompt']
            image = example['image']
            ground_truth_answer = example['answer']

            # Generate
            generated = formatter.enforce_format_during_generation(
                model, prompt, max_length=512
            )

            parsed = formatter.parse_output(generated)

            # Evaluate answer correctness
            is_correct = (parsed['answer'].strip().lower() ==
                         ground_truth_answer.strip().lower())
            correct_answers += int(is_correct)

            # Evaluate context (does it capture key visual elements?)
            context_quality = evaluate_context_against_image(
                parsed['context'], image
            )
            context_quality_scores.append(context_quality)

            # Evaluate reasoning (coherent and grounded?)
            reasoning_quality = evaluate_reasoning_coherence(
                parsed['context'], parsed['thinking']
            )
            reasoning_quality_scores.append(reasoning_quality)

            total += 1

    return {
        'accuracy': correct_answers / total,
        'context_quality': np.mean(context_quality_scores),
        'reasoning_quality': np.mean(reasoning_quality_scores)
    }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| Context reward weight | 0.25 | Equal to reasoning; together 50% of total |
| Answer reward weight | 0.35 | Most important; correct answer is critical |
| Format reward weight | 0.15 | Enforces structure; prevents shortcutting |
| Stage masking | Strict | Never apply answer reward to context tokens |
| Training epochs | 3-5 | GRPO typically converges faster than SFT |
| RL batch size | 16-32 | Larger helps with advantage estimation stability |

**When to use HumanOmniV2 training:**
- You need models that deeply understand visual scenes (not just surface-level)
- You want to prevent shortcutting behavior
- Your task requires reasoning about human intentions, emotions, or context
- You have multimodal data (images, video, audio together)

**When NOT to use HumanOmniV2:**
- Your task is simple object recognition (overkill)
- You don't have explicit context/reasoning ground truth for training
- Inference latency matters (three-stage generation is slower)
- You only care about final answer correctness (context/reasoning not needed)

**Common pitfalls:**
- **Stage order breakdown**: If the model generates </context> before having said anything, the format mask isn't strong enough. Use strict token-level constraints, not just probability adjustment.
- **Context reward too lenient**: Models learn to write plausible but vague context. Use reference context from human annotations and penalize missing key visual elements.
- **Reasoning-answer misalignment**: If reasoning says X but answer says Y, both stage rewards should fail. Ensure reasoning_reward_model checks consistency.
- **Training instability with GRPO**: If advantage estimates vary wildly, increase batch size or reduce stage-specific reward scaling.

## Reference

HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context
https://arxiv.org/abs/2506.21277
