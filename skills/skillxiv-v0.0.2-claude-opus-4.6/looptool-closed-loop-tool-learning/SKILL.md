---
name: looptool-closed-loop-tool-learning
title: "LoopTool: Closing Data-Training Loop for Robust LLM Tool Calls"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.09148"
keywords: [Tool Learning, Data Quality, Iterative Refinement, Self-Improvement, LLM Fine-tuning]
description: "Improve LLM tool-use through automated, closed-loop data curation combining capability probing, error correction, and targeted data expansion—enabling self-refining training pipelines that surpass larger models without expensive APIs."
---

# Refine Tool-Use Training Through Closed-Loop Data Curation

Traditional tool-learning relies on static data generation followed by separate training phases. This decoupled approach fails to address model-specific weaknesses and allows noisy annotations to persist. LoopTool implements a closed-loop framework integrating data synthesis with model training through three coordinated mechanisms: identifying capability gaps, correcting annotation errors, and generating targeted training samples.

The result is that an 8B parameter model trained with LoopTool surpasses 32B data generator models—demonstrating that data quality and curation matter more than quantity or oracle model size.

## Core Concept

LoopTool treats tool-learning as an iterative self-improvement process. Rather than one-shot data generation, the system continuously:

1. **Probes model capabilities** - Identifies which tools the model has mastered vs. where it fails
2. **Corrects annotations** - Removes noisy labels before they corrupt training
3. **Expands training data** - Generates additional samples targeting identified failure modes

This closed-loop approach operates within open-source ecosystems, eliminating dependence on expensive proprietary APIs and enabling efficient, adaptive data pipelines.

## Architecture Overview

- **Capability Probe Module (GCP)**: Tests model on diverse tool-use scenarios; categorizes as mastered/failing
- **Label Verification Module (JGLV)**: Uses open-source judge model to find and correct dataset errors
- **Data Expansion Module (EDDE)**: Generates new training samples addressing failure modes
- **Training Loop**: Fine-tunes model on curated, error-corrected data
- **Feedback Integration**: Iteratively refines data pipeline based on model improvement

## Implementation Steps

**Step 1: Greedy Capability Probing**

Systematically assess which tool-use patterns the model has learned.

```python
from typing import List, Dict, Tuple

class GreedyCapabilityProbe:
    """
    Identifies capabilities model has mastered vs. areas needing improvement.
    """

    def __init__(self, model, eval_dataset: List[Dict]):
        """
        Args:
            model: Language model to evaluate
            eval_dataset: Evaluation samples with tools and expected outputs
        """
        self.model = model
        self.eval_dataset = eval_dataset
        self.capabilities = {}

    def probe_capabilities(self) -> Dict[str, Tuple[float, List[Dict]]]:
        """
        Test model on diverse tool-use scenarios.

        Returns:
            capability_map: {tool_type: (accuracy, failed_examples)}
        """
        # Group evaluation samples by tool type
        tools_samples = {}
        for sample in self.eval_dataset:
            tool = sample.get('tool_type', 'generic')
            if tool not in tools_samples:
                tools_samples[tool] = []
            tools_samples[tool].append(sample)

        capability_map = {}

        for tool_type, samples in tools_samples.items():
            correct = 0
            failed_examples = []

            for sample in samples:
                # Generate tool use for this sample
                prompt = self._construct_tool_prompt(sample)
                response = self.model.generate(prompt, max_tokens=512)

                # Check if tool call is correct
                is_correct = self._validate_tool_call(response, sample)

                if is_correct:
                    correct += 1
                else:
                    failed_examples.append(sample)

            accuracy = correct / max(len(samples), 1)
            capability_map[tool_type] = (accuracy, failed_examples)

        self.capabilities = capability_map
        return capability_map

    def identify_gaps(self, threshold: float = 0.7) -> List[Tuple[str, List[Dict]]]:
        """
        Identify tool types where model underperforms.

        Args:
            threshold: Accuracy threshold for "mastered" classification

        Returns:
            gaps: [(tool_type, failed_examples), ...] for underperforming tools
        """
        gaps = []
        for tool, (accuracy, failed_examples) in self.capabilities.items():
            if accuracy < threshold:
                gaps.append((tool, failed_examples))
        return gaps

    def _construct_tool_prompt(self, sample: Dict) -> str:
        """Build prompt requesting tool invocation."""
        prompt = f"""Given this task, call the appropriate tool:

Task: {sample['task']}
Available tools: {sample.get('tools', [])}

Generate a valid tool call:"""
        return prompt

    def _validate_tool_call(self, response: str, sample: Dict) -> bool:
        """Check if generated tool call matches expected output."""
        expected = sample.get('expected_call', '')
        # Simple match; extend with semantic similarity
        return expected.strip() in response or \
               response.strip() == expected.strip()
```

**Step 2: Judgment-Guided Label Verification**

Use open-source judge model to identify and correct annotation errors.

```python
class JudgmentGuidedLabelVerification:
    """
    Identifies and corrects errors in training dataset labels.
    """

    def __init__(self, judge_model, train_dataset: List[Dict]):
        """
        Args:
            judge_model: Open-source model for verifying labels (e.g., GPT-J)
            train_dataset: Training samples with potential label errors
        """
        self.judge = judge_model
        self.train_dataset = train_dataset
        self.verification_results = []

    def verify_labels(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Verify all labels in training dataset.

        Returns:
            verified_data: Samples with corrected labels
            flagged_errors: Suspicious or corrected labels
        """
        verified_data = []
        flagged_errors = []

        for sample in self.train_dataset:
            # Ask judge to verify label
            is_correct, judgment = self._judge_sample(sample)

            if is_correct:
                verified_data.append(sample)
            else:
                # Try to correct the label
                corrected = self._correct_label(sample, judgment)
                if corrected:
                    verified_data.append(corrected)
                    flagged_errors.append({
                        'original': sample,
                        'corrected': corrected,
                        'judgment': judgment
                    })
                else:
                    # Discard if uncorrectable
                    flagged_errors.append({
                        'sample': sample,
                        'verdict': 'uncorrectable'
                    })

        return verified_data, flagged_errors

    def _judge_sample(self, sample: Dict) -> Tuple[bool, str]:
        """
        Have judge model assess label correctness.

        Args:
            sample: Training sample with task and expected tool call

        Returns:
            is_correct: Whether label seems valid
            judgment: Explanation from judge
        """
        prompt = f"""Verify this tool call is correct:

Task: {sample['task']}
Tool call: {sample['expected_call']}

Is this a valid and correct tool invocation? Explain:"""

        judgment = self.judge.generate(prompt, max_tokens=200)

        # Simple heuristic: judge says "correct" or "valid"
        is_correct = any(word in judgment.lower()
                        for word in ['correct', 'valid', 'appropriate'])

        return is_correct, judgment

    def _correct_label(self, sample: Dict, judgment: str) -> Dict:
        """
        Attempt to correct incorrect label.

        Args:
            sample: Original sample with wrong label
            judgment: Judge's assessment

        Returns:
            corrected_sample: Sample with corrected label, or None
        """
        # Use judge to generate correct version
        prompt = f"""Given this task, generate the correct tool call:

Task: {sample['task']}
Incorrect call: {sample['expected_call']}
Judge feedback: {judgment}

Provide the corrected tool call:"""

        corrected_call = self.judge.generate(prompt, max_tokens=200)

        if corrected_call and corrected_call != sample['expected_call']:
            corrected_sample = sample.copy()
            corrected_sample['expected_call'] = corrected_call
            corrected_sample['was_corrected'] = True
            return corrected_sample

        return None
```

**Step 3: Error-Driven Data Expansion**

Generate new training samples targeting identified failure modes.

```python
class ErrorDrivenDataExpansion:
    """
    Generates additional training data for identified failure modes.
    """

    def __init__(self, generator_model, capability_gaps: List[Tuple[str, List[Dict]]]):
        """
        Args:
            generator_model: Open-source LLM for synthetic data generation
            capability_gaps: Output from GreedyCapabilityProbe.identify_gaps()
        """
        self.generator = generator_model
        self.capability_gaps = capability_gaps

    def expand_training_data(self, expansion_factor: int = 2) -> List[Dict]:
        """
        Generate new samples targeting failure modes.

        Args:
            expansion_factor: How many new samples per gap (e.g., 2x original)

        Returns:
            new_samples: Synthetically generated training data
        """
        new_samples = []

        for tool_type, failed_examples in self.capability_gaps:
            # Analyze failure patterns
            failure_patterns = self._analyze_failures(failed_examples)

            # Generate new samples avoiding failure patterns
            for pattern in failure_patterns:
                num_new = len(failed_examples) * expansion_factor

                generated = self._generate_targeted_samples(
                    tool_type, pattern, num_new
                )
                new_samples.extend(generated)

        return new_samples

    def _analyze_failures(self, failed_examples: List[Dict]) -> List[str]:
        """
        Identify common patterns in failures.

        Args:
            failed_examples: Samples where model failed

        Returns:
            patterns: Common failure characteristics
        """
        patterns = []

        # Simple pattern analysis
        avg_length = sum(len(e['task'].split())
                        for e in failed_examples) / len(failed_examples)

        if avg_length > 100:
            patterns.append('long_tasks')
        if avg_length < 20:
            patterns.append('short_tasks')

        # Parameter count pattern
        param_counts = [len(e.get('parameters', [])) for e in failed_examples]
        if sum(param_counts) / len(param_counts) > 5:
            patterns.append('complex_parameters')

        return patterns if patterns else ['general']

    def _generate_targeted_samples(self, tool_type: str, failure_pattern: str,
                                   num_samples: int) -> List[Dict]:
        """
        Generate samples addressing specific failure pattern.

        Args:
            tool_type: Tool type to generate for
            failure_pattern: Pattern to target ('long_tasks', etc.)
            num_samples: Number to generate

        Returns:
            samples: Generated training samples
        """
        generated = []

        for _ in range(num_samples):
            # Build generation prompt targeting pattern
            if failure_pattern == 'long_tasks':
                task_template = "Create a complex {tool_type} task with multiple steps"
            elif failure_pattern == 'short_tasks':
                task_template = "Create a simple {tool_type} task"
            elif failure_pattern == 'complex_parameters':
                task_template = "Create a {tool_type} task with many parameters"
            else:
                task_template = "Create a {tool_type} task"

            prompt = f"""Generate a training example for {tool_type}:
{task_template}

Provide:
1. task: Clear task description
2. expected_call: Correct tool invocation
3. parameters: Any tool parameters needed"""

            response = self.generator.generate(prompt, max_tokens=300)

            # Parse response into sample
            sample = self._parse_generated_sample(response, tool_type)
            if sample:
                generated.append(sample)

        return generated

    def _parse_generated_sample(self, response: str, tool_type: str) -> Dict:
        """Parse generator output into sample format."""
        # Simple parsing; extend with structured generation
        lines = response.split('\n')
        task = next((l for l in lines if 'task:' in l.lower()), '')
        call = next((l for l in lines if 'call:' in l.lower()), '')

        if task and call:
            return {
                'task': task,
                'expected_call': call,
                'tool_type': tool_type,
                'synthetic': True
            }
        return None
```

**Step 4: Integrated Training Loop**

Combine capability probing, label verification, and data expansion into training pipeline.

```python
def train_with_closed_loop_curation(
        model, initial_dataset: List[Dict],
        num_iterations: int = 3,
        expansion_factor: int = 2):
    """
    Main training loop integrating LoopTool components.

    Args:
        model: LLM to train
        initial_dataset: Starting training data
        num_iterations: Closed-loop iterations
        expansion_factor: Data expansion multiplier

    Returns:
        trained_model: Model after closed-loop training
    """
    import torch.optim as optim

    current_dataset = initial_dataset.copy()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    for iteration in range(num_iterations):
        print(f"Closed-loop iteration {iteration + 1}/{num_iterations}")

        # Step 1: Verify labels in current dataset
        verifier = JudgmentGuidedLabelVerification(model, current_dataset)
        verified_data, errors = verifier.verify_labels()
        print(f"  Verified {len(verified_data)} samples, "
              f"corrected {len(errors)} errors")

        # Step 2: Probe capabilities on verified data
        probe = GreedyCapabilityProbe(model, verified_data)
        capabilities = probe.probe_capabilities()
        gaps = probe.identify_gaps(threshold=0.8)

        if gaps:
            print(f"  Found {len(gaps)} capability gaps")

            # Step 3: Expand data targeting gaps
            expander = ErrorDrivenDataExpansion(model, gaps)
            new_samples = expander.expand_training_data(expansion_factor)
            print(f"  Generated {len(new_samples)} new samples")

            # Merge new samples with verified data
            current_dataset = verified_data + new_samples
        else:
            print("  No capability gaps detected; training complete")
            current_dataset = verified_data
            break

        # Step 4: Fine-tune on improved dataset
        for epoch in range(2):
            total_loss = 0
            for batch_idx in range(0, len(current_dataset), 32):
                batch = current_dataset[batch_idx:batch_idx + 32]

                # Forward pass
                for sample in batch:
                    prompt = f"Task: {sample['task']}\nGenerate tool call:"
                    logits = model.forward(prompt)
                    target = sample['expected_call']
                    loss = compute_tool_loss(logits, target)
                    total_loss += loss.item()

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f"    Epoch {epoch}: Loss {total_loss / len(current_dataset):.4f}")

    return model
```

## Practical Guidance

**When to Use LoopTool:**
- Tool-learning pipelines with data quality concerns
- Scenarios with budget constraints (avoid expensive oracle APIs)
- Tasks where capability gaps can be precisely identified

**When NOT to Use:**
- Tasks with already high-quality annotated data
- Domains requiring human judgment for correctness (judge model insufficient)
- Real-time training pipelines (iteration overhead may be prohibitive)

**Hyperparameters and Configuration:**
- Capability threshold: 0.7-0.8 (balance between being aggressive and conservative)
- Expansion factor: 1.5-3.0 (generate 1.5-3x more samples for gaps)
- Verification sampling: Check all samples in early iterations; sample 10-20% in later ones
- Judge model: Use local, open-source models (Mistral 7B, Llama 2); avoid API calls

**Pitfalls to Avoid:**
1. **Judge model bias** - Judge might systematically favor certain patterns; validate corrections manually
2. **Data drift** - Generated samples might differ from real distribution; monitor diversity
3. **Iteration fatigue** - Diminishing returns after 2-3 iterations; use early stopping
4. **Over-correction** - Judge might over-correct nuanced labels; use confidence scores to filter

---

Reference: https://arxiv.org/abs/2511.09148
