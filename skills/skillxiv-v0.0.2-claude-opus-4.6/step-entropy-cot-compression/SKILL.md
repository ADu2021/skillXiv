---
name: step-entropy-cot-compression
title: Step Entropy - Compressing Chain-of-Thought via Entropy
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03346
keywords: [chain-of-thought, reasoning, compression, inference-efficiency, entropy]
description: "Reduces chain-of-thought verbosity through step entropy metrics that identify and prune low-information reasoning steps while maintaining accuracy."
---

## Step Entropy: Compressing Chain-of-Thought via Step Entropy

### Core Concept

Step Entropy provides a metric to measure the informational contribution of individual reasoning steps in chain-of-thought (CoT) reasoning. By identifying low-entropy steps that contribute minimally to the final answer, this technique enables automatic compression of verbose reasoning chains while preserving reasoning quality and accuracy.

### Architecture Overview

- **Step Entropy Computation**: Quantify information contribution of each reasoning step
- **Low-Entropy Identification**: Detect redundant steps that can be pruned
- **Supervised Fine-Tuning**: Train models with original reasoning chains
- **Reinforcement Learning**: Optimize models to autonomously generate compressed reasoning via [SKIP] tokens
- **Two-Stage Training**: Combine SFT and Group Relative Policy Optimization (GRPO)

### Implementation Steps

**Step 1: Compute Step Entropy Metrics**

Analyze the information content of reasoning steps:

```python
# Pseudocode for step entropy computation
class StepEntropyAnalyzer:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def compute_step_entropy(self, reasoning_chain, answer, model):
        """
        Measure informational contribution of each step.

        Args:
            reasoning_chain: Full CoT with numbered steps
            answer: Final answer
            model: Language model for probability computation

        Returns:
            step_entropies: (num_steps,) entropy values
            step_importance: (num_steps,) importance scores
        """
        steps = self.parse_reasoning_steps(reasoning_chain)
        step_entropies = []
        step_importance = []

        for step_idx, step in enumerate(steps):
            # Compute conditional probability of answer given this step
            partial_reasoning = '\n'.join(steps[:step_idx+1])

            # Get logits for final answer token
            with torch.no_grad():
                outputs = model(self.tokenizer.encode(partial_reasoning))
                answer_logits = outputs.logits[:, -1, :]

            # Compute entropy at this step
            answer_token_id = self.tokenizer.encode(answer)[0]
            answer_prob = F.softmax(answer_logits, dim=-1)[0, answer_token_id]

            # Entropy measures confidence in answer
            entropy = -answer_prob * torch.log(answer_prob + 1e-10)
            step_entropies.append(entropy.item())

            # Importance = change in entropy from previous step
            if step_idx == 0:
                importance = 1.0
            else:
                prev_entropy = step_entropies[step_idx - 1]
                importance = abs(entropy.item() - prev_entropy) / (prev_entropy + 1e-8)

            step_importance.append(importance)

        return torch.tensor(step_entropies), torch.tensor(step_importance)

    def parse_reasoning_steps(self, reasoning_chain):
        """
        Extract individual reasoning steps from chain.
        """
        # Split by numbered patterns or line breaks
        import re
        steps = re.split(r'\n(?=\d+\.)', reasoning_chain)
        return [s.strip() for s in steps if s.strip()]

    def identify_prunable_steps(self, step_entropies, threshold_percentile=20):
        """
        Identify low-entropy steps that can be pruned.
        """
        threshold = torch.quantile(step_entropies, threshold_percentile / 100)
        prunable_mask = step_entropies < threshold
        return prunable_mask
```

**Step 2: Implement Supervised Fine-Tuning**

Train models on original reasoning chains:

```python
# Pseudocode for SFT on full reasoning
class ReasoningSTFTrainer:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def format_training_example(self, question, reasoning_chain, answer):
        """
        Format complete reasoning example for training.
        """
        template = f"""Question: {question}

{reasoning_chain}

Answer: {answer}"""
        return template

    def train_full_reasoning(self, training_data, num_epochs=3):
        """
        SFT stage: train on complete reasoning chains.
        """
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(num_epochs):
            total_loss = 0
            for example in training_data:
                # Format with full reasoning
                text = self.format_training_example(
                    example['question'],
                    example['full_reasoning'],
                    example['answer']
                )

                input_ids = self.tokenizer.encode(text, return_tensors='pt')
                target_ids = input_ids.clone()

                # Mask question tokens (only train on reasoning + answer)
                question_end = text.find('Answer:')
                question_token_count = len(self.tokenizer.encode(text[:question_end]))
                target_ids[:, :question_token_count] = -100

                # Forward pass
                outputs = self.model(input_ids, labels=target_ids)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss = {total_loss / len(training_data):.4f}")

        return self.model
```

**Step 3: Implement GRPO with Skip Tokens**

Train models to autonomously skip low-information steps:

```python
# Pseudocode for GRPO with skip token optimization
class GRPOSkipTokenTrainer:
    def __init__(self, model, tokenizer, verifier_model):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier_model  # Model to verify answers

    def generate_with_skip_option(self, question, max_steps=10):
        """
        Generate reasoning with optional [SKIP] tokens.
        """
        prompt = f"Question: {question}\nLet's think step by step:\n"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        for step_num in range(max_steps):
            # Generate next step or skip token
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9
            )

            generated_text = self.tokenizer.decode(outputs[0])

            # Check if model chose [SKIP]
            if '[SKIP]' in generated_text:
                # Continue without this step
                input_ids = outputs
            else:
                # Include step
                input_ids = outputs

        return generated_text

    def compute_grpo_reward(self, question, reasoning, answer, full_reasoning):
        """
        Compute reward balancing correctness and compression.
        """
        # Task correctness: verify answer
        correctness = self.verifier.verify(question, answer)

        # Compression: measure step reduction
        full_steps = len(full_reasoning.split('\n'))
        compressed_steps = len(reasoning.split('\n'))
        compression_ratio = compressed_steps / (full_steps + 1e-8)

        # Combined reward
        reward = 0.7 * correctness + 0.3 * (1 - compression_ratio)
        return reward

    def train_grpo(self, training_data, num_steps=1000):
        """
        GRPO training: optimize for compressed reasoning.
        """
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        for step in range(num_steps):
            batch = training_data[step % len(training_data)]

            # Generate multiple reasoning variants
            question = batch['question']
            full_answer = batch['answer']

            # Generate with skip tokens
            compressed_reasoning = self.generate_with_skip_option(question)

            # Extract answer from reasoning
            pred_answer = self.extract_answer(compressed_reasoning)

            # Compute GRPO reward
            reward = self.compute_grpo_reward(
                question,
                compressed_reasoning,
                pred_answer,
                batch['full_reasoning']
            )

            # Policy gradient update
            outputs = self.model(
                self.tokenizer.encode(question + '\n' + compressed_reasoning,
                                    return_tensors='pt')
            )
            log_probs = -F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                self.tokenizer.encode(compressed_reasoning).to(outputs.logits.device)
            )

            # GRPO loss
            loss = -log_probs * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.model

    def extract_answer(self, reasoning_text):
        """
        Extract final answer from reasoning.
        """
        import re
        match = re.search(r'Answer[:\s]*(.+?)(?:\n|$)', reasoning_text, re.IGNORECASE)
        return match.group(1) if match else ''
```

**Step 4: Integrate and Evaluate Compression**

Evaluate compression effectiveness:

```python
# Pseudocode for compression evaluation
class CompressionEvaluator:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_compression(self, test_examples):
        """
        Evaluate compression rate and accuracy.
        """
        total_original_steps = 0
        total_compressed_steps = 0
        correct_answers = 0

        for example in test_examples:
            # Generate compressed reasoning
            compressed = self.model.generate(
                self.tokenizer.encode(example['question'], return_tensors='pt'),
                max_length=256
            )

            compressed_text = self.tokenizer.decode(compressed[0])

            # Count steps
            original_steps = len(example['full_reasoning'].split('\n'))
            compressed_steps = len(compressed_text.split('\n'))

            # Verify answer
            extracted_answer = self.extract_answer(compressed_text)
            is_correct = (extracted_answer.lower() in example['answer'].lower())

            total_original_steps += original_steps
            total_compressed_steps += compressed_steps
            correct_answers += int(is_correct)

        compression_rate = total_compressed_steps / (total_original_steps + 1e-8)
        accuracy = correct_answers / len(test_examples)

        return {
            'compression_rate': compression_rate,
            'accuracy': accuracy,
            'avg_compression_per_example': 1 - compression_rate
        }
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Entropy threshold for pruning: 20th percentile of step entropies
- SFT learning rate: 2e-5 to 5e-5
- GRPO learning rate: 1e-5 to 2e-5
- Correctness/compression trade-off weight: 0.7/0.3
- Maximum steps in reasoning: 5-15 steps

**When to Use Step Entropy**:
- Long chain-of-thought reasoning that produces verbose outputs
- Tasks where inference cost is proportional to reasoning length
- Systems where reasoning interpretability is still desired (compressed CoT)
- Benchmarks with 80%+ low-entropy steps

**When NOT to Use**:
- Very short reasoning chains (minimal compression potential)
- Tasks where every reasoning step is critical
- Models already trained for concise reasoning
- Applications requiring exhaustive reasoning documentation

**Implementation Notes**:
- The 80% low-entropy pruning rate is empirically validated but task-dependent
- Step entropy is task-dependent; analysis should be done per-domain
- Two-stage training (SFT then GRPO) is important for stability
- Monitor that compression doesn't significantly degrade answer quality
- Consider domain-specific verifiers for better answer extraction

### Reference

Paper: Compressing Chain-of-Thought via Step Entropy
ArXiv: 2508.03346
Performance: 80% of low-entropy steps can be pruned with minor accuracy degradation; combined with GRPO enables autonomous compressed reasoning generation
