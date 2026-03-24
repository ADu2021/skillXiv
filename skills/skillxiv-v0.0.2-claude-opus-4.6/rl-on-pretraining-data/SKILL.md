---
name: rl-on-pretraining-data
title: "Reinforcement Learning on Pre-Training Data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.19249"
keywords: [reinforcement-learning, pre-training, llm-scaling, reward-signals, language-models]
description: "Scale LLM training using RL on unlabeled pre-training corpora without human annotation. Derive reward signals directly from text segments to optimize both autoregressive generation and in-context reasoning across knowledge and mathematical domains."
---

## Outcome: Scaling LLM Capability Beyond Supervised Training Data Limits

Enable language models to improve reasoning and knowledge through reinforcement learning signals derived directly from unlabeled pre-training data, eliminating dependence on human feedback while maintaining favorable scaling behavior across general knowledge and mathematical reasoning benchmarks.

## Problem Context

Large language models face a fundamental scaling constraint: exponential computational growth requires quadratic growth in high-quality text data, but the supply of human-curated, annotated data remains finite. Methods like RLHF and RLVR depend on human labeling, which introduces bottlenecks in data throughput and scale.

Existing RL approaches require reward models trained on explicit human feedback, limiting scalability to problem domains where obtaining such annotations is expensive or impractical. This creates a hard ceiling on training efficiency: more compute doesn't automatically translate to better performance when reward signal generation becomes the constraint.

## Core Concept

RLPT (Reinforcement Learning on Pre-Training data) derives reward signals directly from existing unlabeled text corpora instead of human annotation. The key insight is that accurate prediction of subsequent text segments—either the next continuation or masked intermediate text—serves as a meaningful learning signal for improving reasoning and generalization.

The method constructs two complementary RL objectives from raw web text:

**Autoregressive Segment Reasoning (ASR):** The model predicts the next logical text segment given preceding context. Correct prediction earns reward; this trains autoregressive generation quality and forward-looking reasoning.

**Middle Segment Reasoning (MSR):** The model predicts masked or removed text spans using bidirectional context. This trains the model to leverage surrounding information and develop more robust semantic understanding.

Both tasks exploit the self-contained structure of pre-training corpora as an unlimited source of reward signals, enabling continuous RL training without external annotation.

## Architecture Overview

- **Data Foundation:** Aggregated web text from Wikipedia, arXiv, conversation threads, and general web corpora with MinHash-based deduplication and PII masking
- **Reward Mechanism:** Generative reward model evaluates semantic consistency between policy predictions and ground-truth text segments
- **Dual Task Structure:** ASR (forward prediction) and MSR (bidirectional reconstruction) objectives simultaneously optimize generation and understanding
- **Policy:** Base language model (e.g., Qwen3-4B-Base) trained via policy gradient methods using the derived reward signal
- **Scaling:** Training operates on the same pre-training data scale without bottlenecks from annotation pipelines

## Implementation

### Step 1: Prepare Pre-Training Corpus with Segment Extraction

The foundation is a large, deduplicated text corpus. Extract training examples by segmenting text into logical chunks and creating segment prediction tasks.

```python
# segment_extraction.py
import hashlib
from collections import defaultdict

class SegmentCorpus:
    def __init__(self, min_segment_length=64, max_segment_length=256):
        self.min_length = min_segment_length
        self.max_length = max_segment_length
        self.hash_cache = defaultdict(list)

    def deduplicate_minihash(self, documents, num_hashes=128):
        """Remove duplicate documents using MinHash fingerprinting."""
        def get_shingles(text, k=4):
            return [text[i:i+k] for i in range(len(text) - k + 1)]

        unique_docs = []
        for doc in documents:
            shingles = get_shingles(doc)
            hashes = [hash(s) % (2**32) for s in shingles[:num_hashes]]
            fingerprint = tuple(sorted(hashes))

            if fingerprint not in self.hash_cache:
                self.hash_cache[fingerprint].append(doc)
                unique_docs.append(doc)

        return unique_docs

    def extract_segments(self, document, stride=None):
        """Break document into logical segments for training."""
        sentences = document.split('. ')
        segments = []
        current_segment = ""

        for sentence in sentences:
            if len(current_segment) + len(sentence) < self.max_segment_length:
                current_segment += sentence + ". "
            else:
                if len(current_segment) >= self.min_length:
                    segments.append(current_segment.strip())
                current_segment = sentence + ". "

        if len(current_segment) >= self.min_length:
            segments.append(current_segment.strip())

        return segments

    def create_training_pairs(self, document):
        """Create (context, target_segment) pairs for ASR and MSR tasks."""
        segments = self.extract_segments(document)
        training_pairs = []

        # Autoregressive Segment Reasoning (ASR)
        for i in range(len(segments) - 1):
            context = " ".join(segments[:i+1])
            target = segments[i+1]
            training_pairs.append(("ASR", context, target))

        # Middle Segment Reasoning (MSR): mask middle segment
        for i in range(1, len(segments) - 1):
            before = " ".join(segments[:i])
            after = " ".join(segments[i+1:])
            target = segments[i]
            training_pairs.append(("MSR", (before, after), target))

        return training_pairs

corpus = SegmentCorpus()
documents = [...]  # Load pre-training documents
unique_docs = corpus.deduplicate_minihash(documents)
```

### Step 2: Initialize Reward Model Using Semantic Consistency

Create a reward model that evaluates whether the policy's predicted segment is semantically consistent with ground-truth continuation.

```python
# reward_model.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SemanticRewardModel:
    def __init__(self, base_model_name="bert-base-uncased", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name).to(device)
        self.device = device

    def get_embeddings(self, texts):
        """Encode text to dense embeddings."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        return F.normalize(embeddings, p=2, dim=1)

    def compute_reward(self, predicted_segment, ground_truth_segment):
        """Compute reward as cosine similarity between embeddings."""
        pred_emb = self.get_embeddings([predicted_segment])
        true_emb = self.get_embeddings([ground_truth_segment])

        # Cosine similarity: higher is better
        similarity = F.cosine_similarity(pred_emb, true_emb)

        # Scale to [0, 1] range and apply margin
        reward = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]
        return reward.item()

    def batch_compute_rewards(self, predicted_segments, ground_truth_segments):
        """Compute rewards for a batch of predictions."""
        pred_embs = self.get_embeddings(predicted_segments)
        true_embs = self.get_embeddings(ground_truth_segments)

        rewards = F.cosine_similarity(pred_embs, true_embs)
        rewards = (rewards + 1) / 2  # Normalize to [0, 1]

        return rewards.detach().cpu().numpy()

reward_model = SemanticRewardModel()
reward = reward_model.compute_reward(
    predicted="The capital of France is Paris",
    ground_truth="The capital of France is Paris"
)
```

### Step 3: Implement Policy Gradient Training Loop

Train the base language model using policy gradient methods with the derived reward signal.

```python
# policy_training.py
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

class RLPTTrainer:
    def __init__(self, model_name, reward_model, learning_rate=5e-5, gamma=0.99):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_model = reward_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.to(self.device)

    def generate_segment(self, context, max_length=128, temperature=0.7):
        """Generate candidate segment using the policy."""
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.policy.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def compute_policy_loss(self, context, generated_segment, ground_truth_segment, reward):
        """Compute policy gradient loss."""
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        target_ids = self.tokenizer.encode(generated_segment, return_tensors="pt").to(self.device)

        # Forward pass through policy
        outputs = self.policy(
            input_ids=input_ids,
            labels=target_ids
        )

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Select log probabilities for generated tokens
        generated_log_probs = log_probs[0, torch.arange(target_ids.shape[1]-1), target_ids[0, 1:]]
        policy_loss = -(generated_log_probs.sum() * reward)

        return policy_loss

    def train_step(self, batch_data):
        """Single training step over a batch of examples."""
        total_loss = 0

        for context, generated_seg, ground_truth_seg in batch_data:
            # Get reward signal from reward model
            reward = self.reward_model.compute_reward(generated_seg, ground_truth_seg)

            # Compute policy gradient loss
            loss = self.compute_policy_loss(context, generated_seg, ground_truth_seg, reward)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(batch_data)

    def train_epoch(self, training_pairs, batch_size=8):
        """Train for one epoch over all training pairs."""
        self.policy.train()
        total_loss = 0

        for i in range(0, len(training_pairs), batch_size):
            batch = training_pairs[i:i+batch_size]

            batch_data = []
            for task_type, context_or_tuple, ground_truth in batch:
                if task_type == "ASR":
                    generated = self.generate_segment(context_or_tuple)
                    batch_data.append((context_or_tuple, generated, ground_truth))
                elif task_type == "MSR":
                    before, after = context_or_tuple
                    context = before + " [MASK] " + after
                    generated = self.generate_segment(context)
                    batch_data.append((context, generated, ground_truth))

            loss = self.train_step(batch_data)
            total_loss += loss

        return total_loss / (len(training_pairs) // batch_size)

trainer = RLPTTrainer("qwen2-4b", reward_model)
epoch_loss = trainer.train_epoch(training_pairs, batch_size=8)
```

### Step 4: Evaluation on Downstream Benchmarks

Assess model improvements on standard benchmarks after RL training.

```python
# evaluation.py
from datasets import load_dataset
import numpy as np

class BenchmarkEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def evaluate_mmlu(self, num_samples=None):
        """Evaluate on MMLU benchmark."""
        dataset = load_dataset("cais/mmlu", "all")
        test_data = dataset["test"]

        if num_samples:
            test_data = test_data.select(range(min(num_samples, len(test_data))))

        correct = 0
        total = 0

        for example in test_data:
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]

            # Format multiple choice
            prompt = f"{question}\nOptions: {', '.join(choices)}\nAnswer: "

            with torch.no_grad():
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(input_ids, max_new_tokens=10)
                response = self.tokenizer.decode(outputs[0])

            # Simple check: does response contain correct answer
            if choices[answer_idx].lower() in response.lower():
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0
        return {"mmlu_accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_aime(self, num_samples=None):
        """Evaluate on AIME-style math problems."""
        # Placeholder for AIME evaluation
        # In practice, would parse structured math problems
        return {"aime_pass_at_1": 0.0}

    def run_all_benchmarks(self):
        """Run comprehensive evaluation."""
        results = {}
        results.update(self.evaluate_mmlu(num_samples=500))
        results.update(self.evaluate_aime(num_samples=100))
        return results

evaluator = BenchmarkEvaluator(trainer.policy, trainer.tokenizer)
benchmark_results = evaluator.run_all_benchmarks()
print(benchmark_results)
```

## Practical Guidance

### Hyperparameter Reference

| Parameter | Recommended Value | Range | Effect |
|-----------|-------------------|-------|--------|
| Learning Rate | 5e-5 | 1e-5 to 1e-4 | Lower = stable but slow; higher = faster convergence but less stable |
| Batch Size | 8-16 | 4 to 64 | Larger enables better gradient estimates; memory-limited by hardware |
| Max Segment Length | 256 tokens | 64-512 | Longer captures broader context; shorter trains faster |
| Reward Temperature | 1.0 | 0.5-2.0 | Lower = sharper reward signal; higher = smoother gradient flow |
| Gamma (Discount) | 0.99 | 0.95-0.99 | Higher values weight future rewards; 0.99 standard for RL |
| Generation Temperature | 0.7 | 0.5-1.0 | Lower = deterministic; higher = diverse exploration |

### When to Use RLPT

- **Abundant unlabeled data but scarce annotations:** You have large pre-training corpora but cannot afford human feedback labeling
- **Scaling reasoning capabilities:** Target improvements in knowledge recall, mathematical reasoning, or logical inference
- **Cost-sensitive RL training:** Eliminate annotation bottlenecks that constrain throughput of other RL methods (RLHF, RLVR)
- **Domain-specific pre-training:** Fine-tune existing models on domain corpora without external reward labels
- **Favorable scaling assumptions:** You expect the model benefits from RL will compound with more compute and data

### When NOT to Use RLPT

- **Preference-aligned training required:** If the goal is safety, instruction-following, or aligning to human preferences, RLPT alone is insufficient. It optimizes for semantic consistency, not human judgment
- **Limited pre-training data:** If your corpus is small or highly curated, the self-derived reward signal becomes weak and less reliable
- **Real-time or online learning:** RLPT assumes static pre-training data; not designed for continual online updates with new user interactions
- **Few-shot or task-specific RL:** When you have labeled data for specific tasks, supervised fine-tuning or supervised RL (like RLHF) will be more sample-efficient
- **Inference speed critical:** The method adds training overhead; if model latency is the bottleneck, focus on quantization or distillation instead

### Common Pitfalls

**Stale reward signals:** If the reward model is not updated as the policy improves, rewards become less meaningful. Periodically retrain or refresh the reward model on new policy samples.

**Reward hacking:** The policy may exploit quirks in the semantic similarity metric (e.g., copying ground-truth text verbatim). Use diverse reward models or regularize for output diversity.

**Data contamination:** If pre-training data contains benchmark test sets, RL training on that data inflates evaluation metrics. Carefully filter benchmark data before RL training.

**Segment length sensitivity:** Segments that are too short lose context; too long make training inefficient. Empirically tune segment extraction to your domain.

**Unstable policy gradients:** Large reward signals combined with high learning rates cause oscillation. Use gradient clipping, lower learning rates, and reward normalization.

## Reference

- **Paper:** Reinforcement Learning on Pre-Training Data (arXiv 2509.19249)
- **arXiv Link:** https://arxiv.org/abs/2509.19249
- **Base Models:** Compatible with transformer-based LLMs (Qwen, LLaMA, Mistral, GPT-style architectures)
- **Key Results:** +3.0 MMLU, +8.1 GPQA-Diamond, +6.6 AIME24 on Qwen3-4B-Base with RLPT training
