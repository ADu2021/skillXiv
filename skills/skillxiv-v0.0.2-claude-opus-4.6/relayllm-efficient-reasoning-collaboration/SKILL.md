---
name: relayllm-efficient-reasoning-collaboration
title: "RelayLLM: Efficient Reasoning via Collaborative Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05167"
keywords: [Efficient Inference, Multi-Model Collaboration, Token-Level Routing, Reasoning Optimization]
description: "Enable small language models to dynamically invoke larger models at critical reasoning tokens rather than offloading entire queries. RelayLLM achieves 49.52% accuracy across benchmarks while invoking the large model for only 1.07% of tokens—98.2% cost reduction compared to non-collaborative approaches."
---

## When to Use This Skill
- Reasoning tasks where a small model can handle most steps independently
- Cost-sensitive inference with access to both capable and efficient models
- Applications requiring balanced accuracy-efficiency trade-offs
- Multi-benchmark scenarios (mathematical reasoning, general knowledge, open-ended problems)
- Model pairs from the same family (Qwen, Llama, etc.) for tokenization consistency

## When NOT to Use This Skill
- Tasks where all steps equally require high reasoning capability
- Scenarios with extreme latency requirements (RL refinement adds training time)
- Applications requiring guaranteed model invocation (e.g., safety-critical steps)
- Single-model inference (no collaborative pair available)

## Problem Summary
Current collaborative approaches use coarse-grained routing, offloading entire queries to large models whenever difficulty emerges. This wastes computational resources since small models typically handle most reasoning steps competently. The small model becomes mere scaffolding—when complexity appears, the entire problem transfers to the expensive model. This approach ignores that reasoning occurs token-by-token: some tokens need teacher guidance while others don't.

## Solution: Dynamic Token-Level Teacher Invocation

Train small models to recognize critical tokens and request help only when necessary, with a two-stage training framework.

```python
class RelayLLMStudent:
    def __init__(self, small_model, large_model):
        self.student = small_model
        self.teacher = large_model

    def forward_with_relaying(self, query, max_calls=None):
        """Generate tokens with selective teacher invocation"""
        generated = []
        token_count = 0
        teacher_tokens_used = 0

        while not_finished():
            # Generate next token in student
            token = self.student.generate_next()

            # Check if token is a call command
            if token == "<call>n</call>":
                # Pause and invoke teacher
                n_tokens = extract_count(token)
                teacher_tokens = self.teacher.generate_n_tokens(n_tokens)
                generated.extend(teacher_tokens)
                teacher_tokens_used += n_tokens
                token_count += n_tokens
            else:
                # Regular token from student
                generated.append(token)
                token_count += 1

        efficiency = 1.0 - (teacher_tokens_used / token_count)
        return generated, efficiency
```

## Key Implementation Details

**Two-Stage Training Framework:**

**Stage 1: Supervised Warm-up**
- Teach small model to generate special command tokens (`<call>n</call>`)
- Commands inserted at synthetically random positions within self-sampled sequences
- Establishes baseline command generation capability

**Stage 2: Reinforcement Learning (GRPO)**
Three difficulty-aware reward scenarios:

```python
# Reward design for difficulty-aware learning
def compute_difficulty_reward(student_output, teacher_output, dataset_label):
    """Conditional rewards based on problem difficulty"""

    if dataset_label == "student_solvable":
        # Easy problems: bonus for independent success
        if student_correct:
            return +1.5  # Encourage self-sufficiency
        else:
            return 0.0

    elif dataset_label == "teacher_dependent":
        # Medium problems: penalty for avoiding help
        if not student_called_teacher:
            return -1.0  # Discourage false confidence
        else:
            return +1.0  # Reward appropriate help-seeking

    elif dataset_label == "teacher_unsolvable":
        # Hard problems: exploration bonus
        if student_attempted_help:
            return +0.5  # Encourage trying despite difficulty
        else:
            return 0.0
```

**Inference Mechanism:**
1. Small model generates tokens normally
2. Upon detecting `<call>` tokens, pause generation
3. Teacher generates specified token count
4. Resume student generation with teacher output as context

**Data Filtering:**
Preprocess datasets to filter unhelpful teacher invocations:
- Require teacher model achieves ≥50% pass rate on samples
- Prevents training on scenarios where teacher cannot improve outcomes
- Ensures relaying is genuinely beneficial

**Dynamic Length Learning:**
Instead of fixed token requests, models learn to request "just enough":
- Fixed-length strategy: Teacher generates 100 tokens regardless of need
- Learned strategy: Models dynamically predict required tokens
- Result: 1.07% call ratio vs. 2.87% for fixed strategies

## Practical Configuration

**Model Architecture:**
- Student: Qwen3-0.6B or Qwen3-1.7B
- Teacher: Qwen3-8B (same family ensures consistency)
- Same family prevents tokenization mismatches

**Training Setup:**
- 32 H100 GPUs
- ~35 hours total training
- Group Relative Policy Optimization (GRPO)

## Performance Results

**Accuracy & Efficiency:**
- Average accuracy: 49.52% across six benchmarks
- Teacher invocation rate: 1.07% of total tokens
- Cost reduction: 98.2% vs. performance-matched routers
- Graceful degradation without teacher access (models internalize patterns)

**Cross-Domain Transfer:**
- Trained on mathematical problems (MATH)
- Generalizes to MMLU-Pro, BBH, SuperGPQA without retraining
- Maintains efficiency gains across domains

**Benchmark Coverage:**
- MATH500 (mathematical reasoning)
- MMLU-Pro (general knowledge)
- BBH (big-bench hard)
- SuperGPQA (factual accuracy)
- GSM8K (grade school math)

## Advantages Over Alternatives

- **vs. Ensemble Routing**: Dynamic token-level decisions vs. coarse query-level
- **vs. Mixture of Experts**: No architectural constraints; works with frozen models
- **vs. Model Distillation**: Student retains independence; teacher is safety net, not replacement
- **vs. Larger Base Model**: 98% cost reduction with comparable accuracy

## Deployment Recommendations
1. Select student-teacher model pair from same family
2. Prepare curriculum with difficulty-labeled examples
3. Run Stage 1 warm-up for command generation convergence
4. Fine-tune with GRPO over 35+ hours
5. Validate teacher invocation rate on holdout set
6. Monitor graceful degradation (student-only inference)
