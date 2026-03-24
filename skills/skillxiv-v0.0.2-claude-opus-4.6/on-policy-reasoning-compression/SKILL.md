---
name: on-policy-reasoning-compression
title: "On-Policy Self-Distillation for Reasoning Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.05433"
keywords: [Reasoning, Compression, Self-Distillation, Token Efficiency, On-Policy Learning]
description: "Compress verbose reasoning by conditioning the same model on a conciseness instruction to create a teacher. Minimize KL divergence between student outputs and concise teacher without ground truth, enabling implicit reward learning that improves accuracy while reducing tokens."
---

# On-Policy Self-Distillation for Reasoning Compression

Long reasoning chains improve model accuracy but consume significant compute and tokens. Traditional compression requires ground-truth short reasoning or external reward models. OPSDC takes a simpler approach: instruct the same model to be concise, then teach it to match that concise version without explicit supervision. The resulting compression naturally adapts to problem difficulty—easy problems receive strong pressure to be brief; hard problems preserve reasoning length.

The core innovation elegantly combines on-policy learning, reverse KL divergence for mode-seeking behavior, and implicit difficulty adaptation. The method simultaneously improves accuracy (by eliminating verbose error-prone reasoning) and reduces tokens.

## Core Concept

OPSDC operates through four coordinated principles:

1. **Same-Model Teacher**: Condition target model on "be concise" instruction to generate a teacher; no external oracle needed
2. **On-Policy Distillation**: Train on student-generated rollouts (on-policy) to avoid distribution shift common in offline distillation
3. **Mode-Seeking Behavior**: Use reverse KL divergence to concentrate probability on teacher's preferred reasoning paths
4. **Implicit Difficulty Scaling**: Hard problems naturally receive weak compression pressure because even the concise teacher needs extended reasoning

## Architecture Overview

- **Student Model**: Base reasoning model generating trajectories
- **Teacher Model**: Same model prompted with "be concise" instruction
- **On-Policy Sampling**: Collect rollouts from student under current policy
- **KL Divergence**: Compute reverse KL from teacher to student for mode-seeking
- **Loss Scaling**: Implicit weighting based on reasoning complexity
- **Output**: Compressed, faster, more accurate reasoning

## Implementation Steps

**Step 1: Prepare student and teacher conditioning**

Design two prompts: one for normal reasoning and one for concise reasoning.

```python
def create_prompt_pair(question, concise=False):
    """Generate dual prompts for student and teacher."""
    base_prompt = f"""Question: {question}

Think step-by-step."""

    if concise:
        prompt = f"""{base_prompt}
Keep your reasoning concise and direct. Skip explanations unless essential.
Reason briefly:"""
    else:
        prompt = f"""{base_prompt}
Reason step-by-step:"""

    return prompt

# Example prompts
math_question = "What is 17 * 23?"
student_prompt = create_prompt_pair(math_question, concise=False)
teacher_prompt = create_prompt_pair(math_question, concise=True)

# Student generates verbose: "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391"
# Teacher generates concise: "17 * 23 = 391"
```

**Step 2: Collect on-policy rollouts from student**

Sample trajectories from the current student model under normal (non-concise) prompting.

```python
def sample_student_rollouts(model, questions, num_rollouts=4, temperature=0.7):
    """
    Collect on-policy rollouts from student model.
    Each question sampled multiple times to capture diversity.
    """
    rollouts = []

    for question in questions:
        student_prompt = create_prompt_pair(question, concise=False)

        for _ in range(num_rollouts):
            # Sample from student (current policy)
            response = model.generate(
                student_prompt,
                max_tokens=256,
                temperature=temperature,
                top_p=0.9
            )

            rollouts.append({
                'question': question,
                'student_response': response,
                'student_prompt': student_prompt
            })

    return rollouts
```

**Step 3: Generate teacher responses and compute KL divergence**

For each student rollout, generate concise teacher response and measure divergence.

```python
def compute_per_token_kl(model, student_response, teacher_response,
                        teacher_prompt, question, tokenizer):
    """
    Compute KL divergence between teacher and student distributions.

    KL(P_teacher || P_student) measures how much student diverges from teacher.
    Reverse KL encourages student to match teacher (mode-seeking).
    """
    # Tokenize responses
    student_tokens = tokenizer.encode(student_response)
    teacher_tokens = tokenizer.encode(teacher_response)

    # Forward passes to get logits
    teacher_prompt_full = f"{teacher_prompt}\n{teacher_response}"
    student_prompt_full = f"{create_prompt_pair(question, False)}\n{student_response}"

    teacher_logits = model.forward(teacher_prompt_full)['logits']
    student_logits = model.forward(student_prompt_full)['logits']

    # Compute log probabilities for teacher and student
    teacher_logprobs = softmax(teacher_logits, dim=-1)
    student_logprobs = softmax(student_logits, dim=-1)

    # Per-token KL divergence
    per_token_kl = []
    for t in range(min(len(teacher_tokens), len(student_tokens))):
        teacher_dist = teacher_logprobs[t]  # Distribution over vocabulary
        student_dist = student_logprobs[t]

        # KL(teacher || student) = sum teacher * log(teacher / student)
        kl_t = (teacher_dist * (torch.log(teacher_dist + 1e-8) -
                               torch.log(student_dist + 1e-8))).sum()
        per_token_kl.append(kl_t.item())

    return np.mean(per_token_kl)

def generate_teacher_responses(model, rollouts):
    """Generate concise responses from teacher for each student rollout."""
    for rollout in rollouts:
        teacher_prompt = create_prompt_pair(rollout['question'], concise=True)

        teacher_response = model.generate(
            teacher_prompt,
            max_tokens=128,  # Shorter max for concise generation
            temperature=0.5  # Lower temperature for more deterministic teacher
        )

        rollout['teacher_response'] = teacher_response
        rollout['teacher_prompt'] = teacher_prompt

        # Compute KL divergence
        rollout['kl_divergence'] = compute_per_token_kl(
            model,
            rollout['student_response'],
            teacher_response,
            teacher_prompt,
            rollout['question'],
            tokenizer
        )

    return rollouts
```

**Step 4: Implicit difficulty adaptation through KL weighting**

Use per-sample KL to weight loss; easy problems with large KL (big divergence) get strong compression; hard problems with small KL (teacher also verbose) get weak compression.

```python
def compute_difficulty_weights(rollouts, threshold=0.5):
    """
    Assign difficulty weights based on KL divergence magnitude.

    High KL = easy problem (teacher very concise, student verbose)
              → strong compression pressure
    Low KL = hard problem (even teacher needs verbosity)
             → weak compression pressure
    """
    kl_values = np.array([r['kl_divergence'] for r in rollouts])
    kl_normalized = (kl_values - kl_values.min()) / (kl_values.max() - kl_values.min() + 1e-8)

    weights = []
    for rollout in rollouts:
        kl_normalized = rollout['kl_divergence'] / (kl_values.max() + 1e-8)

        # Higher KL → higher weight (compress more)
        weight = kl_normalized ** 2  # Quadratic to emphasize easy problems

        rollout['difficulty_weight'] = weight
        weights.append(weight)

    return rollouts
```

**Step 5: On-policy loss computation**

Compute reverse KL divergence loss weighted by implicit difficulty.

```python
def compute_opsdc_loss(model, rollouts, tokenizer, epsilon=1e-8):
    """
    On-Policy Self-Distillation Compression Loss

    Loss = E[KL(P_teacher || P_student)] weighted by difficulty
    """
    total_loss = 0.0
    num_tokens = 0

    for rollout in rollouts:
        student_response = rollout['student_response']
        teacher_response = rollout['teacher_response']
        difficulty_weight = rollout['difficulty_weight']

        # Get logits for student and teacher
        student_logits = model.forward(
            f"{rollout['student_prompt']}\n{student_response}"
        )['logits']

        teacher_logits = model.forward(
            f"{rollout['teacher_prompt']}\n{teacher_response}"
        )['logits']

        # Convert to probabilities
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)

        # Per-token reverse KL divergence
        # KL(teacher || student) = E_teacher[log teacher - log student]
        log_student = torch.log(student_probs + epsilon)
        log_teacher = torch.log(teacher_probs + epsilon)

        kl_loss = (teacher_probs * (log_teacher - log_student)).sum(dim=-1).mean()

        # Weight by difficulty: hard problems (low weight) contribute less loss
        weighted_loss = kl_loss * difficulty_weight

        total_loss += weighted_loss
        num_tokens += len(tokenizer.encode(student_response))

    return total_loss / len(rollouts)
```

**Step 6: Training loop with on-policy updates**

Iteratively collect on-policy rollouts, compute loss, and update model.

```python
def train_opsdc(model, questions, num_iterations=5, batch_size=32,
               learning_rate=2e-5):
    """
    Train model with On-Policy Self-Distillation Compression.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iteration in range(num_iterations):
        # Batch questions
        question_batches = [
            questions[i:i + batch_size]
            for i in range(0, len(questions), batch_size)
        ]

        iteration_loss = 0.0
        num_batches = 0

        for batch_questions in question_batches:
            # Sample student rollouts
            rollouts = sample_student_rollouts(
                model,
                batch_questions,
                num_rollouts=4,
                temperature=0.7
            )

            # Generate teacher responses
            rollouts = generate_teacher_responses(model, rollouts)

            # Compute difficulty weights
            rollouts = compute_difficulty_weights(rollouts)

            # Compute loss
            loss = compute_opsdc_loss(model, rollouts, tokenizer)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            iteration_loss += loss.item()
            num_batches += 1

        avg_loss = iteration_loss / num_batches
        print(f"Iteration {iteration + 1}: Loss = {avg_loss:.4f}")

    return model
```

**Step 7: Evaluation and metric tracking**

Measure compression gains: token reduction and accuracy improvement.

```python
def evaluate_compression(model, test_questions, baseline_model=None):
    """
    Evaluate compression: compare tokens, accuracy, speed.
    """
    results = {
        'token_reductions': [],
        'accuracy_baseline': [],
        'accuracy_compressed': [],
        'speedup': []
    }

    for question in test_questions:
        # Baseline (uncompressed)
        baseline_prompt = create_prompt_pair(question, concise=False)
        baseline_resp = model.generate(baseline_prompt, max_tokens=256)
        baseline_tokens = len(tokenizer.encode(baseline_resp))
        baseline_correct = evaluate_answer(baseline_resp, question)

        # Compressed
        compressed_prompt = create_prompt_pair(question, concise=True)
        start_time = time.time()
        compressed_resp = model.generate(compressed_prompt, max_tokens=128)
        generation_time = time.time() - start_time

        compressed_tokens = len(tokenizer.encode(compressed_resp))
        compressed_correct = evaluate_answer(compressed_resp, question)

        token_reduction = (baseline_tokens - compressed_tokens) / baseline_tokens
        results['token_reductions'].append(token_reduction)
        results['accuracy_baseline'].append(baseline_correct)
        results['accuracy_compressed'].append(compressed_correct)

    print(f"Avg token reduction: {np.mean(results['token_reductions']) * 100:.1f}%")
    print(f"Accuracy improvement: {np.mean(results['accuracy_compressed']) - np.mean(results['accuracy_baseline']) * 100:+.1f}%")

    return results
```

## Practical Guidance

**Hyperparameter Selection:**
- **Concise max tokens**: 50-128. Lower = more aggressive compression; higher = preserves reasoning capacity.
- **Teacher temperature**: 0.3-0.7. Lower = more deterministic conciseness; higher = more varied reasoning.
- **Difficulty weight exponent**: 1.5-2.5. Higher = stronger emphasis on easy-problem compression.
- **Number of on-policy rollouts**: 2-8 per question. More = better coverage; diminishing returns beyond 4.
- **KL epsilon**: 1e-8. Adjust to 1e-6 for numerical instability.

**When to Use:**
- Post-training: compress already-trained models
- Token-budget scenarios: improve efficiency without retraining
- Mixed reasoning tasks: varying difficulty automatically adjusts compression
- Inference optimization: reduce latency/compute with accuracy gains

**When NOT to Use:**
- Tasks requiring exhaustive detailed reasoning (compression may hurt accuracy)
- Real-time systems: on-policy generation adds overhead
- Models already fine-tuned for conciseness

**Common Pitfalls:**
- **Teacher instability**: Concise prompts can cause mode collapse. Use higher teacher temperature if responses become stereotyped.
- **KL saturation**: After few iterations, KL may plateau. Reduce learning rate or switch to curriculum learning.
- **Difficulty miscalibration**: If all problems show similar KL, difficulty weighting provides no benefit. Verify KL variance is > 0.1.
- **Accuracy cliff**: Aggressive compression can suddenly hurt accuracy. Monitor accuracy on held-out set; back off if it drops >5%.

## Reference

arXiv: https://arxiv.org/abs/2603.05433
