---
name: truncated-step-level-retrieval-reasoning
title: "Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.23440"
keywords: [Retrieval-Augmented Generation, Variance Reduction, Step-Level Rewards, Multi-Hop Reasoning, Credit Assignment]
description: "Improve credit assignment in retrieval-augmented reasoning by truncating trajectories at single decision points. Generate k samples sharing a common prefix, differing only at the next step to isolate variation and reduce gradient variance by T-fold on T-step tasks."
---

# Truncated Step-Level Sampling for Retrieval-Augmented Reasoning

Standard reinforcement learning for retrieval-augmented generation (RAG) tasks suffers from poor credit assignment: high-variance gradient estimates make it hard to distinguish which search query or reasoning step improved final outcomes. Truncated step-level sampling fixes this by generating trajectories that share a prefix and diverge only at a single decision point. This isolation dramatically reduces variance while enabling per-step reward supervision through dense LLM-as-Judge scoring.

The core innovation treats each reasoning step as an isolated decision problem. By fixing all prior steps, the method achieves T-fold variance reduction on T-step tasks while maintaining gradient accuracy through reward decomposition.

## Core Concept

Truncated sampling operates through two coordinated mechanisms:

1. **Prefix-Sharing Trajectories**: Generate k candidate continuations from identical prefixes, isolating variation to a single step. This eliminates confounding from earlier decisions.

2. **Dense Step-Level Rewards**: Score each step independently via LLM judge on three dimensions:
   - Reasoning quality (is the thinking sound?)
   - Query quality (is the search formulation effective?)
   - Answer correctness (does the final answer match?)

These two components combine to provide stable, interpretable learning signals.

## Architecture Overview

- **Input**: Multi-hop retrieval task (question requiring multiple document lookups)
- **Prefix Generation**: Execute steps 1 to t-1 deterministically or sample once, reuse across batch
- **Branching**: Generate k different step-t candidates from shared prefix
- **Evaluation**: LLM judges each step t independently on three dimensions
- **Reward Aggregation**: Combine three reward signals (reasoning + query + answer)
- **Policy Update**: Gradient step on isolated step with reduced variance
- **Output**: Improved retrieval and reasoning policy

## Implementation Steps

**Step 1: Structure multi-hop reasoning task with explicit steps**

Define the reasoning process to enable step-level isolation.

```python
def decompose_rag_task(question):
    """
    Decompose RAG into explicit steps: search → retrieve → reason → search → answer.
    Each step is atomic and independently scorable.
    """
    steps = [
        {'step': 0, 'action': 'formulate_initial_query', 'state': question},
        {'step': 1, 'action': 'retrieve_documents', 'state': None},
        {'step': 2, 'action': 'synthesize_reasoning', 'state': None},
        {'step': 3, 'action': 'formulate_followup_query', 'state': None},
        {'step': 4, 'action': 'retrieve_evidence', 'state': None},
        {'step': 5, 'action': 'generate_final_answer', 'state': None}
    ]

    return steps

# Example: "Who was the first person to reach the South Pole?"
# Step 0: User question
# Step 1: Search "first person South Pole" → retrieve docs about Amundsen, Scott
# Step 2: Synthesize: "Roald Amundsen reached it in 1911"
# Step 3: Formulate follow-up: "Roald Amundsen nationality" (verify credibility)
# Step 4: Retrieve clarification docs
# Step 5: Generate final answer with sources
```

**Step 2: Generate prefix and sample k step-t candidates**

Execute deterministically up to step t-1, then generate k diverse continuations for step t.

```python
def generate_truncated_batch(model, question, num_candidates=4, target_step=2):
    """
    Generate batch of trajectories sharing prefix, differing only at target_step.

    Example with target_step=2:
    - Execute steps 0, 1 once (deterministic or single sample)
    - Generate k candidates for step 2
    - Execute steps 3+ identically for all k candidates
    """
    steps = decompose_rag_task(question)

    # Phase 1: Execute steps before target_step deterministically
    prefix_state = question
    for step_idx in range(target_step):
        step = steps[step_idx]
        if step_idx == 0:
            prefix_state = generate_initial_query(model, question)
        elif step_idx == 1:
            documents = retrieve_documents(prefix_state)
            prefix_state = documents
        # ... continue deterministically

    # Phase 2: Generate k candidates for step target_step
    candidates = []
    for _ in range(num_candidates):
        candidate_action = model.generate(
            f"Given context: {prefix_state}, generate next reasoning step:",
            max_tokens=150,
            temperature=0.7,
            top_p=0.9
        )
        candidates.append(candidate_action)

    # Phase 3: Continuation after target_step (reuse for all candidates)
    trajectories = []
    for candidate in candidates:
        # Forward pass through remaining steps
        trajectory = {
            'prefix': prefix_state,
            'step_t_action': candidate,
            'target_step': target_step,
            'full_trajectory': None
        }

        # Execute remaining steps using this candidate
        current_state = candidate
        for step_idx in range(target_step + 1, len(steps)):
            current_state = execute_step(model, steps[step_idx], current_state)

        trajectory['full_trajectory'] = current_state
        trajectories.append(trajectory)

    return trajectories
```

**Step 3: Compute per-step LLM-as-Judge rewards**

Score each step independently on reasoning quality, query quality, and answer correctness.

```python
def compute_three_part_reward(trajectory, question, reference_answer,
                             judge_model):
    """
    Decompose reward into three scoreable components:
    1. Reasoning quality: Is the thinking sound?
    2. Query quality: Is the search formulation likely to retrieve relevant docs?
    3. Answer correctness: Does final answer match reference?
    """
    target_step = trajectory['target_step']
    step_action = trajectory['step_t_action']
    full_trajectory = trajectory['full_trajectory']

    # Component 1: Reasoning quality (LLM judge)
    reasoning_prompt = f"""
Evaluate the reasoning quality of this step:
Question: {question}
Prefix context: {trajectory['prefix'][:100]}
Reasoning step: {step_action}

Rate on scale 0-1: Is the logic sound? Does it make progress toward answer?
"""

    reasoning_score = judge_model.score(
        reasoning_prompt,
        return_numeric=True  # 0-1 score
    )

    # Component 2: Query quality (relevance of search formulation)
    if 'query' in step_action.lower() or 'search' in step_action.lower():
        query_prompt = f"""
Evaluate query formulation quality:
Original question: {question}
Formulated query: {step_action}

Rate 0-1: Will this retrieve relevant documents?
"""

        query_score = judge_model.score(query_prompt, return_numeric=True)
    else:
        query_score = 0.5  # Neutral if not a query step

    # Component 3: Answer correctness (binary: correct/incorrect)
    final_answer = full_trajectory.split('\n')[-1]  # Last line of trajectory
    is_correct = final_answer.lower() in reference_answer.lower()
    answer_score = 1.0 if is_correct else 0.0

    # Combine components (equal weighting)
    total_reward = (reasoning_score + query_score + answer_score) / 3.0

    return {
        'reasoning_score': reasoning_score,
        'query_score': query_score,
        'answer_score': answer_score,
        'total_reward': total_reward
    }

# Evaluate batch
batch = generate_truncated_batch(model, question, num_candidates=4, target_step=2)
rewards = [compute_three_part_reward(traj, question, reference_answer, judge)
           for traj in batch]
```

**Step 4: Compute policy gradients with variance reduction**

Calculate gradients using isolated step rewards and prove T-fold variance reduction.

```python
def compute_pg_loss_truncated(model, batch, rewards, target_step):
    """
    Policy gradient loss with variance reduction from truncation.

    Key insight: By fixing prefix and only varying step t,
    var(gradient) is reduced by ~T-fold relative to full-trajectory sampling.
    """
    total_loss = 0.0

    for trajectory, reward in zip(batch, rewards):
        # Get log probability of the step-t action under current policy
        step_action = trajectory['step_t_action']
        prefix_context = trajectory['prefix']

        logits = model.forward(
            f"Context: {prefix_context}\nAction:",
            output_logits=True
        )

        # Log probability of sampled action
        action_logprob = compute_logprob(logits, step_action, model.tokenizer)

        # Policy gradient: maximize log-prob of high-reward actions
        # Loss = -logprob * reward (negative for gradient descent)
        step_loss = -action_logprob * reward['total_reward']

        total_loss += step_loss

    return total_loss / len(batch)

def analyze_variance_reduction(trajectories_full, trajectories_truncated):
    """
    Empirically verify T-fold variance reduction.
    trajectories_full: full-trajectory samples from baseline
    trajectories_truncated: truncated samples from this method
    """
    # Compute gradient variance
    grads_full = [compute_gradient(traj) for traj in trajectories_full]
    grads_truncated = [compute_gradient(traj) for traj in trajectories_truncated]

    var_full = np.var([g.norm().item() for g in grads_full])
    var_truncated = np.var([g.norm().item() for g in grads_truncated])

    # Expected T-fold reduction
    T = 5  # Assume 5-step tasks
    expected_ratio = T
    observed_ratio = var_full / var_truncated

    print(f"Variance reduction: {observed_ratio:.1f}x (expected ~{expected_ratio}x)")
    return observed_ratio
```

**Step 5: RL training loop with step-level updates**

Integrate truncated sampling into standard policy gradient training.

```python
def train_rag_with_truncated_sampling(model, judge_model, train_tasks,
                                     num_steps=10000, batch_size=32):
    """
    Train retrieval-augmented reasoning with truncated step-level sampling.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(num_steps):
        # Sample tasks from training set
        task_batch = random.sample(train_tasks, batch_size)

        total_loss = 0.0

        for question, reference_answer in task_batch:
            # Randomly select which step to optimize
            target_step = random.randint(1, 4)

            # Generate truncated batch (k candidates for step target_step)
            trajectories = generate_truncated_batch(
                model,
                question,
                num_candidates=4,
                target_step=target_step
            )

            # Compute rewards
            rewards = [
                compute_three_part_reward(traj, question, reference_answer, judge_model)
                for traj in trajectories
            ]

            # Compute loss
            loss = compute_pg_loss_truncated(model, trajectories, rewards, target_step)

            total_loss += loss

        # Gradient step
        optimizer.zero_grad()
        (total_loss / len(task_batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: Loss = {(total_loss / len(task_batch)).item():.4f}")

    return model
```

**Step 6: Evaluation on multi-hop QA benchmarks**

Benchmark on tasks requiring multiple retrieval steps (TREC, HotpotQA, etc.).

```python
def evaluate_on_multihop_qa(model, eval_tasks):
    """
    Evaluate on multi-hop QA: questions requiring 2-4 retrieval steps.
    """
    correct = 0
    total = 0

    for question, reference_answer, num_hops in eval_tasks:
        # Execute full trajectory
        trajectory = execute_full_rag(model, question, max_steps=num_hops)

        # Extract final answer
        final_answer = trajectory.split('\n')[-1]

        # Check correctness
        if final_answer.lower() in reference_answer.lower():
            correct += 1

        total += 1

    accuracy = correct / total
    print(f"Multi-hop QA accuracy: {accuracy * 100:.1f}%")

    return accuracy
```

## Practical Guidance

**Hyperparameter Selection:**
- **Number of candidates k**: 2-4. More candidates = better coverage; diminishing returns beyond 4.
- **Target step selection**: Sample uniformly from [1, T] for balanced training, or focus on early steps for faster learning.
- **Reward weights**: Equal [1/3, 1/3, 1/3] standard; can emphasize answer_score for final tuning.
- **Judge model temperature**: 0.5-0.7 for stable scoring; lower = more consistent, higher = more nuanced.

**When to Use:**
- Multi-hop retrieval tasks (2-5 steps) where single-step methods fail
- Scenarios where every reasoning step affects downstream success
- Settings with access to LLM judge for step-level scoring
- Tasks where variance reduction significantly improves convergence

**When NOT to Use:**
- Single-hop retrieval (no multi-step credit assignment benefit)
- Tasks with very long trajectories (T > 10); variance reduction diminishes
- Real-time systems where multiple forward passes are expensive
- Scenarios without reliable reference answers for reward feedback

**Common Pitfalls:**
- **Inconsistent judge scoring**: If judge is non-deterministic, rewards are noisy. Use deterministic judge or averaging over multiple runs.
- **Step isolation breaks dependencies**: Some steps may be impossible without specific prefix context. Validate that sampled candidates are valid.
- **Reward signal too sparse**: If most samples have zero answer_score, provide auxiliary rewards (query quality, reasoning soundness) to maintain signal.
- **Target step bias**: If always sampling early steps, later steps never get optimized. Ensure uniform or curriculum-based step selection.

## Reference

arXiv: https://arxiv.org/abs/2602.23440
