---
name: mars-rl-multi-agent-reasoning
title: "MarsRL: Multi-Agent Reasoning System via RL with Agentic Pipeline Parallelism"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.11373"
keywords: [Multi-Agent RL, Reasoning, Pipeline Parallelism, Credit Assignment, Extended Reasoning]
description: "Train multi-agent reasoning systems with decoupled reward signals and pipeline parallelism—enable specialized Solver/Verifier/Corrector agents to iteratively refine solutions without waiting for full trajectories, handling extended reasoning up to 320K tokens."
---

# Coordinate Multi-Agent Reasoning with Decoupled Rewards and Pipeline Parallelism

Extended reasoning requires multiple agents: a Solver generates candidate solutions, a Verifier checks correctness, a Corrector refines errors. Traditional RL training waits for complete trajectories before updating, creating bottlenecks when trajectories span hundreds of thousands of tokens. MarsRL solves this via two innovations:

1. **Decoupled Agent-Specific Rewards**: Each agent receives rewards based on its individual performance, not shared trajectory outcomes—avoiding credit assignment noise
2. **Pipeline Parallelism**: Agents begin training as soon as they complete segments (16k tokens), without waiting for full trajectory completion—reducing latency from hours to minutes

This enables efficient training of specialized agents working in concert, where each agent specializes (Solver on generation, Verifier on discrimination, Corrector on refinement) without reward contamination.

## Core Concept

Multi-agent reasoning systems decompose complex problems into specialized subtasks. However, monolithic reward signals (single score for entire trajectory) fail to credit individual agents for their contributions—a Solver's poor generation gets blamed on the Verifier, and vice versa.

MarsRL decouples credit assignment: the Solver is rewarded only for solution correctness, the Verifier for discrimination accuracy, the Corrector for successful refinement. Additionally, rather than accumulating complete trajectories (potentially 320k tokens) before training, MarsRL begins training immediately after each 16k-token segment completes. This "segment rollout" approach combined with pipeline parallelism reduces training latency while maintaining RL signal quality.

## Architecture Overview

- **Specialized Agents**: Solver (generation), Verifier (error detection), Corrector (refinement); each decodes up to 64k tokens
- **Segment-Based Decoding**: Each agent decodes in 16k-token chunks; upon completion, outputs immediately enter training queue
- **Decoupled Reward Functions**: Solver rewarded on solution correctness, Verifier on discrimination, Corrector on refinement success
- **Grouped Agentic Rollouts**: Each problem generates 8 solver outputs; subsequent agents sample 2 outputs each for diverse training signals
- **Pipeline Training Queue**: Training begins on completed segments while downstream agents continue decoding—overlap eliminates bottlenecks

## Implementation Steps

**Step 1: Define Agent-Specific Rewards.** Assign rewards based on individual agent responsibility, not shared outcome.

```python
class DecoupledRewardFunction:
    def __init__(self, reference_answers):
        self.reference_answers = reference_answers

    def reward_solver(self, problem, solution, reference_answer):
        """
        Reward Solver based on solution correctness alone.
        Verifier and Corrector's quality is irrelevant to Solver's reward.
        """
        if self.is_correct(solution, reference_answer):
            return 1.0  # Positive reward for correct solution
        else:
            return -1.0  # Negative reward for incorrect

    def reward_verifier(self, problem, solution, is_correct, verifier_judgment):
        """
        Reward Verifier based on judgment accuracy.
        Correct judgment (regardless of downstream correction) yields reward.
        """
        ground_truth_correct = self.is_correct(solution, self.reference_answers[problem])

        if verifier_judgment == ground_truth_correct:
            return 1.0  # Correct judgment
        else:
            return -1.0  # Incorrect judgment (false positive or negative)

    def reward_corrector(self, problem, original_solution, corrected_solution, reference_answer):
        """
        Reward Corrector based on refinement success.
        Only corrections of originally incorrect solutions yield positive reward.
        """
        original_correct = self.is_correct(original_solution, reference_answer)
        corrected_correct = self.is_correct(corrected_solution, reference_answer)

        if not original_correct and corrected_correct:
            return 1.0  # Successfully fixed an error
        elif corrected_correct:
            return 0.5  # Correct but wasn't fixing an error
        else:
            return -1.0  # Failed to correct or made worse

    def is_correct(self, solution, reference):
        """Evaluate solution against reference."""
        # Problem-specific logic (e.g., exact match, semantic equivalence)
        return solution.strip() == reference.strip()
```

**Step 2: Implement Segment-Based Rollout.** Decode agents in 16k-token chunks and queue for training immediately.

```python
class SegmentRollout:
    def __init__(self, segment_size=16000, agent_budget=64000):
        self.segment_size = segment_size
        self.agent_budget = agent_budget
        self.training_queue = []

    def generate_solver_trajectory(self, problem, solver_model):
        """
        Solver generates solution in segments.
        Each segment enters training queue upon completion.
        """
        solution_tokens = []
        solver_model.reset_state()

        num_segments = self.agent_budget // self.segment_size
        for seg_idx in range(num_segments):
            # Decode one segment
            segment = solver_model.generate(
                problem,
                max_tokens=self.segment_size,
                state=solver_model.get_state()
            )
            solution_tokens.extend(segment)

            # Immediately queue for training (don't wait for full solution)
            self.training_queue.append({
                'agent': 'solver',
                'problem': problem,
                'partial_solution': ''.join(solution_tokens),
                'segment_idx': seg_idx,
                'trajectory': solver_model.get_trajectory()  # logits, actions, etc.
            })

            # Check for early stopping
            if solver_model.is_complete(solution_tokens):
                break

        return ''.join(solution_tokens)

    def process_training_queue(self, reward_fn):
        """
        Process queued segments for RL training.
        Training begins immediately; doesn't wait for downstream agents.
        """
        while self.training_queue:
            item = self.training_queue.pop(0)

            if item['agent'] == 'solver':
                # Compute reward for solver
                reward = reward_fn.reward_solver(
                    item['problem'],
                    item['partial_solution'],
                    # Use final reference for this problem
                )

                # Update solver model via RL (e.g., PPO, GRPO)
                rl_loss = compute_rl_loss(item['trajectory'], reward)
                solver_model.backward(rl_loss)

            # Similar for Verifier, Corrector
```

**Step 3: Grouped Agentic Rollouts.** Each problem generates multiple diverse trajectories per agent.

```python
def grouped_agentic_rollouts(problem, num_solutions=8):
    """
    Generate k solver outputs per problem;
    subsequent agents sample from these for diverse supervision.
    """
    solver_outputs = []
    for i in range(num_solutions):
        solution = solver_model.generate(problem)
        solver_outputs.append(solution)

    # Verifier evaluates all solver outputs
    verification_results = []
    for solution in solver_outputs:
        judgment = verifier_model.judge(problem, solution)
        verification_results.append({
            'solution': solution,
            'is_correct': judgment['is_correct'],
            'confidence': judgment['confidence']
        })

    # Corrector focuses on errors
    for result in verification_results:
        if not result['is_correct']:
            # Corrector attempts to refine error
            corrected = corrector_model.correct(
                problem,
                result['solution'],
                result['confidence']
            )

            # Add to training queue for corrector reward
            training_queue.append({
                'agent': 'corrector',
                'problem': problem,
                'original': result['solution'],
                'corrected': corrected
            })

    return {
        'solver_outputs': solver_outputs,
        'verification_results': verification_results
    }
```

## Practical Guidance

**When to Use:** Complex reasoning tasks (math, coding, multi-step proofs) where extended thinking helps, and where you can decompose reasoning into specialized subtasks (generation, verification, refinement).

**Architecture Choices:**
- Agent budgets: Solver 64k, Verifier 16k, Corrector 32k works well; adjust for problem complexity
- Segment size: 16k balances training latency vs coherence; smaller segments lose long-range context
- Number of diverse rollouts: 4–8 per problem; more increases diversity but cost

**Pitfalls:**
- **Reward contamination**: Avoid using downstream agent quality in upstream rewards; keeps decoupling clean
- **Pipeline bottleneck**: If one agent is much slower, pipeline doesn't parallelize; balance compute budgets
- **Adaptive sampling bias**: Corrector focusing on errors can reduce diversity; add exploration sampling occasionally
- **Training instability**: Multiple agent updates can interfere; use separate optimizers and learning rate schedules

**When NOT to Use:** Single-agent tasks (classification, simple QA) where decomposition adds overhead; problems not benefiting from verification or refinement.

**Integration:** Combine with outcome verification (automated test suites for code) for clearer reward signals; pairs well with expert demonstrations for initialization.

---
Reference: https://arxiv.org/abs/2511.11373
