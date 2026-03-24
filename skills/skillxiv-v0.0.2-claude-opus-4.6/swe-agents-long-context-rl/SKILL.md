---
name: swe-agents-long-context-rl
title: Training Multi-Turn Software Engineering Agents with RL
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03501
keywords: [software-engineering, reinforcement-learning, long-context, code-understanding]
description: "Train LLM-based agents for multi-turn SWE tasks via rejection fine-tuning and DAPO RL, scaling to 131k context length achieving 39% Pass@1."
---

## SWE Agents: Long-Context RL for Code Understanding

This work trains 72B parameter agents for multi-turn software engineering tasks—locating bugs, implementing fixes, running tests—requiring dozens of interaction steps. A two-phase training pipeline (rejection-sampled SFT → DAPO RL) combined with careful handling of long contexts and distribution mismatches achieves 39% Pass@1 on SWE-bench Verified, competitive with much larger models.

### Core Concept

Software engineering requires sustained, multi-turn reasoning: read code → find bug → implement fix → test → iterate. This is fundamentally different from single-turn language modeling. The approach uses rejection-sampled fine-tuning to teach basic tool-calling, then RL to learn exploration and multi-step planning. Critical insights: properly model the POMDP structure, handle long-context issues (131k tokens), and manage sampling biases that violate importance sampling assumptions.

### Architecture Overview

- **Two-Phase Training**: Phase 1 (RFT): rejection-sampled SFT on successful trajectories; Phase 2 (DAPO RL): multi-turn RL with corrected importance sampling
- **POMDP Formulation**: Proper partially-observable Markov model for multi-turn interaction (unlike bandit approximations)
- **Long-Context Scaling**: Handle 131k token contexts without positional encoding disruptions
- **Distribution Mismatch Handling**: Correct for biased sampling (e.g., decoding filters)
- **Data Curation**: Filter for correctness, controlled complexity, non-flaky tests, LLM-assessed quality

### Implementation Steps

**Step 1: Implement Rejection-Sampled Fine-Tuning**

```python
from typing import List, Tuple, Dict
import random

class RejectionSampledFT:
    """
    Phase 1: Collect trajectories from base model, keep only successful ones.
    Rejection sampling filters to high-quality data.
    """

    def __init__(self, base_model, code_executor):
        self.model = base_model
        self.executor = code_executor
        self.successful_trajectories = []

    def sample_trajectories(self, task: str, num_samples: int = 10) -> List[Dict]:
        """
        Sample multiple trajectories for single task.
        Keep only those that solve the task.
        """
        trajectories = []

        for _ in range(num_samples):
            trajectory = {
                'steps': [],
                'success': False,
                'task': task,
                'test_results': None
            }

            # Run agent step-by-step
            current_state = f"Task: {task}"

            for step in range(20):  # Max 20 steps per task
                # LLM decides action
                prompt = f"State: {current_state}\nNext action?"
                action = self.model.generate(prompt, temperature=0.8, max_tokens=500)

                trajectory['steps'].append({
                    'prompt': prompt,
                    'action': action,
                    'step_num': step
                })

                # Execute action (read file, run tests, etc.)
                try:
                    result = self.executor.execute(action)
                except Exception as e:
                    result = f"Error: {str(e)}"

                current_state = result

                # Check if task is solved
                if 'tests passed' in result.lower():
                    trajectory['success'] = True
                    trajectory['test_results'] = result
                    break

            trajectories.append(trajectory)

        return trajectories

    def collect_rft_data(self, tasks: List[str], num_samples_per_task: int = 10) -> List[Dict]:
        """
        Collect rejection-sampled data: only successful trajectories.
        """
        rft_data = []

        for task in tasks:
            trajectories = self.sample_trajectories(task, num_samples_per_task)

            # Keep only successful
            successful = [t for t in trajectories if t['success']]

            for traj in successful:
                # Convert trajectory to training example
                for step in traj['steps']:
                    rft_data.append({
                        'input': step['prompt'],
                        'output': step['action'],
                        'task': task,
                        'success': True
                    })

        return rft_data

    def train_rft(self, rft_data: List[Dict], learning_rate: float = 1e-5):
        """
        Supervised fine-tuning on successful trajectories.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for example in rft_data:
            # Forward
            logits = self.model(example['input'])

            # Loss: cross-entropy on correct action
            loss = torch.nn.functional.cross_entropy(
                logits,
                self.model.tokenizer.encode(example['output'])
            )

            # Backward
            loss.backward()

        optimizer.step()
```

**Step 2: Implement Long-Context Handling**

```python
class LongContextManager:
    """
    Manage long contexts (131k tokens) without positional encoding issues.
    """

    def __init__(self, model, max_context_length: int = 131072):
        self.model = model
        self.max_length = max_context_length
        self.context_cache = {}

    def truncate_context(self, full_context: str, relevant_keywords: List[str] = None) -> str:
        """
        Smart truncation: keep most relevant parts when context exceeds limit.
        """
        if len(full_context) < self.max_length:
            return full_context

        # If keywords provided, prioritize sections containing them
        if relevant_keywords:
            lines = full_context.split('\n')
            relevant_lines = []

            for line in lines:
                if any(kw.lower() in line.lower() for kw in relevant_keywords):
                    relevant_lines.append(line)

            if relevant_lines:
                truncated = '\n'.join(relevant_lines[-1000:])  # Keep recent relevant
                return truncated[:self.max_length]

        # Otherwise keep recent context (recency bias)
        return full_context[-self.max_length:]

    def split_context_into_windows(self, context: str, window_size: int = 8192,
                                   overlap: int = 1024) -> List[str]:
        """
        Split long context into overlapping windows for processing.
        """
        windows = []
        current_pos = 0

        while current_pos < len(context):
            window_end = min(current_pos + window_size, len(context))
            window = context[current_pos:window_end]
            windows.append(window)

            current_pos += window_size - overlap

        return windows

    def process_windowed_context(self, full_context: str, query: str) -> str:
        """
        Process query against long context using windowing.
        """
        windows = self.split_context_into_windows(full_context)
        responses = []

        for window in windows:
            prompt = f"Context:\n{window}\n\nQuery: {query}"
            response = self.model.generate(prompt, max_tokens=100)
            responses.append(response)

        # Aggregate responses (simple: concatenate relevant ones)
        aggregated = '\n'.join(responses)
        return aggregated[:2000]  # Limit aggregated response
```

**Step 3: Implement DAPO (Distributed Asynchronous Policy Optimization)**

```python
class DAPO:
    """
    Multi-turn RL with corrected importance sampling.
    DAPO handles the full POMDP structure and distribution mismatch.
    """

    def __init__(self, model, gamma: float = 0.99, lambda_coef: float = 0.95):
        self.model = model
        self.gamma = gamma
        self.lambda_coef = lambda_coef

    def compute_advantages(self, trajectory: List[Dict], rewards: List[float]) -> List[float]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        Properly handles temporal credit assignment.
        """
        advantages = []
        gae = 0

        # Backward pass through trajectory
        for t in reversed(range(len(trajectory))):
            reward = rewards[t]
            value_t = trajectory[t].get('value', 0.0)  # Estimated value
            value_next = trajectory[t + 1].get('value', 0.0) if t + 1 < len(trajectory) else 0.0

            # TD error
            td_error = reward + self.gamma * value_next - value_t

            # GAE: exponential moving average of TD errors
            gae = td_error + self.gamma * self.lambda_coef * gae
            advantages.insert(0, gae)

        return advantages

    def correct_importance_sampling(self, sampled_actions: List, policy_dist: List,
                                   behavior_dist: List) -> List[float]:
        """
        Correct for distribution mismatch in sampled trajectories.
        This prevents biased training when trajectory collection has filters.
        """
        importance_weights = []

        for action, pi_prob, beta_prob in zip(sampled_actions, policy_dist, behavior_dist):
            # Importance weight: pi(a|s) / beta(a|s)
            # But use clipping to avoid extreme weights
            weight = min(pi_prob / (beta_prob + 1e-6), 1.5)
            importance_weights.append(weight)

        return importance_weights

    def train_step(self, trajectories: List[List[Dict]], rewards_per_trajectory: List[List[float]]):
        """
        One RL training step across multiple trajectories.
        """
        for trajectory, rewards in zip(trajectories, rewards_per_trajectory):
            # Compute advantages
            advantages = self.compute_advantages(trajectory, rewards)

            # Compute policy gradient for each action
            for step, advantage in zip(trajectory, advantages):
                # Get policy logits
                prompt = step['prompt']
                logits = self.model(prompt)

                # Action log probability
                action_logp = torch.nn.functional.log_softmax(logits, dim=-1)[step['action_idx']]

                # Policy gradient: log_pi(a|s) * advantage
                # (with clipping for stability)
                pg_loss = -action_logp * advantage
                pg_loss = torch.clamp(pg_loss, -1.0, 1.0)

                pg_loss.backward()

        self.model.optimizer.step()

    def handle_sampling_bias(self, trajectories: List[List[Dict]]):
        """
        Handle biases from trajectory sampling.
        E.g., filtering out invalid actions changes distribution.
        """
        # Track which actions were filtered/modified
        for trajectory in trajectories:
            for step in trajectory:
                # If step had invalid action filtered, adjust weight
                if step.get('was_filtered', False):
                    # Downweight this step in RL (behavior dist ≠ policy dist)
                    step['importance_weight'] = 0.5  # Heuristic
```

**Step 4: Data Curation Pipeline**

```python
class DataCuration:
    """
    Filter training data for quality and remove noisy examples.
    """

    def __init__(self, code_executor, llm):
        self.executor = code_executor
        self.llm = llm

    def assess_correctness(self, solution: str, test_output: str) -> bool:
        """Check if solution passes all tests."""
        return 'passed' in test_output.lower() and 'failed' not in test_output.lower()

    def assess_complexity(self, task_description: str) -> int:
        """Estimate task difficulty (1-10)."""
        # Heuristic: number of files, lines of code, etc.
        num_files = task_description.count('file:')
        num_lines = task_description.count('\n')
        complexity = min(10, num_files + num_lines // 100)
        return complexity

    def assess_flakiness(self, trajectory: List[Dict]) -> bool:
        """Check if solution is flaky (non-deterministic failures)."""
        # Run twice, check if results match
        run1_success = trajectory.get('success_run1', False)
        run2_success = trajectory.get('success_run2', False)

        return run1_success != run2_success  # Flaky if different

    def llm_assess_quality(self, solution: str) -> float:
        """Use LLM to rate solution quality (0-1)."""
        prompt = f"Rate the code quality (0-10):\n{solution[:1000]}"
        rating = self.llm.generate(prompt, max_tokens=5)

        try:
            score = float(rating) / 10
            return score
        except:
            return 0.5

    def curate_dataset(self, raw_trajectories: List[Dict]) -> List[Dict]:
        """Filter trajectories by quality criteria."""
        curated = []

        for trajectory in raw_trajectories:
            # Criterion 1: Correctness
            if not self.assess_correctness(trajectory['solution'], trajectory['test_output']):
                continue

            # Criterion 2: Controlled complexity (avoid too easy or too hard)
            complexity = self.assess_complexity(trajectory['task'])
            if complexity < 2 or complexity > 8:
                continue

            # Criterion 3: Non-flaky
            if self.assess_flakiness(trajectory):
                continue

            # Criterion 4: Quality (LLM assessment)
            quality = self.llm_assess_quality(trajectory['solution'])
            if quality < 0.6:
                continue

            curated.append(trajectory)

        return curated
```

**Step 5: Full Training Pipeline**

```python
def train_swe_agent(model, code_executor, all_tasks: List[str],
                   num_epochs: int = 3):
    """
    Complete training: RFT → long-context handling → DAPO RL.
    """
    # Phase 1: Rejection-Sampled FT
    print("Phase 1: Rejection-Sampled Fine-Tuning")
    rft_trainer = RejectionSampledFT(model, code_executor)
    rft_data = rft_trainer.collect_rft_data(all_tasks[:50], num_samples_per_task=10)

    print(f"  Collected {len(rft_data)} successful trajectories")
    rft_trainer.train_rft(rft_data)

    # Phase 2: RL Training
    print("Phase 2: DAPO RL Training")
    dapo = DAPO(model)
    context_mgr = LongContextManager(model)
    data_curator = DataCuration(code_executor, model)

    for epoch in range(num_epochs):
        trajectories = []
        rewards = []

        for task in all_tasks:
            # Sample trajectory
            traj = rft_trainer.sample_trajectories(task, num_samples=1)[0]

            # Curate
            if data_curator.assess_correctness(traj.get('solution', ''),
                                              traj.get('test_results', '')):
                trajectories.append(traj['steps'])

                # Compute reward
                num_steps = len(traj['steps'])
                reward = 1.0 - 0.01 * num_steps  # Reward efficiency

                rewards.append([reward] * len(traj['steps']))

        # RL training step
        dapo.train_step(trajectories, rewards)

        print(f"Epoch {epoch}: Trained on {len(trajectories)} trajectories")

    return model
```

### Practical Guidance

**When to Use:**
- Multi-turn coding tasks (bug localization, implementation)
- Long-context understanding required (131k tokens)
- Scenarios with differentiable reward signals (test passing)
- Tasks requiring sustained reasoning

**When NOT to Use:**
- Single-turn coding tasks (standard fine-tuning sufficient)
- Domains without reliable test harnesses
- Real-time inference (training is slow, inference requires full context)

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `rft_samples_per_task` | 10 | Higher = more diverse training, slower collection |
| `max_context_length` | 131k | Match model's context window; larger = more coverage |
| `complexity_range` | 2-8 | Filter out too-easy and too-hard tasks |
| `rl_epochs` | 3 | More epochs = better convergence, more training time |

### Reference

**Paper**: Training Long-Context Multi-Turn Software Engineering Agents with RL (2508.03501)
- 39% Pass@1 on SWE-bench Verified
- Competitive with much larger models like DeepSeek-V3
- Handles 131k token contexts effectively
- Corrected importance sampling for stable RL
