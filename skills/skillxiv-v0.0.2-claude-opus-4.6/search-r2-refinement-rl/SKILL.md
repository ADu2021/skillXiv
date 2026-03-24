---
name: search-r2-refinement-rl
title: "Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03647"
keywords: [Reasoning Refinement, Search Integration, RL Training, Trajectory Correction, GRPO Optimization]
description: "Decompose search-integrated reasoning into actor (generates trajectories) and refiner (identifies and corrects flawed steps). Refiner performs surgical cut-and-regenerate correction preserving valid prefixes. Trained end-to-end with GRPO; adds minimal overhead (2%) to baseline."
---

# Search-R2: Actor-Refiner Architecture for Search-Integrated Reasoning

When solving complex problems with web search integration, agents often generate reasoning trajectories with flawed intermediate steps that invalidate later reasoning. Rather than discarding entire trajectories, Search-R2 uses a two-role system: the actor generates candidate reasoning paths with search queries, and the refiner identifies specific errors and regenerates only the problematic suffix. This targeted correction preserves valid reasoning while fixing errors efficiently.

The key insight is that not all trajectory errors require full re-reasoning. By pinpointing the exact error location and regenerating only that segment, the system maintains coherence while enabling effective learning via RL.

## Core Concept

Search-R2 operates on two complementary roles:

1. **Actor Policy**: Generates search-integrated reasoning trajectories with search queries
2. **Meta-Refiner**: Analyzes trajectories, identifies flawed reasoning, and performs targeted correction

The system uses a discriminator to detect global coherence issues and a trimmer to pinpoint error locations, then regenerates optimal suffixes.

## Architecture Overview

- **Actor Agent**: Standard reasoning agent with search tool access
- **Trajectory Analyzer**: Identifies steps with consistency issues
- **Error Detector**: Binary classifier identifying whether error exists
- **Error Locator**: Pinpoints exact step where reasoning diverges
- **Suffix Generator**: Regenerates reasoning from error point forward
- **GRPO Trainer**: Optimizes both actor and refiner end-to-end

## Implementation

### Step 1: Build Trajectory Analysis Module

Create tools to analyze reasoning trajectories for errors.

```python
# Trajectory analysis
class TrajectoryAnalyzer:
    def __init__(self, model: str = "gpt-4-turbo"):
        """Analyze search-integrated reasoning trajectories."""
        self.model = model

    def extract_reasoning_steps(self, trajectory: str) -> List[dict]:
        """Break trajectory into individual reasoning steps."""
        # Parse trajectory: thought, action, observation, thought, ...
        steps = []
        current_step = {"thought": "", "action": "", "observation": ""}

        lines = trajectory.split("\n")
        for line in lines:
            if line.startswith("Thought:"):
                if current_step["thought"]:  # Save previous step
                    steps.append(current_step.copy())
                    current_step = {"thought": "", "action": "", "observation": ""}
                current_step["thought"] = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                current_step["action"] = line.replace("Action:", "").strip()
            elif line.startswith("Observation:"):
                current_step["observation"] = line.replace("Observation:", "").strip()

        if current_step["thought"]:
            steps.append(current_step)

        return steps

    def compute_trajectory_embedding(self, trajectory: str) -> torch.Tensor:
        """Embed trajectory for consistency analysis."""
        # Pseudo-code: actual implementation uses embeddings
        embedding = torch.randn(768)  # 768-dim embedding
        return embedding / torch.norm(embedding)

    def measure_consistency(
        self,
        step_idx: int,
        trajectory_steps: List[dict],
        full_trajectory: str
    ) -> float:
        """
        Measure consistency of reasoning at step.

        Returns:
            Score [0, 1]: 1 = fully consistent, 0 = contradictory
        """
        if step_idx == 0:
            return 1.0  # First step always consistent

        prev_steps = trajectory_steps[:step_idx]
        current_step = trajectory_steps[step_idx]

        # Compare logical flow
        context = "\n".join([
            f"Step {i}: {s['thought']}"
            for i, s in enumerate(prev_steps)
        ])

        consistency_prompt = f"""
Given this reasoning context:
{context}

Is this next step logically consistent?
{current_step['thought']}

Rate consistency 0-10:"""

        score = self.model.generate_number(consistency_prompt)
        return score / 10.0

    def identify_error_location(
        self,
        trajectory_steps: List[dict],
        full_trajectory: str
    ) -> Optional[int]:
        """
        Find the step index where reasoning first becomes inconsistent.

        Returns:
            Index of first inconsistent step, or None if trajectory is sound
        """
        error_threshold = 0.5

        for step_idx in range(1, len(trajectory_steps)):
            consistency = self.measure_consistency(
                step_idx,
                trajectory_steps,
                full_trajectory
            )

            if consistency < error_threshold:
                return step_idx

        return None
```

### Step 2: Implement Error Detection and Localization

Create discriminator that identifies presence and location of errors.

```python
# Error detection module
class ErrorDetector(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768):
        """Binary classifier for trajectory errors."""
        super().__init__()

        self.trajectory_encoder = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.error_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, trajectory_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify trajectory for errors.

        Args:
            trajectory_embeddings: [batch, num_steps, hidden_dim]

        Returns:
            logits: [batch] binary classification logits
        """
        _, hidden = self.trajectory_encoder(trajectory_embeddings)
        logits = self.error_classifier(hidden.squeeze(0))
        return logits

class ErrorLocator(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768):
        """Identify which step contains the error."""
        super().__init__()

        self.step_scorer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        trajectory_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Score each step for error likelihood.

        Args:
            trajectory_embeddings: [batch, num_steps, hidden_dim]

        Returns:
            scores: [batch, num_steps] error likelihood per step
        """
        batch_size, num_steps, hidden_dim = trajectory_embeddings.shape
        scores = []

        for step_idx in range(num_steps):
            # Concatenate context (steps before) with current step
            if step_idx == 0:
                context = torch.zeros(batch_size, hidden_dim,
                                     device=trajectory_embeddings.device)
            else:
                context = trajectory_embeddings[:, :step_idx].mean(dim=1)

            current = trajectory_embeddings[:, step_idx]
            combined = torch.cat([context, current], dim=-1)

            step_score = self.step_scorer(combined)
            scores.append(step_score)

        scores = torch.cat(scores, dim=-1)  # [batch, num_steps]
        return scores
```

### Step 3: Implement Cut-and-Regenerate Mechanism

Create the refiner that performs surgical trajectory correction.

```python
# Refiner: cut-and-regenerate
class TrajectoryRefiner:
    def __init__(self, model: str = "gpt-4-turbo"):
        """Refine trajectories by targeted regeneration."""
        self.model = model
        self.analyzer = TrajectoryAnalyzer(model)

    def refine_trajectory(
        self,
        original_trajectory: str,
        error_step_idx: int
    ) -> str:
        """
        Cut trajectory at error and regenerate suffix.

        Args:
            original_trajectory: Original reasoning chain
            error_step_idx: Index of first erroneous step

        Returns:
            Refined trajectory with corrected suffix
        """
        steps = self.analyzer.extract_reasoning_steps(original_trajectory)

        # Keep prefix up to error
        prefix = "\n".join([
            f"Thought: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}"
            for s in steps[:error_step_idx]
        ])

        # Identify the problem at error point
        problem_context = prefix + "\n\n[Error detected above]"

        # Prompt refiner to generate corrected suffix
        regeneration_prompt = f"""
The following reasoning has an error:

{problem_context}

Generate a corrected continuation that fixes the error and reaches a conclusion.
Format: Thought: ... Action: ... Observation: ...

Corrected suffix:"""

        corrected_suffix = self.model.generate(
            regeneration_prompt,
            max_tokens=512
        )

        # Combine prefix with corrected suffix
        refined_trajectory = prefix + "\n\n" + corrected_suffix
        return refined_trajectory

    def batch_refine(
        self,
        trajectories: List[str],
        error_indices: List[Optional[int]]
    ) -> List[str]:
        """Refine multiple trajectories in parallel."""
        refined = []

        for trajectory, error_idx in zip(trajectories, error_indices):
            if error_idx is not None:
                refined_traj = self.refine_trajectory(trajectory, error_idx)
                refined.append(refined_traj)
            else:
                refined.append(trajectory)  # No error, keep original

        return refined
```

### Step 4: Implement GRPO Training with Refinement

Train actor and refiner jointly using GRPO.

```python
# GRPO training with refinement
def train_search_r2(
    actor_model: nn.Module,
    error_detector: ErrorDetector,
    error_locator: ErrorLocator,
    dataset: List[dict],
    num_epochs: int = 10,
    group_size: int = 4
):
    """
    Train Search-R2 with actor and refiner via GRPO.

    Args:
        actor_model: Reasoning agent
        error_detector: Binary error classifier
        error_locator: Error location identifier
        dataset: Training tasks
        num_epochs: Training epochs
        group_size: GRPO group size
    """
    optimizer = torch.optim.AdamW(
        list(actor_model.parameters()) +
        list(error_detector.parameters()) +
        list(error_locator.parameters()),
        lr=1e-5
    )

    refiner = TrajectoryRefiner()

    for epoch in range(num_epochs):
        # Group trajectories for GRPO
        for group_idx in range(0, len(dataset), group_size):
            group = dataset[group_idx:group_idx + group_size]
            group_rewards = []
            trajectories_refined = []

            for task in group:
                # Generate trajectory
                trajectory = actor_model.generate(task["description"])

                # Analyze for errors
                trajectory_emb = torch.randn(1, 10, 768)  # Pseudo embeddings
                error_logits = error_detector(trajectory_emb)
                has_error = error_logits > 0

                if has_error:
                    # Localize error
                    error_scores = error_locator(trajectory_emb)
                    error_step_idx = torch.argmax(error_scores).item()

                    # Refine trajectory
                    refined = refiner.refine_trajectory(
                        trajectory,
                        error_step_idx
                    )
                else:
                    refined = trajectory

                trajectories_refined.append(refined)

                # Evaluate
                reward = evaluate_trajectory(refined, task["target"])
                group_rewards.append(reward)

            # GRPO: compute advantages and update
            group_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            group_mean = group_tensor.mean()

            for traj, reward in zip(trajectories_refined, group_rewards):
                advantage = reward - group_mean

                if advantage != 0:
                    loss = -advantage * compute_log_prob(actor_model, traj)
                    loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}: avg_reward={group_tensor.mean():.4f}")

    return actor_model, error_detector, error_locator
```

### Step 5: Inference Pipeline

Use trained models for deployment.

```python
# Inference with refinement
def search_r2_inference(
    actor: nn.Module,
    error_detector: ErrorDetector,
    error_locator: ErrorLocator,
    query: str,
    max_refinement_iterations: int = 1
) -> str:
    """
    Generate response with optional refinement.

    Args:
        actor: Trained reasoning agent
        error_detector: Error detector
        error_locator: Error locator
        query: User query
        max_refinement_iterations: How many refinement passes

    Returns:
        Final refined response
    """
    refiner = TrajectoryRefiner()

    # Generate initial trajectory
    trajectory = actor.generate(query)

    # Iteratively refine
    for iteration in range(max_refinement_iterations):
        # Check for errors
        trajectory_emb = torch.randn(1, 10, 768)  # Pseudo embeddings
        error_logits = error_detector(trajectory_emb)

        if error_logits <= 0:
            break  # No error detected, done

        # Localize and fix
        error_scores = error_locator(trajectory_emb)
        error_idx = torch.argmax(error_scores).item()

        trajectory = refiner.refine_trajectory(trajectory, error_idx)

    return trajectory
```

## Practical Guidance

**When to use Search-R2:**
- Complex reasoning tasks with web search integration
- Scenarios where reasoning trajectories have logical errors
- Problems where intermediate steps can be independently refined
- RL-based training where full trajectory discarding is wasteful

**When not to use:**
- Simple lookup tasks without complex reasoning
- Real-time systems where refinement overhead matters
- Tasks where any trajectory modification breaks coherence
- Systems requiring deterministic behavior

**Common Pitfalls:**
- Error detection false positives: Over-aggressive detection refines correct reasoning
- Prefix contamination: Ensuring cut point doesn't bias suffix generation
- Circular refinement: Error refiner introduces new errors; set max_iterations
- Search query consistency: Regenerated suffix may require different searches

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| error_threshold | 0.3-0.7 | Higher = fewer refinements; lower = more aggressive |
| max_refinement | 1-3 iterations | 1 typical; more for complex reasoning |
| group_size | 4-8 | Standard GRPO setting |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03647

Key results: Outperforms baseline Search-R1; minimal 2-9% overhead. Demonstrates surgical correction preserves reasoning quality. Code and training details released. Applicable to any search-integrated agent.
