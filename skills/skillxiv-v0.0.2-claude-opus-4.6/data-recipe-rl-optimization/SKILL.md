---
name: data-recipe-rl-optimization
title: "DataChef: Cooking Up Optimal Data Recipes via RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.11089"
keywords: [Data Augmentation, RL, Data Curation, Training Data, Dataset Optimization]
description: "Automatically synthesize and optimize training data using GRPO to generate data recipes (specifications for dataset creation). Use a Data Verifier to efficiently evaluate sample quality without full model training. Achieve performance comparable to human expert curation across diverse domains."
---

# DataChef: Cooking Up Optimal Data Recipes via RL

## Problem Context

Creating high-quality training data is labor-intensive and requires domain expertise. DataChef automates this by using RL to generate data recipes—specifications for how to create training datasets. A fast Data Verifier (rubric-based classifier) provides rewards without expensive downstream training, enabling rapid iteration on dataset composition.

## Core Concept

DataChef operates in two phases: (1) cold-start with supervised demonstrations from strong models, (2) RL phase where policies generate data recipes, verifiers assess quality, and GRPO optimizes recipe generation. Recipes specify: data source selection, augmentation strategies, filtering criteria, and mixing ratios.

## Implementation

### Step 1: Data recipe representation and generator

```python
from dataclasses import dataclass
from typing import Dict, List, Any
import json

@dataclass
class DataRecipe:
    """Specification for dataset creation."""
    sources: List[str]  # Data sources (e.g., ["wikipedia", "books"])
    augmentation_strategies: List[str]  # e.g., ["paraphrase", "back_translation"]
    filtering_criteria: Dict[str, Any]  # Quality thresholds
    mixing_ratios: Dict[str, float]  # Per-source weights
    sample_size: int

    def to_dict(self) -> Dict:
        return {
            'sources': self.sources,
            'augmentation': self.augmentation_strategies,
            'filters': self.filtering_criteria,
            'mixing': self.mixing_ratios,
            'size': self.sample_size
        }

class DataRecipeGenerator:
    """Generate data recipes via language model."""

    def __init__(self, model):
        self.model = model

    def generate_recipe(
        self,
        task: str,
        max_tokens: int = 300
    ) -> Dict:
        """
        Generate data recipe for task.

        Args:
            task: Task description
            max_tokens: Recipe generation length

        Returns:
            recipe_dict: Parsed recipe specification
        """
        prompt = f"""Generate a data recipe for training on: {task}

Specify:
1. Data sources (wikipedia, books, web, etc.)
2. Augmentation strategies (paraphrase, back_translation, etc.)
3. Quality filters (minimum length, diversity, etc.)
4. Mixing ratios for different sources
5. Target sample size

Format as JSON.

Data recipe:"""

        recipe_text, log_probs = self.model.generate_with_logprobs(
            prompt, max_tokens=max_tokens, temperature=0.7
        )

        # Parse recipe
        try:
            recipe_dict = json.loads(recipe_text)
        except json.JSONDecodeError:
            # Fallback: extract fields from free text
            recipe_dict = self._parse_recipe_freetext(recipe_text)

        return recipe_dict, log_probs

    def _parse_recipe_freetext(self, text: str) -> Dict:
        """Fallback parsing from free text."""
        return {
            'sources': ['wikipedia'],
            'augmentation': ['paraphrase'],
            'filters': {'min_length': 10},
            'mixing': {'wikipedia': 1.0},
            'size': 10000
        }
```

### Step 2: Data verifier (fast quality assessment)

```python
class DataVerifier:
    """Fast quality assessment without full model training."""

    def __init__(self, quality_rubric: Dict = None):
        """
        Args:
            quality_rubric: Scoring criteria for samples
        """
        self.rubric = quality_rubric or {
            'invalid': 0.0,
            'format_error': 0.1,
            'incorrect': 0.2,
            'task_mismatch': 0.5,
            'pass': 1.0
        }

    def verify_sample(
        self,
        sample: Dict,
        task: str,
        verifier_model=None
    ) -> float:
        """
        Classify sample quality quickly.

        Args:
            sample: Training sample (input, target)
            task: Task description
            verifier_model: Optional LLM for semantic checking

        Returns:
            score: Quality score (0-1)
        """
        # Format check
        if 'input' not in sample or 'target' not in sample:
            return self.rubric['format_error']

        input_text = sample['input']
        target_text = sample['target']

        # Length check
        if len(input_text.split()) < 5 or len(target_text.split()) < 1:
            return self.rubric['format_error']

        # Task relevance check (simple heuristic)
        if not self._is_task_relevant(input_text, task):
            return self.rubric['task_mismatch']

        # Correctness estimation
        if verifier_model:
            is_correct = verifier_model.verify(input_text, target_text)
            if is_correct:
                return self.rubric['pass']
            else:
                return self.rubric['incorrect']

        # Default: assume correct if format is valid
        return self.rubric['pass']

    def _is_task_relevant(self, text: str, task: str) -> bool:
        """Check task relevance (simplified)."""
        task_keywords = task.lower().split()
        text_lower = text.lower()
        return any(kw in text_lower for kw in task_keywords[:3])

    def verify_dataset(
        self,
        dataset: List[Dict],
        task: str,
        sample_fraction: float = 0.1
    ) -> float:
        """
        Estimate dataset quality via sampling.

        Args:
            dataset: Full dataset
            sample_fraction: Fraction to verify

        Returns:
            avg_quality_score: Mean quality across sample
        """
        import random
        sample_size = max(1, int(len(dataset) * sample_fraction))
        sample = random.sample(dataset, min(sample_size, len(dataset)))

        scores = [self.verify_sample(s, task) for s in sample]
        return sum(scores) / len(scores)
```

### Step 3: GRPO training for recipe generation

```python
class DataRecipeRL:
    """GRPO-based optimization of data recipes."""

    def __init__(
        self,
        model,
        optimizer,
        verifier: DataVerifier,
        data_executor,
        group_size: int = 4
    ):
        self.model = model
        self.optimizer = optimizer
        self.verifier = verifier
        self.executor = data_executor  # Creates datasets from recipes
        self.group_size = group_size
        self.recipe_generator = DataRecipeGenerator(model)

    def compute_recipe_reward(
        self,
        recipe: Dict,
        task: str,
        downstream_verifier=None
    ) -> float:
        """
        Compute reward for recipe.

        Args:
            recipe: Generated recipe
            task: Task for which data is created
            downstream_verifier: Optional model for full eval

        Returns:
            reward: Quality score
        """
        try:
            # Execute recipe: create dataset
            dataset = self.executor.create_dataset(recipe, task, sample_size=100)

            # Verify dataset
            quality_score = self.verifier.verify_dataset(dataset, task)

            # Normalize reward
            reward = quality_score

        except Exception as e:
            # Recipe failed to execute
            reward = 0.0

        return reward

    def training_step(
        self,
        tasks: List[str],
        batch_size: int = 4
    ) -> Dict:
        """Single training step."""
        log_probs_list = []
        rewards = []

        for task in tasks[:batch_size]:
            # Generate recipe
            recipe, log_probs = self.recipe_generator.generate_recipe(task)
            log_probs_list.append(log_probs)

            # Compute reward
            reward = self.compute_recipe_reward(recipe, task)
            rewards.append(reward)

        log_probs = torch.stack(log_probs_list)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # GRPO loss
        log_prob_ratio = log_probs - log_probs.detach()
        ratio = torch.exp(log_prob_ratio)

        # Group relative advantage
        group_mean_reward = rewards.mean()
        relative_rewards = rewards - group_mean_reward

        clipped_ratio = torch.clamp(ratio, 0.5, 2.0)
        loss = -torch.min(
            log_prob_ratio * relative_rewards.unsqueeze(-1),
            torch.log(clipped_ratio) * relative_rewards.unsqueeze(-1)
        ).mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_quality': rewards.mean().item()
        }
```

### Step 4: Cold-start with synthetic demonstrations

```python
def generate_cold_start_demonstrations(
    strong_model,
    tasks: List[str],
    num_demos_per_task: int = 5
) -> List[Dict]:
    """
    Generate high-quality data recipes from strong model.

    Uses strong model (e.g., GPT-4) to create initial demonstrations.
    """
    demonstrations = []

    for task in tasks:
        for _ in range(num_demos_per_task):
            # Ask strong model to create data recipe
            prompt = f"""Create a high-quality data recipe for: {task}

Consider data sources, augmentation, filtering, and quality thresholds.
Return as a concrete JSON specification."""

            recipe_json = strong_model.generate(prompt, max_tokens=300)

            demo = {
                'task': task,
                'recipe': recipe_json,
                'is_high_quality': True
            }
            demonstrations.append(demo)

    return demonstrations


def train_datachef_with_cold_start(
    model,
    strong_model,
    tasks: List[str],
    data_executor,
    optimizer,
    num_epochs: int = 10,
    num_rl_steps_per_epoch: int = 100
):
    """
    Full DataChef training: cold-start SFT → RL.
    """
    # Cold-start
    print("Cold-start: Supervised fine-tuning")
    demos = generate_cold_start_demonstrations(strong_model, tasks, num_demos_per_task=5)

    for demo in demos:
        prompt = f"Task: {demo['task']}\n\nData recipe:"
        target = demo['recipe']

        # SFT loss (simplified)
        # Would compute cross-entropy on tokens

    # RL phase
    print("RL phase: GRPO optimization")
    verifier = DataVerifier()
    rl_trainer = DataRecipeRL(model, optimizer, verifier, data_executor)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_reward = 0.0

        for step in range(num_rl_steps_per_epoch):
            metrics = rl_trainer.training_step(tasks, batch_size=4)
            total_loss += metrics['loss']
            total_reward += metrics['avg_reward']

        avg_loss = total_loss / num_rl_steps_per_epoch
        avg_reward = total_reward / num_rl_steps_per_epoch

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")

    return model
```

## Practical Guidance

**When to use**: Data curation bottleneck; need diverse training data; frequent dataset regeneration

**Hyperparameters**:
- **verification_sample_fraction**: 0.05-0.2 (tradeoff: cost vs. confidence)
- **group_size**: 4-8 (GRPO)
- **cold_start_demos**: 3-10 per task
- **recipe_max_tokens**: 200-400

**Key advantages**:
- Automates data curation (labor savings)
- Fast verification without full training
- Curriculum: SFT → RL smooths learning
- Generalizes across domains

**Common pitfalls**:
- Verifier too lenient → low-quality recipes
- Not validating executor can actually create dataset
- Cold-start too brief → RL starts from bad initialization
- Recipe too constrained → fails to explore

**Scaling**: Verification cost is the main bottleneck; parallelize across tasks.

## Reference

Paper: https://arxiv.org/abs/2602.11089
Related work: Data augmentation, dataset curation, GRPO
Benchmarks: Diverse domains including math, code, QA
