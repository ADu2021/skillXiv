---
name: srpo-multimodal-reflection-rl
title: "SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.01713"
keywords: [Multimodal, Reinforcement Learning, Reflection, Self-Correction]
description: "Teach multimodal language models to reflect on their reasoning and improve answers through structured RL training."
---

# SRPO: Teach MLLMs to Think Out Loud and Correct Themselves

Multimodal large language models struggle with complex reasoning because they lack the ability to reflect: to pause, analyze their work, and correct mistakes. SRPO adds explicit self-reflection to MLLMs through a two-stage framework. First, generate high-quality reflection examples using an advanced model. Second, train using Group Relative Policy Optimization to reward both correct answers and meaningful, concise reflections that avoid redundancy. This teaches models to reason visually, express uncertainty, and self-correct—improving performance on tasks like visual math and detailed understanding.

## Core Concept

Explicit reflection is a trainable skill. Most MLLMs generate answers directly; they don't ask themselves "Wait, let me reconsider this visual detail." SRPO makes reflection explicit and trainable by creating datasets of reflection-augmented reasoning, then optimizing a reward function that values both answer correctness and reflection quality. Good reflections are concise, identify errors, propose corrections, and avoid repeating information already stated.

## Architecture Overview

- **Reflection Dataset Construction**: Use advanced MLLM to generate high-quality reflections on reasoning problems; reflections explain uncertainty, identify visual details, propose corrections
- **Dual-Component Rewards**: Separate rewards for answer correctness and reflection quality; avoid rewarding verbose or redundant reflections
- **GRPO Training Loop**: Group Relative Policy Optimization compares samples within groups, upweighting high-reward reflections while downweighting low-reward ones
- **Baseline Multimodal Model**: Standard MLLM architecture (vision transformer + language model); no architectural changes needed
- **Benchmark Integration**: Training on visual math (MathVista), detailed understanding (MMMU-Pro), and multimodal reasoning benchmarks

## Implementation

This implementation demonstrates reflection-aware RL for multimodal models.

First, build a reflection dataset generator using an advanced model:

```python
import torch
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ReflectionExample:
    image_id: str
    problem_text: str
    image: torch.Tensor
    initial_answer: str
    reflection: str
    correct_answer: str
    is_correct: bool

class ReflectionDatasetGenerator:
    """Generate reflection-augmented reasoning examples."""

    def __init__(self, mllm_name: str = "openai/clip-vit-large-patch14"):
        self.processor = CLIPProcessor.from_pretrained(mllm_name)
        self.vision_model = CLIPVisionModel.from_pretrained(mllm_name)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.language_model = AutoModelForCausalLM.from_pretrained("gpt2-large")

    def generate_reflection(self, problem_text: str, image: torch.Tensor,
                           initial_answer: str) -> str:
        """
        Generate high-quality reflection using advanced model.
        Reflection format:
        - Identifies what's visible in image
        - Explains reasoning
        - Identifies errors if any
        - Proposes correction
        """
        # Process image
        image_inputs = self.processor(images=[image], return_tensors="pt")

        # Build reflection prompt
        prompt = f"""Problem: {problem_text}
I initially answered: {initial_answer}

Looking at the image more carefully, here's my reflection:
- Visual details I notice:
- My reasoning was:
- I may have made an error:
- The correct answer should be:"""

        # Generate reflection (placeholder - in practice use advanced model)
        reflection = "Looking at the image, I can see the key visual elements. " \
                    f"My initial answer of {initial_answer} needs revision because " \
                    "I overlooked an important detail. The correct answer should " \
                    "account for this visual information."

        return reflection

    def create_reflection_dataset(self, problems: List[Dict]) -> List[ReflectionExample]:
        """
        Transform raw problem set into reflection-augmented dataset.
        Assumes problems have: image, text, correct_answer
        """
        examples = []

        for i, problem in enumerate(problems):
            # Generate initial answer (could be from base model)
            initial_answer = self._get_initial_answer(problem)

            # Generate reflection
            reflection = self.generate_reflection(
                problem["text"],
                problem["image"],
                initial_answer
            )

            # Check if initial answer was correct
            is_correct = (initial_answer == problem["correct_answer"])

            example = ReflectionExample(
                image_id=f"img_{i}",
                problem_text=problem["text"],
                image=problem["image"],
                initial_answer=initial_answer,
                reflection=reflection,
                correct_answer=problem["correct_answer"],
                is_correct=is_correct
            )
            examples.append(example)

        return examples

    def _get_initial_answer(self, problem: Dict) -> str:
        """Get initial answer from base model (placeholder)."""
        return "42"  # Placeholder

# Create dataset
generator = ReflectionDatasetGenerator()
reflection_examples = generator.create_reflection_dataset(
    [
        {
            "image": torch.randn(3, 224, 224),
            "text": "What is 25% of 80?",
            "correct_answer": "20"
        }
    ]
)
```

Implement dual-component reward model:

```python
class ReflectionRewardModel:
    """Score both answer correctness and reflection quality."""

    def __init__(self):
        # Answer evaluator: checks correctness
        self.answer_verifier = self._build_verifier()
        # Reflection evaluator: scores quality
        self.reflection_scorer = self._build_scorer()

    def _build_verifier(self):
        """Verifier that checks answer correctness."""
        def verify(answer: str, correct_answer: str) -> float:
            if answer.strip().lower() == correct_answer.strip().lower():
                return 1.0
            return 0.0
        return verify

    def _build_scorer(self):
        """Scorer that evaluates reflection quality."""
        def score_reflection(reflection: str, problem: str, image_analysis: str) -> float:
            # Multi-component reflection score
            score = 0.0

            # 1. Conciseness: avoid verbose reflections
            word_count = len(reflection.split())
            if word_count < 100:
                score += 0.3
            elif word_count < 200:
                score += 0.15
            # Very verbose: 0 points

            # 2. Specificity: reference visual details
            visual_keywords = ["image", "see", "visual", "detail", "color", "shape"]
            if any(kw in reflection.lower() for kw in visual_keywords):
                score += 0.3

            # 3. Error identification: acknowledge mistakes
            error_keywords = ["error", "mistake", "incorrect", "wrong", "missed"]
            if any(kw in reflection.lower() for kw in error_keywords):
                score += 0.2

            # 4. Correction: propose fix
            if "should be" in reflection.lower() or "correct answer" in reflection.lower():
                score += 0.2

            return min(1.0, score)

        return score_reflection

    def compute_reward(self, example: ReflectionExample,
                      generated_answer: str, generated_reflection: str) -> dict:
        """
        Compute comprehensive reward for reflection-augmented answer.
        Returns separate components and combined score.
        """
        # Answer correctness reward
        answer_reward = self.answer_verifier(
            generated_answer,
            example.correct_answer
        )

        # Reflection quality reward
        reflection_reward = self.reflection_scorer(
            generated_reflection,
            example.problem_text,
            ""  # Image analysis placeholder
        )

        # Combined: answer is primary, reflection is secondary
        combined_reward = 0.7 * answer_reward + 0.3 * reflection_reward

        return {
            "answer_reward": answer_reward,
            "reflection_reward": reflection_reward,
            "combined": combined_reward,
            "details": {
                "is_correct": answer_reward > 0.5,
                "reflection_quality": reflection_reward
            }
        }

# Test reward model
reward_model = ReflectionRewardModel()

test_example = reflection_examples[0]
gen_answer = "20"
gen_reflection = "Looking at the problem, 25% of 80 means I need to divide 80 by 4. This gives 20."

rewards = reward_model.compute_reward(test_example, gen_answer, gen_reflection)
print(f"Answer reward: {rewards['answer_reward']:.2f}")
print(f"Reflection reward: {rewards['reflection_reward']:.2f}")
print(f"Combined reward: {rewards['combined']:.2f}")
```

Implement GRPO training loop for reflection-augmented answers:

```python
class ReflectionGRPOTrainer:
    """Group Relative Policy Optimization for reflection-aware reasoning."""

    def __init__(self, model, reward_model: ReflectionRewardModel,
                 learning_rate: float = 1e-5):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def generate_with_reflection(self, problem_text: str, image: torch.Tensor,
                                temperature: float = 0.7):
        """
        Generate both answer and reflection from multimodal model.
        """
        prompt = f"Problem: {problem_text}\nThink step-by-step and reflect."

        # In practice: use actual MLLM generation
        answer = "placeholder_answer"
        reflection = "placeholder_reflection"

        return answer, reflection

    def compute_grpo_loss(self, examples: List[ReflectionExample],
                         group_size: int = 4) -> torch.Tensor:
        """
        Compute GRPO loss across groups of examples.
        Within each group, upweight high-reward samples.
        """
        losses = []

        for i in range(0, len(examples), group_size):
            batch = examples[i:i+group_size]

            # Generate and score all samples in group
            group_rewards = []
            group_log_probs = []

            for example in batch:
                # Generate
                answer, reflection = self.generate_with_reflection(
                    example.problem_text,
                    example.image
                )

                # Compute reward
                reward_dict = self.reward_model.compute_reward(
                    example,
                    answer,
                    reflection
                )
                group_rewards.append(reward_dict["combined"])

                # Compute log probability (simplified)
                log_prob = 0.0  # Placeholder: would compute actual log prob
                group_log_probs.append(log_prob)

            # Normalize rewards within group (relative comparison)
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float)
            normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / \
                                (rewards_tensor.std() + 1e-8)

            # GRPO loss: maximize log prob of high-reward samples
            for log_prob, norm_reward in zip(group_log_probs, normalized_rewards):
                loss = -log_prob * norm_reward
                losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def train_step(self, examples: List[ReflectionExample]) -> dict:
        """Single RL training step."""
        loss = self.compute_grpo_loss(examples, group_size=4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.item()}

# Training loop (placeholder structure)
# trainer = ReflectionGRPOTrainer(mllm_model, reward_model)
# for epoch in range(5):
#     for batch in dataloader:
#         stats = trainer.train_step(batch)
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **Reflection Dataset Size** | 5k-20k examples sufficient; use advanced model to bootstrap, then filter |
| **Answer/Reflection Weight** | Start 70/30; visual tasks may need 60/40, pure reasoning 80/20 |
| **Group Size** | 4-8 samples; larger groups provide better relative comparisons |
| **Reflection Length Target** | 50-150 tokens optimal; reward brevity to avoid verbosity |
| **Training Epochs** | 3-5 epochs on reflection data; monitor for overfitting |

**When to Use:**
- Multimodal reasoning tasks requiring visual understanding and self-correction
- Need explainability: reflections show model's reasoning process
- Combining visual and textual information where tradeoffs exist
- Training on benchmarks like MathVista, MMMU where visual math matters
- Want to improve reasoning without scaling model size

**When NOT to Use:**
- Tasks where reflection adds latency costs that outweigh benefits
- Models already achieving ceiling performance (reflection won't help)
- Domains without clear visual-semantic alignment
- Applications requiring fast inference where reflection generation is prohibitive
- Few-shot learning where reflection dataset can't be generated

**Common Pitfalls:**
- Reflection dataset quality low: biased initial model generates bad reflections; validate manually
- Reward model misalignment: if reflection scorer doesn't match human judgment, RL optimizes wrong objective
- Verbosity explosion: models learn to generate long reflections if not penalized; enforce length limits
- Reflection-answer mismatch: models generate good reflections but poor answers; weight answer reward heavily
- Overfitting to reflection dataset: use diverse generation strategies during data collection

## Reference

SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning
https://arxiv.org/abs/2506.01713
