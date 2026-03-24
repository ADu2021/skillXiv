---
name: critique-grpo-reasoning
title: "Critique-GRPO: Advancing LLM Reasoning with Dual Feedback"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.03106"
keywords: [reinforcement-learning, reasoning, feedback, critique, policy-optimization]
description: "Improve LLM reasoning by combining numerical and natural language critique feedback in online RL for policy refinement."
---

# Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback

## Core Concept

Standard reinforcement learning for LLM reasoning using only numerical rewards plateaus in performance and fails to enable effective self-reflection. Critique-GRPO addresses these limitations by integrating both natural language critiques and scalar rewards for policy optimization. This dual-feedback approach enables models to learn from initial failures, refine responses via critique-guided self-improvement, and achieve consistent 15-21% Pass@1 improvements.

## Architecture Overview

- **Three-Step Framework**: Initial response sampling, critique-guided self-refinement, online policy optimization
- **Natural Language Feedback**: Text-based critiques identifying reasoning failures enable verbal credit assignment
- **Numerical Rewards**: Scalar rewards combined with critiques for principled policy gradient updates
- **Critique-Conditioned Refinement**: In-context learning on question-response-critique triplets enables self-improvement
- **Dual Objective**: Train on both initial and refined responses with reward shaping that amplifies successful refinements
- **Advantage**: Maintains higher policy entropy and enables better exploration compared to numerical-only methods

## Implementation

### Step 1: Design Critique Generation

```python
import torch
from typing import Dict, List, Tuple

class CritiqueGenerator:
    """Generate natural language critiques of reasoning responses"""

    def __init__(self, critique_model_name='gpt-4'):
        self.critique_model = load_model(critique_model_name)

    def generate_critique(self, question: str,
                         response: str,
                         ground_truth: str = None) -> Tuple[str, float]:
        """
        Generate detailed critique of response.
        Also provide numerical confidence score.
        """

        critique_prompt = f"""Evaluate this reasoning response:

Question: {question}

Response: {response}

{f"Ground Truth: {ground_truth}" if ground_truth else ""}

Provide a detailed critique covering:
1. Correctness: Is the final answer correct?
2. Reasoning: Are the logical steps sound?
3. Clarity: Is the explanation clear?
4. Completeness: Are there missing steps?
5. Efficiency: Is there a simpler approach?

Also provide a confidence score (0.0-1.0) indicating how likely this response is correct."""

        critique_response = self.critique_model.generate(critique_prompt)

        # Parse response
        critique_text = critique_response
        confidence = self.extract_confidence_score(critique_response)

        return critique_text, confidence

    def extract_confidence_score(self, critique_text: str) -> float:
        """Extract numerical confidence from critique"""

        # Look for patterns like "confidence: 0.85" or "confidence score: 85%"
        import re

        # Try percentage format
        match = re.search(r'confidence.*?(\d+)%', critique_text.lower())
        if match:
            return float(match.group(1)) / 100.0

        # Try decimal format
        match = re.search(r'confidence.*?([0-9.]+)', critique_text.lower())
        if match:
            return float(match.group(1))

        # Default: neutral
        return 0.5

    def classify_critique_type(self, critique: str) -> str:
        """Categorize the type of error identified"""

        error_types = {
            'logical_error': ['logical', 'contradiction', 'inconsistent'],
            'calculation_error': ['calculation', 'arithmetic', 'math'],
            'missing_step': ['missing', 'incomplete', 'skipped'],
            'clarity_issue': ['unclear', 'confusing', 'hard to follow'],
            'wrong_approach': ['wrong approach', 'incorrect method'],
        }

        critique_lower = critique.lower()

        for error_type, keywords in error_types.items():
            if any(kw in critique_lower for kw in keywords):
                return error_type

        return 'other'
```

### Step 2: Implement Critique-Guided Self-Refinement

```python
class CritiqueGuidedRefinement:
    """Enable models to improve responses based on critiques"""

    def __init__(self, model):
        self.model = model
        self.refinement_history = []

    def refine_response(self, question: str,
                       initial_response: str,
                       critique: str,
                       max_refinement_attempts: int = 3) -> List[str]:
        """
        Use critique to guide response refinement.
        Create in-context learning examples showing how to improve.
        """

        refinement_attempts = [initial_response]

        for attempt in range(max_refinement_attempts):
            # Build refinement prompt with few-shot examples
            refinement_prompt = self.build_refinement_prompt(
                question, initial_response, critique, attempt
            )

            # Generate refined response
            refined = self.model.generate(refinement_prompt)

            refinement_attempts.append(refined)

            # Stop if refinement converges
            if self.has_converged(refinement_attempts[-2:]):
                break

        return refinement_attempts

    def build_refinement_prompt(self, question: str,
                               current_response: str,
                               critique: str,
                               attempt: int) -> str:
        """Construct prompt for guided refinement"""

        few_shot_examples = self.get_refinement_examples()

        prompt = f"""You previously answered:
Question: {question}
Your Response: {current_response}

Here's feedback on your response:
{critique}

Based on this feedback, provide an improved response.

Examples of how to address similar feedback:
{few_shot_examples}

Now, please provide your improved response to: {question}"""

        return prompt

    def get_refinement_examples(self) -> str:
        """Few-shot examples of refinement patterns"""

        examples = """Example 1:
Original: "The answer is 5"
Feedback: "Your calculation is wrong. Show the steps."
Improved: "Let me show my work. First, 2+2=4. Then, 4+1=5. So the answer is 5."

Example 2:
Original: "Use method X"
Feedback: "Method X is overcomplicated. Consider a simpler approach."
Improved: "Actually, we can use the simpler method Y: [steps]. This gives us the answer directly."

Example 3:
Original: "I'm not sure"
Feedback: "You need to be more decisive. Pick the best option."
Improved: "After thinking through the options, the best answer is [X] because [reasoning]."""

        return examples

    def has_converged(self, recent_attempts: List[str],
                     similarity_threshold: float = 0.95) -> bool:
        """Check if refinement has stabilized"""

        if len(recent_attempts) < 2:
            return False

        # Compute similarity between last two attempts
        similarity = self.compute_similarity(
            recent_attempts[-2],
            recent_attempts[-1]
        )

        return similarity > similarity_threshold

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity metric"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
```

### Step 3: Implement Critique-GRPO Training Loop

```python
class CritiqueGRPO:
    """GRPO training with dual numerical and natural language feedback"""

    def __init__(self, model, critique_generator: CritiqueGenerator,
                 refinement_module: CritiqueGuidedRefinement):
        self.model = model
        self.critique_gen = critique_generator
        self.refiner = refinement_module
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def training_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step combining initial and refined responses.

        Args:
            batch: {
                'questions': List[str],
                'ground_truths': List[str],
            }

        Returns:
            losses: {
                'initial_loss': float,
                'refinement_loss': float,
                'total_loss': float,
            }
        """

        questions = batch['questions']
        ground_truths = batch['ground_truths']

        total_loss = 0
        all_losses = {}

        for question, ground_truth in zip(questions, ground_truths):
            # Step 1: Initial response sampling
            initial_response = self.model.generate(question, temperature=1.0)

            # Step 2: Generate critique and reward
            critique, confidence = self.critique_gen.generate_critique(
                question, initial_response, ground_truth
            )

            # Compute numerical reward
            is_correct = self.check_correctness(initial_response, ground_truth)
            initial_reward = float(is_correct)

            # Step 3: Critique-guided refinement
            refined_responses = self.refiner.refine_response(
                question, initial_response, critique
            )
            refined_response = refined_responses[-1]  # Best refinement

            refined_is_correct = self.check_correctness(
                refined_response, ground_truth
            )
            refined_reward = float(refined_is_correct)

            # Compute losses
            initial_log_prob = self.model.log_probability(
                question, initial_response
            )

            refined_log_prob = self.model.log_probability(
                question, refined_response
            )

            # Advantage: how much did refinement help?
            advantage = refined_reward - initial_reward

            # Policy gradient loss with reward shaping
            pg_loss = 0
            if initial_reward < refined_reward:
                # Successful refinement: amplify
                pg_loss -= refined_log_prob * (refined_reward - 0.5) * 1.5
            else:
                # Failed refinement or no improvement: normal weight
                pg_loss -= initial_log_prob * (initial_reward - 0.5)

            total_loss += pg_loss
            all_losses[f'q_{len(all_losses)}'] = pg_loss.item()

        # Average loss
        average_loss = total_loss / len(questions)

        # Backward pass
        self.optimizer.zero_grad()
        average_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        all_losses['total'] = average_loss.item()
        all_losses['avg_initial_reward'] = np.mean([
            float(self.check_correctness(
                self.model.generate(q), gt
            ))
            for q, gt in zip(questions[:2], ground_truths[:2])
        ])

        return all_losses

    def check_correctness(self, response: str, ground_truth: str) -> bool:
        """Verify if response matches ground truth"""

        # Extract final answer from response
        predicted = self.extract_answer(response)
        expected = self.extract_answer(ground_truth)

        return predicted == expected

    def extract_answer(self, text: str) -> str:
        """Extract numerical or categorical answer"""

        import re

        # Look for patterns like "answer is X" or "= X"
        match = re.search(r'(?:answer|result|equals?).*?([0-9]+|yes|no|true|false)',
                         text, re.IGNORECASE)

        if match:
            return match.group(1).lower()

        return text.strip().lower()
```

### Step 4: Integrate with Training Pipeline

```python
class CritiqueGRPOTrainer:
    """Full training pipeline with dual feedback"""

    def __init__(self, model, num_epochs: int = 5):
        self.model = model
        self.num_epochs = num_epochs

        self.critique_gen = CritiqueGenerator()
        self.refiner = CritiqueGuidedRefinement(model)
        self.grpo = CritiqueGRPO(model, self.critique_gen, self.refiner)

    def train(self, train_questions: List[str],
             train_answers: List[str],
             val_questions: List[str],
             val_answers: List[str]) -> Dict:
        """Train with critiques and refinement"""

        history = {
            'train_losses': [],
            'val_pass_at_1': [],
            'val_pass_at_4': [],
        }

        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")

            # Training step
            epoch_losses = []

            for question, answer in zip(train_questions, train_answers):
                batch = {
                    'questions': [question],
                    'ground_truths': [answer],
                }

                losses = self.grpo.training_step(batch)
                epoch_losses.append(losses['total'])

            avg_train_loss = np.mean(epoch_losses)
            history['train_losses'].append(avg_train_loss)

            print(f"Train loss: {avg_train_loss:.4f}")

            # Validation
            pass_at_1 = self.evaluate_pass_at_k(
                val_questions, val_answers, k=1
            )
            pass_at_4 = self.evaluate_pass_at_k(
                val_questions, val_answers, k=4
            )

            history['val_pass_at_1'].append(pass_at_1)
            history['val_pass_at_4'].append(pass_at_4)

            print(f"Val Pass@1: {pass_at_1:.1%}")
            print(f"Val Pass@4: {pass_at_4:.1%}")

        return history

    def evaluate_pass_at_k(self, questions: List[str],
                          answers: List[str],
                          k: int = 1) -> float:
        """Compute Pass@K metric"""

        successes = 0

        for question, answer in zip(questions, answers):
            # Generate k samples
            samples = [
                self.model.generate(question, temperature=0.7)
                for _ in range(k)
            ]

            # Check if any matches ground truth
            for sample in samples:
                if self.grpo.check_correctness(sample, answer):
                    successes += 1
                    break

        return successes / len(questions) if questions else 0.0
```

## Practical Guidance

1. **Dual Feedback is Key**: Numerical rewards alone plateau. Adding critique enables 36.47% valid self-refinement rates compared to minimal gains from spontaneous reflection.

2. **Critique Quality Matters**: Use a capable critique model (GPT-4, Claude-3) to generate detailed, specific feedback. Poor critiques won't enable meaningful refinement.

3. **Few-Shot Refinement Examples**: Include concrete examples of how to address different error types. This dramatically improves refinement success rates.

4. **Reward Shaping**: Amplify successful refinements (1.5× weight) while penalizing failed ones. This maintains exploration while concentrating learning on improvements.

5. **Policy Entropy**: The dual-feedback approach maintains higher policy entropy than numerical-only training, enabling better exploration throughout training.

6. **Evaluation Metric**: Use Pass@1 and Pass@4, not just accuracy. Critique-GRPO improves both—initial generations improve, and refinements provide backup paths.

## Reference

- Paper: Critique-GRPO (2506.03106)
- Key Improvement: +15.0-21.6% Pass@1 on Qwen models
- Architecture: Dual feedback (critique + reward) for policy optimization
- Innovation: In-context learning with critique-guided refinement
