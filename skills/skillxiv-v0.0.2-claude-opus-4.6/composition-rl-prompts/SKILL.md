---
name: composition-rl-prompts
title: "Composition-RL: Compose Your Verifiable Prompts for RL of LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.12036"
keywords: [Reinforcement Learning, Prompt Composition, Curriculum Learning, Problem Difficulty, Verifiable Reasoning]
description: "Convert easy, high-accuracy training prompts into harder compositional problems by sequentially chaining multiple prompts together. Use Composition-RL to maintain effective learning signals during RL training when many prompts achieve near-perfect accuracy, enabling curriculum learning through progressive compositional depths."
---

# Composition-RL: Compose Your Verifiable Prompts for RL of LLMs

## Problem Context

During reinforcement learning of language models on mathematical and reasoning tasks, many prompts become too easy—models achieve 100% accuracy, making these prompts uninformative for further learning. This creates a plateau where gradient signals vanish. Standard curriculum learning solutions are expensive, requiring new data collection or external scoring. Composition-RL addresses this by algorithmically combining existing easy prompts into harder compositional ones without new annotation.

## Core Concept

Composition-RL automatically merges K existing prompts into a single compositional prompt that requires solving all K sub-problems sequentially. A two-prompt composition works as follows:

1. Extract the answer from prompt 1 and create a symbolic definition
2. Modify prompt 2 to use a variable instead of a concrete value
3. Link the two by replacing the variable with the symbolic value from prompt 1

The key insight is that solving the composed problem requires solving each sub-problem in sequence, creating a multiplicative difficulty increase without manual curation.

## Architecture Overview

- **Prompt parsing**: Extract structure and numerical values from existing prompts
- **Sequential linking**: Chain solutions from earlier prompts into later prompts
- **Difficulty scaling**: Compose progressively (depth 2, then depth 3)
- **Validation**: Verify composed prompts are solvable and distinct from originals
- **Curriculum stages**: Train on original → depth-2 compositions → depth-3 compositions

## Implementation

### Step 1: Parse problem structure and extract answers

Identify prompt templates and answer patterns.

```python
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParsedProblem:
    """Extracted problem structure."""
    original_text: str
    template_pattern: str
    variables: Dict[str, str]
    answer_value: Optional[str]
    answer_type: str  # 'numeric', 'symbolic', 'boolean'

class ProblemStructureParser:
    """Parse mathematical problem structure for composition."""

    def __init__(self):
        self.numeric_pattern = r'[-+]?\d+(?:\.\d+)?'
        self.variable_pattern = r'[a-zA-Z_]\w*'

    def parse_problem(self, problem_text: str) -> ParsedProblem:
        """
        Extract structure from a problem.
        Assumes verifiable problems with clear answers.
        """
        # Simple extraction: assume last sentence/line is answer
        lines = problem_text.strip().split('\n')

        # Find answer (last line or line with specific markers)
        answer_line = None
        answer_value = None
        for line in reversed(lines):
            if 'Answer:' in line or 'answer:' in line or '=' in line:
                answer_line = line
                answer_match = re.search(r'=\s*(.+?)(?:\s*[.,]|$)', line)
                if answer_match:
                    answer_value = answer_match.group(1).strip()
                break

        if answer_value is None:
            answer_value = lines[-1].strip()

        # Determine answer type
        if re.match(r'[-+]?\d+(?:\.\d+)?$', answer_value):
            answer_type = 'numeric'
        else:
            answer_type = 'symbolic'

        return ParsedProblem(
            original_text=problem_text,
            template_pattern='\n'.join(lines[:-1]),  # Template without answer
            variables={},
            answer_value=answer_value,
            answer_type=answer_type
        )

    def extract_numerical_values(self, problem_text: str) -> List[Tuple[str, str]]:
        """Extract all numerical values and their context."""
        matches = []
        for match in re.finditer(self.numeric_pattern, problem_text):
            value = match.group(0)
            context_start = max(0, match.start() - 20)
            context_end = min(len(problem_text), match.end() + 20)
            context = problem_text[context_start:context_end]
            matches.append((value, context))
        return matches
```

### Step 2: Compose two prompts sequentially

Link answer from first prompt into second prompt.

```python
class PromptComposer:
    """Compose multiple prompts into single difficult problem."""

    def __init__(self, parser: ProblemStructureParser):
        self.parser = parser

    def compose_two_prompts(
        self,
        prompt_1: str,
        prompt_2: str,
        variable_name: str = "X"
    ) -> Tuple[str, str]:
        """
        Compose two prompts into one.

        Strategy:
        1. Parse prompt_1, extract answer as value V
        2. In prompt_2, replace a numerical constant with variable
        3. Create composed prompt that solves p1, then uses answer in p2

        Args:
            prompt_1: First problem
            prompt_2: Second problem to modify
            variable_name: Name for the variable linking problems

        Returns:
            (composed_prompt, expected_answer)
        """
        # Parse both problems
        parsed_1 = self.parser.parse_problem(prompt_1)
        parsed_2 = self.parser.parse_problem(prompt_2)

        answer_1 = parsed_1.answer_value

        # Find a numerical value in prompt_2 to replace
        numerical_values = self.parser.extract_numerical_values(prompt_2)

        if not numerical_values:
            # No numeric value to replace, use direct substitution
            # Modify template of prompt_2 to reference the variable
            modified_prompt_2 = prompt_2.replace(
                parsed_2.answer_value,
                ""  # Will be determined by solving prompt_1
            )
        else:
            # Replace first numerical value with variable
            value_to_replace, _ = numerical_values[0]

            # Create modified prompt_2
            modified_prompt_2 = prompt_2.replace(
                value_to_replace,
                f"{variable_name} (obtained from the first problem)",
                count=1
            )

        # Compose into single problem
        composed_prompt = f"""Solve the following two-part problem:

Part 1: {prompt_1.strip()}

Part 2: {modified_prompt_2.strip()}

Let {variable_name} be the answer to Part 1. Use this value in Part 2 to find the final answer.

What is the final answer?"""

        # Composed answer requires solving both
        return composed_prompt, parsed_2.answer_value  # Final answer from part 2

    def compose_k_prompts(
        self,
        prompts: List[str],
        max_depth: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Compose K prompts with progressive chaining.

        For 3 prompts: (p1, p2, p3) -> p1's answer feeds p2, p2's answer feeds p3
        """
        composed = []

        # Generate compositions of depth 2, 3, ..., min(max_depth, len(prompts))
        for depth in range(2, min(max_depth + 1, len(prompts) + 1)):
            subset = prompts[:depth]

            # Progressive chaining
            current_prompt = subset[0]
            answers_chain = [self.parser.parse_problem(subset[0]).answer_value]

            for i in range(1, len(subset)):
                next_prompt = subset[i]
                current_prompt, final_answer = self.compose_two_prompts(
                    current_prompt, next_prompt,
                    variable_name=f"X{i}"
                )
                answers_chain.append(final_answer)

            composed.append((current_prompt, answers_chain[-1]))

        return composed
```

### Step 3: Validate composed prompts

Ensure composed problems are solvable and distinct.

```python
class CompositionValidator:
    """Validate composed prompts for training."""

    def __init__(self, verifier_model):
        """
        Args:
            verifier_model: Model that verifies if answer is correct.
                          Returns True if correct, False otherwise.
        """
        self.verifier_model = verifier_model

    def validate_composition(
        self,
        composed_prompt: str,
        expected_answer: str,
        original_prompts: List[str]
    ) -> Dict[str, bool]:
        """
        Validate that composed prompt is solvable.

        Returns:
            Dict with keys: 'is_valid', 'is_distinct', 'solvable'
        """
        # Check 1: Is it structurally distinct from originals?
        is_distinct = all(
            composed_prompt.strip() != orig.strip()
            for orig in original_prompts
        )

        # Check 2: Does the composed prompt appear solvable?
        # (Quick check: answer_format is consistent)
        answer_format_valid = isinstance(expected_answer, str)

        # Check 3: Verify with model (sample generation)
        solvable = self._test_solvability(composed_prompt, expected_answer)

        return {
            'is_valid': is_distinct and answer_format_valid,
            'is_distinct': is_distinct,
            'solvable': solvable
        }

    def _test_solvability(
        self,
        prompt: str,
        expected_answer: str,
        num_samples: int = 3
    ) -> bool:
        """
        Quick test: can model generate the expected answer?
        """
        for _ in range(num_samples):
            response = self.verifier_model.generate(
                prompt, max_tokens=500, temperature=0.7
            )

            # Check if response contains expected answer
            if self._answer_matches(response, expected_answer):
                return True

        return False

    @staticmethod
    def _answer_matches(response: str, expected: str) -> bool:
        """Check if response contains expected answer."""
        response_clean = re.sub(r'[^a-zA-Z0-9.-]', '', response.lower())
        expected_clean = re.sub(r'[^a-zA-Z0-9.-]', '', expected.lower())
        return expected_clean in response_clean
```

### Step 4: Build curriculum with progressive composition

Progressively introduce compositional difficulties.

```python
class CompositionCurriculum:
    """Manage curriculum learning through progressive composition."""

    def __init__(
        self,
        original_prompts: List[str],
        composer: PromptComposer,
        validator: CompositionValidator,
        max_composition_depth: int = 3
    ):
        self.original_prompts = original_prompts
        self.composer = composer
        self.validator = validator
        self.max_composition_depth = max_composition_depth

        self.curriculum_stages = []
        self._build_curriculum()

    def _build_curriculum(self):
        """Generate curriculum stages."""

        # Stage 1: Original prompts
        stage_1 = [
            (prompt, self.composer.parser.parse_problem(prompt).answer_value)
            for prompt in self.original_prompts
        ]
        self.curriculum_stages.append(('depth_1', stage_1))

        # Stages 2+: Progressive compositions
        for depth in range(2, self.max_composition_depth + 1):
            composed_prompts = []

            # Sample subsets and compose
            for i in range(0, len(self.original_prompts), depth):
                subset = self.original_prompts[i:i + depth]
                if len(subset) < depth:
                    continue

                compositions = self.composer.compose_k_prompts(subset, max_depth=depth)

                for comp_prompt, answer in compositions:
                    # Validate
                    validation = self.validator.validate_composition(
                        comp_prompt, answer, subset
                    )

                    if validation['is_valid']:
                        composed_prompts.append((comp_prompt, answer))

            if composed_prompts:
                self.curriculum_stages.append((f'depth_{depth}', composed_prompts))

    def get_stage(self, stage_index: int) -> Tuple[str, List[Tuple[str, str]]]:
        """Get curriculum stage."""
        return self.curriculum_stages[min(stage_index, len(self.curriculum_stages) - 1)]

    def get_all_stages(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Return all curriculum stages."""
        return self.curriculum_stages
```

### Step 5: Training with curriculum

Train using progressively harder compositions.

```python
def train_with_composition_curriculum(
    model,
    original_prompts: List[str],
    verifier,
    optimizer,
    num_epochs: int = 3,
    curriculum_stages: int = 3,
    group_size: int = 8,
    device: str = 'cuda'
):
    """
    Train LLM with composition-based curriculum.

    Stages: original → depth-2 compositions → depth-3 compositions
    """
    parser = ProblemStructureParser()
    composer = PromptComposer(parser)
    validator = CompositionValidator(model)

    curriculum = CompositionCurriculum(
        original_prompts, composer, validator,
        max_composition_depth=curriculum_stages
    )

    for epoch in range(num_epochs):
        # Get current curriculum stage
        stage_idx = min(epoch // max(1, num_epochs // curriculum_stages), curriculum_stages - 1)
        stage_name, stage_prompts = curriculum.get_stage(stage_idx)

        print(f"Epoch {epoch + 1}: Curriculum stage {stage_name} "
              f"({len(stage_prompts)} problems)")

        total_loss = 0.0
        num_batches = 0

        # Training on current stage
        for batch_idx in range(0, len(stage_prompts), group_size):
            batch_prompts = stage_prompts[batch_idx:batch_idx + group_size]
            batch_size = len(batch_prompts)

            # Generate responses
            responses = []
            log_probs_list = []

            for prompt, expected_answer in batch_prompts:
                response, log_prob = model.generate_with_logprobs(
                    prompt, max_tokens=500
                )
                responses.append(response)
                log_probs_list.append(log_prob)

            log_probs = torch.stack(log_probs_list).to(device)

            # Verify answers
            rewards = torch.tensor([
                float(verifier(r)) for r in responses
            ], dtype=torch.float32, device=device)

            # GRPO loss
            log_prob_ratio = log_probs - log_probs.detach()
            ratio = torch.exp(log_prob_ratio)

            # Group relative baseline
            group_mean_reward = rewards.mean()
            relative_rewards = rewards - group_mean_reward

            clipped_ratio = torch.clamp(ratio, 0.5, 2.0)
            loss = -torch.min(
                log_prob_ratio * relative_rewards.unsqueeze(-1),
                torch.log(clipped_ratio) * relative_rewards.unsqueeze(-1)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_reward = rewards.mean().item()
        print(f"  Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    return model
```

## Practical Guidance

**When to use**: RL training on mathematical, reasoning, or verification tasks where many prompts reach near-perfect accuracy during training

**Hyperparameters**:
- **max_composition_depth**: 2-4 (deeper compositions harder but may become unsolvable)
- **curriculum_stages**: 2-3 (original → depth-2 → depth-3)
- **group_size**: 4-8 (GRPO grouping)
- **epoch_per_stage**: Spend 1-2 epochs per curriculum stage

**Key advantages**:
- Creates curriculum without new data collection
- Maintains learning signals when originals become easy
- Progressive difficulty scaling
- Works with existing verifiable prompt sets

**Common pitfalls**:
- Compositions becoming too hard (no correct solutions)
- Answer extraction failing on varied formats
- Not validating composed prompts before training
- Curriculum transitions too abrupt

**Scaling**: Composition is O(K²) for K prompts, but typically run offline. Validation can be parallelized.

## Reference

Paper: https://arxiv.org/abs/2602.12036
Related work: Curriculum learning, data augmentation, compositional generalization
Benchmarks: MATH, AIME, arithmetic reasoning
Code concepts: Sequential linking, problem parsing, curriculum management
