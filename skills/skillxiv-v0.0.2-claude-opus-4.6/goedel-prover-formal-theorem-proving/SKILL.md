---
name: goedel-prover-formal-theorem-proving
title: Goedel-Prover-V2 - Scaling Formal Theorem Proving with Expert Iteration
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03613
keywords: [formal-verification, theorem-proving, lean, reinforcement-learning]
description: "Train language models for formal theorem proving via expert iteration with verifier-guided self-correction and checkpoint merging."
---

## Goedel-Prover-V2: Expert Iteration for Formal Theorem Proving

Goedel-Prover-V2 trains LLMs to prove mathematical theorems in the Lean proof assistant through expert iteration: iteratively sampling proofs, checking them against the Lean compiler, and using corrections as training data. The breakthrough is recognizing that verifier feedback (compiler errors) enables models to self-correct, eliminating need for human demonstrations beyond base training.

### Core Concept

Formal theorem proving requires exact syntax and logical correctness—tasks where LLMs typically fail without examples. Rather than requiring humans to write proofs, Goedel-Prover leverages the Lean compiler as an automatic teacher: generate a proof, get compiler feedback (error messages), revise, and learn from successful iterations. This "verifier-guided self-correction" is both more scalable and more aligned with actual proving workflows.

### Architecture Overview

- **Expert Iteration Pipeline**: Sample → Verify → Correct → Train loop with Lean compiler as oracle
- **Scaffolded Data Synthesis**: Generate synthetic theorems of increasing difficulty to create curriculum
- **Verifier-Guided Refinement**: Use Lean compiler error messages to iteratively fix proofs
- **Checkpoint Merging**: Average model checkpoints during training to preserve output diversity and prevent mode collapse in later RL stages
- **Multi-Scale Models**: Train 8B and 32B variants; smaller well-trained models outperform larger untrained ones

### Implementation Steps

**Step 1: Set Up Lean Environment and Proof Checker**

```python
import subprocess
import json
from typing import Tuple, Optional

class LeanProofChecker:
    """Interface to Lean compiler for proof verification."""

    def __init__(self, lean_path: str = "lean"):
        self.lean_path = lean_path

    def check_proof(self, theorem_statement: str, proof_code: str) -> Tuple[bool, str]:
        """
        Check if proof is valid in Lean.

        Returns: (is_valid, feedback)
        """
        full_code = f"""
theorem problem : {theorem_statement} := by
{proof_code}
"""
        try:
            result = subprocess.run(
                [self.lean_path, "--stdin"],
                input=full_code,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check if compilation succeeded
            if result.returncode == 0:
                return True, "Proof verified"
            else:
                # Extract error messages
                error_msg = result.stderr + result.stdout
                return False, error_msg

        except subprocess.TimeoutExpired:
            return False, "Proof checking timeout"
        except Exception as e:
            return False, str(e)

# Example
checker = LeanProofChecker()
is_valid, feedback = checker.check_proof(
    "∀ n : ℕ, n + 0 = n",
    "intro n\nsimp"
)
```

**Step 2: Implement Scaffolded Data Synthesis**

```python
def generate_synthetic_theorems(base_theorems: list, complexity_levels: int = 5) -> dict:
    """
    Generate theorems of increasing difficulty from base set.
    Curriculum learning: start with simple, progress to complex.
    """
    synthetic_data = {}

    for level in range(complexity_levels):
        theorems_at_level = []

        for base_theorem in base_theorems:
            # Level 0: Original theorems
            if level == 0:
                theorems_at_level.append(base_theorem)
            else:
                # Levels 1+: Generalize and extend theorems
                generalized = generalize_theorem(base_theorem, level)
                theorems_at_level.append(generalized)

        synthetic_data[f'level_{level}'] = theorems_at_level

    return synthetic_data

def generalize_theorem(theorem: str, complexity_level: int) -> str:
    """Make theorems more complex by adding parameters, conditions."""
    # Example: transform ∀ n, n+0=n into ∀ a b, a*(n+0) = a*n
    if complexity_level == 1:
        return theorem.replace("n", "(a + b * n)")
    elif complexity_level == 2:
        return theorem.replace("n", "(f (g n))")  # Compose functions
    else:
        return f"({theorem}) ∧ (proof of harder variant)"
```

**Step 3: Expert Iteration Loop**

```python
import random

class ExpertIteration:
    """Expert iteration: sample → check → correct → train."""

    def __init__(self, model, checker: LeanProofChecker, learning_rate=1e-5):
        self.model = model
        self.checker = checker
        self.training_data = []
        self.lr = learning_rate

    def sample_proofs(self, theorem: str, num_samples: int = 5) -> list:
        """Generate multiple proof attempts."""
        proofs = []

        for _ in range(num_samples):
            # Sample from model (use temperature > 0 for diversity)
            proof_text = self.model.generate(
                f"Prove: {theorem}\nProof:",
                temperature=0.8,
                max_tokens=500
            )
            proofs.append(proof_text)

        return proofs

    def verify_and_filter(self, theorem: str, proofs: list) -> Tuple[list, list]:
        """Check proofs, separate correct from incorrect."""
        correct_proofs = []
        incorrect_proofs = []

        for proof in proofs:
            is_valid, feedback = self.checker.check_proof(theorem, proof)

            if is_valid:
                correct_proofs.append((proof, feedback))
            else:
                incorrect_proofs.append((proof, feedback))

        return correct_proofs, incorrect_proofs

    def self_correct(self, theorem: str, incorrect_proofs: list) -> list:
        """
        Use compiler error messages to guide self-correction.
        Generate corrected proofs informed by error feedback.
        """
        corrected = []

        for wrong_proof, error_msg in incorrect_proofs:
            # Prompt model to fix proof given error
            correction_prompt = f"""
Original theorem: {theorem}
Failed proof:
{wrong_proof}

Lean compiler error:
{error_msg[:500]}  # Limit error length

Correct the proof:"""

            fixed_proof = self.model.generate(correction_prompt, max_tokens=500)
            is_valid, feedback = self.checker.check_proof(theorem, fixed_proof)

            if is_valid:
                corrected.append(fixed_proof)

        return corrected

    def train_on_successful_proofs(self, proofs: list, theorems: list):
        """
        Supervised fine-tuning on verified proofs.
        """
        for proof, theorem in zip(proofs, theorems):
            # Format as training example
            prompt = f"Prove: {theorem}\nProof:"
            # SFT: minimize cross-entropy of proof tokens
            loss = self.model.compute_loss(prompt, proof)
            loss.backward()

        self.model.optimizer.step()

    def run_iteration(self, theorems: list, num_samples: int = 5, max_iters: int = 3):
        """
        Run one expert iteration: sample → verify → correct → train.
        """
        all_correct = []

        for theorem in theorems:
            # Sample proofs
            proofs = self.sample_proofs(theorem, num_samples)

            # Verify
            correct, incorrect = self.verify_and_filter(theorem, proofs)
            all_correct.extend([p for p, _ in correct])

            # Self-correct (iterative refinement)
            for _ in range(max_iters):
                if not incorrect:
                    break

                corrected = self.self_correct(theorem, incorrect)
                all_correct.extend(corrected)

                # Re-verify remaining
                correct, incorrect = self.verify_and_filter(
                    theorem,
                    [p for p, _ in incorrect]
                )

        # Train on all successful proofs
        if all_correct:
            self.train_on_successful_proofs(all_correct, theorems)

        return len(all_correct)
```

**Step 4: Implement Checkpoint Merging**

```python
def merge_checkpoints(checkpoints: list, weights: list = None) -> dict:
    """
    Average model checkpoints to prevent mode collapse.
    RL training can reduce output diversity; averaging preserves it.
    """
    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)

    merged_state = {}

    # Average weights across checkpoints
    for param_name in checkpoints[0].keys():
        merged_state[param_name] = sum(
            w * ckpt[param_name] for w, ckpt in zip(weights, checkpoints)
        )

    return merged_state

class CheckpointManager:
    """Manage checkpoints and periodic averaging."""

    def __init__(self, model, averaging_frequency: int = 100):
        self.model = model
        self.checkpoints = []
        self.averaging_frequency = averaging_frequency
        self.step = 0

    def save_checkpoint(self):
        """Save current model state."""
        self.checkpoints.append(self.model.state_dict().copy())

    def maybe_merge(self):
        """Periodically merge recent checkpoints."""
        self.step += 1

        if self.step % self.averaging_frequency == 0 and len(self.checkpoints) > 1:
            # Keep last 5 checkpoints, merge them
            recent = self.checkpoints[-5:]
            merged = merge_checkpoints(recent)
            self.model.load_state_dict(merged)

            # Clear old checkpoints
            self.checkpoints = [merged]
```

**Step 5: Full Training Pipeline**

```python
def train_goedel_prover(
    model,
    theorem_dataset: list,
    num_epochs: int = 3,
    samples_per_theorem: int = 5
):
    """
    Complete training pipeline: expert iteration with curriculum.
    """
    checker = LeanProofChecker()
    expert_iter = ExpertIteration(model, checker)
    checkpoint_mgr = CheckpointManager(model)

    # Sort by difficulty (curriculum)
    difficulties = compute_theorem_difficulty(theorem_dataset)
    sorted_theorems = sorted(zip(theorem_dataset, difficulties), key=lambda x: x[1])

    for epoch in range(num_epochs):
        total_proofs = 0

        for theorem, difficulty in sorted_theorems:
            # Adjust samples based on difficulty
            num_samples = max(3, samples_per_theorem - difficulty // 10)

            num_correct = expert_iter.run_iteration(
                [theorem],
                num_samples=num_samples,
                max_iters=3
            )

            total_proofs += num_correct

            # Periodic checkpoint averaging
            checkpoint_mgr.save_checkpoint()
            checkpoint_mgr.maybe_merge()

        print(f"Epoch {epoch}: {total_proofs} verified proofs")

def compute_theorem_difficulty(theorems: list) -> list:
    """Heuristic: theorem length correlates with difficulty."""
    return [len(t.split()) for t in theorems]
```

### Practical Guidance

**When to Use:**
- Formal verification tasks with checkable proofs (Lean, Coq, Isabelle)
- Scenarios with large theorem libraries for curriculum learning
- Applications where proof correctness is mandatory
- Cases where iterative refinement is preferred to human demonstrations

**When NOT to Use:**
- Informal mathematical reasoning without formal verification
- Real-time inference (proof generation is slow)
- Domains without reliable proof checkers
- Scenarios with <100 training theorems (insufficient curriculum)

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `samples_per_theorem` | 5 | More samples = better coverage, higher compute cost |
| `max_correction_iters` | 3 | Iterations of self-correction; diminishing returns after 3 |
| `checkpoint_averaging_freq` | 100 | More frequent averaging = higher diversity, slower training |
| `curriculum_start_difficulty` | 0 | Begin with easiest theorems; increase gradually |

### Reference

**Paper**: Goedel-Prover-V2: Scaling Formal Theorem Proving (2508.03613)
- Expert iteration with verifier feedback
- 32B model achieves SOTA on formal theorem benchmarks
- Self-correction via compiler error messages
- Checkpoint merging prevents mode collapse
