---
name: decoupled-reasoning-proving
title: "Towards Solving More Challenging IMO Problems via Decoupled Reasoning and Proving"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06804"
keywords: [Automated Theorem Proving, Mathematical Reasoning, Informal to Formal Verification, IMO Problems, Lemma Synthesis]
description: "Bridge the gap between informal mathematical reasoning (80% accuracy) and formal proof synthesis (8% success) by decoupling them: a general-purpose reasoner generates strategic lemmas, then a specialized prover verifies them formally. First open-source solver of post-2000 IMO problems."
---

# Decoupled Reasoning and Proving: Bridging Informal Intuition and Formal Verification

Automated theorem proving faces a stark disconnect: language models understand mathematical problems informally with 80% accuracy but struggle to formalize proofs (only 8% formally verified). This gap exists because reasoning and formal verification require different skills—one needs mathematical intuition about what's true, the other requires meticulous checking against axioms. Decoupled reasoning and proving solves this by separating roles: a general-purpose reasoner generates strategic intermediate lemmas (educated guesses about useful facts), and a specialized formal prover checks each lemma rigorously against a formal system. This division of labor enables models to solve IMO-level problems—for the first time openly, without requiring proprietary systems.

When solving competition mathematics at the highest level, pure formal approaches fail because the search space is vast; pure informal approaches lack rigor. Decoupling lets each component play to its strengths: the reasoner can be creative and exploratory, while the prover acts as a rigorous gatekeeper, ensuring every step is logically sound. This asymmetry mirrors human problem-solving: mathematicians sketch ideas loosely, then formalize carefully.

## Core Concept

The decoupled framework maintains two models in dialogue: the Reasoner generates human-readable strategic lemmas (e.g., "If angle ABC equals angle XYZ, then triangles ABC and XYZ are similar"), and the Prover attempts to formalize and verify each lemma in Lean or similar formal systems. The reasoner works at a high level of abstraction, proposing proof strategy without worrying about implementation details. When the prover rejects a lemma (unable to verify), the reasoner receives feedback and proposes alternatives. This iterative refinement leads to solutions because the reasoner can generate many candidate strategies (each lemma is a small commitment), and only sound strategies survive formal verification. The approach produces a dataset of verified lemmas useful for future problems—mathematical knowledge that accumulates.

## Architecture Overview

- **Informal Reasoner**: General-purpose LLM generating strategic lemmas and proof sketches
- **Formal Prover**: Specialized system (Lean, Coq, Isabelle) attempting to verify lemmas
- **Lemma Synthesis Engine**: Converts reasoner proposals into formal statements
- **Verification Feedback Loop**: Reports which lemmas failed and why
- **Lemma Repository**: Accumulates verified lemmas for pattern reuse
- **Problem Encoder**: Translates IMO problems into formal specifications

## Implementation

This example demonstrates the informal reasoning component that generates strategic lemmas for a given problem.

```python
# Informal reasoner generating strategic lemmas
import torch
from typing import List, Dict

class InformalReasoner:
    def __init__(self, model_name="meta-llama/Llama-2-70b"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_lemma_candidates(self, problem_statement: str, num_candidates: int = 10) -> List[Dict]:
        """Generate multiple strategic lemmas for solving the problem.
        Each lemma is an intermediate fact that helps toward the solution."""

        prompt = f"""Given this geometry/algebra problem:

{problem_statement}

Generate strategic lemmas that would help solve this problem. Each lemma should be:
1. A non-obvious but likely true statement
2. A step toward the final solution
3. In natural mathematical language (not formal notation)

Format each lemma as: LEMMA: [statement]

Generate {num_candidates} candidate lemmas:"""

        with torch.no_grad():
            outputs = self.model.generate(
                self.tokenizer(prompt, return_tensors='pt')['input_ids'],
                max_length=2048,
                temperature=0.8,
                num_return_sequences=num_candidates,
                do_sample=True
            )

        lemmas = []
        for output in outputs:
            text = self.tokenizer.decode(output)
            # Extract lemmas from text
            candidate_lemmas = self._parse_lemmas(text)
            lemmas.extend(candidate_lemmas)

        # Deduplicate and score by usefulness heuristic
        unique_lemmas = []
        seen = set()
        for lemma in lemmas:
            if lemma['statement'] not in seen:
                unique_lemmas.append(lemma)
                seen.add(lemma['statement'])

        return unique_lemmas[:num_candidates]

    def generate_proof_sketch(self, problem_statement: str, lemmas: List[str]) -> str:
        """Generate outline of how lemmas combine to solve the problem."""

        lemma_text = "\n".join([f"- {lemma}" for lemma in lemmas])

        prompt = f"""Problem: {problem_statement}

Strategic lemmas we've established:
{lemma_text}

Now outline a proof strategy that uses these lemmas to solve the problem.
Be precise about the logical flow."""

        with torch.no_grad():
            output = self.model.generate(
                self.tokenizer(prompt, return_tensors='pt')['input_ids'],
                max_length=1024,
                temperature=0.5
            )

        sketch = self.tokenizer.decode(output[0])
        return sketch

    def _parse_lemmas(self, text: str) -> List[Dict]:
        """Extract LEMMA: ... statements from generated text."""
        lemmas = []
        lines = text.split('\n')

        for line in lines:
            if 'LEMMA:' in line:
                statement = line.split('LEMMA:')[1].strip()
                lemmas.append({
                    'statement': statement,
                    'verified': False,
                    'confidence': 0.7
                })

        return lemmas
```

This example shows the formal verification component that attempts to prove lemmas in Lean.

```python
class FormalProver:
    def __init__(self, lean_server_path: str = None):
        """Interface to Lean formal proof assistant."""
        self.lean_path = lean_server_path
        self.theorem_bank = {}  # Cache of verified lemmas

    def formalize_lemma(self, informal_lemma: str, problem_context: str) -> str:
        """Convert informal lemma to Lean statement.
        This is a template-based approach; more sophisticated approaches use neural translation."""

        # Template mapping for common pattern (geometric/algebraic)
        formal_statement = self._lemma_to_lean(informal_lemma, problem_context)
        return formal_statement

    def verify_lemma(self, lemma_statement: str, proof_attempt: str = None) -> Dict:
        """Attempt to verify a lemma in Lean.
        Returns success/failure and any proof hints."""

        # Construct Lean code
        lean_code = f"""theorem lemma_under_test : {lemma_statement} := by
  {proof_attempt if proof_attempt else 'sorry'}
"""

        # Execute Lean checker
        result = self._run_lean_check(lean_code)

        return {
            'verified': result['success'],
            'statement': lemma_statement,
            'proof_code': proof_attempt,
            'error_msg': result.get('error', ''),
            'suggestions': result.get('suggestions', [])
        }

    def _lemma_to_lean(self, informal: str, context: str) -> str:
        """Convert informal lemma to Lean syntax using heuristics."""

        # Example conversions
        conversions = {
            'similar': 'Geometry.similar',
            'parallel': 'Geometry.parallel',
            'perpendicular': 'Geometry.perpendicular',
            'collinear': 'Geometry.collinear',
            'angle': 'Angle.measure',
            'equals': '='
        }

        formal = informal
        for informal_term, formal_term in conversions.items():
            formal = formal.replace(informal_term, formal_term)

        return formal

    def _run_lean_check(self, lean_code: str) -> Dict:
        """Execute Lean checker and return verification result."""
        # This would interface with actual Lean server
        # Simplified mock implementation
        return {
            'success': 'sorry' not in lean_code,  # Only verify if no 'sorry'
            'error': '' if 'sorry' not in lean_code else 'Incomplete proof'
        }
```

This example demonstrates the iterative refinement loop: proposing lemmas, verifying them, and incorporating feedback.

```python
class DecoupledProverSystem:
    def __init__(self, reasoner: InformalReasoner, prover: FormalProver):
        self.reasoner = reasoner
        self.prover = prover
        self.verified_lemmas = []
        self.failed_attempts = []

    def solve_imo_problem(self, problem_statement: str, max_iterations: int = 10) -> Dict:
        """Attempt to solve IMO problem through decoupled reasoning and proving."""

        solution = {
            'problem': problem_statement,
            'lemmas': [],
            'proof_sketch': '',
            'solved': False,
            'iterations': 0
        }

        for iteration in range(max_iterations):
            # Generate candidate lemmas
            candidates = self.reasoner.generate_lemma_candidates(
                problem_statement,
                num_candidates=5
            )

            # Verify each candidate
            verified_this_round = []
            for candidate in candidates:
                result = self.prover.verify_lemma(candidate['statement'])

                if result['verified']:
                    verified_this_round.append(result)
                    self.verified_lemmas.append(result)
                else:
                    # Store failure for feedback
                    self.failed_attempts.append({
                        'lemma': candidate['statement'],
                        'error': result['error_msg'],
                        'iteration': iteration
                    })

            # Check if we have enough lemmas to attempt proof
            if len(self.verified_lemmas) >= 5:
                proof_sketch = self.reasoner.generate_proof_sketch(
                    problem_statement,
                    [l['statement'] for l in self.verified_lemmas[:5]]
                )

                # Attempt formal proof
                solution['proof_sketch'] = proof_sketch
                solution['lemmas'] = self.verified_lemmas
                solution['iterations'] = iteration + 1

                # Check if proof is complete
                if self._is_complete_proof(proof_sketch):
                    solution['solved'] = True
                    break

        return solution

    def _is_complete_proof(self, proof_sketch: str) -> bool:
        """Check if proof sketch covers the problem without gaps."""
        # Heuristic: complete proofs contain 'therefore' and mention the target
        return 'therefore' in proof_sketch.lower() or 'qed' in proof_sketch.lower()

    def get_verified_lemmas(self) -> List[Dict]:
        """Retrieve all verified lemmas for future problems."""
        return self.verified_lemmas

    def save_lemma_repository(self, filepath: str):
        """Save verified lemmas for pattern reuse on similar problems."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.verified_lemmas, f)
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Lemma candidates per iteration | 5-10 | Balance diversity vs. verification cost |
| Max iterations | 8-15 | Time limit before giving up |
| Confidence threshold for reasoner | 0.6+ | Filter low-confidence lemmas |
| Lemma complexity limit | 3-5 predicates | Keep lemmas verifiable |
| Proof sketch length | 500-1000 tokens | Sufficient detail without bloat |
| Reasoner model size | 70B+ | Larger models better at proof strategy |
| Prover (Lean version) | Lean 4 | Most active development |

**When to use:** Apply decoupled reasoning and proving for competition mathematics (IMO, Putnam), formal verification of algorithms, and domains where rigorous proof is mandatory. Use when you have access to formal proof assistants and can annotate problems formally.

**When NOT to use:** Skip for informal mathematical reasoning where proofs aren't required. Avoid if computational budget is extremely limited—formal verification is expensive. Don't use for domains without established formal systems (e.g., empirical science). Skip if training data for your problem domain is very limited; the approach benefits from diverse lemma patterns.

**Common pitfalls:** Using too many lemma candidates (>10) wastes computation on verification. Too few (<3) misses solutions. Not providing enough problem context to the reasoner causes generic lemmas irrelevant to the problem. Formalizing lemmas incorrectly (bad informal→formal translation) causes false verification failures. Forgetting to reuse verified lemmas across iterations wastes verification effort. Setting confidence thresholds too high filters useful exploratory lemmas; too low includes noise.

## Reference

Liang, Z., Song, L., Yang, L., Li, Y., Zhang, F., Mi, H., & Yu, D. (2025). Towards Solving More Challenging IMO Problems via Decoupled Reasoning and Proving. arXiv preprint arXiv:2507.06804. https://arxiv.org/abs/2507.06804
