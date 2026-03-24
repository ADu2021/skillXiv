---
name: hardtests-code-verification
title: "HardTests: Synthesizing High-Quality Test Cases for LLM Coding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24098"
keywords: [Test Synthesis, Code Verification, LLM Reasoning, Edge Cases]
description: "Generate comprehensive test cases for code problems that reliably detect wrong solutions through LLM-based edge case synthesis and test quality ranking."
---

# Synthesize Edge-Case Tests That Catch Wrong Solutions

Verifying code correctness is critical for LLM post-training, but automated test generation struggles with hard problems where wrong solutions are cleverly disguised. HardTests introduces HARDTESTGEN: an LLM-based pipeline that synthesizes high-quality test cases by actively discovering edge cases that reveal subtle bugs. Unlike simple random testing, this approach systematically explores the solution space to find failure-inducing inputs.

The key insight is that humans write effective tests by thinking about edge cases—boundary conditions, off-by-one errors, special cases. LLMs can emulate this reasoning by being prompted to think adversarially: "What inputs would break incorrect implementations?" By generating many candidate tests and ranking them by their ability to discriminate between correct and wrong solutions, the pipeline builds powerful test suites.

## Core Concept

HARDTESTGEN works in three phases:

- **Test generation**: Use LLM to generate diverse test cases, explicitly including edge cases and boundary conditions
- **Test discrimination**: Filter tests by their ability to distinguish correct from incorrect solutions
- **Test diversity**: Ensure tests cover different failure modes and input types
- **Quality ranking**: Rank tests by discriminative power across multiple solution variants

The framework recognizes that correct solutions often fail on specific edge cases while buggy solutions fail elsewhere. By finding tests that separate these failure patterns, you create a verifier that catches disguised wrong answers.

## Architecture Overview

- **Problem analysis module**: Extract key problem characteristics (constraints, edge cases)
- **Candidate test generator**: Create diverse test cases including edge cases, boundaries, and random inputs
- **Solution evaluator**: Run tests against correct and incorrect reference solutions
- **Discriminator function**: Measure test quality by how many wrong solutions it rejects
- **Diversity optimizer**: Select diverse test set rather than redundant hard tests
- **Ranking system**: Prioritize tests by their utility for verification

## Implementation

Build a test synthesis pipeline that generates and ranks test cases:

```python
# HardTests: Test synthesis for LLM code verification
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Set

class HardTestGenerator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_candidate_tests(self, problem_description: str, num_candidates: int = 20) -> List[str]:
        """
        Generate diverse test cases by prompting LLM to think about edge cases.
        """
        prompt = f"""You are a software engineer creating comprehensive tests for a coding problem.
Problem: {problem_description}

Generate {num_candidates} test cases that would catch incorrect solutions. Include:
1. Basic valid inputs
2. Edge cases (empty, single element, maximum/minimum values)
3. Boundary conditions
4. Off-by-one errors
5. Special cases (zero, negative, duplicates)

For each test, provide:
- Input specification
- Expected output
- Why this test matters

Generate test cases:"""

        # Generate multiple test scenarios
        response = self.model.generate(
            self.tokenizer.encode(prompt, return_tensors='pt'),
            max_length=2000,
            temperature=0.8,
            num_return_sequences=1
        )
        test_text = self.tokenizer.decode(response[0])

        # Parse test cases from response
        tests = self._parse_test_cases(test_text)
        return tests[:num_candidates]

    def _parse_test_cases(self, test_text: str) -> List[str]:
        """Parse test specifications from LLM output"""
        tests = []
        lines = test_text.split('\n')
        current_test = []

        for line in lines:
            if line.strip().startswith('Input:') or line.strip().startswith('Test'):
                if current_test:
                    tests.append('\n'.join(current_test))
                current_test = [line]
            elif current_test:
                current_test.append(line)

        if current_test:
            tests.append('\n'.join(current_test))

        return tests

    def rank_test_by_discrimination(self, test: str, correct_solutions: List[str],
                                   wrong_solutions: List[str]) -> float:
        """
        Measure test quality by how well it discriminates between correct and wrong solutions.
        Score = (fraction_wrong_fail) - (fraction_correct_fail)
        """
        correct_passed = 0
        wrong_failed = 0

        # Execute test on correct solutions
        for sol in correct_solutions:
            try:
                if self._test_solution(sol, test):
                    correct_passed += 1
            except:
                pass  # Syntax error on correct solution is bad

        # Execute test on wrong solutions
        for sol in wrong_solutions:
            try:
                if not self._test_solution(sol, test):
                    wrong_failed += 1
            except:
                wrong_failed += 1  # Syntax errors count as solution failure

        correct_pass_rate = correct_passed / len(correct_solutions) if correct_solutions else 1.0
        wrong_fail_rate = wrong_failed / len(wrong_solutions) if wrong_solutions else 0.0

        # Score: reward failing wrong solutions, penalize failing correct solutions
        discrimination_score = wrong_fail_rate - (1 - correct_pass_rate)
        return max(0, discrimination_score)

    def _test_solution(self, solution_code: str, test_spec: str) -> bool:
        """Execute a test against a solution (simplified version)"""
        # In practice, this would:
        # 1. Parse test input from test_spec
        # 2. Execute solution_code with that input
        # 3. Compare output to expected
        # 4. Return whether test passed
        try:
            # Create test execution context
            exec_globals = {}
            exec(solution_code, exec_globals)

            # Extract and run test
            # This is pseudocode; real implementation parses test format
            test_input, expected_output = self._parse_test_spec(test_spec)
            result = exec_globals['solve'](test_input)
            return result == expected_output
        except Exception as e:
            return False

    def _parse_test_spec(self, test_spec: str) -> Tuple:
        """Parse test specification to extract input and expected output"""
        # Implementation depends on test format
        # Returns (input, expected_output)
        return None, None

    def synthesize_test_suite(self, problem: str, num_tests: int = 10) -> List[str]:
        """
        Full pipeline: generate candidates, rank by discrimination, return best tests.
        """
        # Generate candidates (more than needed)
        candidates = self.generate_candidate_tests(problem, num_candidates=50)

        # For ranking, we need reference solutions
        # In practice, get correct solution from problem context
        correct_solutions = [self._get_reference_solution(problem)]

        # Generate intentional wrong solutions for ranking
        wrong_solutions = self._generate_wrong_solutions(problem, num_wrong=5)

        # Rank candidates
        scored_tests = []
        for test in candidates:
            score = self.rank_test_by_discrimination(test, correct_solutions, wrong_solutions)
            scored_tests.append((test, score))

        # Sort by discrimination score
        scored_tests.sort(key=lambda x: x[1], reverse=True)

        # Select diverse subset (not just high-scoring)
        selected = self._select_diverse_subset(scored_tests, num_tests)

        return [t[0] for t in selected]

    def _generate_wrong_solutions(self, problem: str, num_wrong: int = 5) -> List[str]:
        """Generate intentionally wrong solutions to test against"""
        prompt = f"""Generate {num_wrong} intentionally incorrect solutions to this problem:
{problem}

Each solution should have a subtle bug (off-by-one, wrong condition, etc.)
that wouldn't be immediately obvious but would fail on edge cases."""

        # LLM generates wrong solutions
        # (simplified; real version would be more careful)
        return ["# wrong_solution_1", "# wrong_solution_2"]

    def _get_reference_solution(self, problem: str) -> str:
        """Get or generate a correct reference solution"""
        # In practice, extract from problem context or generate with high-confidence model
        return "def solve(x):\n    pass"

    def _select_diverse_subset(self, scored_tests: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
        """Select k diverse tests that cover different failure modes"""
        selected = []
        remaining = scored_tests.copy()

        # Greedy selection: pick highest-scoring, then ones that are most different
        while len(selected) < k and remaining:
            # Pick highest score
            best = remaining.pop(0)
            selected.append(best)

            # Remove similar tests (to ensure diversity)
            remaining = [t for t in remaining if not self._are_similar_tests(best[0], t[0])]

        return selected[:k]

    def _are_similar_tests(self, test1: str, test2: str) -> bool:
        """Check if two tests are too similar (cover same failure mode)"""
        # Simple heuristic: string similarity
        common_tokens = len(set(test1.split()) & set(test2.split()))
        return common_tokens > len(set(test1.split())) * 0.7
```

Implement a verification utility for code solutions using the generated tests:

```python
def verify_solution_with_hard_tests(solution_code: str, test_suite: List[str]) -> dict:
    """
    Verify a code solution against high-quality test suite.
    Returns detailed results about which tests pass/fail.
    """
    results = {
        'passes': 0,
        'fails': 0,
        'errors': 0,
        'failed_tests': [],
        'error_tests': []
    }

    for test_idx, test_spec in enumerate(test_suite):
        try:
            # Execute test
            if test_passes(solution_code, test_spec):
                results['passes'] += 1
            else:
                results['fails'] += 1
                results['failed_tests'].append((test_idx, test_spec))
        except Exception as e:
            results['errors'] += 1
            results['error_tests'].append((test_idx, str(e)))

    results['score'] = results['passes'] / len(test_suite)
    return results

def test_passes(solution_code: str, test_spec: str) -> bool:
    """Execute a test specification against solution code"""
    # Minimal implementation
    try:
        # Run solution with test input
        exec_context = {}
        exec(solution_code, exec_context)
        # Compare output to expected (test_spec parsing needed)
        return True  # Placeholder
    except:
        return False
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|-----------------|-------|
| Candidate generation | 3-5x needed tests | Generate excess, filter by quality |
| Test discrimination threshold | 0.5+ score | Tests should fail >50% wrong solutions |
| Reference solutions | 1 correct + 5-10 wrong | Balance verification cost with ranking quality |
| Test diversity | Ensure coverage of ≥3 failure modes | Avoid redundant tests |
| Test format | Structured input/output pairs | Enables automated execution |

**When to use HardTests:**
- Verifying complex coding problems with subtle edge cases
- Post-training LLMs with RL/DPO on code tasks
- Want to catch disguised wrong solutions (clever bugs)
- Building robust test suites for code competitions
- Need high-confidence verification without human review

**When NOT to use:**
- Simple problems with obvious correct/wrong answers
- Test suites already exist and are comprehensive
- Edge cases don't matter (prototyping, tutorials)
- Testing speed is more important than accuracy
- Problem correctness is hard to formalize (creative coding)

**Common pitfalls:**
- Generating only obvious test cases (boundary value testing sufficient)
- Not ranking tests by discrimination ability
- Too few reference wrong solutions for effective ranking
- Tests that are too similar (redundant coverage)
- Overrelying on LLM-generated wrong solutions without validation
- Not considering solution variability (different approaches, languages)

## Reference

**HardTests: Synthesizing High-Quality Test Cases for LLM Coding**
https://arxiv.org/abs/2505.24098
