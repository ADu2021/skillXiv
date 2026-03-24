---
name: adversarial-code-verification
title: "Rethinking Verification for LLM Code Generation: From Generation to Testing"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06920"
keywords: [Code Verification, Test Generation, Adversarial Testing, Code Generation Evaluation, Program Verification]
description: "Generate adversarial test suites that catch LLM code errors by analyzing constraint differences between correct and incorrect solutions. SAGA framework improves test detection rate 9.55% and verifier accuracy 12.14% by using human bug patterns and multi-input differential analysis."
---

# Adversarial Code Verification: Generating Discriminative Test Cases from Solution Patterns

Current benchmarks for LLM code generation use limited, homogeneous test suites that inflate performance metrics and fail to catch real errors. A model scoring 95% on standard benchmarks may fail 50% of real submissions because test suites are too narrow, allowing solutions with subtle bugs to pass. SAGA solves this through strategic analysis: first, extract constraint insights from correct solutions to identify edge cases; second, analyze patterns in human bug submissions to create targeted test cases; third, generate diverse inputs exploring different error classes. The result is test suites that reliably distinguish correct from incorrect code, giving RL systems honest reward signals.

When training code generation models with RL, weak test suites corrupt the reward signal—models learn to pass bad evaluations rather than produce correct code. SAGA test suites catch bugs that simple unit tests miss: off-by-one errors in loops, incorrect boundary handling, type confusion, and logic inversions. The human-LLM collaborative approach combines human insight about what makes problems hard with LLM efficiency at generating diverse test cases.

## Core Concept

SAGA generates adversarial tests through three coordinated phases. First, multidimensional analysis examines correct solutions to extract constraint patterns: what boundary conditions do they handle, what invariants do they maintain? Second, differential analysis compares failed solutions with corrected versions to identify common error patterns—these patterns become targets for test generation. Third, dual-input strategy combines both ground-truth and buggy solutions to produce tests that distinguish right from wrong. The framework introduces new evaluation metrics beyond simple accuracy: detection rate (what proportion of known bugs does the test suite find), verifier accuracy (does the suite correctly identify all incorrect solutions), and distinct error pattern coverage (breadth of error types detected).

## Architecture Overview

- **Multidimensional Analyzer**: Extracts constraint patterns, boundary conditions, and invariants from correct solutions
- **Differential Analyzer**: Compares correct vs. buggy submissions to identify error patterns
- **Constraint-Differential Test Generator**: Creates test cases targeting specific constraint violations and error patterns
- **Test Discriminator**: Validates that generated tests actually distinguish correct from incorrect code
- **Metrics Engine**: Computes detection rate, verifier accuracy, and error pattern coverage
- **Human-LLM Collaborative Pipeline**: Combines human problem insight with LLM test generation at scale

## Implementation

This example demonstrates extracting constraint patterns from correct solutions to guide test generation.

```python
# Constraint-based adversarial test generation
import ast
import torch

class ConstraintAnalyzer:
    def __init__(self):
        self.constraints = {}
        self.boundaries = {}

    def analyze_correct_solution(self, solution_code, problem_description):
        """Extract constraints and boundary conditions from correct code."""

        # Parse solution AST
        tree = ast.parse(solution_code)

        # Extract constraint patterns
        constraints = {
            'loop_bounds': [],
            'conditionals': [],
            'type_checks': [],
            'boundary_values': []
        }

        # Analyze loops: what are iteration bounds?
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Extract loop range
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        # range(N), range(0, N), range(0, N, 1)
                        bounds = self._extract_range_bounds(node.iter)
                        constraints['loop_bounds'].append(bounds)

            elif isinstance(node, ast.If):
                # Extract condition patterns
                condition = ast.unparse(node.test)
                constraints['conditionals'].append(condition)

            elif isinstance(node, ast.Compare):
                # Extract comparison patterns (boundary checks)
                if len(node.ops) > 0:
                    op_type = type(node.ops[0]).__name__
                    constraints['boundary_values'].append({
                        'operator': op_type,
                        'condition': ast.unparse(node)
                    })

        return constraints

    def extract_boundary_test_cases(self, constraints, variable_bounds):
        """Generate boundary test cases from extracted constraints."""

        test_cases = []

        # For each boundary condition, create test cases at boundaries
        for boundary in constraints['boundary_values']:
            # Extract the compared value
            if 'N' in boundary['condition']:
                # Boundary around N: test N-1, N, N+1
                for n in variable_bounds:
                    test_cases.append(n - 1 if n > 0 else 0)
                    test_cases.append(n)
                    test_cases.append(n + 1)

            # Test zero and large values
            test_cases.extend([0, 1, variable_bounds[-1], variable_bounds[-1] + 1])

        return list(set(test_cases))  # Remove duplicates
```

This example shows differential analysis: comparing correct and incorrect solutions to identify bug patterns.

```python
class DifferentialBugAnalyzer:
    def __init__(self):
        self.error_patterns = {}

    def analyze_bug_difference(self, correct_code, buggy_code, test_input, expected_output):
        """Identify what makes the buggy code fail on this input."""

        # Execute both versions
        try:
            correct_output = execute_solution(correct_code, test_input)
        except:
            return None

        try:
            buggy_output = execute_solution(buggy_code, test_input)
        except Exception as e:
            return {
                'error_type': 'runtime_error',
                'error_msg': str(e),
                'test_input': test_input
            }

        # Compare outputs
        if correct_output != buggy_output:
            # Analyze AST differences
            correct_ast = ast.parse(correct_code)
            buggy_ast = ast.parse(buggy_code)

            diffs = self._compute_ast_differences(correct_ast, buggy_ast)

            return {
                'error_type': 'wrong_output',
                'test_input': test_input,
                'expected': expected_output,
                'actual': buggy_output,
                'code_differences': diffs,
                'error_pattern': self._classify_error_pattern(diffs)
            }

        return None

    def _classify_error_pattern(self, code_diffs):
        """Classify bug type from code differences."""

        if any('off-by-one' in diff for diff in code_diffs):
            return 'off_by_one'
        elif any('<=' in diff or '>=' in diff for diff in code_diffs):
            return 'boundary_condition'
        elif any('not' in diff or '!' in diff for diff in code_diffs):
            return 'logic_inversion'
        elif any('int' in diff or 'float' in diff for diff in code_diffs):
            return 'type_error'
        else:
            return 'other'

    def collect_error_patterns(self, correct_solutions, buggy_submissions, test_suite):
        """Analyze all buggy submissions to build error pattern library."""

        error_patterns = {}

        for buggy_code in buggy_submissions:
            for correct_code in correct_solutions:
                for test_input, expected_output in test_suite:
                    bug_info = self.analyze_bug_difference(
                        correct_code, buggy_code, test_input, expected_output
                    )

                    if bug_info:
                        pattern = bug_info['error_pattern']
                        if pattern not in error_patterns:
                            error_patterns[pattern] = []
                        error_patterns[pattern].append(bug_info)

        return error_patterns
```

This example demonstrates SAGA test generation combining constraints and error patterns.

```python
class SAGATestGenerator:
    def __init__(self, constraint_analyzer, bug_analyzer):
        self.constraint_analyzer = constraint_analyzer
        self.bug_analyzer = bug_analyzer

    def generate_adversarial_tests(self, correct_solutions, buggy_submissions, problem_spec):
        """Generate adversarial test suite using SAGA framework."""

        test_suite = []

        # Phase 1: Extract constraints from correct solutions
        all_constraints = []
        all_boundaries = []

        for correct_code in correct_solutions:
            constraints = self.constraint_analyzer.analyze_correct_solution(
                correct_code, problem_spec
            )
            all_constraints.append(constraints)
            boundaries = self.constraint_analyzer.extract_boundary_test_cases(
                constraints, [1, 10, 100, 1000]
            )
            all_boundaries.extend(boundaries)

        # Phase 2: Analyze error patterns in buggy submissions
        error_patterns = self.bug_analyzer.collect_error_patterns(
            correct_solutions, buggy_submissions, test_suite
        )

        # Phase 3: Generate tests targeting each error pattern
        for pattern_type, bug_examples in error_patterns.items():
            # What inputs trigger this pattern?
            triggering_inputs = [example['test_input'] for example in bug_examples]

            # Generate variations on triggering inputs
            generated_tests = self._mutate_inputs(triggering_inputs, pattern_type)
            test_suite.extend(generated_tests)

        # Phase 4: Add boundary tests
        boundary_tests = self._generate_boundary_tests(all_boundaries)
        test_suite.extend(boundary_tests)

        return test_suite

    def _mutate_inputs(self, base_inputs, error_pattern):
        """Create input variations targeting specific error patterns."""

        mutations = []

        for base_input in base_inputs:
            # For off-by-one errors: test N-1, N, N+1
            if error_pattern == 'off_by_one':
                if isinstance(base_input, int):
                    mutations.extend([base_input - 1, base_input, base_input + 1])
                elif isinstance(base_input, list):
                    mutations.append(base_input[:-1])  # Shorter list
                    mutations.append(base_input + [0])  # Longer list

            # For boundary errors: test at limits
            elif error_pattern == 'boundary_condition':
                mutations.append(0)
                mutations.append(1)
                mutations.append(len(base_input) - 1 if isinstance(base_input, (list, str)) else base_input)

            # For logic inversion: complementary cases
            elif error_pattern == 'logic_inversion':
                mutations.append(not base_input if isinstance(base_input, bool) else -base_input)

        return mutations

    def compute_test_quality_metrics(self, test_suite, correct_solutions, buggy_submissions):
        """Compute detection rate and verifier accuracy."""

        detection_rate = 0.0
        verifier_accuracy = 0.0

        bugs_detected = 0
        total_bugs = len(buggy_submissions)

        for buggy_code in buggy_submissions:
            for test_input, expected_output in test_suite:
                try:
                    actual_output = execute_solution(buggy_code, test_input)
                    if actual_output != expected_output:
                        bugs_detected += 1
                        break  # This test catches this bug
                except:
                    bugs_detected += 1
                    break

        detection_rate = bugs_detected / total_bugs if total_bugs > 0 else 1.0

        # Verifier accuracy: correct solutions pass all tests
        correct_pass_all = 0
        for correct_code in correct_solutions:
            if all(
                execute_solution(correct_code, test_input) == expected_output
                for test_input, expected_output in test_suite
            ):
                correct_pass_all += 1

        verifier_accuracy = correct_pass_all / len(correct_solutions)

        return {
            'detection_rate': detection_rate,
            'verifier_accuracy': verifier_accuracy,
            'bugs_detected': bugs_detected,
            'total_bugs': total_bugs
        }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Boundary offset range | [-2, -1, 0, +1, +2] | Comprehensive boundary testing |
| Max test cases per error pattern | 50 | Balance coverage vs. test suite bloat |
| Constraint extraction depth | 3 iterations | Find nested constraints |
| Error pattern categories | 8+ types | Broad error coverage |
| Detection rate target | > 95% | Catch most buggy submissions |
| Verifier accuracy target | > 98% | Minimize false positives |
| Buggy submission sample size | 100+ | Representative error patterns |

**When to use:** Apply SAGA when evaluating code generation models with RL, where test quality directly affects learned behavior. Use when you have access to buggy human submissions (competitive programming sites) or generated solutions to analyze. Ideal for creating reliable evaluation benchmarks and improving RL reward signals.

**When NOT to use:** Skip if you only have correct reference solutions without buggy examples—differential analysis requires both. Avoid if test generation latency matters (SAGA is compute-intensive). Don't use for simple, well-defined problems where boundary testing suffices. Skip if your problem domain lacks clear error patterns (e.g., creative writing evaluation).

**Common pitfalls:** Using only boundary cases misses logic inversion and type errors. Not analyzing enough buggy submissions leads to sparse error pattern coverage. Generating tests without validation allows degenerate cases that don't actually distinguish correct from incorrect code. Over-weighting boundary tests at expense of pattern-based tests reduces real-world error detection. Forgetting to verify that correct solutions pass the test suite before using it for evaluation. Not computing detection rate separately for different error pattern types hides coverage gaps.

## Reference

Code Verification Team. (2025). Rethinking Verification for LLM Code Generation: From Generation to Testing. arXiv preprint arXiv:2507.06920. https://arxiv.org/abs/2507.06920
