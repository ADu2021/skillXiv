---
name: test-driven-ai-agents
title: "Test-Driven AI Agent Definition: Compiling Tool-Using Agents from Behavioral Specs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.08806"
keywords: [Agent Testing, Behavioral Verification, Test-Driven Development, Prompt Engineering, Agent Safety]
description: "Applies test-driven development to agent prompts by iteratively refining prompts against behavioral test suites until compliance is achieved. Enables measurable agent behavior validation through semantic mutation testing and specification evolution."
---

# Test-Driven AI Agent Definition: Compiling Agent Prompts from Behavioral Specifications

Production LLM agents lack measurable behavioral compliance. Small prompt changes cause silent regressions, tool misuse goes undetected, and policy violations emerge only after deployment. Teams cannot verify agents behave correctly across specified scenarios without tedious manual testing. Test-Driven AI Agent Definition (TDAD) adapts test-driven development (TDD) to agents: write behavioral specs as executable tests, then iteratively refine prompts until all tests pass—like TDD for code but for agent prompts.

## Core Concept

Traditional approach: Write agent prompt → Deploy → Hope it works

TDAD: Write behavioral tests → Iteratively refine prompt to pass tests → Verify robustness through mutation testing

Key insight: Prompts are compilation artifacts. Tests specify desired behavior; the compiler (you + LLM) generates a prompt that makes the agent behave correctly. Tests provide objective verification unavailable with manual inspection.

## Architecture Overview

- **Behavioral Test Specification**: YAML specs defining must-do actions (MFT), invariants (INV), directives (DIR)
- **Visible/Hidden Test Split**: Visible tests guide prompt refinement; hidden tests measure generalization
- **Semantic Mutation Testing**: Generate faulty prompt variants to verify tests catch regressions
- **Spec Evolution**: Test robustness as specifications change over time
- **Multi-Role Compilation**: TestSmith generates tests, PromptSmith refines prompts, MutationSmith creates variants

## Implementation Steps

Implement a test-driven prompt refinement system.

**Behavioral Test Specification**

```python
import yaml
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class BehavioralSpec:
    """Specification of desired agent behavior."""
    task_description: str
    must_follow_tests: List[Dict[str, Any]]  # MFT: actions agent must perform
    invariants: List[Dict[str, Any]]  # INV: properties that must remain true
    directives: List[Dict[str, Any]]  # DIR: explicit instructions

    def to_yaml(self) -> str:
        """Serialize spec to YAML."""
        return yaml.dump({
            'task': self.task_description,
            'must_follow': self.must_follow_tests,
            'invariants': self.invariants,
            'directives': self.directives
        })

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'BehavioralSpec':
        """Load spec from YAML."""
        data = yaml.safe_load(yaml_str)
        return cls(
            task_description=data.get('task', ''),
            must_follow_tests=data.get('must_follow', []),
            invariants=data.get('invariants', []),
            directives=data.get('directives', [])
        )


# Example spec
AGENT_SPEC = """
task: "Financial transaction agent - approve or reject customer requests with proper authorization"

must_follow:
  - action: "check_customer_identity"
    before: ["approve_transaction"]
    reason: "Cannot approve without identity verification"
  - action: "verify_amount_limit"
    before: ["approve_transaction"]
    reason: "Amount must be within customer limits"
  - action: "log_decision"
    after: ["approve_transaction", "reject_transaction"]
    reason: "All decisions must be logged for audit"

invariants:
  - "approved_transactions must have amount < customer_limit"
  - "no transaction can be approved without identity check"
  - "rejection must include reason"
  - "all tool calls must be logged"

directives:
  - "Always confirm identity before proceeding"
  - "Reject requests exceeding limits without exception"
  - "When uncertain, default to rejection (safety first)"
  - "Provide clear reasoning for all decisions"
"""
```

**Test Generation (TestSmith)**

```python
import json
from typing import List, Tuple

class TestSmith:
    """Generates executable tests from behavioral specs."""

    def __init__(self, llm_model):
        self.model = llm_model

    def generate_tests_from_spec(self, spec: BehavioralSpec, num_test_cases: int = 20) -> List[Dict]:
        """
        Generate concrete test cases from abstract specification.

        Args:
            spec: BehavioralSpec instance
            num_test_cases: number of tests to generate

        Returns:
            tests: list of test case dictionaries
        """
        prompt = f"""Given this agent behavior specification:

{spec.to_yaml()}

Generate {num_test_cases} concrete test cases. Each test should:
1. Describe a scenario that tests one requirement
2. Specify the expected tool calls in sequence
3. Include an assertion to verify the requirement

Format: JSON array of {{
  "scenario": "description",
  "input": "user request",
  "expected_actions": ["action1", "action2"],
  "assertion": "verification logic"
}}"""

        # Call LLM to generate tests
        response = self.model.generate(prompt)
        tests = json.loads(response)

        return tests

    def compile_test_harness(self, tests: List[Dict]) -> str:
        """
        Compile test cases into executable harness.

        Args:
            tests: test case specifications

        Returns:
            harness_code: Python code that runs all tests
        """
        harness = """
import unittest

class AgentBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()

"""
        for i, test in enumerate(tests):
            harness += f"""
    def test_case_{i}(self):
        '''Test: {test.get('scenario', '')}'''
        result = self.agent.run('{test.get('input', '')}')
        expected_actions = {test.get('expected_actions', [])}

        # Verify all expected actions were called
        for action in expected_actions:
            self.assertIn(action, result['actions_called'])

        # Assertion
        self.assertTrue({test.get('assertion', 'True')})
"""

        harness += """
if __name__ == '__main__':
    unittest.main()
"""
        return harness
```

**Prompt Refinement (PromptSmith)**

```python
class PromptSmith:
    """Iteratively refines prompts to pass behavioral tests."""

    def __init__(self, model, max_iterations=10):
        self.model = model
        self.max_iterations = max_iterations

    def compile_prompt(self, spec: BehavioralSpec, tests: List[Dict]) -> str:
        """
        Compile a prompt that passes all visible tests.

        Args:
            spec: behavioral specification
            tests: test cases to pass

        Returns:
            compiled_prompt: agent prompt that passes tests
        """
        prompt_candidate = self._initial_prompt(spec)

        for iteration in range(self.max_iterations):
            # Run tests against current prompt
            test_results = self._run_tests(prompt_candidate, tests)
            passing = sum(1 for r in test_results if r['passed'])

            print(f"[Iteration {iteration+1}] {passing}/{len(tests)} tests passing")

            if passing == len(tests):
                print("✓ All tests passing!")
                return prompt_candidate

            # Analyze failures
            failures = [r for r in test_results if not r['passed']]

            # If many failures, split into fast vs full suite
            if len(failures) > len(tests) * 0.5:
                # Run only failing tests for efficiency
                failing_test_subset = [
                    tests[test_results.index(f)]
                    for f in failures[:min(10, len(failures))]
                ]
            else:
                failing_test_subset = [
                    tests[i] for i, r in enumerate(test_results) if not r['passed']
                ]

            # Identify root cause of failures
            root_causes = self._analyze_failures(prompt_candidate, failing_test_subset)

            # Refine prompt with minimal edits
            prompt_candidate = self._apply_minimal_edits(
                prompt_candidate, root_causes, spec
            )

        return prompt_candidate

    def _initial_prompt(self, spec: BehavioralSpec) -> str:
        """Generate initial prompt from spec."""
        base_prompt = f"""You are an agent that must follow these behavioral requirements:

{spec.task_description}

Key requirements:
"""
        for test in spec.must_follow_tests:
            base_prompt += f"\n- {test.get('reason', test)}"

        for invariant in spec.invariants:
            base_prompt += f"\n- MUST: {invariant}"

        for directive in spec.directives:
            base_prompt += f"\n- {directive}"

        return base_prompt

    def _run_tests(self, prompt: str, tests: List[Dict]) -> List[Dict]:
        """Run behavioral tests against prompt."""
        results = []

        for test in tests:
            # Simulate running agent with this prompt
            agent_response = self.model.run_agent(prompt, test['input'])

            # Check if response matches expected actions
            expected_actions = test.get('expected_actions', [])
            actual_actions = self._extract_actions(agent_response)

            passed = all(action in actual_actions for action in expected_actions)

            results.append({
                'test': test,
                'passed': passed,
                'actual_actions': actual_actions,
                'missing_actions': [a for a in expected_actions if a not in actual_actions]
            })

        return results

    def _analyze_failures(self, prompt: str, failing_tests: List[Dict]) -> List[str]:
        """Identify common root causes of test failures."""
        root_causes = []

        for test in failing_tests:
            # What did the prompt fail to enforce?
            failure_reason = f"Test '{test['scenario']}' failed because "
            failure_reason += f"missing actions: {test.get('missing_actions', [])}"
            root_causes.append(failure_reason)

        return root_causes

    def _apply_minimal_edits(self, prompt: str, root_causes: List[str], spec: BehavioralSpec) -> str:
        """Apply minimal edits to fix failures."""
        # Ask LLM for minimal prompt edits
        edit_prompt = f"""Current prompt:
{prompt}

Failures:
{chr(10).join(root_causes)}

Make minimal edits to fix these failures. Return only the refined prompt:"""

        refined = self.model.generate(edit_prompt)
        return refined

    def _extract_actions(self, response: str) -> List[str]:
        """Extract tool calls/actions from agent response."""
        # Simplified: look for tool call patterns
        import re
        actions = re.findall(r'call_(\w+)', response)
        return actions
```

**Semantic Mutation Testing (MutationSmith)**

```python
class MutationSmith:
    """Generate faulty prompt variants to verify test robustness."""

    def __init__(self, model):
        self.model = model

    def generate_mutants(self, prompt: str, num_mutants: int = 5) -> List[Tuple[str, str]]:
        """
        Generate plausible faulty prompt variants.

        Args:
            prompt: original (correct) prompt
            num_mutants: number of variants to create

        Returns:
            mutants: list of (mutant_prompt, mutation_description)
        """
        mutation_types = [
            "remove the requirement to check identity",
            "remove the amount limit check",
            "change rejection default to approval",
            "remove logging requirement",
            "add contradictory instruction"
        ]

        mutants = []
        for i in range(min(num_mutants, len(mutation_types))):
            mutation_desc = mutation_types[i]
            mutant_prompt = self._apply_mutation(prompt, mutation_desc)
            mutants.append((mutant_prompt, mutation_desc))

        return mutants

    def _apply_mutation(self, prompt: str, mutation_desc: str) -> str:
        """Apply a specific mutation to the prompt."""
        mutation_prompt = f"""Original prompt:
{prompt}

Apply this mutation: {mutation_desc}
Return only the mutated prompt:"""

        return self.model.generate(mutation_prompt)

    def verify_tests_catch_mutations(
        self,
        tests: List[Dict],
        mutants: List[Tuple[str, str]]
    ) -> bool:
        """
        Verify that test suite catches faulty prompt variants.

        Args:
            tests: behavioral tests
            mutants: list of (mutant_prompt, description)

        Returns:
            all_caught: whether all mutants were detected
        """
        all_caught = True

        for mutant_prompt, mutation_desc in mutants:
            # Run tests against mutant
            # If any test still passes, the mutation wasn't caught
            mutant_passes_all = all(
                self._test_passes(test, mutant_prompt)
                for test in tests
            )

            if mutant_passes_all:
                print(f"✗ Mutation not caught: {mutation_desc}")
                all_caught = False
            else:
                print(f"✓ Mutation caught: {mutation_desc}")

        return all_caught

    def _test_passes(self, test: Dict, prompt: str) -> bool:
        """Check if single test passes against prompt."""
        # Simplified test execution
        return False  # Would run actual test
```

## Practical Guidance

**Hyperparameters**:
- Max iterations: 8-10 (diminishing returns after that)
- Test subset size when filtering: ~10 failing tests
- Mutation count: 5-10 semantic variants
- Visible/hidden split: 70/30 or 80/20

**When to Apply**:
- Production agents where behavior must be verified
- Systems requiring audit trails and compliance
- Multi-tool agents where tool misuse is risky
- Teams without domain expertise to manually inspect prompts

**When NOT to Apply**:
- Simple single-tool agents
- Research experiments (overhead not justified)
- Real-time agents where compilation latency matters
- Domains where test specification is ambiguous

**Key Pitfalls**:
- Test suite too permissive—catches no regressions
- Mutation space too large—expensive to verify
- Visible tests too similar to hidden tests—overfitting
- Not accounting for stochasticity—flaky tests

**Integration Notes**: Works as a pre-deployment verification step; requires defining behavioral specs in YAML; tests run against mock/simulation of agent; compilation is iterative and may require 5-15 minutes.

**Evidence**: Reduces silent regressions from prompt edits by 95%; catches behavioral violations that manual inspection misses; enables non-experts to verify agent behavior through tests.

Reference: https://arxiv.org/abs/2603.08806
