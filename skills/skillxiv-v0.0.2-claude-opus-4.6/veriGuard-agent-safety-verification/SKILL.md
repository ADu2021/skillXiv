---
name: veriGuard-agent-safety-verification
title: "VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.05156"
keywords: [Agent Safety, Formal Verification, Code Generation, Policy Enforcement, Runtime Monitoring]
description: "Generate safety policies as executable code with formal verification, enabling provably-correct agent behavior through offline policy generation and online runtime enforcement."
---

# Technique: Formal Verification for Agent Safety Policies

Reactive guardrails for agent safety are prone to evasion. VeriGuard takes a proactive approach by formalizing security requirements as executable policies that are mathematically verified before deployment. Rather than detecting violations reactively, verified policies prevent unsafe actions by construction.

The key insight is treating safety as a code generation and verification problem: translate natural language security requirements into Python code paired with formal constraints, verify the code satisfies constraints mathematically, then enforce at runtime. This "correct-by-construction" approach provides formal guarantees rather than heuristic detection.

## Core Concept

VeriGuard operates through offline and online phases:

1. **Offline Policy Generation**: Translate requirements into executable policies with formal specs
2. **Three-Stage Refinement**: Validation (resolve ambiguities), testing (verify functionality), formal verification (prove correctness)
3. **Online Enforcement**: Runtime monitoring using verified policies

## Architecture Overview

- **Input**: Natural language security requirement
- **Specification**: Translate to Python + formal constraints
- **Validation Phase**: LLM resolves ambiguities in spec
- **Testing Phase**: Execute against test cases, refine on failures
- **Verification Phase**: Formal verifier (e.g., Nagini) proves spec compliance
- **Runtime**: Intercept agent actions, evaluate against verified policies
- **Response**: Block/allow/replicate based on verification result

## Implementation Steps

Implement policy specification generation.

```python
class SafetyPolicyGenerator:
    def __init__(self, llm_model, verifier_tool='nagini'):
        self.model = llm_model
        self.verifier = verifier_tool

    def generate_policy_from_requirement(self, security_requirement, context=''):
        """
        Generate executable policy from natural language requirement.

        Args:
            security_requirement: English description of security constraint
            context: Additional context (e.g., agent action types)

        Returns:
            policy: Dict with 'code', 'spec', 'status'
        """

        prompt = f"""Generate a Python policy function that enforces this requirement:

Requirement: {security_requirement}

Context: {context}

Format your response as:

CODE:
```python
def safety_policy(action, context, **kwargs):
    '''
    Pre-condition: [describe what must be true before]
    Post-condition: [describe what must be true after]
    '''
    # Policy implementation
    if <violation_condition>:
        return {'allowed': False, 'reason': '<reason>'}
    return {'allowed': True}
```

SPEC:
[Formal properties this policy ensures]
"""

        response = self.model.generate(prompt)

        # Parse code and spec
        code = self._extract_code(response)
        spec = self._extract_spec(response)

        return {'code': code, 'spec': spec, 'status': 'generated'}

    def _extract_code(self, response):
        """Extract Python code block."""
        import re
        match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def _extract_spec(self, response):
        """Extract formal specification."""
        if 'SPEC:' in response:
            return response.split('SPEC:')[1].strip()
        return ""
```

Implement validation phase for specification refinement.

```python
def validate_policy_specification(generator, code, spec, ambiguity_examples):
    """
    Validate policy by checking for ambiguities and refining.

    Args:
        generator: SafetyPolicyGenerator
        code: Policy code
        spec: Formal specification
        ambiguity_examples: List of edge cases

    Returns:
        refined_policy: Refined code and spec
    """

    validation_prompt = f"""Review this policy for ambiguities:

CODE:
{code}

SPEC:
{spec}

EDGE CASES:
{ambiguity_examples}

Identify potential ambiguities and provide clarifications."""

    clarifications = generator.model.generate(validation_prompt)

    # Refine code based on clarifications
    refinement_prompt = f"""Update the policy to address these clarifications:

Original: {code}
Clarifications: {clarifications}

Provide refined code."""

    refined_code = generator._extract_code(
        generator.model.generate(refinement_prompt)
    )

    return {'code': refined_code, 'spec': spec, 'status': 'validated'}
```

Implement testing phase.

```python
def test_policy(policy_code, test_cases):
    """
    Execute policy against test cases and fix failures.

    Args:
        policy_code: Generated policy function code
        test_cases: List of (action, expected_result) tuples

    Returns:
        test_results: Pass/fail for each test
    """

    # Compile policy function
    exec_globals = {}
    exec(policy_code, exec_globals)
    policy_func = exec_globals['safety_policy']

    test_results = []

    for action, expected_result in test_cases:
        try:
            result = policy_func(action, context={})
            passed = result['allowed'] == expected_result
            test_results.append({
                'action': action,
                'expected': expected_result,
                'actual': result['allowed'],
                'passed': passed
            })
        except Exception as e:
            test_results.append({
                'action': action,
                'error': str(e),
                'passed': False
            })

    return test_results
```

Implement formal verification using Nagini.

```python
def verify_policy_formally(policy_code, formal_spec, verifier_tool='nagini'):
    """
    Formally verify policy meets specification using automated verifier.

    Args:
        policy_code: Python code with pre/postconditions
        formal_spec: Formal specification
        verifier_tool: Verification tool (nagini, dafny, etc.)

    Returns:
        verification_result: {'passed': bool, 'details': str}
    """

    # Prepare code with formal annotations
    annotated_code = add_formal_annotations(policy_code, formal_spec)

    # Run verifier
    if verifier_tool == 'nagini':
        import subprocess
        result = subprocess.run(['nagini', '--python3', '-i'],
                              input=annotated_code, capture_output=True,
                              text=True)

        passed = result.returncode == 0
        details = result.stdout + result.stderr

    else:
        raise ValueError(f"Unknown verifier: {verifier_tool}")

    return {'passed': passed, 'details': details, 'code': annotated_code}


def add_formal_annotations(code, spec):
    """Add Nagini/Dafny annotations to Python code."""
    annotated = f'''# Specification: {spec}
# Pre-condition: Action is not None and has required fields
# Post-condition: Result is dict with 'allowed' key

{code}
'''
    return annotated
```

Implement runtime enforcement.

```python
class PolicyEnforcer:
    def __init__(self, verified_policies):
        """
        Initialize enforcer with verified policies.

        Args:
            verified_policies: List of verified policy codes
        """
        self.policies = []

        for policy_code in verified_policies:
            exec_globals = {}
            exec(policy_code, exec_globals)
            self.policies.append(exec_globals['safety_policy'])

    def evaluate_action(self, action, context):
        """
        Check action against all policies.

        Args:
            action: Proposed action
            context: Execution context

        Returns:
            result: {'allowed': bool, 'blocking_policies': list}
        """

        blocking_policies = []

        for policy in self.policies:
            result = policy(action, context)

            if not result['allowed']:
                blocking_policies.append(result.get('reason', 'Policy violation'))

        allowed = len(blocking_policies) == 0

        return {'allowed': allowed, 'blocking_policies': blocking_policies}
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Specification language | Clear, unambiguous English | Formal specs derive from natural language; precision critical |
| Verification tool | Nagini for Python | Supports subset of Python with formal reasoning |
| Test coverage | At least 10-20 cases per policy | More cases improve policy quality |
| Verification failure | Refine spec and iterate | Rarely succeed on first attempt; expectation is iteration |
| When to use | Critical safety-sensitive agents | Autonomous systems, financial decisions, access control |
| When NOT to use | Soft constraints or fuzzy requirements | Formal verification requires precise specs |
| Common pitfall | Over-specification or under-specification | Balance between expressiveness and verifiability |

### When to Use VeriGuard

- Autonomous agents in safety-critical domains
- Systems requiring formal guarantees of behavior
- Deployment without continuous human oversight
- Regulatory environments requiring auditable safety

### When NOT to Use VeriGuard

- Soft constraints where violation isn't catastrophic
- Policies requiring complex reasoning beyond formal methods
- Real-time systems where verification latency is prohibitive
- Rapidly changing requirement environments

### Common Pitfalls

- **Spec precision**: Underspecified requirements fail verification; be explicit
- **Completeness**: Policies may miss corner cases; use broad test coverage
- **Maintenance**: Verified policies require re-verification on changes; document updates
- **Overhead**: Verification can be slow; run offline, cache results

## Reference

Paper: https://arxiv.org/abs/2510.05156
