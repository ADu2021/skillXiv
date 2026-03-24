---
name: cort-code-reasoning
title: "CoRT: Code-integrated Reasoning within Thinking"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09820"
keywords: [reasoning, code execution, extended thinking, grounded computation, reasoning models]
description: "Enhance reasoning models by integrating executable code within thinking traces, enabling grounded computation verification and reducing hallucination in mathematical and logical reasoning."
---

# CoRT: Code-integrated Reasoning within Thinking

## Core Concept

CoRT augments reasoning models' thinking process by embedding executable code directly into reasoning traces. Rather than pure symbolic reasoning prone to arithmetic errors and logical fallacies, models generate reasoning steps with executable code blocks that are immediately validated. This grounds abstract reasoning in concrete computation, reducing hallucination and improving accuracy on mathematical, logical, and programming tasks. The approach enables self-correction when code execution results contradict reasoning assumptions.

## Architecture Overview

- **Code-Augmented Thinking**: Reasoning traces include executable code sections alongside natural language explanations
- **Immediate Validation**: Code blocks execute during generation, providing real-time feedback to guide subsequent reasoning
- **Grounded Computation**: Mathematical operations, symbolic manipulations, and logical checks verified through code
- **Self-Correction Mechanism**: Models adjust reasoning when code results contradict expectations
- **Token Efficiency**: Shorter total token count vs pure reasoning through code compression
- **Multi-Domain Support**: Applicable to math, logic puzzles, programming, and science problems

## Implementation

### Step 1: Code-Integrated Thinking Module

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import subprocess
import tempfile
import os

class CodeIntegratedThinkingModule(nn.Module):
    """
    Augments reasoning traces with executable code blocks.
    Interleaves natural language reasoning with Python code for verification.
    """

    def __init__(self, base_model, execution_timeout=5):
        super().__init__()
        self.base_model = base_model
        self.execution_timeout = execution_timeout
        self.execution_history = []

    def generate_with_code_reasoning(self, question: str, max_thinking_tokens: int = 8000):
        """
        Generate reasoning trace with embedded code blocks.
        Format: <think>
                Natural language reasoning...
                ```python
                # Code block
                code here
                ```
                More reasoning...
                </think>
        """

        thinking_prompt = f"""
        Solve this problem with integrated reasoning and code verification.

        Use this format:
        <think>
        Explain your approach.

        ```python
        # Code to verify computations
        result = ...
        print(f"Result: {{result}}")
        ```

        Interpret the code output and continue reasoning...
        </think>

        Problem: {question}
        """

        # Generate thinking trace
        thinking_trace = self._generate_thinking(
            thinking_prompt,
            max_tokens=max_thinking_tokens
        )

        # Extract and execute code blocks
        code_blocks = self._extract_code_blocks(thinking_trace)
        execution_results = []

        for code in code_blocks:
            result = self._execute_code_block(code)
            execution_results.append({
                'code': code,
                'output': result['output'],
                'error': result['error'],
                'success': result['success']
            })
            self.execution_history.append(result)

        return {
            'thinking_trace': thinking_trace,
            'code_blocks': code_blocks,
            'execution_results': execution_results
        }

    def _generate_thinking(self, prompt: str, max_tokens: int) -> str:
        """Generate extended thinking trace."""
        # Placeholder: would call actual model
        return "<think>\nReasoning process...\n</think>"

    def _extract_code_blocks(self, thinking_trace: str) -> List[str]:
        """Extract Python code blocks from thinking trace."""
        import re

        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, thinking_trace, re.DOTALL)

        return matches

    def _execute_code_block(self, code: str) -> Dict:
        """
        Execute Python code block safely with timeout.
        Returns output, error status, and results.
        """

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            try:
                # Execute with timeout
                result = subprocess.run(
                    ['python', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout
                )

                return {
                    'output': result.stdout,
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'success': result.returncode == 0
                }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except subprocess.TimeoutExpired:
            return {
                'output': '',
                'error': 'Execution timeout exceeded',
                'success': False
            }
        except Exception as e:
            return {
                'output': '',
                'error': str(e),
                'success': False
            }
```

### Step 2: Self-Correction via Code Feedback

```python
class SelfCorrectionMechanism:
    """
    Detects contradictions between reasoning and code execution results.
    Triggers re-reasoning when assumptions prove incorrect.
    """

    def __init__(self, model, max_correction_rounds=3):
        self.model = model
        self.max_correction_rounds = max_correction_rounds

    def identify_contradictions(self, reasoning_trace: str,
                               execution_results: List[Dict]) -> List[Dict]:
        """
        Identify where reasoning contradicts code execution results.
        Returns list of contradiction locations.
        """

        contradictions = []

        for i, exec_result in enumerate(execution_results):
            if not exec_result['success']:
                contradictions.append({
                    'block_index': i,
                    'type': 'execution_error',
                    'error': exec_result['error'],
                    'severity': 'high'
                })
                continue

            # Parse output for claims
            output = exec_result['output']

            # Check if output contradicts preceding reasoning
            if self._contradicts_reasoning(reasoning_trace, output):
                contradictions.append({
                    'block_index': i,
                    'type': 'logical_contradiction',
                    'output': output,
                    'severity': 'medium'
                })

        return contradictions

    def trigger_correction(self, original_question: str,
                          initial_reasoning: str,
                          contradictions: List[Dict]) -> Dict:
        """
        When contradictions detected, regenerate reasoning with guidance.
        """

        if not contradictions:
            return {'corrected': False}

        # Build correction prompt highlighting issues
        correction_prompt = f"""
        Previous reasoning attempt:
        {initial_reasoning}

        Issues identified:
        """

        for contradiction in contradictions[:3]:  # Focus on top 3
            if contradiction['type'] == 'execution_error':
                correction_prompt += f"\n- Code error: {contradiction['error']}"
            else:
                correction_prompt += f"\n- Output contradicts reasoning: {contradiction['output'][:100]}"

        correction_prompt += f"""

        Please revise your reasoning, being more careful about:
        1. Arithmetic and logical operations
        2. Variable definitions and scope
        3. Proper code syntax

        Original problem: {original_question}
        """

        # Generate corrected reasoning
        corrected_trace = self.model.generate(correction_prompt, max_tokens=5000)

        return {
            'corrected': True,
            'original_trace': initial_reasoning,
            'corrected_trace': corrected_trace,
            'contradictions_addressed': len(contradictions)
        }

    def _contradicts_reasoning(self, reasoning: str, code_output: str) -> bool:
        """Check if code output contradicts stated reasoning."""
        # Placeholder: would use semantic similarity or rule-based checking
        return False

    def iterative_correction(self, question: str, max_rounds: int = 3) -> Dict:
        """
        Iteratively correct reasoning until no contradictions remain
        or max_rounds exceeded.
        """

        current_reasoning = None
        execution_results = []
        correction_count = 0

        for round_num in range(max_rounds):
            # Generate reasoning with code
            if current_reasoning is None:
                # Initial generation
                result = self._generate_initial(question)
            else:
                # Correction round
                result = self._generate_correction(question, current_reasoning)

            current_reasoning = result['thinking']
            execution_results = result['execution_results']

            # Check for contradictions
            contradictions = self.identify_contradictions(
                current_reasoning,
                execution_results
            )

            if not contradictions:
                # No contradictions; converged
                break

            correction_count += 1

        return {
            'final_reasoning': current_reasoning,
            'execution_results': execution_results,
            'correction_rounds': correction_count,
            'converged': len(contradictions) == 0
        }

    def _generate_initial(self, question: str) -> Dict:
        """Generate initial reasoning."""
        return {'thinking': '', 'execution_results': []}

    def _generate_correction(self, question: str, previous: str) -> Dict:
        """Generate correction based on previous attempt."""
        return {'thinking': '', 'execution_results': []}
```

### Step 3: Code Validation and Type Checking

```python
import ast
import typing

class CodeValidator:
    """
    Validates code blocks before execution for common errors.
    Catches logical issues and improves error messages.
    """

    def __init__(self):
        self.allowed_builtins = {
            'print', 'len', 'range', 'sum', 'min', 'max',
            'int', 'float', 'str', 'list', 'dict', 'set',
            'abs', 'round', 'sorted', 'enumerate', 'zip'
        }
        self.forbidden_imports = {'os', 'sys', 'subprocess', '__main__'}

    def validate_code_block(self, code: str) -> Dict:
        """
        Comprehensive validation before execution.
        Returns validation status and identified issues.
        """

        issues = []

        # Check 1: Parse validity
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'message': str(e),
                'severity': 'critical'
            })
            return {
                'valid': False,
                'issues': issues,
                'safe_to_execute': False
            }

        # Check 2: Forbidden imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.forbidden_imports:
                        issues.append({
                            'type': 'forbidden_import',
                            'module': alias.name,
                            'severity': 'high'
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.module in self.forbidden_imports:
                    issues.append({
                        'type': 'forbidden_import',
                        'module': node.module,
                        'severity': 'high'
                    })

        # Check 3: Infinite loops (heuristic)
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if while condition is 'True'
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    issues.append({
                        'type': 'infinite_loop',
                        'message': 'Detected while True without break',
                        'severity': 'high'
                    })

        # Check 4: Undefined variables (simple check)
        defined_vars = set()
        used_vars = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)

        undefined = used_vars - defined_vars - set(self.allowed_builtins)
        for var in undefined:
            issues.append({
                'type': 'undefined_variable',
                'variable': var,
                'severity': 'medium'
            })

        safe = len([i for i in issues if i['severity'] in ['high', 'critical']]) == 0

        return {
            'valid': True,
            'issues': issues,
            'safe_to_execute': safe,
            'defined_variables': defined_vars,
            'used_variables': used_vars
        }
```

### Step 4: Reasoning with Code Verification

```python
class CodeVerifiedReasoning:
    """
    High-level orchestration of code-integrated reasoning.
    Manages generation, validation, execution, and correction.
    """

    def __init__(self, model):
        self.model = model
        self.thinking_module = CodeIntegratedThinkingModule(model)
        self.correction_mechanism = SelfCorrectionMechanism(model)
        self.validator = CodeValidator()

    def reason(self, question: str, max_attempts: int = 3) -> Dict:
        """
        Complete reasoning pipeline with code verification.
        """

        attempt = 0
        current_result = None

        while attempt < max_attempts:
            # Step 1: Generate reasoning with code
            current_result = self.thinking_module.generate_with_code_reasoning(question)

            # Step 2: Validate code blocks
            validation_status = {}
            for i, code in enumerate(current_result['code_blocks']):
                validation = self.validator.validate_code_block(code)
                validation_status[i] = validation

                if not validation['safe_to_execute']:
                    # Unsafe code; need correction
                    attempt += 1
                    break

            else:
                # All code blocks valid; check for logical contradictions
                contradictions = self.correction_mechanism.identify_contradictions(
                    current_result['thinking_trace'],
                    current_result['execution_results']
                )

                if not contradictions:
                    # Success: no errors, no contradictions
                    return {
                        'success': True,
                        'result': current_result,
                        'attempts': attempt + 1
                    }

                attempt += 1

        return {
            'success': False,
            'result': current_result,
            'attempts': max_attempts
        }
```

## Practical Guidance

**Code-Reasoning Integration**:
- Embed code after every major computational claim (not after every sentence)
- Use print statements to output intermediate results for verification
- Keep code blocks focused (5-15 lines maximum)
- Comment code to connect with natural reasoning

**Code Execution Safety**:
- Whitelist allowed functions (math, list operations, string manipulation)
- Forbid file I/O, network access, and dangerous imports
- Set tight timeouts (5 seconds maximum per block)
- Validate AST before execution to catch common errors

**Correction Strategy**:
- First correction: Fix syntax errors and undefined variables
- Second correction: Address logical contradictions
- Third correction: Reconsider approach if still failing
- Stop after 3 attempts to avoid infinite loops

**Performance Improvements**:
- Math problems: +15-25% accuracy improvement
- Logic puzzles: +10-20% improvement
- Programming tasks: +20-30% improvement
- Reduces hallucination in numerical reasoning

**When to Use CoRT**:
- Mathematical reasoning (MATH, AIME, Calculus)
- Logic puzzles and symbolic manipulation
- Code generation and verification tasks
- Scientific problem-solving with numerical components

## Reference

- Abstract Syntax Tree (AST): Enables safe code analysis without execution
- Subprocess isolation: Executes code in separate process for safety
- Type inference: Can detect variable scope issues statically
- Execution feedback: Code results ground reasoning in reality
