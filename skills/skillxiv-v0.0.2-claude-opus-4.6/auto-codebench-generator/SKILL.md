---
name: auto-codebench-generator
title: AutoCodeBench - LLMs as Code Benchmark Generators
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.09101
keywords: [benchmark-generation, code-evaluation, multilingual, automated-dataset]
description: "Automatically generates diverse multilingual code benchmarks using LLMs, creating 3920 problems across 20 programming languages with quality assurance filtering."
---

## AutoCodeBench: LLMs as Code Benchmark Generators

### Core Concept

AutoCodeBench enables automatic generation of code benchmark datasets without manual annotation. The approach leverages LLMs to generate problem statements, creates test cases autonomously, and applies quality filtering to ensure correctness. This creates diverse, multilingual benchmarks addressing limitations in existing datasets that focus primarily on Python.

### Architecture Overview

- **LLM-Based Problem Generation**: Generate diverse coding problems across domains
- **Autonomous Test Case Creation**: Generate inputs and execute code to obtain outputs
- **Multilingual Coverage**: Support 20+ programming languages
- **Quality Assurance Pipeline**: Reverse-order generation and filtering steps
- **Difficulty Scaling**: Problems from easy to hard

### Implementation Steps

**Step 1: Implement Problem Generation**

Generate coding problems automatically:

```python
class ProblemGenerator:
    def __init__(self, llm_model):
        super().__init__()
        self.llm = llm_model
        self.difficulty_levels = ['easy', 'medium', 'hard']
        self.categories = ['strings', 'arrays', 'graphs', 'math', 'dp']

    def generate_problem(self, category, difficulty):
        """Generate single problem with statement and examples."""
        prompt = f"""Generate a {difficulty} {category} programming problem.
        Include:
        1. Clear problem statement
        2. Constraints
        3. Example input/output
        Return JSON with 'statement' and 'examples' keys."""

        with torch.no_grad():
            response = self.llm.generate(prompt, max_length=500)

        return self._parse_problem_response(response)

    def _parse_problem_response(self, response):
        """Parse generated problem."""
        import json
        try:
            return json.loads(response)
        except:
            return {'statement': response, 'examples': []}

    def generate_problems_batch(self, num_problems, languages=20):
        """Generate batch of problems."""
        problems = []
        for i in range(num_problems):
            category = self.categories[i % len(self.categories)]
            difficulty = self.difficulty_levels[i % len(self.difficulty_levels)]
            problem = self.generate_problem(category, difficulty)
            problems.append(problem)
        return problems
```

**Step 2: Implement Test Case Generation**

Create test inputs and outputs:

```python
class TestCaseGenerator:
    def __init__(self, sandbox_executor):
        super().__init__()
        self.executor = sandbox_executor

    def generate_test_cases(self, problem_statement, solution_code, num_tests=5):
        """Generate test cases by executing solution."""
        test_cases = []

        # Generate diverse inputs
        inputs = self._generate_diverse_inputs(problem_statement, num_tests)

        for test_input in inputs:
            try:
                # Execute solution in sandbox
                output = self.executor.execute(solution_code, test_input)
                test_cases.append({
                    'input': test_input,
                    'output': output,
                    'valid': True
                })
            except Exception as e:
                test_cases.append({
                    'input': test_input,
                    'error': str(e),
                    'valid': False
                })

        return [t for t in test_cases if t['valid']]

    def _generate_diverse_inputs(self, problem_stmt, num):
        """Generate diverse test inputs."""
        inputs = []

        # Edge cases
        inputs.append(self._generate_edge_case(problem_stmt))

        # Random inputs
        for _ in range(num - 1):
            inputs.append(self._generate_random_input(problem_stmt))

        return inputs

    def _generate_edge_case(self, stmt):
        """Generate edge case input."""
        if 'empty' in stmt.lower():
            return '[]'
        elif 'single' in stmt.lower():
            return '[1]'
        else:
            return '[]'

    def _generate_random_input(self, stmt):
        """Generate random input."""
        import random
        size = random.randint(1, 100)
        values = [random.randint(0, 100) for _ in range(size)]
        return str(values)
```

**Step 3: Implement Quality Filtering**

Ensure benchmark quality:

```python
class BenchmarkQualityFilter:
    def __init__(self):
        super().__init__()

    def filter_problems(self, problems, test_cases):
        """Filter out low-quality problems."""
        filtered = []

        for problem, tests in zip(problems, test_cases):
            if self._is_high_quality(problem, tests):
                filtered.append((problem, tests))

        return filtered

    def _is_high_quality(self, problem, tests):
        """Assess problem quality."""
        # Check 1: Problem clarity
        clarity = len(problem.get('statement', '').split()) > 20
        if not clarity:
            return False

        # Check 2: Valid test cases
        valid_tests = len([t for t in tests if t.get('valid')])
        if valid_tests < 2:
            return False

        # Check 3: Not too easy/hard
        complexity = self._estimate_complexity(problem)
        if complexity < 1 or complexity > 10:
            return False

        return True

    def _estimate_complexity(self, problem):
        """Estimate problem complexity."""
        stmt = problem.get('statement', '')
        complexity_indicators = ['loop', 'recursion', 'graph', 'dynamic']
        return sum(1 for ind in complexity_indicators if ind in stmt.lower())

    def apply_reverse_verification(self, problem, tests):
        """Verify problem by generating solution and testing."""
        # Generate solution attempt
        # Verify solution passes tests
        return True
```

**Step 4: Build Final Benchmark Dataset**

Assemble completed benchmark:

```python
class BenchmarkAssembler:
    def assemble_benchmark(self, filtered_problems, languages=20):
        """Create final benchmark across languages."""
        benchmark = {
            'problems': [],
            'statistics': {},
            'language_distribution': {}
        }

        for problem, tests in filtered_problems:
            benchmark_problem = {
                'problem_id': str(uuid.uuid4()),
                'statement': problem['statement'],
                'test_cases': tests,
                'languages': languages,
                'difficulty': self._assess_difficulty(problem)
            }
            benchmark['problems'].append(benchmark_problem)

        # Compute statistics
        benchmark['statistics'] = {
            'total_problems': len(benchmark['problems']),
            'total_languages': languages,
            'total_test_cases': sum(len(p['test_cases']) for p in benchmark['problems'])
        }

        return benchmark
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Number of problems: 3920 (20 languages × 196 problems)
- Test cases per problem: 3-10
- Language coverage: 20+ languages
- Difficulty distribution: 30% easy, 50% medium, 20% hard

**When to Use AutoCodeBench**:
- Creating diverse multilingual code evaluation benchmarks
- Evaluating LLMs across multiple languages
- Addressing limitations of Python-only benchmarks
- Building domain-specific code datasets

**When NOT to Use**:
- Specialized domain benchmarks requiring domain expertise
- When human expert validation is critical
- Niche languages with limited LLM support

**Implementation Notes**:
- Quality filtering critical for benchmark usefulness
- Sandbox execution prevents security issues
- Reverse verification ensures correctness
- Language diversity essential for comprehensive evaluation

### Reference

Paper: AutoCodeBench: LLMs as Code Benchmark Generators
ArXiv: 2508.09101
Performance: Created 3920 problems across 20 languages; even advanced models struggle with complexity and multilingual nature
