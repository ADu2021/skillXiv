---
name: exp-bench-ai-research
title: "EXP-Bench: Can AI Conduct AI Research Experiments?"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24878"
keywords: [AI Research, Benchmarking, Automation, Multi-Step Reasoning, Experimental Design]
description: "Evaluate AI systems' ability to conduct autonomous research experiments using EXP-Bench, a benchmark for multi-step scientific reasoning and iterative experimental workflows."
---

# Benchmark AI Capability to Conduct Autonomous Experiments

EXP-Bench addresses a critical capability gap: while AI systems excel at isolated tasks, they struggle with multi-step experimental workflows—the core of scientific research. The benchmark evaluates whether AI can design experiments, run them, analyze results, and iterate based on findings. This requires reasoning about experimental design, error handling, result interpretation, and iterative refinement.

The key insight is that research experiments are complex workflows involving multiple decision points: What parameters to test? How to interpret unexpected results? When to pivot to a different approach? EXP-Bench measures these higher-order research capabilities beyond single-task performance.

## Core Concept

EXP-Bench evaluates AI research capability through:

- **Experimental design**: Choosing appropriate parameters, baselines, and evaluation metrics
- **Workflow execution**: Running multi-step experimental pipelines reliably
- **Result analysis**: Interpreting experimental outputs and drawing conclusions
- **Iterative refinement**: Adjusting experiments based on results
- **Error handling**: Recovering from failures and debugging
- **Documentation**: Recording experimental setup, results, and conclusions

Success requires reasoning about causal relationships, hypothesis testing, and scientific methodology—not just task completion.

## Architecture Overview

- **Experiment specification language**: Format for describing experiments (parameters, steps, evaluation)
- **Execution engine**: Runs experimental pipelines with error recovery
- **Analysis module**: Interprets results and identifies patterns
- **Decision maker**: Decides on next steps (continue, refine, pivot)
- **Logging system**: Records all experimental details for reproducibility
- **Multi-step reasoning**: Chains decisions across multiple experimental phases
- **Error detection**: Identifies when experiments fail or results are anomalous

## Implementation

Build a framework for autonomous experiment execution and analysis:

```python
# EXP-Bench: Autonomous research experiment capability
import json
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class ExperimentStep:
    """Specification for one step in an experiment"""
    name: str
    command: str  # Shell command or Python code to execute
    parameters: Dict[str, Any]  # Configuration parameters
    expected_output: str  # What we expect to capture
    error_handling: str  # How to handle failures

@dataclass
class ExperimentResult:
    """Results from executing one experiment"""
    step_name: str
    success: bool
    output: str
    metrics: Dict[str, float]
    error: Optional[str] = None
    timestamp: Optional[str] = None

class AutonomousExperimentRunner:
    """
    Execute multi-step experiments autonomously with iterative refinement.
    """
    def __init__(self, max_iterations=5, timeout_per_step=300):
        self.max_iterations = max_iterations
        self.timeout_per_step = timeout_per_step
        self.experiment_history = []

    def design_experiment(self, research_question: str) -> List[ExperimentStep]:
        """
        LLM-based experiment design given research question.
        """
        design_prompt = f"""
You are designing an experiment to answer this research question:
{research_question}

Design an experiment with these steps:
1. Setup: Prepare environment and data
2. Baseline: Run baseline implementation
3. Treatment: Run modified implementation
4. Evaluation: Compute metrics comparing both
5. Analysis: Interpret results

For each step, specify:
- What command/code to run
- Expected output format
- What metrics to extract
- How to handle failures

Return as JSON with steps array.
"""
        # In practice, use LLM to generate experiment
        # For now, return example structure
        return [
            ExperimentStep(
                name="setup",
                command="python setup_data.py",
                parameters={},
                expected_output="data_ready",
                error_handling="retry"
            )
        ]

    def execute_experiment_workflow(self, experiment_steps: List[ExperimentStep],
                                   hypothesis: str) -> Dict[str, Any]:
        """
        Execute full experiment with iterative refinement.
        """
        results = []
        iteration = 0
        success = False

        while iteration < self.max_iterations and not success:
            iteration_results = []

            for step in experiment_steps:
                result = self._execute_step(step)
                iteration_results.append(result)

                # Check for critical failures
                if not result.success and step.error_handling == "fail_fast":
                    break

                # Analyze intermediate results
                analysis = self._analyze_intermediate_results(iteration_results)

                # Decide if we should continue or refine
                if analysis['should_refine']:
                    # Refine experiment based on intermediate results
                    experiment_steps = self._refine_experiment(
                        experiment_steps,
                        analysis['issues'],
                        hypothesis
                    )
                    break  # Start new iteration with refined steps

            results.extend(iteration_results)

            # Final analysis
            final_analysis = self._analyze_complete_results(results, hypothesis)
            success = final_analysis['success']

            if not success and iteration < self.max_iterations - 1:
                # Suggest refinements for next iteration
                refinement = self._generate_refinement(final_analysis, hypothesis)
                experiment_steps = refinement

            iteration += 1

        return {
            'success': success,
            'num_iterations': iteration,
            'results': results,
            'analysis': final_analysis,
            'conclusions': self._generate_conclusions(results, final_analysis)
        }

    def _execute_step(self, step: ExperimentStep) -> ExperimentResult:
        """Execute a single experiment step with error handling"""
        try:
            # Execute command
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                timeout=self.timeout_per_step,
                text=True
            )

            # Parse output
            output = result.stdout + result.stderr
            success = result.returncode == 0

            # Extract metrics from output
            metrics = self._parse_metrics(output, step.expected_output)

            return ExperimentResult(
                step_name=step.name,
                success=success,
                output=output,
                metrics=metrics,
                error=result.stderr if not success else None
            )

        except subprocess.TimeoutExpired:
            return ExperimentResult(
                step_name=step.name,
                success=False,
                output="",
                metrics={},
                error=f"Timeout after {self.timeout_per_step}s"
            )
        except Exception as e:
            return ExperimentResult(
                step_name=step.name,
                success=False,
                output="",
                metrics={},
                error=str(e)
            )

    def _analyze_intermediate_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze results during execution to guide next steps"""
        analysis = {
            'should_refine': False,
            'issues': [],
            'insights': []
        }

        for result in results:
            if not result.success:
                analysis['issues'].append(f"{result.step_name}: {result.error}")

        # Detect anomalies or unexpected patterns
        if len(results) > 1:
            # Check for metric trends
            latest_metrics = results[-1].metrics
            if not latest_metrics:
                analysis['issues'].append("No metrics extracted from output")
                analysis['should_refine'] = True

        return analysis

    def _analyze_complete_results(self, results: List[ExperimentResult],
                                 hypothesis: str) -> Dict[str, Any]:
        """Analyze complete experiment results to draw conclusions"""
        all_metrics = {}
        for result in results:
            all_metrics.update(result.metrics)

        analysis = {
            'success': all(r.success for r in results),
            'metrics': all_metrics,
            'hypothesis_confirmed': None,
            'confidence': 0.0
        }

        # Determine if hypothesis is supported
        if 'accuracy' in all_metrics:
            # Example: simple metric-based conclusion
            analysis['hypothesis_confirmed'] = all_metrics['accuracy'] > 0.8
            analysis['confidence'] = all_metrics['accuracy']

        return analysis

    def _refine_experiment(self, steps: List[ExperimentStep],
                          issues: List[str],
                          hypothesis: str) -> List[ExperimentStep]:
        """
        Generate refined experiment based on identified issues.
        Uses LLM to suggest improvements.
        """
        refinement_prompt = f"""
The experiment encountered these issues: {issues}
Original hypothesis: {hypothesis}

Suggest refinements to the experiment:
1. What parameters should be adjusted?
2. Should we add additional steps?
3. Are there alternative approaches to test?

Return refined experiment steps as JSON.
"""
        # In practice, use LLM
        # For now, return same steps (simplified)
        return steps

    def _parse_metrics(self, output: str, expected_format: str) -> Dict[str, float]:
        """Extract metrics from experiment output"""
        metrics = {}

        # Look for common metric patterns
        import re
        patterns = {
            'accuracy': r'accuracy[:\s=]+([0-9.]+)',
            'loss': r'loss[:\s=]+([0-9.]+)',
            'f1': r'f1[:\s=]+([0-9.]+)',
            'auc': r'auc[:\s=]+([0-9.]+)'
        }

        for metric_name, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[metric_name] = float(match.group(1))

        return metrics

    def _generate_refinement(self, analysis: Dict[str, Any],
                           hypothesis: str) -> List[ExperimentStep]:
        """Generate refined experiment based on analysis"""
        # Would use LLM to generate refinements
        return []

    def _generate_conclusions(self, results: List[ExperimentResult],
                            analysis: Dict[str, Any]) -> str:
        """Generate written conclusions from experiment"""
        conclusion = f"""
Experiment conducted with {len(results)} steps.
All steps successful: {all(r.success for r in results)}

Key metrics:
{json.dumps(analysis.get('metrics', {}), indent=2)}

Hypothesis confirmed: {analysis.get('hypothesis_confirmed')}
Confidence: {analysis.get('confidence', 0):.2%}

Key findings:
- Experiment completed successfully
- Results support hypothesis: {analysis.get('hypothesis_confirmed')}
"""
        return conclusion
```

Implement an experimental workflow coordinator:

```python
class ResearchWorkflowCoordinator:
    """
    Coordinate multiple related experiments for iterative research.
    """
    def __init__(self, research_goal: str):
        self.research_goal = research_goal
        self.experiments = []
        self.conclusions = []

    def plan_research_direction(self) -> List[str]:
        """
        Generate sequence of experiments to answer research question.
        """
        planning_prompt = f"""
Research Goal: {self.research_goal}

Plan a sequence of {3-5} experiments to systematically investigate this goal.
Each experiment should build on previous findings.

For each experiment specify:
1. Hypothesis being tested
2. Key variables to manipulate
3. Control conditions
4. Success criteria

Return as JSON array with experiments.
"""
        # In practice, use LLM to plan
        return []

    def run_research_cycle(self) -> Dict[str, Any]:
        """Execute planned sequence of experiments"""
        runner = AutonomousExperimentRunner()
        cycle_results = []

        for exp_spec in self.planned_experiments:
            print(f"\nRunning experiment: {exp_spec['name']}")

            # Design experiment
            steps = runner.design_experiment(exp_spec['hypothesis'])

            # Execute
            result = runner.execute_experiment_workflow(steps, exp_spec['hypothesis'])
            cycle_results.append(result)

            # Extract insights
            insights = self._extract_insights(result)

            # Decide on next experiment
            if not result['success']:
                print(f"Experiment failed. Analyzing failure...")
                next_action = self._decide_next_action(result)
                if next_action == 'pivot':
                    print("Pivoting to alternative hypothesis")
                    # Modify research direction

        return {
            'research_goal': self.research_goal,
            'num_experiments': len(cycle_results),
            'results': cycle_results,
            'overall_success': all(r['success'] for r in cycle_results)
        }

    def _extract_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract key learnings from experiment"""
        insights = []
        if result['success']:
            insights.append(f"Hypothesis confirmed")
        else:
            insights.append(f"Need to refine approach")
        return insights

    def _decide_next_action(self, result: Dict[str, Any]) -> str:
        """Decide whether to refine or pivot based on results"""
        if result['success']:
            return 'continue'
        elif len(self.experiments) < 3:
            return 'refine'
        else:
            return 'pivot'
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|----------------|----|
| Max iterations per experiment | 3 - 5 | More allows refinement; raises cost/time |
| Step timeout | 300 - 600 seconds | Prevents hanging; adjust for your domain |
| Metric extraction | Regex + structured output | Parse common metrics from output |
| Error handling | Fail-fast vs retry | Depends on error severity |
| Experiment documentation | JSON + markdown | Enables reproducibility and analysis |

**When to use EXP-Bench approach:**
- Evaluating AI capability on research tasks
- Building autonomous research systems
- Need multi-step experimental reasoning
- Want to benchmark scientific methodology capability
- Developing AI research assistants

**When NOT to use:**
- Single-task performance evaluation (use standard benchmarks)
- Experiments don't have verifiable/parseable outputs
- Iterative refinement isn't needed (pre-determined workflow sufficient)
- Computational budget is extremely limited
- Experiments require human judgment for interpretation

**Common pitfalls:**
- Metrics not properly parsed from outputs (malformed detection)
- Experiments too complex for iterative refinement (define simpler cycles)
- No error recovery mechanism (system fails on first error)
- Iterations don't actually improve results (refinement logic too simple)
- Not tracking experimental history (hard to learn from failures)
- Assuming AI will match human research intuition (it won't, needs supervision)

## Reference

**EXP-Bench: Can AI Conduct AI Research Experiments?**
https://arxiv.org/abs/2505.24878
