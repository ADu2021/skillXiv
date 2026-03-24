---
name: terminal-bench-agent-evaluation
title: "Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.11868"
keywords: [agent-benchmark, terminal-tasks, command-line, realistic-evaluation, agent-capability]
description: "Evaluate agents on 89 challenging terminal-based tasks where frontier models score below 65%, providing realistic assessment of command-line interface automation capability. Use when benchmarking agents designed for system administration, automation, or DevOps tasks."
---

# Terminal-Bench: Agent Evaluation on CLI Tasks

This skill provides a comprehensive benchmarking suite for evaluating agents on realistic, hard terminal-based tasks, revealing significant gaps in even frontier models' abilities to automate command-line workflows.

## When to Use
- Evaluating CLI automation agents (DevOps, system administration)
- Measuring agent capability on realistic computer-use tasks
- Identifying gaps in agent reasoning for terminal-based workflows
- Benchmarking improvements in agent autonomy and problem-solving
- Comparing agent performance across different models/architectures

## When NOT to Use
- Tasks not involving command-line interfaces
- Simple command execution (Terminal-Bench is for complex workflows)
- Systems where safety is critical (tests include risky operations)
- Evaluating GUI-based agents (Terminal-Bench is CLI-specific)

## Key Concept
Terminal-Bench 2.0 contains 89 challenging tasks across realistic scenarios:
- File system operations (finding, organizing, searching)
- Process management (monitoring, controlling, debugging)
- Network operations (connectivity, configuration)
- Software installation and management
- Log analysis and troubleshooting

These tasks reflect real-world challenges where even frontier models fail < 65% of the time.

## Benchmark Task Categories

Terminal-Bench covers:
- **File Management**: Complex operations requiring understanding of file types, permissions, search patterns
- **System Monitoring**: Extracting metrics from multiple sources, identifying patterns
- **Configuration**: Managing system settings, understanding configuration formats
- **Troubleshooting**: Diagnosing issues from partial information, using multiple diagnostic tools
- **Scripting**: Creating bash scripts for complex multi-step workflows

## Using the Benchmark

Evaluate an agent on Terminal-Bench tasks:

```python
# Pseudocode for Terminal-Bench evaluation
class TerminalBench:
    def __init__(self, tasks_path="terminal_bench_tasks.json"):
        self.tasks = load_tasks(tasks_path)  # 89 tasks

    def evaluate_agent(self, agent, timeout_per_task=300):
        results = {"passed": 0, "failed": 0, "details": []}

        for task in self.tasks:
            try:
                # Execute task with timeout
                output = agent.execute_terminal_task(task.description)

                # Validate output against expected results
                if self.validate_output(output, task.expected_output):
                    results["passed"] += 1
                    results["details"].append({
                        "task": task.id,
                        "success": True
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "task": task.id,
                        "success": False,
                        "reason": "output validation failed"
                    })
            except TimeoutError:
                results["failed"] += 1

        return results
```

## Key Insights
- Frontier models achieve < 65% success rate on Terminal-Bench tasks
- Clear gap between simple command execution and complex workflows
- Multi-step reasoning and tool composition remain challenging
- Real-world CLI automation is significantly harder than assumed

## Research Context
Terminal-Bench 2.0 is designed to reveal genuine limitations of agent reasoning on realistic computer-use tasks. The high difficulty (most frontier models fail 35%+ of tasks) demonstrates that CLI automation remains an open challenge despite advances in other agent domains.
