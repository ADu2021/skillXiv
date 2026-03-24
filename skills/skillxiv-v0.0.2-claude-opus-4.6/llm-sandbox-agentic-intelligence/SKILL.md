---
name: llm-sandbox-agentic-intelligence
title: "LLM-in-Sandbox Elicits General Agentic Intelligence"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.16206"
keywords: [agentic-intelligence, code-sandbox, reinforcement-learning, general-capability, llm-agent]
description: "Enable language models to explore code sandboxes to solve diverse tasks like mathematics and long-context reasoning without additional training, with reinforcement learning further enhancing capabilities. Use when you need LLMs to perform tasks requiring exploration and code execution in isolated environments."
---

# LLM-in-Sandbox: Eliciting Agentic Intelligence

This skill demonstrates how to enable language models to explore isolated code sandboxes and develop general agentic intelligence across diverse task domains through exploration and optional reinforcement learning.

## When to Use
- Tasks requiring code execution and exploration without modifying the main system
- Building agents that solve math problems, reasoning tasks, or long-context understanding
- Implementing safe environments for LLM experimentation and learning
- Creating agents that don't require additional fine-tuning for diverse tasks

## When NOT to Use
- Tasks requiring interaction with external APIs or network resources
- Real-time systems where sandbox overhead is problematic
- Simple inference tasks (overhead of exploration not justified)
- Systems that need persistent state across sandbox sessions

## Key Concept
LLM-in-Sandbox gives language models the ability to freely explore isolated code execution environments. The model can write and execute code, observe results, and iterate—similar to how humans solve problems through trial and error.

The system provides:
1. **Sandbox Access**: Safe code execution environment
2. **Exploration Freedom**: LLM can write and run arbitrary code
3. **Feedback Loop**: Results guide further exploration
4. **Reinforcement Learning**: Optional training on successful trajectories

## Implementation Pattern

Create a sandbox environment where the LLM can explore and execute code:

```python
# Pseudocode for LLM-Sandbox system
class LLMSandbox:
    def __init__(self, llm_model, sandbox_executor):
        self.llm = llm_model
        self.executor = sandbox_executor
        self.context_window = []

    def solve_task(self, task_description):
        messages = [{"role": "user", "content": task_description}]

        for step in range(max_steps):
            # LLM decides next action (code to execute or answer)
            response = self.llm.generate(messages)

            if "EXECUTE_CODE:" in response:
                code = response.split("EXECUTE_CODE:")[1]
                result = self.executor.run(code)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Execution result: {result}"})
            else:
                return response

        return "Max steps exceeded"
```

The LLM learns to write exploratory code, observe execution results, and refine its approach without modification to the base model.

## Key Results
- Solves diverse tasks: mathematics, algorithms, reasoning, long-context understanding
- No additional training required—sandbox exploration alone elicits capability
- Reinforcement learning can further enhance performance
- Demonstrates emergence of general agentic behavior through freedom to explore

## Research Context
This work shows that simply giving LLMs access to execution environments with feedback unlocks agentic reasoning capabilities. The key insight: exploration in sandboxes naturally develops agency and problem-solving without explicit training.
