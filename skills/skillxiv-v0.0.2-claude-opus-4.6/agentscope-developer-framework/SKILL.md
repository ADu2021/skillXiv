---
name: agentscope-developer-framework
title: "AgentScope 1.0: Developer-Centric Framework for Agentic Applications"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.16279
keywords: [agent-framework, developer-tools, async-design, react-paradigm, agentic-applications]
description: "Build agentic applications using unified agent interfaces, asynchronous design patterns, ReAct paradigm grounding, and developer-centric evaluation and deployment tools."
---

# AgentScope 1.0: Developer-Centric Framework

## Core Concept

AgentScope 1.0 provides a comprehensive framework for building production-ready agentic applications. It features unified component architecture for easy model/tool integration, asynchronous design for efficient multi-agent systems, ReAct paradigm grounding combining reasoning and action, built-in agents for common tasks, visual evaluation interfaces, and runtime sandboxes for safe deployment.

## Architecture Overview

- **Unified Component Interfaces**: Extensible abstractions for models, tools, memory
- **Asynchronous Design**: Event-driven architecture supporting diverse interaction patterns
- **ReAct Paradigm**: Structured reasoning and action loops
- **Built-in Agents**: Pre-configured solutions for common scenarios
- **Developer Tools**: Visualization, evaluation, and sandbox execution

## Implementation Steps

### 1. Implement Core Component Abstraction

Create unified interfaces for models and tools:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str

class ModelInterface(ABC):
    """Abstract base for LLM integration."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        pass

class Tool(ABC):
    """Abstract base for agent tools."""

    @abstractmethod
    async def execute(self, input_str: str, **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass

class OpenAIModel(ModelInterface):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

class Calculator(Tool):
    async def execute(self, expression: str, **kwargs) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"expression": {"type": "string", "description": "Math expression"}}
```

### 2. Implement ReAct Agent Loop

Structure agents around reasoning and acting:

```python
from enum import Enum

class ActionType(Enum):
    THINK = "think"
    ACT = "act"
    CONCLUDE = "conclude"

@dataclass
class ReActTrace:
    thoughts: List[str]
    actions: List[Dict[str, str]]
    observations: List[str]
    final_answer: str
    success: bool

class ReActAgent:
    def __init__(
        self,
        model: ModelInterface,
        tools: Dict[str, Tool],
        max_steps: int = 10
    ):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.trace: Optional[ReActTrace] = None

    async def run(self, task: str) -> ReActTrace:
        """Execute ReAct loop."""
        self.trace = ReActTrace([], [], [], "", False)
        messages = [Message("user", task)]

        for step in range(self.max_steps):
            # Reasoning phase
            thought = await self._think(messages, task)
            self.trace.thoughts.append(thought)
            messages.append(Message("assistant", thought))

            # Check if should act or conclude
            if "use tool:" in thought.lower() or "call:" in thought.lower():
                # Acting phase
                action_str = self._extract_action(thought)
                tool_name, tool_input = self._parse_action(action_str)

                if tool_name == "conclude":
                    self.trace.final_answer = tool_input
                    self.trace.success = True
                    break

                # Execute tool
                if tool_name in self.tools:
                    observation = await self.tools[tool_name].execute(tool_input)
                    self.trace.actions.append({"tool": tool_name, "input": tool_input})
                    self.trace.observations.append(observation)
                    messages.append(Message("user", f"Tool result: {observation}"))

        return self.trace

    async def _think(self, messages: List[Message], task: str) -> str:
        """Generate reasoning step."""
        system_msg = Message("system",
            "You are a reasoning agent. Think step by step. "
            "When ready, use tools by saying 'use tool: <tool_name>(<input>)' "
            "or conclude by saying 'conclude: <answer>'")

        full_messages = [system_msg] + messages
        return await self.model.generate(full_messages)

    def _extract_action(self, thought: str) -> str:
        """Extract action specification from thought."""
        import re
        match = re.search(r'(?:use tool:|call:)\s*(.+?)(?:\n|$)', thought, re.IGNORECASE)
        return match.group(1) if match else ""

    def _parse_action(self, action_str: str) -> tuple:
        """Parse action into tool name and input."""
        import re
        match = re.match(r'(\w+)\s*\((.+)\)', action_str)
        if match:
            return match.group(1), match.group(2)
        return action_str, ""
```

### 3. Implement Asynchronous Multi-Agent Coordination

Enable concurrent agent interactions:

```python
class AgentPool:
    """Manages multiple agents with async execution."""

    def __init__(self):
        self.agents: Dict[str, ReActAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

    def register_agent(self, name: str, agent: ReActAgent):
        """Register agent in pool."""
        self.agents[name] = agent

    async def execute_task(
        self,
        task: str,
        primary_agent: str,
        parallel_agents: Optional[List[str]] = None
    ) -> Dict[str, ReActTrace]:
        """
        Execute task with primary agent and optional parallel agents.
        """
        results = {}

        # Primary agent
        if primary_agent in self.agents:
            results[primary_agent] = await self.agents[primary_agent].run(task)

        # Parallel agents
        if parallel_agents:
            tasks = [
                self.agents[agent].run(task)
                for agent in parallel_agents
                if agent in self.agents
            ]
            parallel_results = await asyncio.gather(*tasks)
            for agent_name, trace in zip(parallel_agents, parallel_results):
                results[agent_name] = trace

        return results

    async def agent_collaboration(
        self,
        agents: List[str],
        task: str,
        max_rounds: int = 3
    ) -> Dict[str, Any]:
        """
        Multi-round collaboration between agents.
        """
        messages = {agent: [Message("user", task)] for agent in agents}

        for round_idx in range(max_rounds):
            # All agents think and share results
            round_results = {}

            for agent_name in agents:
                agent = self.agents[agent_name]
                thought = await agent._think(messages[agent_name], task)
                messages[agent_name].append(Message("assistant", thought))
                round_results[agent_name] = thought

            # Share thoughts across agents
            for agent_name in agents:
                for other_agent in agents:
                    if other_agent != agent_name:
                        messages[agent_name].append(
                            Message("user", f"Agent {other_agent}: {round_results[other_agent]}")
                        )

        return {agent: messages[agent][-1].content for agent in agents}
```

### 4. Implement Evaluation Interface

Create tools for assessing agent performance:

```python
class AgentEvaluator:
    """Evaluate agent performance on tasks."""

    async def evaluate_on_dataset(
        self,
        agent: ReActAgent,
        dataset: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Run agent on dataset and compute metrics.
        """
        results = []

        for example in dataset:
            trace = await agent.run(example["task"])
            correct = trace.final_answer == example["expected_answer"]
            results.append({
                "correct": correct,
                "steps": len(trace.actions),
                "tools_used": [a["tool"] for a in trace.actions]
            })

        # Compute metrics
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        return {
            "accuracy": accuracy,
            "avg_steps": avg_steps,
            "efficiency": accuracy / (avg_steps + 1)
        }

    def visualize_trace(self, trace: ReActTrace) -> str:
        """Generate visualization of reasoning trace."""
        viz = "ReAct Trace Visualization\n"
        viz += "=" * 50 + "\n"

        for i, (thought, action, obs) in enumerate(zip(
            trace.thoughts,
            trace.actions,
            trace.observations
        )):
            viz += f"\nStep {i+1}:\n"
            viz += f"  Thought: {thought}\n"
            viz += f"  Action: {action}\n"
            viz += f"  Observation: {obs}\n"

        viz += f"\nFinal Answer: {trace.final_answer}\n"
        return viz
```

### 5. Implement Sandbox Execution Environment

Create safe execution context:

```python
import docker
import json

class SandboxExecutor:
    """Execute agents in isolated containers."""

    def __init__(self, image_name: str = "agentscope-runtime"):
        self.client = docker.from_env()
        self.image_name = image_name

    async def run_agent_sandboxed(
        self,
        agent_code: str,
        task: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run agent code in isolated sandbox.
        """
        container_input = {
            "agent_code": agent_code,
            "task": task
        }

        try:
            container = self.client.containers.run(
                self.image_name,
                stdin_open=True,
                stdout=True,
                stderr=True,
                detach=True
            )

            # Send task to container
            container.exec_run(
                f"python /app/agent.py",
                input=json.dumps(container_input).encode()
            )

            # Wait for completion or timeout
            exit_code = container.wait(timeout=timeout)

            # Get output
            logs = container.logs().decode()

            container.remove()

            return {
                "success": exit_code == 0,
                "output": logs,
                "exit_code": exit_code
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exit_code": -1
            }
```

## Practical Guidance

### When to Use AgentScope

- Building production agent applications
- Multi-agent collaboration systems
- Rapid prototyping of agent architectures
- Applications requiring safe sandboxed execution
- Complex workflows mixing reasoning and tools

### When NOT to Use

- Simple single-prompt inference
- Real-time low-latency applications (<100ms)
- Scenarios without clear tool definitions
- Extremely resource-constrained environments

### Key Hyperparameters

- **max_steps**: 5-20 per agent task
- **async_batch_size**: 4-16 parallel agents
- **timeout**: 30-600 seconds based on task complexity
- **temperature**: 0.7 for reasoning, 0.0 for determinism

### Performance Expectations

- Framework Overhead: <100ms per agent initialization
- Concurrent Agents: 10-100s feasible on single machine
- Tool Latency: Dominated by tool, not framework

## Reference

Researchers. (2024). AgentScope 1.0: A Developer-Centric Framework for Building Agentic Applications. arXiv preprint arXiv:2508.16279.
