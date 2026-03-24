---
name: agent-data-protocol
title: "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24702"
keywords: [Data Format, Fine-tuning, Agent Training, Standardization, Interoperability]
description: "Standardizes agent training data representation across diverse sources (API use, web browsing, coding, software engineering). Single lightweight protocol unifies 13 datasets enabling 20% performance gains without domain-specific tuning. Enables reproducible agent training and scalable data combination."
---

# Agent Data Protocol: Unified Training Data Format

Agent training suffers from dataset fragmentation: different sources use incompatible formats (ReAct, OpenAI, custom). ADP provides lightweight, expressive standard format that unifies diverse datasets into single training format.

A single model trained on unified data outperforms task-specific variants, enabling scalable agent training.

## Core Concept

Key innovation: **single interlingua that captures diverse agent interactions**:
- API calls, tool use, planning, execution
- Web browsing, DOM interaction, form filling
- Code generation, execution, debugging
- Generic enough for new domains without redesign

Format is simple to parse and extensible for new agent capabilities.

## Architecture Overview

- Minimal schema capturing agent interactions
- Tool/API invocation representation
- Observation and action specification
- Task metadata and success signals

## Implementation Steps

Define the core ADP schema. Keep it minimal to reduce parsing overhead:

```python
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ADPAction:
    """Unified agent action representation."""
    type: str  # "tool", "code", "text", "navigate"
    tool_name: str = None  # e.g., "search", "click", "execute"
    parameters: Dict[str, Any] = None  # tool parameters
    code: str = None  # if type=="code"
    text: str = None  # if type=="text"

@dataclass
class ADPObservation:
    """Result from agent action."""
    status: str  # "success", "error", "timeout"
    output: str  # tool/code output
    metadata: Dict[str, Any] = None  # additional context

@dataclass
class ADPStep:
    """Single agent interaction step."""
    action: ADPAction
    observation: ADPObservation
    reward: float = 0.0  # optional reward signal

@dataclass
class ADPTraj:
    """Complete agent trajectory/episode."""
    task_instruction: str
    steps: List[ADPStep]
    final_success: bool
    domain: str  # "api", "web", "code", "robotics"
```

Implement converters from existing formats to ADP. This is the key to unifying data:

```python
class FormatConverter:
    """Convert various agent dataset formats to ADP."""

    @staticmethod
    def convert_react_format(react_trajectory):
        """Convert ReAct format (Thought/Action/Observation) to ADP."""
        adp_steps = []

        for step in react_trajectory['steps']:
            # Parse ReAct action format
            action_text = step['action']
            action_type, tool, params = FormatConverter._parse_react_action(action_text)

            action = ADPAction(
                type='tool',
                tool_name=tool,
                parameters=params
            )

            observation = ADPObservation(
                status='success' if step['observation'] else 'error',
                output=step['observation']
            )

            adp_steps.append(ADPStep(action=action, observation=observation))

        return ADPTraj(
            task_instruction=react_trajectory['task'],
            steps=adp_steps,
            final_success=react_trajectory.get('success', False),
            domain=react_trajectory.get('domain', 'unknown')
        )

    @staticmethod
    def convert_web_dataset(web_trajectory):
        """Convert web navigation format to ADP."""
        adp_steps = []

        for web_action in web_trajectory['actions']:
            # Parse web action (click, type, navigate, etc)
            action = ADPAction(
                type='navigate',
                tool_name=web_action['action_type'],
                parameters={'target': web_action.get('target')}
            )

            observation = ADPObservation(
                status='success',
                output=web_action['resulting_page_html'][:500]  # Truncate
            )

            adp_steps.append(ADPStep(action=action, observation=observation))

        return ADPTraj(
            task_instruction=web_trajectory['goal'],
            steps=adp_steps,
            final_success=web_trajectory.get('completed', False),
            domain='web'
        )

    @staticmethod
    def convert_code_dataset(code_trajectory):
        """Convert code execution format to ADP."""
        adp_steps = []

        for code_step in code_trajectory['execution_trace']:
            action = ADPAction(
                type='code',
                code=code_step['code']
            )

            observation = ADPObservation(
                status='success' if code_step['exit_code'] == 0 else 'error',
                output=code_step['stdout'] + code_step['stderr']
            )

            adp_steps.append(ADPStep(action=action, observation=observation))

        return ADPTraj(
            task_instruction=code_trajectory['task'],
            steps=adp_steps,
            final_success=code_trajectory.get('solved', False),
            domain='code'
        )

    @staticmethod
    def _parse_react_action(action_text):
        """Parse ReAct format: Action: toolname(params)."""
        import re
        match = re.match(r'Action:\s*(\w+)\((.*)\)', action_text)
        if match:
            tool = match.group(1)
            params_str = match.group(2)
            # Simple param parsing
            params = {'raw': params_str}
            return 'tool', tool, params
        return 'text', None, {}
```

Implement training on unified ADP format. Models trained on combined data:

```python
def train_unified_agent(model, adp_trajectories, num_epochs=5):
    """Train agent model on unified ADP format."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_trajs in adp_trajectories:
            batch_loss = 0

            for traj in batch_trajs:
                # Encode task instruction
                task_emb = model.encode_instruction(traj.task_instruction)

                for step in traj.steps:
                    # Predict action given task and history
                    action_logits = model.predict_action(
                        task_emb,
                        step.action.type,
                        step.action.tool_name
                    )

                    # Loss for predicting action
                    loss = torch.nn.functional.cross_entropy(
                        action_logits,
                        model.encode_action(step.action)
                    )

                    batch_loss += loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

## Practical Guidance

| Aspect | Recommendation |
|--------|-----------------|
| Format simplicity | Minimize fields (easier parsing) |
| Extensibility | Add domain-specific metadata without breaking compatibility |
| Data proportion | Balance domains (avoid one dominating) |
| Missing fields | Use sensible defaults or None |

**When to use:**
- Multi-source agent training projects
- Sharing agent training data across teams
- Building general-purpose agents from diverse domains
- Reproducible research on agent training data

**When NOT to use:**
- Single dataset with specialized format (conversion overhead)
- Real-time data ingestion (parsing latency)
- Proprietary formats where standardization isn't possible

**Common pitfalls:**
- Schema too complex (defeats standardization purpose)
- Lossy conversion (domain-specific information discarded)
- Imbalanced dataset combination (one domain dominates)
- Not validating converted data (quality issues propagate)

Reference: [Agent Data Protocol on arXiv](https://arxiv.org/abs/2510.24702)
