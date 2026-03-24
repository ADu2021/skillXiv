---
name: verltool-agentic-rl-tool-use
title: "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.01055"
keywords: [reinforcement learning, tool use, agentic AI, multi-turn reasoning, modular framework]
description: "Train agents to leverage external tools across domains using VerlTool's unified RL framework. Coordinate code execution, search, SQL queries, and vision utilities in multi-turn interactions without domain-specific redesign. 2× faster asynchronous rollouts on mathematical reasoning, knowledge QA, and software engineering tasks."
---

## Train Agentic Systems to Solve Complex Tasks with Tools

**Outcome:** Build agents that iteratively reason, call external tools, observe results, and adapt across diverse problem domains using a single unified framework.

### Problem Context

Existing approaches to agentic AI fragment tool use into domain-specific systems. A knowledge QA system handles search differently than a code execution system; SQL agents use different APIs than visual reasoning systems. When researchers want agents to solve multi-domain tasks—or when practitioners need to add new tool capabilities—the cost of integration is high: custom pipelines, reimplemented coordination logic, and repeated infrastructure investment.

Single-turn language models also cannot naturally handle the sequential decision-making required for tool use: acting, observing results from that action, and choosing the next action based on new information. Reinforcement learning (RL) can optimize these multi-turn trajectories, but extending RL frameworks to support diverse tools requires careful trajectory representation, observation tokenization, and reward alignment across modalities.

VerlTool solves this by introducing a unified, modular framework that extends Reinforcement Learning with Verifiable Rewards (RLVR) to multi-turn agentic settings with tool use.

### Core Concept

VerlTool treats tool use in RL as a trajectory of alternating actions and observations: the agent selects an action (tool call with arguments), the tool executes and returns an observation, and this cycle repeats until the task is solved. The key insight is that observations in tool-use trajectories are environment-generated facts outside the agent's control—they should not influence the policy gradient during training, only provide information for subsequent decisions.

The framework is "holistic" because it unifies:
- Multiple tool types (code, search, SQL, vision) under one API
- Multiple modalities (text, images, video) in observation tokens
- Multi-turn RL training with asynchronous execution for efficiency
- Upstream compatibility with VeRL for seamless maintenance

Instead of building separate agents for code execution, search, and SQL, teams develop a single agent that learns to compose any combination of these tools effectively.

### Architecture Overview

VerlTool splits into two primary subsystems that communicate via standardized APIs:

- **VeRL Workflow (Training Side):** Orchestrates policy training, reward computation, and model updates. Inherits VeRL as a submodule to stay aligned with upstream improvements and avoid duplicating core RL logic.

- **Tool Server (Execution Side):** Manages execution of tool calls. Processes rollouts asynchronously on a trajectory-by-trajectory basis instead of enforcing synchronous batch alignment, eliminating idle waiting and achieving approximately 2× speedup during rollout phases.

**Modular Tool Registration:**
Each tool implements a common BaseTool interface with methods for parsing actions, managing environment state, and executing operations. New tools are added by creating lightweight Python definition files without modifying training code. This design separates concerns and enables domain experts to contribute tools without deep RL knowledge.

**Tokenization Strategy:**
A critical detail for stability: action and observation strings are tokenized separately, then concatenated. This prevents boundary-related token mismatches that can cause training instability during multi-turn rollouts. The framework ensures consistent token alignment across all turns.

**Observation Masking in Policy Optimization:**
Since observations are environment-generated and off-policy with respect to the model being trained, the framework masks observation tokens during gradient computation. Only action tokens and the final reward contribute to policy updates. This prevents the model from learning spurious correlations with stale observation data and maintains RL stability.

### Implementation

#### 1. Set Up VerlTool Modular Architecture

Begin by inheriting VeRL and structuring the tool server as a separate component. This ensures training logic remains decoupled from tool execution infrastructure.

```python
# verltool/config.py
# Configuration for dual-component architecture

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from verl.config import BaseConfig

@dataclass
class ToolServerConfig(BaseConfig):
    """Configuration for asynchronous tool execution server."""
    host: str = "localhost"
    port: int = 8888
    max_workers: int = 32  # Async worker threads for tool calls
    timeout_seconds: int = 300
    enable_async_rollout: bool = True  # Critical for 2× speedup

@dataclass
class VerlToolConfig(BaseConfig):
    """Top-level VerlTool configuration."""
    verl_config: Dict = field(default_factory=dict)  # Inherited VeRL settings
    tool_server: ToolServerConfig = field(default_factory=ToolServerConfig)
    modalities: List[str] = field(default_factory=lambda: ["text"])  # text, image, video
    observation_masking_enabled: bool = True  # Mask off-policy observations
    tokenization_mode: str = "separate"  # Separate action/observation tokens
```

#### 2. Implement Unified Tool Interface

Define the BaseTool abstraction that all tools inherit from. This enables dynamic registration and consistent handling of diverse tools.

```python
# verltool/tools/base.py
# Unified tool interface for consistent behavior across domains

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class ToolAction:
    """Structured representation of a tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: Optional[float] = None

@dataclass
class ToolObservation:
    """Result returned by tool execution."""
    content: Union[str, bytes]
    modality: str = "text"  # text, image, video
    token_count: Optional[int] = None
    error: Optional[str] = None

class BaseTool(ABC):
    """Base class for all tools in VerlTool ecosystem."""

    def __init__(self, name: str, modalities: List[str] = None):
        self.name = name
        self.modalities = modalities or ["text"]
        self.state = {}  # Maintains context across turns

    @abstractmethod
    def parse_action(self, action_string: str) -> ToolAction:
        """Convert raw action string to structured ToolAction."""
        pass

    @abstractmethod
    def execute(self, action: ToolAction) -> ToolObservation:
        """Execute the tool and return observation."""
        pass

    def update_state(self, observation: ToolObservation):
        """Update internal state based on execution result."""
        self.state['last_result'] = observation

    def reset(self):
        """Clear state for new trajectory."""
        self.state = {}
```

#### 3. Register Tools Dynamically

Create a registry that discovers and manages tools without hardcoding them into training logic.

```python
# verltool/tools/registry.py
# Dynamic tool registration system for extensibility

from typing import Dict, Type, Optional
from verltool.tools.base import BaseTool
import importlib
import os

class ToolRegistry:
    """Registry for tool discovery and management."""

    def __init__(self, tool_dir: str = "verltool/tools"):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_dir = tool_dir

    def register(self, tool_class: Type[BaseTool]):
        """Explicitly register a tool class."""
        instance = tool_class()
        self.tools[instance.name] = instance
        return self

    def discover_from_directory(self):
        """Auto-discover tools from tool_dir by importing modules."""
        for filename in os.listdir(self.tool_dir):
            if filename.endswith('_tool.py') and not filename.startswith('_'):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f'verltool.tools.{module_name}')
                    # Assume each module exports TOOL_CLASS
                    if hasattr(module, 'TOOL_CLASS'):
                        self.register(module.TOOL_CLASS)
                except Exception as e:
                    print(f"Warning: Could not load tool {module_name}: {e}")
        return self

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> Dict[str, BaseTool]:
        """Return all registered tools."""
        return self.tools.copy()
```

#### 4. Implement Code Execution Tool

Demonstrate a concrete tool implementation for executing Python code, a common requirement in reasoning and software engineering tasks.

```python
# verltool/tools/code_execution_tool.py
# Code execution tool with sandboxed environment support

from verltool.tools.base import BaseTool, ToolAction, ToolObservation
from typing import Dict, Any
import subprocess
import tempfile
import os

class CodeExecutionTool(BaseTool):
    """Execute Python code and capture output."""

    def __init__(self, timeout_seconds: int = 30):
        super().__init__(
            name="code_execution",
            modalities=["text"]
        )
        self.timeout = timeout_seconds
        self.execution_history = []

    def parse_action(self, action_string: str) -> ToolAction:
        """Extract Python code from action string."""
        # Expected format: <code>python_code_here</code>
        start = action_string.find('<code>')
        end = action_string.find('</code>')

        if start == -1 or end == -1:
            return ToolAction(
                tool_name=self.name,
                arguments={"code": action_string}
            )

        code = action_string[start + 6:end].strip()
        return ToolAction(
            tool_name=self.name,
            arguments={"code": code}
        )

    def execute(self, action: ToolAction) -> ToolObservation:
        """Run code in subprocess and capture output."""
        code = action.arguments.get("code", "")

        if not code:
            return ToolObservation(
                content="Error: No code provided",
                modality="text",
                error="empty_code"
            )

        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout
            if result.stderr:
                output += "\nStderr:\n" + result.stderr

            self.execution_history.append({
                "code": code,
                "output": output,
                "returncode": result.returncode
            })

            return ToolObservation(
                content=output or "(No output)",
                modality="text",
                error=None if result.returncode == 0 else "execution_error"
            )

        except subprocess.TimeoutExpired:
            return ToolObservation(
                content=f"Error: Code execution timeout (>{self.timeout}s)",
                modality="text",
                error="timeout"
            )
        except Exception as e:
            return ToolObservation(
                content=f"Error: {str(e)}",
                modality="text",
                error="execution_failed"
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

TOOL_CLASS = CodeExecutionTool
```

#### 5. Build Multi-Turn Trajectory Handling with Observation Masking

Implement trajectory construction that alternates actions and observations, with proper masking for off-policy observations.

```python
# verltool/training/trajectory.py
# Multi-turn trajectory representation with observation masking

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import torch

@dataclass
class MultiTurnTrajectory:
    """Represents a complete multi-turn agent trajectory."""

    actions: List[str] = field(default_factory=list)  # Agent actions/tool calls
    observations: List[str] = field(default_factory=list)  # Environment observations
    action_tokens: List[torch.Tensor] = field(default_factory=list)
    observation_tokens: List[torch.Tensor] = field(default_factory=list)
    observation_masks: List[bool] = field(default_factory=list)  # True = mask out
    reward: float = 0.0
    episode_done: bool = False

    def add_turn(
        self,
        action: str,
        observation: str,
        action_token_ids: torch.Tensor,
        observation_token_ids: torch.Tensor,
        mask_observation: bool = True
    ):
        """Add a single action-observation pair to trajectory."""
        self.actions.append(action)
        self.observations.append(observation)
        self.action_tokens.append(action_token_ids)
        self.observation_tokens.append(observation_token_ids)
        # Observation tokens are off-policy and should be masked
        self.observation_masks.append(mask_observation)

    def get_concatenated_tokens(self) -> torch.Tensor:
        """Interleave action and observation tokens."""
        sequence = []
        for i in range(len(self.actions)):
            sequence.append(self.action_tokens[i])
            if i < len(self.observation_tokens):
                sequence.append(self.observation_tokens[i])
        return torch.cat(sequence, dim=0)

    def get_loss_mask(self) -> torch.Tensor:
        """Create mask: True where loss should be computed, False otherwise."""
        mask = []
        for i in range(len(self.actions)):
            # Action tokens contribute to loss
            mask.append(torch.ones_like(self.action_tokens[i], dtype=torch.bool))
            # Observation tokens are masked (do not contribute to loss)
            if i < len(self.observation_masks):
                is_masked = self.observation_masks[i]
                obs_mask = torch.zeros_like(
                    self.observation_tokens[i],
                    dtype=torch.bool
                ) if is_masked else torch.ones_like(
                    self.observation_tokens[i],
                    dtype=torch.bool
                )
                mask.append(obs_mask)
        return torch.cat(mask, dim=0)

class TrajectoryBuilder:
    """Construct trajectories from agent rollouts."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_from_rollout(
        self,
        actions: List[str],
        observations: List[str],
        reward: float,
        mask_all_observations: bool = True
    ) -> MultiTurnTrajectory:
        """Convert rollout data into a trajectory with proper tokenization."""
        trajectory = MultiTurnTrajectory(reward=reward)

        for i, action in enumerate(actions):
            # Tokenize separately to avoid boundary issues
            action_tokens = self.tokenizer.encode(action)
            action_tensor = torch.tensor(action_tokens)

            if i < len(observations):
                obs = observations[i]
                obs_tokens = self.tokenizer.encode(obs)
                obs_tensor = torch.tensor(obs_tokens)

                trajectory.add_turn(
                    action=action,
                    observation=obs,
                    action_token_ids=action_tensor,
                    observation_token_ids=obs_tensor,
                    mask_observation=mask_all_observations
                )

        return trajectory
```

#### 6. Implement Asynchronous Tool Server for 2× Speedup

Deploy tool execution asynchronously so rollouts don't block waiting for tool completion.

```python
# verltool/server/async_tool_server.py
# Asynchronous tool execution with trajectory-level batching

import asyncio
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import queue
from verltool.tools.registry import ToolRegistry
from verltool.tools.base import ToolAction, ToolObservation

class AsyncToolServer:
    """Non-blocking tool execution server for efficient rollouts."""

    def __init__(self, tool_registry: ToolRegistry, max_workers: int = 32):
        self.registry = tool_registry
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks: Dict[str, asyncio.Future] = {}
        self.task_counter = 0

    def execute_tool_async(
        self,
        tool_name: str,
        action_string: str
    ) -> str:
        """Submit tool execution without blocking, return task ID."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return None

        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        def _run():
            action = tool.parse_action(action_string)
            observation = tool.execute(action)
            tool.update_state(observation)
            return observation

        future = asyncio.get_event_loop().run_in_executor(
            self.executor,
            _run
        )
        self.pending_tasks[task_id] = future
        return task_id

    async def wait_for_result(self, task_id: str, timeout: int = 300) -> ToolObservation:
        """Await tool completion and retrieve result."""
        if task_id not in self.pending_tasks:
            raise ValueError(f"Unknown task: {task_id}")

        try:
            result = await asyncio.wait_for(
                self.pending_tasks[task_id],
                timeout=timeout
            )
            del self.pending_tasks[task_id]
            return result
        except asyncio.TimeoutError:
            return ToolObservation(
                content=f"Timeout waiting for {task_id}",
                modality="text",
                error="timeout"
            )

    async def process_trajectory_batch(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> List[List[ToolObservation]]:
        """Execute all tool calls in a batch of trajectories concurrently."""
        all_observations = []

        for traj in trajectories:
            tool_calls = traj.get("tool_calls", [])
            tasks = [
                self.execute_tool_async(call["tool"], call["action"])
                for call in tool_calls
            ]

            observations = []
            for task_id in tasks:
                result = await self.wait_for_result(task_id)
                observations.append(result)

            all_observations.append(observations)

        return all_observations
```

#### 7. Configure Reward and Loss Computation

Define how rewards are computed for tool-use trajectories and how the loss respects observation masking.

```python
# verltool/training/reward.py
# Reward computation for multi-turn agentic trajectories

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

class VerlToolRewardComputer:
    """Compute rewards from verifiable task outcomes."""

    def __init__(self, reward_fn: callable, use_per_step_rewards: bool = False):
        """
        reward_fn: Function taking (final_output, expected_output) -> float
        use_per_step_rewards: If True, grant intermediate rewards for progress
        """
        self.reward_fn = reward_fn
        self.use_per_step_rewards = use_per_step_rewards

    def compute(
        self,
        final_output: str,
        expected_output: Optional[str] = None,
        intermediate_outputs: Optional[Dict[int, str]] = None
    ) -> float:
        """
        Compute scalar reward for trajectory.
        Follows RLVR principle: reward only depends on verifiable task outcome.
        """
        if expected_output is None:
            # Fallback: treat any valid output as success
            return 1.0 if final_output else 0.0

        base_reward = self.reward_fn(final_output, expected_output)

        if self.use_per_step_rewards and intermediate_outputs:
            # Optional: add small bonuses for intermediate progress
            bonus = 0.0
            for step, output in intermediate_outputs.items():
                if output and step < len(intermediate_outputs) - 1:
                    bonus += 0.05  # Small step reward
            return min(base_reward + bonus, 1.0)

        return base_reward

class PolicyLoss(nn.Module):
    """Compute policy gradient loss with observation masking."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        trajectory,  # MultiTurnTrajectory instance
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy loss, masking observation tokens.
        logits: Model output logits for entire sequence
        """
        loss_mask = trajectory.get_loss_mask()  # Bool tensor

        # Shift logits for language modeling objective
        shift_logits = logits[:-1]
        target_tokens = trajectory.get_concatenated_tokens()[1:]

        # Cross-entropy loss only on unmasked (action) tokens
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, logits.shape[-1]),
            target_tokens.view(-1),
            reduction='none'
        )

        # Apply mask to loss
        ce_loss = ce_loss * loss_mask[1:].float()

        # Weight by trajectory reward
        trajectory_reward = max(trajectory.reward, 0.0)
        weighted_loss = ce_loss.mean() * trajectory_reward

        return weighted_loss
```

### Practical Guidance

**Hyperparameter Configuration:**

| Parameter | Domain | Recommended | Rationale |
|-----------|--------|-------------|-----------|
| `max_turns` | Math/SQL | 8–12 | Allows sufficient tool calls without excessive sequences |
| `max_turns` | Code execution | 5–8 | Shorter: limited debugging iterations needed |
| `max_turns` | Web search | 4–6 | Search queries return quick results |
| `observation_masking` | All | True | Prevents off-policy observation leakage into gradients |
| `async_workers` | All | 32–64 | Balances concurrency; 2× speedup with ≥32 |
| `tokenization_mode` | All | "separate" | Avoids token boundary mismatches across turns |
| `timeout_seconds` | Code/search | 30–60 | Prevents runaway tool calls; adjust by domain |
| `learning_rate` | All | 1e-5 to 5e-5 | Tool-use RL requires careful tuning |

**When to Use VerlTool:**

- Multi-domain agents where code, search, SQL, and vision tools must coexist
- Organizations with shared RL infrastructure that multiple teams extend
- Tasks requiring multi-turn reasoning where outcomes are verifiable (rewards are computable)
- Scenarios demanding fast rollout throughput (async execution provides 2× improvement)
- Environments where observation tokens are abundant and masking prevents noise in gradients

**When NOT to Use VerlTool:**

- Single-task systems where domain-specific optimization is critical (specialized agents may outperform)
- Scenarios where tool observations directly determine rewards and policy (observation masking would discard signal)
- Real-time systems with strict latency budgets (asynchronous design adds queueing overhead)
- Environments with non-verifiable rewards (RLVR extension assumes ground-truth task outcomes)
- Teams without RL expertise managing training infrastructure (requires careful reward engineering and convergence tuning)

**Pitfalls to Avoid:**

1. **Neglecting Tool Timeout Configuration:** Set realistic timeouts per tool. Code execution might need 60s; search should timeout faster. Timeouts that are too long create training bottlenecks; too short causes spurious failures.

2. **Improper Reward Alignment:** Verify that reward functions reflect the actual task goal. Weak reward signals lead to high variance in RL and poor convergence. Use verification functions that match your evaluation metric.

3. **Forgetting to Reset Tool State:** Each trajectory should start fresh. Stale state from previous episodes corrupts observations. Always call `tool.reset()` between trajectories.

4. **Mixing Observation Masking Strategies:** Once you enable observation masking, apply it consistently across training. Inconsistent masking destabilizes gradients and causes sudden performance drops.

5. **Inadequate Batch Size for Async Server:** The speedup assumes enough concurrent trajectories to saturate workers. Small batch sizes (< 8) negate async benefits. Use batch sizes ≥ 16 when enabling async rollout.

6. **Skipping Tool Registry Discovery:** Hardcoding tools into training loops defeats modularity. Use the registry's `discover_from_directory()` method to enable true plug-and-play tool integration.

### Reference

**Paper:** VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use. Jiang, D., Lu, Y., et al. (2025). arXiv:2509.01055 [cs.AI].

**Full Citation:** https://arxiv.org/abs/2509.01055

**Key Contributions:** The framework demonstrates that unified, modular tool-use RL matches or exceeds domain-specific systems across six diverse benchmarks (mathematical reasoning at 62.2%, knowledge QA at 45.9%, SQL generation matching SkyRL-SQL, visual reasoning at 82.7%, web search at 34.0% GAIA accuracy, and software engineering at 19.5% SWE-Verified). Its 2× rollout speedup via async execution and observation masking strategy provide both efficiency and training stability for multi-turn agentic RL.
