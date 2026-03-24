---
name: aworld-distributed-agent-training
title: AWorld Distributed Training Recipe for Agentic AI
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.20404
keywords: [distributed-training, agent-learning, reinforcement-learning, scalability, experience-collection]
description: "Accelerate agentic AI training by distributing task execution across clusters, achieving 14.6x speedup in experience collection and enabling practical large-scale agent development"
---

# AWorld: Orchestrating Training Recipe for Agentic AI

## Core Concept

AWorld addresses the bottleneck in reinforcement learning for agents: the slow collection of environmental interactions. By distributing task execution across multiple cluster nodes, rather than sequential single-machine execution, the system dramatically accelerates experience generation. This enables practical training of capable agents on complex benchmarks like GAIA, where computational efficiency is critical.

## Architecture Overview

- **Distributed Executor**: Parallel task execution across cluster nodes
- **Centralized Experience Aggregation**: Collects trajectories from all workers
- **Efficient Communication**: Minimal overhead for distributed coordination
- **Scalable RL Pipeline**: From experience collection through model training
- **Production-Ready Implementation**: Open-source reference implementation

## Implementation Steps

### Stage 1: Design Distributed Task Execution Framework

Create infrastructure for parallel task distribution and result collection.

```python
# Distributed task execution framework
import asyncio
from typing import Dict, List, Any
import pickle
import queue
from concurrent.futures import ProcessPoolExecutor

class DistributedExecutor:
    """Execute tasks across multiple nodes in parallel"""

    def __init__(self, num_workers: int, max_queue_size: int = 1000):
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.running_tasks = {}

    def submit_task(self, task_id: str, task_fn, *args, **kwargs):
        """Submit a task for distributed execution"""
        future = self.executor.submit(task_fn, *args, **kwargs)
        self.running_tasks[task_id] = future
        return task_id

    def collect_results(self, timeout: float = None) -> Dict:
        """Collect completed task results"""
        completed = {}
        for task_id, future in list(self.running_tasks.items()):
            if future.done():
                try:
                    result = future.result(timeout=timeout)
                    completed[task_id] = {"success": True, "result": result}
                except Exception as e:
                    completed[task_id] = {"success": False, "error": str(e)}
                del self.running_tasks[task_id]

        return completed

    def shutdown(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)


class RemoteNodeExecutor:
    """Execute on remote cluster nodes (for distributed setup)"""

    def __init__(self, nodes: List[str]):
        """
        Args:
            nodes: List of remote node addresses (e.g., "worker1.cluster:5000")
        """
        self.nodes = nodes
        self.node_load = {node: 0 for node in nodes}

    async def submit_to_least_loaded(self, task_fn, *args, **kwargs) -> str:
        """Route task to least-loaded node"""
        node = min(self.nodes, key=lambda n: self.node_load[n])
        self.node_load[node] += 1

        task_id = await self.submit_remote_task(node, task_fn, *args, **kwargs)
        return task_id

    async def submit_remote_task(self, node: str, task_fn, *args, **kwargs) -> str:
        """Send task to remote node via RPC"""
        import rpc_client  # Hypothetical RPC library

        client = rpc_client.connect(node)
        serialized_fn = pickle.dumps(task_fn)
        serialized_args = pickle.dumps((args, kwargs))

        task_id = await client.submit_task(
            function=serialized_fn,
            arguments=serialized_args
        )
        return task_id
```

### Stage 2: Implement Task and Trajectory Definition

Define what constitutes a task and how to collect trajectories from execution.

```python
# Task and trajectory definitions
from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    """Single task for agent to solve"""
    task_id: str
    instruction: str
    tools: List[str]  # Available tools
    ground_truth: Any  # Expected solution
    metadata: Dict = None

    def to_prompt(self) -> str:
        """Convert task to agent prompt"""
        return f"Instruction: {self.instruction}\nAvailable tools: {self.tools}"


@dataclass
class Trajectory:
    """Single execution trace from task"""
    task_id: str
    agent_id: str
    steps: List[Dict]  # Each step: action, observation, reward
    final_result: Any
    success: bool
    token_count: int
    execution_time: float
    metadata: Dict = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "steps": self.steps,
            "success": self.success,
            "token_count": self.token_count,
            "execution_time": self.execution_time
        }
```

### Stage 3: Create Parallel Task Executor with Agent Integration

Execute tasks on distributed workers with agent inference.

```python
# Parallel task execution with agent
import time
from typing import Callable

class ParallelTaskExecutor:
    """Execute tasks in parallel across workers"""

    def __init__(
        self,
        agent_model: Any,
        distributed_executor: DistributedExecutor,
        task_loader: Callable
    ):
        self.agent = agent_model
        self.executor = distributed_executor
        self.task_loader = task_loader
        self.trajectories = []

    def execute_single_task(
        self,
        task: Task,
        max_steps: int = 10,
        timeout: float = 60
    ) -> Trajectory:
        """
        Execute a single task with agent.
        Can run in parallel across workers.
        """
        steps = []
        current_state = {"instruction": task.instruction}
        total_tokens = 0
        start_time = time.time()
        success = False

        try:
            for step_idx in range(max_steps):
                # Agent generates action
                action, action_tokens = self.agent.generate_action(
                    current_state,
                    max_tokens=256
                )
                total_tokens += action_tokens

                # Execute action
                observation = self.execute_action(action, task)

                # Record step
                steps.append({
                    "step": step_idx,
                    "action": action,
                    "observation": observation,
                    "tokens": action_tokens
                })

                # Update state
                current_state["last_action"] = action
                current_state["last_observation"] = observation

                # Check success
                if observation.get("done"):
                    success = observation.get("success", False)
                    break

        except Exception as e:
            steps.append({
                "step": "error",
                "error": str(e)
            })

        execution_time = time.time() - start_time

        trajectory = Trajectory(
            task_id=task.task_id,
            agent_id=self.agent.model_id,
            steps=steps,
            final_result=current_state.get("last_observation"),
            success=success,
            token_count=total_tokens,
            execution_time=execution_time
        )

        return trajectory

    def execute_action(self, action: str, task: Task) -> Dict:
        """Execute an action in the environment"""
        # Parse action (e.g., "use_tool[search] with query[...]")
        tool_name = self.parse_tool_name(action)
        tool_args = self.parse_tool_args(action)

        try:
            if tool_name in task.tools:
                # Simulate or call actual tool
                result = self.call_tool(tool_name, tool_args)
                return {"observation": result, "done": False}
            else:
                return {"observation": f"Unknown tool: {tool_name}", "done": True}

        except Exception as e:
            return {"observation": f"Error: {str(e)}", "done": True}

    def distribute_tasks(self, tasks: List[Task]) -> List[Trajectory]:
        """
        Distribute all tasks across workers.
        This is where the 14.6x speedup comes from!
        """
        # Submit all tasks to distributed executor
        submitted_tasks = {}
        for task in tasks:
            task_id = self.executor.submit_task(
                task.task_id,
                self.execute_single_task,
                task
            )
            submitted_tasks[task_id] = task

        # Collect results as they complete
        trajectories = []
        while submitted_tasks:
            results = self.executor.collect_results(timeout=1.0)

            for task_id, result in results.items():
                if result["success"]:
                    trajectories.append(result["result"])
                else:
                    # Handle failed execution
                    print(f"Task {task_id} failed: {result['error']}")

                del submitted_tasks[task_id]

        return trajectories
```

### Stage 4: Implement Experience Aggregation and RL Training

Collect experiences from all workers and train the model.

```python
# Experience aggregation and RL training
import numpy as np

class ExperienceBuffer:
    """Buffer for collecting trajectories"""

    def __init__(self, max_size: int = 100000):
        self.trajectories = []
        self.max_size = max_size

    def add_trajectory(self, traj: Trajectory):
        """Add trajectory to buffer"""
        self.trajectories.append(traj)
        if len(self.trajectories) > self.max_size:
            self.trajectories.pop(0)

    def add_batch(self, trajs: List[Trajectory]):
        """Add multiple trajectories"""
        for traj in trajs:
            self.add_trajectory(traj)

    def sample_batch(self, batch_size: int) -> List[Trajectory]:
        """Sample random batch"""
        indices = np.random.choice(len(self.trajectories), batch_size)
        return [self.trajectories[i] for i in indices]

    def compute_returns(self, discount: float = 0.99):
        """Compute cumulative returns for trajectories"""
        for traj in self.trajectories:
            returns = []
            cumulative = 0
            for step in reversed(traj.steps):
                reward = step.get("reward", 1.0 if traj.success else 0.0)
                cumulative = reward + discount * cumulative
                returns.insert(0, cumulative)
            traj.returns = returns


class RLTrainer:
    """Train agent using RL on collected trajectories"""

    def __init__(self, agent_model, learning_rate: float = 1e-5):
        self.model = agent_model
        self.optimizer = agent_model.optimizer_class(
            agent_model.parameters(), lr=learning_rate
        )

    def train_step(self, batch: List[Trajectory]) -> float:
        """Single RL training step"""
        total_loss = 0

        for traj in batch:
            # Compute policy gradient loss
            for step_idx, step in enumerate(traj.steps):
                action_tokens = step.get("tokens", 0)
                # In practice, compute log probs of actions taken
                log_prob = self.model.get_log_prob(
                    step["action"],
                    traj.steps[:step_idx]
                )

                # Return as advantage signal
                advantage = traj.returns[step_idx]

                # Policy gradient: maximize log_prob * advantage
                loss = -(log_prob * advantage)
                total_loss += loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return (total_loss / len(batch)).item()


class DistributedTrainingPipeline:
    """Full pipeline: distribute execution, collect experiences, train"""

    def __init__(
        self,
        agent_model,
        num_workers: int,
        task_dataset: List[Task]
    ):
        self.agent = agent_model
        self.executor = DistributedExecutor(num_workers)
        self.task_executor = ParallelTaskExecutor(
            agent_model,
            self.executor,
            lambda: task_dataset
        )
        self.experience_buffer = ExperienceBuffer()
        self.trainer = RLTrainer(agent_model)
        self.task_dataset = task_dataset

    def train_epoch(self, tasks_per_epoch: int = 1000):
        """Single training epoch"""
        # Select random tasks
        selected_tasks = np.random.choice(
            self.task_dataset,
            size=min(tasks_per_epoch, len(self.task_dataset)),
            replace=False
        ).tolist()

        print(f"Executing {len(selected_tasks)} tasks in parallel...")

        # Execute all tasks in parallel across workers
        trajectories = self.task_executor.distribute_tasks(selected_tasks)

        print(f"Collected {len(trajectories)} trajectories")

        # Add to experience buffer
        self.experience_buffer.add_batch(trajectories)

        # Compute returns
        self.experience_buffer.compute_returns()

        # Train on batches
        print("Training on collected experiences...")
        batch_size = 32
        for _ in range(len(trajectories) // batch_size):
            batch = self.experience_buffer.sample_batch(batch_size)
            loss = self.trainer.train_step(batch)

        return {
            "trajectories": len(trajectories),
            "success_rate": sum(t.success for t in trajectories) / len(trajectories),
            "avg_tokens": np.mean([t.token_count for t in trajectories])
        }
```

## Practical Guidance

### Scaling Configuration

- **Worker Count**: Start with 8-16 workers; scale up to 64+ for large benchmarks
- **Task Batch Size**: 100-1000 tasks per distribution cycle
- **Experience Buffer**: Keep last 10,000-100,000 trajectories for training stability
- **Training Frequency**: After each 1000 task executions, perform 1-5 training epochs

### Performance Optimization

- **Network Efficiency**: Compress trajectories before transfer (usually 10-100KB each)
- **Load Balancing**: Use least-loaded routing to avoid hotspots
- **Fault Tolerance**: Implement retry logic for failed tasks
- **Memory Management**: Stream results to disk if buffer exceeds RAM

### Baseline Metrics

- **Single-Node Baseline**: ~10 tasks/minute on 1 GPU
- **16-Worker Cluster**: ~140-150 tasks/minute (14.6x speedup)
- **GAIA Performance**: 32.23% pass@1 with Qwen3-32B agent (vs 27.91% GPT-4o)

### When to Use

- Training agents on complex benchmarks (GAIA, ScienceBoard)
- Settings with access to multi-node clusters
- Problems requiring millions of task executions
- Scenarios where wall-clock training time matters

### When NOT to Use

- Single-machine environments without cluster access
- Tasks with very fast execution (<1 second)
- Strongly sequential dependencies between tasks
- Scenarios with limited network bandwidth

## Reference

AWorld: Orchestrating Training Recipe for Agentic AI. arXiv:2508.20404
- https://arxiv.org/abs/2508.20404
