---
name: coda-dual-brain-agent
title: CODA Dual-Brain Computer Use Agent with Decoupled RL
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.20096
keywords: [agent-architecture, decoupled-training, planning, execution, computer-use, specialization]
description: "Train dual-brain agents with specialized planner (Cerebrum) and executor (Cerebellum) through decoupled RL, resolving planning-execution trade-off for scientific GUI agents"
---

# CODA: Dual-Brain Computer Use Agent with Decoupled RL

## Core Concept

CODA implements a compositional dual-brain architecture for computer use agents: a generalist planner (Cerebrum) and specialist executor (Cerebellum). Rather than a monolithic agent, the system trains specialized components for specific domains while maintaining a unified planning interface. Decoupled RL enables learning from limited task trajectories by first training domain experts, then aggregating their knowledge into a generalist.

## Architecture Overview

- **Cerebrum**: Generalist planner that reasons about task steps
- **Cerebellum**: Specialist executors for specific domains (CV, AI, ML)
- **Decoupled GRPO**: Train specialists individually on limited data, then consolidate
- **Domain Specialization**: Each executor optimized for specific tool patterns
- **Two-Stage Training**: Specialization phase → Generalization phase

## Implementation Steps

### Stage 1: Define Domain-Specific Task Trajectories

Collect task examples for each scientific domain.

```python
# Domain-specific trajectory collection
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TaskStep:
    """Single action in a task"""
    action: str  # e.g., "click", "type", "scroll"
    target: str  # e.g., button name, coordinate
    parameters: Dict[str, Any]  # Action parameters
    observation: str  # What happened after action
    reward: float = 0.0

@dataclass
class DomainTask:
    """Complete task in a specific domain"""
    task_id: str
    domain: str  # "computer_vision", "ai_ml", "biology"
    instruction: str
    steps: List[TaskStep]
    success: bool
    metadata: Dict = None


# Collect domain-specific trajectories
class DomainTrajectoryCollector:
    """Gather trajectories for each domain"""

    def __init__(self, domains: List[str]):
        self.domains = domains
        self.trajectories = {domain: [] for domain in domains}

    def collect_from_demonstrations(self, domain: str, demos: List[Dict]):
        """Collect human demonstration trajectories"""
        for demo in demos:
            steps = []
            for action in demo["actions"]:
                step = TaskStep(
                    action=action["type"],
                    target=action["target"],
                    parameters=action.get("params", {}),
                    observation=action["result"]
                )
                steps.append(step)

            task = DomainTask(
                task_id=demo["id"],
                domain=domain,
                instruction=demo["instruction"],
                steps=steps,
                success=demo["success"]
            )
            self.trajectories[domain].append(task)

    def get_domain_trajectories(self, domain: str) -> List[DomainTask]:
        """Get all trajectories for a domain"""
        return self.trajectories[domain]
```

### Stage 2: Train Domain Specialists

Train separate GRPO agents for each domain using limited task data.

```python
# Domain specialist training
import torch
from torch import nn

class DomainSpecialist(nn.Module):
    """Specialist agent for a specific domain"""

    def __init__(
        self,
        domain: str,
        model_dim: int = 2048,
        num_actions: int = 100
    ):
        super().__init__()
        self.domain = domain
        self.model_dim = model_dim

        # State encoder (GUI observations)
        self.state_encoder = nn.Sequential(
            nn.Linear(256, model_dim),  # Assume 256-dim state embeddings
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Action policy
        self.policy_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, num_actions)
        )

        # Value function
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1)
        )

    def forward(self, observation: torch.Tensor) -> tuple:
        """
        Predict action and value for observation.
        """
        state = self.state_encoder(observation)
        action_logits = self.policy_head(state)
        value = self.value_head(state)
        return action_logits, value


class SpecialistGRPOTrainer:
    """Train specialists using decoupled GRPO"""

    def __init__(self, specialist: DomainSpecialist, lr=1e-5):
        self.specialist = specialist
        self.optimizer = torch.optim.Adam(specialist.parameters(), lr=lr)

    def train_on_domain(
        self,
        trajectories: List[DomainTask],
        num_epochs: int = 3
    ) -> Dict:
        """
        Train specialist on domain-specific trajectories.
        """
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0

            for traj in trajectories:
                # Convert trajectory to training data
                states = []
                actions = []
                returns = []

                cumulative_return = traj.success * 100.0  # Reward for success
                for step in reversed(traj.steps):
                    returns.insert(0, cumulative_return)
                    actions.insert(0, step.action)
                    states.insert(0, self.encode_observation(step.observation))

                # GRPO update
                loss = self.grpo_step(states, actions, returns)
                epoch_loss += loss

            epoch_loss /= len(trajectories)
            losses.append(epoch_loss)

        return {
            "domain": self.specialist.domain,
            "final_loss": losses[-1],
            "loss_curve": losses
        }

    def grpo_step(
        self,
        states: List[torch.Tensor],
        actions: List[str],
        returns: List[float]
    ) -> float:
        """Single GRPO update"""
        # Get action logits and values
        state_batch = torch.stack(states)
        action_logits, values = self.specialist(state_batch)

        # Compute advantages
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advantages = returns_tensor - values.squeeze()

        # Policy loss: maximize log_prob * advantage
        action_indices = torch.tensor([self.action_to_index(a) for a in actions])
        log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), action_indices]

        policy_loss = -(selected_log_probs * advantages).mean()

        # Value loss: minimize (return - value)^2
        value_loss = ((returns_tensor - values.squeeze()) ** 2).mean()

        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def encode_observation(self, obs_str: str) -> torch.Tensor:
        """Encode GUI observation to tensor"""
        # Placeholder: in practice, use vision encoder + text encoder
        return torch.randn(256)

    def action_to_index(self, action_str: str) -> int:
        """Map action string to index"""
        action_dict = {
            "click": 0, "type": 1, "scroll": 2,
            "drag": 3, "wait": 4, "end_task": 99
        }
        return action_dict.get(action_str, 0)
```

### Stage 3: Create Unified Planner (Cerebrum)

Build a generalist planner that decides which domain executor to use.

```python
# Unified planner architecture
class UnifiedPlanner(nn.Module):
    """Cerebrum: Generalist planner routing to specialists"""

    def __init__(
        self,
        model_dim: int = 2048,
        num_domains: int = 3,
        num_actions: int = 100
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_domains = num_domains

        # Task encoder
        self.instruction_encoder = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Domain selector (which specialist to use)
        self.domain_selector = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, num_domains)
        )

        # Action planner (high-level steps)
        self.action_planner = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, num_actions)
        )

    def forward(self, instruction: str) -> Dict:
        """
        Plan task execution:
        1. Understand task
        2. Determine which domain
        3. Output high-level plan
        """
        # Encode instruction
        instruction_embedding = self.encode_instruction(instruction)
        encoded = self.instruction_encoder(instruction_embedding)

        # Predict domain
        domain_logits = self.domain_selector(encoded)
        domain_probs = torch.nn.functional.softmax(domain_logits, dim=-1)

        # Predict action sequence
        action_logits = self.action_planner(encoded)
        action_probs = torch.nn.functional.softmax(action_logits, dim=-1)

        return {
            "domain_logits": domain_logits,
            "domain_probs": domain_probs,
            "action_logits": action_logits,
            "action_probs": action_probs
        }

    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode task instruction"""
        return torch.randn(512)  # Placeholder
```

### Stage 4: Two-Stage Training Pipeline

First train specialists, then train unified planner using expert trajectories.

```python
# Two-stage training orchestration
class DualBrainTrainer:
    """Training pipeline: specialists → planner"""

    def __init__(
        self,
        domain_specialists: Dict[str, DomainSpecialist],
        planner: UnifiedPlanner,
        trajectory_collector: DomainTrajectoryCollector
    ):
        self.specialists = domain_specialists
        self.planner = planner
        self.trajectories = trajectory_collector
        self.planner_optimizer = torch.optim.Adam(planner.parameters(), lr=1e-5)

    def stage1_train_specialists(self) -> Dict:
        """
        Stage 1: Train each specialist on domain-specific trajectories.
        Done with limited data.
        """
        print("Stage 1: Training domain specialists...")
        specialist_results = {}

        for domain, specialist in self.specialists.items():
            print(f"  Training {domain} specialist...")

            domain_trajs = self.trajectories.get_domain_trajectories(domain)
            trainer = SpecialistGRPOTrainer(specialist)

            results = trainer.train_on_domain(domain_trajs, num_epochs=3)
            specialist_results[domain] = results

        return specialist_results

    def stage2_aggregate_and_train_planner(self) -> Dict:
        """
        Stage 2: Aggregate trajectories from all specialists.
        Train unified planner on consolidated dataset.
        """
        print("Stage 2: Aggregating and training unified planner...")

        # Aggregate successful trajectories from all domains
        aggregated_trajs = []
        for domain in self.specialists.keys():
            domain_trajs = self.trajectories.get_domain_trajectories(domain)
            successful = [t for t in domain_trajs if t.success]
            aggregated_trajs.extend(successful)

        # Train planner on aggregated data
        planner_loss = self.train_planner(aggregated_trajs)

        return {
            "aggregated_trajectory_count": len(aggregated_trajs),
            "planner_loss": planner_loss
        }

    def train_planner(self, trajectories: List[DomainTask]) -> float:
        """Train unified planner"""
        epoch_loss = 0

        for traj in trajectories:
            # Predict domain for this task
            plan = self.planner(traj.instruction)

            # Get ground truth domain
            domain_index = self.domain_to_index(traj.domain)

            # Cross-entropy loss on domain prediction
            domain_logits = plan["domain_logits"]
            domain_loss = torch.nn.functional.cross_entropy(
                domain_logits.unsqueeze(0),
                torch.tensor([domain_index])
            )

            # Action loss
            action_logits = plan["action_logits"]
            action_targets = torch.tensor([
                self.action_to_index(step.action) for step in traj.steps
            ])

            action_loss = torch.nn.functional.cross_entropy(
                action_logits.unsqueeze(0).expand(len(action_targets), -1),
                action_targets
            )

            total_loss = domain_loss + action_loss
            epoch_loss += total_loss.item()

            # Backward
            self.planner_optimizer.zero_grad()
            total_loss.backward()
            self.planner_optimizer.step()

        return epoch_loss / len(trajectories)

    def domain_to_index(self, domain: str) -> int:
        domain_map = {
            "computer_vision": 0,
            "ai_ml": 1,
            "biology": 2
        }
        return domain_map.get(domain, 0)

    def action_to_index(self, action: str) -> int:
        action_map = {
            "click": 0, "type": 1, "scroll": 2,
            "drag": 3, "wait": 4, "end_task": 99
        }
        return action_map.get(action, 0)
```

### Stage 5: Integrated Inference and Evaluation

Execute tasks using the dual-brain system and evaluate performance.

```python
# Integrated inference and evaluation
class DualBrainAgent:
    """Complete dual-brain agent for execution"""

    def __init__(
        self,
        planner: UnifiedPlanner,
        specialists: Dict[str, DomainSpecialist]
    ):
        self.planner = planner
        self.specialists = specialists

    def execute_task(self, instruction: str, max_steps: int = 100) -> Dict:
        """
        Execute task using dual-brain:
        1. Planner decides domain
        2. Domain specialist executes steps
        """
        # Step 1: Planning
        plan = self.planner(instruction)
        domain_idx = plan["domain_probs"].argmax().item()
        domain = self.index_to_domain(domain_idx)

        print(f"Task: {instruction}")
        print(f"Domain: {domain}")

        # Step 2: Specialist execution
        specialist = self.specialists[domain]
        steps_executed = 0
        total_reward = 0

        for step_idx in range(max_steps):
            # Get action from specialist
            # (In practice, this would be actual GUI interaction)
            action = self.get_specialist_action(specialist, instruction)

            if action == "end_task":
                break

            steps_executed += 1

        return {
            "task": instruction,
            "domain": domain,
            "steps": steps_executed,
            "success": steps_executed > 0
        }

    def get_specialist_action(self, specialist: DomainSpecialist, instruction: str) -> str:
        """Get action from domain specialist"""
        return "click"  # Placeholder

    def index_to_domain(self, idx: int) -> str:
        domains = {0: "computer_vision", 1: "ai_ml", 2: "biology"}
        return domains.get(idx, "general")


class DualBrainEvaluator:
    """Evaluate dual-brain agent performance"""

    def evaluate_on_benchmark(
        self,
        agent: DualBrainAgent,
        test_tasks: List[DomainTask]
    ) -> Dict:
        """Evaluate on scientific benchmark"""
        results = {"overall_success": 0, "by_domain": {}}

        for task in test_tasks:
            execution = agent.execute_task(task.instruction)

            if task.domain not in results["by_domain"]:
                results["by_domain"][task.domain] = {"success": 0, "total": 0}

            results["by_domain"][task.domain]["total"] += 1
            if execution["success"]:
                results["by_domain"][task.domain]["success"] += 1
                results["overall_success"] += 1

        # Compute success rates
        results["overall_success_rate"] = results["overall_success"] / len(test_tasks)

        for domain in results["by_domain"]:
            domain_stats = results["by_domain"][domain]
            domain_stats["success_rate"] = (
                domain_stats["success"] / domain_stats["total"]
            )

        return results
```

## Practical Guidance

### Specialist Training

- **Data per Domain**: 50-200 demonstrations per domain sufficient
- **Training Time**: Specialists converge in 1-3 epochs on limited data
- **Domain Definition**: 3-5 domains work well; more requires more data

### Planner Training

- **Aggregation Strategy**: Use all successful trajectories across domains
- **Domain Routing**: Softmax over domain predictions for differentiable routing
- **Fine-tuning**: Can further specialize planner on new domains

### When to Use

- Multi-domain agents with specialization requirements
- Scenarios with limited task examples per domain
- Computer use / GUI automation tasks
- Systems where planning and execution conflict

### When NOT to Use

- Single-domain problems (use monolithic agent)
- Real-time systems (domain routing adds latency)
- Domains requiring frequent cross-specialization

### Design Insights

The dual-brain architecture solves the planning-execution trade-off: generalist planners excel at reasoning across domains but struggle at execution details, while specialists execute well but lack generalization. By training them separately then consolidating, CODA gets the best of both.

## Reference

CODA: Dual-Brain Computer Use Agent with Decoupled RL. arXiv:2508.20096
- https://arxiv.org/abs/2508.20096
