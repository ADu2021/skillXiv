---
name: robofactory-embodied-agent-collaboration
title: "RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16408"
keywords: [Multi-Robot Systems, Embodied AI, Imitation Learning, Compositional Constraints, Manipulation Tasks]
description: "Learn coordinated manipulation behaviors for multi-robot systems using compositional constraints that enforce safe and efficient collaboration. Generate training data through automated collection with task-specific constraint interfaces, then train imitation learning policies adaptable to varying difficulty levels."
---

## Core Concept

RoboFactory addresses the challenge of training multiple embodied agents to collaborate on complex manipulation tasks. The key innovation is compositional constraints—structured specifications that govern how agents should interact and coordinate. Rather than unstructured multi-agent learning, compositional constraints provide safety guardrails and efficiency guidance during both data collection and policy training. An automated data collection framework uses task-specific interfaces adapted to different constraint types, enabling safe generation of training trajectories. Imitation learning policies trained on this data learn coordinated behaviors that generalize across difficulty levels.

## Architecture Overview

The system has three primary components:

- **Compositional Constraint Specification**: Formal definitions of coordination rules and safety boundaries for multi-agent systems
- **Constraint-Aware Data Collection**: Automated framework using task-specific interaction mechanisms to generate training trajectories that respect constraints
- **Imitation Learning Policies**: Neural network policies trained from constraint-compliant demonstrations, adapted for varying task difficulty

The framework emphasizes both safety (constraints prevent dangerous collisions or invalid states) and efficiency (constraints guide agents toward coordinated solutions).

## Implementation

The compositional constraint system defines valid multi-agent state spaces and transitions:

```python
import numpy as np
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

class CompositionConstraint(ABC):
    """Base class for compositional constraints in multi-robot systems."""

    @abstractmethod
    def is_valid(self, agent_states: Dict[str, np.ndarray]) -> bool:
        """Check if current agent configuration satisfies constraint."""
        pass

    @abstractmethod
    def get_allowed_actions(self, agent_id: str, current_state: np.ndarray) -> np.ndarray:
        """Return mask of valid actions for agent given current state."""
        pass

    def sample_valid_state(self) -> Dict[str, np.ndarray]:
        """Sample a valid multi-agent configuration."""
        pass


class CollisionAvoidanceConstraint(CompositionConstraint):
    """Prevents robots from colliding with each other or environment."""

    def __init__(self, min_distance: float = 0.1):
        self.min_distance = min_distance

    def is_valid(self, agent_states: Dict[str, np.ndarray]) -> bool:
        """
        Args:
            agent_states: dict mapping agent_id -> position (3,)
        Returns:
            True if all pairwise distances exceed min_distance
        """
        agent_ids = list(agent_states.keys())
        for i, id1 in enumerate(agent_ids):
            for id2 in agent_ids[i+1:]:
                pos1 = agent_states[id1][:3]
                pos2 = agent_states[id2][:3]
                distance = np.linalg.norm(pos1 - pos2)
                if distance < self.min_distance:
                    return False
        return True

    def get_allowed_actions(self, agent_id: str, current_state: np.ndarray) -> np.ndarray:
        """Return action mask preventing collision trajectories."""
        # Discretize action space (e.g., 6 directions + no-op)
        action_mask = np.ones(7, dtype=bool)

        # Simulate potential next positions for each action
        for action_idx in range(7):
            next_pos = self._simulate_action(current_state, action_idx)
            # Check if next position would violate constraint
            if not self._is_collision_free(next_pos):
                action_mask[action_idx] = False

        return action_mask

    def _simulate_action(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        """Predict next state given action."""
        delta = 0.01  # step size
        next_state = state.copy()
        directions = [
            [delta, 0, 0], [-delta, 0, 0],
            [0, delta, 0], [0, -delta, 0],
            [0, 0, delta], [0, 0, -delta],
            [0, 0, 0]  # no-op
        ]
        next_state[:3] += directions[action_idx]
        return next_state

    def _is_collision_free(self, position: np.ndarray) -> bool:
        """Check if position is collision-free."""
        # In practice, query physics simulator
        return True


class CoordinationConstraint(CompositionConstraint):
    """Enforces task-specific coordination patterns."""

    def __init__(self, pattern: str = "simultaneous"):
        """
        Args:
            pattern: 'simultaneous', 'sequential', or 'interleaved'
        """
        self.pattern = pattern
        self.phase = 0

    def is_valid(self, agent_states: Dict[str, np.ndarray]) -> bool:
        """Validate that agents follow coordination pattern."""
        if self.pattern == "simultaneous":
            # All agents should be active
            return all(state[6] > 0.5 for state in agent_states.values())
        elif self.pattern == "sequential":
            # Only one agent active at a time
            active_count = sum(1 for state in agent_states.values() if state[6] > 0.5)
            return active_count == 1
        return True

    def get_allowed_actions(self, agent_id: str, current_state: np.ndarray) -> np.ndarray:
        """Restrict actions based on coordination pattern."""
        action_mask = np.ones(7, dtype=bool)

        if self.pattern == "sequential":
            # Only allow this agent to act if it's its turn
            is_active = current_state[6] > 0.5
            if not is_active:
                action_mask[:-1] = False  # only allow no-op

        return action_mask
```

The constraint-aware data collection framework generates safe training trajectories:

```python
class ConstraintAwareDataCollector:
    """Generates imitation learning data respecting compositional constraints."""

    def __init__(self, constraints: List[CompositionConstraint], max_episode_length: int = 100):
        self.constraints = constraints
        self.max_episode_length = max_episode_length

    def collect_episode(self, task: str, agent_ids: List[str]) -> Dict:
        """
        Collects one episode of multi-agent trajectories.

        Args:
            task: task description (e.g., 'pick_and_place')
            agent_ids: list of robot identifiers
        Returns:
            episode: dict with states, actions, rewards
        """
        episode = {
            'states': {agent_id: [] for agent_id in agent_ids},
            'actions': {agent_id: [] for agent_id in agent_ids},
            'rewards': [],
            'valid': True
        }

        # Initialize valid multi-agent state
        agent_states = self._initialize_valid_state(agent_ids)

        for step in range(self.max_episode_length):
            # Collect current state
            for agent_id in agent_ids:
                episode['states'][agent_id].append(agent_states[agent_id].copy())

            # Get allowed actions per constraint
            allowed_actions = {}
            for agent_id in agent_ids:
                masks = [
                    c.get_allowed_actions(agent_id, agent_states[agent_id])
                    for c in self.constraints
                ]
                # Intersect all masks (logical AND)
                allowed_actions[agent_id] = np.logical_and.reduce(masks)

            # Sample actions respecting constraints
            actions = {}
            for agent_id in agent_ids:
                valid_action_indices = np.where(allowed_actions[agent_id])[0]
                if len(valid_action_indices) == 0:
                    # No valid actions; episode invalid
                    episode['valid'] = False
                    break
                action_idx = np.random.choice(valid_action_indices)
                actions[agent_id] = action_idx
                episode['actions'][agent_id].append(action_idx)

            if not episode['valid']:
                break

            # Execute actions and update states
            agent_states = self._step_environment(agent_states, actions)

            # Verify constraints still satisfied
            for constraint in self.constraints:
                if not constraint.is_valid(agent_states):
                    episode['valid'] = False
                    break

            if not episode['valid']:
                break

            # Compute reward
            reward = self._compute_task_reward(agent_states, task)
            episode['rewards'].append(reward)

        return episode

    def _initialize_valid_state(self, agent_ids: List[str]) -> Dict[str, np.ndarray]:
        """Initialize multi-agent state satisfying all constraints."""
        # Simple initialization; in practice, use constraint sampling
        agent_states = {}
        for i, agent_id in enumerate(agent_ids):
            state = np.zeros(7)  # position (3) + velocity (3) + active flag (1)
            state[:3] = np.array([i * 0.2, 0, 0])  # Spread agents
            agent_states[agent_id] = state
        return agent_states

    def _step_environment(self, agent_states: Dict[str, np.ndarray], actions: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Simulate environment step with multi-robot physics."""
        next_states = {}
        for agent_id, state in agent_states.items():
            action_idx = actions[agent_id]
            next_states[agent_id] = state.copy()
            # Update position based on action (simple kinematics)
            if action_idx < 6:
                directions = [
                    [0.01, 0, 0], [-0.01, 0, 0],
                    [0, 0.01, 0], [0, -0.01, 0],
                    [0, 0, 0.01], [0, 0, -0.01]
                ]
                next_states[agent_id][:3] += directions[action_idx]
        return next_states

    def _compute_task_reward(self, agent_states: Dict[str, np.ndarray], task: str) -> float:
        """Compute task-specific reward signal."""
        # Placeholder; implement task-specific reward functions
        return 0.0 if task == "pick_and_place" else 0.0
```

Imitation learning policies are trained on constraint-compliant data:

```python
class MultiAgentImitationPolicy(nn.Module):
    """Neural network policy for multi-robot imitation learning."""

    def __init__(self, state_dim: int = 7, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, 7) agent state
        Returns:
            action_logits: (B, 7) action distribution
            confidence: (B, 1) policy confidence [0, 1]
        """
        encoded = self.state_encoder(state)
        action_logits = self.action_head(encoded)
        confidence = torch.sigmoid(self.confidence_head(encoded))
        return action_logits, confidence


def train_imitation_policy(
    episodes: List[Dict],
    policy: MultiAgentImitationPolicy,
    num_epochs: int = 100,
    learning_rate: float = 1e-3
):
    """
    Train policy on constraint-compliant demonstrations.

    Args:
        episodes: list of collected episodes
        policy: neural network policy
        num_epochs: training epochs
        learning_rate: optimizer learning rate
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for episode in episodes:
            if not episode['valid']:
                continue

            agent_ids = list(episode['states'].keys())
            for agent_id in agent_ids:
                states = torch.tensor(np.array(episode['states'][agent_id]), dtype=torch.float32)
                actions = torch.tensor(np.array(episode['actions'][agent_id]), dtype=torch.long)

                # Forward pass
                action_logits, confidence = policy(states)

                # Imitation loss
                loss = criterion(action_logits, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(episodes):.4f}")
```

## Practical Guidance

**When to Use:**
- Training multi-robot systems for collaborative manipulation tasks
- Scenarios where safety and coordination are critical
- Tasks with clear compositional structure (collision avoidance, turn-taking, etc.)
- Difficulty-varying domains (easy, medium, hard) requiring generalization
- Imitation learning settings with access to expert demonstrations

**When NOT to Use:**
- Single-robot control (overhead not justified)
- Fully decentralized systems with no coordination requirements
- Real-time learning scenarios (constraint checking adds overhead)
- Unstructured environments where compositional constraints don't apply
- Reinforcement learning from scratch without demonstrations

**Key Hyperparameters:**
- `min_distance`: 0.05-0.2 meters for collision avoidance
- `max_episode_length`: 100-500 steps depending on task complexity
- `hidden_dim`: 256-512 for policy networks
- `learning_rate`: 1e-3 to 1e-4 for stable training
- Constraint combination strategy: AND (all constraints) vs. weighted sum

**Common Pitfalls:**
- Constraints too restrictive, leaving few valid actions (training data bottleneck)
- Constraints too loose, allowing unsafe or inefficient behaviors
- Not verifying constraints are satisfiable before data collection
- Overfitting to constraint interface; poor generalization to real robots
- Ignoring task rewards in favor of purely constraint satisfaction

## Performance Notes

- Benchmark: RoboFactory embodied multi-agent manipulation
- Difficulty levels: Evaluated across easy, medium, hard configurations
- Imitation learning success rate: Varies by task and constraint complexity
- Training time: Hours to days depending on episode count and model size
- Generalization: Policies trained on medium difficulty generalize to hard difficulty

## References

- Multi-agent reinforcement learning and coordination
- Imitation learning from demonstrations
- Compositional AI and constraint-based systems
- Robot manipulation and multi-robot coordination
- Safety in embodied AI systems
