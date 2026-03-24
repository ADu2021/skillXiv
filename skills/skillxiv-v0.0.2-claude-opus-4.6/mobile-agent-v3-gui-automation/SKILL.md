---
name: mobile-agent-v3-gui-automation
title: "Mobile-Agent-v3: Foundational GUI Automation with Self-Evolving Trajectories"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.15144
keywords: [gui-automation, mobile-agents, trajectory-generation, reinforcement-learning, desktop-automation]
description: "Build GUI automation agents using self-evolving trajectory generation, trajectory-aware policy optimization, and integrated action semantics for cross-platform interaction."
---

# Mobile-Agent-v3: Foundational GUI Automation with Self-Evolving Trajectories

## Core Concept

Mobile-Agent-v3 enables robust GUI automation across desktop and mobile platforms by combining self-evolving trajectory generation, integrated agent capabilities (UI grounding, planning, action semantics), and trajectory-aware reinforcement learning. The system generates high-quality interaction data automatically through iterative refinement, eliminating manual annotation. GUI-Owl, the foundational model, grounds visual understanding with semantic action capabilities, achieving state-of-the-art performance on diverse benchmarks.

## Architecture Overview

- **Self-Evolving Trajectory Production**: Cloud infrastructure for autonomous interaction data generation
- **GUI-Owl Foundation Model**: Integrated visual grounding with action semantics
- **Trajectory-Aware Relative Policy Optimization (TRPO)**: RL training with trajectory awareness
- **Multi-Platform Support**: Android, Windows, macOS, Linux execution
- **Modular Agent Capabilities**: UI grounding, planning, action semantics, reasoning

## Implementation Steps

### 1. Implement GUI State Representation

Create unified abstraction for visual and semantic information:

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

@dataclass
class GUIElement:
    """Represents interactive element in GUI."""
    element_id: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    element_type: str  # button, text_input, dropdown, etc.
    text_content: str
    clickable: bool
    visible: bool
    parent_id: str = None
    semantic_role: str = None  # based on context

@dataclass
class GUIState:
    """Complete GUI state snapshot."""
    screenshot: np.ndarray  # RGB image
    timestamp: float
    platform: str  # android, windows, macos, linux
    elements: List[GUIElement]
    dom_tree: Dict[str, Any]  # Hierarchical element structure
    text_content: str  # OCR extracted text

    def get_clickable_elements(self) -> List[GUIElement]:
        return [e for e in self.elements if e.clickable and e.visible]

    def get_element_by_id(self, element_id: str) -> GUIElement:
        return next((e for e in self.elements if e.element_id == element_id), None)
```

### 2. Implement Action Semantics

Define action space and execution:

```python
from enum import Enum

class ActionType(Enum):
    CLICK = "click"
    SCROLL = "scroll"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    TYPE_TEXT = "type_text"
    KEY_PRESS = "key_press"
    DRAG = "drag"

@dataclass
class Action:
    """Semantic action representation."""
    action_type: ActionType
    target_element_id: str = None
    parameters: Dict[str, Any] = None  # x, y, text, direction, duration, etc.

    def to_execution_command(self, gui_state: GUIState) -> Dict[str, Any]:
        """Convert semantic action to platform-specific command."""
        element = gui_state.get_element_by_id(self.target_element_id) if self.target_element_id else None

        if self.action_type == ActionType.CLICK:
            x, y = self._get_click_coordinates(element)
            return {"type": "click", "x": x, "y": y}

        elif self.action_type == ActionType.SCROLL:
            direction = self.parameters.get("direction", "down")
            amount = self.parameters.get("amount", 5)
            return {"type": "scroll", "direction": direction, "amount": amount}

        elif self.action_type == ActionType.TYPE_TEXT:
            text = self.parameters.get("text", "")
            return {"type": "type", "text": text}

        return {}

    def _get_click_coordinates(self, element: GUIElement) -> Tuple[int, int]:
        if element:
            x1, y1, x2, y2 = element.bbox
            return ((x1 + x2) // 2, (y1 + y2) // 2)
        else:
            return (self.parameters.get("x", 0), self.parameters.get("y", 0))
```

### 3. Implement UI Grounding Module

Ground visual elements with semantic understanding:

```python
import torch
from typing import Callable

class UIGrounder:
    """Grounds visual elements to semantic understanding."""

    def __init__(self, vision_model: Callable, language_model: Callable):
        self.vision_model = vision_model
        self.language_model = language_model

    def ground_elements(
        self,
        gui_state: GUIState,
        task_description: str
    ) -> List[Tuple[GUIElement, float, str]]:
        """
        Ground elements: (element, relevance_score, semantic_role)
        """
        results = []

        for element in gui_state.get_clickable_elements():
            # Extract visual features around element
            x1, y1, x2, y2 = element.bbox
            element_patch = gui_state.screenshot[y1:y2, x1:x2]

            # Get visual embedding
            visual_embedding = self.vision_model.encode(element_patch)

            # Create semantic context
            context = f"Task: {task_description}\nElement text: {element.text_content}\nType: {element.element_type}"

            # Compute relevance
            semantic_embedding = self.language_model.encode(context)
            relevance = torch.cosine_similarity(
                torch.tensor(visual_embedding).unsqueeze(0),
                torch.tensor(semantic_embedding).unsqueeze(0)
            ).item()

            # Determine semantic role
            role = self._infer_semantic_role(element, task_description, context)

            results.append((element, relevance, role))

        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _infer_semantic_role(
        self,
        element: GUIElement,
        task: str,
        context: str
    ) -> str:
        """Infer what role this element plays in task."""
        # Use language model to classify role
        role_prompt = f"Given task '{task}', element '{element.text_content}' of type '{element.element_type}' has semantic role: "
        role = self.language_model.generate(role_prompt, max_tokens=10)
        return role
```

### 4. Implement Trajectory Collection and Evolution

Generate and iteratively improve interaction trajectories:

```python
@dataclass
class Trajectory:
    """Complete interaction sequence."""
    task_description: str
    initial_state: GUIState
    actions: List[Action]
    states: List[GUIState]
    rewards: List[float]
    success: bool
    task_completion_rate: float

class TrajectoryCollector:
    def __init__(self, executor: "GUIExecutor", evaluator: "TaskEvaluator"):
        self.executor = executor
        self.evaluator = evaluator

    def collect_trajectory(
        self,
        task_description: str,
        max_steps: int = 50,
        initial_state: GUIState = None
    ) -> Trajectory:
        """Execute task and collect trajectory."""

        if initial_state is None:
            initial_state = self.executor.get_current_state()

        current_state = initial_state
        states = [initial_state]
        actions = []
        rewards = []

        for step in range(max_steps):
            # Generate action (initially: rule-based or random)
            action = self._generate_action(task_description, current_state, step)
            actions.append(action)

            # Execute action
            current_state = self.executor.execute_action(action)
            states.append(current_state)

            # Evaluate progress
            reward = self.evaluator.compute_reward(
                task_description,
                current_state,
                previous_state=states[-2] if len(states) > 1 else None
            )
            rewards.append(reward)

            # Check completion
            if self.evaluator.is_task_complete(task_description, current_state):
                break

        # Evaluate final trajectory
        success = self.evaluator.is_task_complete(task_description, current_state)
        completion_rate = self.evaluator.compute_completion_rate(
            task_description, current_state
        )

        return Trajectory(
            task_description=task_description,
            initial_state=initial_state,
            actions=actions,
            states=states,
            rewards=rewards,
            success=success,
            task_completion_rate=completion_rate
        )

    def _generate_action(
        self,
        task: str,
        state: GUIState,
        step: int
    ) -> Action:
        """Generate action using learned policy or heuristics."""
        # In practice: use fine-tuned model to select action
        # For initialization: use rule-based approach
        pass

    def evolve_trajectories(
        self,
        base_trajectory: Trajectory,
        num_variations: int = 5
    ) -> List[Trajectory]:
        """Generate trajectory variations through self-play."""
        evolved = [base_trajectory]

        for _ in range(num_variations):
            # Perturb trajectory slightly
            modified_actions = self._perturb_actions(base_trajectory.actions)

            # Execute modified trajectory
            # Collect improved trajectory if better
            new_traj = self.collect_trajectory(
                base_trajectory.task_description
            )

            if new_traj.task_completion_rate > base_trajectory.task_completion_rate:
                evolved.append(new_traj)

        return evolved

    def _perturb_actions(self, actions: List[Action]) -> List[Action]:
        """Introduce variations to action sequence."""
        # Randomly skip steps, modify parameters, etc.
        pass
```

### 5. Implement Trajectory-Aware Policy Optimization (TRPO)

Fine-tune agents using collected trajectories:

```python
class TrajectoryAwarePolicy:
    """Policy model aware of trajectory context."""

    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_trajectory_loss(
        self,
        trajectory: Trajectory,
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute loss considering entire trajectory context.
        """
        # Compute discounted cumulative rewards
        returns = []
        cumulative = 0
        for reward in reversed(trajectory.rewards):
            cumulative = reward + gamma * cumulative
            returns.insert(0, cumulative)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        total_loss = 0.0

        for step, (state, action, ret) in enumerate(zip(
            trajectory.states[:-1],
            trajectory.actions,
            returns
        )):
            # Get model prediction for this state
            with torch.no_grad():
                state_repr = self._encode_state(state)

            logits = self.model(state_repr)
            log_prob = torch.log_softmax(logits, dim=-1)
            action_idx = self._encode_action(action)

            # Policy gradient loss
            policy_loss = -log_prob[action_idx] * ret

            # Trajectory-aware regularization
            # Encourage consistency within trajectory
            if step > 0:
                prev_state_repr = self._encode_state(trajectory.states[step - 1])
                trajectory_consistency = torch.nn.functional.cosine_similarity(
                    state_repr.unsqueeze(0),
                    prev_state_repr.unsqueeze(0)
                )
                regularization = -trajectory_consistency * 0.01

                policy_loss = policy_loss + regularization

            total_loss += policy_loss

        return total_loss / len(trajectory.actions)

    def train_on_trajectories(self, trajectories: List[Trajectory]):
        """Train policy on batch of trajectories."""
        total_loss = 0.0

        for trajectory in trajectories:
            loss = self.compute_trajectory_loss(trajectory)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(trajectories)

    def _encode_state(self, state: GUIState) -> torch.Tensor:
        """Encode GUI state for policy input."""
        # Combine visual and semantic features
        visual_feat = self._extract_visual_features(state.screenshot)
        semantic_feat = self._extract_semantic_features(state)
        return torch.cat([visual_feat, semantic_feat], dim=-1)

    def _encode_action(self, action: Action) -> int:
        """Map semantic action to discrete action index."""
        pass

    def _extract_visual_features(self, screenshot: np.ndarray) -> torch.Tensor:
        pass

    def _extract_semantic_features(self, state: GUIState) -> torch.Tensor:
        pass
```

## Practical Guidance

### When to Use Mobile-Agent-v3

- GUI automation across desktop and mobile platforms
- Task learning from limited demonstrations
- Interactive systems requiring visual understanding
- Scenarios with procedural task variations
- Production deployment of GUI agents

### When NOT to Use

- Tasks without clear visual interface
- Real-time systems with strict latency (<100ms)
- Scenarios requiring semantic understanding beyond UI
- Proprietary or closed systems without API access

### Key Hyperparameters

- **max_steps_per_trajectory**: 30-100 based on task complexity
- **trajectory_variations**: 3-10 for evolution
- **learning_rate**: 1e-5 to 1e-4
- **gamma (discount factor)**: 0.99 standard
- **trajectory_consistency_weight**: 0.01-0.1

### Performance Expectations

- AndroidWorld: 73.3% success rate
- OSWorld: 37.7% success rate
- Cross-platform Generalization: Strong
- Data Efficiency: High-quality trajectories reduce annotation

## Reference

Researchers. (2024). Mobile-Agent-v3: Foundational Agents for GUI Automation. arXiv preprint arXiv:2508.15144.
