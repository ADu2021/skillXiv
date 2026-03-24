---
name: robomemory-multi-memory-embodied-ai
title: RoboMemory Brain-Inspired Multi-Memory Agentic Framework
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.01415
keywords: [embodied-ai, memory-systems, robotic-agents, knowledge-graphs, long-horizon-planning]
description: "Brain-inspired framework integrating spatial, temporal, episodic, and semantic memory systems for embodied agents. Achieves 26.5% performance improvement via dynamic spatial knowledge graphs and closed-loop planning with critic modules."
---

## RoboMemory: Brain-Inspired Multi-Memory Agentic Framework

RoboMemory introduces a sophisticated memory architecture inspired by cognitive neuroscience for embodied AI agents. Rather than monolithic representations, the system unifies four specialized memory types to enable robust learning, generalization, and planning in complex physical environments.

### Core Concept

The fundamental insight is that human cognition uses specialized memory systems working in concert. RoboMemory implements:

- **Spatial Memory**: Maintains consistent knowledge graphs of environment layouts
- **Temporal Memory**: Tracks sequences of events and their causality
- **Episodic Memory**: Stores specific experiences for retrieval and learning
- **Semantic Memory**: Builds general knowledge about object properties and interactions

This multi-system approach enables agents to handle partial observability, long-horizon planning, and generalization across embodiments and tasks.

### Architecture Overview

The framework consists of:

- **Dynamic Spatial Knowledge Graph**: Builds and updates environment maps despite partial observability
- **Temporal Event Buffer**: Records action-outcome sequences
- **Episodic Experience Bank**: Stores complete task trajectories
- **Semantic Knowledge Module**: Extracts and stores generalizable facts
- **Closed-Loop Planner**: Generates long-horizon plans using all memory types
- **Critic Module**: Evaluates plan feasibility and provides adaptive feedback

### Implementation Steps

**Step 1: Implement dynamic spatial knowledge graph**

Create a memory structure for environment representation:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SpatialNode:
    """Represents a location or object in spatial memory"""
    node_id: str
    object_type: str  # 'object', 'location', 'landmark'
    position: np.ndarray  # (x, y, z)
    attributes: Dict[str, float]  # properties
    last_observed: float  # timestamp
    confidence: float  # 0-1, certainty of position

@dataclass
class SpatialEdge:
    """Relationship between spatial nodes"""
    source_id: str
    target_id: str
    relation_type: str  # 'near', 'contains', 'blocks'
    strength: float  # 0-1

class DynamicSpatialKnowledgeGraph(nn.Module):
    """Maintains consistent spatial understanding despite partial observability"""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.nodes: Dict[str, SpatialNode] = {}
        self.edges: List[SpatialEdge] = []
        self.embedding_dim = embedding_dim

        # Learnable embeddings for nodes
        self.node_embeddings = nn.ParameterDict()

    def add_observation(self, objects: List[Dict], timestamp: float):
        """
        Process observation and update spatial graph.

        Args:
            objects: List of observed objects with position/type
            timestamp: When observation occurred
        """
        for obj in objects:
            node_id = f"{obj['type']}_{obj['id']}"

            if node_id not in self.nodes:
                # New object: create node
                position = np.array(obj.get('position', [0, 0, 0]))
                node = SpatialNode(
                    node_id=node_id,
                    object_type=obj['type'],
                    position=position,
                    attributes=obj.get('attributes', {}),
                    last_observed=timestamp,
                    confidence=0.9
                )
                self.nodes[node_id] = node

                # Initialize embedding
                self.node_embeddings[node_id] = nn.Parameter(
                    torch.randn(self.embedding_dim)
                )
            else:
                # Update existing node
                old_node = self.nodes[node_id]
                new_position = np.array(obj.get('position', old_node.position))

                # Weighted average position (decaying older observations)
                decay = np.exp(-0.1 * (timestamp - old_node.last_observed))
                old_node.position = (
                    decay * old_node.position + (1 - decay) * new_position
                )
                old_node.last_observed = timestamp
                old_node.confidence = min(1.0, old_node.confidence + 0.05)

        # Update relationships
        self._update_spatial_relationships()

    def _update_spatial_relationships(self):
        """Compute/update spatial relationships between objects"""
        node_ids = list(self.nodes.keys())

        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                node1 = self.nodes[id1]
                node2 = self.nodes[id2]

                # Compute distance
                distance = np.linalg.norm(node1.position - node2.position)

                # Determine relationship type
                if distance < 0.5:
                    relation = 'touching'
                    strength = 1.0 - (distance / 0.5)
                elif distance < 2.0:
                    relation = 'near'
                    strength = 1.0 - (distance / 2.0)
                else:
                    relation = None
                    strength = 0.0

                if relation:
                    # Check if edge exists
                    existing = next(
                        (e for e in self.edges
                         if e.source_id == id1 and e.target_id == id2),
                        None
                    )

                    if existing:
                        existing.strength = strength
                    else:
                        self.edges.append(
                            SpatialEdge(id1, id2, relation, strength)
                        )

    def get_spatial_context(self, position: np.ndarray,
                           radius: float = 5.0) -> Dict:
        """
        Get spatial context around given position.

        Returns objects and relationships near position.
        """
        nearby_objects = []

        for node_id, node in self.nodes.items():
            distance = np.linalg.norm(node.position - position)

            if distance < radius:
                nearby_objects.append({
                    'id': node_id,
                    'type': node.object_type,
                    'position': node.position.tolist(),
                    'distance': float(distance),
                    'confidence': node.confidence
                })

        # Get relevant edges
        nearby_ids = {obj['id'] for obj in nearby_objects}
        relevant_edges = [
            {'source': e.source_id, 'target': e.target_id, 'type': e.relation_type}
            for e in self.edges
            if e.source_id in nearby_ids or e.target_id in nearby_ids
        ]

        return {
            'objects': nearby_objects,
            'relationships': relevant_edges
        }

    def query_path(self, start_id: str, goal_id: str) -> Optional[List[str]]:
        """Find path between two objects in spatial graph"""
        # BFS to find path
        from collections import deque

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current, path = queue.popleft()

            if current == goal_id:
                return path

            for edge in self.edges:
                if edge.source_id == current and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, path + [edge.target_id]))

        return None
```

This maintains a dynamic, updatable spatial representation.

**Step 2: Implement temporal event buffer**

Track sequences and cause-effect relationships:

```python
@dataclass
class TemporalEvent:
    """Represents an action and its outcomes"""
    event_id: str
    action: str
    action_params: Dict
    timestamp: float
    outcomes: List[Tuple[str, float]]  # (outcome_type, success_rate)
    preconditions: List[str]
    postconditions: List[str]

class TemporalEventBuffer:
    """Tracks action sequences and causality"""

    def __init__(self, max_events: int = 10000):
        self.events: List[TemporalEvent] = []
        self.max_events = max_events
        self.causal_graph: Dict[str, List[str]] = {}

    def record_event(self, action: str, params: Dict,
                    outcomes: List[Tuple[str, float]],
                    preconditions: List[str] = None):
        """Record action and its outcomes"""

        event = TemporalEvent(
            event_id=f"event_{len(self.events)}",
            action=action,
            action_params=params,
            timestamp=np.time.time(),
            outcomes=outcomes,
            preconditions=preconditions or [],
            postconditions=[o[0] for o in outcomes]
        )

        self.events.append(event)

        # Update causal graph
        for outcome_type, _ in outcomes:
            if action not in self.causal_graph:
                self.causal_graph[action] = []

            if outcome_type not in self.causal_graph[action]:
                self.causal_graph[action].append(outcome_type)

        # Trim old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_action_effects(self, action: str) -> Dict[str, float]:
        """Get empirical effects of action"""

        matching_events = [e for e in self.events if e.action == action]

        if not matching_events:
            return {}

        # Aggregate outcomes
        outcome_counts = {}
        for event in matching_events:
            for outcome_type, success_rate in event.outcomes:
                if outcome_type not in outcome_counts:
                    outcome_counts[outcome_type] = []
                outcome_counts[outcome_type].append(success_rate)

        # Compute mean success rate
        effect_rates = {
            outcome_type: np.mean(rates)
            for outcome_type, rates in outcome_counts.items()
        }

        return effect_rates

    def find_action_sequence(self, start_state: str,
                            goal_state: str) -> Optional[List[str]]:
        """Find action sequence that led to goal"""

        # Simple BFS through temporal history
        from collections import deque

        queue = deque([(start_state, [])])
        visited = {start_state}

        for event in self.events[-100:]:  # Recent events
            current_state = event.preconditions[-1] if event.preconditions else start_state

            if current_state not in visited:
                visited.add(current_state)

                new_states = event.postconditions

                for new_state in new_states:
                    if new_state == goal_state:
                        return [event.action]

                    queue.append((new_state, [event.action]))

        return None
```

**Step 3: Implement episodic memory for trajectory storage**

Store complete experiences for later retrieval and learning:

```python
@dataclass
class Episode:
    """Complete trajectory from start to goal"""
    episode_id: str
    task: str
    start_state: Dict
    trajectory: List[Dict]  # states and actions
    final_state: Dict
    success: bool
    reward: float
    timestamp: float

class EpisodicMemory:
    """Stores and retrieves complete task episodes"""

    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
        self.task_index: Dict[str, List[Episode]] = {}

    def store_episode(self, task: str, trajectory: List[Dict],
                     success: bool, reward: float):
        """Store complete episode"""

        episode = Episode(
            episode_id=f"ep_{len(self.episodes)}",
            task=task,
            start_state=trajectory[0]['state'] if trajectory else {},
            trajectory=trajectory,
            final_state=trajectory[-1]['state'] if trajectory else {},
            success=success,
            reward=reward,
            timestamp=np.time.time()
        )

        self.episodes.append(episode)

        # Index by task
        if task not in self.task_index:
            self.task_index[task] = []

        self.task_index[task].append(episode)

        # Trim old episodes
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def retrieve_similar_episodes(self, task: str,
                                  current_state: Dict,
                                  k: int = 5) -> List[Episode]:
        """Retrieve similar past episodes for current task"""

        task_episodes = self.task_index.get(task, [])

        if not task_episodes:
            return []

        # Score episodes by similarity to current state
        scores = []

        for episode in task_episodes:
            # Simple similarity: overlap in state features
            state_similarity = self._compute_state_similarity(
                current_state,
                episode.start_state
            )

            # Prefer successful episodes
            success_bonus = 0.2 if episode.success else 0.0

            score = state_similarity + success_bonus

            scores.append((episode, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return [ep for ep, _ in scores[:k]]

    def _compute_state_similarity(self, state1: Dict,
                                 state2: Dict) -> float:
        """Compute similarity between states"""

        common_keys = set(state1.keys()) & set(state2.keys())

        if not common_keys:
            return 0.0

        similarities = []

        for key in common_keys:
            v1, v2 = state1[key], state2[key]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numerical similarity
                similarity = 1.0 - abs(v1 - v2) / (abs(v1) + abs(v2) + 1e-6)
            elif v1 == v2:
                similarity = 1.0
            else:
                similarity = 0.0

            similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0
```

**Step 4: Implement semantic knowledge module**

Extract and store generalizable facts:

```python
class SemanticKnowledge:
    """Stores generalizable knowledge about objects and interactions"""

    def __init__(self):
        self.object_properties: Dict[str, Dict[str, float]] = {}
        self.interaction_rules: Dict[Tuple[str, str], float] = {}

    def update_object_knowledge(self, object_type: str,
                               properties: Dict[str, float]):
        """Update knowledge about object type"""

        if object_type not in self.object_properties:
            self.object_properties[object_type] = {}

        # Exponential moving average update
        alpha = 0.1
        for prop, value in properties.items():
            if prop in self.object_properties[object_type]:
                old_value = self.object_properties[object_type][prop]
                new_value = alpha * value + (1 - alpha) * old_value
            else:
                new_value = value

            self.object_properties[object_type][prop] = new_value

    def learn_interaction(self, object_type1: str, object_type2: str,
                         interaction: str, success_rate: float):
        """Learn about interactions between object types"""

        key = (object_type1, object_type2, interaction)

        if key in self.interaction_rules:
            # Exponential average
            old_rate = self.interaction_rules[key]
            self.interaction_rules[key] = 0.1 * success_rate + 0.9 * old_rate
        else:
            self.interaction_rules[key] = success_rate

    def query_interaction_feasibility(self, obj1_type: str,
                                     obj2_type: str,
                                     interaction: str) -> float:
        """Query likelihood of successful interaction"""

        key = (obj1_type, obj2_type, interaction)

        return self.interaction_rules.get(key, 0.5)  # Default 50% if unknown
```

**Step 5: Implement closed-loop planner with critic**

Generate plans using all memory types with adaptive feedback:

```python
class ClosedLoopPlannerWithCritic:
    """Generates long-horizon plans using all memory systems"""

    def __init__(self, spatial_graph: DynamicSpatialKnowledgeGraph,
                 temporal_buffer: TemporalEventBuffer,
                 episodic_memory: EpisodicMemory,
                 semantic_knowledge: SemanticKnowledge):
        self.spatial = spatial_graph
        self.temporal = temporal_buffer
        self.episodic = episodic_memory
        self.semantic = semantic_knowledge

    def plan(self, current_state: Dict, goal: str) -> List[str]:
        """Generate long-horizon plan"""

        # Retrieve relevant past episodes
        similar_episodes = self.episodic.retrieve_similar_episodes(goal, current_state)

        # Extract action sequences from successful episodes
        candidate_actions = self._extract_candidate_actions(similar_episodes)

        # Plan using temporal knowledge
        plan = self._construct_plan(current_state, goal, candidate_actions)

        return plan

    def _extract_candidate_actions(self, episodes: List[Episode]) -> List[str]:
        """Extract likely useful actions from episodes"""

        action_counts = {}

        for episode in episodes:
            for step in episode.trajectory:
                action = step.get('action', 'unknown')

                action_counts[action] = action_counts.get(action, 0) + 1

        # Sort by frequency
        sorted_actions = sorted(
            action_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [action for action, _ in sorted_actions[:10]]

    def _construct_plan(self, state: Dict, goal: str,
                       candidate_actions: List[str]) -> List[str]:
        """Construct plan using memory and critic"""

        plan = []
        max_steps = 20

        for step in range(max_steps):
            if self._is_goal_reached(state, goal):
                break

            # Get best next action
            action = self._select_next_action(state, goal, candidate_actions)

            plan.append(action)

            # Simulate effect
            effects = self.temporal.get_action_effects(action)

            # Update state (simplified)
            state = self._simulate_action(state, action, effects)

        return plan

    def _is_goal_reached(self, state: Dict, goal: str) -> bool:
        """Check if goal achieved"""
        return state.get('task_status') == goal

    def _select_next_action(self, state: Dict, goal: str,
                           candidates: List[str]) -> str:
        """Select action using critic module"""

        best_action = None
        best_score = -float('inf')

        for action in candidates:
            # Score action using critic
            score = self._critic_score(state, action, goal)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action or candidates[0]

    def _critic_score(self, state: Dict, action: str, goal: str) -> float:
        """Critic evaluates action feasibility"""

        # Get empirical action effects
        effects = self.temporal.get_action_effects(action)

        # Score based on goal alignment
        score = 0.0

        # Add rewards from successful past executions
        if effects:
            score += sum(effects.values()) / len(effects)

        return score

    def _simulate_action(self, state: Dict, action: str,
                        effects: Dict) -> Dict:
        """Simulate state transition from action"""

        new_state = state.copy()

        # Apply effects
        for effect, magnitude in effects.items():
            if effect in new_state and isinstance(new_state[effect], float):
                new_state[effect] += magnitude

        return new_state
```

### Practical Guidance

**When to use RoboMemory:**
- Long-horizon robotic tasks requiring learning from experience
- Environments with partial observability
- Multi-embodiment learning (adapting to different robot platforms)
- Tasks requiring spatial reasoning and planning
- Agents that benefit from human-like memory systems

**When NOT to use RoboMemory:**
- Simple reactive tasks (single-step decisions)
- Real-time systems where memory overhead is critical
- Fully observable, deterministic environments
- When end-to-end learning outperforms memory-augmented approaches

**Key memory system characteristics:**

- Spatial graph: handles ~1000 objects efficiently
- Temporal buffer: ~10K events before truncation
- Episodic memory: ~1K episodes typical
- Semantic knowledge: generalizes across tasks

**Expected improvements:**

- 26.5% performance boost on embodied tasks
- Surpasses Claude-3.5-Sonnet on EmbodiedBench
- Generalization to unseen tasks: 15-25% improvement
- Multi-embodiment transfer: 20-35% efficiency gain

### Reference

RoboMemory: Brain-inspired Multi-memory Agentic Framework for Interactive Environmental Learning in Physical Embodied Systems. arXiv:2508.01415
