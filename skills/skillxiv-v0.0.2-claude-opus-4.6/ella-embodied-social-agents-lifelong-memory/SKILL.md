---
name: ella-embodied-social-agents-lifelong-memory
title: "Ella: Embodied Social Agents with Lifelong Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.24019"
keywords: [Embodied AI, Social Agents, Long-term Memory, Knowledge Graphs, Multi-agent Coordination]
description: "Enable autonomous embodied agents to function in 3D communities with structured memory systems. Combines semantic memory (scene graphs, knowledge graphs) and episodic memory (spatiotemporal experiences) for social intelligence and multi-agent coordination."
---

# Ella: Teaching Agents to Remember and Socialize

Robots and embodied AI agents in shared spaces must do more than perceive and act—they must remember interactions, understand social dynamics, and coordinate with other agents over time. Current embodied AI systems operate in narrow spatial domains and brief time horizons. Ella addresses this by implementing structured memory systems that enable agents to learn continuously from their environment, maintain social relationships, and coordinate complex group activities over extended periods.

The key innovation is separating memory into two complementary systems: semantic memory (what agents know about places and facts) and episodic memory (what they've personally experienced). This combination enables agents to understand context globally while reasoning about specific past interactions, supporting both individual autonomy and social coordination.

## Core Concept

Embodied agents in shared 3D worlds face unique challenges: spatial understanding, temporal reasoning about past interactions, and social reasoning about other agents' goals and beliefs. Ella solves this through:

1. **Semantic Memory**: Hierarchical scene graphs and knowledge graphs that organize spatial and relational information, enabling agents to understand "where am I and what does this place mean"

2. **Episodic Memory**: A spatiotemporal database of experienced events—what happened, where, and when—indexed by location and time for efficient retrieval

3. **Integration with Foundation Models**: Connecting these memory systems with GPT-4 enables natural language reasoning about stored knowledge and planning future actions

This enables agents to generate daily schedules, react appropriately to observed events, and coordinate with other agents—capabilities impossible with single-frame perception or stateless language models.

## Architecture Overview

The Ella system consists of these key components:

- **Scene Graph Memory**: Hierarchical representation of 3D environments (objects, rooms, buildings, city structure)
- **Knowledge Graph**: Structured facts about entities and relationships (who lives where, what activities happen when)
- **Spatiotemporal Episodic Memory**: Database of multimodal events indexed by location, time, and semantic content
- **Memory Retrieval Engine**: Queries past experiences based on current context (location-based, time-based, semantic similarity)
- **Foundation Model Integration**: GPT-4 interface for reasoning over memory to generate actions, plans, and social responses
- **Multi-Agent Coordination Layer**: Sharing relevant memories and intentions with other agents for collaborative tasks
- **Evaluation Benchmarks**: "Influence Battle" (social persuasion) and "Leadership Quest" (group coordination)

## Implementation

This section demonstrates how to implement Ella's memory systems and agent coordination.

**Step 1: Build semantic memory using scene graphs and knowledge graphs**

This code constructs structured representations of environments and facts:

```python
import networkx as nx
import json
from typing import Dict, List, Tuple

class SemanticMemory:
    """
    Structured knowledge about the environment: scene graphs and knowledge graphs.
    Scene graphs: spatial structure of 3D world
    Knowledge graphs: facts about entities and relationships
    """

    def __init__(self):
        self.scene_graph = nx.DiGraph()  # Spatial hierarchy
        self.knowledge_graph = nx.DiGraph()  # Facts and relationships

    def add_location(self, location_id: str, location_name: str, location_type: str, parent_id: str = None):
        """Add a location to the scene graph hierarchy."""
        self.scene_graph.add_node(location_id, name=location_name, type=location_type)
        if parent_id:
            self.scene_graph.add_edge(parent_id, location_id, relation="contains")

    def add_entity(self, entity_id: str, entity_name: str, entity_type: str, location_id: str):
        """Add an entity to the scene graph (object or agent)."""
        self.scene_graph.add_node(entity_id, name=entity_name, type=entity_type, location=location_id)

    def add_fact(self, subject_id: str, relation: str, object_id: str):
        """Add a knowledge graph edge (fact) connecting entities."""
        self.knowledge_graph.add_edge(subject_id, object_id, relation=relation)

    def query_location_hierarchy(self, location_id: str) -> Dict:
        """Query what's inside a location."""
        contents = list(self.scene_graph.successors(location_id))
        return {
            'location_id': location_id,
            'location_name': self.scene_graph.nodes[location_id].get('name'),
            'contents': contents
        }

    def query_entity_facts(self, entity_id: str) -> Dict:
        """Query all facts about an entity."""
        facts = []
        # Outgoing relations
        for target, data in self.scene_graph[entity_id].items():
            facts.append((entity_id, data.get('relation', ''), target))
        # Incoming relations
        for source in self.scene_graph.pred[entity_id]:
            data = self.scene_graph[source][entity_id]
            facts.append((source, data.get('relation', ''), entity_id))

        return facts

    def serialize(self) -> Dict:
        """Export as JSON for persistent storage."""
        return {
            'scene_graph': nx.node_link_data(self.scene_graph),
            'knowledge_graph': nx.node_link_data(self.knowledge_graph)
        }

# Build semantic memory for a city environment
semantic_mem = SemanticMemory()

# Add locations
semantic_mem.add_location("city_1", "Metropolis", "city")
semantic_mem.add_location("district_1", "Downtown", "district", parent_id="city_1")
semantic_mem.add_location("building_1", "Community Center", "building", parent_id="district_1")
semantic_mem.add_location("room_101", "Main Hall", "room", parent_id="building_1")

# Add agents
semantic_mem.add_entity("agent_alice", "Alice", "agent", location_id="room_101")
semantic_mem.add_entity("agent_bob", "Bob", "agent", location_id="room_101")

# Add facts
semantic_mem.add_fact("agent_alice", "likes", "agent_bob")
semantic_mem.add_fact("building_1", "hosts_events", "meetings")

print("Semantic Memory Built:")
print(semantic_mem.query_location_hierarchy("building_1"))
```

This creates structured spatial and relational knowledge about the world.

**Step 2: Implement spatiotemporal episodic memory**

This code stores and retrieves experienced events indexed by location and time:

```python
import numpy as np
from datetime import datetime
from typing import List, Dict
import faiss

class EpisodicMemory:
    """
    Stores multimodal experiences: what happened, where, and when.
    Indexed for efficient retrieval by location, time, and semantic content.
    """

    def __init__(self, embedding_dim=384):
        self.memories = []  # List of memory records
        self.embedding_dim = embedding_dim
        self.embedding_index = faiss.IndexFlatL2(embedding_dim)  # Semantic search
        self.embeddings = np.zeros((0, embedding_dim), dtype=np.float32)

    def store_event(
        self,
        timestamp: datetime,
        location_id: str,
        event_type: str,
        agents_involved: List[str],
        description: str,
        embedding: np.ndarray
    ):
        """Store a memory of an event."""

        memory_record = {
            'timestamp': timestamp,
            'location_id': location_id,
            'event_type': event_type,
            'agents_involved': agents_involved,
            'description': description,
            'embedding': embedding,
            'memory_id': len(self.memories)
        }

        self.memories.append(memory_record)

        # Add to semantic embedding index for similarity search
        self.embedding_index.add(embedding.reshape(1, -1))
        self.embeddings = np.vstack([self.embeddings, embedding])

    def retrieve_by_location(self, location_id: str, time_window_hours=24) -> List[Dict]:
        """Retrieve all memories from a location in recent time window."""
        recent_memories = []

        for mem in self.memories:
            if mem['location_id'] == location_id:
                time_diff = (datetime.now() - mem['timestamp']).total_seconds() / 3600
                if time_diff <= time_window_hours:
                    recent_memories.append(mem)

        return sorted(recent_memories, key=lambda x: x['timestamp'], reverse=True)

    def retrieve_by_similarity(self, query_embedding: np.ndarray, k=5) -> List[Dict]:
        """Retrieve semantically similar memories."""
        distances, indices = self.embedding_index.search(query_embedding.reshape(1, -1), k)

        similar_memories = [self.memories[idx] for idx in indices[0] if idx < len(self.memories)]
        return similar_memories

    def retrieve_by_agent_interaction(self, agent_id: str) -> List[Dict]:
        """Retrieve memories involving a specific agent."""
        agent_memories = [mem for mem in self.memories if agent_id in mem['agents_involved']]
        return agent_memories

# Create episodic memory
episodic_mem = EpisodicMemory()

# Store some events
now = datetime.now()
event_embedding = np.random.randn(384).astype(np.float32)

episodic_mem.store_event(
    timestamp=now,
    location_id="room_101",
    event_type="conversation",
    agents_involved=["agent_alice", "agent_bob"],
    description="Alice and Bob discussed the upcoming community event",
    embedding=event_embedding
)

episodic_mem.store_event(
    timestamp=now,
    location_id="room_101",
    event_type="activity",
    agents_involved=["agent_alice"],
    description="Alice prepared decorations for the event",
    embedding=np.random.randn(384).astype(np.float32)
)

# Retrieve memories
location_memories = episodic_mem.retrieve_by_location("room_101", time_window_hours=24)
print(f"Memories from room_101: {len(location_memories)} events")

agent_memories = episodic_mem.retrieve_by_agent_interaction("agent_alice")
print(f"Alice's memories: {len(agent_memories)} events")
```

This indexes and retrieves personal experiences for context-aware reasoning.

**Step 3: Integrate with foundation models for reasoning**

This code uses GPT-4 to reason over agent memories for planning and social interaction:

```python
import openai

class FoundationModelInterface:
    """
    Interface to GPT-4 for reasoning about memories and planning actions.
    """

    def __init__(self, semantic_mem: SemanticMemory, episodic_mem: EpisodicMemory):
        self.semantic_mem = semantic_mem
        self.episodic_mem = episodic_mem

    def generate_daily_schedule(self, agent_id: str, location_id: str) -> str:
        """Generate a daily schedule based on memory and environment."""

        # Retrieve relevant memories
        location_info = self.semantic_mem.query_location_hierarchy(location_id)
        agent_memories = self.episodic_mem.retrieve_by_agent_interaction(agent_id)

        # Format context for GPT-4
        memory_context = f"""
Agent: {agent_id}
Current location: {location_info['location_name']}

Recent activities:
{chr(10).join([f"- {mem['description']}" for mem in agent_memories[-5:]])}
"""

        prompt = f"""{memory_context}

Based on the agent's memories and current location, generate a realistic daily schedule for tomorrow.
Include 4-5 activities, considering what the agent likes to do and who they interact with regularly.
Format as a timeline."""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    def generate_social_response(self, agent_id: str, other_agent_id: str, context: str) -> str:
        """Generate socially appropriate response based on relationship history."""

        # Retrieve interaction history with this agent
        all_memories = self.episodic_mem.retrieve_by_agent_interaction(agent_id)
        interaction_memories = [m for m in all_memories if other_agent_id in m['agents_involved']]

        # Query knowledge graph for relationship info
        relationship_facts = self.semantic_mem.knowledge_graph[agent_id]

        # Format context
        interaction_context = f"""
Agent {agent_id} interacting with {other_agent_id}

Past interactions:
{chr(10).join([f"- {m['description']}" for m in interaction_memories[-3:]])}

Current context: {context}
"""

        prompt = f"""{interaction_context}

Generate a natural, socially appropriate response for {agent_id}. Consider the relationship history
and what we know about both agents' preferences and past interactions."""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300
        )

        return response.choices[0].message.content

# Use foundation model interface
fm_interface = FoundationModelInterface(semantic_mem, episodic_mem)

# Generate schedule
schedule = fm_interface.generate_daily_schedule("agent_alice", "room_101")
print(f"Generated schedule:\n{schedule}")

# Generate social response
response = fm_interface.generate_social_response(
    "agent_alice",
    "agent_bob",
    "Bob asks Alice if she wants to help organize the event"
)
print(f"\nAlice's response:\n{response}")
```

This uses foundation models to reason over structured memories.

**Step 4: Multi-agent coordination using shared memory**

This code enables agents to coordinate by sharing and reasoning about shared memories:

```python
class MultiAgentCoordinator:
    """
    Enables agents to coordinate by sharing memories and reasoning about joint goals.
    """

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.shared_memories = []  # Memories relevant to multiple agents
        self.agent_states = {agent: {'location': None, 'goal': None} for agent in agents}

    def add_shared_memory(self, memory_record: Dict, relevant_agents: List[str]):
        """Add memory relevant to multiple agents."""
        self.shared_memories.append({
            'memory': memory_record,
            'relevant_agents': relevant_agents,
            'broadcast_time': datetime.now()
        })

    def coordinate_group_task(self, task_description: str, agents_needed: List[str]) -> Dict:
        """Coordinate a task requiring multiple agents."""

        # Retrieve shared memories about these agents
        relevant_shared_mems = [
            sm for sm in self.shared_memories
            if any(agent in sm['relevant_agents'] for agent in agents_needed)
        ]

        # Use GPT-4 to create a coordination plan
        coordination_prompt = f"""
Task: {task_description}
Required agents: {', '.join(agents_needed)}

Shared recent events:
{chr(10).join([f"- {sm['memory']['description']}" for sm in relevant_shared_mems[-5:]])}

Create a coordination plan:
1. Assign roles to each agent
2. Specify actions and sequence
3. Note communication points
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": coordination_prompt}],
            temperature=0.7,
            max_tokens=600
        )

        plan = response.choices[0].message.content

        # Update agent states to reflect coordination
        for agent in agents_needed:
            self.agent_states[agent]['goal'] = task_description

        return {'plan': plan, 'participants': agents_needed}

# Create coordinator for multiple agents
coordinator = MultiAgentCoordinator(['agent_alice', 'agent_bob', 'agent_charlie'])

# Add shared memory about an upcoming event
event_mem = {
    'timestamp': datetime.now(),
    'event_type': 'group_event',
    'description': 'Community festival happening tomorrow at the downtown square'
}
coordinator.add_shared_memory(event_mem, ['agent_alice', 'agent_bob', 'agent_charlie'])

# Coordinate a task
plan = coordinator.coordinate_group_task(
    "Organize volunteers to help set up for the community festival",
    ['agent_alice', 'agent_bob', 'agent_charlie']
)
print(f"Coordination plan:\n{plan['plan']}")
```

This enables multi-agent coordination through shared memory systems.

## Practical Guidance

**When to use Ella's memory systems:**
- Building embodied agents that operate in shared 3D environments
- Multi-agent systems requiring social coordination and relationship reasoning
- Applications needing agents to learn and adapt from long-term interactions
- Systems where context (where am I, what happened here before) drives behavior
- Scenarios requiring persistent knowledge across agent lifespans

**When NOT to use:**
- Single-agent, short-horizon tasks without environmental complexity
- Systems with real-time constraints (memory operations add latency)
- Environments with minimal social interaction requirements
- Scenarios where stateless behavior is sufficient (web agents, API consumers)
- Massive agent swarms (memory management becomes prohibitive)

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Scene Graph Depth | 3-5 levels | City → District → Building → Room; more levels = more detail |
| Episodic Memory Capacity | 10,000-100,000 events | Larger for longer deployment; older events can be archived |
| Semantic Embedding Dim | 384 | Standard for moderate-quality similarity; 768 for higher precision |
| Memory Retrieval Window | 24-168 hours | One week for long-term pattern detection |
| Foundation Model Queries | 1-5 per action | Minimize API calls; batch reasoning when possible |
| Agent Update Frequency | 1-5 minutes | Less frequent for computational efficiency |

**Common Pitfalls:**
- Storing redundant memories (use similarity-based deduplication)
- Forgetting to age out old memories (memory bloat hurts retrieval)
- Over-relying on GPT-4 (expensive and slow for every decision)
- Not maintaining consistency between semantic and episodic memory
- Failing to validate that stored facts match actual agent behavior
- Insufficient spatial indexing (makes location-based retrieval slow)

**Key Design Decisions:**
Ella separates semantic memory (facts, structure) from episodic memory (experiences, events), mirroring human cognition. Scene graphs provide efficient spatial reasoning; knowledge graphs handle relational facts. Episodic memories are indexed by location, time, and semantic content for flexible retrieval. Foundation models reason over this structure, not replace it—enabling both efficient computation and powerful generalization.

## Reference

Gao, T., Xie, C., Nie, Y., Sun, Y., Dey, P., Tiwari, R., ... & Weller, A. (2025). Ella: Embodied Social Agents with Lifelong Memory. arXiv preprint arXiv:2506.24019. https://arxiv.org/abs/2506.24019
