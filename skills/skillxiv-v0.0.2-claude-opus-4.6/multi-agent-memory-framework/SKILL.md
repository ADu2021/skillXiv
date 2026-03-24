---
name: multi-agent-memory-framework
title: "BMAM: Brain-inspired Multi-Agent Memory Framework"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.20465"
keywords: [multi-agent, memory-management, neural-architecture, agent-coordination, shared-state]
description: "Design multi-agent systems with brain-inspired memory mechanisms that enable efficient information sharing and coordination. Implement hierarchical memory structures (working memory, episodic memory, semantic memory) similar to neuroscience models to improve multi-agent reasoning, planning, and task completion."
---

## Problem

Multi-agent systems struggle with inefficient information sharing and coordination. Agents often duplicate computational effort or fail to leverage collective knowledge. Traditional approaches don't effectively balance shared memory (enabling coordination) with individual agent autonomy (enabling parallelism).

## Solution

Implement BMAM: a brain-inspired memory framework that structures multi-agent memory hierarchically:

1. **Working Memory**: Short-term, high-capacity state shared between agents for immediate coordination
2. **Episodic Memory**: Persistent records of agent interactions, decisions, and outcomes
3. **Semantic Memory**: Abstracted knowledge and learned patterns shared across the team
4. **Memory Consolidation**: Mechanism for moving information between memory types based on relevance and frequency

This approach mirrors neuroscience models while enabling efficient multi-agent coordination.

## When to Use

- Multi-agent systems performing collaborative tasks (research, planning, problem-solving)
- Scenarios requiring shared knowledge while maintaining agent specialization
- Long-running agent systems needing persistent learning
- Complex tasks requiring both coordination and parallel execution
- Systems where agents build on each other's discoveries

## When NOT to Use

- Single-agent systems (overhead not justified)
- Scenarios with strict memory/latency constraints
- Competitive or adversarial multi-agent settings
- Tasks requiring immediate responses without consolidation

## Implementation

### Step 1: Design the Memory Architecture

Create hierarchical memory structures inspired by cognitive science.

```python
class BrainInspiredMemory:
    """
    Hierarchical memory system for multi-agent coordination
    Inspired by working, episodic, and semantic memory in neuroscience
    """

    def __init__(self, num_agents):
        # Working Memory: High-capacity, short-lived state
        self.working_memory = {
            "current_observations": {},  # Latest observations from each agent
            "recent_actions": deque(maxlen=100),  # Last 100 actions across agents
            "shared_goals": [],
            "active_subtasks": {}
        }

        # Episodic Memory: Historical records of agent interactions
        self.episodic_memory = {
            "interaction_history": [],  # (agent_a, agent_b, action, outcome)
            "decision_outcomes": [],    # (agent, decision, outcome, success)
            "problem_solutions": {}     # problem_id -> successful_solutions
        }

        # Semantic Memory: Abstracted knowledge and learned patterns
        self.semantic_memory = {
            "agent_capabilities": {},  # agent_id -> capabilities
            "task_strategies": {},     # task_type -> effective_strategies
            "learned_relationships": {},  # entity -> related_entities
            "domain_knowledge": {}     # abstracted domain facts
        }

        self.num_agents = num_agents
        self.consolidation_counter = 0

    def record_observation(self, agent_id, observation):
        """Add agent observation to working memory"""
        self.working_memory["current_observations"][agent_id] = {
            "data": observation,
            "timestamp": time.time()
        }

    def record_action(self, agent_id, action, result):
        """Log action execution"""
        action_record = {
            "agent": agent_id,
            "action": action,
            "result": result,
            "timestamp": time.time()
        }
        self.working_memory["recent_actions"].append(action_record)

    def record_interaction(self, agent_a, agent_b, action, outcome):
        """Log multi-agent interaction"""
        interaction = {
            "agents": (agent_a, agent_b),
            "action": action,
            "outcome": outcome,
            "timestamp": time.time()
        }
        self.episodic_memory["interaction_history"].append(interaction)

    def get_agent_working_context(self, agent_id):
        """Retrieve relevant working memory for an agent"""
        context = {
            "own_observation": self.working_memory["current_observations"].get(agent_id),
            "recent_actions": list(self.working_memory["recent_actions"])[-10:],
            "shared_goals": self.working_memory["shared_goals"],
            "relevant_subtasks": self.working_memory["active_subtasks"].get(agent_id, [])
        }
        return context
```

### Step 2: Implement Memory Consolidation

Move information from working to episodic to semantic memory based on relevance.

```python
class MemoryConsolidation:
    """Manage information movement through memory hierarchy"""

    def consolidate_memory(self, memory_system):
        """
        Periodically consolidate working memory to episodic/semantic
        Similar to sleep-based memory consolidation in brains
        """
        # Step 1: Extract decision patterns from recent actions
        recent_actions = list(memory_system.working_memory["recent_actions"])

        decision_patterns = self.extract_decision_patterns(recent_actions)

        # Step 2: Identify frequently successful action sequences
        successful_sequences = self.identify_frequent_patterns(
            memory_system.episodic_memory["decision_outcomes"],
            min_success_rate=0.7,
            min_frequency=3
        )

        # Step 3: Move successful patterns to semantic memory (learned strategies)
        for sequence, success_rate in successful_sequences:
            task_type = self.infer_task_type(sequence)
            if task_type not in memory_system.semantic_memory["task_strategies"]:
                memory_system.semantic_memory["task_strategies"][task_type] = []

            memory_system.semantic_memory["task_strategies"][task_type].append({
                "strategy": sequence,
                "success_rate": success_rate,
                "learned_at": time.time()
            })

        # Step 4: Extract agent capability profiles
        for agent_id in range(memory_system.num_agents):
            capabilities = self.extract_agent_capabilities(
                memory_system.episodic_memory["decision_outcomes"],
                agent_id
            )
            memory_system.semantic_memory["agent_capabilities"][agent_id] = capabilities

        # Step 5: Prune old entries from working memory
        memory_system.working_memory["recent_actions"] = deque(
            list(memory_system.working_memory["recent_actions"])[-50:],
            maxlen=100
        )

    def extract_decision_patterns(self, actions):
        """Find repeated decision patterns"""
        patterns = {}
        for action in actions:
            action_type = action["action"]["type"]
            if action_type not in patterns:
                patterns[action_type] = 0
            patterns[action_type] += 1

        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)

    def identify_frequent_patterns(self, decision_outcomes, min_success_rate=0.7, min_frequency=3):
        """Extract successful action sequences that should become learned strategies"""
        sequence_success = {}

        for outcome in decision_outcomes:
            decision = outcome["decision"]
            success = outcome["success"]

            # Convert decision to sequence representation
            seq_key = tuple(decision) if isinstance(decision, list) else (decision,)

            if seq_key not in sequence_success:
                sequence_success[seq_key] = {"success": 0, "total": 0}

            sequence_success[seq_key]["total"] += 1
            if success:
                sequence_success[seq_key]["success"] += 1

        # Filter by frequency and success rate
        frequent_patterns = [
            (seq, data["success"] / data["total"])
            for seq, data in sequence_success.items()
            if data["total"] >= min_frequency and (data["success"] / data["total"]) >= min_success_rate
        ]

        return frequent_patterns

    def extract_agent_capabilities(self, decision_outcomes, agent_id):
        """Build capability profile for an agent"""
        agent_outcomes = [
            o for o in decision_outcomes
            if o["agent"] == agent_id
        ]

        capabilities = {}
        for outcome in agent_outcomes:
            task_type = self.infer_task_type(outcome["decision"])
            if task_type not in capabilities:
                capabilities[task_type] = {"success": 0, "total": 0}

            capabilities[task_type]["total"] += 1
            if outcome["success"]:
                capabilities[task_type]["success"] += 1

        # Convert to success rates
        return {
            task: data["success"] / data["total"]
            for task, data in capabilities.items()
            if data["total"] >= 2
        }
```

### Step 3: Implement Semantic Memory for Learned Knowledge

Store and retrieve learned patterns and relationships.

```python
class SemanticMemoryManager:
    """Manage semantic/abstract knowledge across agents"""

    def __init__(self, semantic_memory):
        self.semantic_memory = semantic_memory

    def get_best_strategy_for_task(self, task_type, agent_capabilities=None):
        """
        Retrieve learned strategy for a task, optionally filtered by agent capability
        """
        if task_type not in self.semantic_memory["task_strategies"]:
            return None

        strategies = self.semantic_memory["task_strategies"][task_type]
        strategies.sort(key=lambda s: s["success_rate"], reverse=True)

        if agent_capabilities:
            # Prefer strategies the agent is good at
            best_fit = None
            for strategy in strategies:
                if agent_capabilities.get(task_type, 0) > 0.5:
                    best_fit = strategy
                    break
            return best_fit or strategies[0]

        return strategies[0]

    def find_capable_agent(self, task_type, min_capability=0.6):
        """Find which agent is best suited for a task"""
        capabilities = self.semantic_memory["agent_capabilities"]

        best_agent = None
        best_score = min_capability

        for agent_id, agent_caps in capabilities.items():
            score = agent_caps.get(task_type, 0)
            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    def get_related_knowledge(self, concept):
        """Find related domain knowledge"""
        if concept in self.semantic_memory["learned_relationships"]:
            return self.semantic_memory["learned_relationships"][concept]

        return []
```

### Step 4: Coordinate Multi-Agent Actions Using Memory

Use memory to guide agent coordination.

```python
class MultiAgentCoordinator:
    """Use brain-inspired memory for agent coordination"""

    def __init__(self, memory_system):
        self.memory = memory_system
        self.semantic_manager = SemanticMemoryManager(memory_system.semantic_memory)

    def assign_task_to_capable_agent(self, task_type):
        """
        Find best agent for task using semantic memory
        """
        capable_agent = self.semantic_manager.find_capable_agent(task_type, min_capability=0.5)

        if not capable_agent:
            # Fall back to first available agent
            capable_agent = 0

        return capable_agent

    def get_learned_strategy_for_agent(self, agent_id, task_type):
        """
        Provide agent with learned strategy for task
        """
        agent_capabilities = self.memory.semantic_memory["agent_capabilities"].get(agent_id, {})

        strategy = self.semantic_manager.get_best_strategy_for_task(
            task_type,
            agent_capabilities
        )

        return strategy

    def record_team_decision(self, agents_involved, decision, outcome):
        """
        Log multi-agent collaborative decision
        """
        for agent_id in agents_involved:
            self.memory.record_action(agent_id, decision, outcome)

        # Cross-agent interaction
        if len(agents_involved) > 1:
            for i in range(len(agents_involved) - 1):
                self.memory.record_interaction(
                    agents_involved[i],
                    agents_involved[i+1],
                    decision,
                    outcome
                )

    def coordinate_multi_step_task(self, agents, task_sequence):
        """
        Execute complex task requiring multiple agents
        Use memory for coordination
        """
        results = []

        for step_idx, task in enumerate(task_sequence):
            # Find best agent using semantic memory
            assigned_agent = self.assign_task_to_capable_agent(task["type"])

            # Provide learned strategy if available
            strategy = self.get_learned_strategy_for_agent(assigned_agent, task["type"])

            # Execute with strategy
            outcome = agents[assigned_agent].execute(task, strategy)
            results.append(outcome)

            # Record interaction
            self.record_team_decision([assigned_agent], task, outcome)

            # Consolidate memory periodically
            if step_idx % 10 == 0:
                consolidator = MemoryConsolidation()
                consolidator.consolidate_memory(self.memory)

        return results
```

## Key Neuroscience Insights

- **Working Memory**: Enables immediate coordination, limited capacity (~100 items)
- **Episodic Memory**: Records what happened, when, with whom - enables learning from history
- **Semantic Memory**: Abstracted facts and strategies - efficient knowledge representation
- **Consolidation**: Regularly move frequently-used episodic knowledge to semantic storage

## Benefits

- Agents can coordinate without direct communication (via shared memory)
- Learned strategies improve over time through consolidation
- Efficient knowledge reuse across multiple agents
- Scalable: adding agents doesn't require retraining

## References

- arXiv:2601.20465: BMAM brain-inspired multi-agent memory framework
- Based on neuroscience models of human working, episodic, and semantic memory
