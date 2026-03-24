---
name: multiagent-communication
title: "Thought Communication in Multiagent Collaboration"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.20733"
keywords: [multiagent systems, latent communication, thought sharing, collaboration, information theory]
description: "Enable agents to communicate through shared latent thoughts rather than natural language, recovering both shared and private latent representations with theoretical guarantees for more efficient collaboration."
---

# Technique: Latent Thought Communication — Direct Mind-to-Mind Agent Interaction

Traditional multi-agent systems have agents exchange information through natural language, but language is lossy and indirect. Thought Communication enables agents to share internal latent representations (thoughts) directly, bypassing the information bottleneck of language.

The theoretical framework shows that both shared thoughts (beneficial for coordination) and private thoughts (individual reasoning) can be identified and recovered with guarantees. This enables agents to collaborate more efficiently than natural language allows, particularly for complex reasoning tasks where misunderstandings are costly.

## Core Concept

Thought Communication operates on three principles:
- **Latent Extraction**: Before communication, extract internal thoughts from each agent
- **Thought Assignment**: Share relevant thoughts with agents that need them
- **Structure Recovery**: Identify the global sharing pattern (which agents share which thoughts)
- **Theoretical Guarantees**: Under mild assumptions, uniquely identify shared/private latent factors

The insight is that agent communication can happen at the level of internal representations, not just outputs. This reduces redundancy and misunderstanding.

## Architecture Overview

- **Thought Extractor**: Access/extract latent representations from agent models
- **Relevance Detector**: Determine which thoughts are relevant to which agents
- **Communication Channel**: Transmit thought vectors between agents
- **Thought Integrator**: Agents incorporate received thoughts into their reasoning
- **Structure Learner**: Identify global thought-sharing topology
- **Theoretical Validator**: Verify identifiability of shared/private factors

## Implementation Steps

The core algorithm extracts latents, identifies sharing patterns, and enables structured communication. This example shows latent extraction and communication.

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class ThoughtExtractor:
    """
    Extract latent thoughts from agent internal states.
    """

    def __init__(self, model):
        self.model = model

    def extract_thoughts(self, input_data) -> torch.Tensor:
        """
        Extract latent representation before output layer.
        Args: input_data (observations, context)
        Returns: latent_thoughts (hidden_dim,)
        """
        # Get intermediate layer activations
        with torch.no_grad():
            # Forward through model, capturing hidden state
            hidden_states = []

            def hook_fn(module, input, output):
                hidden_states.append(output)

            # Register hook on model's latent layer
            hook = self.model.latent_layer.register_forward_hook(hook_fn)

            # Forward pass
            _ = self.model(input_data)

            # Remove hook
            hook.remove()

        # Extract latent thought (usually the hidden state before output)
        latent_thought = hidden_states[0]
        return latent_thought


class ThoughtCommunicationGraph:
    """
    Manage shared and private latent thoughts across agents.
    """

    def __init__(self, num_agents: int, latent_dim: int):
        self.num_agents = num_agents
        self.latent_dim = latent_dim

        # Shared thoughts: common to multiple agents
        self.shared_thoughts = nn.Parameter(torch.randn(10, latent_dim) * 0.01)

        # Private thoughts: specific to each agent
        self.private_thoughts = nn.ParameterList([
            nn.Parameter(torch.randn(5, latent_dim) * 0.01)
            for _ in range(num_agents)
        ])

        # Sharing matrix: which agents share which thoughts
        self.sharing_matrix = nn.Parameter(
            torch.bernoulli(0.5 * torch.ones(num_agents, 10))
        )  # (num_agents, num_shared)

    def get_agent_thoughts(self, agent_id: int) -> torch.Tensor:
        """
        Get combined representation: shared + private thoughts for agent.
        """
        # Shared thoughts this agent uses
        shared_mask = self.sharing_matrix[agent_id]
        shared_for_agent = (self.shared_thoughts * shared_mask.unsqueeze(1)).sum(0)

        # Private thoughts
        private_for_agent = self.private_thoughts[agent_id].sum(0)

        # Combined thought vector
        combined = shared_for_agent + private_for_agent
        return combined

    def receive_thought(
        self,
        agent_id: int,
        incoming_thought: torch.Tensor,
        source_agent_id: int
    ):
        """
        Agent receives thought from another agent.
        Integrate received thought into internal state.
        """
        # Check if this is a valid shared thought (source agent uses it)
        if self.sharing_matrix[source_agent_id].sum() > 0:
            # Incorporate incoming thought with learned weight
            integration_weight = 0.3  # Tune based on task
            self.private_thoughts[agent_id][-1] = (
                (1 - integration_weight) * self.private_thoughts[agent_id][-1] +
                integration_weight * incoming_thought
            )


class MultiagentCollaborationWithThoughtComm:
    """
    Multi-agent team where agents share latent thoughts.
    """

    def __init__(self, agents: List[nn.Module], num_shared_thoughts: int = 10):
        self.agents = agents
        self.num_agents = len(agents)
        self.comm_graph = ThoughtCommunicationGraph(
            num_agents=self.num_agents,
            latent_dim=768  # Typical hidden dim
        )
        self.extractors = [ThoughtExtractor(agent) for agent in agents]

    def collaborative_step(
        self,
        observations: List[torch.Tensor],
        task_context: str
    ) -> List[torch.Tensor]:
        """
        Execute one step of multi-agent collaboration with thought sharing.
        """
        # Phase 1: Each agent thinks (extract latent thoughts)
        agent_thoughts = []
        for agent_id, (agent, obs) in enumerate(zip(self.agents, observations)):
            thought = self.extractors[agent_id].extract_thoughts(obs)
            agent_thoughts.append(thought)

        # Phase 2: Share relevant thoughts
        for agent_id in range(self.num_agents):
            # Determine which other agents this agent should share with
            relevant_agents = self._find_relevant_agents(agent_id)

            for target_agent_id in relevant_agents:
                # Share thought with target agent
                self.comm_graph.receive_thought(
                    agent_id=target_agent_id,
                    incoming_thought=agent_thoughts[agent_id],
                    source_agent_id=agent_id
                )

        # Phase 3: Each agent acts based on own + received thoughts
        actions = []
        for agent_id, agent in enumerate(self.agents):
            # Get augmented thought (own + shared)
            augmented_thought = self.comm_graph.get_agent_thoughts(agent_id)

            # Generate action
            action = agent.generate_action(augmented_thought)
            actions.append(action)

        return actions

    def _find_relevant_agents(self, source_agent_id: int) -> List[int]:
        """
        Determine which agents should receive thoughts from source agent.
        Based on task context and thought relevance.
        """
        # Simple heuristic: agents with high similarity to source
        source_private = self.comm_graph.private_thoughts[source_agent_id].mean(0)

        similarities = []
        for target_id in range(self.num_agents):
            if target_id != source_agent_id:
                target_private = self.comm_graph.private_thoughts[target_id].mean(0)
                sim = torch.cosine_similarity(
                    source_private.unsqueeze(0),
                    target_private.unsqueeze(0)
                )
                similarities.append(sim.item())
            else:
                similarities.append(-float('inf'))

        # Share with top-k relevant agents
        top_k = 2
        relevant = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )[:top_k]

        return relevant

    def learn_communication_structure(self, trajectories: List[Dict]) -> Dict:
        """
        Analyze trajectories to identify which agents should share which thoughts.
        Uses mutual information to recover the sharing structure.
        """
        # Compute mutual information between agent pairs
        mi_matrix = torch.zeros(self.num_agents, self.num_agents)

        for trajectory in trajectories:
            # MI[i, j] = H(thought_i) + H(thought_j) - H(thought_i, thought_j)
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    thought_i = trajectory[f"thought_{i}"]
                    thought_j = trajectory[f"thought_{j}"]

                    # Estimate entropy (simplified)
                    mi_ij = mutual_information(thought_i, thought_j)
                    mi_matrix[i, j] = mi_ij
                    mi_matrix[j, i] = mi_ij

        # Identify significant connections (mutual information above threshold)
        threshold = mi_matrix.mean() + mi_matrix.std()
        significant_pairs = (mi_matrix > threshold).nonzero(as_tuple=True)

        return {
            "mi_matrix": mi_matrix,
            "significant_pairs": significant_pairs,
            "sharing_structure": mi_matrix > threshold
        }
```

The theoretical contribution is proving that with sufficient interaction data, shared and private latent factors can be uniquely identified. This enables discovering "what should be shared" automatically.

## Practical Guidance

| Scenario | Language Overhead | Thought Comm Overhead | Win |
|----------|---|---|---|
| Math coordination | -30% efficiency | Direct latent | +25% |
| Complex reasoning | -40% clarity loss | Perfect transfer | +35% |
| Simple tasks | Minimal | Overhead | No win |

**When to Use:**
- Multi-agent systems with complex interdependencies
- Latent representations matter (not purely discrete actions)
- Agents have similar architectures/training
- You want to analyze collaboration structure theoretically

**When NOT to Use:**
- Heterogeneous agent architectures (different latent spaces)
- Simple coordination tasks (language sufficient)
- Agents with non-differentiable/discrete outputs
- Interpretability required (latent communication less transparent)

**Common Pitfalls:**
- Sharing too much → information overload, agents confused
- Sharing only numerical data without semantic context → loss of meaning
- Not normalizing thought vectors → dimension mismatch across agents
- Assuming all agents benefit from same shared thoughts (learn selective sharing)
- Ignoring private thoughts → agents lose individual reasoning capability

## Reference

[Thought Communication in Multiagent Collaboration](https://arxiv.org/abs/2510.20733)
