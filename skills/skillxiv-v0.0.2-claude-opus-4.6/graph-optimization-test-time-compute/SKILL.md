---
name: graph-optimization-test-time-compute
title: "Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.00086"
keywords: [Test-Time Scaling, Graph Optimization, Multi-LLM Collaboration, Inference, Probabilistic Search]
description: "Optimize test-time computation through graph-based collaborative architecture where nodes represent models, edges represent information flow, and topology itself is optimizable via reinforcement learning to discover ideal model assignments and configurations."
---

# Title: Discover Optimal Model Collaboration Architectures Through Graph Search

Rather than assuming fixed LLM collaboration patterns, treat the entire architecture as a searchable space. Nodes represent LLM instances with assigned roles (assistant, fusion), edges represent information flow, and the framework uses reinforcement learning to search for configurations that maximize accuracy within latency budgets. This enables discovery of novel architectures: some tasks prefer deep sequential chains, others benefit from parallel specialists.

The approach generalizes test-time scaling beyond conventional ensembles.

## Core Concept

**Probabilistic Graph Optimization for Multi-LLM Systems**:
- **Nodes**: LLM computation units with role assignments (assistant, fusion) and model choices
- **Edges**: Information flow directions forming DAG structure
- **Topology Optimization**: Search over edge probabilities, role assignments, model selections
- **RL Optimization**: Agent-REINFORCE uses textual feedback as gradient signals
- **Adaptive Architecture**: Discover task-specific ideal configurations without human design

## Architecture Overview

- **Node Types**: Assistant (refines previous outputs), Fusion (aggregates multiple inputs)
- **Parameterization**: θ (topology), π (roles), ψ (models) form probabilistic graph
- **Search Algorithm**: Agent-REINFORCE iteratively samples and refines distributions
- **Feedback**: Task performance + latency feedback guides optimization
- **Historical Archive**: Record all tried configurations to guide future search

## Implementation Steps

**1. Model Collaboration as Probabilistic Graph**

Represent architecture as learnable probability distributions.

```python
class MultiLLMCollaborationGraph:
    def __init__(self, num_nodes=5, available_models=None):
        self.num_nodes = num_nodes
        self.available_models = available_models or ['gpt4', 'gpt3.5', 'claude']

        # Learnable parameters
        self.theta = nn.Parameter(torch.randn(num_nodes, num_nodes))  # Topology
        self.pi = nn.Parameter(torch.randn(num_nodes, 2))  # Role: assistant vs fusion
        self.psi = nn.Parameter(torch.randn(num_nodes, len(available_models)))  # Model assignment

    def sample_graph(self):
        """Sample a graph from learned distributions"""
        # Edge probabilities: sigmoid(theta)
        edge_probs = torch.sigmoid(self.theta)
        edges = (torch.rand_like(edge_probs) < edge_probs).float()

        # Role assignments: softmax(pi)
        role_probs = F.softmax(self.pi, dim=-1)
        roles = torch.argmax(role_probs, dim=-1)  # 0=assistant, 1=fusion

        # Model assignments: softmax(psi)
        model_probs = F.softmax(self.psi, dim=-1)
        models = torch.argmax(model_probs, dim=-1)

        # Ensure DAG structure
        edges = self._enforce_acyclicity(edges)

        return {
            'edges': edges,
            'roles': roles,
            'models': models
        }

    def _enforce_acyclicity(self, adjacency_matrix):
        """Ensure graph is a DAG"""
        # Topological ordering: process nodes in fixed order
        for i in range(self.num_nodes):
            adjacency_matrix[i, :i] = 0  # No backward edges
        return adjacency_matrix

    def execute_graph(self, query, graph):
        """Execute computation following graph topology"""
        edges, roles, models = graph['edges'], graph['roles'], graph['models']

        # Initialize node outputs
        outputs = {}
        outputs[0] = self._get_model(models[0]).generate(query)

        # Execute in topological order
        for node in range(1, self.num_nodes):
            if roles[node] == 0:  # Assistant
                # Refine previous best output
                pred_input = outputs.get(node - 1, outputs[0])
                outputs[node] = self._get_model(models[node]).refine(pred_input, query)
            else:  # Fusion
                # Aggregate inputs from predecessors
                predecessors = torch.nonzero(edges[:node, node]).squeeze(-1)
                if len(predecessors) > 0:
                    inputs_to_fuse = [outputs[p.item()] for p in predecessors]
                    outputs[node] = self._get_model(models[node]).fuse(inputs_to_fuse, query)

        # Final output from sink node
        return outputs[self.num_nodes - 1]

    def _get_model(self, model_idx):
        """Get LLM for given index"""
        return ModelRegistry.get(self.available_models[model_idx])
```

**2. Implement Agent-REINFORCE Search**

Use RL to optimize graph parameters.

```python
class AgentReinforceOptimizer:
    def __init__(self, graph_model, num_tasks=100):
        self.graph = graph_model
        self.archive = []  # Store tried configurations
        self.optimizer = torch.optim.Adam(graph_model.parameters(), lr=1e-4)
        self.num_tasks = num_tasks

    def optimize(self, tasks, num_iterations=100, budget_tokens=10000):
        """Optimize graph architecture using REINFORCE"""
        for iteration in range(num_iterations):
            # Sample multiple graphs
            graphs = [self.graph.sample_graph() for _ in range(3)]

            rewards_batch = []
            for graph in graphs:
                # Evaluate on sample of tasks
                task_rewards = []
                total_tokens = 0

                for task in random.sample(tasks, min(5, len(tasks))):
                    result = self.graph.execute_graph(task['query'], graph)
                    accuracy = self.evaluate_result(result, task['ground_truth'])
                    tokens_used = self._estimate_tokens(result)

                    # Reward: balance accuracy and efficiency
                    latency_penalty = max(0, (tokens_used - budget_tokens) / budget_tokens)
                    reward = accuracy - 0.1 * latency_penalty

                    task_rewards.append(reward)
                    total_tokens += tokens_used

                avg_reward = np.mean(task_rewards)
                rewards_batch.append(avg_reward)

                # Store in archive
                self.archive.append({
                    'graph': graph,
                    'reward': avg_reward,
                    'tokens': total_tokens / len(task_rewards)
                })

            # REINFORCE update: maximize expected reward
            rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32)
            baseline = rewards_tensor.mean()
            advantages = rewards_tensor - baseline

            # Compute log probabilities of sampled graphs
            log_probs = self._graph_log_prob(graphs)

            # Policy gradient
            loss = -(log_probs * advantages.detach()).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iteration % 10 == 0:
                best_reward = max(r['reward'] for r in self.archive)
                print(f"Iteration {iteration}: Best reward {best_reward:.3f}")

    def _graph_log_prob(self, graphs):
        """Compute log probability of graphs under current distribution"""
        log_probs = []
        for graph in graphs:
            # Log prob of topology
            edge_probs = torch.sigmoid(self.graph.theta)
            topology_log_prob = (graph['edges'] * torch.log(edge_probs) +
                                (1 - graph['edges']) * torch.log(1 - edge_probs)).sum()

            # Log prob of roles and models
            role_probs = F.softmax(self.graph.pi, dim=-1)
            model_probs = F.softmax(self.graph.psi, dim=-1)

            log_probs.append(topology_log_prob)

        return torch.stack(log_probs)
```

## Practical Guidance

**When to Use**:
- Test-time scaling with flexible latency budgets
- Diverse tasks with varying optimal architectures
- Scenarios where model ensemble is already planned

**Hyperparameters**:
- num_nodes: 3-8 (more nodes = larger search space)
- learning_rate: 1e-4 (conservative for stability)
- num_samples_per_iteration: 3-5 graphs

**When NOT to Use**:
- Single-model settings
- Strict latency constraints (search overhead)
- Tasks where single best model dominates

**Pitfalls**:
- **ACyclicity enforcement**: Topological ordering can be too restrictive
- **Inference cost**: Graph execution adds communication overhead
- **Sample efficiency**: RL requires many evaluations; expensive with large models

## Reference

arXiv: https://arxiv.org/abs/2511.00086
