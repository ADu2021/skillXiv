---
name: redsearcher-long-horizon-agents
title: "REDSearcher: Scalable Framework for Long-Horizon Search Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.14234"
keywords: [Information Seeking, Long-Horizon Planning, Reinforcement Learning, Search Agents, Tool Use]
description: "Train language models for multi-step information-seeking using dual-constrained task synthesis and cost-efficient staged learning. Generate complex queries by controlling topological complexity and information dispersion, then train atomic reasoning skills before introducing environment interaction. Achieve state-of-the-art on long-horizon search benchmarks with 30B parameter models."
---

# REDSearcher: Scalable Framework for Long-Horizon Search Agents

## Problem Context

Language models for information-seeking (web search, retrieval) face extreme sparsity: high-quality search trajectories are rare, multi-step reasoning is expensive, and real-world API costs prohibit extensive exploration. REDSearcher addresses this by (1) generating diverse, challenging synthetic tasks, (2) training fundamental skills in controlled environments, (3) scaling to real interaction only after skills solidify.

## Core Concept

REDSearcher operates in three phases: (1) dual-constrained task synthesis creating complex search problems, (2) atomic skill learning (grounding, planning) in simulation, (3) hierarchical trajectory training with environment interaction. This staged approach reduces expensive real-world costs while maintaining quality.

## Architecture Overview

- **Dual constraints**: Control task complexity (graph treewidth) and information dispersion
- **Tool-augmented synthesis**: Convert facts into tool-resolvable constraints (API calls)
- **Atomic skills**: Intent grounding, hierarchical planning, fact composition
- **Simulation environment**: Local search environment with millions of documents
- **Staged training**: Skills → hierarchical planning → environment interaction
- **Long-horizon support**: 30+ step trajectories with explicit reasoning

## Implementation

### Step 1: Dual-constrained task synthesis

```python
import networkx as nx
from typing import Dict, List, Tuple, Set
import random

class DualConstraintedTaskSynthesis:
    """Generate search tasks with controlled complexity."""

    def __init__(
        self,
        document_corpus: List[str],
        entity_graph: nx.DiGraph,
        max_treewidth: int = 5
    ):
        self.corpus = document_corpus
        self.entity_graph = entity_graph
        self.max_treewidth = max_treewidth

    def compute_treewidth(self, subgraph: nx.DiGraph) -> int:
        """Estimate graph treewidth (NP-hard; use heuristic)."""
        # Simplified: use degree-based approximation
        if len(subgraph.nodes) == 0:
            return 0
        degrees = [subgraph.degree(n) for n in subgraph.nodes]
        return max(degrees) if degrees else 0

    def synthesize_query_with_complexity(
        self,
        target_treewidth: int = 3,
        information_dispersion: float = 0.7
    ) -> Dict:
        """
        Generate query with target topological complexity.

        Args:
            target_treewidth: Desired graph complexity (higher = harder routing)
            information_dispersion: Fraction of facts in different documents (0-1)

        Returns:
            task: {question, gold_facts, required_searches, optimal_path}
        """
        # Sample entities to connect
        num_entities = min(target_treewidth + 2, len(self.entity_graph.nodes))
        selected_entities = random.sample(list(self.entity_graph.nodes), num_entities)

        # Create subgraph and verify treewidth
        subgraph = self.entity_graph.subgraph(selected_entities)
        actual_treewidth = self.compute_treewidth(subgraph)

        if actual_treewidth > self.max_treewidth:
            # Simplify by removing high-degree nodes
            high_degree_nodes = [n for n in subgraph.nodes if subgraph.degree(n) > 3]
            selected_entities = [e for e in selected_entities if e not in high_degree_nodes]

        # Generate gold facts from entity connections
        gold_facts = []
        for source, target in subgraph.edges:
            relation = self.entity_graph.edges[source, target].get('relation', 'connects_to')
            gold_facts.append(f"{source} {relation} {target}")

        # Distribute facts across documents (information dispersion)
        facts_per_doc = max(1, len(gold_facts) // max(1, int(len(self.corpus) * information_dispersion)))

        document_assignments = {}
        for doc_idx in range(len(self.corpus)):
            assigned_facts = gold_facts[
                doc_idx * facts_per_doc:(doc_idx + 1) * facts_per_doc
            ]
            if assigned_facts:
                document_assignments[doc_idx] = assigned_facts

        # Generate question requiring all facts
        question = self._generate_question(selected_entities, gold_facts)

        return {
            'question': question,
            'gold_facts': gold_facts,
            'document_assignments': document_assignments,
            'required_entities': selected_entities,
            'treewidth': actual_treewidth,
            'dispersion': information_dispersion
        }

    def _generate_question(self, entities: List[str], facts: List[str]) -> str:
        """Generate natural language question from entities and facts."""
        if len(entities) >= 2:
            return f"What is the connection between {entities[0]} and {entities[-1]}? " \
                   f"List all intermediate steps."
        return f"Describe the relationships for: {', '.join(entities)}"
```

### Step 2: Tool-augmented learning

```python
class ToolAugmentedConstraintRepresentation:
    """Convert facts into tool-resolvable constraints."""

    def __init__(self, available_tools: Dict[str, callable]):
        """
        Args:
            available_tools: Dict of tool_name -> tool_function
                           e.g., {search_web, get_entity_facts, route_query}
        """
        self.tools = available_tools

    def convert_fact_to_tool_constraint(
        self,
        fact: str
    ) -> Dict:
        """
        Convert fact into tool call constraint.

        Example:
            Fact: "Alice works at Acme Corp"
            Constraint: {tool: "get_entity_facts", entity: "Alice", expected: "Acme Corp"}
        """
        # Parse fact structure (simplified; would use NER in practice)
        parts = fact.split()

        if "connects" in fact.lower() or "relationship" in fact.lower():
            # Entity-relation-entity triple
            entity1, entity2 = parts[0], parts[-1]
            constraint = {
                'type': 'entity_relationship',
                'entity_1': entity1,
                'entity_2': entity2,
                'required_tool': 'get_entity_relationship',
                'expected_relation': fact
            }
        else:
            # Property fact
            entity = parts[0]
            property_val = ' '.join(parts[1:])
            constraint = {
                'type': 'entity_property',
                'entity': entity,
                'property': property_val,
                'required_tool': 'get_entity_facts',
                'expected_value': property_val
            }

        return constraint

    def create_tool_sequence(self, task: Dict) -> List[Dict]:
        """
        Create sequence of tool calls needed to resolve task.

        Returns:
            tool_sequence: List of {tool_name, arguments, expected_output}
        """
        tool_sequence = []

        for fact in task['gold_facts']:
            constraint = self.convert_fact_to_tool_constraint(fact)

            tool_call = {
                'tool': constraint['required_tool'],
                'arguments': {
                    k: v for k, v in constraint.items()
                    if k not in ['type', 'required_tool', 'expected_relation', 'expected_value']
                },
                'expected_output': constraint.get('expected_relation', constraint.get('expected_value'))
            }

            tool_sequence.append(tool_call)

        return tool_sequence
```

### Step 3: Atomic skill learning in simulation

```python
class AtomicSkillTrainer:
    """Train fundamental search skills in simulation."""

    def __init__(
        self,
        model,
        optimizer,
        simulated_environment: dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.env = simulated_environment

    def train_intent_grounding(
        self,
        tasks: List[Dict],
        num_steps: int = 1000
    ) -> float:
        """
        Train skill: ground natural language intent into API calls.

        Task: Given question, predict correct tool and arguments.
        """
        total_loss = 0.0

        for step in range(num_steps):
            task = random.choice(tasks)
            question = task['question']

            # Ground intent: predict first tool call
            tool_sequence = task['tool_sequence']
            if not tool_sequence:
                continue

            first_tool = tool_sequence[0]

            # Generate grounding
            grounding_prompt = f"Question: {question}\n\nPredicted next search:"
            predicted_grounding, log_probs = self.model.generate_with_logprobs(
                grounding_prompt, max_tokens=100
            )

            # Reward: does predicted grounding match first tool?
            is_correct = self._match_grounding(
                predicted_grounding,
                first_tool['tool']
            )

            reward = 1.0 if is_correct else 0.0

            # Policy gradient update
            loss = -log_probs.mean() * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_steps

    def train_hierarchical_planning(
        self,
        tasks: List[Dict],
        num_steps: int = 1000
    ) -> float:
        """
        Train skill: decompose multi-step questions into tool sequence.

        Task: Given question and facts, predict tool call order.
        """
        total_loss = 0.0

        for step in range(num_steps):
            task = random.choice(tasks)
            question = task['question']
            tool_sequence = task['tool_sequence']

            # Predict full tool sequence
            planning_prompt = f"Question: {question}\n\nPlan the search steps:"
            predicted_plan, log_probs = self.model.generate_with_logprobs(
                planning_prompt, max_tokens=200
            )

            # Reward: how many tools in correct order?
            num_correct = self._score_plan(predicted_plan, tool_sequence)
            reward = num_correct / max(1, len(tool_sequence))

            # Update
            loss = -log_probs.mean() * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_steps

    def _match_grounding(self, prediction: str, expected_tool: str) -> bool:
        """Check if prediction matches expected tool."""
        return expected_tool.lower() in prediction.lower()

    def _score_plan(self, prediction: str, expected_sequence: List[Dict]) -> int:
        """Count correctly ordered tools in prediction."""
        correct = 0
        for tool_dict in expected_sequence:
            if tool_dict['tool'] in prediction:
                correct += 1
        return correct
```

### Step 4: Environment interaction stage

```python
class LongHorizonSearchRL:
    """RL training with actual environment interaction."""

    def __init__(
        self,
        model,
        optimizer,
        environment,
        max_steps: int = 30
    ):
        self.model = model
        self.optimizer = optimizer
        self.env = environment
        self.max_steps = max_steps

    def generate_trajectory(
        self,
        task: Dict,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate trajectory by interacting with environment.

        Returns:
            trajectory: {question, steps, observations, reward, success}
        """
        question = task['question']
        trajectory = {
            'question': question,
            'steps': [],
            'observations': [],
            'success': False
        }

        current_state = question
        discovered_facts = set()

        for step in range(self.max_steps):
            # Predict next action
            prompt = f"Question: {question}\nCurrent findings: {current_state}\n\nNext action:"
            action, log_probs = self.model.generate_with_logprobs(
                prompt, max_tokens=50, temperature=temperature
            )

            trajectory['steps'].append({
                'action': action,
                'log_probs': log_probs
            })

            # Execute in environment
            observation = self.env.execute_action(action)
            trajectory['observations'].append(observation)

            # Check for success
            current_state = f"{current_state}\n{observation}"
            for fact in task['gold_facts']:
                if fact in observation:
                    discovered_facts.add(fact)

            if len(discovered_facts) == len(task['gold_facts']):
                trajectory['success'] = True
                break

        trajectory['num_facts_found'] = len(discovered_facts)
        trajectory['reward'] = 1.0 if trajectory['success'] else 0.5 * (len(discovered_facts) / len(task['gold_facts']))

        return trajectory

    def update_from_trajectory(self, trajectory: Dict):
        """Update policy from trajectory."""
        total_loss = 0.0

        for step in trajectory['steps']:
            log_probs = step['log_probs']
            # Discounted reward
            reward = trajectory['reward']

            loss = -log_probs.mean() * reward

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(trajectory['steps'])
```

### Step 5: Staged training pipeline

```python
def train_redsearcher_staged(
    model,
    corpus: List[str],
    entity_graph,
    optimizer,
    num_task_synthesis_iter: int = 500,
    num_skill_steps: int = 5000,
    num_env_steps: int = 10000,
    device: str = 'cuda'
):
    """
    Full REDSearcher training: synthesis → skills → environment.
    """
    # Stage 1: Task synthesis
    print("Stage 1: Task Synthesis")
    synthesizer = DualConstraintedTaskSynthesis(corpus, entity_graph)
    tool_converter = ToolAugmentedConstraintRepresentation({})

    tasks = []
    for i in range(num_task_synthesis_iter):
        # Vary complexity
        treewidth = (i % 5) + 1
        dispersion = 0.5 + 0.3 * (i % 10) / 10

        task = synthesizer.synthesize_query_with_complexity(
            target_treewidth=treewidth,
            information_dispersion=dispersion
        )
        task['tool_sequence'] = tool_converter.create_tool_sequence(task)
        tasks.append(task)

    print(f"  Generated {len(tasks)} tasks")

    # Stage 2: Atomic skill learning
    print("Stage 2: Atomic Skill Training")
    skill_trainer = AtomicSkillTrainer(model, optimizer, {})

    loss_grounding = skill_trainer.train_intent_grounding(tasks, num_skill_steps // 2)
    loss_planning = skill_trainer.train_hierarchical_planning(tasks, num_skill_steps // 2)

    print(f"  Grounding loss: {loss_grounding:.4f}")
    print(f"  Planning loss: {loss_planning:.4f}")

    # Stage 3: Environment interaction
    print("Stage 3: Environment Interaction")
    env_trainer = LongHorizonSearchRL(model, optimizer, {}, max_steps=30)

    total_success = 0
    for step in range(num_env_steps):
        task = random.choice(tasks)
        trajectory = env_trainer.generate_trajectory(task)
        loss = env_trainer.update_from_trajectory(trajectory)

        if trajectory['success']:
            total_success += 1

        if (step + 1) % 1000 == 0:
            success_rate = total_success / (step + 1)
            print(f"  Step {step + 1}: Success rate={success_rate:.2%}")

    return model
```

## Practical Guidance

**When to use**: Multi-step information-seeking agents; questions requiring 10+ search steps; domains with clear document retrieval

**Hyperparameters**:
- **treewidth_range**: 1-5 (complexity)
- **information_dispersion**: 0.4-0.9 (sparsity)
- **max_steps**: 20-40 (trajectory length)
- **skill_training_ratio**: Allocate steps 40% grounding, 40% planning, 20% integration

**Key advantages**:
- Systematic curriculum from simple to complex
- Staged training reduces expensive environment calls
- Atomic skills enable transfer across domains
- Long-horizon support (30+ steps)

**Common pitfalls**:
- Too much task complexity too soon → exploration failure
- Skill training too brief → weak foundations
- Not validating facts are actually retrievable
- Environment cost not amortized enough

**Scaling**: Simulation environment enables large-scale training; real interaction only for final polish.

## Reference

Paper: https://arxiv.org/abs/2602.14234
Related work: Information retrieval, long-horizon planning, RL for agents
Benchmarks: Humanity's Last Exam, BrowseComp, custom search tasks
