---
name: agent-lightning-rl-training
title: Agent Lightning - Framework-Agnostic RL Training for Any Agent
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03680
keywords: [reinforcement-learning, agent-training, framework-agnostic, rl-infrastructure]
description: "Train RL on diverse agent frameworks (LangChain, AutoGen, custom) via unified data interface and transition-based RL decomposition."
---

## Agent Lightning: Universal RL Training Infrastructure

Agent Lightning decouples RL training from agent execution by providing a unified data interface. Agents run on their native frameworks; lightning captures transitions as semantic state snapshots. A novel hierarchical RL algorithm decomposes episode returns across individual LLM actions, enabling seamless integration with existing RL methods without agent code changes.

### Core Concept

RL training typically requires deep integration with agent code, making it framework-specific. Agent Lightning inverts this: agents remain unchanged; lightning server observes state snapshots and learns. The key insight: abstract agent execution as a state machine where each LLM call is an action. This enables training any agent with minimal modifications while reusing standard RL algorithms.

### Architecture Overview

- **Unified Data Interface**: Agent execution as sequence of state snapshots with semantic variables
- **Markov Decision Process Formulation**: States (snapshots), actions (LLM outputs), rewards (action quality)
- **Hierarchical RL**: Decompose episode returns to individual actions, apply existing RL at token level
- **Lightning Server/Client**: Training (server) separated from execution (client/agent runtime)
- **Framework Agnostic**: Works with LangChain, OpenAI SDK, AutoGen, custom agents

### Implementation Steps

**Step 1: Define Unified State Snapshot Interface**

```python
from dataclasses import dataclass
from typing import Any, Dict, List
from enum import Enum

class CallType(Enum):
    LLM = "llm"
    TOOL = "tool"
    DECISION = "decision"
    ACTION = "action"

@dataclass
class Call:
    """Single component invocation in agent execution."""
    component: str  # "gpt4", "web_search", "calculator"
    input: Dict[str, Any]  # Input parameters
    output: Any  # Execution result
    metadata: Dict[str, Any] = None  # Additional info (latency, cost, etc.)

@dataclass
class StateSnapshot:
    """Complete agent execution state at a moment in time."""
    step_number: int
    task: str  # Current task/objective
    semantic_variables: Dict[str, Any]  # Variables relevant to task
    call_history: List[Call]  # History of component calls
    current_context: str  # Relevant context for decision
    timestamp: float

    def to_dict(self) -> Dict:
        """Serialize to dict for transmission."""
        return {
            'step': self.step_number,
            'task': self.task,
            'variables': self.semantic_variables,
            'calls': [c.__dict__ for c in self.call_history],
            'context': self.current_context,
        }

class StateCapture:
    """Capture execution state without modifying agent code."""

    def __init__(self, agent):
        self.agent = agent
        self.snapshots = []

    def capture_state(self, step: int, task: str, variables: Dict,
                     calls: List[Call]) -> StateSnapshot:
        """Create state snapshot from current execution."""
        snapshot = StateSnapshot(
            step_number=step,
            task=task,
            semantic_variables=variables,
            call_history=calls,
            current_context=self._extract_context(variables),
            timestamp=time.time()
        )
        self.snapshots.append(snapshot)
        return snapshot

    def _extract_context(self, variables: Dict) -> str:
        """Extract relevant context from semantic variables."""
        relevant_keys = ['query', 'search_results', 'current_answer']
        context_parts = []
        for key in relevant_keys:
            if key in variables:
                context_parts.append(f"{key}: {variables[key]}")
        return '\n'.join(context_parts)
```

**Step 2: Build Agent-Server Communication**

```python
import json
from typing import Callable

class LightningClient:
    """Agent-side client for reporting execution to training server."""

    def __init__(self, server_url: str = "localhost:5000"):
        self.server_url = server_url
        self.session_id = None

    def register_agent(self, agent_name: str) -> str:
        """Register agent execution session."""
        response = requests.post(f"{self.server_url}/register", json={
            'agent_name': agent_name,
            'timestamp': time.time()
        })
        self.session_id = response.json()['session_id']
        return self.session_id

    def report_transition(self, state: StateSnapshot, action: str,
                        next_state: StateSnapshot, reward: float):
        """Report (s, a, s', r) transition to training server."""
        transition = {
            'session_id': self.session_id,
            'state': state.to_dict(),
            'action': action,
            'next_state': next_state.to_dict(),
            'reward': reward,
            'timestamp': time.time()
        }
        requests.post(f"{self.server_url}/transition", json=transition)

class LightningServer:
    """Training-side server that collects and processes transitions."""

    def __init__(self, model, learning_rate: float = 1e-5):
        self.model = model
        self.lr = learning_rate
        self.transitions = []

    def receive_transition(self, transition: Dict):
        """Receive and queue transition from agent."""
        self.transitions.append(transition)

    def process_batch(self, batch_transitions: List[Dict]):
        """Process batch of transitions for RL update."""
        for transition in batch_transitions:
            state_dict = transition['state']
            action = transition['action']
            reward = transition['reward']

            # Convert state dict back to semantic representation
            # (in practice would reconstruct embeddings or features)
            state_features = self._state_to_features(state_dict)

            # Compute loss for this action
            logp = self.model.compute_logp(state_features, action)
            loss = -logp * reward  # Policy gradient

            loss.backward()

        self.model.optimizer.step()

    def _state_to_features(self, state_dict: Dict):
        """Convert state dict to model-compatible features."""
        # Reconstruct embeddings from task, variables, context
        task_text = state_dict['task']
        context = state_dict['context']
        prompt = f"Task: {task_text}\nContext: {context}"
        features = self.model.encode(prompt)
        return features
```

**Step 3: Implement Transition-Based RL Decomposition**

```python
from typing import List, Tuple

class HierarchicalRL:
    """
    Hierarchical RL: decompose episode return across individual LLM actions.
    """

    def __init__(self, model, gamma: float = 0.99):
        self.model = model
        self.gamma = gamma  # Discount factor

    def decompose_episode_return(self, episode: List[Dict], episode_return: float) -> List[float]:
        """
        Distribute episode return across individual actions.

        episode: List of transitions
        episode_return: Total reward for episode

        Returns: Per-action rewards (credit assignment)
        """
        num_actions = len(episode)

        # Method 1: Simple decomposition - equal credit per action
        # action_rewards = [episode_return / num_actions] * num_actions

        # Method 2: Temporally-discounted credit assignment
        action_rewards = []
        for t in range(num_actions):
            # Reward for action t: contribution to future returns
            future_steps = num_actions - t
            discount = self.gamma ** future_steps
            action_reward = episode_return * discount / num_actions

            action_rewards.append(action_reward)

        # Method 3: Advantage estimation with baseline
        # (more sophisticated)
        baseline_returns = self._estimate_baseline(episode)
        action_rewards_with_baseline = [
            (ep_r - bl_r) for ep_r, bl_r in zip(action_rewards, baseline_returns)
        ]

        return action_rewards_with_baseline

    def _estimate_baseline(self, episode: List[Dict]) -> List[float]:
        """Estimate expected return at each step (value function)."""
        baselines = []
        remaining_steps = len(episode)

        for transition in episode:
            # Heuristic: baseline = average of future rewards
            expected_return = sum(t.get('reward', 0) for t in episode[len(baselines):])
            baseline = expected_return / max(1, remaining_steps)
            baselines.append(baseline)
            remaining_steps -= 1

        return baselines

    def train_on_episode(self, episode: List[Dict], episode_return: float):
        """Train on single episode with action-level credit assignment."""
        action_rewards = self.decompose_episode_return(episode, episode_return)

        for transition, action_reward in zip(episode, action_rewards):
            state_features = transition['state']
            action = transition['action']

            # Update policy
            logp = self.model.compute_logp(state_features, action)
            loss = -logp * action_reward  # Weighted by credit

            loss.backward()

        self.model.optimizer.step()
```

**Step 4: Integrate with Diverse Agent Frameworks**

```python
class AgentAdapterLangChain:
    """Adapter for LangChain agents."""

    def __init__(self, agent_chain):
        self.agent = agent_chain
        self.client = LightningClient()
        self.client.register_agent('langchain-agent')

    def run_with_lightning(self, task: str) -> str:
        """Run agent, report transitions to training server."""
        variables = {'task': task}
        calls = []
        state_number = 0

        # Capture initial state
        state = StateSnapshot(
            step_number=state_number,
            task=task,
            semantic_variables=variables,
            call_history=calls,
            current_context=task,
            timestamp=time.time()
        )

        # Run LangChain agent
        result = self.agent.run(task)
        state_number += 1

        # Capture final state
        next_state = StateSnapshot(
            step_number=state_number,
            task=task,
            semantic_variables={'result': result},
            call_history=calls,
            current_context=result,
            timestamp=time.time()
        )

        # Compute reward (e.g., success or quality metric)
        reward = self._compute_reward(result, task)

        # Report to training server
        self.client.report_transition(state, result, next_state, reward)

        return result

    def _compute_reward(self, result: str, task: str) -> float:
        """Compute reward for this action."""
        # Would implement actual reward function
        return 1.0 if len(result) > 0 else 0.0

class AgentAdapterAutoGen:
    """Adapter for AutoGen agents."""

    def __init__(self, user_proxy, assistant):
        self.user_proxy = user_proxy
        self.assistant = assistant
        self.client = LightningClient()
        self.client.register_agent('autogen-agent')

    def run_with_lightning(self, task: str) -> str:
        """Run AutoGen with lightning integration."""
        # Similar to LangChain adapter
        self.user_proxy.initiate_chat(self.assistant, message=task)
        # Extract result and report
```

**Step 5: End-to-End Training Loop**

```python
def train_agents_lightning(agent_definitions: Dict, num_episodes: int = 100):
    """
    Train multiple diverse agents with unified RL infrastructure.
    """
    # Start lightning server
    server = LightningServer(model=gpt4_model)

    # Create agents
    agents = {}
    for agent_name, agent_def in agent_definitions.items():
        if agent_name == 'langchain':
            agents[agent_name] = AgentAdapterLangChain(agent_def)
        elif agent_name == 'autogen':
            agents[agent_name] = AgentAdapterAutoGen(*agent_def)

    # Training loop
    for episode in range(num_episodes):
        for agent_name, agent in agents.items():
            # Generate task
            task = generate_random_task()

            # Run agent (reports transitions to server)
            result = agent.run_with_lightning(task)

            # Compute episode return
            episode_return = evaluate_result(result, task)

            # Server processes batch when ready
            if len(server.transitions) > 32:
                server.process_batch(server.transitions[-32:])

        if episode % 10 == 0:
            print(f"Episode {episode}")

    return agents
```

### Practical Guidance

**When to Use:**
- Multi-framework agent training
- RL training without modifying agent code
- Scenarios with diverse agent architectures
- Infrastructure-level agent training

**When NOT to Use:**
- Single-agent systems (direct training simpler)
- Real-time agents requiring <100ms latency (overhead of communication)
- Proprietary agents without SDK access

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `gamma` (discount factor) | 0.99 | Higher = values future rewards more; 0.99 standard for control |
| `learning_rate` | 1e-5 | Standard LLM RL rate |
| `batch_size` | 32 | Larger = more stable but slower updates |
| `decomposition_method` | temporal-discount | How to assign credit per action |

### Reference

**Paper**: Agent Lightning: Train ANY AI Agents with RL (2508.03680)
- Framework-agnostic through unified state interface
- Hierarchical RL decomposes episode returns
- Seamless integration with LangChain, AutoGen, custom agents
