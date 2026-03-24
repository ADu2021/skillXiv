---
name: agent-r1-end-to-end-rl
title: "Agent-R1: Training Powerful LLM Agents with End-to-End RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.14460"
keywords: [LLM Agents, RL Training, Tool Use, MDP Formulation, Policy Optimization]
description: "Train LLM-based agents with end-to-end RL by extending MDPs to handle tool invocation and environmental stochasticity—enable dense process rewards for intermediate steps and masked policy gradients for learnable actions."
---

# Train LLM Agents with RL by Extending MDP Formulations

Standard LLM training treats models as passive token generators within fixed contexts. LLM agents are fundamentally different: they interact with environments (call APIs, execute code, read files), adapt to feedback, and make sequential decisions. Agent-R1 formalizes this via extended MDPs, then applies end-to-end RL with careful credit assignment.

The key innovations are (1) extending state-action-reward formalisms to capture tool invocation and environmental stochasticity, (2) computing dense process rewards for intermediate steps, and (3) masking policy gradients to only update learnable agent actions (excluding prompts and environment feedback).

## Core Concept

Language models generate tokens; agents generate tokens AND invoke tools, receiving environmental feedback. This creates a fundamental departure from standard language modeling:

- **State Space**: History of multi-turn interactions and environmental feedback, not just text context
- **Action Space**: Token generation AND tool invocation commands; specific token sequences trigger external tools
- **State Transitions**: Two transition types—generative (deterministic token generation) and environmental (stochastic tool execution)
- **Reward Signal**: Dense rewards (process rewards for intermediate steps) + outcome rewards (final result)

Agent-R1 formalizes these distinctions, then applies RL with careful masking to avoid backpropagating gradients through immutable environmental responses.

## Architecture Overview

- **Extended MDP**: State includes dialogue history + environment state; actions include token generation + tool commands
- **Tool Module**: Standardized interface for atomic tool invocation (function calling); returns raw execution outcomes
- **ToolEnv Module**: Transforms tool outputs into environment state transitions and computes reward signals
- **Action Masking**: Distinguish agent-generated tokens from prompts and environmental responses; policy gradients only flow through agent actions
- **Advantage Alignment**: Combine process rewards (intermediate progress) with outcome rewards (final success) for advantage estimation
- **Masked Loss Computation**: Only agent-controlled tokens contribute to policy gradients

## Implementation Steps

**Step 1: Extend MDP Formulation.** Define states, actions, transitions, and rewards for agentic LLMs.

```python
class AgentMDP:
    """
    Extended MDP capturing tool use and environmental stochasticity.
    """
    def __init__(self):
        self.state_history = []      # Multi-turn dialogue + environment state
        self.tool_interface = ToolInterface()  # Standard tool calling API
        self.env = Environment()     # External environment

    class State:
        def __init__(self, dialogue_history, env_state):
            self.dialogue_history = dialogue_history  # Conversation so far
            self.env_state = env_state                # Current environment
            self.timestamp = time.time()

        def to_tensor(self, tokenizer):
            """Convert state to token embeddings for LLM processing."""
            full_text = self._format_dialogue(self.dialogue_history)
            full_text += "\n[ENV_STATE]:\n" + str(self.env_state)
            tokens = tokenizer.encode(full_text)
            return torch.tensor(tokens)

        def _format_dialogue(self, history):
            lines = []
            for msg in history:
                lines.append(f"{msg['role']}: {msg['content']}")
            return "\n".join(lines)

    class Action:
        def __init__(self, action_type, content):
            self.action_type = action_type  # 'generate_tokens' or 'invoke_tool'
            self.content = content          # Token sequence or tool command

        def is_tool_invocation(self):
            return self.action_type == 'invoke_tool'

        def parse_tool_call(self):
            """Extract tool name and arguments from token sequence."""
            # Heuristic: parse <tool_call>name(args)</tool_call>
            match = re.search(r'<tool_call>(\w+)\((.*?)\)</tool_call>', self.content)
            if match:
                return {'tool': match.group(1), 'args': match.group(2)}
            return None

    def compute_state_transition(self, state, action):
        """
        Compute next state given current state and action.
        Distinguishes generative (deterministic) and environmental (stochastic) transitions.
        """
        if not action.is_tool_invocation():
            # Generative transition: LLM generated tokens
            next_dialogue = state.dialogue_history.copy()
            next_dialogue.append({'role': 'assistant', 'content': action.content})

            # Environment state unchanged
            next_env_state = state.env_state

        else:
            # Environmental transition: tool invocation
            tool_call = action.parse_tool_call()
            if not tool_call:
                return None  # Invalid tool call

            # Execute tool (stochastic)
            tool_result = self.tool_interface.invoke(
                tool_call['tool'],
                tool_call['args']
            )

            # Update dialogue
            next_dialogue = state.dialogue_history.copy()
            next_dialogue.append({'role': 'assistant', 'content': action.content})
            next_dialogue.append({'role': 'user', 'content': f"Tool result: {tool_result}"})

            # Update environment state
            next_env_state = self.env.update_state(state.env_state, tool_call)

        next_state = self.State(next_dialogue, next_env_state)
        return next_state

    def compute_reward(self, state, action, next_state):
        """
        Compute dense rewards for both process and outcome.
        """
        process_reward = 0.0
        outcome_reward = 0.0

        # Process reward: intermediate progress
        if action.is_tool_invocation():
            tool_call = action.parse_tool_call()
            # Reward for valid tool invocation
            process_reward += 0.5
            # Bonus for tools that advance the task
            if self.env.is_progress(next_state.env_state):
                process_reward += 0.5

        # Outcome reward: task completion
        if self.env.is_task_complete(next_state.env_state):
            outcome_reward = 1.0
        elif self.env.is_task_failed(next_state.env_state):
            outcome_reward = -1.0

        total_reward = process_reward + outcome_reward
        return total_reward
```

**Step 2: Implement Tool Module and ToolEnv.** Standardize tool invocation and result interpretation.

```python
class ToolInterface:
    """Standardized tool invocation interface."""
    def __init__(self):
        self.tools = {}  # Registry of available tools

    def register_tool(self, name, fn, description):
        """Register a callable tool."""
        self.tools[name] = {'fn': fn, 'description': description}

    def invoke(self, tool_name, args_str):
        """
        Invoke tool and return result.
        args_str: string representation of arguments (e.g., "x=5, y=10")
        """
        if tool_name not in self.tools:
            return f"Error: unknown tool '{tool_name}'"

        try:
            tool_fn = self.tools[tool_name]['fn']
            # Parse arguments
            args = self._parse_args(args_str)
            result = tool_fn(**args)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def _parse_args(self, args_str):
        """Parse 'key=value, key2=value2' into dict."""
        args = {}
        for pair in args_str.split(','):
            if '=' in pair:
                key, val = pair.split('=', 1)
                args[key.strip()] = val.strip().strip('"\'')
        return args


class ToolEnv:
    """Transform tool results into RL signals."""
    def __init__(self, task_spec):
        self.task_spec = task_spec
        self.tool_history = []

    def process_tool_result(self, tool_name, result):
        """Convert tool result to state transition and reward."""
        # Track tool invocation
        self.tool_history.append({'tool': tool_name, 'result': result})

        # Determine if tool result advances task
        is_progress = self._evaluate_progress(tool_name, result)
        reward = 0.5 if is_progress else -0.1

        return {
            'next_state_info': {'tool_result': result},
            'reward': reward,
            'is_progress': is_progress
        }

    def _evaluate_progress(self, tool_name, result):
        """Heuristic: did this tool call advance the task?"""
        # Task-specific logic; example: code execution that doesn't error
        if tool_name == 'code_execute':
            return 'error' not in result.lower()
        return True
```

**Step 3: Policy Gradient with Action Masking.** Only update tokens generated by the agent.

```python
def compute_masked_policy_loss(
    logits, target_tokens, action_mask, advantages, seq_mask
):
    """
    Compute policy loss with action masking.
    logits: (batch, seq_len, vocab_size)
    target_tokens: (batch, seq_len) ground truth tokens
    action_mask: (batch, seq_len) binary mask (1=agent action, 0=prompt/env)
    advantages: (batch,) advantage estimates
    seq_mask: (batch, seq_len) valid tokens (1=valid, 0=padding)
    """
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(
        log_probs, -1, target_tokens.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len)

    # Apply action mask: zero gradients for non-agent actions
    selected_log_probs = selected_log_probs * action_mask

    # Apply sequence mask: ignore padding
    selected_log_probs = selected_log_probs * seq_mask

    # Policy gradient: -advantage * log_prob
    # Advantage is per-trajectory, broadcast to all tokens
    advantages_expanded = advantages.unsqueeze(1)  # (batch, 1)
    policy_loss = -(selected_log_probs.sum(dim=1) * advantages_expanded.squeeze(-1)).mean()

    return policy_loss


def build_action_mask(tokens, tokenizer):
    """
    Identify which tokens are agent-generated vs. prompt/environment.
    Heuristic: tokens between <agent_start> and <agent_end> markers.
    """
    mask = torch.zeros_like(tokens)

    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token])
        if '<agent_start>' in token_str:
            # Start marking agent actions
            agent_start = True
        elif '<agent_end>' in token_str:
            agent_start = False
        elif agent_start:
            mask[i] = 1

    return mask
```

**Step 4: Train Agent with Advantage Alignment.** Combine process and outcome rewards.

```python
def train_agent_r1(agent_model, environment, num_episodes=1000, lr=1e-5):
    """
    End-to-end RL training for agent.
    """
    optimizer = torch.optim.AdamW(agent_model.parameters(), lr=lr)

    for episode in range(num_episodes):
        # Initialize trajectory
        state = environment.reset()
        trajectory = []
        rewards_list = []

        # Rollout: agent acts until task complete
        while not environment.is_done():
            # Agent generates action
            logits = agent_model(state.to_tensor())
            action = sample_action(logits)

            # Environment transitions
            next_state = environment.step(action)
            reward = environment.get_reward(state, action, next_state)

            trajectory.append({
                'state': state,
                'action': action,
                'logits': logits,
                'reward': reward
            })
            rewards_list.append(reward)

            state = next_state

        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = compute_gae(rewards_list, gamma=0.99, lam=0.95)

        # Compute losses
        policy_losses = []
        for i, step in enumerate(trajectory):
            logits = step['logits']
            target_tokens = step['action']
            action_mask = build_action_mask(target_tokens, agent_model.tokenizer)
            advantage = advantages[i]

            loss = compute_masked_policy_loss(
                logits,
                target_tokens,
                action_mask,
                advantage,
                seq_mask=torch.ones_like(target_tokens)
            )
            policy_losses.append(loss)

        total_loss = torch.stack(policy_losses).mean()

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_model.parameters(), 1.0)
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: loss={total_loss.item():.4f}, "
                  f"reward={sum(rewards_list):.2f}")
```

## Practical Guidance

**When to Use:** Training agents that need to solve multi-step problems involving external tool use (code execution, API calls, web navigation). Use when standard supervised fine-tuning doesn't capture the sequential decision-making needed.

**Hyperparameters:**
- Process reward magnitude: 0.5 for tool invocation; adjust upward if agent avoids tools
- Outcome reward: ±1.0 for task success/failure; scale based on problem difficulty
- Action mask strategy: use XML markers (<agent_start>/<agent_end>) or attention masks
- Advantage normalization: crucial for stability; use GAE with λ=0.95, γ=0.99

**Pitfalls:**
- **Gradient masking errors**: Incorrect masks can lead to backprop through environment, causing divergence; validate masks carefully
- **Reward sparsity**: Dense process rewards crucial; define them clearly for intermediate progress
- **Exploiting tool errors**: Agents may learn to invoke broken tools if reward signal is weak; penalize failed tool invocations
- **Dialogue explosion**: Long multi-turn histories blow up context; implement periodic summarization

**When NOT to Use:** Simple classification or generation tasks where supervised fine-tuning suffices; single-step predictions with no feedback loops.

**Integration:** Compatible with any LLM (GPT, Llama, Claude); requires well-defined tool interface and reward computation for your task domain.

---
Reference: https://arxiv.org/abs/2511.14460
