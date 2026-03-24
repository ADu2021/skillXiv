---
name: moe-routing-alignment
title: "Stabilizing MoE RL by Aligning Training and Inference Routers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.11370"
keywords: [MoE, Routing, Reinforcement Learning, Stability, Training]
description: "Prevents MoE router instability during RL training by recording and replaying inference-phase routing distributions back into training. Reduces training-inference routing divergence and KL divergence, enabling stable MoE RL scaling without sacrificing training speed."
---

# Rollout Routing Replay: Stabilizing Mixture-of-Experts RL

Mixture-of-Experts models with RL training suffer from routing instability: routers diverge between training and inference phases, causing catastrophic collapse. Rollout Routing Replay (R3) synchronizes routing behavior across both phases.

By recording inference routing and replaying those decisions during training, R3 prevents divergence and enables stable scaling.

## Core Concept

The core problem: **routers make different decisions during training vs. inference**, even with identical inputs. This divergence causes:
- Policy collapse in RL training
- Inconsistent expert utilization
- Degraded final model performance

R3 solves this by:
- Recording routing decisions during inference rollouts
- Replaying those same routing distributions during policy gradient training
- Synchronizing training and inference for the same states

## Architecture Overview

- Inference-phase routing distribution recording
- Training-phase replay mechanism to enforce learned routing
- KL divergence minimization between phases
- Efficient storage of routing patterns

## Implementation Steps

Record routing distributions during inference rollouts. When the model makes decisions in the environment, capture which experts were selected:

```python
class RoutingRecorder:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.routing_history = []

    def record_routing(self, inputs, routing_logits):
        """Capture routing decisions during inference."""
        # Get expert selection distribution
        routing_probs = torch.softmax(routing_logits, dim=-1)

        # For each token, record which expert was selected
        selected_experts = torch.argmax(routing_probs, dim=-1)

        record = {
            'inputs': inputs.detach().cpu(),
            'routing_logits': routing_logits.detach().cpu(),
            'routing_probs': routing_probs.detach().cpu(),
            'selected_experts': selected_experts.detach().cpu()
        }

        self.routing_history.append(record)
        return record

    def get_routing_distribution(self):
        """Retrieve recorded routing distributions for training."""
        return self.routing_history

    def clear(self):
        """Clear history after training batch."""
        self.routing_history = []
```

Implement training with enforced routing replay. During policy gradient updates, use recorded routing to constrain router behavior:

```python
class RoutingReplayPG:
    def __init__(self, model, recorder, learning_rate=1e-4):
        self.model = model
        self.recorder = recorder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_loss_with_routing_replay(self, batch_inputs, recorded_routing, rewards):
        """Compute RL loss while enforcing routing consistency."""
        # Forward pass through model
        logits, routing_logits = self.model(batch_inputs)

        # Get current routing probabilities
        current_probs = torch.softmax(routing_logits, dim=-1)
        recorded_probs = recorded_routing['routing_probs']

        # Policy gradient loss
        action_log_probs = torch.log(current_probs.gather(-1, recorded_routing['selected_experts'].unsqueeze(-1)))
        pg_loss = -(action_log_probs * rewards).mean()

        # Routing consistency loss: penalize divergence from recorded routing
        kl_divergence = torch.nn.functional.kl_div(
            torch.log(current_probs + 1e-10),
            recorded_probs,
            reduction='batchmean'
        )

        # Total loss: PG + routing alignment
        total_loss = pg_loss + 0.5 * kl_divergence

        return total_loss, {
            'pg_loss': pg_loss.item(),
            'kl_div': kl_divergence.item(),
            'total_loss': total_loss.item()
        }

    def training_step(self, batch_inputs, recorded_routing, rewards):
        """Perform one gradient update with routing replay."""
        loss, metrics = self.compute_loss_with_routing_replay(
            batch_inputs, recorded_routing, rewards
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return metrics
```

Implement the full training loop that alternates between inference (recording) and training (replaying):

```python
def train_moe_with_routing_replay(model, env, num_episodes=1000):
    """Main training loop with routing replay."""
    recorder = RoutingRecorder(num_experts=model.num_experts)
    pg_trainer = RoutingReplayPG(model, recorder)

    for episode in range(num_episodes):
        # Phase 1: Inference rollouts with routing recording
        episode_transitions = []
        for _ in range(batch_size):
            state = env.reset()
            done = False

            while not done:
                # Forward pass records routing
                with torch.no_grad():
                    action, routing_logits = model(state)
                    recorder.record_routing(state, routing_logits)

                state, reward, done, _ = env.step(action)
                episode_transitions.append((state, reward))

        # Phase 2: Training with recorded routing
        recorded_routing = recorder.get_routing_distribution()
        episode_reward = sum(r for _, r in episode_transitions)

        # Train on batch with enforced routing
        metrics = pg_trainer.training_step(
            batch_inputs=torch.stack([s for s, _ in episode_transitions]),
            recorded_routing=recorded_routing[0],  # Use first recorded distribution
            rewards=torch.tensor([r for _, r in episode_transitions])
        )

        recorder.clear()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"KL Div: {metrics['kl_div']:.4f}")
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| KL divergence weight | 0.5-1.0 (balance PG loss and routing consistency) |
| Routing history retention | Single batch (memory-efficient) |
| Recording overhead | ~5-10% inference time |
| Batch size for replay | Same as inference batch |

**When to use:**
- MoE models with RL training (training collapse without R3)
- Scenarios where router stability is critical
- Large-scale models where divergence is pronounced
- Environments with exploration requirements

**When NOT to use:**
- Supervised fine-tuning only (routing already stable)
- Fixed routing strategies (no learnable router)
- Memory-constrained environments (recording adds overhead)

**Common pitfalls:**
- KL weight too high (suppresses learning new routing)
- Recording stale data across episodes (outdated routing patterns)
- Not clearing history between batches (information leakage)
- Mismatch between recorded batch size and training batch size

Reference: [Stabilizing MoE RL on arXiv](https://arxiv.org/abs/2510.11370)
