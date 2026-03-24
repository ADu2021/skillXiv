---
name: sotopia-rl-social-intelligence
title: Sotopia-RL - Reward Design for Social Intelligence
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03905
keywords: [social-intelligence, reinforcement-learning, dialogue, reward-design]
description: "Train socially intelligent LLMs via utterance-level credit assignment and multi-dimensional reward aggregation for social interactions."
---

## Sotopia-RL: Reward Design for Social Intelligence

Sotopia-RL addresses the unique challenge of training LLMs for social tasks via reinforcement learning. Unlike verifiable domains (math, coding), social interactions have no ground truth. The breakthrough: two-stage reward design decomposing episode-level feedback into utterance-level contributions across multiple social dimensions (goal completion, relationship, knowledge-seeking).

### Core Concept

Standard RL rewards (task success/failure) work well for verifiable problems. Social tasks are fundamentally different: success means balancing goal completion with relationship maintenance and mutual understanding. An agent could complete the goal but damage the relationship, or vice versa. Sotopia-RL's key insight: decompose episode outcomes into individual utterance contributions across multiple dimensions, enabling nuanced reward signals for social learning.

### Architecture Overview

- **Offline Reward Labeling**: GPT-4o generates utterance-level scores across 7 dimensions with full episode context
- **Online Reward Model**: Smaller LLM trained on offline labels provides real-time signals during RL
- **Utterance-Level Credit Assignment**: Rather than binary task success, attribute progress to specific utterances
- **Multi-Dimensional Aggregation**: Combine goal completion, relationship, knowledge, formality, etc.
- **GRPO Training**: Group Relative Policy Optimization using learned reward model

### Implementation Steps

**Step 1: Design Multi-Dimensional Reward Framework**

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class RewardDimension(Enum):
    """Dimensions of social interaction quality."""
    GOAL_COMPLETION = "goal_completion"  # Progress toward conversational goal
    RELATIONSHIP = "relationship"  # Maintaining/building relationship
    KNOWLEDGE = "knowledge"  # Sharing/acquiring information
    POLITENESS = "politeness"  # Social norms and courtesy
    ENGAGEMENT = "engagement"  # Keeping other person interested
    COHERENCE = "coherence"  # Logical consistency
    SAFETY = "safety"  # Avoiding harmful/offensive content

@dataclass
class UtteranceReward:
    """Reward for single utterance across dimensions."""
    utterance_idx: int
    utterance_text: str
    scores: Dict[RewardDimension, float]  # 0-1 per dimension
    attribution: Dict[RewardDimension, str]  # Reasoning per dimension

    def aggregate_score(self, weights: Dict[RewardDimension, float] = None) -> float:
        """Combine dimensions into single reward."""
        if weights is None:
            # Default: equal weight
            weights = {dim: 1.0 / len(RewardDimension) for dim in RewardDimension}

        weighted_score = sum(
            self.scores[dim] * weights[dim]
            for dim in RewardDimension
        )
        return weighted_score
```

**Step 2: Implement Offline Reward Labeling with GPT-4o**

```python
class OfflineRewardLabeler:
    """
    Use GPT-4o to label utterances in complete episodes.
    Full episode context enables accurate credit assignment.
    """

    def __init__(self, labeling_model='gpt-4o'):
        self.model = labeling_model
        self.dimensions = list(RewardDimension)

    def label_episode(self, episode: Dict) -> List[UtteranceReward]:
        """
        Label all utterances in episode with multi-dimensional scores.
        Uses full episode context for accuracy.
        """
        task_context = episode['task']
        dialogue_history = episode['dialogue']
        final_outcome = episode['outcome']

        utterance_rewards = []

        for turn_idx, turn in enumerate(dialogue_history):
            utterance = turn['agent_utterance']

            # Prompt for multi-dimensional evaluation
            eval_prompt = f"""Evaluate this utterance in context of a social task.

Task: {task_context}

Full Conversation History:
{self._format_dialogue(dialogue_history)}

Utterance #{turn_idx}:
"{utterance}"

Task Outcome: {final_outcome}

For this specific utterance, rate (0-10) how much it contributes to:
1. GOAL_COMPLETION: Progress toward task goal
2. RELATIONSHIP: Maintaining/building rapport with other person
3. KNOWLEDGE: Sharing relevant information
4. POLITENESS: Following social norms and courtesy
5. ENGAGEMENT: Keeping other person interested
6. COHERENCE: Logical consistency with conversation
7. SAFETY: Avoiding harmful/offensive content

Provide scores and brief reasoning (1-2 sentences) for each dimension."""

            # Get detailed evaluation
            evaluation = self.model.generate(eval_prompt)

            # Parse evaluation
            scores, attributions = self._parse_evaluation(evaluation)

            utterance_reward = UtteranceReward(
                utterance_idx=turn_idx,
                utterance_text=utterance,
                scores=scores,
                attribution=attributions
            )

            utterance_rewards.append(utterance_reward)

        return utterance_rewards

    def _format_dialogue(self, dialogue: List[Dict]) -> str:
        """Format dialogue for context in prompt."""
        formatted = []
        for i, turn in enumerate(dialogue):
            formatted.append(f"Agent: {turn.get('agent_utterance', '')}")
            formatted.append(f"Other: {turn.get('other_utterance', '')}")
        return '\n'.join(formatted)

    def _parse_evaluation(self, text: str) -> Tuple[Dict, Dict]:
        """Parse GPT evaluation into scores and reasoning."""
        scores = {}
        attributions = {}

        # Extract scores (0-10) for each dimension
        import re
        for dimension in RewardDimension:
            pattern = rf"{dimension.value}.*?(\d+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_score = int(match.group(1))
                scores[dimension] = min(raw_score / 10, 1.0)  # Normalize to [0,1]
            else:
                scores[dimension] = 0.5

            # Extract reasoning
            if dimension.value in text.lower():
                relevant_section = text.split(dimension.value)[1].split('\n')[0]
                attributions[dimension] = relevant_section[:100]
            else:
                attributions[dimension] = "No explanation"

        return scores, attributions
```

**Step 3: Train Online Reward Model**

```python
class OnlineRewardModel(torch.nn.Module):
    """
    Smaller LLM that predicts utterance-level rewards during RL.
    Trained on offline labels from GPT-4o.
    """

    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        super().__init__()
        self.encoder = load_pretrained_model(model_name)

        # Reward heads for each dimension
        self.reward_heads = torch.nn.ModuleDict({
            dim.value: torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid()  # Output in [0, 1]
            )
            for dim in RewardDimension
        })

    def forward(self, utterance: str, context: str = "") -> Dict[RewardDimension, float]:
        """
        Predict reward for utterance given context.
        """
        # Encode utterance with context
        prompt = f"Context: {context}\nUtterance: {utterance}"
        embeddings = self.encoder(prompt)
        pooled = embeddings.pooler_output

        # Predict each dimension
        predictions = {}
        for dim in RewardDimension:
            score = self.reward_heads[dim.value](pooled).squeeze()
            predictions[dim] = score.item()

        return predictions

    def train_on_offline_data(self, offline_rewards: List[List[UtteranceReward]],
                             learning_rate: float = 1e-5):
        """
        Fine-tune on offline labels.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for episode_rewards in offline_rewards:
            for utterance_reward in episode_rewards:
                utterance = utterance_reward.utterance_text

                # Predict
                predictions = self.forward(utterance)

                # Loss: MSE against offline labels
                loss = 0.0
                for dim in RewardDimension:
                    target = utterance_reward.scores[dim]
                    pred = predictions[dim]
                    loss += torch.nn.functional.mse_loss(torch.tensor(pred), torch.tensor(target))

                loss.backward()

        optimizer.step()
```

**Step 4: Implement GRPO Training with Multi-Dimensional Rewards**

```python
class SocialGRPO:
    """
    Group Relative Policy Optimization for social tasks.
    Uses multi-dimensional reward model.
    """

    def __init__(self, model, reward_model: OnlineRewardModel):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def compute_utterance_advantage(self, utterance: str, context: str,
                                   group_reward_baseline: float) -> float:
        """
        Compute advantage: how much did this utterance exceed baseline?
        """
        # Get multi-dimensional reward
        rewards_dict = self.reward_model.forward(utterance, context)

        # Aggregate across dimensions (weighted average)
        aggregated_reward = sum(rewards_dict.values()) / len(rewards_dict)

        # Advantage relative to group baseline
        advantage = aggregated_reward - group_reward_baseline

        return advantage

    def train_on_trajectory(self, dialogue: List[Dict], task: str):
        """
        Train on single dialogue trajectory.
        """
        context = task

        # Group baseline: average reward for this group of trajectories
        group_baseline = 0.5  # Heuristic

        # For each utterance
        for turn_idx, turn in enumerate(dialogue):
            utterance = turn['agent_utterance']

            # Compute context (conversation history up to this point)
            history = '\n'.join([
                f"Agent: {t['agent_utterance']}\nOther: {t['other_utterance']}"
                for t in dialogue[:turn_idx]
            ])

            # Compute advantage
            advantage = self.compute_utterance_advantage(utterance, history, group_baseline)

            # Get log probability of utterance
            logp = self.model.compute_logp(utterance, history)

            # Policy gradient loss
            pg_loss = -logp * advantage
            pg_loss.backward()

        self.optimizer.step()
```

**Step 5: Evaluate on Sotopia Benchmark**

```python
def evaluate_social_agent(agent_model, reward_model: OnlineRewardModel,
                         test_episodes: List[Dict]) -> Dict:
    """
    Evaluate social agent on Sotopia benchmark.
    Sotopia-hard and full benchmark.
    """
    results = {
        'hard_avg_score': 0.0,
        'full_avg_score': 0.0,
        'dimension_scores': {dim.value: 0.0 for dim in RewardDimension}
    }

    for episode_type in ['hard', 'full']:
        episode_list = [e for e in test_episodes if e['type'] == episode_type]

        episode_scores = []

        for episode in episode_list:
            task = episode['task']

            # Run agent
            dialogue = run_social_agent(agent_model, task, num_turns=10)

            # Evaluate with reward model
            total_reward = 0.0
            dimension_totals = {dim: 0.0 for dim in RewardDimension}

            for turn in dialogue:
                utterance = turn['agent_utterance']
                context = '\n'.join([
                    f"Agent: {t['agent_utterance']}\nOther: {t['other_utterance']}"
                    for t in dialogue[:dialogue.index(turn)]
                ])

                rewards_dict = reward_model.forward(utterance, context)
                for dim, score in rewards_dict.items():
                    dimension_totals[dim] += score

                total_reward += sum(rewards_dict.values()) / len(rewards_dict)

            # Average per turn
            num_turns = len(dialogue)
            episode_score = total_reward / num_turns

            episode_scores.append(episode_score)

            # Aggregate dimensions
            for dim in RewardDimension:
                dimension_totals[dim] /= num_turns
                results['dimension_scores'][dim.value] += dimension_totals[dim]

        # Average across episodes
        avg_score = sum(episode_scores) / len(episode_scores)
        if episode_type == 'hard':
            results['hard_avg_score'] = avg_score
        else:
            results['full_avg_score'] = avg_score

    # Normalize dimension scores
    num_episodes = len(test_episodes) // 2
    for dim in results['dimension_scores']:
        results['dimension_scores'][dim] /= num_episodes

    return results
```

### Practical Guidance

**When to Use:**
- Social task automation (customer service, negotiation, discussion)
- Scenarios emphasizing relationship and engagement
- Multi-turn dialogue systems
- Tasks where fairness and politeness matter

**When NOT to Use:**
- Single-turn transactions (overkill complexity)
- Fully automated systems (humans prefer human interaction for social tasks)
- Domains where speed/cost dominates quality
- Scenarios requiring formal credentials/expertise

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `dimension_weights` | equal | Adjust for different emphasis (goals vs. relationships) |
| `group_baseline` | 0.5 | Relative advantage threshold; lower = easier improvement |
| `learning_rate` | 1e-5 | Standard LLM RL rate |
| `num_dialogue_turns` | 10-20 | Conversation length; longer = more nuanced interactions |

### Reference

**Paper**: Sotopia-RL: Reward Design for Social Intelligence (2508.03905)
- Multi-dimensional reward design for social tasks
- Utterance-level credit assignment with offline + online rewards
- Achieves 7.17 on Sotopia-hard and 8.31 on full benchmark
- Robust to different evaluators and partner models
