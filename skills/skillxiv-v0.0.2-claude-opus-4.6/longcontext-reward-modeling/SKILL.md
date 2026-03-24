---
name: longcontext-reward-modeling
title: "LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.06915"
keywords: [Reward Modeling, Long Context, Faithfulness, RL Training, Extended Sequences]
description: "Train reward models to evaluate long-context responses by introducing faithfulness as a criterion and using consistency-based alignment to maintain judgment-explanation coherence."
---

# Technique: Long-Context Reward Modeling via Faithfulness and Consistency Training

Reward models fail dramatically on long-context scenarios (4K+ tokens), with accuracy collapsing below 50% despite working well on short contexts. LongRM identifies two root causes: format compliance failures and judgment-explanation misalignment. By introducing faithfulness (grounding in provided context) and training via consistency voting, reward models scale to extended contexts.

The key insight is that long-context evaluation requires two explicit criteria: correctness and faithfulness. A response may be factually true but not grounded in the provided context, or it may be grounded but incorrect. Teaching models to evaluate both, with consistency checking between explicit judgments and explanations, enables reliable long-context evaluation.

## Core Concept

LongRM employs a two-stage training approach:

1. **Cold Start via SFT**: Introduce faithfulness criterion alongside standard correctness, teaching models to evaluate context grounding explicitly

2. **Fine-Grained Alignment via RL**: Use consistency-based majority voting to ensure judgment-explanation alignment

## Architecture Overview

- **Input**: Long-context scenario with response (up to 128K tokens)
- **Evaluation Criteria**: Both correctness and faithfulness judgments
- **Explanation Generation**: Produce reasoning for why response is (un)faithful
- **Consistency Checking**: Ensure explanations align with final judgments
- **Output**: Reliable reward signal for long-context RL

## Implementation Steps

Implement the dual-criterion evaluation framework.

```python
class LongContextRewardModel:
    def __init__(self, base_model):
        self.model = base_model

    def evaluate_long_context(self, context, response, verbose=False):
        """
        Evaluate both correctness and faithfulness.

        Args:
            context: Long-context document/conversation
            response: Response to evaluate
            verbose: Include explanation in output

        Returns:
            result: Dict with 'correctness', 'faithfulness', 'explanation'
        """

        # Evaluation prompt with both criteria
        eval_prompt = f"""Context:
{context}

Response:
{response}

Evaluate this response on two criteria:

1. CORRECTNESS: Is the response factually accurate?
2. FAITHFULNESS: Is the response grounded in the provided context?

First, provide your judgment:
CORRECTNESS: [YES/NO]
FAITHFULNESS: [YES/NO]

Then explain your reasoning:
EXPLANATION: [detailed reasoning]

Finally, provide an overall reward score [0.0 to 1.0]:
REWARD: [score]"""

        output = self.model.generate(eval_prompt)

        # Parse structured output
        correctness = self._extract_boolean(output, 'CORRECTNESS')
        faithfulness = self._extract_boolean(output, 'FAITHFULNESS')
        explanation = self._extract_text(output, 'EXPLANATION')
        reward = self._extract_float(output, 'REWARD')

        return {
            'correctness': correctness,
            'faithfulness': faithfulness,
            'explanation': explanation,
            'reward': reward
        }

    def _extract_boolean(self, text, key):
        """Extract YES/NO judgment."""
        search_str = f"{key}: "
        if search_str in text:
            value = text.split(search_str)[1].split('\n')[0].strip()
            return value.upper() == 'YES'
        return None

    def _extract_text(self, text, key):
        """Extract explanation text."""
        search_str = f"{key}: "
        if search_str in text:
            value = text.split(search_str)[1].split('\n')[0].strip()
            return value
        return ""

    def _extract_float(self, text, key):
        """Extract float reward score."""
        search_str = f"{key}: "
        if search_str in text:
            try:
                value = float(text.split(search_str)[1].split('\n')[0].strip())
                return value
            except:
                return 0.5
        return 0.5
```

Implement consistency-based training via majority voting.

```python
def train_consistency_aligned_rm(rm_model, training_data, optimizer,
                                num_epochs=3, num_votes=3):
    """
    Train reward model with consistency-based alignment.

    Args:
        rm_model: LongContextRewardModel instance
        training_data: List of (context, response, label) tuples
        optimizer: PyTorch optimizer
        num_epochs: Training epochs
        num_votes: Number of evaluations for consistency voting

    Returns:
        losses: Training loss curve
    """

    import torch
    import torch.nn.functional as F

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for context, response, ground_truth in training_data:
            # Get multiple evaluations for consistency voting
            evaluations = []
            for _ in range(num_votes):
                eval_result = rm_model.evaluate_long_context(
                    context, response, verbose=True
                )
                evaluations.append(eval_result)

            # Majority voting on judgments
            correctness_votes = sum(1 for e in evaluations if e['correctness'])
            faithfulness_votes = sum(1 for e in evaluations if e['faithfulness'])

            consensus_correctness = correctness_votes >= (num_votes / 2)
            consensus_faithfulness = faithfulness_votes >= (num_votes / 2)

            # Check if consensus matches ground truth
            matches_gt = (consensus_correctness == ground_truth['correctness'] and
                         consensus_faithfulness == ground_truth['faithfulness'])

            # Loss: maximize likelihood of consensus judgments
            # when they match ground truth
            loss = -1.0 if matches_gt else 0.1

            # Backprop through the model
            # (Simplified; actual implementation would use policy gradients)

            epoch_loss += abs(loss)

        losses.append(epoch_loss / len(training_data))
        print(f"Epoch {epoch+1}: Consistency Loss={losses[-1]:.4f}")

    return losses
```

Implement long-context adaptation training.

```python
def sft_on_faithfulness(rm_model, training_examples, optimizer, num_epochs=3):
    """
    Supervised fine-tuning on faithfulness criterion.

    Args:
        rm_model: LongContextRewardModel
        training_examples: List of (context, response, is_faithful, is_correct)
        optimizer: PyTorch optimizer
        num_epochs: Training epochs

    Returns:
        losses: Training loss curve
    """

    import torch
    import torch.nn.functional as F

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for context, response, is_faithful, is_correct in training_examples:
            # Generate with both criteria in prompt
            prompt = f"""Evaluate faithfulness and correctness.

Context: {context[:2000]}...  # Truncate for efficiency

Response: {response}

FAITHFULNESS (grounded in context): [1=YES, 0=NO]
CORRECTNESS (factually accurate): [1=YES, 0=NO]"""

            # Get model logits for both judgments
            output = rm_model.model.generate(prompt, return_logits=True)

            faith_logits = output['faithfulness_logits']
            correct_logits = output['correctness_logits']

            # Compute cross-entropy loss
            faith_loss = F.binary_cross_entropy_with_logits(
                faith_logits, torch.tensor([float(is_faithful)])
            )
            correct_loss = F.binary_cross_entropy_with_logits(
                correct_logits, torch.tensor([float(is_correct)])
            )

            total_loss = faith_loss + correct_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        losses.append(epoch_loss / len(training_examples))

    return losses
```

Implement long-context benchmarking.

```python
def evaluate_long_context_rm(rm_model, test_data, max_context_length=4096):
    """
    Evaluate reward model on long-context scenarios.

    Args:
        rm_model: LongContextRewardModel
        test_data: List of (context, response, label) tuples
        max_context_length: Maximum context length to test

    Returns:
        metrics: Performance metrics by context length
    """

    results = {'short': [], 'medium': [], 'long': []}

    for context, response, label in test_data:
        context_length = len(context.split())

        # Evaluate
        evaluation = rm_model.evaluate_long_context(context, response)

        # Check correctness
        correct = (evaluation['correctness'] == label['correctness'])

        # Categorize by length
        if context_length < 500:
            results['short'].append(correct)
        elif context_length < 2000:
            results['medium'].append(correct)
        else:
            results['long'].append(correct)

    # Compute accuracies
    metrics = {
        'short_acc': sum(results['short']) / len(results['short']) if results['short'] else 0,
        'medium_acc': sum(results['medium']) / len(results['medium']) if results['medium'] else 0,
        'long_acc': sum(results['long']) / len(results['long']) if results['long'] else 0,
    }

    return metrics
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Faithfulness weight | Equal to correctness | Both criteria essential for long-context; don't subsume |
| Consistency voting | 3-5 votes per example | More votes improve robustness; balance with training cost |
| Context truncation | 4K-8K token windows | Longer contexts slower but more realistic |
| Training data | Mix short and long contexts | Ensure generalization across context lengths |
| When to use | Long-document evaluation or LLM critique | Research papers, conversations, transcripts |
| When NOT to use | Short-context tasks where standard RMs work | Overhead of faithfulness training not justified |
| Common pitfall | Faithfulness and correctness conflation | Keep separate; a response can be faithful but incorrect |

### When to Use LongRM

- Evaluating long-form responses (research summaries, conversations, articles)
- RL training on extended contexts (book summarization, document analysis)
- Scenarios requiring grounding in provided context
- Multi-document reasoning tasks

### When NOT to Use LongRM

- Short-context evaluation where standard RMs work well
- Tasks without clear context-response relationship
- Real-time evaluation where computational cost is critical

### Common Pitfalls

- **Context truncation issues**: Losing important information at boundaries; use careful windowing
- **Faithfulness overfitting**: Model learns dataset-specific patterns rather than generalized faithfulness
- **Judgment inconsistency**: Explanations contradict judgments; enforce consistency losses
- **Long-context bias**: Models may favor short responses to avoid faithfulness penalties; monitor response length distribution

## Reference

Paper: https://arxiv.org/abs/2510.06915
