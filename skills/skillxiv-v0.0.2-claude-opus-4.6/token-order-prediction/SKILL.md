---
name: token-order-prediction
title: Predicting Token Order Improves Language Model Performance
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.19228
keywords: [auxiliary-objective, token-ordering, learning-to-rank, language-modeling, multi-token-prediction]
description: "Improve LM performance with token order prediction (TOP) auxiliary loss using learning-to-rank instead of exact multi-token prediction, achieving gains across math, code, and NLP tasks"
---

# Predicting the Order of Upcoming Tokens Improves Language Models

## Core Concept

Token Order Prediction (TOP) is a lightweight auxiliary training objective that teaches models to rank upcoming tokens by proximity. Instead of predicting exact future tokens (which is difficult), TOP uses learning-to-rank loss to teach the model to understand token sequencing. This simple auxiliary objective improves performance across mathematics, coding, and standard NLP benchmarks while requiring only a single additional unembedding layer.

## Architecture Overview

- **Learning-to-Rank Loss**: Rank tokens by distance rather than exact prediction
- **Single Extra Layer**: Minimal architectural overhead vs. multi-token prediction
- **Auxiliary Objective**: Combined with next-token prediction during training
- **Task Generalization**: Benefits across math, code, and general NLP
- **Efficiency**: No increase in inference latency or model size

## Implementation Steps

### Stage 1: Design Token Order Prediction Objective

Formulate the learning-to-rank loss for token ordering.

```python
# Token order prediction: learning-to-rank objective
import torch
from torch import nn
import torch.nn.functional as F

class TokenOrderPredictionHead(nn.Module):
    """Predict ordering of upcoming tokens"""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_future_tokens: int = 5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_future = num_future_tokens

        # Projection from hidden state to ranking scores
        self.ranking_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking scores for upcoming tokens.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            future_tokens: [batch, seq_len, num_future]

        Returns:
            ranking_scores: [batch, seq_len, vocab_size]
        """
        # Project hidden states to token scores
        scores = self.ranking_head(hidden_states)  # [batch, seq_len, vocab_size]

        return scores


class TopLoss(nn.Module):
    """Learning-to-rank loss for token ordering"""

    def __init__(self, loss_type: str = "listwise"):
        """
        Args:
            loss_type: "listwise", "pairwise", or "pointwise"
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predicted_scores: torch.Tensor,  # [batch*seq_len, vocab_size]
        target_tokens: torch.Tensor,     # [batch*seq_len, num_future]
        target_positions: torch.Tensor   # [batch*seq_len, num_future] - distance from current
    ) -> torch.Tensor:
        """
        Compute learning-to-rank loss.

        The model should score tokens higher if they appear soon.
        """
        if self.loss_type == "listwise":
            return self._listwise_loss(predicted_scores, target_tokens, target_positions)
        elif self.loss_type == "pairwise":
            return self._pairwise_loss(predicted_scores, target_tokens, target_positions)
        else:
            return self._pointwise_loss(predicted_scores, target_tokens, target_positions)

    def _listwise_loss(
        self,
        predicted_scores: torch.Tensor,
        target_tokens: torch.Tensor,
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Listwise loss: maximize probability of correct ranking.

        Tokens appearing sooner should have higher scores.
        """
        batch_size = predicted_scores.shape[0]
        loss = 0

        for i in range(batch_size):
            scores = predicted_scores[i]  # [vocab_size]
            tokens = target_tokens[i]      # [num_future]
            positions = target_positions[i]  # [num_future]

            # Get scores for target tokens
            token_scores = scores[tokens]

            # Closer tokens should have higher scores
            # Create preference pairs
            for j in range(len(tokens)):
                for k in range(j + 1, len(tokens)):
                    # Token at position j should have higher score if closer (smaller position)
                    if positions[j] < positions[k]:
                        # score[j] should be > score[k]
                        margin = token_scores[j] - token_scores[k]
                    else:
                        # score[k] should be > score[j]
                        margin = token_scores[k] - token_scores[j]

                    # Hinge-like loss
                    loss += F.relu(1.0 - margin)

        return loss / batch_size

    def _pairwise_loss(
        self,
        predicted_scores: torch.Tensor,
        target_tokens: torch.Tensor,
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Pairwise loss: compare tokens at different distances.
        """
        batch_size = predicted_scores.shape[0]

        # Get scores for target tokens
        batch_idx = torch.arange(batch_size).unsqueeze(1)
        token_scores = predicted_scores[batch_idx, target_tokens]  # [batch, num_future]

        # Create pairs: closer token vs farther token
        # Closer should have higher score
        loss = 0
        for dist_close in range(target_positions.shape[1]):
            for dist_far in range(dist_close + 1, target_positions.shape[1]):
                score_close = token_scores[:, dist_close]
                score_far = token_scores[:, dist_far]

                # Loss: margin between close and far
                margin_loss = F.relu(1.0 - (score_close - score_far))
                loss += margin_loss.mean()

        return loss

    def _pointwise_loss(
        self,
        predicted_scores: torch.Tensor,
        target_tokens: torch.Tensor,
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Pointwise loss: score tokens relative to position.
        """
        batch_size = predicted_scores.shape[0]

        # Get scores for target tokens
        batch_idx = torch.arange(batch_size).unsqueeze(1)
        token_scores = predicted_scores[batch_idx, target_tokens]

        # Target: closer tokens should have higher scores
        # Invert positions (smaller distance = larger target score)
        max_pos = target_positions.float().max()
        position_targets = (max_pos - target_positions.float()) / max_pos

        # Regression loss
        loss = F.mse_loss(token_scores, position_targets)

        return loss
```

### Stage 2: Integrate TOP with Next-Token Prediction

Combine TOP auxiliary objective with standard language modeling.

```python
# Combined training with TOP auxiliary objective
class LanguageModelWithTOP(nn.Module):
    """LM with top-k token order prediction"""

    def __init__(
        self,
        hidden_dim: int = 4096,
        vocab_size: int = 32000,
        num_layers: int = 32
    ):
        super().__init__()

        # Standard transformer LM
        self.transformer = TransformerLM(hidden_dim, vocab_size, num_layers)

        # Next-token prediction head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Token order prediction head
        self.top_head = TokenOrderPredictionHead(hidden_dim, vocab_size)

        self.top_loss_fn = TopLoss(loss_type="pairwise")

    def forward(
        self,
        input_ids: torch.Tensor,
        future_token_ids: torch.Tensor = None
    ) -> Dict:
        """
        Forward pass with both NTP and TOP.

        Args:
            input_ids: [batch, seq_len]
            future_token_ids: [batch, seq_len, num_future] (for TOP)

        Returns:
            ntp_logits, top_scores
        """
        # Get hidden states
        hidden_states = self.transformer(input_ids)

        # Next-token prediction
        ntp_logits = self.lm_head(hidden_states)

        # Token order prediction (if provided)
        top_scores = None
        if future_token_ids is not None:
            top_scores = self.top_head(hidden_states, future_token_ids)

        return {
            "ntp_logits": ntp_logits,
            "top_scores": top_scores,
            "hidden_states": hidden_states
        }


class CombinedTrainingLoop:
    """Train with NTP + TOP objectives"""

    def __init__(
        self,
        model: LanguageModelWithTOP,
        lr: float = 1e-4,
        top_weight: float = 0.1
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.ntp_loss_fn = nn.CrossEntropyLoss()
        self.top_loss_fn = TopLoss(loss_type="pairwise")
        self.top_weight = top_weight

    def extract_future_tokens(
        self,
        input_ids: torch.Tensor,
        num_future: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract future tokens and their distances.

        Returns:
            future_token_ids: [batch, seq_len, num_future]
            future_positions: [batch, seq_len, num_future]
        """
        batch_size, seq_len = input_ids.shape

        future_tokens = torch.zeros(
            batch_size, seq_len, num_future,
            dtype=torch.long, device=input_ids.device
        )
        future_positions = torch.zeros_like(future_tokens)

        for i in range(seq_len):
            for j in range(num_future):
                if i + j + 1 < seq_len:
                    future_tokens[:, i, j] = input_ids[:, i + j + 1]
                    future_positions[:, i, j] = j + 1

        return future_tokens, future_positions

    def train_step(self, batch: Dict) -> Dict:
        """Single training step with NTP + TOP"""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Extract future tokens for TOP
        future_tokens, future_positions = self.extract_future_tokens(input_ids)

        # Forward pass
        outputs = self.model(input_ids, future_tokens)
        ntp_logits = outputs["ntp_logits"]
        top_scores = outputs["top_scores"]

        # Next-token prediction loss
        ntp_loss = self.ntp_loss_fn(
            ntp_logits.view(-1, ntp_logits.shape[-1]),
            input_ids.view(-1)
        )

        # Token order prediction loss
        top_loss = 0
        if top_scores is not None:
            top_loss = self.top_loss_fn(
                top_scores.view(-1, top_scores.shape[-1]),
                future_tokens.view(-1, future_tokens.shape[-1]),
                future_positions.view(-1, future_positions.shape[-1])
            )

        # Combined loss
        total_loss = ntp_loss + self.top_weight * top_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "ntp_loss": ntp_loss.item(),
            "top_loss": top_loss.item() if isinstance(top_loss, torch.Tensor) else top_loss
        }

    def set_top_weight(self, weight: float):
        """Adjust TOP weight (useful for curriculum learning)"""
        self.top_weight = weight
```

### Stage 3: Evaluation on Benchmarks

Test TOP across diverse tasks.

```python
# Benchmark evaluation
class TOPEvaluator:
    """Evaluate TOP improvements"""

    def __init__(self, model_with_top: LanguageModelWithTOP):
        self.model = model_with_top

    def evaluate_on_benchmarks(self) -> Dict:
        """
        Evaluate on standard benchmarks.
        Reproduction of paper results.
        """
        benchmarks = {
            "arc_challenge": self.evaluate_benchmark("arc_challenge"),
            "hellaswag": self.evaluate_benchmark("hellaswag"),
            "mmlu": self.evaluate_benchmark("mmlu"),
            "gsm8k": self.evaluate_benchmark("gsm8k"),
            "math": self.evaluate_benchmark("math"),
            "humanevals": self.evaluate_benchmark("humanevals"),
            "mbpp": self.evaluate_benchmark("mbpp")
        }

        return benchmarks

    def evaluate_benchmark(self, benchmark_name: str) -> Dict:
        """Evaluate on single benchmark"""
        # Load test set
        test_set = self.load_benchmark_data(benchmark_name)

        correct = 0
        total = len(test_set)

        for example in test_set:
            prompt = example["prompt"]
            answer = example["answer"]

            # Generate
            generated = self.generate(prompt, max_length=256)

            # Check correctness
            if self.check_correctness(generated, answer, benchmark_name):
                correct += 1

        accuracy = correct / total

        return {
            "benchmark": benchmark_name,
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }

    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate with model"""
        tokens = self.model.transformer.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0)

        for _ in range(max_length):
            outputs = self.model(tokens)
            logits = outputs["ntp_logits"][:, -1, :]
            next_token = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

        return self.model.transformer.tokenizer.decode(tokens[0])

    def check_correctness(self, generated: str, reference: str, benchmark: str) -> bool:
        """Check if generation matches reference"""
        if benchmark in ["gsm8k", "math"]:
            # Extract final number
            gen_num = self.extract_number(generated)
            ref_num = self.extract_number(reference)
            return gen_num == ref_num

        else:
            # Exact match or substring
            return reference in generated

    def extract_number(self, text: str) -> float:
        """Extract final number from text"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])
        return None

    def load_benchmark_data(self, name: str):
        """Load benchmark data"""
        # Placeholder: in practice, load from HuggingFace
        return []
```

## Practical Guidance

### Hyperparameters

- **TOP Weight**: 0.1-0.3 relative to NTP loss (higher = more TOP emphasis)
- **Future Tokens**: 5 is optimal (further tokens have diminishing signal)
- **Loss Type**: Pairwise generally works best
- **Training Duration**: TOP converges quickly; add after 10-20% of pretraining

### When to Use TOP

- Improving language model performance across diverse tasks
- Math and coding tasks (TOP particularly helps here)
- General pretraining where you want better few-shot performance
- Resource-constrained training (minimal overhead)

### When NOT to Use

- Models already performing optimally on target tasks
- Very long-context scenarios (future token extraction becomes expensive)
- Real-time systems where any overhead matters

### Performance Expectations

- Math benchmarks: +2-4% improvement
- Coding benchmarks: +1-3% improvement
- General NLP: +0.5-2% improvement
- Computational overhead: <5% training time increase

### Design Insights

TOP works because predicting exact future tokens is a harder auxiliary task than necessary. By reformulating as a ranking problem ("which tokens appear first?"), the model learns token sequencing patterns without the regression difficulty. This is a goldilocks auxiliary objective—harder than next-token only, but easier than exact multi-token prediction.

## Reference

Predicting the Order of Upcoming Tokens Improves Language Models. arXiv:2508.19228
- https://arxiv.org/abs/2508.19228
