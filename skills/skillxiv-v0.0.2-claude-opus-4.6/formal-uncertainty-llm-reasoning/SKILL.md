---
name: formal-uncertainty-llm-reasoning
title: "Grammars of Formal Uncertainty: When to Trust LLMs in Automated Reasoning Tasks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.20047"
keywords: [Uncertainty Quantification, Verification, Automated Reasoning, Formal Logic]
description: "Predict when LLM outputs are trustworthy for formal reasoning by analyzing domain-specific uncertainty signals."
---

# Know When to Trust Your LLM: Uncertainty for Formal Reasoning

LLMs generate confident-sounding outputs regardless of actual correctness, creating dangerous blind spots in automated reasoning where formal verification demands certainty. This skill teaches selective verification: analyzing uncertainty signals to predict which LLM-generated formalizations are worth verifying versus which are likely correct without checking. The approach discovers that different domains require different uncertainty metrics—entropy-based signals work for logic tasks but fail for factual problems, where token statistics matter more.

By learning task-dependent uncertainty patterns, you can reduce verification costs 14-100% while maintaining accuracy, transforming LLM-driven formalization from unreliable to deployable.

## Core Concept

Traditional confidence calibration treats all tasks the same. Formal uncertainty recognition is domain-aware: it learns that logic problems exhibit high uncertainty in specific token positions (decision points where multiple formalizations are valid), while factual reasoning tasks show uncertainty through different statistical patterns. A lightweight meta-model trained on your target domain learns which uncertainty signals reliably predict correctness, enabling selective formal verification that validates only high-risk outputs.

## Architecture Overview

- **Probabilistic Context-Free Grammar (PCFG) Framework**: Models the structure of LLM outputs to identify uncertainty signals at multiple linguistic levels (token, phrase, logical clause)
- **Domain-Specific Signal Extraction**: Task-dependent metrics including token probability entropy, grammatical complexity, logical branching points
- **Signal Fusion Module**: Lightweight combination mechanism that weights different uncertainty indicators based on domain
- **Selective Verification**: Routes high-uncertainty outputs to formal verification, passes low-uncertainty through
- **Iterative Calibration**: Improves signal weights over time as verification results accumulate

## Implementation

This implementation demonstrates uncertainty quantification and selective verification for formal reasoning tasks.

Build a framework to extract uncertainty signals from LLM outputs:

```python
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

class UncertaintyAnalyzer:
    """Extract uncertainty signals from LLM reasoning outputs."""

    def __init__(self, model_name: str = "gpt2-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def get_token_probabilities(self, text: str, tokenized_output: List[int]):
        """
        Get probability of each predicted token given context.
        Returns sequence of log probabilities.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Get probabilities for generated tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = []

        for i, token_id in enumerate(tokenized_output):
            if i < len(log_probs):
                token_log_probs.append(log_probs[i, token_id].item())

        return token_log_probs

    def compute_entropy(self, text: str) -> float:
        """
        Token-level entropy: measures uncertainty in LLM predictions.
        High entropy = multiple plausible continuations.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]

        # Compute entropy for each position
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        return float(entropy.mean().item())

    def compute_branching_points(self, text: str) -> List[Tuple[int, float]]:
        """
        Identify positions where LLM has high uncertainty about next token.
        These are logical branching points in reasoning.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]

        probs = torch.softmax(logits, dim=-1)

        # Find positions with high entropy (branching)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropy_threshold = entropy.mean() + entropy.std()

        branching_points = [
            (i, entropy[i].item())
            for i, e in enumerate(entropy)
            if e > entropy_threshold
        ]

        return branching_points

# Example usage
analyzer = UncertaintyAnalyzer()

reasoning_output = "To formalize this: ∃x (P(x) ∧ Q(x))"
entropy = analyzer.compute_entropy(reasoning_output)
branching_pts = analyzer.compute_branching_points(reasoning_output)

print(f"Average entropy: {entropy:.3f}")
print(f"Branching points: {len(branching_pts)}")
```

Implement domain-specific signal fusion for predicting correctness:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class DomainUncertaintyCalibrator:
    """Learn domain-specific mapping from uncertainty signals to correctness."""

    def __init__(self):
        self.signal_weights = {
            "entropy": 0.3,
            "token_probability": 0.2,
            "grammatical_complexity": 0.3,
            "logical_branching": 0.2
        }
        self.calibrator = LogisticRegression()
        self.scaler = StandardScaler()
        self.fitted = False

    def extract_signals(self, text: str, analyzer: UncertaintyAnalyzer) -> Dict:
        """Extract all uncertainty signals from a text."""
        signals = {}

        # Token-level entropy
        signals["entropy"] = analyzer.compute_entropy(text)

        # Average token probability
        tokens = analyzer.tokenizer.encode(text)
        log_probs = analyzer.get_token_probabilities(text[:100], tokens)
        signals["token_probability"] = -np.mean(log_probs) if log_probs else 0.5

        # Grammatical complexity (proxy: length variation)
        token_lengths = [len(analyzer.tokenizer.decode([t])) for t in tokens]
        signals["grammatical_complexity"] = np.std(token_lengths) / (np.mean(token_lengths) + 1e-8)

        # Logical branching density
        branching_pts = analyzer.compute_branching_points(text)
        signals["logical_branching"] = len(branching_pts) / (len(tokens) + 1e-8)

        return signals

    def fit_to_domain(self, texts: List[str], correctness_labels: List[int],
                     analyzer: UncertaintyAnalyzer):
        """
        Train domain-specific uncertainty calibration.
        Assumes correctness_labels: 1 = correct, 0 = incorrect
        """
        all_signals = []
        for text in texts:
            sig = self.extract_signals(text, analyzer)
            all_signals.append([
                sig["entropy"],
                sig["token_probability"],
                sig["grammatical_complexity"],
                sig["logical_branching"]
            ])

        X = np.array(all_signals)
        y = np.array(correctness_labels)

        # Normalize and fit
        X_scaled = self.scaler.fit_transform(X)
        self.calibrator.fit(X_scaled, y)

        # Update weights based on learned coefficients
        coef_abs = np.abs(self.calibrator.coef_[0])
        coef_sum = coef_abs.sum()
        self.signal_weights = {
            "entropy": coef_abs[0] / coef_sum,
            "token_probability": coef_abs[1] / coef_sum,
            "grammatical_complexity": coef_abs[2] / coef_sum,
            "logical_branching": coef_abs[3] / coef_sum
        }
        self.fitted = True

    def predict_correctness(self, text: str, analyzer: UncertaintyAnalyzer) -> Tuple[float, Dict]:
        """
        Predict probability that text is correct based on uncertainty signals.
        Returns (confidence, signal_dict).
        """
        signals = self.extract_signals(text, analyzer)

        X = np.array([[
            signals["entropy"],
            signals["token_probability"],
            signals["grammatical_complexity"],
            signals["logical_branching"]
        ]])

        if self.fitted:
            X_scaled = self.scaler.transform(X)
            confidence = self.calibrator.predict_proba(X_scaled)[0, 1]
        else:
            # Fallback: weighted combination
            confidence = sum(
                signals[key] * weight
                for key, weight in self.signal_weights.items()
            )
            confidence = min(1.0, max(0.0, confidence))

        return confidence, signals

# Example training and prediction
analyzer = UncertaintyAnalyzer()
calibrator = DomainUncertaintyCalibrator()

# Training data: logic formalization tasks
train_texts = [
    "All dogs are animals, so Fido is an animal.",
    "If P then Q. P is true. Therefore Q is true.",
]
train_labels = [1, 1]  # Both correct

# Fit domain calibrator
calibrator.fit_to_domain(train_texts, train_labels, analyzer)

# Predict on new reasoning
test_text = "Some birds fly. Therefore, all animals fly."
confidence, signals = calibrator.predict_correctness(test_text, analyzer)
print(f"Predicted correctness: {confidence:.2f}")
print(f"Signals: {signals}")
```

Implement selective verification: only formally check low-confidence outputs:

```python
class SelectiveVerifier:
    """Route LLM outputs to verification based on uncertainty."""

    def __init__(self, calibrator: DomainUncertaintyCalibrator,
                 formal_verifier_fn, confidence_threshold: float = 0.7):
        self.calibrator = calibrator
        self.formal_verifier = formal_verifier_fn  # Function that returns correctness
        self.threshold = confidence_threshold
        self.verification_cache = {}

    def process_reasoning(self, text: str, analyzer: UncertaintyAnalyzer) -> Dict:
        """
        Process reasoning output with selective verification.
        High confidence → skip verification, return predicted correct.
        Low confidence → formally verify.
        """
        # Predict correctness from uncertainty signals
        confidence, signals = self.calibrator.predict_correctness(text, analyzer)

        result = {
            "text": text,
            "predicted_confidence": confidence,
            "signals": signals,
            "verified": False,
            "actual_correctness": None
        }

        # Selective verification
        if confidence >= self.threshold:
            # High confidence: skip formal verification
            result["decision"] = "TRUST"
            result["actual_correctness"] = 1  # Assume correct
        else:
            # Low confidence: formally verify
            result["decision"] = "VERIFY"
            result["actual_correctness"] = self.formal_verifier(text)
            result["verified"] = True

        return result

# Example usage with mock verifier
def mock_formal_verifier(reasoning: str) -> int:
    """Placeholder formal verifier (e.g., SMT solver output)."""
    # In practice: run Z3, Dafny, or Coq to verify
    return 1 if "all" not in reasoning.lower() else 0

verifier = SelectiveVerifier(
    calibrator,
    mock_formal_verifier,
    confidence_threshold=0.7
)

test_reasoning = [
    "All birds have wings. Penguins are birds. Penguins have wings.",
    "Most flowers are red. This is a flower. This is red.",
]

for reasoning in test_reasoning:
    result = verifier.process_reasoning(reasoning, analyzer)
    print(f"Text: {result['text']}")
    print(f"Decision: {result['decision']}")
    if result['verified']:
        print(f"Formal verification result: {result['actual_correctness']}")
    print()
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **Confidence Threshold** | Start at 0.7; lower if verification is expensive, raise if cost of errors is high |
| **Domain Calibration Data** | 100-500 annotated examples sufficient; must be representative of target task |
| **Signal Extraction Cost** | ~10-50ms per text; negligible vs. formal verification (often seconds) |
| **Verification Backend** | Use SMT solvers (Z3), proof assistants (Coq), or domain-specific checkers |
| **Update Frequency** | Recalibrate quarterly or when error patterns shift |

**When to Use:**
- Formal reasoning with high cost of errors (code synthesis, theorem proving, policy formalization)
- Limited verification budget (can't check every output)
- Multiple reasoning domains with different uncertainty patterns
- Need transparency about which outputs are trustworthy

**When NOT to Use:**
- Real-time systems where added latency is critical (verification delay dominates)
- Tasks where all outputs must be verified anyway (no cost savings)
- Insufficient historical correctness data to calibrate domain signals
- Highly novel reasoning tasks where calibration patterns don't transfer

**Common Pitfalls:**
- Domain shift: uncertainty patterns from logic don't transfer to factual reasoning; recalibrate for each domain
- Overconfident predictions: if LLM is generally unreliable, signals become noise; invest in better base model first
- Verification coverage bias: if verification is biased (easier problems verified), calibration becomes skewed
- Ignoring model updates: retraining LLM changes uncertainty signals; recalibrate whenever base model changes

## Reference

Grammars of Formal Uncertainty: When to Trust LLMs in Automated Reasoning Tasks
https://arxiv.org/abs/2505.20047
