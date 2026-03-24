---
name: llm-forecasting-evaluation
title: "Evaluating LLMs on Real-World Forecasting Against Human Superforecasters"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04562"
keywords: [Forecasting, LLM Evaluation, Prediction, Human Comparison, Brier Score]
description: "Benchmark LLMs on real-world forecasting questions from Metaculus, comparing against human crowds and expert forecasters. Identifies which domains LLMs handle well and where they fall short relative to human intelligence."
---

# LLM Forecasting Evaluation: Assessing Predictive Reasoning on Out-of-Distribution Tasks

Evaluating whether LLMs can forecast future events reveals critical gaps in reasoning and planning capabilities. Unlike benchmarks that test memorization or in-distribution reasoning, forecasting requires understanding causal mechanisms, weighing uncertain information, and updating beliefs as new evidence emerges. This framework uses real Metaculus forecasting questions to assess frontier LLMs against human crowds and expert superforecasters—revealing that while frontier models exceed crowd performance, they dramatically underperform human experts.

The challenge is that forecasting is inherently out-of-distribution: answers depend on events that haven't occurred yet, cannot be contaminated in training data, and require genuine reasoning rather than pattern matching.

## Core Concept

The evaluation framework operates on three key insights:

1. **Real-world tasks matter**: Use actual forecasting questions from Metaculus (a tournament platform with incentive-aligned forecasters) rather than synthetic benchmarks.

2. **Multiple evaluation modes**: Compare frontier models under different prompting strategies to understand reasoning failure modes.

3. **Heterogeneous baselines**: Compare against both crowd forecasters (wisdom of crowds) and expert superforecasters (domain specialists) to understand capability gaps.

The Brier score (0 = perfect, 1 = worst possible) quantifies forecast accuracy and enables direct comparison across all forecasting attempts.

## Architecture Overview

- **Question dataset**: 464 real forecasting questions from July-December 2024 spanning politics, economics, healthcare, sports, technology
- **Information pipeline**: Integration with AskNews to provide relevant news articles for each question
- **Multiple model evaluation**: 12 frontier models including GPT-4o, o3, Claude variants, and DeepSeek
- **Prompting strategies**: Direct forecasting vs. narrative (script-based) prompting to test robustness
- **Aggregation method**: 5 predictions per question, averaged for stability
- **Human baselines**: Crowd forecaster aggregates and individual expert predictions from Metaculus

## Implementation

Set up the evaluation framework by loading questions and preparing news context:

```python
from forecasting_eval.data import MetaculusDataset
from forecasting_eval.context import NewsContextualizer

# Load real forecasting questions from Metaculus
dataset = MetaculusDataset(
    date_range=("2024-07-01", "2024-12-31"),
    n_questions=464
)

# Add contextual news articles for each question
contextualizer = NewsContextualizer(provider="AskNews")

for question in dataset:
    news_articles = contextualizer.fetch_context(
        query=question["title"],
        max_articles=5
    )
    question["context"] = news_articles
```

Implement direct forecasting evaluation with a frontier LLM:

```python
from forecasting_eval.models import LLMForecaster
from forecasting_eval.metrics import BrierScore

forecaster = LLMForecaster(model="gpt-4o")
scorer = BrierScore()

predictions = []

for question in dataset:
    # Use direct prompting: ask for probability estimate
    prompt = f"""
    Question: {question['title']}

    Context: {format_news(question['context'])}

    Based on current information, what is the probability this occurs?
    Provide a number between 0 and 1.
    """

    prediction = forecaster.forecast(prompt)  # Returns probability
    predictions.append(prediction)

# Calculate Brier score (lower is better)
brier = scorer.calculate(
    predictions=predictions,
    ground_truth=[q["outcome"] for q in dataset]
)

print(f"Direct forecasting Brier score: {brier:.4f}")
```

Evaluate using narrative (story-based) prompting to test robustness:

```python
# Narrative prompting: ask model to write reasoning, then extract forecast
predictions_narrative = []

for question in dataset:
    prompt = f"""
    Consider this question: {question['title']}

    Relevant context:
    {format_news(question['context'])}

    Write your reasoning about what will likely happen, then estimate
    the probability as a percentage.
    """

    response = forecaster.forecast(prompt)

    # Extract probability from narrative response
    prob = extract_probability(response)
    predictions_narrative.append(prob)

brier_narrative = scorer.calculate(
    predictions=predictions_narrative,
    ground_truth=[q["outcome"] for q in dataset]
)

print(f"Narrative prompting Brier score: {brier_narrative:.4f}")
```

Compare LLM performance against human baselines:

```python
from forecasting_eval.baselines import MetaculusBaselines

baselines = MetaculusBaselines()

# Get crowd forecast aggregates (wisdom of crowds)
crowd_predictions = baselines.get_crowd_predictions(dataset)
crowd_brier = scorer.calculate(crowd_predictions, ground_truth)

# Get expert superforecaster predictions
expert_predictions = baselines.get_expert_predictions(dataset)
expert_brier = scorer.calculate(expert_predictions, ground_truth)

print(f"LLM (GPT-4o) Brier: {brier:.4f}")
print(f"Crowd average Brier: {crowd_brier:.4f}")
print(f"Expert superforecasters Brier: {expert_brier:.4f}")

# Analyze performance by domain
domain_analysis = analyze_by_domain(
    predictions=predictions,
    ground_truth=dataset,
    domains=["politics", "economics", "healthcare", "sports"]
)

for domain, score in domain_analysis.items():
    print(f"{domain}: {score:.4f}")
```

## Practical Guidance

### When to Use This Framework

Use LLM forecasting evaluation for:
- Assessing whether frontier models improve reasoning capability
- Identifying specific domains where LLMs succeed or fail
- Testing robustness of prompting strategies
- Measuring progress as models evolve
- Understanding capability gaps between AI and human expertise
- Detecting potential contamination or overfitting on historical events

### When NOT to Use

Avoid forecasting for:
- Evaluating closed-vocabulary tasks
- Domains with no clear ground truth or outcomes
- Tasks where all training data is historical
- Real-time decision-making requiring human accountability
- Markets or applications where model deployment could influence outcomes

### Key Findings from Research

| Model | Brier Score | vs. Crowd | vs. Experts |
|-------|------------|----------|------------|
| o3 (frontier) | 0.1352 | Better | +11.2pp worse |
| GPT-4o | 0.1540 | Better | +13.2pp worse |
| Claude-3.5-Sonnet | 0.1680 | Similar | +14.6pp worse |
| Human crowd avg | 0.1490 | Baseline | +2.5pp worse |
| Expert superforecasters | 0.0230 | Much better | Baseline |

**Critical insight**: Frontier LLM o3 beats crowd average but lags expert forecasters by ~11 percentage points of Brier score—suggesting that domain expertise and empirical judgment remain superior to raw reasoning ability.

### Performance by Domain

LLMs typically perform better on:
- **Political questions** (more media coverage, clearer signals)
- **Technology predictions** (within AI understanding)
- **Binary outcomes** (easier than continuous estimates)

LLMs typically perform worse on:
- **Economic forecasts** (require subtle causal reasoning)
- **Scientific breakthrough predictions** (require deep domain knowledge)
- **Questions with evolving information** (require belief updating)

### Prompting Strategy Impact

| Prompting Strategy | Best For | Pitfall |
|-------------------|----------|---------|
| Direct probability | Speed and consistency | Anchoring effects |
| Narrative + extraction | Interpretability and verification | False reasoning in stories |
| Structured reasoning | Transparent multi-step reasoning | Length bias toward extremes |

**Warning**: Narrative prompting can degrade accuracy if models generate plausible-sounding but incorrect reasoning.

### Common Pitfalls

1. **Ignoring calibration**: Models may predict probabilities but be poorly calibrated (60% predictions occur 75% of the time).
2. **Contamination concerns**: Ensure questions weren't in training data. Use recent events (post-training cutoff).
3. **Overinterpreting single runs**: Forecasting has inherent randomness. Aggregate multiple predictions per question.
4. **Forgetting base rates**: Compare against simple baselines (uniform random, historical frequencies) before celebrating improvements.
5. **Missing error analysis**: Identify which question types cause model failures to understand capability gaps.

### Evaluation Checklist

- [ ] Questions span diverse domains and difficulty levels
- [ ] All questions are out-of-distribution (post-training)
- [ ] Ground truth is objectively verifiable
- [ ] Multiple predictions per question are aggregated
- [ ] Human baselines include crowds and experts
- [ ] Brier score is primary metric; secondary metrics capture coverage/calibration
- [ ] Failure modes are analyzed and documented

## Reference

"Evaluating LLMs on Real-World Forecasting Against Human Superforecasters" - [arXiv:2507.04562](https://arxiv.org/abs/2507.04562)
