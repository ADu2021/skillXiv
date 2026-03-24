---
name: timeseries-scientist-agentic-forecasting
title: "TimeSeriesScientist: Autonomous Time Series Forecasting via Multi-Agent Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.01538
keywords: [time-series, agents, forecasting, reasoning, automation]
description: "Automate univariate time series forecasting through a four-agent system orchestrating preprocessing, model selection, validation, and reporting. Use when reducing manual effort in forecasting pipelines and improving reproducibility."
---

# TimeSeriesScientist: Autonomous Time Series Forecasting via Multi-Agent Reasoning

TimeSeriesScientist introduces the first end-to-end agentic framework automating univariate time series forecasting through specialized agents handling preprocessing, model selection, validation, and reporting. The approach achieves 38.2% error reduction versus pure LLM baselines.

## Core Architecture

- **Four specialized agents**: Curator, Planner, Forecaster, Reporter
- **Curator agent**: Data preprocessing and outlier detection
- **Planner agent**: Model selection and hyperparameter configuration
- **Forecaster agent**: Ensemble forecasting with validation
- **Reporter agent**: Result summarization and uncertainty quantification
- **21 model implementations**: Diverse algorithms from statistical to neural

## Implementation Steps

Setup multi-agent forecasting system:

```python
# Initialize TimeSeriesScientist framework
from timeseries_scientist import ForecastingMAS, Agent, PreprocessingPipeline

# Create specialized agents
curator = Agent(
    role="data_curator",
    capabilities=["outlier_detection", "missing_value_handling", "detrending", "deseasonalization"],
    model="gpt-4o"
)

planner = Agent(
    role="model_planner",
    capabilities=["model_selection", "hyperparameter_tuning", "ensemble_design"],
    model="gpt-4o"
)

forecaster = Agent(
    role="forecaster",
    capabilities=["forecast_generation", "uncertainty_quantification", "ensemble_combination"],
    model="gpt-4o"
)

reporter = Agent(
    role="report_generator",
    capabilities=["result_summarization", "insight_extraction", "limitation_discussion"],
    model="gpt-4o"
)

# Initialize multi-agent orchestrator
mas = ForecastingMAS(
    agents=[curator, planner, forecaster, reporter],
    models_available=21,  # statistical, ML, neural network models
    ensemble_strategy="weighted_average"
)
```

Execute end-to-end forecasting pipeline:

```python
# Run autonomous forecasting system
time_series_data = load_data("univariate_series.csv")

# Stage 1: Data curation via curator agent
curation_report = curator.execute(
    data=time_series_data,
    tasks={
        "detect_outliers": True,
        "handle_missing": True,
        "assess_trend": True,
        "assess_seasonality": True
    }
)

preprocessed_data = curation_report["processed_data"]
data_insights = curation_report["insights"]

# Stage 2: Model planning via planner agent
plan = planner.execute(
    data=preprocessed_data,
    insights=data_insights,
    tasks={
        "select_models": True,
        "configure_hyperparameters": True,
        "design_ensemble": True
    }
)

# Selected models: e.g., [ARIMA, Prophet, LSTM, XGBoost]
selected_models = plan["selected_models"]
hyperparameters = plan["hyperparameters"]

# Stage 3: Forecasting via forecaster agent
forecast_result = forecaster.execute(
    data=preprocessed_data,
    models=selected_models,
    hyperparameters=hyperparameters,
    tasks={
        "train_models": True,
        "generate_forecasts": True,
        "ensemble_combination": True,
        "compute_uncertainty": True
    }
)

# Get forecast and uncertainty bounds
forecast = forecast_result["forecast"]
confidence_intervals = forecast_result["confidence_intervals"]
individual_forecasts = forecast_result["individual_model_forecasts"]

# Stage 4: Reporting via reporter agent
report = reporter.execute(
    forecast=forecast,
    confidence_intervals=confidence_intervals,
    data_insights=data_insights,
    plan=plan,
    tasks={
        "summarize_results": True,
        "extract_insights": True,
        "discuss_limitations": True,
        "provide_recommendations": True
    }
)

print(report["summary"])
```

## Practical Guidance

**When to use TimeSeriesScientist:**
- Automating routine univariate forecasting workflows
- Reducing manual effort in preprocessing and model selection
- Improving reproducibility and documentation quality
- Scenarios where LLM reasoning can add value (e.g., seasonal pattern detection)
- Multi-step pipelines where agent orchestration provides organization

**When NOT to use:**
- Multivariate forecasting (framework handles univariate only)
- Real-time streaming forecasting (batch processing model)
- Scenarios requiring domain-expert judgment beyond LLM reasoning
- High-frequency trading requiring sub-second latency
- Tasks where simple statistical methods already sufficient

**Model coverage (21 implementations):**
- **Statistical**: ARIMA, SARIMA, Exponential Smoothing, Prophet
- **Machine Learning**: Random Forest, Gradient Boosting (XGBoost, LightGBM)
- **Neural Networks**: LSTM, GRU, Transformer-based, Temporal Convolutional Networks
- **Hybrid**: Statistical + neural combinations
- **Ensemble**: Multiple combinations of above

**Hyperparameters:**
- **Forecast horizon**: Default 12 (1 year for monthly data); adjust to task scale
- **Train/test split**: Default 80/20; adjust for seasonal patterns (keep multiple full seasons)
- **Ensemble method**: "weighted_average" standard; test "median" for robustness
- **Uncertainty quantification**: "quantile_regression" for distribution-free bounds
- **Agent model**: gpt-4o standard; test gpt-4-turbo for cost reduction

## Performance Metrics

- **38.2% error reduction** vs. pure LLM baseline
- **Consistency across 8 benchmarks**: ARIMA, AutoTS, M4, Tourism, Electricity, Weather, Stock, Kaggle
- **Robustness**: Handles diverse time series characteristics (trend, seasonality, noise)

## Agent Specialization Benefits

Each agent contributes uniquely:
- **Curator**: Reduces noise; improves data quality signal
- **Planner**: Optimal model selection reduces trial-and-error
- **Forecaster**: Ensemble combination exploits diversity; uncertainty bounds add value
- **Reporter**: Human-readable insights improve decision-making

## Key Implementation Details

The framework implements:
- Automated model parameter search over discrete spaces
- Ensemble weighting based on validation performance
- Uncertainty quantification via bootstrap and quantile methods
- Natural language reasoning about data characteristics

## References

Builds on multi-agent systems literature, automated machine learning, and time series forecasting practices.
