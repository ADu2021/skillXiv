---
name: pretraining-midtraining-rl-interplay
title: "On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.07783
keywords: [reinforcement learning, reasoning, pre-training, mid-training, capability development]
description: "Understand when RL genuinely expands reasoning beyond pre-training through controlled experiments on synthetic tasks. Discover that RL works best at the edge of competence and process rewards reduce hacking—critical for designing effective reasoning model training."
---

## Overview

This work investigates whether RL in language models genuinely expands reasoning capabilities beyond what models acquire during initial training. Using synthetic reasoning tasks to isolate different training phases, the research reveals conditions under which RL generates real capability improvements versus when it merely optimizes existing knowledge.

## When to Use

- Designing RL post-training curricula for reasoning models
- Understanding contribution of pre-training versus RL to capabilities
- Scenarios where process-level rewards matter for reasoning integrity
- Determining whether additional RL training is worthwhile
- Models where RL seems to plateau or hit diminishing returns

## When NOT to Use

- Simple supervised fine-tuning scenarios
- Tasks where pre-training alone suffices
- Models with no room for RL improvements
- Situations where you need results immediately without careful analysis

## Core Technique

Controlled experimental framework isolating training phase contributions:

```python
# Analysis framework for training phase interactions
class TrainingPhaseAnalysis:
    def __init__(self):
        self.synthetic_tasks = SyntheticReasoningTasks()

    def study_rl_effectiveness(self, model, task_distribution):
        """
        Analyze when RL generates genuine capability improvements
        versus optimizing existing knowledge. Three key findings:
        1. RL needs adequate room for growth from pre-training
        2. RL targets "edge of competence" for effectiveness
        3. Generalization requires minimal but adequate pre-training exposure
        """
        results = {}

        # Dimension 1: Pre-training exposure level
        for exposure_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model_pretrained = self.pretrain_with_exposure(
                task_distribution,
                exposure_level
            )

            # RL training on this model
            model_rl = self.apply_rl(model_pretrained)

            # Measure extrapolative generalization
            # (composing learned atomic operations)
            extrap_improvement = self.measure_extrapolation_gain(
                model_pretrained,
                model_rl
            )

            results[f'exposure_{exposure_level}'] = extrap_improvement

        return results

    def analyze_competence_edge(self, model, tasks):
        """
        RL is most effective when targeting "edge of competence":
        tasks that are challenging but achievable with current knowledge.
        """
        performance_metrics = []

        for task in tasks:
            # Compute base performance
            base_perf = self.evaluate_model(model, task)

            # RL training focused on this task
            model_rl = self.apply_focused_rl(model, task)
            rl_perf = self.evaluate_model(model_rl, task)

            # Improvement is highest when base_perf is ~50% accuracy
            # (edge of competence)
            improvement = rl_perf - base_perf
            performance_metrics.append({
                'task': task,
                'base_performance': base_perf,
                'rl_performance': rl_perf,
                'improvement': improvement
            })

        return performance_metrics

    def study_midtraining_importance(self, dataset):
        """
        Mid-training significantly boosts performance relative to compute
        compared to RL-only approaches. Often overlooked in practice.
        """
        # Path 1: Pre-training only
        model_pretrain = self.pretrain(dataset)
        perf_pretrain = self.evaluate(model_pretrain)

        # Path 2: Pre-training + RL
        model_rl = self.apply_rl(model_pretrain)
        perf_rl = self.evaluate(model_rl)

        # Path 3: Pre-training + Mid-training + RL
        model_midtrain = self.midtrain(model_pretrain, dataset)
        model_full = self.apply_rl(model_midtrain)
        perf_full = self.evaluate(model_full)

        return {
            'pretrain_only': perf_pretrain,
            'rl_only': perf_rl,
            'full_pipeline': perf_full
        }

    def compare_reward_designs(self, model, tasks):
        """
        Process-level rewards reduce reward hacking and strengthen
        reasoning integrity vs outcome-only rewards.
        """
        # Outcome-only rewards (standard approach)
        model_outcome = self.train_with_outcome_rewards(model, tasks)
        perf_outcome = self.evaluate(model_outcome)
        integrity_outcome = self.measure_reasoning_integrity(model_outcome)

        # Process-level rewards (step-by-step correctness)
        model_process = self.train_with_process_rewards(model, tasks)
        perf_process = self.evaluate(model_process)
        integrity_process = self.measure_reasoning_integrity(model_process)

        return {
            'outcome_rewards': {
                'performance': perf_outcome,
                'integrity': integrity_outcome
            },
            'process_rewards': {
                'performance': perf_process,
                'integrity': integrity_process
            }
        }

    def generalization_analysis(self, model, contexts):
        """
        Contextual generalization requires minimal but adequate pre-training.
        Beyond this threshold, RL reliably transfers knowledge.
        """
        # Extrapolative: composing complex operations from atomic units
        extrapolative_gen = self.measure_extrapolative_generalization(
            model, contexts
        )

        # Contextual: applying knowledge across different surface contexts
        contextual_gen = self.measure_contextual_generalization(
            model, contexts
        )

        return {
            'extrapolative': extrapolative_gen,
            'contextual': contextual_gen
        }
```

Methodology employs synthetic reasoning tasks with explicit atomic operations and parseable step-by-step reasoning traces.

## Key Results

- RL effectiveness depends on pre-training exposure (need room for growth)
- Edge of competence is optimal RL target region
- Minimal pre-training sufficient for contextual generalization
- Process-level rewards outperform outcome-only
- Mid-training significantly boosts compute efficiency

## Implementation Notes

- Use synthetic tasks to isolate phase contributions
- Explicit reasoning traces enable process-level reward analysis
- Test at edge of competence for RL effectiveness
- Consider mid-training phase in pipeline design
- Process rewards recommended over outcome-only

## References

- Original paper: https://arxiv.org/abs/2512.07783
- Focus: Understanding RL contributions to reasoning
- Domain: Language model post-training, reinforcement learning
