---
name: dover-auto-debugging
title: "DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.06749
keywords: [multi-agent systems, auto-debugging, intervention-driven, failure recovery, agent coordination]
description: "Diagnose and fix multi-agent system failures through targeted interventions (message edits, plan changes) rather than static log analysis. DoVer recovers 18-28% of failed trials with 30-60% hypothesis validation—essential for autonomous multi-agent reliability."
---

## Overview

DoVer augments traditional log-based debugging with active verification through targeted system interventions. Rather than accepting single-point attributions, the framework systematically tests modifications to agent communications and planning to determine which changes resolve failures, providing practical mechanisms for improving multi-agent reliability.

## When to Use

- Multi-agent systems experiencing task failures
- Debugging attribution is uncertain or insufficient
- Need to determine which modifications resolve failures
- Testing multiple potential fixes for the same failure
- Applications where failure recovery is critical
- Autonomous agent systems requiring reliability improvement

## When NOT to Use

- Single-agent systems or simple pipelines
- Deterministic workflows without failure modes
- Cases where static log analysis suffices
- Real-time systems where intervention overhead is unacceptable
- Scenarios where testing modifications on live systems is risky

## Core Technique

Hypothesis generation and verification through targeted interventions:

```python
# Intervention-Driven Debugging Framework
class DoVerDebugger:
    def __init__(self, agent_framework):
        self.framework = agent_framework  # e.g., AG2, Anthropic Framework
        self.failed_trials = []

    def analyze_failure(self, trial):
        """
        Analyze failed trial to generate debugging hypotheses.
        Goes beyond log analysis to test interventions.
        """
        hypothesis_candidates = []

        # Extract relevant context
        history = trial.execution_trace
        agents = trial.agents
        final_state = trial.final_state
        task = trial.task

        # Hypothesis 1: Agent A made incorrect decision
        for agent in agents:
            hypothesis = {
                'type': 'agent_decision',
                'agent': agent,
                'hypothesis': f"Agent {agent.name} made suboptimal decision"
            }
            hypothesis_candidates.append(hypothesis)

        # Hypothesis 2: Communication failure
        for agent_pair in self.get_agent_pairs(agents):
            hypothesis = {
                'type': 'communication',
                'agents': agent_pair,
                'hypothesis': f"Communication between {agent_pair} failed"
            }
            hypothesis_candidates.append(hypothesis)

        # Hypothesis 3: Plan was suboptimal
        hypothesis = {
            'type': 'plan',
            'hypothesis': 'Task decomposition or planning was incorrect'
        }
        hypothesis_candidates.append(hypothesis)

        return hypothesis_candidates

    def test_hypothesis_via_intervention(self, trial, hypothesis):
        """
        Verify hypothesis by intervening in system and observing outcome.
        Multiple modification types enable comprehensive testing.
        """
        if hypothesis['type'] == 'agent_decision':
            # Intervention: Suggest alternative action to agent
            return self.test_agent_intervention(trial, hypothesis)

        elif hypothesis['type'] == 'communication':
            # Intervention: Edit messages between agents
            return self.test_message_intervention(trial, hypothesis)

        elif hypothesis['type'] == 'plan':
            # Intervention: Alter task decomposition
            return self.test_plan_intervention(trial, hypothesis)

    def test_agent_intervention(self, trial, hypothesis):
        """
        Re-run trial with suggested action changes for problem agent.
        """
        agent = hypothesis['agent']
        original_trial = trial

        # Generate alternative actions agent could take
        alternatives = self.generate_alternative_actions(
            agent,
            original_trial.execution_trace
        )

        results = []
        for alternative_action in alternatives:
            # Run modified trial with alternative action
            modified_trial = self.run_with_intervention(
                original_trial,
                agent,
                alternative_action
            )

            # Check if intervention resolved failure
            success = modified_trial.completed_successfully
            results.append({
                'intervention': alternative_action,
                'outcome': 'success' if success else 'failure',
                'trial': modified_trial
            })

        return results

    def test_message_intervention(self, trial, hypothesis):
        """
        Test whether editing agent messages resolves failure.
        """
        agent_pair = hypothesis['agents']
        agent_a, agent_b = agent_pair

        # Identify messages between agents
        messages = self.extract_messages(
            trial.execution_trace,
            agent_a,
            agent_b
        )

        results = []
        for message in messages:
            # Generate improved message versions
            improved_messages = self.improve_message(message)

            for improved_msg in improved_messages:
                # Re-run trial with modified message
                modified_trial = self.run_with_message_edit(
                    trial,
                    message,
                    improved_msg
                )

                success = modified_trial.completed_successfully
                results.append({
                    'original_message': message,
                    'improved_message': improved_msg,
                    'outcome': 'success' if success else 'failure'
                })

        return results

    def test_plan_intervention(self, trial, hypothesis):
        """
        Test if altering task decomposition/plan resolves failure.
        """
        original_plan = trial.plan
        problem_stage = self.identify_problem_stage(trial)

        # Generate alternative decompositions
        alternative_plans = self.generate_alternative_plans(
            trial.task,
            original_plan,
            problem_stage
        )

        results = []
        for alt_plan in alternative_plans:
            # Re-run trial with modified plan
            modified_trial = self.run_with_plan_intervention(
                trial,
                alt_plan
            )

            success = modified_trial.completed_successfully
            progress = self.measure_progress(modified_trial)

            results.append({
                'alternative_plan': alt_plan,
                'outcome': 'success' if success else 'failure',
                'progress': progress
            })

        return results

    def validate_hypothesis(self, hypothesis, intervention_results):
        """
        Determine if hypothesis is validated/refuted based on intervention results.
        Focus on outcomes rather than single causation.
        """
        successful_interventions = [
            r for r in intervention_results
            if r['outcome'] == 'success'
        ]

        if len(successful_interventions) > 0:
            return 'validated', successful_interventions
        else:
            # Check for partial progress
            progress_results = [
                r for r in intervention_results
                if r.get('progress', 0) > 0
            ]
            if len(progress_results) > 0:
                return 'partial', progress_results
            else:
                return 'refuted', []

    def apply_successful_intervention(self, trial_id, successful_intervention):
        """
        Apply validated intervention to recover from failure.
        """
        self.framework.apply_intervention(
            trial_id,
            successful_intervention
        )

        # Re-run task with modification
        recovered_trial = self.framework.retry_with_intervention(
            trial_id,
            successful_intervention
        )

        return recovered_trial
```

Outcome-oriented evaluation measures success-focused metrics: failure recovery rate and progress toward task completion.

## Key Results

- 18-28% recovery of failed trials across GAIA and AssistantBench
- 30-60% of hypotheses validated or refuted
- 49% recovery rate on GSMPlus with AG2 agent framework
- Generalizable across multiple agent frameworks

## Implementation Notes

- Interventions test modifications without requiring code changes
- Hypothesis generation considers agent decisions, communication, planning
- Outcome-oriented evaluation focuses on practical fixes vs attribution
- Multiple hypotheses can resolve same failure
- Results feed back into agent system improvements

## References

- Original paper: https://arxiv.org/abs/2512.06749
- Focus: Multi-agent system debugging and reliability
- Domain: Autonomous agents, system debugging, agent coordination
