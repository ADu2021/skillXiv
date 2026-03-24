---
name: metaclaw-continual-agent-learning
title: "MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.17187"
keywords: [Meta-Learning, Continual Learning, Skill Synthesis, LLM Agents, In-Context Learning]
description: "Enable LLM agents to evolve behavioral skills and policies online through skill synthesis from failures and opportunistic gradient-based refinement, without service interruption."
---

# MetaClaw: Continual Agent Meta-Learning in Production

Production LLM agents face a fundamental tension: they must improve continuously while serving users, but traditional retraining creates downtime and disruption. MetaClaw solves this through a dual-mechanism architecture that runs skill refinement and policy optimization in parallel, triggered intelligently during user-inactive windows.

The core innovation is recognizing two complementary learning pathways. Skill synthesis (converting failures into reusable behavioral patterns) provides immediate improvement with zero latency impact. Policy optimization (using gradient-based RL during idle time) builds on the new skills to refine overall decision-making. These two mechanisms are mutually reinforcing: better policies generate cleaner training data for skills, while richer skills accelerate policy learning.

## Core Concept

MetaClaw operates through two interdependent learning loops:

**Skill-Driven Fast Adaptation:**
- Analyze failure trajectories when agents make mistakes
- Use an LLM evolver to synthesize new behavioral skills on-the-fly
- Deploy new skills immediately to prevent recurrence, zero downtime

**Opportunistic Policy Optimization:**
- Monitor system activity and user calendars
- During idle windows, run gradient-based RL with process reward models (PRM)
- Refine the base policy to leverage newly synthesized skills

The system maintains version separation to prevent data contamination: support sets (for policy updates) and query sets (for evaluation) remain distinct.

## Architecture Overview

- **Failure Capture Module**: Detects task failures and extracts error trajectories
- **LLM Skill Evolver**: Synthesizes new skills from failure patterns using in-context examples
- **Skill Repository**: Versioned skill library with temporal tracking
- **Opportunistic Meta-Learning Scheduler (OMLS)**: Monitors system state, triggers RL during idle time
- **Policy Optimizer**: Cloud-based LoRA fine-tuning + RL with PRM
- **Versioning System**: Prevents support/query data leakage across training iterations

## Implementation Steps

### Step 1: Failure Analysis and Skill Synthesis

Capture failed trajectories and convert them into generalizable skills.

```python
import json
from datetime import datetime

def analyze_failure_and_synthesize_skill(task, trajectory, error, llm_evolver):
    """
    Extract failure pattern and synthesize new skill to address it.
    Returns: (skill_code, skill_name, success_metric)
    """
    # Extract the failure context
    failure_context = {
        'task': task,
        'steps_before_failure': trajectory[-3:],  # Last 3 steps
        'error_type': error['type'],
        'error_message': error['message'],
        'task_goal': task.get('objective'),
        'failure_point': trajectory[-1]
    }

    # Prompt the skill evolver to create a new skill
    skill_synthesis_prompt = f"""
    Given this failure in an agent trajectory:
    Task: {failure_context['task']}
    Failed at step: {failure_context['failure_point']}
    Error: {failure_context['error_message']}

    Create a reusable skill (Python function) that prevents this error type.
    The skill should:
    1. Take (state, context) as input
    2. Return an action or guidance
    3. Be general enough to apply to similar tasks

    Format:
    ```python
    def skill_name(state, context):
        # Implementation
        return action
    ```
    """

    # LLM generates the skill
    skill_definition = llm_evolver.generate(skill_synthesis_prompt)
    skill_name = extract_skill_name(skill_definition)

    return {
        'skill_code': skill_definition,
        'skill_name': skill_name,
        'created_at': datetime.now().isoformat(),
        'triggered_by': failure_context
    }
```

### Step 2: Deploy Skill and Update Router

Add new skill to library and update the skill router's decision logic.

```python
class SkillRepository:
    """Versioned skill library with temporal tracking."""

    def __init__(self):
        self.skills = {}  # skill_name -> skill_definition
        self.versions = {}  # skill_name -> list of versions with timestamps
        self.router_weights = {}  # skill_name -> router confidence

    def add_skill(self, skill_definition, base_skill=None):
        """Add synthesized skill to repository."""
        skill_name = skill_definition['skill_name']

        # Track version history
        if skill_name not in self.versions:
            self.versions[skill_name] = []
        self.versions[skill_name].append({
            'timestamp': skill_definition['created_at'],
            'code': skill_definition['skill_code'],
            'triggered_by': skill_definition['triggered_by']
        })

        # Store active skill
        self.skills[skill_name] = skill_definition['skill_code']

        # Initialize router weight (low initially, refined by policy)
        self.router_weights[skill_name] = 0.1

        return skill_name

    def select_skill(self, state, context, learned_weights=None):
        """
        Route to best skill based on learned weights and state.
        Uses learned_weights from policy optimization if available.
        """
        if learned_weights:
            self.router_weights = learned_weights

        scores = {}
        for skill_name in self.skills.keys():
            # Combine learned weight with context similarity
            context_match = self._context_similarity(skill_name, context)
            scores[skill_name] = (
                0.7 * self.router_weights.get(skill_name, 0.0) +
                0.3 * context_match
            )

        selected_skill = max(scores.items(), key=lambda x: x[1])[0]
        return selected_skill, self.skills[selected_skill]

    def _context_similarity(self, skill_name, context):
        """Compute how well skill applies to current context."""
        trigger = self.versions[skill_name][-1]['triggered_by']
        # Simple similarity: compare task types
        similarity = 1.0 if trigger['task'].get('type') == context.get('type') else 0.5
        return similarity
```

### Step 3: Opportunistic Policy Optimization Scheduler

Monitor system activity and trigger training during idle windows.

```python
import psutil
from datetime import datetime, timedelta
import threading

class OpportunisticMetaLearningScheduler:
    """
    Monitors system state and user activity.
    Triggers policy optimization during idle windows.
    """

    def __init__(self, agent, skill_repo, rl_trainer, check_interval=60):
        self.agent = agent
        self.skill_repo = skill_repo
        self.rl_trainer = rl_trainer
        self.check_interval = check_interval
        self.is_optimizing = False
        self.last_optimization = datetime.now()

    def should_optimize(self, user_calendar=None):
        """
        Determine if it's safe to run optimization.
        Checks: CPU idle, no active users, no upcoming meetings, cool-down time.
        """
        # Check system CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 20:  # System is in use
            return False

        # Check if last optimization was too recent (cool-down)
        time_since_last = (datetime.now() - self.last_optimization).total_seconds()
        if time_since_last < 3600:  # 1 hour cool-down
            return False

        # Check user calendar (if provided)
        if user_calendar:
            now = datetime.now()
            next_meeting_in = user_calendar.time_to_next_meeting()
            if next_meeting_in < timedelta(minutes=30):
                return False

        return True

    def run_policy_optimization(self):
        """
        Run RL training loop on collected trajectories.
        Uses Process Reward Model (PRM) for dense signals.
        """
        if self.is_optimizing:
            return  # Prevent concurrent runs

        self.is_optimizing = True
        try:
            # Collect trajectories from skill_repo (support set)
            support_trajectories = self._get_support_set()

            # Run RL with PRM reward signal
            self.rl_trainer.train(
                trajectories=support_trajectories,
                num_steps=500,
                learning_rate=1e-4,
                use_process_reward_model=True
            )

            # Extract updated router weights from policy
            new_weights = self.rl_trainer.extract_skill_router_weights()
            self.skill_repo.router_weights.update(new_weights)

            self.last_optimization = datetime.now()

        finally:
            self.is_optimizing = False

    def _get_support_set(self, max_trajectories=100):
        """
        Get recent successful trajectories for RL training.
        Maintains separation from evaluation (query) set.
        """
        recent_successes = []
        for skill_name, versions in self.skill_repo.versions.items():
            # Get trajectories that used this skill successfully
            for version in versions[-5:]:  # Last 5 versions
                recent_successes.append({
                    'skill_name': skill_name,
                    'trajectory': version.get('trajectory', []),
                    'success': True
                })
        return recent_successes[:max_trajectories]

    def monitor_and_optimize(self):
        """Background thread that periodically checks and runs optimization."""
        while True:
            if self.should_optimize():
                self.run_policy_optimization()
            threading.Event().wait(self.check_interval)
```

### Step 4: Integrate into Agent Loop

Wire skill synthesis and policy optimization into the main agent execution.

```python
class MetaLearningAgent:
    """
    LLM agent with continual meta-learning capabilities.
    Synthesizes skills from failures and optimizes policy opportunistically.
    """

    def __init__(self, base_policy, skill_repo, scheduler):
        self.base_policy = base_policy
        self.skill_repo = skill_repo
        self.scheduler = scheduler
        self.trajectory_buffer = []

    def execute_task(self, task):
        """Execute task with skill-enhanced policy."""
        state = task.get_initial_state()
        trajectory = []

        while not task.is_done():
            # Select best skill for current state
            skill_name, skill_fn = self.skill_repo.select_skill(state, task.context)

            # Execute skill + base policy
            if skill_fn is not None:
                action = skill_fn(state, task.context)
            else:
                action = self.base_policy(state, task.context)

            # Execute action
            next_state, reward = task.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        task_result = task.get_result()

        # Log trajectory
        self.trajectory_buffer.append({
            'task': task.task_id,
            'trajectory': trajectory,
            'success': task_result['success'],
            'error': task_result.get('error')
        })

        # If failure, synthesize new skill
        if not task_result['success']:
            new_skill = analyze_failure_and_synthesize_skill(
                task, trajectory, task_result['error'], self.base_policy
            )
            self.skill_repo.add_skill(new_skill)

        return task_result

    def background_learning_loop(self):
        """Run opportunistic policy optimization."""
        while True:
            if self.scheduler.should_optimize():
                self.scheduler.run_policy_optimization()
            threading.Event().wait(60)  # Check every minute
```

## Practical Guidance

**Hyperparameters:**
- Skill repository refresh rate: 1-5 new skills per 100 tasks (tune by monitoring quality)
- Optimization cool-down: 1-4 hours between RL runs (balance improvement vs. stability)
- Router weight update rate: 0.1-0.3 (controls speed of skill adoption)
- Process reward model (PRM) training frequency: parallel to main task execution

**When to Use:**
- Long-running deployed agents serving diverse user requests
- Tasks with learnable error patterns (agents make systematic mistakes)
- Environments where skill synthesis is feasible (structured action spaces)
- When you have access to user calendar / system activity data

**When NOT to Use:**
- Real-time low-latency scenarios (synthesis adds inference overhead)
- Highly chaotic task distributions (skills become overfitted)
- Systems without clear failure patterns or error diagnostics
- Environments requiring immediate consistency (gradual improvement only)

**Pitfalls:**
- Synthesized skills can perpetuate errors if failure analysis is shallow; validate before deployment
- Policy drift: outdated skills in repository increase confusion; prune regularly
- Feedback loop amplification: if PRM is poorly calibrated, bad skills get reinforced
- Version contamination: ensure support/query sets remain strictly separated or data leakage occurs

## Reference

Paper: [arxiv.org/abs/2603.17187](https://arxiv.org/abs/2603.17187)
