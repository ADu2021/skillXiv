---
name: autoenv-environment-synthesis
title: "AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.19304"
keywords: [Environment Synthesis, Coding Agents, Automated Testing, Agent Evaluation, DSL-to-Code]
description: "Generate diverse, validated game environments automatically using domain-specific language specifications and LLM coding agents with self-repair, enabling cost-effective (≈$4/env) creation of heterogeneous test domains for evaluating cross-environment agent generalization."
---

# AutoEnv: Automated Environments for Agent Testing

Creating diverse, realistic environments for training and testing agents is expensive and time-consuming. This skill demonstrates how to automate environment generation using a three-layer architectural abstraction combined with LLM coding agents that iteratively synthesize and repair implementation code. The system enables systematic creation of heterogeneous game environments with validated dynamics, observation functions, and rendering pipelines—all driven from high-level YAML specifications.

The core innovation is decomposing environment complexity into three manageable layers (dynamics, observations, rendering) and using coding agents to translate natural language specifications into executable Python code with built-in self-repair loops.

## Core Concept

AutoEnv employs a three-layer abstraction architecture to decompose environments systematically:

1. **BaseEnv**: Implements core dynamics—state space definition, action effects, transition logic, and reward computation
2. **ObsEnv**: Specializes observation functions through configurable policies that determine what information agents perceive
3. **SkinEnv**: Applies rendering to convert observations into agent-facing modalities (pixels, text, features)

This layering enables each component to be synthesized and tested independently, with composition handled by the system automatically.

## Architecture Overview

- **YAML DSL Frontend**: High-level environment specification with themes, goals, rules, state variables, and reward conditions
- **Prompt Assembly Pipeline**: Converts DSL specifications into detailed, contextualized prompts for coding agents
- **LLM Coding Agent**: Generates implementation code using patterns and examples from libraries
- **Three-Stage Verification**: Execution testing, level generation validation, and differential model testing
- **Self-Repair Loop**: Iterative syntax and semantic error correction through LLM-guided debugging

## Implementation Steps

The environment synthesis process flows through specification, code generation, validation, and repair stages.

**1. Parse and Validate YAML Specification**

Convert high-level environment themes into detailed specifications with goals, mechanics, and state variables.

```yaml
# Example: Simple grid-world specification
environment:
  name: "Treasure Hunt"
  theme: "exploration"
  goal: "collect 3 treasures before time expires"
  state_space:
    - agent_position: [int, int]
    - treasures: list
    - time_remaining: int
  actions:
    - move_up
    - move_down
    - move_left
    - move_right
    - collect
  reward_conditions:
    - collect_treasure: +10
    - step_taken: -0.1
    - timeout: -100
```

**2. Generate Implementation with LLM Coding Agent**

Use an LLM agent to convert YAML specs into three Python classes implementing the architecture layers.

```python
def generate_environment_code(yaml_spec, example_patterns):
    """
    Generate BaseEnv, ObsEnv, and SkinEnv implementations from YAML.
    The LLM receives the specification and code patterns to follow.
    """
    prompt = f"""
    Given this environment specification:
    {yaml_spec}

    And these code patterns:
    {example_patterns}

    Generate three Python classes:
    1. BaseEnv: Implements state transitions and rewards
    2. ObsEnv: Implements observation policies
    3. SkinEnv: Implements rendering/modality conversion

    Return complete, runnable code with proper class definitions and docstrings.
    """

    code = llm_agent.generate_code(prompt)
    return code
```

**3. Execute and Test Generated Code**

Run the generated environment with brief agent rollouts to detect crashes and verify reward signals.

```python
def verify_execution(generated_code, test_episodes=10):
    """
    Test generated environment for crashes and meaningful reward structure.
    Runs brief agent rollouts to validate dynamics.
    """
    try:
        # Execute the generated code
        env = exec_and_create_env(generated_code)

        # Run test episodes
        for episode in range(test_episodes):
            obs = env.reset()
            total_reward = 0

            for step in range(100):
                # Random action
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    break

            # Verify non-zero rewards
            if total_reward == 0:
                raise ValueError("No reward signal detected")

        return True, "Execution successful"

    except Exception as e:
        return False, str(e)
```

**4. Generate Levels and Validate Reachability**

Test that generated levels are solvable—agents can reach valid terminal states.

```python
def validate_level_structure(env, num_tests=20):
    """
    Verify generated levels have valid, reachable goal states.
    Tests agent ability to solve at least one episode per level variation.
    """
    success_count = 0

    for test_idx in range(num_tests):
        obs = env.reset()

        # Simple greedy solver for validation
        solved = attempt_greedy_solution(env, obs, max_steps=500)

        if solved:
            success_count += 1

    solvability_rate = success_count / num_tests

    if solvability_rate < 0.3:
        raise ValueError(f"Too few solvable levels: {solvability_rate:.1%}")

    return solvability_rate
```

**5. Self-Repair Loop for Errors**

When verification fails, pass error messages back to the LLM to fix implementation.

```python
def repair_and_retry(yaml_spec, error_message, attempt=1, max_attempts=3):
    """
    Iteratively repair code errors using LLM feedback.
    Continues until code passes all verification stages.
    """
    if attempt >= max_attempts:
        raise RuntimeError(f"Failed after {max_attempts} repair attempts")

    prompt = f"""
    The generated environment code had this error:
    {error_message}

    Original specification:
    {yaml_spec}

    Fix the code to resolve this error. Return the corrected classes.
    """

    repaired_code = llm_agent.generate_code(prompt)
    success, error = verify_execution(repaired_code)

    if not success:
        return repair_and_retry(yaml_spec, error, attempt + 1, max_attempts)

    return repaired_code
```

**6. Differential Model Testing**

Verify generated environments produce consistent, non-random reward signals across multiple agent policies.

```python
def differential_model_testing(env, num_models=3, episodes_per_model=5):
    """
    Test that multiple agent policies produce differentiated rewards.
    Validates that environment has meaningful, non-random dynamics.
    """
    policies = [
        RandomPolicy(env.action_space),
        GreedyPolicy(env),
        ExplorationPolicy(env)
    ]

    reward_distributions = []

    for policy in policies:
        episode_rewards = []

        for _ in range(episodes_per_model):
            obs = env.reset()
            total_reward = 0

            for step in range(100):
                action = policy.select_action(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

                if done:
                    break

            episode_rewards.append(total_reward)

        reward_distributions.append(episode_rewards)

    # Check that different policies get different reward distributions
    variance_across_policies = np.var([np.mean(r) for r in reward_distributions])

    if variance_across_policies < 1.0:
        raise ValueError("Policies not differentiated by rewards")

    return True
```

## Practical Guidance

**When to Use AutoEnv:**
- Creating diverse test suites for agent generalization (10+ environments needed)
- Developing procedural game environments at scale
- Evaluating cross-environment transfer learning
- Rapid prototyping of new RL domains

**When NOT to Use:**
- Single, highly specialized environments with unique physics (handcraft instead)
- Environments requiring complex 3D graphics or physics engines
- Tasks where environment diversity is less important than fidelity

**Key Hyperparameters:**
- `max_repair_attempts`: How many times to retry on error (typically 2-3)
- `verification_episodes`: Brief rollouts for execution testing (10-20)
- `solvability_threshold`: Minimum fraction of levels that must be solvable (0.3-0.5)
- `num_models_for_differential_testing`: How many policies to test (3-5)

**Cost Optimization:**
- Batch YAML specifications into single prompts to reduce API calls
- Use smaller LLM models for self-repair loops (can achieve 70-80% success with smaller models)
- Cache verified code patterns to reduce regeneration

**Integration Pattern:**
The generated environments are standard OpenAI Gym-compatible, enabling drop-in integration with existing RL pipelines. Set `num_envs=K` to run K environments in parallel for efficient cross-environment agent training.

## Reference

Research paper: https://arxiv.org/abs/2511.19304
