---
name: agent0-vl-self-evolution
title: "Agent0-VL: Self-Evolving Agent for Tool-Integrated VL Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.19900"
keywords: [Self-Evolving Agents, Tool-Grounded Verification, Visual Reasoning, Reinforcement Learning, GRPO]
description: "Enable vision-language agents to self-evolve by grounding verification in tool outputs rather than text: implement nested loops where Solver+Verifier generate trajectories and tool-based feedback, then optimize via GRPO using self-generated rewards without external supervision."
---

# Agent0-VL: Self-Evolving Vision-Language Agents

Vision-language agents typically struggle with self-evaluation—they can easily hallucinate confidence scores or generate false critiques without external grounding. This skill demonstrates how to build agents that self-evolve by grounding their verification process in tool-generated evidence, creating a unified loop where reasoning, verification, and self-repair happen through the same agentic mechanism, all optimized via reinforcement learning.

The key innovation is tool-grounded verification: instead of text-based self-evaluation (prone to hallucination), the Verifier evaluates reasoning steps by checking tool outputs, enabling genuine self-correction before policy updates.

## Core Concept

Agent0-VL implements a Self-Evolving Reasoning Cycle (SERC) with two nested loops:

1. **Inner Loop (Generation & Verification)**: Solver generates reasoning trajectories with tool calls; Verifier evaluates using tool-generated evidence and produces structured feedback. When confidence is low, self-repair mechanisms correct reasoning before re-execution.

2. **Outer Loop (Policy Update)**: GRPO optimizes the unified policy using self-generated rewards, requiring zero external reward supervision.

The system operates entirely on tool-grounded evidence, avoiding evaluation hallucination common in text-only LLM self-evaluation.

## Architecture Overview

- **Unified Solver-Verifier Model**: Single model with roles for both generation and verification
- **Tool-Grounded Verification**: Verification uses tool outputs to assess correctness, not subjective text confidence
- **Self-Repair Mechanism**: On low confidence, agent regenerates and re-executes before moving forward
- **Confidence Thresholds**: Numerical gates determining when self-repair triggers
- **Reward Signal Design**: Self-generated rewards from tool-based correctness signals
- **GRPO Optimization**: Policy gradient RL using self-generated, tool-grounded rewards

## Implementation Steps

The self-evolution process cycles through reasoning generation, tool-grounded verification, optional repair, and policy updates.

**1. Initialize Unified Solver-Verifier Model**

Create model infrastructure supporting both generation (Solver) and verification (Verifier) roles.

```python
def create_solver_verifier_model(base_model_name, max_tokens=2048):
    """
    Initialize unified model capable of both reasoning generation and verification.
    Both roles output structured tokens for tool calls and confidence scores.
    """
    model = load_vlm_model(base_model_name)

    # Solver prompt template
    solver_template = """
    Image: [image]
    Question: [question]

    Reason step by step:
    1. Observe the image carefully
    2. Plan which tools to call
    3. Call tools and interpret results
    4. Generate final answer

    Your reasoning:
    """

    # Verifier prompt template
    verifier_template = """
    Image: [image]
    Question: [question]
    Reasoning trajectory: [trajectory]

    Tool outputs:
    [tool_results]

    Evaluate the reasoning step. Provide:
    - score (0-100): How correct is this step based on tool outputs?
    - confidence (0-1): How certain are you in this score?
    - feedback: Specific issues if score < 80
    """

    return {
        'model': model,
        'solver_template': solver_template,
        'verifier_template': verifier_template,
        'max_tokens': max_tokens
    }
```

**2. Implement Solver: Generate Reasoning with Tool Calls**

Solver generates multi-step reasoning trajectories with explicit tool call instructions.

```python
def solver_generate_trajectory(image, question, model_config, tools_available):
    """
    Generate reasoning trajectory with tool calls.
    Solver outputs structured reasoning including tool invocations.
    """
    prompt = model_config['solver_template'].replace('[image]', image).replace('[question]', question)

    trajectory = {
        'steps': [],
        'tool_calls': [],
        'image': image,
        'question': question
    }

    # Generate step-by-step with tool calls
    for step_idx in range(5):  # Max 5 reasoning steps
        response = model_config['model'].generate(
            prompt + trajectory_to_text(trajectory['steps']),
            max_tokens=512,
            stop_tokens=['[TOOL_CALL_END]']
        )

        # Parse for tool calls
        if '[TOOL_CALL]' in response:
            tool_call = extract_tool_call(response)
            trajectory['tool_calls'].append(tool_call)

            # Execute tool
            tool_result = execute_tool(tool_call, tools_available)
            trajectory['steps'].append({
                'type': 'reasoning',
                'content': response,
                'tool_result': tool_result
            })

        else:
            # Final answer
            trajectory['steps'].append({
                'type': 'answer',
                'content': response
            })
            break

    return trajectory
```

**3. Implement Tool-Grounded Verifier**

Verifier evaluates reasoning steps using tool outputs as ground truth evidence.

```python
def verifier_evaluate_trajectory(trajectory, model_config, max_confidence=0.95):
    """
    Verify reasoning using tool-generated evidence.
    Returns score and confidence purely based on tool outputs, not subjective text.
    """
    verifier_prompt = model_config['verifier_template'].replace('[image]', trajectory['image']).replace('[question]', trajectory['question'])

    verifier_prompt = verifier_prompt.replace('[trajectory]', trajectory_to_text(trajectory['steps']))
    verifier_prompt = verifier_prompt.replace('[tool_results]', format_tool_results(trajectory['tool_calls']))

    verification = model_config['model'].generate(
        verifier_prompt,
        max_tokens=256,
        response_format={'score': 'int', 'confidence': 'float', 'feedback': 'str'}
    )

    return {
        'score': verification['score'],
        'confidence': min(verification['confidence'], max_confidence),
        'feedback': verification['feedback'],
        'step_index': len(trajectory['steps']) - 1
    }
```

**4. Implement Self-Repair Loop**

When verification confidence is low, automatically regenerate the problematic step.

```python
def self_repair_step(trajectory, verification_result, model_config, tools_available, max_repairs=2):
    """
    Repair low-confidence reasoning steps.
    Re-generates the problematic step with explicit guidance from verification feedback.
    """
    if verification_result['confidence'] >= 0.75 or max_repairs <= 0:
        return trajectory  # No repair needed or max repairs reached

    # Generate repair prompt
    repair_prompt = f"""
    Previous reasoning had issues:
    {verification_result['feedback']}

    Image: [image]
    Question: [question]
    Previous steps: [steps]

    Please re-reason about the next step, addressing the feedback:
    """

    # Regenerate the step
    repair_response = model_config['model'].generate(repair_prompt, max_tokens=512)

    # Parse and execute tools if present
    if '[TOOL_CALL]' in repair_response:
        tool_call = extract_tool_call(repair_response)
        tool_result = execute_tool(tool_call, tools_available)

        # Replace the problematic step
        trajectory['steps'][verification_result['step_index']] = {
            'type': 'reasoning',
            'content': repair_response,
            'tool_result': tool_result,
            'repaired': True
        }

    # Re-verify
    new_verification = verifier_evaluate_trajectory(trajectory, model_config)

    if new_verification['confidence'] > verification_result['confidence']:
        return trajectory
    else:
        # Recursive repair if still low confidence
        return self_repair_step(trajectory, new_verification, model_config, tools_available, max_repairs - 1)
```

**5. Design Reward Signal from Tool-Grounded Verification**

Create rewards purely from tool outputs and correctness indicators.

```python
def compute_self_generated_reward(trajectory, final_answer, ground_truth_answer):
    """
    Compute reward signal from tool-based verification without external labels.
    Rewards are based on correctness indicators and verification scores.
    """
    reward = 0.0

    # Base reward: answer correctness
    if final_answer == ground_truth_answer:
        reward += 1.0
    else:
        # Partial credit for close answers
        similarity = compute_answer_similarity(final_answer, ground_truth_answer)
        reward += similarity * 0.5

    # Step-wise rewards from tool correctness
    for step in trajectory['steps']:
        if 'tool_result' in step:
            # Tool executed successfully
            reward += 0.1

    # Penalty for excessive repairs
    repair_count = sum(1 for s in trajectory['steps'] if s.get('repaired', False))
    reward -= repair_count * 0.05

    # Efficiency bonus for shorter trajectories
    num_steps = len(trajectory['steps'])
    if num_steps <= 3:
        reward += 0.2

    return max(0.0, reward)  # Clip to [0, inf)
```

**6. GRPO Optimization with Self-Generated Rewards**

Update policy using self-generated rewards through gradient-based optimization.

```python
def grpo_step(batch_trajectories, model_config, optimizer, lr=1e-5):
    """
    Single GRPO step optimizing policy with self-generated rewards.
    No external reward model needed; uses trajectory-internal verification and tool correctness.
    """
    losses = []

    for trajectory in batch_trajectories:
        # Get self-generated reward
        reward = trajectory['self_reward']

        # Compute policy gradient
        log_prob = compute_log_probability(trajectory, model_config['model'])

        # GRPO loss: maximize expected reward
        loss = -log_prob * reward

        losses.append(loss)

    # Backpropagate and update
    total_loss = sum(losses) / len(losses)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {'loss': total_loss.item(), 'mean_reward': np.mean([t['self_reward'] for t in batch_trajectories])}
```

## Practical Guidance

**When to Use Agent0-VL:**
- Tasks with clear tool outputs that can verify reasoning (VQA, visual navigation, document understanding)
- Scenarios where evaluation hallucination is a major problem
- Multi-step reasoning where intermediate feedback is valuable
- Systems where you want end-to-end self-improvement without external reward models

**When NOT to Use:**
- Tasks without executable tools or clear verification signals
- Open-ended reasoning where correctness is subjective
- Real-time applications where repair loops add too much latency

**Key Hyperparameters:**
- `confidence_threshold`: When to trigger repair (0.70-0.80 typical)
- `max_repairs_per_trajectory`: Prevent infinite repair loops (1-3)
- `num_grpo_steps`: Training iterations before evaluation (100-1000)
- `batch_size`: Trajectories per GRPO update (16-64)
- `learning_rate`: Gradient step size (1e-6 to 1e-4 typical)

**Optimization Tips:**
- Cache tool results to avoid redundant executions
- Use confidence scores to dynamically adjust repair thresholds during training
- Batch-generate multiple trajectories in parallel for GRPO updates
- Track repair statistics to detect inadequate reward signals

**Pitfalls to Avoid:**
- Verification feedback becoming circular (verifier hallucinating like Solver)
- Repair loops creating infinite cycles without improvement
- Reward signals too weak to drive policy updates (test with deterministic tasks first)
- Using same tool outputs for both verification and reward (use independent evaluation)

## Reference

Research paper: https://arxiv.org/abs/2511.19900
