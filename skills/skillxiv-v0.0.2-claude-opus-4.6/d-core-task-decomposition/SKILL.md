---
name: d-core-task-decomposition
title: "D-CORE: Incentivizing Task Decomposition for Complex Tool Use"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02160"
keywords: [Task Decomposition, Reasoning Models, Tool Use, Self-Distillation, GRPO]
description: "Mitigate lazy reasoning in Large Reasoning Models via self-distillation teaching task decomposition, followed by Diversity-Aware GRPO with entropy-based advantage functions, enabling effective decomposition without external teachers while balancing structured reasoning with diversity."
---

# D-CORE: Incentivizing Task Decomposition via Self-Distillation and DA-GRPO

Large Reasoning Models often exhibit "lazy reasoning"—generating extensive but ineffective processes instead of decomposing complex tasks into executable subtasks. D-CORE addresses this through a two-stage approach: self-distillation that bootstraps decomposition learning without external teachers, followed by Diversity-Aware GRPO (DA-GRPO) that encourages high-entropy reasoning while maintaining structured decomposition.

## Core Concept

The key insight is that lazy reasoning and over-diversification are dual problems. Standard self-distillation produces overly-structured trajectories with low diversity, while GRPO trained on sparse rewards encourages entropy but loses decomposition structure. D-CORE solves this by combining bootstrapped decomposition learning with entropy-based advantage functions that reward both correctness and reasoning diversity.

## Architecture Overview

- **Self-Distillation Stage**: Prompt LRM to decompose queries, generate reasoning, apply SFT without external data
- **Entropy-Aware GRPO (DA-GRPO)**: Substitute entropy for advantage when standard advantage approaches zero (sparse reward case)
- **Bootstrapping Process**: Iteratively improve decomposition quality by training on synthesized data
- **Diversity-Correctness Balance**: Entropy encourages exploration; correctness reward provides signal
- **Generalization**: Demonstrated across multiple benchmarks with 5-30% improvements

## Implementation

### Step 1: Self-Distillation via Bootstrapped Decomposition

Generate and synthesize training data for task decomposition.

```python
def self_distillation_stage(model, queries, reasoning_model, num_iterations=3):
    """
    Bootstrap task decomposition without external teacher.
    1. Prompt model to decompose query
    2. Generate reasoning for subtasks
    3. Compose into trajectory
    4. Apply SFT on synthesized data
    """
    synthesized_trajectories = []
    
    for iteration in range(num_iterations):
        for query in queries:
            # Step 1: Prompt decomposition
            decomposition_prompt = f"""
            Break down this complex task into executable subtasks:
            {query}
            
            Format: 
            Subtask 1: [description]
            Subtask 2: [description]
            ...
            """
            decomposition = model.generate(decomposition_prompt, max_tokens=200)
            subtasks = parse_subtasks(decomposition)
            
            # Step 2: Generate reasoning and tool calls for each subtask
            trajectory = {'query': query, 'decomposition': decomposition, 'steps': []}
            current_context = ""
            
            for subtask in subtasks:
                reasoning_prompt = f"""
                Previous context: {current_context}
                
                Solve this subtask: {subtask}
                Provide reasoning and any tool calls needed.
                """
                
                reasoning = reasoning_model.generate(reasoning_prompt, max_tokens=300)
                tool_calls = extract_tool_calls(reasoning)
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    result = execute_tool(tool_call)
                    tool_results.append(result)
                
                trajectory['steps'].append({
                    'subtask': subtask,
                    'reasoning': reasoning,
                    'tool_calls': tool_calls,
                    'tool_results': tool_results
                })
                
                # Update context for next subtask
                current_context += f"\nSubtask result: {'; '.join(tool_results)}"
            
            # Step 3: Compose final answer
            final_prompt = f"Given all subtask results: {current_context}\nProvide final answer to: {query}"
            final_answer = model.generate(final_prompt, max_tokens=200)
            trajectory['final_answer'] = final_answer
            
            synthesized_trajectories.append(trajectory)
        
        # Step 4: Apply SFT on synthesized trajectories
        sft_model = model
        optimizer = torch.optim.Adam(sft_model.parameters(), lr=1e-5)
        
        for trajectory in synthesized_trajectories:
            # Reconstruct full trajectory text
            trajectory_text = f"Query: {trajectory['query']}\n"
            trajectory_text += f"Decomposition: {trajectory['decomposition']}\n"
            for step in trajectory['steps']:
                trajectory_text += f"Reasoning: {step['reasoning']}\n"
                trajectory_text += f"Tool results: {'; '.join(step['tool_results'])}\n"
            trajectory_text += f"Answer: {trajectory['final_answer']}"
            
            # SFT loss
            logits = sft_model(trajectory_text, return_logits=True)
            loss = compute_language_model_loss(logits, trajectory_text)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model
```

### Step 2: Identify Decomposition Failures

Analyze where lazy reasoning occurs.

```python
def identify_lazy_reasoning(trajectory, ground_truth):
    """
    Detect when model generates extensive reasoning without meaningful decomposition.
    Lazy reasoning: long reasoning sequences without tool use or clear subtask progress.
    """
    reasoning_length = len(trajectory['reasoning'].split())
    num_tool_calls = len(trajectory.get('tool_calls', []))
    num_subtasks = len(trajectory.get('subtasks', []))
    
    # Metrics for laziness
    reasoning_per_tool = reasoning_length / (num_tool_calls + 1)
    reasoning_per_subtask = reasoning_length / (num_subtasks + 1)
    
    is_lazy = (
        reasoning_per_tool > 100 and  # Too much reasoning per tool call
        num_subtasks < 3 and           # Insufficient decomposition
        reasoning_per_subtask > 200    # Verbose reasoning without clear structure
    )
    
    return is_lazy, {'reasoning_length': reasoning_length, 'num_tool_calls': num_tool_calls}
```

### Step 3: Implement Diversity-Aware GRPO (DA-GRPO)

Train with entropy-based advantages when standard advantages vanish.

```python
class DiversityAwareGRPO:
    """
    GRPO variant that incentivizes both correctness and reasoning diversity.
    """
    def __init__(self, model, reference_model, reward_fn):
        self.model = model
        self.reference_model = reference_model
        self.reward_fn = reward_fn
        self.entropy_coefficient = 0.1
    
    def compute_advantage(self, trajectory, group_size=4, min_advantage_threshold=0.01):
        """
        Compute advantage with entropy fallback for sparse rewards.
        """
        # Compute reward
        reward = self.reward_fn(trajectory)
        
        # Group trajectories for GRPO
        # Standard GRPO advantage
        group_rewards = get_group_rewards(trajectory, group_size)
        standard_advantage = reward - group_rewards.mean()
        
        # If standard advantage is too small (sparse reward), use entropy
        if abs(standard_advantage) < min_advantage_threshold:
            # Entropy advantage: encourage high-entropy token generation
            entropy = compute_trajectory_entropy(trajectory, self.model)
            entropy_advantage = entropy * self.entropy_coefficient
            return entropy_advantage
        else:
            return standard_advantage
    
    def training_step(self, batch, group_size=4):
        """
        DA-GRPO training step with entropy-aware advantages.
        """
        total_loss = 0.0
        
        for trajectory in batch:
            # Get log probabilities
            log_probs = self.model.get_log_probs(trajectory['tokens'])
            reference_log_probs = self.reference_model.get_log_probs(trajectory['tokens'])
            
            # Compute advantage with entropy fallback
            advantage = self.compute_advantage(trajectory, group_size)
            
            # GRPO loss
            grpo_loss = -log_probs * advantage
            
            # KL regularization
            kl_div = torch.nn.functional.kl_div(log_probs, reference_log_probs)
            
            loss = grpo_loss + 0.1 * kl_div
            total_loss += loss.mean()
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        return total_loss.item()
```

### Step 4: Entropy-Based Advantage Computation

Calculate entropy-based rewards when standard signals are weak.

```python
def compute_trajectory_entropy(trajectory, model, normalize=True):
    """
    Measure entropy of generated reasoning.
    Higher entropy = more diverse reasoning paths.
    """
    token_ids = trajectory['tokens']
    logits = model(token_ids, return_logits=True)
    probs = torch.softmax(logits, dim=-1)
    
    # Entropy per token
    entropy_per_token = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    # Average entropy
    trajectory_entropy = entropy_per_token.mean()
    
    if normalize:
        max_entropy = torch.log(torch.tensor(model.vocab_size, dtype=torch.float32))
        trajectory_entropy = trajectory_entropy / max_entropy
    
    return trajectory_entropy

def compute_group_rewards(trajectories, group_size):
    """
    Partition trajectories into groups for relative advantage computation.
    """
    num_groups = len(trajectories) // group_size
    group_rewards = []
    
    for g in range(num_groups):
        group = trajectories[g*group_size:(g+1)*group_size]
        group_reward = sum(t['reward'] for t in group) / group_size
        group_rewards.extend([group_reward] * group_size)
    
    return torch.tensor(group_rewards)
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|----------------|-------|
| Self-Distillation Iterations | 2-5 iterations | More iterations improve decomposition quality but show diminishing returns |
| Entropy Coefficient | 0.05-0.2 | Higher = more emphasis on diversity over correctness |
| Advantage Threshold | 0.01-0.05 | Determines when to switch to entropy-based advantages |
| Group Size | 4-8 trajectories | Affects GRPO stability; larger groups reduce variance |
| Training Data Scale | 100-1000 queries | Scaling law: more data -> better generalization |

**When to Use:**
- Large reasoning models with multi-step tasks
- Tasks suffering from lazy reasoning (extensive but ineffective reasoning)
- Scenarios needing both structured decomposition and diverse exploration

**When Not to Use:**
- Single-step reasoning tasks (decomposition not helpful)
- Tasks with dense reward signals (entropy-based advantages less useful)
- Models already showing good decomposition behavior

## Reference

Achieves 5.7-30.8% improvements on accuracy across benchmarks (BFCLv3, τ-bench, ACEBench), with D-CORE-8B matching 70B model performance and D-CORE-14B demonstrating strong generalization across diverse reasoning tasks.
