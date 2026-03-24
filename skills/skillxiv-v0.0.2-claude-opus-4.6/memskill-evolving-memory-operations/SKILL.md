---
name: memskill-evolving-memory-operations
title: "MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02474"
keywords: [Agent Memory, Skill Learning, Self-Improvement, LLM Controller, Reinforcement Learning]
description: "Build agents that evolve their own memory operations by learning a skill bank of memory transformations and periodically discovering new skills from challenging cases, enabling adaptive memory management that improves with scale."
---

# MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents

## Problem Context

Existing LLM agent memory systems rely on static, hand-designed operations (insert, update, delete) that hard-code human assumptions. These fixed procedures fail when interaction patterns vary, histories grow longer, or task structure changes. Agents cannot adapt their memory strategies or learn what truly matters for their specific workflow.

## Core Concept

MemSkill introduces [learnable skill banks, controller-executor architecture, LLM-guided evolution] to enable agents to discover and refine memory operations autonomously. A learned policy (controller) selects which skills to apply per context, while an LLM-based designer periodically identifies missing behaviors and proposes new skills through mining hard cases.

## Architecture Overview

- **Skill bank**: Learnable memory operations (Insert, Update, Delete, Skip) stored as prompts; initially hand-designed
- **Controller**: Neural policy selecting relevant skills for each context span (trained with PPO)
- **Executor**: LLM-based component applying selected skills to generate memories
- **Designer**: LLM mining hard cases to identify skill gaps and propose new operations
- **Training loop**: Alternate between RL-based skill selection and skill bank evolution

## Implementation

### Step 1: Define initial skill bank

Create a basic set of learnable memory operations. Each skill is represented as a prompt specifying when and how to transform interaction traces.

```python
# Initial skill bank definition
class MemorySkillBank:
    def __init__(self):
        self.skills = {
            'insert': {
                'description': 'Add new information to memory',
                'prompt': 'Given the interaction: {interaction}, '
                          'extract and add the most important fact to memory.',
                'parameters': {}
            },
            'update': {
                'description': 'Modify existing memory entries',
                'prompt': 'Given the interaction: {interaction} and '
                          'current memory: {memory}, update the memory '
                          'if new information contradicts or extends existing facts.',
                'parameters': {}
            },
            'delete': {
                'description': 'Remove outdated or irrelevant memories',
                'prompt': 'Given the interaction: {interaction} and '
                          'current memory: {memory}, remove memories that '
                          'are no longer relevant or are contradicted.',
                'parameters': {}
            },
            'skip': {
                'description': 'Do not modify memory',
                'prompt': 'No memory modification needed.',
                'parameters': {}
            }
        }
        self.skill_names = list(self.skills.keys())

    def get_skill_prompt(self, skill_name, interaction, memory):
        """Get the prompt for a specific skill."""
        skill = self.skills[skill_name]
        return skill['prompt'].format(
            interaction=interaction, memory=memory
        )

    def add_skill(self, skill_name, description, prompt):
        """Add a new skill to the bank (called by Designer)."""
        self.skills[skill_name] = {
            'description': description,
            'prompt': prompt,
            'parameters': {}
        }
        self.skill_names.append(skill_name)
```

### Step 2: Implement controller for skill selection

Train a neural network to select which skills are relevant for each interaction context. Use PPO with task performance as reward.

```python
# Skill selection controller
class MemorySkillController:
    def __init__(self, input_dim=768, num_skills=4):
        self.encoder = nn.LSTM(input_dim, 256, batch_first=True)
        self.policy_head = nn.Linear(256, num_skills)
        self.value_head = nn.Linear(256, 1)

    def select_skills(self, interaction_embedding, memory_embedding):
        """
        Select skills based on interaction and memory context.
        Returns skill probabilities and sampled skill indices.
        """
        combined = torch.cat([interaction_embedding, memory_embedding], dim=-1)
        encoded, _ = self.encoder(combined)

        # Policy output: skill probabilities
        logits = self.policy_head(encoded[:, -1, :])
        probs = F.softmax(logits, dim=-1)

        # Value output: state value estimate
        value = self.value_head(encoded[:, -1, :])

        # Sample skills (allow multiple skills per context)
        skill_mask = torch.multinomial(probs, num_samples=1, replacement=False)

        return probs, skill_mask, value
```

### Step 3: Implement executor for applying skills

Use an LLM to execute selected skills on interaction traces. Apply skills in sequence to transform memories.

```python
# Executor: apply selected skills
class MemoryExecutor:
    def __init__(self, llm_client, skill_bank):
        self.llm = llm_client
        self.skill_bank = skill_bank

    def execute_skills(self, interaction, memory, skill_indices):
        """
        Apply selected skills to update memory.
        """
        current_memory = memory
        execution_trace = []

        for skill_idx in skill_indices:
            skill_name = self.skill_bank.skill_names[skill_idx]
            skill_prompt = self.skill_bank.get_skill_prompt(
                skill_name, interaction, current_memory
            )

            # LLM executes skill
            response = self.llm.generate(
                prompt=skill_prompt,
                max_tokens=200,
                temperature=0.7
            )

            execution_trace.append({
                'skill': skill_name,
                'input_memory': current_memory,
                'output': response
            })

            # Update memory for next skill
            current_memory = response

        return current_memory, execution_trace
```

### Step 4: Train controller with PPO

Optimize the controller to select skills that maximize task performance. Use task reward signals to guide skill selection.

```python
# PPO training for skill controller
def train_controller_ppo(
    controller, executor, skill_bank, trajectories,
    task_reward_fn, optimizer, num_epochs=3
):
    """
    Train skill selection controller with PPO on task rewards.
    """
    for epoch in range(num_epochs):
        total_loss = 0.0

        for trajectory in trajectories:
            interaction_seq = trajectory['interactions']
            memory_seq = trajectory['memories']
            task_reward = task_reward_fn(trajectory)

            # Forward pass: select skills
            probs_seq = []
            skill_masks = []
            values = []

            for i, (interaction, memory) in enumerate(
                zip(interaction_seq, memory_seq)
            ):
                inter_emb = embed_interaction(interaction)
                mem_emb = embed_memory(memory)

                probs, skill_mask, value = controller.select_skills(
                    inter_emb, mem_emb
                )

                probs_seq.append(probs)
                skill_masks.append(skill_mask)
                values.append(value)

            # Compute advantages
            returns = compute_returns([task_reward] * len(values), gamma=0.99)
            advantages = returns - torch.tensor(values)

            # PPO loss
            old_log_probs = torch.log(torch.gather(
                torch.stack(probs_seq), 1, torch.stack(skill_masks)
            )).detach()

            new_probs = torch.stack(probs_seq)
            new_log_probs = torch.log(torch.gather(new_probs, 1, torch.stack(skill_masks)))

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.95, 1.05) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(torch.cat(values), returns)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(trajectories)}")
```

### Step 5: Implement designer for skill discovery

Mine challenging cases where the current skill set fails. Use LLM to propose new skills addressing gaps.

```python
# Designer: discover new skills
class MemorySkillDesigner:
    def __init__(self, llm_client):
        self.llm = llm_client

    def mine_hard_cases(self, trajectories, success_threshold=0.5):
        """
        Identify trajectories where memory management failed.
        """
        hard_cases = []

        for trajectory in trajectories:
            final_reward = trajectory.get('reward', 0)

            if final_reward < success_threshold:
                hard_cases.append({
                    'interactions': trajectory['interactions'],
                    'memory_operations': trajectory['skills_applied'],
                    'outcome': trajectory.get('failure_reason', ''),
                    'reward': final_reward
                })

        return hard_cases

    def propose_new_skill(self, hard_case, existing_skills):
        """
        Use LLM to propose a new skill addressing a failure case.
        """
        case_desc = f"""
        Interaction sequence: {hard_case['interactions']}
        Applied skills: {hard_case['memory_operations']}
        Failure: {hard_case['outcome']}
        Existing skills: {list(existing_skills.keys())}
        """

        prompt = f"""
        Given this memory management failure case:
        {case_desc}

        Propose a new memory operation skill that would address this failure.
        Provide:
        1. Skill name (kebab-case)
        2. Description of what it does
        3. Prompt template for LLM execution

        Format:
        Name: [skill-name]
        Description: [description]
        Prompt: [prompt template with {{interaction}} and {{memory}} placeholders]
        """

        response = self.llm.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )

        # Parse response to extract skill definition
        new_skill = parse_skill_proposal(response)
        return new_skill

    def evolve_skill_bank(self, skill_bank, hard_cases, top_k=3):
        """
        Discover and add top-k new skills to the skill bank.
        """
        proposed_skills = []

        for hard_case in hard_cases[:top_k]:
            new_skill = self.propose_new_skill(hard_case, skill_bank.skills)
            proposed_skills.append(new_skill)

        # Add to skill bank
        for skill in proposed_skills:
            skill_bank.add_skill(
                skill['name'],
                skill['description'],
                skill['prompt']
            )

        return proposed_skills
```

### Step 6: Main training loop alternating controller and designer

Implement the core loop: train controller, mine hard cases, evolve skills, repeat.

```python
# Main training loop
def train_memskill_agent(
    skill_bank, controller, executor, designer,
    all_trajectories, task_reward_fn,
    num_iterations=10, hard_case_threshold=0.5
):
    """
    Main loop: train controller → mine failures → evolve skills → repeat.
    """
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4)

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")

        # Step 1: Train controller
        print("  Training controller with PPO...")
        train_controller_ppo(
            controller, executor, skill_bank,
            all_trajectories, task_reward_fn,
            controller_optimizer
        )

        # Step 2: Mine hard cases
        print("  Mining hard cases...")
        hard_cases = designer.mine_hard_cases(
            all_trajectories,
            success_threshold=hard_case_threshold
        )

        if len(hard_cases) > 0:
            # Step 3: Evolve skill bank
            print(f"  Proposing new skills ({len(hard_cases)} failures)...")
            new_skills = designer.evolve_skill_bank(
                skill_bank, hard_cases, top_k=3
            )
            print(f"    Added {len(new_skills)} new skills")

        # Step 4: Evaluate improvement
        eval_reward = evaluate_agent(
            controller, executor, skill_bank, all_trajectories
        )
        print(f"  Evaluation reward: {eval_reward}")
```

## Practical Guidance

**When to use**: Long-horizon agent tasks (conversational, embodied control) where memory management is critical and task patterns vary. Less beneficial for one-off reasoning or static retrieval scenarios.

**Hyperparameters**:
- Controller hidden size (256-512): larger for more complex contexts
- PPO learning rate (1e-4 to 5e-4): conservative to avoid instability
- Skill mining threshold (0.3-0.7): lower catches more failure cases
- Evolution frequency: every 10-20 iterations if using growing trajectory data

**Common pitfalls**:
- Skill bank explosion: periodically prune low-utility skills by measuring frequency
- Designer over-proposes: filter new skills by simulating on hard cases first
- Controller convergence: use gradient clipping and value normalization for stability
- Memory bloat: cap memory size; implement retention policies (recency, importance)

**Scaling**: Skill bank grows linearly with iterations (add 2-3 skills per round). Controller training cost is linear in trajectory count. Recommend evaluating every 5-10 iterations to track improvement.

## Reference

Paper: https://arxiv.org/abs/2602.02474
Code: Available at author's repository
Related work: Memory-augmented neural networks, meta-learning, agent architecture design
