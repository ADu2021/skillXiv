---
name: skill-rl-recursive-distillation
title: "SkillRL: Evolving Agents via Recursive Skill-Augmented RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.08234"
keywords: [Skill Distillation, Agent Learning, Experience Reuse, Hierarchical Skills, Policy Improvement]
description: "Improve agent performance by autonomously distilling behavioral patterns from trajectories into reusable skills, then using these skills to guide future decisions. Achieves 89.9% success on ALFWorld through differential processing of success vs failure episodes and dynamic skill library evolution."
---

# SkillRL: Recursive Skill-Augmented Reinforcement Learning

Agent performance plateaus when policies must rediscover past insights repeatedly. SkillRL addresses this by automatically extracting behavioral patterns from interaction history into compact, reusable skills that guide future decision-making. Rather than storing raw trajectories, the system distills strategic patterns and failure lessons, creating a skill library that grows with the agent.

## Core Concept

SkillRL processes trajectories differentially:
- **Success episodes** → extract strategic patterns (10-20× compression vs raw trajectory)
- **Failure episodes** → extract failure lessons capturing what went wrong

Skills organize hierarchically: general skills (exploration, state management) and task-specific skills. During decision-making, the agent retrieves relevant skills via semantic similarity, reducing context overhead while maintaining reasoning quality.

The skill library evolves recursively: after validation epochs, failure modes generate new skills or refine existing ones, creating a virtuous cycle where improved policies encounter new challenges.

## Architecture Overview

- **Experience Processing**: Separate success trajectories (extract patterns) and failure trajectories (extract lessons)
- **Skill Library (SkillBank)**: Two-tier organization—general skills (universal) and task-specific skills
- **Semantic Retrieval**: Use embedding similarity to retrieve relevant skills for current state
- **Dynamic Evolution**: Analyze failure modes to generate new skills or update existing ones
- **Integration**: Skills prepend to context during policy rollouts

## Implementation

Process trajectories into skills by extracting high-level patterns:

```python
def extract_skills_from_trajectory(trajectory, success=True):
    """Extract skills from a trajectory (success or failure)."""
    if success:
        # For successful trajectories: extract strategic patterns
        # E.g., "When encountering X, use strategy Y"
        skill = {
            'type': 'strategic',
            'condition': identify_key_decision_points(trajectory),
            'action_pattern': abstract_action_sequence(trajectory),
            'outcome': 'success'
        }
    else:
        # For failures: extract lessons
        # E.g., "Avoid X because it leads to Y failure"
        failure_point = identify_failure_point(trajectory)
        skill = {
            'type': 'lesson',
            'condition': failure_point['state'],
            'anti_action': failure_point['action_taken'],
            'failure_mode': failure_point['reason'],
            'outcome': 'failure'
        }
    return skill

def compress_skill_text(skill, max_tokens=100):
    """Compress skill into concise text representation."""
    if skill['type'] == 'strategic':
        return f"When {skill['condition']}, {skill['action_pattern']} → success"
    else:
        return f"Avoid {skill['anti_action']} in {skill['condition']} (→ {skill['failure_mode']})"

# Build skill library from trajectory batch
skill_library = {'general': [], 'task_specific': {}}
for traj in trajectories:
    if traj['success']:
        skill = extract_skills_from_trajectory(traj, success=True)
        skill_library['general'].append(compress_skill_text(skill))
    else:
        skill = extract_skills_from_trajectory(traj, success=False)
        task = traj['task_id']
        if task not in skill_library['task_specific']:
            skill_library['task_specific'][task] = []
        skill_library['task_specific'][task].append(compress_skill_text(skill))
```

Retrieve relevant skills via embedding similarity:

```python
import torch
import torch.nn.functional as F

def retrieve_relevant_skills(current_state, skill_library, embedding_model, top_k=5):
    """Retrieve top-k skills relevant to current state."""
    # Embed current state
    state_embedding = embedding_model.encode(current_state)

    # Embed all skills
    all_skills = skill_library['general'] + sum(
        skill_library['task_specific'].values(), []
    )
    skill_embeddings = embedding_model.encode(all_skills)

    # Compute similarities
    similarities = F.cosine_similarity(
        state_embedding.unsqueeze(0),
        skill_embeddings,
        dim=-1
    )

    # Get top-k
    top_indices = torch.topk(similarities, min(top_k, len(all_skills)))[1]
    relevant_skills = [all_skills[i] for i in top_indices]

    return relevant_skills

# During rollout
def policy_forward_with_skills(state, policy, skill_library, embedding_model):
    """Generate action using policy augmented with relevant skills."""
    # Retrieve skills
    skills = retrieve_relevant_skills(state, skill_library, embedding_model, top_k=5)
    skill_context = "\n".join([f"- {s}" for s in skills])

    # Augment prompt with skills
    augmented_prompt = f"Skills:\n{skill_context}\n\nState: {state}\nAction:"

    # Generate action
    action = policy.generate(augmented_prompt, max_tokens=50)
    return action
```

Evolve skill library by analyzing failures:

```python
def analyze_failures_and_evolve_skills(trajectories, skill_library, embedding_model):
    """Analyze failures to generate new skills."""
    failed_trajectories = [t for t in trajectories if not t['success']]

    for failed_traj in failed_trajectories:
        # Extract failure lesson
        failure_skill = extract_skills_from_trajectory(failed_traj, success=False)
        skill_text = compress_skill_text(failure_skill)

        # Check if similar skill already exists
        existing_skills = skill_library['general'] + \
            sum(skill_library['task_specific'].values(), [])
        existing_embeddings = embedding_model.encode(existing_skills)
        new_embedding = embedding_model.encode(skill_text)

        similarities = F.cosine_similarity(
            new_embedding.unsqueeze(0),
            existing_embeddings,
            dim=-1
        )

        # If no similar skill exists, add it
        if similarities.max() < 0.8:
            task = failed_traj['task_id']
            if task not in skill_library['task_specific']:
                skill_library['task_specific'][task] = []
            skill_library['task_specific'][task].append(skill_text)

    return skill_library
```

## Practical Guidance

| Component | Recommendation | Notes |
|-----------|-----------------|-------|
| Skill extraction | Success + failure | Both provide value; failures prevent anti-patterns. |
| Skill retrieval top-k | 3-7 | Balance context length with coverage. |
| Similarity threshold | 0.75-0.85 | Avoid storing near-duplicate skills. |
| Evolution frequency | After each validation epoch | Timely skill updates enable policy adaptation. |

**When to Use**
- Long-horizon tasks where agents need to remember patterns (web navigation, dialogue)
- Multi-task learning where general skills apply across domains
- Domains with clear success/failure distinction

**When NOT to Use**
- Single-episode tasks with no trajectory reuse
- Domains where all trajectories are unique (low pattern reusability)

## Reference

See https://arxiv.org/abs/2602.08234 for full implementation, including skill library management, embedding models, and validation on ALFWorld household tasks.
