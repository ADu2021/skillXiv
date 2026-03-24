---
name: web-coach-self-evolving-agents
title: "WebCoach: Self-Evolving Web Agents with Cross-Session Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.12997"
keywords: [Web Agents, Episodic Memory, Self-Evolution, Cross-Session Learning, Runtime Guidance]
description: "Enable web agents to improve over time by curating episodic memory from navigation trajectories and dynamically injecting task-specific guidance at runtime—no retraining required, persistent improvement across sessions."
---

# Enable Web Agents to Self-Improve via Cross-Session Episodic Memory

Web agents (browser automation, form filling, multi-step navigation) often fail on edge cases or novel sites. Retraining on every failure is expensive. WebCoach enables **self-evolution without retraining**: the agent maintains an External Memory Store (EMS) of past navigation episodes, continuously refining its decision-making by retrieving and injecting relevant past experiences at runtime.

A Coach component evaluates the agent's current trajectory and injects guidance when it detects failure risk or identifies superior strategies in memory. This achieves persistent agent improvement across sessions without model updates—learning happens through retrieval and composition, not gradient descent.

## Core Concept

Web agents typically operate in isolation: each session is independent, successful and failed trajectories are discarded. WebCoach breaks this pattern by maintaining a shared episodic memory across sessions. When an agent encounters a new task, the Coach retrieves similar past episodes and injects relevant guidance into the agent's context.

The system comprises three components working in tandem:

1. **WebCondenser**: Summarizes raw navigation logs into structured episodes with embeddings, success labels, and error patterns
2. **External Memory Store (EMS)**: Persists completed episodes with semantic search capability
3. **Coach**: LLM-based runtime decision engine that retrieves relevant episodes and injects guidance when needed

Together, they enable agents to learn from cross-session experience without modifying the underlying policy.

## Architecture Overview

- **Navigation Trajectory Logging**: Capture observation-action-reward sequences; store only completed episodes (success or failure endpoints)
- **WebCondenser**: Convert raw logs to structured episodes with (summary, embedding, success_label, error_patterns)
- **External Memory Store (EMS)**: Semantic index enabling fast retrieval of similar past episodes; stores up to millions of episodes
- **Coach Decision Engine**: Evaluate current trajectory; retrieve top-k similar episodes; decide if intervention needed; inject guidance as system message
- **No-Retraining Integration**: Coach advice appends to message history; agent processes without policy modification

## Implementation Steps

**Step 1: WebCondenser—Normalize Navigation Logs.** Convert raw traces to structured episodes.

```python
class WebCondenser:
    def __init__(self, llm_model='gpt2', embedding_dim=1536):
        self.llm = load_small_llm(llm_model)  # ≤8B params for speed
        self.embedding_model = load_embedding_model()
        self.embedding_dim = embedding_dim

    def condense_trajectory(self, trajectory):
        """
        Convert observation-action-reward trajectory to structured episode.
        trajectory: list of (observation, action, reward) tuples
        """
        # Format trajectory as narrative
        narrative = self._format_trajectory_text(trajectory)

        # Summarize with LLM
        summary = self.llm.summarize(
            narrative,
            max_tokens=150,
            system_prompt="Concisely summarize this web navigation trajectory in 3-5 sentences, noting task, key actions, and outcome."
        )

        # Extract success/failure from final reward
        final_reward = trajectory[-1][2]
        success = final_reward > 0

        # Extract error patterns
        error_patterns = self._extract_error_patterns(trajectory)

        # Compute embedding
        embedding = self.embedding_model.encode(summary)  # (embedding_dim,)

        return {
            'summary': summary,
            'embedding': embedding,
            'success': success,
            'error_patterns': error_patterns,
            'trajectory_hash': hash(str(trajectory))  # Deduplication
        }

    def _format_trajectory_text(self, trajectory):
        """Convert trajectory to human-readable text."""
        lines = []
        for obs, action, reward in trajectory:
            lines.append(f"Observation: {obs}")
            lines.append(f"Action: {action}")
            lines.append(f"Reward: {reward}")
        return "\n".join(lines)

    def _extract_error_patterns(self, trajectory):
        """Identify recurring error patterns in trajectory."""
        patterns = []

        for obs, action, reward in trajectory:
            if reward < 0:  # Error detected
                # Extract action type and context
                action_type = action.split()[0] if action else "unknown"
                patterns.append({
                    'action_type': action_type,
                    'observation': obs,
                    'error_context': obs
                })

        # Deduplicate and compress
        return patterns[:5]  # Keep top-5 error patterns
```

**Step 2: External Memory Store (EMS).** Persist and index episodes for retrieval.

```python
class ExternalMemoryStore:
    def __init__(self, embedding_dim=1536, max_episodes=1000000):
        self.episodes = []  # List of structured episodes
        self.embeddings = np.zeros((0, embedding_dim))
        self.embedding_dim = embedding_dim
        self.max_episodes = max_episodes
        self.index = None  # FAISS index for fast retrieval

    def add_episode(self, episode):
        """
        Persist episode to EMS.
        episode: dict from WebCondenser.condense_trajectory
        """
        # Deduplication: skip if trajectory_hash already exists
        if any(e['trajectory_hash'] == episode['trajectory_hash'] for e in self.episodes):
            return

        self.episodes.append(episode)
        self.embeddings = np.vstack([
            self.embeddings,
            episode['embedding'].reshape(1, -1)
        ])

        # Rebuild FAISS index periodically
        if len(self.episodes) % 1000 == 0:
            self._rebuild_index()

        # Evict oldest if at capacity
        if len(self.episodes) > self.max_episodes:
            self._evict_oldest()

    def retrieve_similar(self, query_embedding, k=5):
        """
        Retrieve k most similar past episodes.
        query_embedding: (embedding_dim,) array
        """
        if self.index is None or len(self.episodes) == 0:
            return []

        # FAISS search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(k, len(self.episodes))
        )

        # Return episodes sorted by similarity
        retrieved = [self.episodes[i] for i in indices[0]]
        return retrieved

    def _rebuild_index(self):
        """Build FAISS index for fast similarity search."""
        import faiss
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))

    def _evict_oldest(self):
        """Remove oldest episodes when at capacity."""
        # Remove oldest 10% by creation timestamp
        num_remove = int(0.1 * len(self.episodes))
        self.episodes = self.episodes[num_remove:]
        self.embeddings = self.embeddings[num_remove:]
        self._rebuild_index()
```

**Step 3: Coach Decision Engine.** Evaluate trajectory and decide when to inject guidance.

```python
class Coach:
    def __init__(self, coach_llm, ems):
        self.coach_llm = coach_llm  # Small LLM (≤8B)
        self.ems = ems

    def evaluate_trajectory(self, current_trajectory, current_embedding):
        """
        Evaluate current trajectory for intervention need.
        current_trajectory: list of (obs, action, reward)
        current_embedding: embedding of current partial trajectory
        """
        # Retrieve similar past episodes
        similar_episodes = self.ems.retrieve_similar(current_embedding, k=5)

        if not similar_episodes:
            return None  # No guidance available

        # Assess failure risk
        failure_risk = self._assess_failure_risk(current_trajectory)

        # Identify better strategies
        better_strategies = [
            ep for ep in similar_episodes if ep['success'] and ep['reward'] > self._get_current_reward(current_trajectory)
        ]

        # Decide intervention
        intervene = (failure_risk > 0.6) or (len(better_strategies) > 0 and failure_risk > 0.3)

        if intervene:
            guidance = self._generate_guidance(current_trajectory, similar_episodes, better_strategies)
            return guidance
        return None

    def _assess_failure_risk(self, trajectory):
        """Estimate probability of trajectory failure."""
        if not trajectory:
            return 0.0

        # Heuristic: failure risk increases with repeated errors
        recent_rewards = [r for _, _, r in trajectory[-5:]]
        num_negative = sum(1 for r in recent_rewards if r < 0)

        failure_risk = num_negative / len(recent_rewards) if recent_rewards else 0.0
        return failure_risk

    def _generate_guidance(self, current_trajectory, similar_episodes, better_strategies):
        """
        Generate actionable guidance from similar episodes.
        Output as JSON for injection into agent's message history.
        """
        guidance_prompt = f"""
        Current trajectory (partial): {current_trajectory[-3:]}

        Similar successful episodes from past:
        {[ep['summary'] for ep in similar_episodes[:3]]}

        Better strategies identified:
        {[ep['summary'] for ep in better_strategies[:2]]}

        Generate JSON guidance with:
        - "suggested_action": next action to try
        - "rationale": why this action
        - "error_to_avoid": common mistakes in similar situations
        """

        guidance_json = self.coach_llm(guidance_prompt)
        return json.loads(guidance_json)

    def inject_guidance(self, agent_message_history, guidance):
        """
        Append guidance to agent's message history as system message.
        No agent retraining needed; guidance processed at inference.
        """
        guidance_message = {
            'role': 'system',
            'content': f"""Coach guidance from past episodes:
            Suggested next action: {guidance['suggested_action']}
            Rationale: {guidance['rationale']}
            Error to avoid: {guidance['error_to_avoid']}"""
        }

        agent_message_history.append(guidance_message)
        return agent_message_history
```

**Step 4: Integration Loop.** Continuously update memory and guide agents.

```python
def self_evolving_agent_loop(agent, coach, ems, condenser, task):
    """
    Main loop: agent acts, coach guides, memory updates.
    """
    trajectory = []
    message_history = [{'role': 'user', 'content': task}]

    while not agent.is_task_complete():
        # Agent generates action
        action = agent.step(message_history)
        observation = execute_action(action)
        reward = evaluate_reward(task, observation)

        trajectory.append((observation, action, reward))

        # Coach evaluates and guides
        episode_embedding = condenser.embedding_model.encode(str(trajectory))
        guidance = coach.evaluate_trajectory(trajectory, episode_embedding)

        if guidance:
            message_history = coach.inject_guidance(message_history, guidance)

        # Append to message history
        message_history.append({'role': 'assistant', 'content': action})
        message_history.append({'role': 'user', 'content': f"Observation: {observation}"})

    # Store episode for future reference
    final_episode = condenser.condense_trajectory(trajectory)
    ems.add_episode(final_episode)

    return trajectory
```

## Practical Guidance

**When to Use:** Web automation tasks (form filling, web navigation, data extraction) where agents encounter similar problems repeatedly across sessions; cost of retraining prohibitive.

**Architecture Decisions:**
- Condenser LLM size: 8B is good balance of speed and quality; use smaller for latency-critical apps
- EMS capacity: 100K–1M episodes depending on RAM budget; use FAISS for efficient retrieval
- Guidance injection frequency: check every 3–5 steps; avoid excessive interruptions
- Success criteria: define clearly (e.g., task complete, form submitted) for episode labeling

**Pitfalls:**
- **Memory corruption**: Conditioning on irrelevant past episodes degrades performance; ensure similarity-based retrieval is accurate
- **Guidance misalignment**: Coach guidance might conflict with agent's current thinking; add confidence thresholds before injection
- **Unbounded memory growth**: Implement strict eviction policies; monitor EMS size regularly
- **Cold start**: Agent lacks guidance on first session; seed EMS with expert demonstrations or synthetic data

**When NOT to Use:** One-off tasks; tasks with highly variable state spaces where similarity is hard to judge; real-time systems where retrieval latency matters.

**Integration:** Compatible with any web automation agent (Selenium, Playwright, LLM-based); no retraining, works with frozen policies.

---
Reference: https://arxiv.org/abs/2511.12997
