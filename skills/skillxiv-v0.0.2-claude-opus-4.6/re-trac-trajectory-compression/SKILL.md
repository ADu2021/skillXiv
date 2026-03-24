---
name: re-trac-trajectory-compression
title: "RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02486"
keywords: [Search Agents, Trajectory Compression, Recursive Planning, Evidence Tracking]
description: "Compress search trajectories into structured states capturing partial answers, evidence, and uncertainties. Recursive execution leverages compressed states to avoid redundant exploration, improving resource efficiency by 50%."
---

# RE-TRAC: Recursive Trajectory Compression for Search

## Problem
ReAct-style search agents suffer from incomplete branch exploration—up to 93% of planned branches never execute due to linear trajectory structure. Independent parallel attempts (Pass@k scaling) miss cross-trajectory learning.

Long sequences consume tokens and computational resources. Agents need structured, compressible representations of search progress.

## Core Concept
RE-TRAC compresses search trajectories into structured state representations capturing answer progress, evidence base, and unresolved questions. These compressed states seed the next search round, enabling targeted exploration of incomplete branches.

Recursive execution with cross-trajectory learning reduces resource consumption while improving final performance.

## Architecture Overview

- **Structured State Representation**: Answer & Conclusions, Evidence Base, Uncertainties
- **Evidence Tracking**: Source verification and provenance for all claims
- **Recursive Execution**: Use compressed states as input for continuation rounds
- **Branch Exploration**: Systematically address unresolved questions from previous round
- **Resource Monitoring**: Track cumulative computation across rounds

## Implementation

### Step 1: Define Structured State Representation
Create compress format for search progress.

```python
class StructuredSearchState:
    def __init__(self):
        self.partial_answers = []  # Best-supported answers found so far
        self.evidence_base = []    # All discovered evidence with sources
        self.uncertainties = []    # Open questions needing exploration

    def add_partial_answer(self, answer_text, confidence, supporting_evidence):
        """Record a potential answer with its support."""
        self.partial_answers.append({
            'text': answer_text,
            'confidence': confidence,
            'evidence': supporting_evidence
        })

    def add_evidence(self, claim, source_url, verification_status):
        """Store evidence with provenance tracking."""
        self.evidence_base.append({
            'claim': claim,
            'source': source_url,
            'verified': verification_status,
            'timestamp': datetime.now().isoformat()
        })

    def add_uncertainty(self, question, reason, next_steps):
        """Record unresolved questions."""
        self.uncertainties.append({
            'question': question,
            'why_unresolved': reason,
            'suggested_exploration': next_steps
        })

    def to_prompt(self):
        """Convert state to prompt for next search round."""
        prompt = "Previous search progress:\n\n"

        if self.partial_answers:
            prompt += "Current answers (by confidence):\n"
            sorted_answers = sorted(self.partial_answers, key=lambda x: x['confidence'], reverse=True)
            for ans in sorted_answers[:3]:
                prompt += f"- {ans['text']} (confidence: {ans['confidence']})\n"

        if self.evidence_base:
            prompt += "\nKnown evidence:\n"
            for ev in self.evidence_base[:5]:
                status = "verified" if ev['verified'] else "unverified"
                prompt += f"- {ev['claim']} [{status}] ({ev['source']})\n"

        if self.uncertainties:
            prompt += "\nOutstanding questions:\n"
            for unc in self.uncertainties:
                prompt += f"- {unc['question']}\n"
                prompt += f"  Suggested: {unc['suggested_exploration']}\n"

        return prompt
```

### Step 2: Compress Search Trajectory
Extract structured state from completed search trajectory.

```python
def compress_trajectory_to_state(trajectory, answer_model, evidence_model):
    """Convert trajectory of actions into structured state."""
    state = StructuredSearchState()

    # Extract partial answers from trajectory
    for step in trajectory:
        if 'answer_attempt' in step:
            # Evaluate answer quality
            confidence = answer_model.score_answer(
                step['answer_attempt'],
                step['question'],
                step['evidence_seen']
            )

            state.add_partial_answer(
                step['answer_attempt'],
                confidence,
                step['evidence_seen']
            )

    # Build evidence base with verification
    unique_claims = extract_claims_from_trajectory(trajectory)
    for claim in unique_claims:
        sources = find_sources_for_claim(trajectory, claim)
        verified = verify_claim(claim, sources)

        for source in sources:
            state.add_evidence(claim, source, verified)

    # Identify unresolved questions
    planned_branches = extract_planned_exploration(trajectory)
    executed_branches = extract_executed_actions(trajectory)

    for branch in planned_branches:
        if branch not in executed_branches:
            state.add_uncertainty(
                question=branch,
                reason="Planned but not explored in this round",
                next_steps=f"Execute planned branch: {branch}"
            )

    return state
```

### Step 3: Recursive Search with Compressed States
Execute multiple search rounds using compressed states.

```python
def recursive_search_with_compression(question, search_agent, num_rounds=3, max_total_tokens=5000):
    """Perform recursive search leveraging trajectory compression."""
    current_state = None
    total_tokens_used = 0
    all_trajectories = []

    for round_num in range(num_rounds):
        # Build search prompt
        if current_state is None:
            search_prompt = f"Answer this question: {question}"
        else:
            search_prompt = f"Question: {question}\n\n{current_state.to_prompt()}\n\nContinue exploring unanswered questions."

        # Execute search (capped tokens)
        tokens_remaining = max_total_tokens - total_tokens_used
        trajectory = search_agent.search(search_prompt, max_tokens=tokens_remaining)

        all_trajectories.append(trajectory)
        total_tokens_used += trajectory['tokens_used']

        # Compress trajectory for next round
        current_state = compress_trajectory_to_state(trajectory, answer_model, evidence_model)

        # Check convergence
        best_answer = max(current_state.partial_answers, key=lambda x: x['confidence'])
        if best_answer['confidence'] > 0.95 or total_tokens_used >= max_total_tokens:
            break

    return current_state, all_trajectories, total_tokens_used
```

### Step 4: Targeted Branch Completion
Systematically address previously unexecuted branches.

```python
def complete_unexecuted_branches(state, search_agent, tokens_per_branch=500):
    """Target remaining unexecuted branches from previous rounds."""
    for uncertainty in state.uncertainties:
        if 'Execute planned branch' in uncertainty['suggested_exploration']:
            # Extract branch description
            branch_query = uncertainty['question']

            # Focused search on this branch
            branch_prompt = f"Focus on this specific question: {branch_query}\n\n"
            branch_prompt += f"Context:\n{state.to_prompt()}\n\n"
            branch_prompt += "Provide thorough exploration of this specific branch."

            trajectory = search_agent.search(branch_prompt, max_tokens=tokens_per_branch)

            # Update state with new findings
            new_state_fragment = compress_trajectory_to_state(trajectory, answer_model, evidence_model)

            # Merge new findings into existing state
            state.partial_answers.extend(new_state_fragment.partial_answers)
            state.evidence_base.extend(new_state_fragment.evidence_base)

    return state
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max rounds per search | 3-5 | Diminishing returns beyond 5 |
| Total token budget | 4000-8000 | Comparable to single long trajectory |
| Confidence threshold | 0.8-0.95 | Convergence criterion |
| Evidence verification | Full sampling | Cite all claims |
| Branch selection strategy | Uncertainty-guided | Prioritize unresolved questions |

### When to Use

- Open-domain search agents (web search, knowledge graphs)
- Multi-round research tasks with incomplete information
- Scenarios where recursion enables better branch coverage
- Tasks benefiting from explicit uncertainty tracking
- Long-horizon exploration with structured intermediate states

### When Not to Use

- Single-round question answering (overhead not justified)
- Deterministic environments with single optimal path
- Real-time systems requiring immediate response
- Tasks without clear evidence-gathering structure
- Environments where state compression loses critical information

### Common Pitfalls

1. **State representation bloat**: Careful pruning prevents state explosion. Keep only high-confidence findings.
2. **Branch redundancy**: Earlier rounds may re-explore same branches. Track and skip already-executed branches.
3. **Evidence hallucination**: Verify all claims before adding to evidence base. Use verification model.
4. **Premature convergence**: Early high-confidence answers may be incorrect. Require multiple confirming sources.

## Reference
RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents
https://arxiv.org/abs/2602.02486
