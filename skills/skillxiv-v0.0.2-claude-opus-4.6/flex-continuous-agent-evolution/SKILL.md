---
name: flex-continuous-agent-evolution
title: "FLEX: Continuous Agent Evolution via Forward Learning from Experience"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.06449"
keywords: [Reinforcement Learning, In-Context Learning, Experience Libraries, Agent Learning, Gradient-Free Optimization]
description: "Enable LLM agents to improve continuously during deployment by constructing structured experience libraries through self-reflection on successes and failures—achieving 23% improvement on reasoning without gradient-based parameter updates or external training."
---

# Evolve Agents Through Structured Experience Accumulation

Deployed language model agents are typically static—once trained, they don't improve from real-world interactions. FLEX solves this through gradient-free continuous learning: agents maintain a structured experience library recording successes, failures, and their contexts. During subsequent interactions, the agent retrieves and reflects on relevant past experiences, incorporating these lessons into prompting without retraining.

The approach demonstrates substantial gains: 23% improvement on mathematical reasoning (AIME25), 10% on chemical synthesis, 14% on protein engineering—all from self-refinement during deployment, not additional training.

## Core Concept

FLEX treats deployed agent improvement as a problem of structured experience management rather than parameter optimization. The system maintains three components:

1. **Experience Library**: Structured records of past interactions (state, action, outcome, reflection)
2. **Retrieval Mechanism**: Finding relevant precedents for current problems
3. **Self-Reflection**: Agents analyze successes/failures and distill lessons as prompting context

This approach is particularly powerful because it requires no gradient computation, model retraining, or API calls to external LLMs during learning—only structured reflection during inference.

## Architecture Overview

- **Experience Capture Module**: Records interactions (problem, solution attempt, outcome, contextual factors)
- **Structured Library**: Organizes experiences by problem domain, difficulty, technique type
- **Semantic Retrieval**: Finds relevant past experiences using embedding similarity or keyword matching
- **Reflection Engine**: Generates natural language summaries of why solutions succeeded/failed
- **Prompt Augmentation**: Incorporates retrieved experiences into in-context examples
- **Performance Tracking**: Measures improvement over time; identifies learning plateaus

## Implementation Steps

**Step 1: Experience Data Structure**

Define a structured format for recording and retrieving agent interactions.

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json
from datetime import datetime

@dataclass
class Experience:
    """Single interaction record in the experience library."""
    problem: str                    # Problem description
    solution_attempt: str           # Agent's attempted solution
    ground_truth: str              # Correct answer (if available)
    is_correct: bool               # Did the solution succeed?
    domain: str                    # Problem domain (math, coding, etc.)
    difficulty: str                # Estimated difficulty
    timestamp: str                 # When this occurred
    techniques_used: List[str]     # Techniques employed (e.g., 'divide-and-conquer')
    failure_reason: str            # Why it failed (if applicable)
    reflection: str                # Agent's own analysis of the attempt
    metadata: Dict[str, Any]       # Additional context (tokens used, latency, etc.)

    def to_dict(self):
        return {
            'problem': self.problem,
            'solution': self.solution_attempt,
            'correct': self.is_correct,
            'domain': self.domain,
            'difficulty': self.difficulty,
            'timestamp': self.timestamp,
            'techniques': self.techniques_used,
            'failure_reason': self.failure_reason,
            'reflection': self.reflection,
            'metadata': self.metadata
        }

class ExperienceLibrary:
    """Maintains structured experience collection."""

    def __init__(self, storage_path='./experience_library.jsonl'):
        self.storage_path = storage_path
        self.experiences: List[Experience] = []
        self.load_from_disk()

    def add_experience(self, exp: Experience):
        """Record a new experience."""
        self.experiences.append(exp)
        # Append to disk for persistence
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(exp.to_dict()) + '\n')

    def retrieve_relevant(self, problem: str, domain: str, k=3) -> List[Experience]:
        """
        Find most relevant past experiences for a given problem.

        Args:
            problem: Current problem description
            domain: Problem domain
            k: Number of experiences to retrieve

        Returns:
            relevant_experiences: Top-k similar past experiences
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Filter by domain first
        domain_exps = [e for e in self.experiences if e.domain == domain]

        if len(domain_exps) < k:
            return domain_exps

        # Compute similarity between current problem and past problems
        all_problems = [e.problem for e in domain_exps] + [problem]
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf = vectorizer.fit_transform(all_problems)

        # Similarity of current problem to all past problems
        similarities = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]

        # Sort by similarity and return top-k
        top_indices = similarities.argsort()[-k:][::-1]
        return [domain_exps[i] for i in top_indices]

    def load_from_disk(self):
        """Load experiences from persistent storage."""
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    exp_dict = json.loads(line)
                    self.experiences.append(Experience(**exp_dict))
        except FileNotFoundError:
            pass  # First run: empty library
```

**Step 2: Self-Reflection Engine**

Generate structured reflections on why attempts succeeded or failed.

```python
def generate_reflection(problem: str, solution: str, is_correct: bool,
                       ground_truth: str = None, llm_api=None) -> str:
    """
    Generate agent's reflection on an attempt.

    Args:
        problem: Original problem
        solution: Agent's attempted solution
        is_correct: Whether solution was correct
        ground_truth: Correct solution (if available)
        llm_api: LLM API for generating reflection (e.g., GPT-4, Claude)

    Returns:
        reflection: Natural language analysis
    """
    if is_correct:
        prompt = f"""Analyze why this solution was correct:

Problem: {problem}

Solution: {solution}

Provide a brief reflection on what techniques made this solution work:"""
    else:
        prompt = f"""Analyze why this solution failed:

Problem: {problem}

Your solution: {solution}

Correct solution: {ground_truth}

Identify the key mistake or misconception:"""

    # Call LLM to generate reflection
    if llm_api:
        reflection = llm_api.generate(prompt, max_tokens=200)
    else:
        # Fallback: simple pattern matching
        if "ValueError" in solution or "TypeError" in solution:
            reflection = "Code had syntax or type error"
        elif is_correct:
            reflection = "Solution approach was sound"
        else:
            reflection = "Solution logic was flawed"

    return reflection
```

**Step 3: Experience-Augmented Prompting**

Incorporate retrieved experiences into prompts during inference.

```python
def augment_prompt_with_experiences(
        original_prompt: str,
        relevant_experiences: List[Experience],
        include_failures: bool = True) -> str:
    """
    Create augmented prompt including relevant past experiences.

    Args:
        original_prompt: User's problem description
        relevant_experiences: Retrieved past experiences
        include_failures: Whether to include negative examples

    Returns:
        augmented_prompt: Enhanced prompt with examples
    """
    augmented = "You have access to relevant past experiences. Use insights from successes:\n\n"

    successful_exps = [e for e in relevant_experiences if e.is_correct]
    for i, exp in enumerate(successful_exps):
        augmented += f"Example {i+1} - Success:\n"
        augmented += f"Problem: {exp.problem}\n"
        augmented += f"Solution: {exp.solution_attempt}\n"
        augmented += f"Key insight: {exp.reflection}\n\n"

    if include_failures:
        failed_exps = [e for e in relevant_experiences if not e.is_correct]
        if failed_exps:
            augmented += "Learn from past mistakes:\n\n"
            for i, exp in enumerate(failed_exps):
                augmented += f"Past Mistake {i+1}:\n"
                augmented += f"Problem: {exp.problem}\n"
                augmented += f"Failed attempt: {exp.solution_attempt}\n"
                augmented += f"Why it failed: {exp.failure_reason}\n\n"

    augmented += f"Now solve this new problem:\n{original_prompt}"
    return augmented
```

**Step 4: Agent Deployment Loop**

Main loop integrating experience capture and retrieval during deployment.

```python
class DeployedAgent:
    """LLM agent that learns from deployment experiences."""

    def __init__(self, base_model, experience_library, domain='general'):
        self.model = base_model
        self.library = experience_library
        self.domain = domain
        self.success_count = 0
        self.total_attempts = 0

    def solve_problem(self, problem: str, ground_truth: str = None) -> Dict[str, Any]:
        """
        Solve a problem, recording experience for future learning.

        Args:
            problem: Problem description
            ground_truth: Correct answer (if available for offline validation)

        Returns:
            result: {solution, is_correct, experience_recorded}
        """
        # Step 1: Retrieve relevant past experiences
        relevant_exps = self.library.retrieve_relevant(problem, self.domain, k=3)

        # Step 2: Augment prompt with relevant experiences
        augmented_prompt = augment_prompt_with_experiences(
            problem, relevant_exps, include_failures=True
        )

        # Step 3: Generate solution
        solution = self.model.generate(augmented_prompt, max_tokens=2048)

        # Step 4: Validate (if ground truth available)
        is_correct = False
        if ground_truth:
            is_correct = self._validate_solution(solution, ground_truth)

        self.total_attempts += 1
        if is_correct:
            self.success_count += 1

        # Step 5: Generate reflection
        reflection = generate_reflection(problem, solution, is_correct, ground_truth)

        # Step 6: Record experience
        failure_reason = None
        if not is_correct:
            failure_reason = self._analyze_failure(solution, ground_truth)

        experience = Experience(
            problem=problem,
            solution_attempt=solution,
            ground_truth=ground_truth or '',
            is_correct=is_correct,
            domain=self.domain,
            difficulty=self._estimate_difficulty(problem),
            timestamp=datetime.now().isoformat(),
            techniques_used=self._extract_techniques(solution),
            failure_reason=failure_reason,
            reflection=reflection,
            metadata={
                'model': self.model.name,
                'num_examples': len(relevant_exps),
                'accuracy_rate': self.success_count / self.total_attempts
            }
        )

        self.library.add_experience(experience)

        return {
            'solution': solution,
            'is_correct': is_correct,
            'experience_recorded': True
        }

    def _validate_solution(self, solution: str, ground_truth: str) -> bool:
        """Check if solution matches ground truth."""
        # Simple string matching; extend for domain-specific validation
        return solution.strip() == ground_truth.strip()

    def _analyze_failure(self, solution: str, ground_truth: str) -> str:
        """Identify type of failure."""
        if "ValueError" in solution or "TypeError" in solution:
            return "Syntax/Type Error"
        elif len(solution) == 0:
            return "No Output Generated"
        else:
            return "Incorrect Logic"

    def _estimate_difficulty(self, problem: str) -> str:
        """Estimate problem difficulty."""
        word_count = len(problem.split())
        if word_count < 50:
            return "easy"
        elif word_count < 200:
            return "medium"
        else:
            return "hard"

    def _extract_techniques(self, solution: str) -> List[str]:
        """Identify reasoning techniques used."""
        techniques = []
        if "divide" in solution.lower() or "split" in solution.lower():
            techniques.append("divide-and-conquer")
        if "recursion" in solution.lower():
            techniques.append("recursion")
        if "dynamic" in solution.lower():
            techniques.append("dynamic-programming")
        if "greedy" in solution.lower():
            techniques.append("greedy")
        return techniques
```

**Step 5: Continuous Monitoring and Adaptation**

Track learning over time and identify when improvements plateau.

```python
def monitor_agent_learning(agent: DeployedAgent, window_size: int = 100):
    """
    Monitor improvement trends in agent performance.

    Args:
        agent: Deployed agent instance
        window_size: Number of recent attempts to analyze

    Yields:
        metrics: Performance statistics
    """
    while True:
        recent_exps = agent.library.experiences[-window_size:]

        if len(recent_exps) > 0:
            success_rate = sum(1 for e in recent_exps if e.is_correct) / len(recent_exps)
            avg_reflection_length = sum(
                len(e.reflection.split()) for e in recent_exps
            ) / len(recent_exps)

            metrics = {
                'success_rate': success_rate,
                'sample_count': agent.total_attempts,
                'improvement': success_rate,  # Compare to baseline if available
                'avg_reflection_length': avg_reflection_length,
                'unique_domains': len(set(e.domain for e in recent_exps))
            }

            yield metrics

            # If plateau detected, could trigger additional strategies
            if success_rate > 0.9:
                print("Agent has reached high performance; consider expanding domain")

        import time
        time.sleep(60)  # Monitor every minute
```

## Practical Guidance

**When to Use FLEX:**
- Deployed agents handling diverse, evolving tasks (continuous learning essential)
- Scenarios where retraining is expensive or time-consuming
- Systems where interpretability matters (experience reflections are human-readable)

**When NOT to Use:**
- Static batch learning (no deployment or evaluation feedback)
- Tasks with no clear success/failure signal (reflection requires outcome validation)
- Privacy-sensitive applications (experience library stores past interactions)

**Hyperparameters and Configuration:**
- Retrieval k: 3-5 examples per problem (balance context size with diversity)
- Reflection model: Use same LLM or smaller model to reduce cost
- Library retention: Keep all experiences initially; prune redundant ones after scale-up
- Domain granularity: Separate libraries for distinct problem types (math, code, reasoning)

**Pitfalls to Avoid:**
1. **Stale experiences** - Old experiences from early deployment may become outdated; prioritize recent successes
2. **Confirmation bias** - Over-weighting successful experiences; include failures to avoid learning wrong patterns
3. **Library bloat** - Experiences accumulate; implement deduplication or archival strategies
4. **No validation** - Without ground truth, incorrectly-classified successes pollute library; add validation when possible

---

Reference: https://arxiv.org/abs/2511.06449
