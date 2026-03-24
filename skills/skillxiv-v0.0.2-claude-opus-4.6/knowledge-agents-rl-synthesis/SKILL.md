---
name: knowledge-agents-rl-synthesis
title: "KARL: Knowledge Agents via Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.05218"
keywords: [Knowledge Agents, Synthetic Data, Reinforcement Learning, Multi-Task Training, Off-Policy Learning]
description: "Build knowledge agents that generalize across grounded reasoning tasks by combining agentic data synthesis via vector search with off-policy reinforcement learning. Use multi-task training on synthetic question-answer pairs to achieve robust cross-domain performance."
---

# KARL: Knowledge Agents via Reinforcement Learning

Retrieval-augmented agents struggle with distribution shift: training on one task's synthetic data doesn't transfer well to different retrieval demands. KARL addresses this through two coordinated mechanisms: agentic data synthesis that dynamically explores corpora to generate challenging examples, and multi-task off-policy RL that improves generalization. The key insight is to treat data synthesis as an agent behavior itself, then learn policies that work across diverse reasoning patterns.

The core innovation combines active learning (agents explore corpora) with meta-learning (training across diverse synthetic distributions) to build robust knowledge agents.

## Core Concept

KARL implements three coordinated components:

1. **Agentic Data Synthesis**: Question-Answer synthesizer explores document corpus via vector search, generates diverse QA pairs grounded in actual retrieval scenarios
2. **Multi-Task Training**: Train on heterogeneous tasks (cross-document synthesis, constraint-driven search) simultaneously for generalization
3. **Off-Policy RL**: Large-batch off-policy learning avoids online instabilities and enables efficient multi-task optimization

## Architecture Overview

- **Input**: Document corpus + task specifications (e.g., TREC, BrowseComp)
- **Synthesis Module**: Agent explores corpus with vector search, generates challenging QA pairs
- **Multi-Task Buffer**: Accumulate synthetic data from multiple task types
- **RL Trainer**: Optimize agent policy via advantage-based off-policy learning (OAPL)
- **Output**: Robust knowledge agent generalizing across tasks

## Implementation Steps

**Step 1: Design agentic data synthesis with vector search**

Create a synthesis agent that explores the corpus actively to generate training data.

```python
class SynthesisAgent:
    """
    Dynamically synthesizes QA pairs by exploring corpus via retrieval.
    """

    def __init__(self, corpus, encoder_model):
        """
        corpus: list of documents
        encoder_model: dense encoder for vector search (e.g., BERT, CLIP)
        """
        self.corpus = corpus
        self.encoder = encoder_model

        # Build vector index
        self.corpus_embeddings = [
            encoder_model.encode(doc) for doc in corpus
        ]
        self.corpus_index = self._build_faiss_index()

    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search."""
        import faiss

        embeddings = np.array(self.corpus_embeddings)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

        return index

    def explore_and_synthesize(self, num_examples=100, difficulty_schedule=None):
        """
        Actively explore corpus and generate synthetic QA pairs.

        difficulty_schedule: optional function(iteration) -> difficulty_level
        """
        synthetic_pairs = []

        for iteration in range(num_examples):
            # Determine difficulty target
            if difficulty_schedule:
                difficulty = difficulty_schedule(iteration)
            else:
                difficulty = 0.5  # Fixed medium difficulty

            # Step 1: Sample a document as context
            doc_idx = random.randint(0, len(self.corpus) - 1)
            context_doc = self.corpus[doc_idx]

            # Step 2: Generate question from document (using LLM)
            question = self._generate_question_from_doc(context_doc, difficulty)

            # Step 3: Execute retrieval search for this question
            query_embedding = self.encoder.encode(question)
            distances, indices = self.corpus_index.search(
                np.array([query_embedding]).astype(np.float32),
                k=5
            )

            # Step 4: Synthesize answer from top retrieved documents
            retrieved_docs = [self.corpus[idx] for idx in indices[0]]
            answer = self._synthesize_answer(question, retrieved_docs)

            synthetic_pairs.append({
                'question': question,
                'answer': answer,
                'retrieved_docs': retrieved_docs,
                'difficulty': difficulty
            })

        return synthetic_pairs

    def _generate_question_from_doc(self, doc, difficulty):
        """Generate question of specified difficulty from document."""
        prompt = f"""
Document: {doc[:200]}

Generate a {'simple' if difficulty < 0.3 else 'moderate' if difficulty < 0.7 else 'complex'} question about this document.
Question:
"""

        question = self.llm.generate(prompt, max_tokens=50)
        return question

    def _synthesize_answer(self, question, retrieved_docs):
        """Synthesize answer from retrieved documents."""
        prompt = f"""
Question: {question}

Relevant documents:
{' '.join(retrieved_docs[:3])}

Synthesize a concise answer based on the documents.
Answer:
"""

        answer = self.llm.generate(prompt, max_tokens=100)
        return answer
```

**Step 2: Create multi-task training buffer**

Accumulate synthetic data from different tasks and domains.

```python
class MultiTaskBuffer:
    """
    Replay buffer storing synthetic examples from multiple tasks.
    Enables multi-task learning via random task sampling.
    """

    def __init__(self, max_size_per_task=10000):
        self.tasks = {}
        self.max_size_per_task = max_size_per_task

    def add_task(self, task_name, task_description):
        """Register a new task."""
        self.tasks[task_name] = {
            'description': task_description,
            'buffer': []
        }

    def add_examples(self, task_name, examples):
        """Add synthetic examples to a task."""
        if task_name not in self.tasks:
            self.add_task(task_name, "Unnamed task")

        buffer = self.tasks[task_name]['buffer']
        buffer.extend(examples)

        # Enforce size limit (keep most recent examples)
        if len(buffer) > self.max_size_per_task:
            self.tasks[task_name]['buffer'] = buffer[-self.max_size_per_task:]

    def sample_batch(self, batch_size, task_distribution=None):
        """
        Sample batch across tasks.
        task_distribution: optional dict mapping task_name -> probability
        """
        if task_distribution is None:
            # Uniform task distribution
            task_distribution = {
                name: 1.0 / len(self.tasks)
                for name in self.tasks
            }

        batch = []

        for _ in range(batch_size):
            # Sample task according to distribution
            task_name = np.random.choice(
                list(task_distribution.keys()),
                p=list(task_distribution.values())
            )

            # Sample example from task
            task_buffer = self.tasks[task_name]['buffer']
            example = random.choice(task_buffer)

            batch.append((task_name, example))

        return batch

    def sample_diverse_batch(self, batch_size, diversity_metric='task'):
        """
        Sample diverse batch: ensure coverage across tasks.
        diversity_metric: 'task' (equal task coverage) or 'difficulty'
        """
        if diversity_metric == 'task':
            num_per_task = batch_size // len(self.tasks)
            batch = []

            for task_name in self.tasks:
                task_buffer = self.tasks[task_name]['buffer']
                task_samples = random.sample(
                    task_buffer,
                    min(num_per_task, len(task_buffer))
                )
                batch.extend([(task_name, ex) for ex in task_samples])

            return batch

        elif diversity_metric == 'difficulty':
            # Sort by difficulty, select evenly spaced
            all_examples = []
            for task_name in self.tasks:
                for ex in self.tasks[task_name]['buffer']:
                    all_examples.append((task_name, ex))

            all_examples.sort(key=lambda x: x[1].get('difficulty', 0.5))

            # Stratified sampling
            step = len(all_examples) // batch_size
            return [all_examples[i * step] for i in range(batch_size)]
```

**Step 3: Implement off-policy RL (OAPL)**

Train knowledge agent using large-batch off-policy optimization.

```python
def compute_advantages(rollouts, baseline_model, gamma=0.99):
    """
    Compute advantage estimates for rollouts.
    A_t = R_t - V_baseline(s_t)
    """
    advantages = []

    for rollout in rollouts:
        trajectory = rollout['trajectory']
        task_success = rollout['success']

        # Compute returns
        returns = []
        cumulative_return = 0.0

        for t in reversed(range(len(trajectory))):
            if t == len(trajectory) - 1:
                cumulative_return = float(task_success)
            else:
                cumulative_return = float(task_success) + gamma * cumulative_return

            returns.insert(0, cumulative_return)

        # Baseline values
        values = []
        for step in trajectory:
            value = baseline_model(step['state']).item()
            values.append(value)

        # Advantages
        step_advantages = [r - v for r, v in zip(returns, values)]
        advantages.append(step_advantages)

    return advantages

def off_policy_update(agent, baseline, buffer, batch_size=128,
                     learning_rate=1e-4, num_epochs=3):
    """
    Off-policy RL update using importance sampling.
    Collects large batch, then performs multiple gradient steps.
    """
    optimizer = torch.optim.AdamW(agent.parameters(), lr=learning_rate)

    # Sample diverse batch
    batch = buffer.sample_diverse_batch(batch_size, diversity_metric='task')

    # Collect rollouts
    rollouts = []
    for task_name, example in batch:
        trajectory, success = agent.rollout(example)
        rollouts.append({
            'trajectory': trajectory,
            'success': success,
            'task': task_name
        })

    # Compute advantages
    advantages = compute_advantages(rollouts, baseline)

    # Multiple gradient steps on same batch
    for epoch in range(num_epochs):
        total_loss = 0.0

        for rollout, step_advantages in zip(rollouts, advantages):
            for t, (step, advantage) in enumerate(
                zip(rollout['trajectory'], step_advantages)
            ):
                # Policy gradient
                action = step['action']
                logprob = agent.compute_logprob(step['state'], action)

                # PG loss with advantage weighting
                pg_loss = -logprob * advantage

                # KL regularization for stability
                ref_logprob = agent.reference_model.compute_logprob(
                    step['state'], action
                )
                kl_loss = logprob - ref_logprob

                # Combined loss
                loss = pg_loss + 0.01 * kl_loss

                total_loss += loss

        # Gradient step
        optimizer.zero_grad()
        (total_loss / len(rollouts)).backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss.item() / len(rollouts)
```

**Step 4: Implement multi-task training loop**

Alternate between data synthesis and RL updates.

```python
def train_knowledge_agent(corpus, agent_model, baseline_model,
                         num_iterations=1000):
    """
    Train knowledge agent via multi-task learning:
    1. Synthesize data from different tasks
    2. Update agent via off-policy RL
    3. Iterate
    """
    # Initialize components
    synthesizer = SynthesisAgent(corpus)
    buffer = MultiTaskBuffer()

    # Register tasks
    buffer.add_task('cross_document', 'Synthesize answers from multiple documents')
    buffer.add_task('constraint_search', 'Answer questions with specific constraints')

    for iteration in range(num_iterations):
        # Phase 1: Synthesize data
        if iteration % 10 == 0:  # Re-synthesize periodically
            print(f"Iteration {iteration}: Synthesizing data...")

            # Synthesize examples for each task
            cross_doc_examples = synthesizer.explore_and_synthesize(
                num_examples=50,
                difficulty_schedule=lambda i: min(i / 50, 1.0)
            )
            buffer.add_examples('cross_document', cross_doc_examples)

            constraint_examples = synthesizer.explore_and_synthesize(
                num_examples=30
            )
            buffer.add_examples('constraint_search', constraint_examples)

        # Phase 2: Off-policy RL update
        loss = off_policy_update(
            agent_model,
            baseline_model,
            buffer,
            batch_size=32,
            num_epochs=2
        )

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}")

    return agent_model
```

**Step 5: Evaluation on diverse benchmarks**

Test generalization across different reasoning tasks.

```python
def evaluate_generalization(agent, eval_tasks):
    """
    Evaluate: does agent generalize across diverse tasks?
    """
    results = {}

    for task_name, examples in eval_tasks.items():
        correct = 0
        total = len(examples)

        for example in examples:
            # Execute agent
            trajectory, success = agent.rollout(example)

            # Check correctness
            predicted_answer = trajectory[-1]['output']
            expected_answer = example['answer']

            if predicted_answer.lower() in expected_answer.lower():
                correct += 1

        accuracy = correct / total
        results[task_name] = accuracy

        print(f"{task_name}: {accuracy * 100:.1f}%")

    # Compute harmonic mean for overall generalization score
    accuracies = list(results.values())
    harmonic_mean = len(accuracies) / sum(1.0 / (a + 1e-8) for a in accuracies)

    print(f"\nOverall generalization score: {harmonic_mean * 100:.1f}%")

    return results
```

## Practical Guidance

**Hyperparameter Selection:**
- **Synthesis exploration budget**: 50-200 examples per task. More = better coverage; diminishing returns beyond 100.
- **Task distribution**: Start uniform; can weight difficult tasks higher in later training.
- **Off-policy batch size**: 32-128. Larger batches = more stable gradients; memory-limited systems use smaller.
- **KL regularization strength**: 0.01-0.05. Prevents instability; too high can prevent improvement.
- **Difficulty schedule**: Linear 0→1 over synthesis examples; exponential schedules also work well.

**When to Use:**
- Knowledge-grounded QA tasks requiring cross-document reasoning
- Multi-domain scenarios where single-task training fails
- Settings with access to large document corpus for synthesis
- Tasks benefiting from synthetic data augmentation

**When NOT to Use:**
- Single-task, well-defined problems (simpler approaches sufficient)
- Real-time systems with low latency budgets (synthesis is expensive)
- Domains with limited or unreliable document corpus
- Tasks where synthetic data distribution significantly differs from real distribution

**Common Pitfalls:**
- **Synthesis distribution mismatch**: If synthesized QA pairs don't match evaluation distribution, generalization fails. Validate synthesis quality on held-out set.
- **Task imbalance**: If one task has much larger buffer, it dominates training. Use explicit task weighting or equal-size buffers.
- **Baseline instability**: Poor baseline estimates lead to high-variance advantages. Pre-train baseline on supervised data.
- **Off-policy divergence**: If policy drifts too far from data collection policy, importance weights explode. Monitor KL divergence; add explicit bounds.

## Reference

arXiv: https://arxiv.org/abs/2603.05218
