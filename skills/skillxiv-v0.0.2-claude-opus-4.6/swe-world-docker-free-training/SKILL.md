---
name: swe-world-docker-free-training
title: "SWE-World: Building Software Engineering Agents in Docker-Free Environments"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03419"
keywords: [Software Engineering Agents, Training Environment, Learned Simulators, LLM-Based Surrogates, Cost Reduction]
description: "Replace Docker environments with learned LLM surrogates comprising a lightweight sandbox for deterministic file operations, a transition model predicting execution feedback, and a reward model acting as virtual test runner. Reduces training infrastructure costs while scaling SWE agent training data."
---

# SWE-World: LLM-Based Surrogate Training Environments

Training software engineering agents typically requires expensive infrastructure to run code in isolated Docker containers. SWE-World replaces this with learned surrogates that simulate execution feedback, dramatically reducing infrastructure overhead while enabling larger-scale training data.

The key insight is that execution can be decomposed: deterministic file operations (navigation, editing) can be handled directly, while non-deterministic execution effects (stdout, exit codes, test results) are predicted by lightweight models. This hybrid approach maintains fidelity while eliminating Docker overhead.

## Core Concept

SWE-World comprises three components working in tandem:

1. **Lightweight Sandbox**: Handles all deterministic state transitions (file system operations) directly without invoking external processes
2. **Transition Model (SWT)**: Predicts step-level execution output—stdout, stderr, exit codes—based on file state and command
3. **Reward Model (SWR)**: Acts as a virtual test runner, generating test reports and binary success signals

This design preserves task semantics while replacing expensive runtime dependencies with efficient learned components.

## Architecture Overview

- **Deterministic Sandbox**: Fast, stateless file operation handling (read, write, delete, navigate) with in-memory state
- **SWT (Transition Model)**: Small LLM fine-tuned to predict execution trace outputs given command and file context
- **SWR (Reward Model)**: Outputs test success/failure signals, trained on real test execution traces
- **Training Loop**: Agents generate sequences of commands; sandbox updates file state; models predict execution effects
- **Data Scaling**: Larger training sets possible since models are faster than real Docker execution

## Implementation

### Step 1: Build the Deterministic Sandbox

Create a lightweight virtual file system that tracks state without external process invocation.

```python
# Lightweight deterministic sandbox
class DeterministicSandbox:
    def __init__(self):
        self.file_system = {}
        self.current_dir = "/"
        self.env_vars = {}

    def read_file(self, path: str) -> str:
        """Read file from virtual filesystem."""
        full_path = self._resolve_path(path)
        if full_path not in self.file_system:
            return f"Error: File not found: {path}"
        return self.file_system[full_path]

    def write_file(self, path: str, content: str) -> str:
        """Write to virtual filesystem."""
        full_path = self._resolve_path(path)
        self.file_system[full_path] = content
        return f"Wrote {len(content)} bytes to {path}"

    def list_files(self, path: str = ".") -> str:
        """List directory contents."""
        full_path = self._resolve_path(path)
        files = [f for f in self.file_system.keys()
                 if f.startswith(full_path)]
        return "\n".join(files)

    def _resolve_path(self, path: str) -> str:
        """Resolve relative path to absolute."""
        if path.startswith("/"):
            return path
        return f"{self.current_dir}/{path}".replace("//", "/")
```

### Step 2: Train the Transition Model (SWT)

Fine-tune an LLM to predict execution outputs given command and file context.

```python
# Transition model training
def prepare_swe_training_data(execution_traces: List[dict]) -> List[dict]:
    """Convert execution traces to SWT training examples."""
    examples = []
    for trace in execution_traces:
        for step in trace['steps']:
            example = {
                'input': f"""
Command: {step['command']}
Current directory: {step['cwd']}
File context: {format_file_context(step['files'])}
Environment: {step['env']}
                """,
                'target': step['output']  # stdout/stderr
            }
            examples.append(example)
    return examples

def train_transition_model(traces: List[dict], model_name: str = 'gpt-3.5-turbo'):
    """Fine-tune transition model on execution traces."""
    train_data = prepare_swe_training_data(traces)

    # Create training dataset
    training_examples = [
        f"Input: {ex['input']}\nOutput: {ex['target']}"
        for ex in train_data
    ]

    # Fine-tune on lightweight model
    # (Using API-based fine-tuning for illustration)
    response = openai.FineTuningJob.create(
        training_file=upload_to_openai(training_examples),
        model=model_name,
        n_epochs=3,
        batch_size=32
    )
    return response.fine_tuned_model
```

### Step 3: Train the Reward Model (SWR)

Create a model that predicts test success/failure based on execution state.

```python
# Reward model training
def prepare_reward_training_data(test_results: List[dict]) -> List[dict]:
    """Convert test results to SWR training examples."""
    examples = []
    for result in test_results:
        example = {
            'input': f"""
Test command: {result['command']}
Exit code: {result['exit_code']}
Output: {result['output']}
Error: {result['error']}
            """,
            'label': 1 if result['passed'] else 0
        }
        examples.append(example)
    return examples

def train_reward_model(test_results: List[dict]):
    """Train reward model on test execution outcomes."""
    train_data = prepare_reward_training_data(test_results)

    # Prepare binary classification dataset
    texts = [ex['input'] for ex in train_data]
    labels = [ex['label'] for ex in train_data]

    # Use lightweight classifier fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='./reward_model',
            num_train_epochs=3,
            per_device_train_batch_size=32
        ),
        train_dataset=Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
    )
    trainer.train()
    return trainer.model
```

### Step 4: Integrate Sandbox and Models into Agent Training Loop

Combine the sandbox and learned models to simulate execution during training.

```python
# Agent training with surrogate environment
def train_swe_agent_with_surrogate(
    agent_model: str,
    sandbox: DeterministicSandbox,
    transition_model,
    reward_model,
    dataset: List[dict]
):
    """Train SWE agent using surrogate environment."""

    for episode in range(num_episodes):
        task = dataset[episode % len(dataset)]
        sandbox.reset()

        agent_trajectory = []
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # Get agent's next action
            state_repr = format_state(sandbox.file_system, sandbox.current_dir)
            action = agent_model.get_next_action(
                task_description=task['description'],
                current_state=state_repr,
                previous_actions=agent_trajectory
            )

            # Execute in sandbox (deterministic file ops)
            file_effect = sandbox.execute_file_operation(action)

            # Predict execution output using SWT
            predicted_output = transition_model.predict(
                command=action,
                file_context=format_file_context(sandbox.file_system),
                cwd=sandbox.current_dir
            )

            # Check if task is complete using SWR
            task_success = reward_model.predict(
                output=predicted_output,
                command=action,
                task=task['description']
            )

            agent_trajectory.append({
                'action': action,
                'predicted_output': predicted_output,
                'success': task_success
            })

            if task_success > 0.5:  # Threshold for success
                done = True

            step_count += 1

        # Compute episode reward and update agent
        episode_reward = 1.0 if done else -0.1 * step_count
        agent_model.update(
            trajectory=agent_trajectory,
            reward=episode_reward
        )

    return agent_model
```

## Practical Guidance

**When to use Docker-free surrogate training:**
- Large-scale SWE agent training with 10K+ task instances
- Development/prototyping where Docker overhead slows iteration
- Scenarios where exact execution fidelity is less critical than scalability
- Cost-constrained environments lacking containerization infrastructure

**When to use real Docker:**
- Production evaluation requiring exact execution behavior
- Tasks with complex external dependencies (databases, services)
- Security-critical scenarios requiring true isolation
- High-consequence deployments where fidelity is paramount

**Common Pitfalls:**
- Train-inference mismatch: SWT/SWR trained on limited domains may hallucinate on novel code patterns
- Context window saturation: Large file contents can exceed model input limits; use summary representations
- Missing failure modes: Surrogates may not capture rare error conditions; validate on real environment periodically
- Stale model assumptions: Periodically retrain models as agent behavior and command distributions drift

**Hyperparameter Guidelines:**
- SWT training data: Use 5K-10K execution traces; beyond 10K, diminishing returns
- Context window budget: Reserve 1K-2K tokens for file content; summarize larger files
- Reward threshold: Use 0.5-0.7 for binary success classification
- Validation interval: Test agent trajectories on real Docker every 500 episodes to catch distribution shift

## Reference

See the full paper at: https://arxiv.org/abs/2602.03419

Key results: Qwen2.5-Coder improved from 6.2% to 52.0% resolve rate via Docker-free SFT, reaching 68.2% with test-time scaling. Public code released; training scaled from thousands to 16.6K task instances.
