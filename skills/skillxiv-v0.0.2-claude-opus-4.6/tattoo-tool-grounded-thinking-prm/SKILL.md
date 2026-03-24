---
name: tattoo-tool-grounded-thinking-prm
title: "TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.06217"
keywords: [process reward model, tabular reasoning, tool grounding, test-time scaling, table verification]
description: "Build process reward models for tabular reasoning by grounding intermediate reasoning steps in executable tool operations. Train on 60K+ step-level annotations combining verification rationales with tool execution feedback. TaTToo achieves 30.9% improvement over policies using only final rewards, outperforming 72B reasoning models with just 8B parameters via better step-level supervision."
---

# TaTToo: Tool-Grounded Thinking PRM for Tabular Reasoning

## Core Concept

Process Reward Models (PRMs) enable test-time scaling via intermediate step-level feedback, but existing PRMs struggle with table-specific operations (sub-table retrieval, schema navigation). TaTToo grounds each reasoning step in executable tool operations—SQL queries, table transforms, schema checks—generating precise reward signals for what constitutes progress in data analysis tasks.

## Architecture Overview

- **Dual-Stage Training**: (1) Supervised fine-tuning on verification rationales; (2) reinforcement learning with tool-grounded rewards
- **Tool-Based Verification**: Each step grounds in actual tool execution (pandas operations, SQL queries), not just text descriptions
- **60K+ Step Annotations**: Scalable pipeline generates high-quality step-level training data from tool execution traces
- **Reward Shaping**: Reward models learn to evaluate reasoning quality based on tool-execution feasibility and correctness
- **Generalization**: Trained TaTToo transfers across benchmarks and test-time scaling strategies

## Implementation Steps

### 1. Grounding Reasoning Steps in Tool Operations

Each reasoning step must map to executable tool operations for verification.

```python
class ToolGroundedStep:
    def __init__(self, reasoning_text, tool_operation, expected_output):
        """
        Reasoning step grounded in tool execution.

        Args:
            reasoning_text: Natural language explanation ("Filter rows where age > 25")
            tool_operation: Executable code (SQL/pandas/etc)
            expected_output: Ground truth result
        """
        self.reasoning = reasoning_text
        self.operation = tool_operation
        self.expected = expected_output
        self.actual_output = None
        self.verification_status = None

    def verify(self):
        """Execute tool and verify step correctness."""
        try:
            self.actual_output = execute_tool(self.operation)
            self.verification_status = matches_expected(self.actual_output, self.expected)
        except Exception as e:
            self.verification_status = False
            self.actual_output = f"Error: {e}"

        return self.verification_status

class TabularReasoningTrace:
    def __init__(self, question, table, expected_answer):
        self.question = question
        self.table = table
        self.expected_answer = expected_answer
        self.steps = []  # List of ToolGroundedStep

    def add_step(self, reasoning, tool_op, expected_output):
        step = ToolGroundedStep(reasoning, tool_op, expected_output)
        self.steps.append(step)
        return step

    def verify_full_trace(self):
        """Verify all steps and final answer."""
        step_verifications = []
        for step in self.steps:
            is_correct = step.verify()
            step_verifications.append(is_correct)

        # Final answer check
        final_answer = self.steps[-1].actual_output if self.steps else None
        final_correct = matches_expected(final_answer, self.expected_answer)

        return {
            'step_correctness': step_verifications,
            'final_correct': final_correct,
            'num_correct_steps': sum(step_verifications),
            'total_steps': len(self.steps)
        }
```

### 2. Data Generation Pipeline: 60K+ Step Annotations

Automated pipeline generating training data from tool execution traces.

```python
def generate_step_level_annotations(questions, tables, max_traces=10000):
    """
    Generate 60K+ step-level training examples from tabular reasoning traces.
    """

    training_examples = []

    for question, table in zip(questions, tables):
        # Generate multiple reasoning traces per question
        for trace_idx in range(5):  # 5 traces per question
            trace = generate_reasoning_trace(question, table)

            # Verify each step
            verification_result = trace.verify_full_trace()

            # Extract step-level training examples
            for step_idx, step in enumerate(trace.steps):
                # Determine if step is on "good path" (leads to correct answer)
                is_on_correct_path = all(
                    trace.steps[i].verification_status
                    for i in range(step_idx + 1)
                ) and verification_result['final_correct']

                example = {
                    'question': question,
                    'table': table,
                    'step_number': step_idx,
                    'reasoning': step.reasoning,
                    'tool_operation': step.operation,
                    'step_correct': step.verification_status,
                    'on_correct_path': is_on_correct_path,
                    'intermediate_output': step.actual_output,
                    'label': 1.0 if is_on_correct_path else 0.0
                }

                training_examples.append(example)

    return training_examples

# Example: SQL-grounded tabular reasoning
trace_example = TabularReasoningTrace(
    question="How many employees have salary > 100k and joined after 2020?",
    table="employee_table",
    expected_answer=42
)

trace_example.add_step(
    reasoning="First, filter employees with salary > 100k",
    tool_op="SELECT * FROM employee_table WHERE salary > 100000",
    expected_output="[150000 rows]"
)

trace_example.add_step(
    reasoning="Filter those who joined after 2020",
    tool_op="SELECT * FROM filtered_above WHERE join_date > '2020-12-31'",
    expected_output="[42 rows]"
)

trace_example.add_step(
    reasoning="Count the results",
    tool_op="SELECT COUNT(*) FROM filtered_2020_above",
    expected_output=42
)
```

### 3. Dual-Stage Training: SFT + RL

Stage 1: Supervised fine-tuning on verification rationales. Stage 2: RL with tool-grounded rewards.

```python
class TaTTooTrainer:
    def __init__(self, base_model='qwen-8b'):
        self.model = base_model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

    def stage1_supervised_finetuning(self, training_examples, num_epochs=3):
        """
        SFT on step-level annotations: learn to evaluate step quality.
        """
        for epoch in range(num_epochs):
            total_loss = 0
            for example in training_examples:
                # Input: question + table + step reasoning
                input_text = f"""Question: {example['question']}
Table: {example['table']}
Step {example['step_number']}: {example['reasoning']}

Is this step correct? (Binary classification)"""

                # Target: label (0 or 1)
                target = example['label']

                # Forward pass
                logits = self.model.predict(input_text)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Stage 1 Epoch {epoch}: Loss={total_loss/len(training_examples):.4f}")

    def stage2_reinforcement_learning(self, eval_traces, num_steps=5000):
        """
        RL: reward model learns from tool execution feedback.
        Policy (reasoning model) learns to maximize expected rewards.
        """

        policy = self.model  # PRM acts as reward signal for policy

        for step in range(num_steps):
            # Sample trace
            trace = random.choice(eval_traces)

            # Generate reasoning step
            step_reasoning = trace.steps[random.randint(0, len(trace.steps) - 1)]

            # Execute tool to get verification
            step_reasoning.verify()

            # Compute reward
            tool_execution_successful = step_reasoning.verification_status
            on_correct_path = all(s.verify() for s in trace.steps)

            reward = 1.0 if (tool_execution_successful and on_correct_path) else 0.0

            # Policy gradient update
            log_prob = self.model.log_probability(step_reasoning.reasoning)
            loss = -log_prob * reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"Stage 2 Step {step+1}: Avg reward tracking...")

    def train_tattoo(self, train_examples, eval_traces):
        """Full training pipeline."""
        print("Stage 1: Supervised fine-tuning...")
        self.stage1_supervised_finetuning(train_examples)

        print("Stage 2: Reinforcement learning...")
        self.stage2_reinforcement_learning(eval_traces)

        return self.model
```

### 4. Evaluation on Tabular Benchmarks

Test-time scaling with TaTToo-trained reward models.

```python
def evaluate_with_prm(policy, prm_reward_model, test_examples, num_beams=5):
    """
    Use PRM for test-time scaling via beam search.
    At each step, select highest-reward continuations.
    """

    correct = 0
    for example in test_examples:
        question = example['question']
        table = example['table']

        # Beam search with PRM guidance
        beams = [(question, [])]  # (current_state, step_history)

        for step_num in range(20):  # Max 20 steps
            scored_beams = []

            for state, history in beams:
                # Generate next reasoning step
                next_step = policy.generate_step(state, table, history)

                # Score with PRM
                reward = prm_reward_model.score(
                    question=question,
                    table=table,
                    step_number=step_num,
                    reasoning=next_step
                )

                scored_beams.append((reward, state + next_step, history + [next_step]))

            # Keep top-k beams
            scored_beams.sort(reverse=True)
            beams = [(s, h) for _, s, h in scored_beams[:num_beams]]

        # Evaluate final answer
        final_answer = extract_answer_from_reasoning(beams[0][0])
        if final_answer == example['expected_answer']:
            correct += 1

    accuracy = correct / len(test_examples)
    return accuracy

# Benchmark results
results = {
    'without_prm': {'accuracy': 0.45},  # Policy alone
    'with_tattoo_prm': {'accuracy': 0.55},  # +30.9% improvement
    'baseline_72b': {'accuracy': 0.52},  # Larger model without PRM
}
```

## Practical Guidance

**Tool Grounding**: Every reasoning step must map to an executable operation. Abstract reasoning without tool execution doesn't generate reliable reward signals.

**Training Data Scale**: 60K+ examples provides good coverage across table schemas and operation types. Collect diverse traces (SQL, pandas, Excel operations) to ensure generalization.

**Reward Shaping**: Combine step-level correctness (tool execution) with trajectory-level signal (on-correct-path) to balance immediate feedback with long-horizon alignment.

**Generalization**: Train on 5 benchmarks, evaluate on held-out tasks. TaTToo generalizes better than task-specific PRMs due to tool-based grounding.

## When to Use / When NOT to Use

**Use When**:
- Training agents on data analysis tasks (SQL, pandas, spreadsheets)
- Verifiable intermediate steps are available (query results, transformed tables)
- Test-time scaling via beam search or planning is feasible
- Smaller models need to match larger model performance

**NOT For**:
- Tasks without executable verification (open-ended reasoning)
- Domains where intermediate steps lack ground truth
- Real-time low-latency inference (beam search is expensive)

## Reference

This skill synthesizes findings from "TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning" (arXiv:2510.06217). Tool grounding enables reliable step-level reward signals for process-based RL in data analysis.
