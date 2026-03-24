---
name: agentic-llm-data-science
title: "DeepAnalyze: Agentic LLMs for Autonomous Data Science"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.16872"
keywords: [agentic LLM, autonomous analytics, data science, curriculum learning, end-to-end workflows]
description: "Train agentic LLMs through curriculum-based learning to autonomously execute full data science workflows from raw data to analysis reports, enabling 8B models to match proprietary systems."
---

# Technique: Curriculum-Based Agentic Data Science Training

Traditional data science workflows require chaining multiple tools (SQL, visualization, statistical testing), but most LLMs lack the emergent ability to coordinate these tools autonomously. DeepAnalyze addresses this by training models through a **curriculum** that progressively teaches data science competencies: from simple data QA through specialized analytics to open-ended research.

Rather than single-task supervised fine-tuning, the curriculum approach mirrors how human data scientists learn—starting with basic skills, building to intermediate analysis, and finally solving complex research questions. This enables even 8B models to achieve performance comparable to proprietary larger systems.

## Core Concept

Curriculum-based agentic training operates on three levels:
- **Level 1 (Data QA)**: Answer specific questions about provided data
- **Level 2 (Specialized Analytics)**: Execute targeted analyses (correlation, clustering, forecasting)
- **Level 3 (Open-Ended Research)**: Conduct comprehensive exploratory analysis without predefined constraints

This progression develops interconnected capabilities: basic QA teaches data handling, specialized tasks teach analysis patterns, open-ended tasks teach hypothesis formation and verification.

## Architecture Overview

- **Tool Executor**: SQL, Python (pandas, sklearn), visualization wrappers
- **Query Planner**: Decompose user request into tool sequences
- **Tool Invoker**: Map planned steps to actual tool calls with parameter binding
- **Result Interpreter**: Parse tool outputs and decide on next steps
- **Report Generator**: Synthesize findings into clear narrative
- **Curriculum Scheduler**: Progressively mix training data from three difficulty levels

## Implementation Steps

The key insight is structuring training data as progressively harder tasks that build on earlier capabilities. This example shows how to implement the three-level curriculum.

```python
from dataclasses import dataclass
from typing import List, Callable, Dict
import random

@dataclass
class DataScienceTask:
    """Represents a data science training task."""
    level: int  # 1=QA, 2=specialized, 3=open-ended
    question: str
    dataset: Dict  # Data dictionary
    required_tools: List[str]  # SQL, Python, viz, etc.
    expected_output: str
    difficulty_score: float  # 0.0 to 1.0


class CurriculumDataScienceTrainer:
    """
    Trains agentic LLM through curriculum of increasing difficulty.
    """

    def __init__(self, model, tool_executor):
        self.model = model
        self.executor = tool_executor
        self.qa_tasks = []      # Level 1: Basic QA
        self.specialized_tasks = []  # Level 2: Targeted analysis
        self.research_tasks = []     # Level 3: Open-ended

    def create_curriculum_batch(
        self,
        batch_size: int,
        epoch: int,
        total_epochs: int
    ):
        """
        Mix tasks from all three levels with curriculum weighting.
        Early epochs: more QA, mid: more specialized, late: more research.
        """
        # Curriculum schedule: shift distribution over training
        qa_weight = max(0.3, 0.6 - epoch * 0.2 / total_epochs)
        spec_weight = 0.5 * (epoch / total_epochs)
        research_weight = 1.0 - qa_weight - spec_weight

        # Sample batch maintaining curriculum balance
        batch = []
        num_qa = int(batch_size * qa_weight)
        num_spec = int(batch_size * spec_weight)
        num_research = batch_size - num_qa - num_spec

        batch.extend(random.sample(self.qa_tasks, min(num_qa, len(self.qa_tasks))))
        batch.extend(random.sample(self.specialized_tasks,
                                   min(num_spec, len(self.specialized_tasks))))
        batch.extend(random.sample(self.research_tasks,
                                   min(num_research, len(self.research_tasks))))

        return batch

    def level1_qa_task(self):
        """
        Create a Level 1 task: answer specific question about data.
        Example: "What is the average age of customers?"
        """
        task = DataScienceTask(
            level=1,
            question="What is the average customer age?",
            dataset=self.sample_dataset(),
            required_tools=["SQL"],
            expected_output="numeric_answer",
            difficulty_score=0.2
        )
        return task

    def level2_specialized_task(self):
        """
        Create a Level 2 task: targeted analysis.
        Example: "Perform correlation analysis between age and purchase amount"
        """
        task = DataScienceTask(
            level=2,
            question=("Analyze the relationship between customer age and "
                     "purchase amounts. Include correlation coefficient and visualization."),
            dataset=self.sample_dataset(),
            required_tools=["SQL", "Python", "Visualization"],
            expected_output="analysis_with_visualization",
            difficulty_score=0.5
        )
        return task

    def level3_research_task(self):
        """
        Create a Level 3 task: open-ended research.
        Example: "Conduct comprehensive customer segmentation analysis"
        """
        task = DataScienceTask(
            level=3,
            question=("Conduct a comprehensive analysis of our customer base. "
                     "Identify segments, behavioral patterns, and actionable insights."),
            dataset=self.sample_dataset(),
            required_tools=["SQL", "Python", "Visualization", "Statistical Testing"],
            expected_output="research_report",
            difficulty_score=0.8
        )
        return task

    def train_step(self, task: DataScienceTask):
        """
        Single training step: agent attempts task, compare to ground truth.
        """
        # Format task for model
        prompt = f"""
You are an autonomous data scientist. Execute this analysis:

Question: {task.question}

Data Schema: {task.dataset['schema']}

Available Tools: {', '.join(task.required_tools)}

Provide:
1. Analysis plan (step-by-step)
2. SQL queries and/or Python code
3. Results and visualization commands
4. Interpretation and insights

"""

        # Generate plan and code
        response = self.model.generate(prompt)

        # Execute tools
        results = self.executor.execute_plan(response, task.dataset)

        # Compare to expected output
        accuracy = self.evaluate_output(results, task.expected_output)

        # Compute loss
        loss = 1.0 - accuracy
        return loss


def train_agentic_data_science_model(
    model,
    executor,
    num_epochs: int = 10,
    batch_size: int = 32
):
    """
    Full training loop with curriculum scheduling.
    """
    trainer = CurriculumDataScienceTrainer(model, executor)

    # Create task pools
    for _ in range(500):
        trainer.qa_tasks.append(trainer.level1_qa_task())
        trainer.specialized_tasks.append(trainer.level2_specialized_task())
        trainer.research_tasks.append(trainer.level3_research_task())

    for epoch in range(num_epochs):
        # Get curriculum batch
        batch = trainer.create_curriculum_batch(
            batch_size,
            epoch,
            num_epochs
        )

        epoch_loss = 0.0
        for task in batch:
            loss = trainer.train_step(task)
            epoch_loss += loss

        print(f"Epoch {epoch + 1}: Loss={epoch_loss / len(batch):.4f}")
```

The curriculum is key: starting with simple QA teaches basic data handling, specialized tasks teach analysis patterns, and open-ended research forces integration of all skills. This mirrors human learning and produces more capable generalist agents.

## Practical Guidance

| Task Level | Training Focus | Typical Questions |
|-----------|----------|----------|
| Level 1 (QA) | Data fluency | "How many records?" "What's the max value?" |
| Level 2 (Specialized) | Analysis patterns | "Find correlations," "Cluster customers," "Forecast trends" |
| Level 3 (Research) | Integration | "Conduct comprehensive analysis," "Identify opportunities," "Explain patterns" |

**When to Use:**
- Building autonomous data science agents for enterprise analytics
- You have diverse data science tasks of varying complexity
- You want to develop end-to-end reasoning without orchestration frameworks
- Cost matters (8B models > larger proprietary systems)

**When NOT to Use:**
- Domain-specific data science (medical, legal) needing specialized tools
- Real-time streaming analytics (curriculum training is offline)
- Single-task optimization (curriculum overhead not justified)

**Common Pitfalls:**
- Imbalanced curriculum: too much early-stage task repetition → slow convergence
- Curriculum progression too fast → model skips necessary skills
- Task difficulty mismatch with level (Level 2 shouldn't be as hard as Level 3)
- Not validating curriculum order (test different orderings on dev set)
- Over-relying on tool fidelity (poor tool executors → poor training signal)

## Reference

[DeepAnalyze: Agentic LLMs for Autonomous Data Science](https://arxiv.org/abs/2510.16872)
