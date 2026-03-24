---
name: mact-document-understanding
title: MACT - Multi-Agent for Visual Document Understanding with Test-Time Scaling
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03404
keywords: [document-understanding, multi-agent, test-time-scaling, vision-language]
description: "Decomposes document processing into specialized agents (planning, execution, judgment, answer) with agent-wise adaptive test-time scaling. Achieves 9.9-11.5% performance gain with smaller models while maintaining reasoning."
---

# MACT: Multi-Agent for Visual Document Understanding with Test-Time Scaling

## Core Concept

Vision-Language Models struggle with document understanding because documents require procedural reasoning, cognitive complexity, and factual accuracy. Rather than scaling model size, MACT decomposes document processing into specialized agents that collaborate through adaptive test-time scaling. A planning agent breaks tasks into steps, execution agents process content, a judgment agent verifies results, and an answer agent synthesizes responses. Crucially, each agent receives adaptive compute based on task difficulty.

## Architecture Overview

- **Planning Agent**: Decomposes document task into reasoning steps
- **Execution Agents**: Process specific document components (text extraction, layout, tables)
- **Judgment Agent**: Verifies intermediate results and flags issues
- **Answer Agent**: Synthesizes final response with self-correction
- **Adaptive Test-Time Scaling**: Allocate more computation to difficult subtasks

## Implementation Steps

### Step 1: Build Planning Agent

Create agent that decomposes document tasks strategically.

```python
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DocumentTask:
    """Represents a document understanding task."""
    document_image: str  # Path or image tensor
    query: str
    task_type: str  # extraction, qa, summarization, layout
    complexity: float = None  # 0-1

class PlanningAgent:
    """
    Decomposes complex document task into procedural steps.
    """

    def __init__(self, planning_model):
        self.model = planning_model

    def decompose_task(self, task: DocumentTask) -> List[Dict]:
        """
        Break document task into procedural steps.

        Args:
            task: Document understanding task

        Returns:
            List of procedural steps with context
        """
        prompt = f"""
        Document Query: {task.query}
        Task Type: {task.task_type}

        Break this document task into 3-8 concrete procedural steps.
        For each step:
        1. What specifically needs to be done?
        2. What information from the document is needed?
        3. What are success criteria?

        Return as JSON list of steps.
        """

        response = self.model.generate(prompt)
        steps = self._parse_steps(response)

        # Estimate complexity of each step
        for step in steps:
            step["complexity"] = self._estimate_step_complexity(step)

        return steps

    def _estimate_step_complexity(self, step: Dict) -> float:
        """
        Estimate computational complexity of step.

        Args:
            step: Step description

        Returns:
            Complexity 0-1
        """
        # Heuristic: complexity based on step description
        complexity_keywords = {
            "complex": 0.8,
            "detailed": 0.6,
            "comprehensive": 0.7,
            "compare": 0.7,
            "analyze": 0.6,
            "extract": 0.3,
            "identify": 0.4
        }

        step_text = step.get("description", "").lower()

        complexity = 0.5  # Default medium complexity

        for keyword, score in complexity_keywords.items():
            if keyword in step_text:
                complexity = max(complexity, score)

        return complexity

    def _parse_steps(self, response: str) -> List[Dict]:
        """Parse steps from model response."""
        import json
        import re

        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
```

### Step 2: Build Specialized Execution Agents

Create agents optimized for specific document processing tasks.

```python
class ExecutionAgent:
    """
    Base execution agent for document processing subtasks.
    """

    def __init__(self, model, agent_type: str):
        self.model = model
        self.agent_type = agent_type
        self.confidence_scores = []

    def execute(self, task_step: Dict, document, context: Dict) -> Dict:
        """
        Execute single procedural step on document.

        Args:
            task_step: Step to execute
            document: Document image/content
            context: Prior results/context

        Returns:
            Execution result with output and confidence
        """
        prompt = self._build_prompt(task_step, context)

        output = self.model.generate(prompt)
        confidence = self._estimate_confidence(output, task_step)

        self.confidence_scores.append(confidence)

        return {
            "agent_type": self.agent_type,
            "step": task_step,
            "output": output,
            "confidence": confidence,
            "requires_verification": confidence < 0.7
        }

    def _build_prompt(self, step: Dict, context: Dict) -> str:
        """Build prompt for execution step."""
        prompt = f"""
        Task: {step.get('description', '')}

        Prior context: {context}

        Complete this task on the document.
        Be specific and factual.
        """

        return prompt

    def _estimate_confidence(self, output: str, step: Dict) -> float:
        """
        Estimate confidence in output.

        Returns:
            Confidence 0-1
        """
        # Simple heuristics
        confidence = 0.5

        # Check for specificity
        if len(output.split()) > 5:
            confidence += 0.2

        # Check for uncertainty language
        uncertainty_words = ["maybe", "possibly", "unclear", "uncertain"]
        if not any(word in output.lower() for word in uncertainty_words):
            confidence += 0.3

        return min(confidence, 1.0)


class TextExtractionAgent(ExecutionAgent):
    """Specialized for text extraction from documents."""

    def __init__(self, model):
        super().__init__(model, "text_extraction")

    def _build_prompt(self, step: Dict, context: Dict) -> str:
        """Build text extraction prompt."""
        return f"""
        Extract the following text from the document:
        {step.get('extraction_target', '')}

        Return only the extracted text, no explanations.
        """


class TableAnalysisAgent(ExecutionAgent):
    """Specialized for table and structured data analysis."""

    def __init__(self, model):
        super().__init__(model, "table_analysis")

    def _build_prompt(self, step: Dict, context: Dict) -> str:
        """Build table analysis prompt."""
        return f"""
        Analyze the table in the document:
        {step.get('table_task', '')}

        Return analysis as structured JSON.
        """


class LayoutAnalysisAgent(ExecutionAgent):
    """Specialized for document layout and structure."""

    def __init__(self, model):
        super().__init__(model, "layout_analysis")

    def _build_prompt(self, step: Dict, context: Dict) -> str:
        """Build layout analysis prompt."""
        return f"""
        Analyze the layout and structure of the document:
        {step.get('layout_task', '')}

        Describe the organization and key sections.
        """
```

### Step 3: Build Judgment Agent

Create agent that verifies intermediate results.

```python
class JudgmentAgent:
    """
    Verifies intermediate results and flags potential issues.
    """

    def __init__(self, model):
        self.model = model

    def verify_execution_result(
        self,
        step: Dict,
        execution_output: str,
        document_context: str,
        confidence_score: float
    ) -> Dict:
        """
        Verify quality of execution result.

        Args:
            step: Original task step
            execution_output: Output from execution agent
            document_context: Document content for reference
            confidence_score: Execution agent's confidence

        Returns:
            Verification result with accept/reject and feedback
        """
        verification_prompt = f"""
        Task: {step.get('description', '')}
        Execution Output: {execution_output}
        Agent Confidence: {confidence_score:.2f}

        Verify this output:
        1. Does it complete the task?
        2. Is it factually correct based on the document?
        3. Is it complete and unambiguous?

        Respond with JSON:
        {{
            "valid": true/false,
            "confidence": 0.0-1.0,
            "issues": ["issue1", "issue2"],
            "feedback": "explanation"
        }}
        """

        response = self.model.generate(verification_prompt)

        result = self._parse_verification(response)

        # Accept if high confidence and agent was confident
        result["accepted"] = (
            result.get("valid", False) and
            confidence_score > 0.6 and
            result.get("confidence", 0) > 0.7
        )

        result["requires_refinement"] = not result["accepted"]

        return result

    def _parse_verification(self, response: str) -> Dict:
        """Parse verification result."""
        import json
        import re

        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {"valid": False, "confidence": 0.0, "issues": [], "feedback": ""}
```

### Step 4: Implement Adaptive Test-Time Scaling

Create system that allocates compute based on task difficulty.

```python
class AdaptiveTestTimeScaling:
    """
    Adaptively allocate computational resources to difficult subtasks.
    """

    def __init__(self, execution_agents: Dict, judgment_agent: JudgmentAgent):
        self.execution_agents = execution_agents
        self.judgment = judgment_agent

    def execute_with_adaptive_scaling(
        self,
        task_steps: List[Dict],
        document,
        base_compute: int = 1,
        max_compute: int = 5
    ) -> List[Dict]:
        """
        Execute steps with adaptive compute allocation.

        Args:
            task_steps: Procedural steps to execute
            document: Document to process
            base_compute: Base compute units per step
            max_compute: Maximum compute units available

        Returns:
            List of execution results
        """
        results = []
        remaining_compute = max_compute
        context = {}

        for step_idx, step in enumerate(task_steps):
            # Estimate compute needed for this step
            complexity = step.get("complexity", 0.5)

            # Allocate compute: high complexity gets more compute
            step_compute = int(base_compute + (complexity * (max_compute - base_compute)))
            step_compute = min(step_compute, remaining_compute)

            # Execute with allocated compute
            result = self._execute_step_with_compute(
                step,
                document,
                context,
                compute_budget=step_compute
            )

            # Verify result
            verification = self.judgment.verify_execution_result(
                step,
                result["output"],
                str(document)[:500],
                result["confidence"]
            )

            # If rejected and compute remaining, try again
            if not verification["accepted"] and remaining_compute > step_compute:
                extra_compute = min(step_compute, remaining_compute - step_compute)

                result = self._execute_step_with_compute(
                    step,
                    document,
                    context,
                    compute_budget=step_compute + extra_compute
                )

            result["verification"] = verification

            # Update remaining compute
            remaining_compute -= step_compute

            # Update context for next step
            context[f"step_{step_idx}"] = result["output"]

            results.append(result)

        return results

    def _execute_step_with_compute(
        self,
        step: Dict,
        document,
        context: Dict,
        compute_budget: int
    ) -> Dict:
        """
        Execute step with given compute budget.

        Higher compute budget = multiple attempts, temperature sampling, etc.

        Args:
            step: Task step
            document: Document
            context: Prior context
            compute_budget: Compute units available

        Returns:
            Execution result
        """
        agent_type = step.get("agent_type", "general")
        agent = self.execution_agents.get(agent_type, self.execution_agents.get("general"))

        # With more compute, try multiple approaches
        outputs = []
        confidences = []

        num_attempts = min(compute_budget, 3)

        for attempt in range(num_attempts):
            # Vary temperature with attempts
            temperature = 0.3 + (attempt * 0.2)

            # Set temperature for model
            agent.model.temperature = temperature

            result = agent.execute(step, document, context)

            outputs.append(result["output"])
            confidences.append(result["confidence"])

        # Select best output (highest confidence)
        best_idx = confidences.index(max(confidences))

        return {
            "output": outputs[best_idx],
            "confidence": confidences[best_idx],
            "num_attempts": num_attempts,
            "all_outputs": outputs
        }
```

### Step 5: Build Answer Synthesis Agent

Create agent that synthesizes final answer with self-correction.

```python
class AnswerAgent:
    """
    Synthesize final answer from execution results.
    """

    def __init__(self, model):
        self.model = model

    def synthesize_answer(
        self,
        original_query: str,
        execution_results: List[Dict],
        document_context: str
    ) -> str:
        """
        Synthesize final answer from all procedural results.

        Args:
            original_query: Original document query
            execution_results: Results from all execution steps
            document_context: Document content

        Returns:
            Final synthesized answer
        """
        # Build context from execution results
        execution_summary = self._summarize_execution_results(execution_results)

        synthesis_prompt = f"""
        Original Query: {original_query}

        Execution Results:
        {execution_summary}

        Synthesize a comprehensive answer to the original query.
        Use only information from the execution results.
        If results conflict, note the conflict and provide most reliable answer.

        Answer:
        """

        answer = self.model.generate(synthesis_prompt)

        # Self-check: verify answer addresses query
        is_valid = self._verify_answer_validity(original_query, answer)

        if not is_valid:
            # Refine answer
            answer = self._refine_answer(original_query, answer, execution_summary)

        return answer

    def _summarize_execution_results(self, results: List[Dict]) -> str:
        """Summarize execution results for synthesis."""
        summary = ""

        for i, result in enumerate(results):
            summary += f"Step {i + 1}: {result['output']}\n"

            if result.get("verification"):
                if not result["verification"]["accepted"]:
                    summary += f"  (Note: Verification flagged issues)\n"

        return summary

    def _verify_answer_validity(self, query: str, answer: str) -> bool:
        """Check if answer addresses query."""
        # Simple check: answer length, query keywords in answer
        return len(answer) > 50

    def _refine_answer(self, query: str, initial_answer: str, context: str) -> str:
        """Refine answer based on verification."""
        refinement_prompt = f"""
        Query: {query}
        Initial Answer: {initial_answer}
        Supporting Information: {context}

        Improve this answer to better address the query.
        """

        refined = self.model.generate(refinement_prompt)

        return refined
```

## Practical Guidance

### When to Use MACT

- **Complex document processing**: Multi-step reasoning required
- **Mixed document types**: Tables, text, layout analysis needed
- **High accuracy requirements**: Verification step ensures quality
- **Variable task complexity**: Adaptive scaling handles mixed difficulties
- **Medium-sized models**: Reach large-model performance without scaling

### When NOT to Use MACT

- **Simple document tasks**: Single extraction task doesn't need decomposition
- **Real-time constraints**: Multi-agent overhead adds latency
- **Streaming documents**: Can't decompose until full document available
- **Very large documents**: Decomposition context may be insufficient

### Hyperparameter Recommendations

- **Base compute units**: 1-2 per step
- **Max compute budget**: 3-5 total units per task
- **Complexity threshold for extra compute**: 0.6+
- **Verification confidence threshold**: 0.7+
- **Number of execution agents**: 3-5 specialized + 1 general

### Key Insights

The critical insight is recognizing that document understanding isn't monolithic but requires specialized reasoning for different document components. By decomposing tasks, using specialized agents, and adaptively allocating compute to difficult subtasks, MACT achieves strong performance with smaller models. The judgment agent prevents propagation of errors through the pipeline.

## Reference

**Visual Document Understanding via Multi-Agent with Test-Time Scaling** (arXiv:2508.03404)

Decomposes document processing into planning, execution, judgment, and answer agents. Introduces agent-wise adaptive test-time scaling that allocates computational resources proportionally to task difficulty. Achieves 9.9-11.5% performance improvement with smaller base models.
