---
name: recon-act-self-evolving-browser-agents
title: "Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance, Tool Generation, and Task Execution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.21072"
keywords: [multi-agent systems, web automation, browser agents, tool generation, self-evolution, reconnaissance, task execution, autonomous web browsing]
description: "Build self-evolving multi-agent browser systems that combine web reconnaissance with dynamic tool generation and execution. Enables autonomous agents to analyze failed trajectories, generate specialized tools on-the-fly, and adapt to novel web environments without pre-defined action sets, achieving 36.48% success on VisualWebArena."
---

# Recon-Act Self-Evolving Browser Agents

## Outcome

Deploy a multi-agent browser automation system that evolves through iterative reconnaissance, dynamically generates task-specific tools, and executes complex web interactions with minimal human intervention while maintaining accuracy across unknown websites.

## Problem Context

Autonomous web browsers currently suffer from two critical limitations: disordered action sequencing and excessive trial-and-error during execution. Traditional approaches rely on exhaustive pre-defined toolsets or require extensive labeled training data for each web domain. When agents encounter novel interfaces or fail on complex tasks, they lack mechanisms to learn from failures and adapt. This forces teams to either manually expand tool libraries or collect more training data—both expensive and time-consuming approaches.

The Recon-Act framework addresses this by implementing a closed-loop, data-to-tool-to-action evolutionary pipeline. Rather than treating failures as dead ends, the system treats them as learning opportunities, systematically analyzing what went wrong and synthesizing domain-specific tools to prevent recurrence.

## Core Concept

Recon-Act operates as a dual-team hierarchical system:

**Reconnaissance Team**: Analyzes failed execution trajectories to identify root causes, explores the web environment for additional observational data, and synthesizes generalized tools (either natural language hints or rule-based code). Comprises an Analyst (reasoning layer) and Coder (implementation layer), both receiving human oversight.

**Action Team**: Executes tasks using available tools. Comprises a Master (interprets natural language queries and decomposes intent), Tool Manager (orchestrates tool registration and selection), and Execution Agent (generates fallback actions when existing tools prove insufficient).

The innovation lies in the feedback loop: when the Action Team fails, the Reconnaissance Team automatically activates to understand why and generate fixes that persist in the tool archive for future use. This self-evolution requires no retraining or data collection—only iterative synthesis of specialized tools.

## Architecture Overview

- **Reconnaissance Loop**: Comparative analysis of erroneous vs. successful trajectories. Agents conduct exploratory interactions with the webpage environment, gathering additional observational data about DOM structure, element interactivity, and dynamically loaded content.

- **Tool Generation**: Dynamic synthesis of generalized tools tailored to specific task contexts. Tools are expressed either as natural language hints or as rule-based code blocks that encapsulate discovered patterns or workarounds.

- **Tool Archive**: Persistent registry of all synthesized tools, searchable by domain and task type. Enables rapid selection and reuse across similar tasks without regeneration overhead.

- **Action Execution Pipeline**: Multi-stage processing of tool-assisted actions: intent parsing from natural language queries, tool selection from archive via semantic matching, parameterization with task-specific variables, and execution with state validation across multi-turn interactions.

- **Multi-Agent Communication**: Hierarchical message passing between Reconnaissance and Action teams using structured JSON schemas. Includes failure reports, trajectory summaries, generated tool specifications, and execution results.

- **State Management**: Maintains execution context across long-horizon task sequences, tracking webpage state, DOM snapshots, interaction history, and cumulative observations for root-cause analysis.

## Implementation Details

### Step 1: Initialize the Dual-Team Architecture

Set up the hierarchical agent structure with clear separation of concerns between reconnaissance and action capabilities. The Reconnaissance Team operates in analysis mode, activated only when Action Team failures occur. The Action Team runs continuously on incoming tasks.

```python
class ReconAct:
    def __init__(self, model="gpt-4-turbo", human_oversight=True):
        # Reconnaissance team components
        self.analyst = Agent(
            role="trajectory_analyzer",
            model=model,
            system_prompt="""You are a trajectory analyzer. Examine failed browser
            interactions and identify root causes. Look for pattern breaks,
            missing preconditions, and interface-specific quirks. Generate
            hypotheses about why the action failed."""
        )
        self.coder = Agent(
            role="tool_synthesizer",
            model=model,
            system_prompt="""You are a tool implementation expert. Given failure
            analysis, implement generalized tools as either natural language hints
            or rule-based code. Ensure tools are domain-specific but generalizable
            to similar task patterns."""
        )

        # Action team components
        self.master = Agent(
            role="query_decomposer",
            model=model,
            system_prompt="""You are a task decomposer. Parse high-level user
            queries and decompose them into sub-intent sequences. Maintain
            sequential constraints and identify dependencies between actions."""
        )
        self.tool_manager = ToolManager(oversight=human_oversight)
        self.executor = Agent(
            role="action_generator",
            model=model,
            system_prompt="""You are an action generator. Execute tool invocations
            and generate fallback actions when tools are insufficient."""
        )

        # Persistent storage
        self.tool_archive = ToolArchive()
        self.trajectory_log = []
        self.state_manager = StateManager()
```

### Step 2: Implement the Reconnaissance Loop

Create the failure analysis and tool generation pipeline. When tasks fail, automatically trigger reconnaissance to extract insights and synthesize fixes.

```python
def run_reconnaissance_cycle(self, failed_trajectory, task_query):
    """
    Analyze a failed trajectory and generate tools to prevent recurrence.

    Args:
        failed_trajectory: List of (action, observation) tuples
        task_query: Original user query that triggered the failure

    Returns:
        List of generated Tool objects
    """
    # Step 1: Extract failure context
    failure_point = failed_trajectory[-1]
    action_attempt, error_state = failure_point

    # Retrieve successful trajectories for the same or similar tasks
    similar_successes = self.trajectory_log.find_similar_successes(
        task_type=self._infer_task_type(task_query),
        domain=self._extract_domain(error_state)
    )

    # Step 2: Comparative analysis via Analyst
    analysis_prompt = f"""Compare these trajectories:

    FAILED trajectory (steps 1-{len(failed_trajectory)}):
    {self._format_trajectory(failed_trajectory)}

    SUCCESSFUL trajectory (reference):
    {self._format_trajectory(similar_successes[0] if similar_successes else [])}

    Original query: {task_query}
    Error state: {error_state}

    Identify:
    1. Where did the failed trajectory diverge from successful patterns?
    2. What preconditions were missing?
    3. What environmental quirks (DOM structure, timing, etc.) caused failure?
    4. What specific action sequence should prevent this failure?"""

    analysis_result = self.analyst.reason(analysis_prompt)

    # Step 3: Tool synthesis via Coder
    synthesis_prompt = f"""Based on this failure analysis:
    {analysis_result}

    Generate 1-3 actionable tools:
    - Option A: Natural language hint (concise rule for agents to follow)
    - Option B: Rule-based code (pseudocode/Python for structured domains)

    Tools must be:
    - Generalizable beyond this single failure
    - Domain-specific for the detected website
    - Executable by future agents without retraining"""

    generated_tools = self.coder.synthesize_tools(synthesis_prompt)

    # Step 4: Register tools with human oversight
    for tool in generated_tools:
        review = self.tool_manager.request_human_review(tool)
        if review.approved:
            self.tool_archive.register(tool)

    return generated_tools
```

### Step 3: Implement the Tool Archive and Retrieval System

Create a searchable registry of tools with semantic indexing for rapid retrieval based on task context.

```python
class ToolArchive:
    def __init__(self):
        self.tools = {}  # domain -> list of Tool objects
        self.embeddings = {}  # tool_id -> embedding vector
        self.metadata = {}  # tool_id -> metadata

    def register(self, tool: Tool):
        """Register a new tool in the archive."""
        if tool.domain not in self.tools:
            self.tools[tool.domain] = []

        tool.id = self._generate_id(tool)
        self.tools[tool.domain].append(tool)

        # Semantic indexing for retrieval
        self.embeddings[tool.id] = self._embed_tool_description(tool)
        self.metadata[tool.id] = {
            "created_at": datetime.now(),
            "domain": tool.domain,
            "task_type": tool.task_type,
            "success_rate": 0.0,
            "use_count": 0
        }

    def retrieve(self, task_query: str, domain: str, top_k=5) -> List[Tool]:
        """
        Retrieve most relevant tools for a task.

        Scoring: combine domain match + semantic similarity + success rate
        """
        if domain not in self.tools:
            return []

        candidate_tools = self.tools[domain]
        query_embedding = self._embed_tool_description(task_query)

        scored_tools = []
        for tool in candidate_tools:
            # Semantic similarity (cosine)
            sim_score = cosine_similarity(
                query_embedding,
                self.embeddings[tool.id]
            )

            # Success rate (empirical)
            success_score = self.metadata[tool.id]["success_rate"]

            # Combined score: 60% semantic + 40% empirical
            combined = (0.6 * sim_score) + (0.4 * success_score)
            scored_tools.append((tool, combined))

        # Sort by score and return top-k
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in scored_tools[:top_k]]

    def update_success_rate(self, tool_id: str, success: bool):
        """Update tool success metrics after execution."""
        if tool_id in self.metadata:
            meta = self.metadata[tool_id]
            meta["use_count"] += 1

            # Exponential moving average
            alpha = 0.3
            old_rate = meta["success_rate"]
            new_rate = (alpha * float(success)) + ((1 - alpha) * old_rate)
            meta["success_rate"] = new_rate
```

### Step 4: Implement the Action Execution Pipeline

Build the multi-agent execution pipeline that processes tasks through intent decomposition, tool selection, and execution.

```python
def execute_task(self, user_query: str, context: ExecutionContext) -> ExecutionResult:
    """
    Execute a high-level task using the Action Team.

    Args:
        user_query: Natural language task description
        context: Browser state, previous interactions, domain info

    Returns:
        ExecutionResult with success status and observational data
    """
    # Phase 1: Intent decomposition via Master
    decomposition = self.master.decompose_intent(
        query=user_query,
        domain=context.current_domain,
        page_state=context.page_state
    )

    # decomposition format:
    # {
    #   "sub_intents": ["navigate to X", "fill field Y", "click Z"],
    #   "dependencies": [[0, 1], [1, 2]],  # sequential
    #   "preconditions": ["page must be loaded", "user must be logged in"]
    # }

    # Phase 2: Execute sub-intents sequentially
    execution_trace = []
    for step_idx, sub_intent in enumerate(decomposition["sub_intents"]):
        # Retrieve candidate tools
        tools = self.tool_archive.retrieve(
            task_query=sub_intent,
            domain=context.current_domain,
            top_k=3
        )

        # Attempt execution with best tool
        for tool in tools:
            try:
                action = tool.instantiate(
                    intent=sub_intent,
                    context=context
                )

                # Execute action in browser environment
                observation = context.execute_action(action)

                # Log successful execution
                execution_trace.append({
                    "step": step_idx,
                    "intent": sub_intent,
                    "tool_used": tool.id,
                    "action": action,
                    "observation": observation,
                    "success": True
                })

                # Update tool success metrics
                self.tool_archive.update_success_rate(tool.id, success=True)

                # Update state for next step
                context.update_state(observation)
                break  # Move to next sub-intent

            except ToolExecutionError as e:
                # Tool failed; try next candidate
                self.tool_archive.update_success_rate(tool.id, success=False)
                continue

        else:
            # No tool succeeded; attempt fallback action generation
            fallback_action = self.executor.generate_fallback_action(
                intent=sub_intent,
                context=context,
                failed_tools=[t.id for t in tools]
            )

            try:
                observation = context.execute_action(fallback_action)
                execution_trace.append({
                    "step": step_idx,
                    "intent": sub_intent,
                    "tool_used": "fallback_generated",
                    "action": fallback_action,
                    "observation": observation,
                    "success": True
                })
                context.update_state(observation)
            except Exception as e:
                # Fallback also failed; trigger reconnaissance
                execution_trace.append({
                    "step": step_idx,
                    "intent": sub_intent,
                    "tool_used": None,
                    "action": None,
                    "observation": None,
                    "success": False,
                    "error": str(e)
                })

                # Store failed trajectory for analysis
                self.trajectory_log.append(execution_trace)

                # Trigger reconnaissance cycle (async recommended)
                self.run_reconnaissance_cycle(
                    failed_trajectory=execution_trace,
                    task_query=user_query
                )

                # Return partial result to user
                return ExecutionResult(
                    success=False,
                    partial_trace=execution_trace,
                    failed_at_step=step_idx
                )

    # Phase 3: Return successful result
    self.trajectory_log.append(execution_trace)
    return ExecutionResult(
        success=True,
        full_trace=execution_trace,
        final_observation=context.get_page_state()
    )
```

### Step 5: Implement Multi-Agent Communication Protocol

Establish structured communication between teams using JSON schemas for tool specifications and failure reports.

```python
class Tool:
    """Generalized tool representation with flexible expression."""

    def __init__(
        self,
        id: str,
        name: str,
        domain: str,
        task_type: str,
        expression_type: str,  # "hint" or "rule_code"
        content: str,
        context_requirements: dict
    ):
        self.id = id
        self.name = name
        self.domain = domain
        self.task_type = task_type
        self.expression_type = expression_type
        self.content = content
        self.context_requirements = context_requirements

    def to_json(self) -> dict:
        """Serialize for inter-agent communication."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "task_type": self.task_type,
            "expression_type": self.expression_type,
            "content": self.content,
            "context_requirements": self.context_requirements
        }

    def instantiate(self, intent: str, context: ExecutionContext):
        """
        Convert tool specification into executable action.

        For "hint" type: pass to executor as instruction
        For "rule_code" type: parameterize and execute directly
        """
        if self.expression_type == "hint":
            return HintBasedAction(
                hint=self.content,
                intent=intent,
                context=context
            )
        elif self.expression_type == "rule_code":
            # Parse rule_code and bind context variables
            return RuleBasedAction(
                rule_code=self.content,
                context=context
            )


class FailureReport:
    """Structured failure communication from Action Team to Reconnaissance Team."""

    def __init__(
        self,
        failed_trajectory: List[dict],
        original_query: str,
        error_summary: str,
        context_snapshot: dict
    ):
        self.timestamp = datetime.now()
        self.failed_trajectory = failed_trajectory
        self.original_query = original_query
        self.error_summary = error_summary
        self.context_snapshot = context_snapshot

    def to_json(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "trajectory_length": len(self.failed_trajectory),
            "original_query": self.original_query,
            "error_summary": self.error_summary,
            "failed_step_index": len(self.failed_trajectory) - 1,
            "context_snapshot": self.context_snapshot
        }
```

## Practical Guidance

### Hyperparameters and Configuration

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| tool_retrieval_top_k | 3 | 1-10 | Number of candidate tools to attempt per sub-intent |
| similarity_weight | 0.6 | 0.5-0.8 | Weight of semantic similarity vs. empirical success in tool scoring |
| success_rate_alpha | 0.3 | 0.1-0.5 | Exponential moving average for tool success tracking (lower = slower adaptation) |
| fallback_generation_timeout | 30s | 10-60s | Max time for executor to synthesize fallback actions |
| reconnaissance_activation_delay | 0 | 0-300s | Delay reconnaissance cycle for batch failure analysis |
| tool_context_window | 2048 | 512-8192 | Token budget for tool description in semantic indexing |
| max_trajectory_length | 50 | 20-200 | Maximum steps before forced failure and reconnaissance |

### When to Use Recon-Act

**Ideal scenarios:**
- Multi-step web automation tasks with variable domain layouts (e-commerce, classifieds, content sites)
- Environments where you cannot pre-define all possible action sequences
- Long-horizon tasks (10+ interaction steps) where failure analysis provides compounding value
- Domains that evolve over time (sites update layouts, UI patterns change)
- Teams with limited training data or labeled examples for specific sites
- Scenarios where human oversight is available for tool review (small bottleneck)

**Key advantages:**
- Achieves 36%+ success on diverse, unseen web domains
- Generates tools incrementally from failures rather than upfront
- Minimal training overhead (11 tools sufficed for 3-domain benchmark)
- Closed-loop evolution: failures automatically become learning signals
- Generalizes across domain variations without retraining

### When NOT to Use Recon-Act

- **Real-time constraints**: Reconnaissance cycles add latency (minutes to hours for analysis). Not suitable for sub-second task execution.
- **Fully-deterministic workflows**: If tasks have zero variation and can be hard-coded, simpler state machines suffice.
- **Strictly adversarial environments**: If the website actively obfuscates or blocks tool generation, reconnaissance synthesis becomes futile.
- **No human oversight available**: The Analyst and Tool Manager require human-in-the-loop review. Fully autonomous tool generation without oversight risks executing harmful actions.
- **Offline-only execution**: Reconnaissance requires live website interaction and state observation. Purely offline workflows (e.g., batch document processing) don't benefit.
- **Cost-sensitive applications**: Language model calls for agent reasoning and tool synthesis accumulate costs. Count model API expenses carefully.
- **Highly constrained action spaces**: If your domain has a small, stable set of actions (e.g., ATM operations), traditional automation tools are more efficient.

### Common Pitfalls

1. **Insufficient failure diversity**: Tool generation relies on comparative analysis. If failures are too similar, generated tools over-specialize. Collect failures from multiple domains and task types.

2. **Weak semantic indexing**: Tool retrieval quality depends on accurate embeddings. Use domain-aware embedding models or fine-tune on your specific website/task vocabulary.

3. **Human bottleneck**: Tool Manager review can become a bottleneck if high volumes of tools are generated. Implement filtering (e.g., only review tools with novel patterns) or raise human approval thresholds.

4. **State drift**: Browser state can diverge between Reconnaissance and Action team execution (DOM changes, session expiry). Snapshot and restore state explicitly.

5. **Tool overfitting**: Generated tools may encode environment-specific quirks that don't generalize. Validate tools across multiple similar tasks before archiving.

6. **Missing preconditions**: When synthesizing tools, the Coder may overlook prerequisites (e.g., "user must be logged in"). Explicitly validate preconditions before execution.

7. **Recursive tool dependencies**: Avoid generating tools that depend on other recently-generated tools. This creates fragile dependency chains. Prefer tools grounded in stable, observable webpage features.

8. **Fallback action loops**: If fallback action generation frequently succeeds but produces wrong results, the system may converge to incorrect behaviors. Monitor fallback success metrics separately and periodically audit.

## Reference

**Paper**: He, K., Wang, Z., Zhuang, C., & Gu, J. (2025). Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance, Tool Generation, and Task Execution. arXiv:2509.21072.

**Available at**: https://arxiv.org/abs/2509.21072

**Benchmark**: VisualWebArena (3 domains: shopping, classifieds, Reddit)

**Key Results**: 36.48% overall success rate, achieved with only 11 generated tools and <10 training examples per domain.

**Related Work**: Builds on multi-agent orchestration, tool learning, and trajectory analysis. Complements systems like WebArena, VisualWebArena, and agent-based planning frameworks.
