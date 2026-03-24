---
name: livemcpbench-agent-tool-navigation
title: LiveMCPBench - Evaluating Agents in Large-Scale Tool Ecosystems
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.01780
keywords: [benchmarking, agent-evaluation, tool-selection, mcp-tools]
description: "Benchmark framework for evaluating LLM agents navigating large-scale Model Context Protocol ecosystems with multi-tool composition across 95 daily tasks."
---

## LiveMCPBench: Agent Navigation in Tool Ecosystems

LiveMCPBench evaluates how well LLM-based agents can discover and compose tools from large, realistic Model Context Protocol (MCP) environments. The core challenge: with thousands of available tools, agents must develop sophisticated retrieval and composition strategies. The benchmark addresses this by providing 70 MCP servers with 527 tools, real-world tasks, and an LLM-as-Judge evaluation framework that handles dynamic data and multiple valid solutions.

### Core Concept

Standard agent benchmarks test tool use in toy environments with 10-50 tools. Real MCP ecosystems contain thousands of servers and tools—requiring agents to solve a dual problem: (1) retrieve relevant tools from massive search spaces, and (2) compose multiple tools coherently to solve tasks. LiveMCPBench bridges this gap by benchmarking both retrieval effectiveness and multi-tool reasoning with reproducible, task-agnostic evaluation.

### Architecture Overview

- **Task Design**: 95 daily tasks across 6 domains (Office, Lifestyle, Leisure, Finance, Travel, Shopping) emphasizing temporal dynamics and real-time information retrieval
- **LiveMCPTool Collection**: 70 Docker-packaged MCP servers providing 527 tools without external API dependencies, enabling reproducible evaluation
- **MCP Copilot Agent**: ReACT-based agent operating as a POMDP with Route, Execute, Response operations; tool selection via joint server-tool description alignment
- **LiveMCPEval**: LLM-as-Judge framework using "key points" (critical subtasks) for human-aligned evaluation supporting dynamic solutions
- **Tool Taxonomy**: Hierarchical organization enabling efficient retrieval and composition analysis

### Implementation Steps

**Step 1: Build the Task Definition Framework**

Design tasks that require multi-step tool composition with temporal reasoning:

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict

class TaskDomain(Enum):
    OFFICE = "office"
    LIFESTYLE = "lifestyle"
    LEISURE = "leisure"
    FINANCE = "finance"
    TRAVEL = "travel"
    SHOPPING = "shopping"

@dataclass
class Task:
    """Represents a single evaluation task."""
    id: str
    domain: TaskDomain
    description: str
    key_points: List[str]  # Critical subtasks for success
    required_tools: List[str]  # Tools needed (for analysis)
    temporal_constraints: Dict[str, str]  # Time-dependent requirements
    success_criteria: str

    def to_agent_prompt(self):
        """Convert task to prompt for agent."""
        prompt = f"""Task: {self.description}

Success Criteria:
{self.success_criteria}

Key Milestones to Achieve:
"""
        for kp in self.key_points:
            prompt += f"- {kp}\n"

        if self.temporal_constraints:
            prompt += "\nTemporal Constraints:\n"
            for constraint, details in self.temporal_constraints.items():
                prompt += f"- {constraint}: {details}\n"

        return prompt

# Example task definitions
tasks = [
    Task(
        id="office-meeting-001",
        domain=TaskDomain.OFFICE,
        description="Schedule a team meeting for next Tuesday at 2 PM with attendees from sales and engineering, then send calendar invites with meeting notes.",
        key_points=[
            "Find available time slots for all attendees",
            "Create calendar event",
            "Send calendar invites",
            "Attach meeting agenda"
        ],
        required_tools=["calendar", "email", "employee_lookup"],
        temporal_constraints={"meeting_date": "next Tuesday 2 PM", "invite_deadline": "24 hours before"},
        success_criteria="Calendar event created, all invites sent, meeting notes attached"
    ),
    Task(
        id="shopping-price-compare-001",
        domain=TaskDomain.SHOPPING,
        description="Find the best price for a specific laptop model from 3+ retailers, check current discounts, and compare total cost including shipping.",
        key_points=[
            "Search multiple retailers",
            "Extract price and shipping cost",
            "Check for active coupon codes",
            "Calculate total cost",
            "Identify cheapest option"
        ],
        required_tools=["product_search", "price_scraper", "coupon_finder"],
        temporal_constraints={"discount_validity": "current day only"},
        success_criteria="3+ retailer options with price comparison and cheapest option identified"
    ),
]

def create_task_dataset(num_tasks_per_domain=15):
    """Create balanced dataset of tasks across all domains."""
    all_tasks = []
    for domain in TaskDomain:
        domain_tasks = [t for t in tasks if t.domain == domain]
        # Expand with variations as needed
        all_tasks.extend(domain_tasks[:num_tasks_per_domain])
    return all_tasks
```

**Step 2: Implement the Tool Retrieval System**

Build effective tool discovery from large tool spaces:

```python
from typing import Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class ToolRetriever:
    """
    Retrieve relevant tools from massive tool catalog using semantic search.
    Addresses the critical bottleneck: retrieval accounts for ~50% of agent failures.
    """
    def __init__(self, tool_catalog, embedding_model='all-MiniLM-L6-v2'):
        self.tool_catalog = tool_catalog
        self.embedder = SentenceTransformer(embedding_model)

        # Pre-compute embeddings for all tools
        self.tool_descriptions = [
            f"{tool['name']}: {tool['description']}"
            for tool in tool_catalog
        ]
        self.tool_embeddings = self.embedder.encode(self.tool_descriptions)

    def retrieve_tools(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k tools using semantic similarity on combined server-tool description.
        Joint alignment is more effective than individual descriptions.
        """
        query_embedding = self.embedder.encode(query)

        # Compute similarities
        similarities = np.dot(self.tool_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        retrieved_tools = [self.tool_catalog[i] for i in top_indices]
        return retrieved_tools

    def retrieve_with_joint_alignment(self, task_query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Advanced retrieval: align both server name and tool description together.
        This "joint alignment" approach outperforms individual component matching.
        """
        task_embedding = self.embedder.encode(task_query)

        scored_tools = []
        for tool in self.tool_catalog:
            # Joint description: combine server and tool info
            joint_desc = f"Server: {tool.get('server_name', 'unknown')} | Tool: {tool['name']} | {tool['description']}"
            joint_embedding = self.embedder.encode(joint_desc)

            # Compute alignment score
            alignment_score = np.dot(joint_embedding, task_embedding)
            scored_tools.append((tool, alignment_score))

        # Sort and return top-k
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[:k]

# Example usage
retriever = ToolRetriever(tool_catalog)
retrieved = retriever.retrieve_with_joint_alignment(
    "Schedule a meeting for Tuesday at 2 PM and send calendar invites",
    k=5
)
```

**Step 3: Build the ReACT Agent for Tool Composition**

Implement the agent loop with Route, Execute, Response operations:

```python
from enum import Enum
from typing import Callable, Any

class AgentOperation(Enum):
    ROUTE = "route"
    EXECUTE = "execute"
    RESPONSE = "response"
    REVISE = "revise"

class MCPCopilotAgent:
    """
    ReACT-based agent formulated as a POMDP.
    - State: current task progress, available tools, previous actions
    - Actions: Route (select tools), Execute (call tools), Respond (generate answer), Revise (recover from errors)
    """
    def __init__(self, model_name: str, retriever: ToolRetriever):
        self.model_name = model_name
        self.retriever = retriever
        self.action_history = []
        self.tool_execution_results = {}

    def route(self, state: Dict, task: Task, k: int = 5) -> List[Dict]:
        """
        Route operation: Select k candidate tools for current task state.
        Uses retriever with joint alignment.
        """
        # Construct query from task and current state
        query = task.description
        if state.get('recent_failures'):
            query += f" (Note: Previously tried and failed: {state['recent_failures']})"

        # Retrieve candidate tools
        candidates = self.retriever.retrieve_with_joint_alignment(query, k=k)
        selected_tools = [tool for tool, score in candidates]

        return selected_tools

    def execute(self, tool: Dict, input_params: Dict) -> Any:
        """
        Execute operation: Call the selected tool and capture result.
        """
        tool_name = tool['name']
        server_name = tool.get('server_name')

        # Invoke tool via MCP protocol
        result = {
            'tool': tool_name,
            'server': server_name,
            'status': 'executing',
            'input': input_params
        }

        try:
            # Call actual tool (would be via MCP client in practice)
            output = self.call_mcp_tool(server_name, tool_name, input_params)
            result['status'] = 'success'
            result['output'] = output
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        # Store for state tracking
        self.action_history.append(result)
        self.tool_execution_results[f"{tool_name}_{len(self.action_history)}"] = result

        return result

    def response(self, state: Dict, task: Task) -> str:
        """
        Response operation: Generate final answer from execution results.
        """
        # Compile evidence from tool execution
        evidence = ""
        for action in self.action_history:
            if action['status'] == 'success':
                evidence += f"- {action['tool']}: {action['output']}\n"

        # Generate response via LLM
        prompt = f"""Task: {task.description}

Execution Results:
{evidence}

Key Points to Address:
{chr(10).join('- ' + kp for kp in task.key_points)}

Provide a clear response addressing all key points based on the tool results above."""

        response = self.model.generate(prompt)
        return response

    def run_episode(self, task: Task, max_steps: int = 10) -> Tuple[str, Dict]:
        """
        Run complete agent episode for a single task.
        Returns final response and execution statistics.
        """
        state = {
            'task': task,
            'step': 0,
            'completed_keypoints': [],
            'recent_failures': []
        }

        for step in range(max_steps):
            state['step'] = step

            # Step 1: Route (select tools)
            candidate_tools = self.route(state, task, k=5)

            if not candidate_tools:
                break

            # Step 2: Execute (call selected tool)
            # In multi-tool scenarios, call multiple tools and aggregate
            for tool in candidate_tools[:3]:  # Execute top 3
                params = self.generate_tool_params(tool, state, task)
                result = self.execute(tool, params)

                if result['status'] == 'error':
                    state['recent_failures'].append(tool['name'])
                else:
                    # Check if key points satisfied
                    for kp in task.key_points:
                        if kp not in state['completed_keypoints']:
                            state['completed_keypoints'].append(kp)

            # Step 3: Generate response if enough progress
            if len(state['completed_keypoints']) >= len(task.key_points):
                break

        # Generate final response
        final_response = self.response(state, task)

        stats = {
            'steps_taken': state['step'] + 1,
            'tools_used': len(self.action_history),
            'successful_calls': sum(1 for a in self.action_history if a['status'] == 'success'),
            'failed_calls': sum(1 for a in self.action_history if a['status'] == 'error'),
            'keypoints_completed': len(state['completed_keypoints']),
            'keypoints_total': len(task.key_points),
        }

        return final_response, stats

    def call_mcp_tool(self, server: str, tool: str, params: Dict) -> Any:
        """Placeholder for actual MCP tool invocation."""
        # Would use mcp.connect() and call actual tool
        pass

    def generate_tool_params(self, tool: Dict, state: Dict, task: Task) -> Dict:
        """Generate appropriate parameters for tool based on task context."""
        # LLM generates tool-specific parameters
        prompt = f"""Tool: {tool['name']}
Tool description: {tool['description']}
Task: {task.description}

Generate appropriate input parameters for this tool in JSON format."""
        params = self.model.generate_json(prompt)
        return params
```

**Step 4: Implement LLM-as-Judge Evaluation**

Evaluate agent responses using key points instead of fixed ground truth:

```python
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    task_id: str
    success: bool
    key_points_achieved: int
    key_points_total: int
    response_quality: float  # 0-1
    confidence: float  # Evaluator confidence
    reasoning: str

class LiveMCPEval:
    """
    LLM-as-Judge evaluation framework.
    Uses key points (critical subtasks) for human-aligned evaluation.
    Handles dynamic data and multiple valid solutions.
    """
    def __init__(self, judge_model_name: str):
        self.judge_model = judge_model_name

    def evaluate_response(self, response: str, task: Task, execution_stats: Dict) -> EvaluationResult:
        """
        Evaluate agent response against key points.
        Key points are subtasks that must be satisfied for success.
        """
        prompt = f"""Evaluate this agent response for the following task:

Task: {task.description}

Agent Response:
{response}

Execution Statistics:
- Steps taken: {execution_stats.get('steps_taken')}
- Tools used: {execution_stats.get('tools_used')}
- Successful tool calls: {execution_stats.get('successful_calls')}

Key Points that MUST be addressed:
"""
        for i, kp in enumerate(task.key_points, 1):
            prompt += f"{i}. {kp}\n"

        prompt += """For each key point, determine if the response adequately addresses it.
Provide:
1. Key point satisfaction: Which key points were addressed? (list numbers)
2. Quality assessment: How well was the overall task completed? (0-100)
3. Confidence: How confident are you in this evaluation? (0-100)
4. Reasoning: Briefly explain your assessment."""

        # Get LLM judgment
        judgment = self.judge_model.generate(prompt)

        # Parse judgment
        result = self._parse_judgment(judgment, task)
        return result

    def _parse_judgment(self, judgment: str, task: Task) -> EvaluationResult:
        """Parse structured judgment from LLM."""
        # Extract key points achieved (would parse from LLM output)
        lines = judgment.split('\n')

        keypoints_achieved = 0
        for line in lines:
            if 'key point' in line.lower() and any(str(i) in line for i in range(len(task.key_points))):
                keypoints_achieved += 1

        quality = 75  # Default, would be parsed from judgment
        confidence = 80  # Default, would be parsed from judgment

        success = keypoints_achieved >= len(task.key_points) * 0.8  # 80% threshold

        return EvaluationResult(
            task_id=task.id,
            success=success,
            key_points_achieved=keypoints_achieved,
            key_points_total=len(task.key_points),
            response_quality=quality / 100.0,
            confidence=confidence / 100.0,
            reasoning=judgment
        )

    def evaluate_batch(self, responses: List[str], tasks: List[Task],
                      execution_stats: List[Dict]) -> List[EvaluationResult]:
        """Evaluate multiple responses."""
        results = []
        for response, task, stats in zip(responses, tasks, execution_stats):
            result = self.evaluate_response(response, task, stats)
            results.append(result)
        return results
```

**Step 5: Analyze Bottlenecks**

Identify failure sources to improve agent design:

```python
def analyze_failure_modes(results: List[EvaluationResult], execution_stats: List[Dict]) -> Dict:
    """
    Analyze agent performance to identify bottlenecks.
    LiveMCPBench shows retrieval accounts for ~50% of failures.
    """
    failures = [r for r in results if not r.success]

    analysis = {
        'total_tasks': len(results),
        'successful_tasks': sum(1 for r in results if r.success),
        'success_rate': sum(1 for r in results if r.success) / len(results),
        'failures': {
            'retrieval_failures': 0,
            'execution_failures': 0,
            'composition_failures': 0,
            'reasoning_failures': 0
        }
    }

    for failure, stats in zip(failures, [execution_stats[results.index(r)] for r in failures]):
        # Retrieval failure: too few successful tool calls
        if stats['successful_calls'] == 0:
            analysis['failures']['retrieval_failures'] += 1
        # Execution failure: tools called but no results
        elif stats['tools_used'] > 0 and failure.key_points_achieved < len(failure.key_points_total) * 0.3:
            analysis['failures']['execution_failures'] += 1
        # Composition failure: tools worked but not combined effectively
        elif stats['successful_calls'] > 0 and failure.key_points_achieved < len(failure.key_points_total) * 0.5:
            analysis['failures']['composition_failures'] += 1
        # Reasoning failure: response incoherent
        else:
            analysis['failures']['reasoning_failures'] += 1

    return analysis
```

### Practical Guidance

**When to Use:**
- Evaluating agents that must discover tools from large ecosystems (>500 tools)
- Scenarios emphasizing multi-tool composition and reasoning
- Applications requiring reproducible benchmarking without external API dependencies
- Cases where evaluation must support multiple valid solution paths

**When NOT to Use:**
- Simple, single-tool tasks with clear ground truth answers
- Real-time evaluation (LLM-as-Judge adds latency)
- Domains with strict, deterministic output requirements (medical)
- Low-resource settings where maintaining 70 servers is infeasible

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `retrieval_k` | 5 | Number of candidate tools retrieved per decision; higher = better coverage, higher latency |
| `max_agent_steps` | 10 | Episode length limit; balance exploration vs. computational cost |
| `keypoint_success_threshold` | 0.8 | Fraction of key points needed for task success; lower = more lenient |
| `evaluator_confidence_threshold` | 0.7 | Minimum confidence required for evaluation; higher = stricter |

**Common Challenges:**
- Tool explosion: 527 tools causes retrieval challenges; improve via better descriptions and taxonomy
- Task ambiguity: Multiple solution paths; key points framework handles this better than ground truth
- Real-time data: Temporal constraints require live data; use mock data for reproducibility

### Reference

**Paper**: LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? (2508.01780)
- 95 realistic daily tasks across 6 domains with temporal dynamics
- 70 reproducible MCP servers with 527 tools (no external API dependencies)
- Identifies retrieval as dominant bottleneck (~50% of failures)
- LLM-as-Judge evaluation with 81% human agreement
- Performance ceiling at 78.95% (Claude-Sonnet-4), most models at 30-50%
