---
name: mcp-bench-tool-agents
title: MCP-Bench Evaluation Framework for Tool-Using Agents
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.20453
keywords: [model-context-protocol, tool-agents, benchmarking, multi-hop, orchestration]
description: "Evaluate LLM agents on realistic tool-use tasks via 28 live MCP servers with 250 tools, assessing fuzzy tool discovery, multi-step planning, and cross-domain workflow coordination"
---

# MCP-Bench: Benchmarking Tool-Using LLM Agents via MCP Servers

## Core Concept

MCP-Bench evaluates language models on authentic tool-use scenarios using the Model Context Protocol to connect live tool servers. Rather than simulating tools, it employs 28 real MCP servers spanning finance, travel, scientific computing, and academic search. Tasks require agents to discover tools from ambiguous instructions, plan multi-hop execution, and orchestrate cross-domain workflows—revealing genuine limitations in current model capabilities.

## Architecture Overview

- **Live MCP Infrastructure**: 28 real servers providing 250 distinct tools with genuine input-output coupling
- **Fuzzy Tool Discovery**: Agents must locate appropriate tools without explicit names in instructions
- **Multi-Hop Planning**: Complex objectives requiring sequential tool invocations
- **Grounded Reasoning**: Agents must interpret and react to intermediate outputs
- **Cross-Domain Orchestration**: Workflows spanning multiple domains with shared context

## Implementation Steps

### Stage 1: Setup MCP Server Infrastructure

Deploy live MCP servers and establish connections to tool providers.

```python
# MCP server setup and connection management
import json
import asyncio
from typing import Dict, List, Any

class MCPServerManager:
    """Manage connections to live MCP servers"""

    def __init__(self, server_configs: List[Dict]):
        self.servers = {}
        self.tools_by_domain = {}
        self.server_configs = server_configs

    async def initialize_servers(self):
        """Connect to all MCP servers"""
        for config in self.server_configs:
            try:
                server = await self.connect_to_server(config)
                self.servers[config["name"]] = server

                # Index tools by domain
                tools = await server.list_tools()
                domain = config.get("domain", "general")
                self.tools_by_domain.setdefault(domain, []).extend(tools)

                print(f"Connected to {config['name']}: {len(tools)} tools")
            except Exception as e:
                print(f"Failed to connect to {config['name']}: {e}")

    async def connect_to_server(self, config: Dict):
        """Establish connection to MCP server"""
        # Connect via stdio or HTTP based on config
        if config.get("type") == "stdio":
            return await self.connect_stdio(config["command"], config["args"])
        elif config.get("type") == "http":
            return self.connect_http(config["url"])

    async def execute_tool(self, server_name: str, tool_name: str, args: Dict) -> Any:
        """Execute a specific tool on a server"""
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")

        result = await server.call_tool(tool_name, arguments=args)
        return result

    def get_tools_for_domain(self, domain: str) -> List[Dict]:
        """Get available tools for a domain"""
        return self.tools_by_domain.get(domain, [])
```

### Stage 2: Define Benchmark Tasks with Clear Objectives

Create realistic multi-step tasks requiring tool discovery and orchestration.

```python
# Benchmark task definitions
class MCPTask:
    """Individual benchmark task"""

    def __init__(
        self,
        task_id: str,
        domain: str,
        objective: str,
        required_tools: List[str],
        expected_output_schema: Dict
    ):
        self.task_id = task_id
        self.domain = domain
        self.objective = objective
        self.required_tools = required_tools  # Gold standard
        self.expected_output_schema = expected_output_schema
        self.examples = []

    def add_example(self, instruction: str, ground_truth: Dict):
        """Add example with expected output"""
        self.examples.append({
            "instruction": instruction,
            "ground_truth": ground_truth
        })

# Sample benchmark tasks
BENCHMARK_TASKS = [
    MCPTask(
        task_id="finance_portfolio",
        domain="finance",
        objective="Retrieve stock prices and compute portfolio value",
        required_tools=["get_stock_price", "compute_portfolio"],
        expected_output_schema={"portfolio_value": float, "holdings": list}
    ),
    MCPTask(
        task_id="travel_itinerary",
        domain="travel",
        objective="Book flights and hotels for a week-long trip",
        required_tools=["search_flights", "search_hotels", "book_reservation"],
        expected_output_schema={"flights": list, "hotels": list, "total_cost": float}
    ),
    MCPTask(
        task_id="academic_search",
        domain="academic",
        objective="Find recent papers on a topic and summarize key findings",
        required_tools=["search_papers", "extract_abstract", "summarize"],
        expected_output_schema={"papers": list, "summary": str}
    ),
]
```

### Stage 3: Implement Tool Discovery Mechanism

Create evaluation logic that tests agents on finding the right tools without explicit names.

```python
# Tool discovery evaluation
class ToolDiscoveryEvaluator:
    """Evaluate fuzzy tool discovery capability"""

    def __init__(self, mcp_manager: MCPServerManager):
        self.mcp_manager = mcp_manager

    async def evaluate_discovery(
        self,
        agent,
        task: MCPTask,
        instruction: str
    ) -> Dict:
        """
        Test agent's ability to discover appropriate tools from ambiguous instruction
        """
        # Agent must determine which tools to use from task description
        # without explicit tool names
        discovered_tools = await agent.discover_tools(
            instruction,
            available_domains=[task.domain]
        )

        # Score based on precision and recall
        required_set = set(task.required_tools)
        discovered_set = set(discovered_tools)

        precision = len(required_set & discovered_set) / len(discovered_set) \
            if discovered_set else 0
        recall = len(required_set & discovered_set) / len(required_set) \
            if required_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0

        return {
            "discovered_tools": discovered_tools,
            "required_tools": task.required_tools,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
```

### Stage 4: Implement Multi-Hop Planning Evaluator

Test agents on sequential tool invocations with output grounding.

```python
# Multi-hop planning evaluation
class MultiHopPlanningEvaluator:
    """Evaluate multi-step execution planning"""

    async def evaluate_planning(
        self,
        agent,
        task: MCPTask,
        instruction: str
    ) -> Dict:
        """
        Test agent's ability to plan and execute multi-step workflows
        """
        execution_trace = []
        agent_output = None

        try:
            # Agent generates plan and executes
            plan = await agent.generate_plan(instruction, task)
            execution_trace.append({
                "step": "planning",
                "plan": plan
            })

            # Execute plan step by step
            current_state = {}
            for step_idx, step in enumerate(plan):
                # Execute tool call
                result = await self.execute_plan_step(step, current_state)
                execution_trace.append({
                    "step": step_idx,
                    "action": step,
                    "result": result,
                    "success": result is not None
                })

                if result is None:
                    break

                current_state.update(result)

            agent_output = current_state

        except Exception as e:
            execution_trace.append({
                "step": "error",
                "error": str(e)
            })

        # Evaluate output against expected schema
        output_correct = self.validate_output(
            agent_output,
            task.expected_output_schema
        )

        return {
            "execution_trace": execution_trace,
            "output_correct": output_correct,
            "num_steps": len([s for s in execution_trace if "step" in s and isinstance(s["step"], int)]),
            "plan_length": len(plan) if plan else 0
        }

    async def execute_plan_step(self, step: Dict, current_state: Dict) -> Dict:
        """Execute a single planning step"""
        try:
            result = await self.mcp_manager.execute_tool(
                step["server"],
                step["tool"],
                step.get("arguments", {})
            )
            return {step.get("output_key", f"step_{step['tool']}"): result}
        except Exception:
            return None
```

### Stage 5: Implement Cross-Domain Orchestration Evaluator

Test agents on workflows spanning multiple domains.

```python
# Cross-domain orchestration evaluation
class CrossDomainEvaluator:
    """Evaluate cross-domain workflow coordination"""

    async def evaluate_orchestration(
        self,
        agent,
        workflow_task: Dict
    ) -> Dict:
        """
        Test agent on tasks requiring coordination across domains
        """
        domains_used = set()
        tools_used = []
        errors = []

        # Execute workflow
        current_context = {}

        for step_def in workflow_task["steps"]:
            domain = step_def["domain"]
            domains_used.add(domain)

            try:
                # Get available tools for domain
                tools = self.mcp_manager.get_tools_for_domain(domain)

                # Agent selects appropriate tool
                selected_tool = await agent.select_tool(
                    step_def["objective"],
                    tools,
                    current_context
                )

                # Execute and update context
                result = await self.mcp_manager.execute_tool(
                    domain,
                    selected_tool["name"],
                    selected_tool.get("args", {})
                )

                tools_used.append({
                    "domain": domain,
                    "tool": selected_tool["name"],
                    "success": result is not None
                })

                current_context[step_def.get("key", f"step_{len(tools_used)}")] = result

            except Exception as e:
                errors.append({
                    "step": step_def,
                    "error": str(e)
                })

        return {
            "domains_used": list(domains_used),
            "num_domains": len(domains_used),
            "tools_used": tools_used,
            "num_tools": len(tools_used),
            "errors": errors,
            "success": len(errors) == 0
        }
```

### Stage 6: Run Full Benchmark

Execute comprehensive evaluation across all tasks and models.

```python
# Full benchmark runner
async def run_full_benchmark(
    agents: Dict[str, Any],
    mcp_manager: MCPServerManager,
    tasks: List[MCPTask]
) -> Dict:
    """Run complete MCP-Bench evaluation"""

    results = {agent_name: {} for agent_name in agents}

    for task in tasks:
        print(f"\nEvaluating task: {task.task_id}")

        for agent_name, agent in agents.items():
            # Run all three evaluation dimensions
            discovery_results = await ToolDiscoveryEvaluator(mcp_manager).evaluate_discovery(
                agent, task, task.objective
            )

            planning_results = await MultiHopPlanningEvaluator(mcp_manager).evaluate_planning(
                agent, task, task.objective
            )

            orchestration_results = await CrossDomainEvaluator(mcp_manager).evaluate_orchestration(
                agent, {
                    "steps": [
                        {"domain": task.domain, "objective": task.objective}
                    ]
                }
            )

            results[agent_name][task.task_id] = {
                "discovery": discovery_results,
                "planning": planning_results,
                "orchestration": orchestration_results
            }

    return results
```

## Practical Guidance

### Task Design

- **Ambiguous Instructions**: Avoid mentioning tool names; use natural language
- **Multi-Hop Requirements**: Design 3-5 step workflows requiring sequential planning
- **Cross-Domain Tasks**: Include objectives spanning 2+ domains with shared context
- **Realistic Constraints**: Use actual API limitations, rate limits, and error conditions

### Evaluation Metrics

- **Schema Coverage**: Tool discovery precision and recall
- **Planning Accuracy**: Correct step sequence and parameter selection
- **Execution Success**: Actual completion rate and output correctness
- **Domain Diversity**: Coverage of tools across multiple domains

### When to Use

- Assessing real-world agent capabilities on authentic workflows
- Comparing models on standardized tool-use benchmarks
- Identifying specific failure modes in tool discovery or planning
- Evaluating cross-domain reasoning abilities

### When NOT to Use

- Isolated single-tool scenarios (use simpler benchmarks)
- Domains without available live tool servers
- Real-time production evaluation (use offline logs instead)

## Reference

MCP-Bench: Benchmarking Tool-Using LLM Agents via MCP Servers. arXiv:2508.20453
- https://arxiv.org/abs/2508.20453
