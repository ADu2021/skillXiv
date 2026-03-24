---
name: youtu-agent-framework
title: "Youtu-Agent: Scaling Agent Productivity with Automated Generation and Hybrid Policy Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.24615"
keywords: [Agent Framework, Automated Agent Generation, LLM Agents, Policy Optimization, Agent RL]
description: "Build scalable LLM agent frameworks with automated generation of tools, prompts, and configurations, plus continuous optimization through experience accumulation and reinforcement learning—achieving 71.47% on WebWalkerQA and 72.8% on GAIA."
---

## Overview

Youtu-Agent is a modular framework for constructing and continuously improving LLM-based agents. It addresses two critical bottlenecks: the high manual effort required for agent configuration (tool integration, prompt engineering) and the difficulty in adapting deployed agents without expensive fine-tuning.

**Key Innovation:** Automated generation of complete agent configurations from high-level task descriptions, combined with two optimization mechanisms: Agent Practice (low-cost in-context learning) and Agent RL (end-to-end reinforcement learning at scale).

## Core Architecture

The framework organizes agent execution into three hierarchical layers:

**Environment Layer**
Provides foundational execution context and primitives. Typical backends include browser automation (Playwright), OS shells, and sandboxed code execution (E2B).

**Tools Layer**
Encapsulates atomic and composite operations: (1) environment-related tools wrapping low-level APIs, (2) environment-independent utilities, (3) MCP tools for external services.

**Agent Layer**
Houses the LLM-driven planner with a perceive-reason-act loop and Context Manager for managing long-horizon interactions.

## Automated Generation Mechanisms

### Workflow Mode
A deterministic four-stage pipeline for routine tasks:

1. **Intent Clarification and Decomposition** - Analyze and structure task requirements
2. **Tool Retrieval and Ad-hoc Synthesis** - Search existing toolkit library; auto-generate missing tools
3. **Prompt Engineering** - Generate optimized system instructions
4. **Configuration Assembly** - Compile all components into YAML configuration

This mode achieves 100% configuration validity on the 80-task AgentGen benchmark.

### Meta-Agent Mode
For complex or ambiguous requirements, deploy an Architect Agent with tools: `search_tool`, `create_tool`, `ask_user`, and `create_agent_config`. The meta-agent dynamically plans generation through multi-turn clarification and tool synthesis.

**Example:** Given "Summarize today's trending papers on multi-agent systems and download PDFs," the meta-agent:
- Calls `search_tool` to find existing arxiv toolkit
- Calls `create_tool` to synthesize `fetch_daily_papers` tool
- Calls `create_agent_config` to assemble final configuration

Tool synthesis implementation:

```python
def fetch_daily_papers(date: str) -> str:
    """Crawl daily papers from aggregation site.

    Args:
        date (str): date in format YYYY-MM-DD
    """
    papers = list_daily_papers(date=date)
    return "\n".join([f"{asdict(paper)}" for paper in papers])
```

Meta-Agent mode achieves 98.75% configuration validity and 68.75% end-to-end task completion on AgentGen-80.

## Continuous Optimization: Agent Practice

The Agent Practice module enables low-cost agent improvement without parameter updates through Training-free Group Relative Policy Optimization (Training-free GRPO).

**Mechanism:**
1. Agent performs multiple rollouts per task, generating diverse solution trajectories
2. LLM evaluator compares trajectories and extracts semantic group advantage (textual learning direction)
3. During inference, learned experiences are injected as "textual LoRA" to guide reasoning

**Results:**
- +2.7% improvement on AIME 2024 with 100 samples and ~$18 cost
- +5.4% improvement on AIME 2025
- Works with API-based models where fine-tuning is inaccessible
- No gradient computation required

Training dynamics show steady performance improvement with decreasing tool usage, indicating more efficient problem-solving strategies.

## Continuous Optimization: Agent RL

For applications requiring significant lasting improvement, Agent RL provides end-to-end reinforcement learning at production scale.

**Scalability Solutions:**
- RESTful API wrapping for distributed agent execution
- Ray-based concurrency for parallel rollout collection
- Hierarchical timeout logic to handle failures gracefully
- Enables scaling to 128 GPUs with 40% speedup vs. baseline

**Stability Solutions:**
- Filter invalid/anomalous tool calls during training
- Prevent policy overfitting through controlled off-policy updates
- Correct bias in advantage estimation for turn-level GRPO
- Mitigate "entropy explosion" in long-horizon tasks

**Results on Qwen2.5-7B:**
- Math tasks (AIME 2024): +35% improvement (10% → 45%)
- Search tasks: +17% to +21% across multiple QA benchmarks
- Stable training dynamics with consistent KL divergence and gradient norms

## Implementation Pattern

YAML-based configuration system enables both manual composition and automated generation:

```yaml
agent:
  name: research_agent
  instructions: "You are a helpful research assistant..."
env:
  name: e2b
  config: {}
context_manager:
  name: base
  config: {}
toolkits:
  search:
    activated_tools: ["search", "web_qa"]
  python_executor:
    activated_tools: ["execute_python_code"]
```

The standardized format serves as the target schema for automated generation and facilitates sharing of agent variants across teams.

## Benchmark Performance

**WebWalkerQA (680 questions):**
- Youtu-Agent: 71.47% pass@1
- Tests multi-step web navigation and deep reasoning
- Uses only open-source models (DeepSeek-V3)

**GAIA Text-only Subset (466 questions):**
- Youtu-Agent: 72.8% pass@1
- Evaluates real-world QA with tool-use proficiency
- Effective tool selection and multi-step reasoning

## When to Use

**Use Youtu-Agent when:**
- Building complex multi-step automation with web navigation, code execution, or external APIs
- Needing to rapidly deploy agents for diverse tasks without manual configuration
- Deploying agents that need to adapt over time without expensive fine-tuning cycles
- Scaling agent training across distributed systems for performance improvement
- Integrating LLM agents into production systems with reliability constraints

**When NOT to use:**
- Simple single-step tasks better served by direct API calls
- Scenarios requiring proprietary model APIs (Claude, GPT) only
- Applications where reproducibility of agent behavior is paramount over optimization
- Low-latency inference where overhead of agent reasoning is prohibitive

## Related Patterns

- **Agent Frameworks:** MetaGPT, AutoGen, ChatDev (focus on role orchestration vs. automated configuration synthesis)
- **Automated Agent Design:** ADAS, AutoAgents (generate agent designs; Youtu-Agent additionally synthesizes tool code)
- **Agent Optimization:** Reflexion, ReAct (verbal reinforcement); Youtu-Agent combines inference-time optimization with scalable RL

## Code Availability

Full implementation available at: https://github.com/TencentCloudADP/youtu-agent

## References

- Youtu-Agent achieves state-of-the-art with open-source models on WebWalkerQA and GAIA benchmarks
- Training-free GRPO enables low-cost improvement with 100 samples at ~$18 learning cost
- Agent RL module achieves 40% speedup with stable 128-GPU scaling
