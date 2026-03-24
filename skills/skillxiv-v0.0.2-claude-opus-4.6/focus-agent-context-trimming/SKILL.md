---
name: focus-agent-context-trimming
title: "FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.03204"
keywords: [Web Agents, Context Management, Accessibility Trees, Observation Filtering, Token Efficiency]
description: "Use a lightweight LLM to filter accessibility tree observations by task relevance, reducing agent context size by 50-80% while maintaining equivalent task performance."
---

# Technique: Task-Guided LLM-Based Observation Filtering

Web agents navigate complex pages by parsing accessibility trees (AxTree)—DOM representations containing hundreds or thousands of elements. Traditional agents pass the entire tree as context, consuming excessive tokens and slowing inference. FocusAgent uses a small language model to selectively extract only task-relevant content lines from the accessibility tree at each step.

The key insight is that web navigation requires understanding page state and action consequences within a specific task context. A general embedding-based retrieval would miss crucial interactive relationships. By having a dedicated LLM analyze what matters for the current goal, the agent retains planning-relevant information while discarding layout noise, redundant navigation elements, and off-topic content.

## Core Concept

FocusAgent employs a two-level filtering architecture. A base web agent generates the full accessibility tree observation. A smaller "retriever" LLM (GPT-4o-mini) examines this tree along with a task goal and current step context, then identifies which lines are relevant to future decision-making. The filtered observation becomes the input for the main agent policy.

The retriever uses a "soft" strategy: when uncertain, it errs toward inclusion. This preserves task-critical information at the cost of slightly higher token usage, avoiding the risk of filtering away essential context.

## Architecture Overview

- **Observation Layer**: Generate full accessibility tree with numbered lines for each DOM element
- **Retriever Module**: Lightweight LLM that analyzes the tree and task context
- **Filtering Decision**: Line-level selection producing a compressed tree
- **Policy Input**: Main agent receives filtered observation as context
- **Feedback Loop**: Agent actions and results inform the next observation filtering step

## Implementation Steps

Create a function to generate numbered accessibility tree lines. Each element receives a unique ID for reference.

```python
def generate_numbered_axtree(page_html, include_attributes=True):
    """
    Parse page HTML into numbered accessibility tree lines.

    Args:
        page_html: Raw HTML from the web page
        include_attributes: Include element attributes in lines

    Returns:
        axtree_lines: List of (line_number, element_text, attributes) tuples
    """
    from html.parser import HTMLParser

    axtree_lines = []
    line_num = 1

    # Parse HTML and build tree representation
    for element in parse_html_to_tree(page_html):
        element_text = extract_element_text(element)
        attributes = extract_attributes(element) if include_attributes else {}

        # Format: "[1] Button: Click here" or "[1] Link: https://example.com - Example Page"
        formatted_line = f"[{line_num}] {element_text}"
        if attributes and 'href' in attributes:
            formatted_line += f" - {attributes['href']}"

        axtree_lines.append((line_num, formatted_line, attributes))
        line_num += 1

    return axtree_lines
```

Implement the retriever LLM call that identifies relevant lines.

```python
def filter_axtree_with_retriever(axtree_lines, task_goal, current_step,
                                  retriever_model="gpt-4o-mini"):
    """
    Use a lightweight LLM to filter accessibility tree by task relevance.

    Args:
        axtree_lines: List of formatted accessibility tree lines
        task_goal: The overall task objective (e.g., "Find and click on checkout button")
        current_step: Current action context (e.g., what was just attempted)
        retriever_model: LLM to use for filtering decision

    Returns:
        relevant_line_numbers: Set of line indices to retain
    """
    axtree_text = "\n".join([line[1] for line in axtree_lines])

    prompt = f"""Task Goal: {task_goal}
Current Step: {current_step}

Accessibility Tree:
{axtree_text}

Which lines are most relevant for the next decision? List line numbers [1], [2], etc.
Be inclusive when uncertain—better to keep slightly verbose context than miss crucial information.
Return only line numbers, one per line."""

    response = call_llm(retriever_model, prompt)

    # Parse response to extract line numbers
    relevant_lines = extract_line_numbers_from_response(response)

    return relevant_lines
```

Build the compressed observation by selecting only relevant lines.

```python
def compress_observation(axtree_lines, relevant_line_numbers):
    """
    Extract relevant lines and reconstruct accessibility tree.

    Args:
        axtree_lines: Full accessibility tree
        relevant_line_numbers: Set of line indices to keep

    Returns:
        compressed_tree: Filtered accessibility tree as text
        compression_ratio: Fraction of lines retained
    """
    compressed_lines = [
        line[1] for line in axtree_lines
        if line[0] in relevant_line_numbers
    ]

    compressed_tree = "\n".join(compressed_lines)
    compression_ratio = len(compressed_lines) / len(axtree_lines)

    return compressed_tree, compression_ratio
```

Integrate into the web agent's observation-action loop.

```python
def web_agent_step_with_focus(agent_policy, page_html, task_goal,
                               action_history, retriever_model="gpt-4o-mini"):
    """
    Single step of web agent with FocusAgent filtering.

    Args:
        agent_policy: Main agent LLM policy
        page_html: Current page HTML
        task_goal: Task objective
        action_history: Previous actions taken
        retriever_model: Retriever LLM for filtering

    Returns:
        action: Next action to execute (e.g., click element [42])
        compressed_observation: The filtered context used for decision
    """
    # Generate full accessibility tree
    axtree_lines = generate_numbered_axtree(page_html)

    # Get current step context from recent actions
    current_step = action_history[-1] if action_history else "Initial page load"

    # Filter using lightweight retriever
    relevant_lines = filter_axtree_with_retriever(
        axtree_lines, task_goal, current_step, retriever_model
    )

    # Compress observation
    compressed_obs, ratio = compress_observation(axtree_lines, relevant_lines)

    # Main agent policy uses compressed observation
    prompt = f"""Goal: {task_goal}

Available elements:
{compressed_obs}

What is the next action?"""

    action = agent_policy(prompt)

    return action, compressed_obs
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Retriever model | GPT-4o-mini or similar | Small models sufficient; balance cost vs. accuracy |
| Inclusion strategy | Soft/erring toward inclusion | Avoids filtering critical elements at cost of slightly higher tokens |
| Filter frequency | Every step | Recompute for each observation to capture changing relevance |
| Batch operations | Apply filter to multiple pages | Amortize retriever LLM cost across agents |
| Compression target | 50-80% reduction | Achievable on most pages; higher reduction increases risk |
| When to use | Complex multi-step web navigation | E-commerce, SaaS, information retrieval tasks |
| When NOT to use | Highly visual tasks or pages with minimal text | May filter out layout-critical information |
| Common pitfall | Over-filtering with aggressive thresholds | Safety margin is important; preserve questionable elements |

### When to Use FocusAgent

- Web agents navigating e-commerce sites, SaaS dashboards, or complex document sites
- Scenarios where token budget is constrained (cost-sensitive inference)
- Tasks requiring multiple steps with evolving context
- Pages with hundreds+ DOM elements

### When NOT to Use FocusAgent

- Single-page lookups where compression adds overhead
- Highly visual tasks where layout structure is essential
- Pages with sparse text and important implicit structure
- Real-time systems where latency of additional LLM call is prohibitive

### Common Pitfalls

- **Retriever timeout**: Set timeouts to fall back to full tree if retriever is slow
- **Loss of critical navigation elements**: Headers/footers often filtered despite importance for multi-step tasks
- **Lack of context carry-over**: Each step's filtering ignores previous steps' progress; maintain memory
- **Attribute loss**: Filtering removes element IDs/classes needed for reliable clicking—preserve these
- **Task goal drift**: Retriever may misinterpret task scope; provide precise goal descriptions

## Reference

Paper: https://arxiv.org/abs/2510.03204
