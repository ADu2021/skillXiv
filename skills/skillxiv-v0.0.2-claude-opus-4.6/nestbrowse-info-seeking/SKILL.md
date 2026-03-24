---
name: nestbrowse-info-seeking
title: "Nested Browser-Use Learning for Agentic Information Seeking"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.23647
keywords: [information-seeking, web-agents, nested-architecture, efficiency]
description: "Scale information-seeking agents via nested architecture separating outer reasoning from inner page exploration. Minimal toolkit (search, visit, click, fill) handles dynamic web interactions, inner loop filters content before context injection, multi-task learning trains outer/inner jointly—enabling 4B models to match larger competitors."
---

## Overview

Nested architecture decouples reasoning from page content extraction.

## Core Technique

**Nested Loop Structure:**

```python
# Outer loop: reasoning
reasoning = outer_agent.think(state)
tool_call = outer_agent.select_tool()

# Inner loop: extract relevant content
relevant_content = inner_agent.extract(page, goal)

# Return minimal content to reasoning
```

## When to Use

Use when: Information-seeking agents, large-scale web interaction, efficiency critical.

## References

- Nested outer/inner loop architecture
- Minimal browser toolkit
- Content filtering before context injection
