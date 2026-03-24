---
name: trust-sql-rl-text-to-sql
title: "TRUST-SQL: Tool-Integrated Multi-Turn RL for Text-to-SQL over Unknown Schemas"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.16448"
keywords: [Text-to-SQL, Reinforcement Learning, Schema Discovery, Agent Reasoning, Policy Gradient]
description: "Convert natural language to SQL for unknown database schemas by formulating the task as a partially observable MDP. Use dual-track GRPO (token-level masked advantages) to learn schema discovery and query generation jointly."
---

# TRUST-SQL: Multi-Turn RL for Text-to-SQL over Unknown Schemas

Most text-to-SQL systems assume full schema information is available, but production databases contain hundreds of tables with noisy metadata. TRUST-SQL treats this realistically: an autonomous agent must discover relevant schema information iteratively before generating queries. By formulating the task as a Partially Observable Markov Decision Process (POMDP) and applying a novel dual-track reinforcement learning strategy (token-level masked advantages for credit assignment), the system achieves 30% absolute improvement on 4B models and 16% on 8B models while operating without pre-loaded schema.

The key innovation is token-level advantage masking that isolates exploration rewards from execution outcomes, solving the credit assignment problem in multi-turn tool use.

## Core Concept

TRUST-SQL operates through an iterative four-phase schema discovery protocol:

1. **Schema Exploration** — Agent identifies potentially relevant tables using natural language search
2. **Metadata Verification** — Agent verifies column names and types for selected tables
3. **Constraint Discovery** — Agent identifies key relationships and constraints
4. **Query Generation** — Agent generates SQL query based on discovered schema information

The agent learns both schema discovery strategies (which tables/columns to explore) and query generation simultaneously through reinforcement learning.

## Architecture Overview

- **Schema Exploration API** — Tool interface for querying database metadata
- **Observation History Tracker** — Maintains what the agent has already discovered
- **Multi-Turn Agent** — Decides which tools to invoke and when to transition to query generation
- **POMDP State Representation** — Encodes partial schema knowledge and reasoning history
- **Dual-Track Reward** — Separates schema discovery rewards from query execution rewards
- **Token-Level Advantage Masking** — Applies per-token credit assignment for multi-turn reasoning
- **SQL Validator** — Verifies query syntax and executability

## Implementation Steps

Start by defining the POMDP state space and environment interface.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class PODMPState:
    """Partially observable state during schema discovery and query generation."""
    natural_language_question: str
    discovered_tables: Dict[str, Dict]  # table_name -> {columns, types}
    explored_columns: set  # (table, column) tuples examined
    tool_history: List[Dict]  # Log of tool calls and results
    current_turn: int

    def to_context(self) -> str:
        """Convert state to language model context."""
        context = f"Question: {self.natural_language_question}\n\n"
        context += "Discovered Schema:\n"
        for table, info in self.discovered_tables.items():
            context += f"  Table: {table}\n"
            for col, col_type in info.get('columns', {}).items():
                context += f"    - {col}: {col_type}\n"
        context += f"\nCurrent Turn: {self.current_turn}\n"
        return context


class SchemaDiscoveryEnvironment:
    """Environment for POMDP-based schema discovery."""

    def __init__(self, database_connection, initial_tables_hidden=True):
        self.db = database_connection
        self.initial_tables_hidden = initial_tables_hidden
        self.all_tables = self._get_all_tables()
        self.all_columns = self._get_all_columns()

    def _get_all_tables(self) -> Dict:
        """Get complete schema (for reference/evaluation only)."""
        tables = {}
        query = "SELECT table_name FROM information_schema.tables"
        for row in self.db.execute(query):
            table_name = row[0]
            columns = {}
            col_query = f"""SELECT column_name, data_type
                           FROM information_schema.columns
                           WHERE table_name = '{table_name}'"""
            for col_row in self.db.execute(col_query):
                columns[col_row[0]] = col_row[1]
            tables[table_name] = {'columns': columns}
        return tables

    def search_tables(self, query_text: str, limit=5) -> List[str]:
        """Search for relevant tables using text similarity."""
        # Use semantic search or keyword matching
        candidates = []
        query_lower = query_text.lower()

        for table_name in self.all_tables.keys():
            score = self._similarity_score(query_lower, table_name.lower())
            candidates.append((table_name, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [table for table, _ in candidates[:limit]]

    def inspect_table(self, table_name: str) -> Optional[Dict]:
        """Get columns and types for a specific table."""
        if table_name not in self.all_tables:
            return None

        return self.all_tables[table_name]

    def search_columns(self, table_name: str, query_text: str,
                      limit=5) -> List[str]:
        """Search for relevant columns in a table."""
        if table_name not in self.all_tables:
            return []

        columns = self.all_tables[table_name]['columns']
        candidates = []

        for col_name in columns.keys():
            score = self._similarity_score(query_text.lower(),
                                          col_name.lower())
            candidates.append((col_name, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in candidates[:limit]]

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Simple substring/token similarity."""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union

    def validate_sql(self, sql_query: str) -> (bool, str):
        """Check if SQL is valid and executable."""
        try:
            # Parse and validate without executing
            result = self.db.execute(f"EXPLAIN {sql_query}", timeout=5)
            return True, "Valid SQL"
        except Exception as e:
            return False, str(e)
```

Now implement the multi-turn agent with tool use and the dual-track reward system.

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW

class SchemaDiscoveryAgent:
    """Multi-turn RL agent for iterative schema discovery."""

    def __init__(self, language_model, environment, max_turns=10):
        self.llm = language_model
        self.env = environment
        self.max_turns = max_turns

        # Tool definitions
        self.tools = {
            'search_tables': self.env.search_tables,
            'inspect_table': self.env.inspect_table,
            'search_columns': self.env.search_columns,
            'generate_sql': self._generate_sql_candidate
        }

    def step(self, state: PODMPState) -> (Dict, float):
        """One turn: decide next action and compute reward."""
        context = state.to_context()
        context += "\nAvailable tools: search_tables, inspect_table, search_columns, generate_sql\n"
        context += "Choose next action:"

        # LLM selects next action
        action_text = self.llm.generate(context, max_tokens=100)
        parsed_action = self._parse_action(action_text)

        if not parsed_action:
            return {'error': 'Could not parse action'}, -0.5

        tool_name = parsed_action['tool']
        tool_args = parsed_action['args']

        # Execute tool
        try:
            result = self.tools[tool_name](**tool_args)
        except Exception as e:
            return {'error': str(e)}, -0.2

        # Compute reward
        reward = self._compute_reward(tool_name, result, state)

        return {
            'tool': tool_name,
            'args': tool_args,
            'result': result
        }, reward

    def _parse_action(self, action_text: str) -> Optional[Dict]:
        """Parse LLM output into structured action."""
        # Simple parsing; in production use more robust extraction
        if 'search_tables' in action_text:
            query = action_text.split("query=")[-1].split('"')[1] if 'query=' in action_text else ''
            return {'tool': 'search_tables', 'args': {'query_text': query}}
        elif 'inspect_table' in action_text:
            table = action_text.split('table=')[-1].split('"')[1] if 'table=' in action_text else ''
            return {'tool': 'inspect_table', 'args': {'table_name': table}}
        # ... other tools
        return None

    def _compute_reward(self, tool_name: str, result: Dict,
                       state: PODMPState) -> float:
        """Compute exploration vs execution reward."""
        exploration_reward = 0.0
        execution_reward = 0.0

        if tool_name == 'search_tables':
            # Reward discovering new tables
            new_tables = len([t for t in result
                            if t not in state.discovered_tables])
            exploration_reward = 0.1 * new_tables

        elif tool_name == 'inspect_table':
            # Reward discovering new columns
            exploration_reward = 0.05 * len(result.get('columns', {}))

        elif tool_name == 'generate_sql':
            # Execution reward: correctness of generated query
            is_valid, msg = self.env.validate_sql(result)
            execution_reward = 0.5 if is_valid else -0.5

        # Composite reward
        return 0.6 * exploration_reward + 0.4 * execution_reward

    def _generate_sql_candidate(self, discovered_schema: Dict,
                               question: str) -> str:
        """Generate SQL based on discovered schema."""
        prompt = f"""Based on this schema:
{json.dumps(discovered_schema, indent=2)}

Answer the question with SQL: {question}"""
        return self.llm.generate(prompt, max_tokens=200)
```

Finally, implement the dual-track GRPO with token-level masked advantages.

```python
class TokenLevelMaskedGRPO:
    """Dual-track GRPO with token-level advantage masking for credit assignment."""

    def __init__(self, model, agent, environment):
        self.model = model
        self.agent = agent
        self.env = environment
        self.optimizer = AdamW(model.parameters(), lr=1e-5)

    def compute_masked_advantages(self, trajectory: List[Dict],
                                 rewards: List[float],
                                 action_types: List[str]) -> torch.Tensor:
        """Compute per-token advantages, masking by action type."""
        advantages = []

        for i, (action_type, reward) in enumerate(zip(action_types, rewards)):
            # Separate advantage computation by tool type
            if action_type == 'generate_sql':
                # Execution reward applies to all tokens
                advantage = reward
            else:
                # Exploration reward applies to decision tokens only
                advantage = reward * 0.5

            advantages.append(advantage)

        return torch.tensor(advantages, dtype=torch.float32)

    def update_step(self, question: str, max_turns: int = 10):
        """One RL update step."""
        state = PODMPState(
            natural_language_question=question,
            discovered_tables={},
            explored_columns=set(),
            tool_history=[],
            current_turn=0
        )

        trajectory = []
        rewards = []
        action_types = []
        logprobs = []

        # Rollout: collect trajectory
        for turn in range(max_turns):
            action_dict, reward = self.agent.step(state)

            if 'error' in action_dict:
                break

            trajectory.append(action_dict)
            rewards.append(reward)
            action_types.append(action_dict['tool'])

            # Compute log-probability for this action
            context = state.to_context()
            logprob = self._compute_action_logprob(context, action_dict)
            logprobs.append(logprob)

            # Update state
            state.tool_history.append(action_dict)
            state.current_turn += 1

            if action_dict['tool'] == 'generate_sql':
                break

        # Compute masked advantages
        advantages = self.compute_masked_advantages(trajectory, rewards,
                                                    action_types)

        # Policy gradient loss with per-token masking
        logprobs_tensor = torch.stack(logprobs)
        loss = -(advantages * logprobs_tensor).mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), sum(rewards)

    def _compute_action_logprob(self, context: str,
                               action_dict: Dict) -> torch.Tensor:
        """Compute log-probability of action under current policy."""
        # Generate action from context, compute log-prob of actual action
        logits = self.model(context)
        # Simplified; in practice extract token probabilities
        return torch.tensor(0.0)  # Placeholder

    def train(self, training_questions: List[str], num_steps: int = 100):
        """Train the agent."""
        for step in range(num_steps):
            question = training_questions[step % len(training_questions)]
            loss, total_reward = self.update_step(question)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}: Loss={loss:.4f}, "
                      f"Total Reward={total_reward:.3f}")
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Maximum turns typically 5-15; more turns enable thorough schema discovery but increase latency
- Exploration reward weight 0.6, execution reward 0.4; adjust based on schema complexity
- Use when schemas are large (100+ tables) and partially unknown
- Ideal for real-world database query tasks with noisy or incomplete metadata

**When NOT to use:**
- For small, fully-known schemas where direct prompting suffices
- When database access is restricted or very slow
- For applications requiring sub-100ms latency (schema discovery is inherently iterative)

**Common Pitfalls:**
- Agent getting stuck exploring irrelevant tables; use importance weighting to guide exploration
- Tool calls failing silently; add explicit error handling and retry logic
- Insufficient diversity in discovered schema; apply epsilon-greedy exploration
- Tool outputs being too verbose; summarize/filter tool results before feeding to LLM

## Reference

Paper: [TRUST-SQL: Tool-Integrated Multi-Turn RL for Text-to-SQL over Unknown Schemas](https://arxiv.org/abs/2603.16448)
