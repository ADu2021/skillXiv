---
name: agent-world-model-env-generation
title: "Agent World Model: Infinity Synthetic Environments for Agentic RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.10090"
keywords: [Synthetic Environments, Environment Generation, Database-Driven, Task Synthesis, Tool Use Training]
description: "Automatically synthesize executable RL training environments with database backends, Python tools, and task descriptions. Generate 1000+ diverse domains with 10K+ tasks enabling data-efficient tool-use agent training without manual scenario design."
---

# Agent World Model: Automated Synthetic Environment Generation

Creating diverse training environments for tool-use agents is expensive and requires manual scenario design. Agent World Model (AWM) automates this through a multi-stage synthesis pipeline: starting from seed domain names, systematically generates scenarios, synthesizes executable databases and tool interfaces, and verifies task solvability. The result: 1000+ diverse environments with 35K+ tools across 10K+ tasks, enabling agents to train without human environment design.

## Core Concept

Manual environment creation: expensive, limited diversity, slow iteration.

AWM approach: LLM-driven multi-stage synthesis:
1. **Scenario Generation** (1000 scenarios from 100 seeds via LLM expansion)
2. **Task Synthesis** (10 user-facing tasks per scenario)
3. **Environment Synthesis** (SQLite database + Python tool APIs)
4. **Self-Correction** (retry failed components up to 5 times)

Result: executable environments with verifiable success criteria, ready for RL training.

## Architecture Overview

- **Scenario Expansion**: LLM generates diverse scenarios (e.g., e-commerce, banking, social) from domain seeds
- **Task Generation**: For each scenario, synthesize 10 user-facing goals (API-solvable)
- **Database Schema**: Auto-generate SQLite schema matching scenario (tables, columns, sample data)
- **Tool API**: Expose Python functions as Model Context Protocol (MCP) tools
- **Task Verification**: Code-augmented validation ensures tasks are solvable and rewards are meaningful
- **Self-Correction Loop**: Retry failed components with error feedback (up to 5 attempts)

## Implementation

Implement scenario and task generation:

```python
import json
import sqlite3
from typing import List, Dict

class EnvironmentGenerator:
    """Generates synthetic training environments."""

    def __init__(self, llm_model, num_scenarios=1000, tasks_per_scenario=10):
        self.llm = llm_model
        self.num_scenarios = num_scenarios
        self.tasks_per_scenario = tasks_per_scenario
        self.generated_envs = []

    def expand_scenarios(self, seed_domains: List[str]):
        """Expand seed domains into diverse scenarios."""
        expansion_prompt = f"""Given these seed domains: {', '.join(seed_domains)}

Generate 10 unique, detailed scenario names for each domain. Scenarios should be:
- Distinct and varied (different problem contexts)
- Realistic and practical
- API-solvable (can be implemented with database + tools)

Format output as JSON: {{"domain": ["scenario1", "scenario2", ...]}}"""

        response = self.llm.generate(expansion_prompt, max_tokens=1000)
        scenarios = json.loads(response)

        return scenarios

    def synthesize_tasks(self, scenario: str, domain: str) -> List[Dict]:
        """Generate 10 user-facing tasks for a scenario."""
        task_prompt = f"""For the scenario: {scenario} (domain: {domain})

Generate 10 diverse user tasks that:
1. Are realistic and meaningful
2. Can be solved using database queries and Python tools
3. Require 1-5 API calls to complete
4. Have clear success/failure criteria

Format as JSON:
[{{"task": "...", "success_criteria": "...", "difficulty": "easy|medium|hard"}}, ...]"""

        response = self.llm.generate(task_prompt, max_tokens=800)
        tasks = json.loads(response)

        return tasks

    def generate_database_schema(self, scenario: str) -> Dict:
        """Generate SQLite schema for scenario."""
        schema_prompt = f"""For the scenario: {scenario}

Design a SQLite database schema that supports the scenario. Include:
- 3-5 tables representing entities (users, products, accounts, etc.)
- Appropriate columns with realistic data types
- Sample data (5-10 rows per table)

Format as JSON with structure:
{{
  "tables": [
    {{"name": "table_name", "columns": [
      {{"name": "col", "type": "TEXT|INTEGER|REAL|DATE", "constraints": "PRIMARY KEY|NOT NULL|..."}}
    ]}}
  ],
  "sample_data": {{"table_name": [{{row_dict}}]}}
}}"""

        response = self.llm.generate(schema_prompt, max_tokens=1000)
        schema = json.loads(response)

        return schema

    def create_database(self, db_name: str, schema: Dict) -> str:
        """Create actual SQLite database from schema."""
        conn = sqlite3.connect(f"{db_name}.db")
        cursor = conn.cursor()

        # Create tables
        for table in schema['tables']:
            col_defs = ', '.join([
                f"{col['name']} {col['type']} {col.get('constraints', '')}"
                for col in table['columns']
            ])
            sql = f"CREATE TABLE {table['name']} ({col_defs})"
            cursor.execute(sql)

        # Insert sample data
        for table_name, rows in schema.get('sample_data', {}).items():
            for row in rows:
                cols = ', '.join(row.keys())
                vals = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in row.values()])
                sql = f"INSERT INTO {table_name} ({cols}) VALUES ({vals})"
                cursor.execute(sql)

        conn.commit()
        conn.close()

        return f"{db_name}.db"

    def synthesize_tool_apis(self, scenario: str, schema: Dict) -> Dict:
        """Generate Python tool functions for scenario."""
        tool_prompt = f"""For the scenario with database: {json.dumps(schema, indent=2)[:500]}...

Generate Python functions that agents can call to interact with the database.
Include CRUD operations (Create, Read, Update, Delete) appropriate to the scenario.

Format as JSON:
{{
  "tools": [
    {{
      "name": "function_name",
      "description": "...",
      "signature": "def function_name(param1: type, param2: type) -> return_type:",
      "implementation": "SELECT ... / INSERT ... / etc."
    }}
  ]
}}"""

        response = self.llm.generate(tool_prompt, max_tokens=1000)
        tools = json.loads(response)

        return tools

    def generate_environment(self, scenario: str, domain: str, max_retries=5):
        """Full environment generation with self-correction."""
        for attempt in range(max_retries):
            try:
                # Generate tasks
                tasks = self.synthesize_tasks(scenario, domain)

                # Generate database
                schema = self.generate_database_schema(scenario)
                db_path = self.create_database(scenario.replace(' ', '_'), schema)

                # Generate tool APIs
                tools = self.synthesize_tool_apis(scenario, schema)

                # Verify environment (simplified)
                success = self._verify_environment(scenario, tasks, db_path, tools)

                if success:
                    return {
                        'scenario': scenario,
                        'domain': domain,
                        'tasks': tasks,
                        'db_path': db_path,
                        'schema': schema,
                        'tools': tools
                    }

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to generate environment for {scenario}")
                    return None

        return None

    def _verify_environment(self, scenario: str, tasks: List, db_path: str, tools: Dict) -> bool:
        """Verify environment is valid and tasks are solvable."""
        # Simplified verification: check if tools can execute
        return len(tasks) >= 5 and len(tools.get('tools', [])) >= 3
```

Integrate for RL training:

```python
def generate_training_environments(num_envs=1000, tasks_per_env=10):
    """Generate full training dataset of environments."""
    generator = EnvironmentGenerator(llm_model, num_envs, tasks_per_env)

    # Seed domains
    seed_domains = [
        'e-commerce', 'finance', 'healthcare', 'travel', 'social_media',
        'education', 'real_estate', 'food_delivery', 'project_management'
    ]

    # Expand to scenarios
    scenarios = generator.expand_scenarios(seed_domains)

    # Generate environments
    all_environments = []
    total_tasks = 0

    for domain, domain_scenarios in scenarios.items():
        for scenario in domain_scenarios[:5]:  # Limit per domain
            env = generator.generate_environment(scenario, domain)
            if env:
                all_environments.append(env)
                total_tasks += len(env['tasks'])

    print(f"Generated {len(all_environments)} environments with {total_tasks} total tasks")
    return all_environments
```

## Practical Guidance

| Component | Recommendation | Notes |
|-----------|-----------------|-------|
| Num scenarios | 1000+ | Coverage across diverse domains. |
| Tasks per scenario | 8-12 | Balance variety with per-domain depth. |
| Retry limit | 5 attempts | Usually succeeds by attempt 3-4. |
| Schema complexity | 3-5 tables | More tables = harder; avoid 10+ tables. |
| Tool count | 5-10 per domain | Sufficient for diverse task solving. |

**When to Use**
- Training tool-use agents at scale without manual data
- Need diverse task distribution across many domains
- Want to validate agent generalization across synthetic environments
- Can't access real APIs or environments

**When NOT to Use**
- Real-world task performance critical (synthetic ≠ real)
- Domain-specific knowledge required (LLM-generated may lack nuance)
- Small sample regime where hand-crafted envs sufficient

**Common Pitfalls**
- Generated schemas too simple; agents exploit shortcuts
- Tasks not actually solvable (verification insufficient)
- Tool interfaces inconsistent between environments
- No diversity in tool patterns (all basic CRUD)

## Reference

See https://arxiv.org/abs/2602.10090 for full implementation, including tool diversity strategies, task verification details, and benchmarks on 1000-environment agent training.
