---
name: toucan-mcp-tool-agentic-dataset
title: "TOUCAN: Synthesizing Tool-Agentic Data from MCP Environments"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.01179
keywords: [MCP, agent-training, dataset-synthesis, tool-use, LLM-agents]
description: "Generate 1.5M realistic agent training trajectories from 495 real-world MCP servers without human annotation. Use to build tool-agentic training datasets with authentic tool execution, multi-turn conversations, and error handling."
---

# TOUCAN: Synthesizing Tool-Agentic Data from MCP Environments

TOUCAN introduces the largest open-source tool-agentic dataset with 1.5M trajectories generated from real-world MCP (Model Context Protocol) servers. By leveraging authentic MCP execution rather than simulation, the dataset enables LLM agent training with realistic tool behavior, error patterns, and complex workflows.

## Core Architecture

- **Real MCP execution**: 495 servers with actual tool behavior (no simulation)
- **1.5M trajectories**: Multi-turn conversations with parallel tool calls
- **Quality filtering**: Six-dimension validation (relevance, correctness, completeness, diversity, executability, safety)
- **Diverse interactions**: Error recovery, irrelevant queries, multi-step reasoning
- **State-of-the-art results**: Fine-tuned 32B models outperform GPT-4.5-Preview on multi-turn benchmarks

## Implementation Steps

Setup MCP server discovery and trajectory synthesis:

```python
# Initialize TOUCAN dataset generation pipeline
from toucan import MCPDatasetSynthesizer, QualityFilter

synthesizer = MCPDatasetSynthesizer(
    num_servers=495,
    trajectory_budget=1_500_000,
    parallel_workers=128,
    server_timeout=30
)

# Discover available MCP servers
servers = synthesizer.discover_servers(
    exclude_api_key_dependent=True,  # avoid credential exposure
    max_per_category=50
)

# Configure quality filtering (6-dimension validation)
quality_filter = QualityFilter(
    dimensions={
        "relevance": 0.85,
        "correctness": 0.90,
        "completeness": 0.80,
        "diversity": 0.75,
        "executability": 0.95,
        "safety": 1.0
    }
)
```

Generate trajectories with error handling and quality assurance:

```python
# Generate realistic agent trajectories from MCP servers
trajectories = []

for server_config in servers:
    # Stage 1: Instruction generation via LLM
    instructions = synthesizer.generate_instructions(
        server=server_config,
        num_instructions_per_server=3000,
        temperature=0.8
    )

    # Stage 2: Trajectory rollout with actual MCP execution
    for instruction in instructions:
        trajectory = synthesizer.rollout(
            instruction=instruction,
            server=server_config,
            max_turns=15,
            allow_parallel_calls=True,
            error_handling="recovery"
        )

        # Stage 3: LLM-as-judge validation
        score = quality_filter.evaluate(trajectory)

        if score >= 0.80:  # aggregated quality threshold
            trajectories.append(trajectory)

        # Stage 4: Safety screening
        if not quality_filter.check_safety(trajectory):
            continue

        # Stage 5: Deduplication
        if not synthesizer.is_duplicate(trajectory, existing=trajectories):
            trajectories.append(trajectory)
```

Fine-tune agents on TOUCAN:

```python
# Train LLM agents using TOUCAN trajectories
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("Qwen2.5-32B")

training_args = TrainingArguments(
    output_dir="./agent-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    max_seq_length=8192
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=toucan_dataset,  # 1.5M trajectories
    data_collator=agent_collator
)

trainer.train()
```

## Practical Guidance

**When to use TOUCAN:**
- Training LLM agents for MCP-compatible tool use
- Building production systems requiring robust multi-turn handling
- Rapid prototyping with high-quality real-world data
- Fine-tuning on diverse tool ecosystems (495 servers across 50+ categories)

**When NOT to use:**
- Training for entirely new tool categories not covered in MCP ecosystem
- Scenarios requiring credentials or API keys (TOUCAN excludes these)
- Specialized domains with unique interaction patterns (e.g., medical devices)
- June 2025 data is historical; newer server capabilities not included

**Dataset statistics:**
- **1.5M trajectories**: Mix of single-turn and multi-turn (15 turns average)
- **Average trajectory length**: 8-12 turns with 2-4 tool calls per turn
- **Error scenarios**: 25% of trajectories include error handling/recovery
- **Parallel execution**: 30% include simultaneous multi-tool calls

**Training hyperparameters:**
- **Learning rate (5e-5)**: Adjust down to 2e-5 for 70B models; up to 1e-4 for 7B
- **Batch size (4)**: Increase to 8-16 if GPU memory permits (8192 context length demanding)
- **Max sequence length (8192)**: Necessary for multi-turn; reduce to 4096 for faster training
- **Epochs (3)**: Increase to 5 for smaller datasets; single epoch sufficient for 1.5M trajectories

## Quality Metrics

TOUCAN achieves state-of-the-art results on established benchmarks:
- **BFCL V3**: Fine-tuned 32B outperforms GPT-4.5-Preview on multi-turn tool calls
- **MCP-Universe**: Comprehensive coverage across diverse server categories
- **Error recovery**: Models trained on TOUCAN handle failures more gracefully

## Data Composition

- **Diverse servers**: 495 real MCP implementations
- **Balanced categories**: Even distribution across tool types (retrieval, manipulation, analysis)
- **Real execution**: No synthetic simplifications or simulation artifacts
- **Safety-filtered**: All dangerous operations removed; credentials excluded

## References

Relates to dataset synthesis for agent training and MCP ecosystem standardization.
