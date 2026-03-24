---
name: governed-autonomy-drug-discovery
title: "Mozi: Governed Autonomy for Drug Discovery LLM Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.03655"
keywords: [Agentic Systems, Governed Autonomy, Drug Discovery, Scientific Reasoning, Workflow Management]
description: "Balance LLM flexibility with domain rigor in scientific agents through dual-layer architecture. Enforce role-based access control and artifact-centric state management to prevent hallucinations, while preserving free-form reasoning for lower-risk tasks."
---

# Mozi: Governed Autonomy for Drug Discovery Agents

Fully autonomous LLM agents make costly errors in scientific workflows like drug discovery: they hallucinate about non-existent data, waste expensive computational resources, and propagate early mistakes through multi-stage pipelines. Mozi introduces **governed autonomy**: a dual-layer architecture separating reasoning (flexible) from execution (controlled). The system constrains agent behavior through role-based tool access, explicit state artifacts, and human-in-the-loop checkpoints at high-uncertainty decision boundaries.

The core insight is that scientific agents need dual modes: free-form reasoning for interpretation and planning, but strictly controlled execution for expensive or irreversible actions. This enables faster iteration on safe tasks while maintaining safety guardrails on risky ones.

## Core Concept

Mozi implements three coordinated mechanisms:

1. **Dual-Layer Architecture**:
   - **Control Plane (Layer A)**: Hierarchical supervisor-worker system with role-based access control
   - **Workflow Plane (Layer B)**: Stateful skill graphs encoding canonical domain workflows

2. **Hard-Coded Constraints**: Role-based tool filtering prevents unauthorized access to expensive resources (computational, chemical reagents)

3. **Artifact-Centric State**: Synchronize unstructured reasoning with structured scientific artifacts to detect hallucinations

## Architecture Overview

- **Input**: Scientific task specification (drug discovery objective, constraints)
- **Control Plane**: Supervisor agent assigns roles, monitors execution, enforces constraints
- **Workflow Plane**: Skill graph with defined stages, data contracts, checkpoints
- **State Manager**: Maintains both reasoning traces (flexible) and validated artifacts (strict)
- **Output**: Executed workflow with decision audit trail and human checkpoints

## Implementation Steps

**Step 1: Design workflow with explicit stages and data contracts**

Define scientific workflow as a DAG with strict data validation at each stage.

```python
class WorkflowStage:
    """
    Represents a single stage in a scientific workflow.
    Enforces input/output contracts and checkpoints.
    """

    def __init__(self, name, description, required_inputs, output_schema,
                 cost_estimate, requires_human_approval=False):
        """
        name: stage identifier
        required_inputs: list of required input artifact names
        output_schema: JSON schema for output validation
        cost_estimate: computational cost (for role-based filtering)
        requires_human_approval: flag for high-risk stages
        """
        self.name = name
        self.description = description
        self.required_inputs = required_inputs
        self.output_schema = output_schema
        self.cost_estimate = cost_estimate
        self.requires_human_approval = requires_human_approval

    def validate_inputs(self, artifacts):
        """Verify required inputs exist and conform to schema."""
        for input_name in self.required_inputs:
            if input_name not in artifacts:
                raise ValueError(f"Missing required input: {input_name}")

            # Type validation
            artifact = artifacts[input_name]
            if not self._validate_schema(artifact):
                raise ValueError(f"Input {input_name} violates schema")

        return True

    def _validate_schema(self, artifact):
        """Validate artifact against expected schema."""
        # Implementation: JSON schema validation, type checking, etc.
        return isinstance(artifact, dict) or isinstance(artifact, str)

    def execute(self, agent, artifacts):
        """Execute stage with agent reasoning + constraint enforcement."""
        self.validate_inputs(artifacts)

        # Agent generates action
        reasoning_prompt = f"""
Stage: {self.name}
Description: {self.description}

Available artifacts:
{' '.join(self.required_inputs)}

Generate next action:
"""

        action = agent.generate(reasoning_prompt)

        # Validate output against schema
        output = self._apply_action(action, artifacts)

        if not self._validate_output(output):
            raise ValueError(f"Output violates schema for {self.name}")

        return output

    def _validate_output(self, output):
        """Validate execution output against output_schema."""
        # Implementation: JSON schema validation
        return True

    def _apply_action(self, action, artifacts):
        """Apply agent action and return output."""
        # Implementation-specific execution
        return action

# Define drug discovery workflow
workflow_stages = [
    WorkflowStage(
        name='target_identification',
        description='Identify biological targets for drug',
        required_inputs=['disease_context'],
        output_schema={'type': 'object', 'properties': {'target_id': {'type': 'string'}}},
        cost_estimate=1.0,
        requires_human_approval=False
    ),
    WorkflowStage(
        name='molecular_design',
        description='Design candidate molecules',
        required_inputs=['target_id', 'design_constraints'],
        output_schema={'type': 'array', 'items': {'type': 'string'}},
        cost_estimate=10.0,
        requires_human_approval=False
    ),
    WorkflowStage(
        name='property_prediction',
        description='Predict ADMET properties (expensive)',
        required_inputs=['candidate_molecules'],
        output_schema={'type': 'object'},
        cost_estimate=100.0,
        requires_human_approval=True  # High cost, requires approval
    ),
    WorkflowStage(
        name='synthesis_planning',
        description='Plan synthesis routes',
        required_inputs=['top_candidates'],
        output_schema={'type': 'object'},
        cost_estimate=50.0,
        requires_human_approval=False
    )
]
```

**Step 2: Implement role-based access control**

Define agent roles and restrict tool access based on role permissions.

```python
class AgentRole:
    """Role definition with tool permissions."""

    def __init__(self, role_name, allowed_tools, cost_budget):
        """
        role_name: 'planner', 'designer', 'analyst'
        allowed_tools: list of tool names this role can call
        cost_budget: max computational cost allowed
        """
        self.role_name = role_name
        self.allowed_tools = set(allowed_tools)
        self.cost_budget = cost_budget
        self.cost_used = 0.0

    def can_execute_stage(self, stage):
        """Check if role has permission to execute this stage."""
        stage_cost = stage.cost_estimate

        # Check cost budget
        if self.cost_used + stage_cost > self.cost_budget:
            return False, f"Exceeds cost budget: {stage_cost} > {self.cost_budget - self.cost_used}"

        # Check tool access
        required_tools = self._extract_tools_from_stage(stage)
        for tool in required_tools:
            if tool not in self.allowed_tools:
                return False, f"Tool not allowed for {self.role_name}: {tool}"

        return True, "Permission granted"

    def _extract_tools_from_stage(self, stage):
        """Extract required tools from stage description."""
        # Implementation: parse stage requirements
        return ['search', 'compute']

    def charge_cost(self, stage):
        """Track cumulative cost."""
        self.cost_used += stage.cost_estimate

# Define roles
ROLES = {
    'planner': AgentRole(
        'planner',
        allowed_tools=['search', 'reasoning', 'literature_search'],
        cost_budget=100.0
    ),
    'designer': AgentRole(
        'designer',
        allowed_tools=['molecular_design', 'property_prediction', 'optimization'],
        cost_budget=500.0
    ),
    'analyst': AgentRole(
        'analyst',
        allowed_tools=['data_analysis', 'visualization', 'reporting'],
        cost_budget=50.0
    )
}
```

**Step 3: Implement artifact-centric state management**

Maintain synchronized reasoning traces and validated artifacts to detect hallucinations.

```python
class ArtifactStore:
    """
    Manages both reasoning traces (unstructured, flexible)
    and artifacts (structured, validated).
    """

    def __init__(self):
        self.reasoning_traces = []  # Log of agent reasoning
        self.artifacts = {}  # Validated scientific artifacts
        self.artifact_history = {}  # History of each artifact

    def add_reasoning_trace(self, trace_text):
        """Log agent reasoning (flexible, unstructured)."""
        self.reasoning_traces.append({
            'timestamp': time.time(),
            'content': trace_text
        })

    def add_artifact(self, artifact_name, artifact_data, validation_schema):
        """
        Create validated artifact (structured).
        Raises error if data doesn't conform to schema.
        """
        # Validate
        if not self._validate_against_schema(artifact_data, validation_schema):
            raise ValueError(f"Artifact {artifact_name} violates schema")

        # Store with history
        if artifact_name not in self.artifact_history:
            self.artifact_history[artifact_name] = []

        self.artifacts[artifact_name] = artifact_data
        self.artifact_history[artifact_name].append({
            'timestamp': time.time(),
            'value': artifact_data,
            'trace_reference': len(self.reasoning_traces) - 1
        })

    def detect_hallucination(self):
        """
        Detect hallucinations by checking for reasoning about
        non-existent artifacts.
        """
        artifacts_mentioned = self._extract_artifact_refs_from_traces()

        for mentioned_artifact in artifacts_mentioned:
            if mentioned_artifact not in self.artifacts:
                return True, f"Hallucination detected: {mentioned_artifact} not in store"

        return False, "No hallucinations detected"

    def _extract_artifact_refs_from_traces(self):
        """Extract artifact references from reasoning traces."""
        mentioned = set()

        for trace in self.reasoning_traces[-10:]:  # Check recent traces
            # Simple pattern matching: [artifact_name]
            matches = re.findall(r'\[([a-z_]+)\]', trace['content'].lower())
            mentioned.update(matches)

        return mentioned

    def _validate_against_schema(self, data, schema):
        """Validate data against JSON schema."""
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False

    def get_synchronized_state(self):
        """Return state consistent between reasoning and artifacts."""
        return {
            'reasoning_trace_length': len(self.reasoning_traces),
            'artifact_count': len(self.artifacts),
            'artifacts': self.artifacts,
            'hallucination_check': self.detect_hallucination()
        }
```

**Step 4: Implement supervisor agent with approval checkpoints**

Create supervisor that monitors execution and decides when to escalate for human approval.

```python
class SupervisorAgent:
    """
    Supervisor that orchestrates worker agents and enforces governance.
    """

    def __init__(self, workflow_stages, roles, artifact_store):
        self.workflow = workflow_stages
        self.roles = roles
        self.artifacts = artifact_store
        self.approvals = []

    def execute_workflow(self, task_spec, worker_agent, human_handler):
        """
        Execute workflow with governance:
        1. Assign worker role based on task
        2. Check permissions before each stage
        3. Request human approval for risky stages
        4. Execute and validate
        """
        worker_role = self._assign_role(task_spec)

        for stage in self.workflow:
            # Permission check
            can_execute, reason = worker_role.can_execute_stage(stage)

            if not can_execute:
                print(f"Permission denied: {reason}")
                continue

            # High-risk check
            if stage.requires_human_approval:
                artifacts_state = self.artifacts.get_synchronized_state()

                # Request human approval
                approval = human_handler.request_approval(
                    stage=stage,
                    artifacts=artifacts_state,
                    reasoning_log=self.artifacts.reasoning_traces[-5:]
                )

                if not approval['approved']:
                    print(f"Stage {stage.name} rejected by human reviewer")
                    continue

                self.approvals.append(approval)

            # Execute stage
            try:
                output = stage.execute(worker_agent, self.artifacts.artifacts)

                # Store output as artifact
                self.artifacts.add_artifact(
                    f"{stage.name}_output",
                    output,
                    stage.output_schema
                )

                # Charge cost to role
                worker_role.charge_cost(stage)

            except Exception as e:
                print(f"Stage {stage.name} failed: {e}")
                return False

        return True

    def _assign_role(self, task_spec):
        """Assign worker role based on task requirements."""
        # Simple heuristic: molecule design → designer, planning → planner
        if 'molecule' in task_spec.lower() or 'design' in task_spec.lower():
            return self.roles['designer']
        elif 'plan' in task_spec.lower():
            return self.roles['planner']
        else:
            return self.roles['analyst']
```

**Step 5: Integration with LLM agents**

Connect governance components to LLM-based agent execution.

```python
def run_drug_discovery_agent(task_spec, human_handler=None):
    """
    Execute drug discovery workflow with governed autonomy.
    """
    # Initialize components
    artifact_store = ArtifactStore()
    supervisor = SupervisorAgent(
        workflow_stages=workflow_stages,
        roles=ROLES,
        artifact_store=artifact_store
    )

    worker_agent = LLMAgent()

    # Execute with governance
    success = supervisor.execute_workflow(
        task_spec,
        worker_agent,
        human_handler or DummyHumanHandler()
    )

    return {
        'success': success,
        'artifacts': artifact_store.artifacts,
        'reasoning_trace': artifact_store.reasoning_traces,
        'approvals': supervisor.approvals,
        'hallucination_check': artifact_store.detect_hallucination()
    }
```

## Practical Guidance

**Hyperparameter Selection:**
- **Cost budgets**: Scale with agent capability (advanced agents: 10x budget)
- **Human approval threshold**: Trigger on cost > 50, or novel molecule types
- **Hallucination detection threshold**: Flag any mention of non-existent artifact
- **Stage timeout**: 5-10 minutes per stage; escalate on timeout

**When to Use:**
- High-stakes scientific workflows (drug discovery, materials science)
- Multi-stage pipelines where early errors compound
- Settings with expensive computational resources
- Scenarios requiring audit trails and human oversight

**When NOT to Use:**
- Low-risk tasks (question answering, text generation)
- Real-time systems where human approval latency is unacceptable
- Single-stage workflows
- Domains where all reasoning can be safely automated

**Common Pitfalls:**
- **Over-restrictive permissions**: Too many approval barriers slow iteration. Calibrate thresholds based on error rates.
- **Artifact validation too strict**: False rejections block valid workflows. Validate schema pragmatically.
- **Hallucination detection false positives**: Filter mentions that are clearly hypothetical. Use allowlist of known hallucination patterns.
- **Role exhaustion**: If agents quickly deplete cost budget, re-allocate or add higher-capacity roles.

## Reference

arXiv: https://arxiv.org/abs/2603.03655
