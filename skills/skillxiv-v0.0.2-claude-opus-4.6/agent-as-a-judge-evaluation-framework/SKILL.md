---
name: agent-as-a-judge-evaluation-framework
title: "Agent-as-a-Judge"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05111"
keywords: [Evaluation Systems, Agent Design, LLM Evaluation, Multi-Agent Systems]
description: "Transition from simple LLM-based evaluation to agentic judges that employ planning, tool-augmented verification, multi-agent collaboration, and persistent memory. Survey of sophisticated evaluation paradigms for complex, specialized, and multi-step assessment tasks across diverse domains."
---

## When to Use This Skill
- Complex multi-step evaluation requiring decomposition into subtasks
- Scenarios needing verification beyond model inference (code execution, theorem proving)
- Assessments where tool access and external data are crucial
- Domains requiring domain-expert involvement or specialization
- Evaluation where context persistence across decisions improves judgment

## When NOT to Use This Skill
- Simple classification tasks (LLM-as-Judge is sufficient)
- Real-time latency-critical evaluation scenarios
- Evaluations where tool access creates safety or security risks
- Scenarios with limited computational budget for multi-agent systems

## Problem Summary
Traditional LLM-as-a-Judge evaluation suffers from inherent limitations as assessments become increasingly complex, specialized, and multi-step: models exhibit biases, perform only shallow single-pass reasoning, and cannot verify assessments against real-world observations. For example, code correctness evaluation requires execution verification; mathematical proof assessment benefits from formal verification tools; and nuanced evaluations require consultation with domain experts. These limitations compound when evaluation decisions require persistent context or adaptive strategy adjustment.

## Solution: Agent-as-a-Judge Framework

Evolve from static LLM evaluation to autonomous agents employing structured planning, tool integration, multi-agent collaboration, and persistent memory.

```python
class AgentAsAJudge:
    def __init__(self, llm_backbone, tools):
        self.llm = llm_backbone
        self.tools = tools  # Code executor, search, theorem prover, etc.
        self.evaluation_history = {}
        self.domain_experts = []

    def evaluate_complex_submission(self, submission, rubric, context=None):
        """Multi-step agentic evaluation with tool verification"""

        # Step 1: Planning - decompose evaluation into subtasks
        evaluation_plan = self.create_evaluation_plan(submission, rubric)
        # Output: List of specialized subtasks (code correctness, efficiency, style)

        # Step 2: Distributed Task Execution
        subtask_results = {}
        for subtask in evaluation_plan.subtasks:
            if subtask.requires_code_execution:
                # Delegate to code executor tool
                result = self.tools.execute_code(submission.code, subtask.test_cases)
                subtask_results[subtask.id] = result
            elif subtask.requires_search:
                # Delegate to search tool
                result = self.tools.search(subtask.query)
                subtask_results[subtask.id] = result
            elif subtask.requires_expert_review:
                # Route to domain expert (human-in-the-loop)
                result = self.domain_experts[subtask.expert_domain].review(submission)
                subtask_results[subtask.id] = result

        # Step 3: Multi-Agent Consensus (if multiple experts)
        if len(self.domain_experts) > 1:
            consensus_rating = self.aggregate_expert_opinions(subtask_results)
        else:
            consensus_rating = self.synthesize_results(subtask_results)

        # Step 4: Persistent Memory Update
        self.evaluation_history[submission.id] = {
            "plan": evaluation_plan,
            "results": subtask_results,
            "reasoning_trace": consensus_rating.trace,
            "final_score": consensus_rating.score,
            "confidence": consensus_rating.confidence
        }

        return consensus_rating

    def create_evaluation_plan(self, submission, rubric):
        """Planning agent designs evaluation workflow"""
        prompt = f"""
        Given submission:
        {submission}

        And evaluation rubric:
        {rubric}

        Design an evaluation plan with steps:
        - What aspects require code execution verification?
        - What aspects need external tool use (search, calculation)?
        - What aspects require domain expert input?
        - In what order should evaluations proceed?
        """
        plan_text = self.llm.generate(prompt)
        return parse_evaluation_plan(plan_text)

    def aggregate_expert_opinions(self, expert_results):
        """Multi-agent debate mechanism"""
        debate_prompt = f"""
        Multiple experts have evaluated a submission:
        {format_expert_opinions(expert_results)}

        Please synthesize their opinions into a unified assessment.
        Highlight agreements and discuss disagreements.
        """
        synthesis = self.llm.generate(debate_prompt)
        return parse_synthesis(synthesis)
```

## Three Developmental Stages

**Stage 1: Procedural**
- Fixed workflows with predetermined decision rules
- Example: Template-based rubric following
- Limitation: Cannot adapt to submission-specific characteristics

**Stage 2: Reactive**
- Adaptive routing based on intermediate feedback
- Tool invocation triggered by observed issues
- Example: Code evaluation → runs tests → if failures, analyzes error messages
- Improvement: Responds to evidence but still within pre-defined pathways

**Stage 3: Self-Evolving**
- Autonomous refinement of evaluation rubrics during operation
- Updates criteria based on new submission types
- Example: Discovers new code anti-patterns → adds to evaluation criteria
- Most sophisticated: Continuous improvement of evaluation process

## Five Key Methodological Dimensions

**1. Multi-Agent Collaboration**
- Collective consensus mechanisms (voting, debate)
- Task specialization (separate agents for different aspects)
- Expert diversity reduces individual model biases

**2. Planning**
- Workflow orchestration (execution order matters)
- Rubric discovery (adaptively refine evaluation criteria)
- Strategy adaptation (route to appropriate tools/experts)

**3. Tool Integration**
- Code execution (verify correctness)
- Theorem provers (validate mathematical proofs)
- Search engines (fact-checking, evidence gathering)
- Specialized calculators and validators

**4. Memory & Personalization**
- Intermediate state tracking (evaluation history)
- User context persistence (remember prior interactions)
- Pattern learning (identify common issues)

**5. Optimization Paradigms**
- Training-time (fine-tune judges on feedback)
- Inference-time (adaptive evaluation strategy selection)
- Hybrid (continuous learning + strategic optimization)

## Application Domains

**Implemented in:**
- **Mathematics**: Proof verification, solution correctness
- **Code Analysis**: Execution correctness, efficiency, style
- **Fact-Checking**: Multi-source verification, claim validation
- **Conversation Quality**: Turn-level assessment, coherence evaluation
- **Medicine**: Diagnosis assessment, treatment plan evaluation
- **Law**: Case analysis, precedent relevance
- **Finance**: Risk assessment, decision reasoning
- **Education**: Student understanding, learning progress

## Key Challenges & Mitigation

**Challenge**: Computational Expense
- Mitigation: Cache evaluation results, use faster models for initial screening

**Challenge**: Inference Latency
- Mitigation: Parallelize independent subtask evaluation

**Challenge**: Safety Risks from Tool Access
- Mitigation: Sandboxed execution, permission-based tool access

**Challenge**: Privacy Concerns with Persistent Memory
- Mitigation: Anonymization, access controls, retention policies

## Implementation Recommendations

1. **Start with Procedural Stage**: Template-based evaluation is baseline
2. **Integrate Tools Gradually**: Add code execution, then search, then specialized tools
3. **Implement Multi-Agent Review**: Deploy domain experts for high-stakes decisions
4. **Build Memory Infrastructure**: Log evaluation decisions for pattern analysis
5. **Add Self-Evolution Loop**: Periodically review and refine rubrics

## Survey Coverage
The full paper provides comprehensive taxonomy of 50+ published approaches across these dimensions and application domains.
