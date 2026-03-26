---
name: caid-asynchronous-software-agents
title: "Effective Strategies for Asynchronous Software Engineering Agents"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.21489"
keywords: [Multi-Agent Coordination, Git Workflows, Asynchronous Execution, Software Engineering, Agent Architecture]
description: "Coordinate multiple LLM agents via CAID framework: centralized task delegation, asynchronous execution in isolated git worktrees, structured integration through git merges. Branch-and-merge with worktree isolation yields +26.7% absolute improvement on PaperBench; ranked strategies show structured JSON communication and dependency-aware delegation outperform soft isolation."
---

## Ranked Findings

### 1. Branch-and-Merge with Git Worktree Isolation (Most Effective)
**Strategy**: Each engineer operates in a separate git worktree; changes merge to main through explicit merge+test validation.

**Performance**:
- **PaperBench**: +26.7% absolute improvement over single-agent baseline
- **Commit0**: +14.3% absolute improvement
- **Key advantage**: Physical workspace separation prevents concurrent edit conflicts; enables true parallelism

**Implementation**:
```bash
git worktree add ./worktree-engineer-1 main
cd ./worktree-engineer-1
# Engineer 1 works independently, commits locally
git commit -m "feature A"
cd ..
git merge ./worktree-engineer-1
git test --scope=modified  # Validate before main branch
```

### 2. Dependency-Aware Task Delegation
**Strategy**: Manager constructs dependency graph before assigning work; prioritizes upstream tasks enabling earlier validation.

**Prioritization criteria**:
- Enable earlier test execution on dependent modules
- Expose more evaluation signals sooner
- Lie upstream in dependency chain (less reversal risk)

**Impact**: Reduces integration overhead and catches issues before downstream work begins.

### 3. Structured JSON Communication
**Strategy**: Manager outputs machine-parsable JSON specifications for task assignments (vs. free-form dialogue).

**Specification includes**:
- Task assignments with target scope
- File paths and target functions
- Dependency information
- Execution constraints

**Advantage**: Prevents alignment failures; engineers execute precisely defined work vs. interpreting ambiguous language.

### 4. Self-Verification Before Integration
**Strategy**: Engineers execute tests within their worktree before submitting commits to main.

**Workflow**:
```bash
# Engineer in worktree-engineer-1
git test --scope=local
# If passing, submit to main
git merge --to main
```

**Impact**: Catches conflicts early in engineer's local context; fewer merge conflicts at integration point.

### 5. Soft Isolation (Least Effective)
**Strategy**: Instruction-level constraints without physical workspace separation.

**Result**: Underperforms on open-ended tasks; engineers still encounter concurrent edit conflicts at file level.

## Decision Checklist

- [ ] **Use physical git worktrees**: Not instruction-only isolation—each engineer gets separate workspace
- [ ] **Build dependency graph**: Manager analyzes task dependencies before delegation
- [ ] **Output structured JSON**: Manager provides machine-parsable task specifications, not natural language
- [ ] **Engineer self-tests**: Validate work in local worktree before main integration
- [ ] **Explicit merge gates**: Changes only enter main after tests pass
- [ ] **Monitor parallelism**: Track that increasing engineers improves performance; if performance plateaus, reduce engineer count

## Conditions

### Effective Scenarios
- Long-horizon software engineering (PaperBench, code generation, complex refactoring)
- Tasks decomposable into modular units with identifiable dependencies
- Git repository with test infrastructure
- Teams of 3–8 agents (beyond this, coordination overhead exceeds benefits)

### Environmental Requirements
- Git-aware version control with worktree support
- Local test execution (or fast integration testing)
- Dependency graph constructible from codebase
- Asynchronous execution model (agents work in parallel, integrate periodically)

### When Performance Plateaus
**Key insight**: Increasing engineer count does not monotonically improve performance.

- **Root cause**: Task modularity limits independent parallelism; manager coordination overhead grows
- **Remedy**: Reduce engineer count to match actual task modularity; focus on high-priority modules
- **Optimal range**: 3–6 agents for typical medium-sized projects; diminishing returns beyond 8

## Git Primitives and Coordination Roles

| Git Primitive | Coordination Role |
|---------------|------------------|
| `git worktree add` | Physical workspace isolation—prevents concurrent edits |
| `git commit` | Structured completion signal with message context |
| `git merge` | Explicit output integration—synchronization point |
| `git merge conflict resolution` | Engineer handles conflicts in local context before main |
| `git test` | Executable validation gate—catch errors before integration |

