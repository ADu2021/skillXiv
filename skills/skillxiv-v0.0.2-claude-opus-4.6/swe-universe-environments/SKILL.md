---
name: swe-universe-environments
title: "SWE-Universe: Scale Real-World Verifiable Environments to Millions"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02361"
keywords: [Software Engineering Agents, Environment Generation, Scalable Verification, GitHub, Autonomous Building]
description: "Automatically generate executable software engineering environments from GitHub pull requests at million-scale using an autonomous building agent. Detects superficial verification patterns to force genuine code execution testing."
---

# SWE-Universe: Scalable Environment Generation

## Problem
Software engineering (SWE) task datasets are small and limited to curated repositories. Agent training requires diverse, real-world verification environments. Manual environment construction scales poorly.

Current approaches use string-matching verification (grep-based) instead of actually executing code, enabling agents to cheat without solving real problems.

## Core Concept
SWE-Universe uses an autonomous agent (custom Qwen model) to build executable verification scripts from GitHub pull requests. The agent iteratively tests scripts against buggy and fixed states, with in-loop detection of superficial verification patterns.

This creates the largest real-world SWE task collection with genuine executable verification.

## Architecture Overview

- **PR Processing Pipeline**: Separate test and fix patches; filter low-quality PRs
- **Autonomous Building Agent**: Generates verification scripts from PR descriptions
- **Iterative Validation**: Test scripts against both buggy and fixed repository states
- **Hack Detection Module**: Identifies superficial patterns (grep instead of execution)
- **Self-Correction Loop**: Agent diagnoses failures and revises procedures
- **Scaling**: 807K+ instances across 52K repositories, 8 programming languages

## Implementation

### Step 1: Parse and Process GitHub PRs
Extract test and fix patches from pull requests.

```python
def process_github_pr(pr_data):
    """Extract test and fix components from PR."""
    # Parse PR title and description
    title = pr_data['title']
    description = pr_data['body']

    # Separate modified files into fix components
    fix_patch = {}
    test_files = []

    for file_change in pr_data['files']:
        if 'test' in file_change['filename'].lower():
            test_files.append(file_change)
        else:
            fix_patch[file_change['filename']] = file_change['patch']

    # Extract language from repository
    language = infer_language(pr_data['repo'])

    return {
        'title': title,
        'description': description,
        'language': language,
        'fix_patch': fix_patch,
        'test_files': test_files,
        'repo_url': pr_data['repo_url']
    }

def infer_language(repo):
    """Determine primary programming language."""
    file_extensions = count_extensions_in_repo(repo)
    primary_ext = max(file_extensions, key=file_extensions.get)
    return extension_to_language(primary_ext)
```

### Step 2: Autonomous Agent Builds Verification Script
LLM-based agent generates executable test scripts.

```python
def autonomous_build_agent(pr_processed, building_agent, max_iterations=5):
    """Build verification script via iterative agent loop."""
    context = f"""Generate a bash script that verifies this PR:

Title: {pr_processed['title']}
Description: {pr_processed['description']}

Requirements:
1. Script MUST execute actual code, not string matching
2. Test against both buggy (original) and fixed (patched) repository states
3. Output 'PASS' if fix resolves issue, 'FAIL' otherwise
"""

    verification_script = building_agent.generate(context)

    # Iteratively test and refine
    for iteration in range(max_iterations):
        # Clone repo and test script
        buggy_result = run_script_on_repo(verification_script, pr_processed['repo_url'], buggy=True)
        fixed_result = run_script_on_repo(verification_script, pr_processed['repo_url'], buggy=False)

        # Check for hack patterns (superficial verification)
        is_hacking = detect_hack_patterns(verification_script, buggy_result, fixed_result)

        if is_hacking:
            # Force genuine execution
            feedback = f"Script uses superficial verification. Buggy: {buggy_result}, Fixed: {fixed_result}. Must execute actual code."
            verification_script = building_agent.refine(verification_script, feedback)

        elif buggy_result != 'PASS' and fixed_result == 'PASS':
            # Converged to valid verification
            return verification_script, True

    return verification_script, False

def detect_hack_patterns(script, buggy_result, fixed_result):
    """Identify superficial verification (string matching vs execution)."""
    # Patterns indicating string matching
    hack_patterns = [
        'grep -q',
        'grep -c',
        'echo.*|.*grep',
        'if.*grep',
        'test -f.*grep'
    ]

    for pattern in hack_patterns:
        if re.search(pattern, script):
            # Check if results differ despite hack
            if buggy_result == fixed_result:
                return True  # Hack didn't differentiate states

    return False
```

### Step 3: Validate Verification Scripts
Ensure scripts differentiate buggy from fixed states.

```python
def validate_verification_script(script, repo_url, pr_data, language):
    """Confirm script properly verifies the PR fix."""
    # Test on buggy version
    buggy_repo = clone_repo_at_state(repo_url, 'buggy')
    apply_patches_to_repo(buggy_repo, pr_data['test_files'])

    buggy_output = execute_script(script, buggy_repo, language)
    success_on_buggy = (buggy_output.returncode != 0) or ('FAIL' in buggy_output.stdout)

    # Test on fixed version
    fixed_repo = clone_repo_at_state(repo_url, 'fixed')
    apply_patches_to_repo(fixed_repo, pr_data['fix_patch'])
    apply_patches_to_repo(fixed_repo, pr_data['test_files'])

    fixed_output = execute_script(script, fixed_repo, language)
    success_on_fixed = (fixed_output.returncode == 0) and ('PASS' in fixed_output.stdout)

    return success_on_buggy and success_on_fixed
```

### Step 4: Scale Across Repositories
Generate environments for diverse repositories and languages.

```python
def generate_swe_universe(pr_list, building_agent, output_dir, batch_size=100):
    """Generate SWE environments for PR collection."""
    successful_envs = []

    for batch_start in range(0, len(pr_list), batch_size):
        batch = pr_list[batch_start:batch_start + batch_size]

        for pr in batch:
            # Process PR
            pr_proc = process_github_pr(pr)

            # Build and validate script
            script, success = autonomous_build_agent(pr_proc, building_agent)

            if success:
                is_valid = validate_verification_script(
                    script, pr_proc['repo_url'], pr_proc, pr_proc['language']
                )

                if is_valid:
                    # Save environment
                    env_id = f"{pr['repo_name']}__{pr['number']}"
                    save_environment(env_id, script, pr_proc, output_dir)
                    successful_envs.append(env_id)

    return successful_envs
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Building agent iterations | 3-5 | Balance quality with latency |
| Hack detection patterns | Language-specific | Adapt for target languages |
| Success rate threshold | 80%+ | Before/after differentiation |
| Supported languages | Python, JS, Java, C++, etc. | Expand as needed |
| Batch size | 50-200 | Process efficiency |

### When to Use

- Training software engineering agents at scale
- Creating diverse, real-world benchmark environments
- Validating that agents solve problems, not exploiting verification
- Building datasets for code-related tasks (debugging, refactoring)
- Multi-language software engineering benchmarks

### When Not to Use

- Single-repository, specialized benchmarks (manual curation fine)
- Proprietary code repositories (open source only)
- Real-time environment generation (precompute in batches)
- Environments requiring human judgment beyond correctness
- Tasks where string matching is legitimate verification

### Common Pitfalls

1. **Insufficient hack detection**: Agents find creative ways to cheat. Regularly audit generated scripts against expected behavior.
2. **Test quality bias**: Old, unmaintained tests may not validate fixes properly. Filter by test coverage and recency.
3. **Language support gaps**: Build agent needs language-specific knowledge. Validate generation per language.
4. **False failures**: Some PRs may have valid fixes that tests don't fully capture. Include metadata for known exceptions.

## Reference
SWE-Universe: Scale Real-World Verifiable Environments to Millions
https://arxiv.org/abs/2602.02361
