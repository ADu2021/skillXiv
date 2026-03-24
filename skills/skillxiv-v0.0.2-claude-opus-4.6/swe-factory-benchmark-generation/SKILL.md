---
name: swe-factory-benchmark-generation
title: "SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.10954"
keywords: [automated dataset generation, GitHub issues, software engineering, benchmarks, fail2pass validation]
description: "Automate construction of GitHub issue resolution benchmarks via multi-agent LLM coordination, binary file recovery, and exit-code-based validation, achieving 337 valid instances at $0.047 per instance."
---

# SWE-Factory: Automated Benchmark Generation

## Core Concept

SWE-Factory fully automates GitHub issue resolution dataset construction through a multi-agent LLM system that coordinates environment setup, binary file recovery, and fail2pass validation. Instead of manual inspection and custom log parsing, the system uses exit-code-based testing (standardized Unix return codes), successfully generating 337 valid benchmarks from 671 issues across Python, Java, JavaScript, and TypeScript at $0.047 per instance and 96% quality.

## Architecture Overview

- **SWE-Builder Multi-Agent System**: Four collaborative agents (Repository Explorer, Environment Manager, Test Manager, Test Analyst) with shared Environment Memory Pool for configuration reuse
- **Binary File Recovery**: Automatically downloads missing test binaries and removes incomplete hunks from patches
- **Exit-Code-Based Validation**: Standardized fail2pass testing via Unix exit codes ($rc) instead of manual log parsing, achieving F1=0.99
- **Environment Memory Pool**: Caches successful Docker configurations from similar repository versions, reducing redundant iterations
- **Language Support**: Python, Java, JavaScript, TypeScript with unified testing interface

## Implementation

### Step 1: Binary File Recovery System

```python
import os
import subprocess
from pathlib import Path

class BinaryFileRecoverer:
    """
    Automatically downloads missing binary test files from repositories
    and removes incomplete binary hunks from patches.
    """

    def __init__(self, repo_path, repo_url):
        self.repo_path = repo_path
        self.repo_url = repo_url

    def identify_missing_binaries(self, patch_content):
        """
        Parse patch file to identify missing binary files.
        Binary patches show "Binary files ... differ" marker.
        """
        missing_binaries = []
        lines = patch_content.split('\n')

        for i, line in enumerate(lines):
            if 'Binary files' in line and 'differ' in line:
                # Extract filename from patch header
                for j in range(i-1, max(0, i-10), -1):
                    if lines[j].startswith('+++'):
                        filename = lines[j].split('\t')[0].replace('+++ b/', '')
                        missing_binaries.append(filename)
                        break

        return missing_binaries

    def download_binaries_from_repo(self, filenames):
        """
        Download binary files from original repository.
        Handles different VCS (GitHub, GitLab, etc.)
        """
        downloaded = []

        for filename in filenames:
            # Construct raw file URL
            raw_url = self.repo_url.replace('github.com', 'raw.githubusercontent.com')
            raw_url = f"{raw_url}/HEAD/{filename}"

            try:
                result = subprocess.run(
                    ['curl', '-L', '-o', filename, raw_url],
                    cwd=self.repo_path,
                    timeout=30,
                    capture_output=True
                )

                if result.returncode == 0 and os.path.exists(os.path.join(self.repo_path, filename)):
                    downloaded.append(filename)
                    print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

        return downloaded

    def clean_patch_binaries(self, patch_content):
        """
        Remove incomplete binary hunks from patches.
        Keeps only well-formed binary sections.
        """
        lines = patch_content.split('\n')
        cleaned = []
        skip_binary_section = False

        for i, line in enumerate(lines):
            if 'Binary files' in line and 'differ' in line:
                skip_binary_section = True
                cleaned.append(line)  # Keep the "differs" marker
            elif line.startswith('+++') or line.startswith('---'):
                skip_binary_section = False
                cleaned.append(line)
            elif not skip_binary_section or line.startswith(('GIT binary patch', 'literal')):
                cleaned.append(line)

        return '\n'.join(cleaned)
```

### Step 2: Multi-Agent Environment Builder

```python
class SWEBuilderMultiAgent:
    """
    Orchestrates four collaborative agents to construct evaluation environment.
    Uses shared memory pool for configuration reuse.
    """

    def __init__(self, repo_path, model_name='gpt-4-mini'):
        self.repo_path = repo_path
        self.model = model_name
        self.env_memory_pool = {}

    def repository_explorer(self):
        """Agent 1: Collects environment dependencies and test commands."""
        exploration_prompt = f"""
        Analyze repository at {self.repo_path}:
        1. Identify all dependency files (requirements.txt, pom.xml, package.json, etc.)
        2. Extract all test commands (pytest, maven, npm test, etc.)
        3. List any configuration files (setup.cfg, tsconfig.json, etc.)
        4. Identify language versions needed (Python 3.9+, Java 11+, etc.)

        Return structured JSON with sections: dependencies, test_commands, configs, versions
        """
        return self._llm_call(exploration_prompt)

    def environment_manager(self, exploration_result):
        """Agent 2: Generates Dockerfile configuration."""
        docker_prompt = f"""
        Based on repository analysis:
        {exploration_result}

        Generate complete Dockerfile that:
        1. Installs base image matching language version requirements
        2. Installs all dependencies in correct order
        3. Sets environment variables
        4. Copies repository code
        5. Runs any setup scripts

        Output valid Dockerfile content.
        """

        docker_content = self._llm_call(docker_prompt)

        # Check memory pool for similar configurations
        similar = self._find_similar_config(exploration_result)
        if similar:
            print(f"Found similar configuration: {similar}")
            docker_content = self._merge_configs(docker_content, similar)

        return docker_content

    def test_manager(self, exploration_result):
        """Agent 3: Creates executable test scripts with exit code reporting."""
        test_prompt = f"""
        Based on identified test commands:
        {exploration_result['test_commands']}

        Create a master test script that:
        1. Sets up test environment
        2. Runs each test command sequentially
        3. Captures exit codes
        4. Appends standardized marker: OMNIGRIL_EXIT_CODE=$rc
        5. Returns appropriate exit code (0=pass, 1=fail)

        Language: {self._detect_language()}
        """

        test_script = self._llm_call(test_prompt)

        return test_script

    def test_analyst(self, docker_content, test_script):
        """Agent 4: Validates environment and orchestrates refinement."""

        # Try to build Docker image
        build_success = self._build_docker(docker_content)

        if not build_success:
            # Iteratively refine via LLM
            refinement_prompt = f"""
            Docker build failed. Previous Dockerfile:
            {docker_content}

            Common issues:
            - Missing base image layer
            - Dependency installation order
            - Working directory not set

            Provide corrected Dockerfile.
            """

            docker_content = self._llm_call(refinement_prompt)
            return self.test_analyst(docker_content, test_script)

        # Build passed; store in memory pool
        self.env_memory_pool[hash(docker_content)] = {
            'dockerfile': docker_content,
            'test_script': test_script,
            'success': True
        }

        return {'dockerfile': docker_content, 'test_script': test_script}

    def _find_similar_config(self, exploration_result):
        """Find similar configurations in memory pool."""
        repo_hash = hash(exploration_result.get('dependencies', ''))

        # Simple similarity: same language/version
        for stored_hash, config in self.env_memory_pool.items():
            if config['success']:
                return config

        return None

    def _merge_configs(self, new_config, similar_config):
        """Merge new config with similar successful one."""
        # Simplified: prefer new, fallback to similar
        return new_config

    def _llm_call(self, prompt):
        """Call LLM with prompt."""
        # Simplified placeholder
        return f"Generated from: {prompt[:50]}..."

    def _detect_language(self):
        """Detect primary language of repository."""
        if os.path.exists(os.path.join(self.repo_path, 'requirements.txt')):
            return 'python'
        elif os.path.exists(os.path.join(self.repo_path, 'pom.xml')):
            return 'java'
        elif os.path.exists(os.path.join(self.repo_path, 'package.json')):
            return 'javascript'
        return 'unknown'

    def _build_docker(self, dockerfile_content):
        """Test build Docker image."""
        # Simplified: return True for now
        return True
```

### Step 3: Exit-Code-Based Fail2Pass Validation

```python
import re
import subprocess

class ExitCodeValidator:
    """
    Standardized fail2pass validation using Unix exit codes.
    Eliminates custom log parsing; achieves 100% precision, 98.3% recall.
    """

    def create_test_wrapper(self, original_test_command, language):
        """
        Wraps test command to append standardized exit code marker.
        Format: OMNIGRIL_EXIT_CODE=$rc
        """

        if language == 'python':
            wrapper = f"""#!/bin/bash
set -e
bash -c '{original_test_command}'
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
"""
        elif language == 'java':
            wrapper = f"""#!/bin/bash
set -e
{original_test_command}
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
"""
        else:
            wrapper = f"""#!/bin/bash
set -e
{original_test_command}
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
"""

        return wrapper

    def extract_exit_code(self, test_output):
        """
        Parse test output and extract exit code marker.
        Returns (exit_code, is_valid).
        """

        pattern = r'OMNIGRIL_EXIT_CODE=(\d+)'
        match = re.search(pattern, test_output)

        if match:
            exit_code = int(match.group(1))
            return exit_code, True

        return None, False

    def validate_fail2pass(self, buggy_version_output, fixed_version_output):
        """
        Verify that test failed on buggy version and passed on fixed version.
        Returns (is_valid, buggy_code, fixed_code).
        """

        buggy_code, buggy_valid = self.extract_exit_code(buggy_version_output)
        fixed_code, fixed_valid = self.extract_exit_code(fixed_version_output)

        if not (buggy_valid and fixed_valid):
            return False, None, None

        # Fail2pass: buggy version failed (non-zero), fixed version passed (zero)
        is_fail2pass = (buggy_code != 0) and (fixed_code == 0)

        return is_fail2pass, buggy_code, fixed_code
```

## Practical Guidance

**Dataset Construction Workflow**:
1. Collect GitHub issues (use GitHub API with issue labels like "bug", "fix")
2. Extract patches from issue comments or linked PRs
3. Run Binary Recovery to handle test fixtures
4. Execute SWE-Builder multi-agent pipeline to construct environment
5. Validate with Exit-Code-Based testing

**Cost Optimization**:
- Average: $0.047 per valid instance with GPT-4 mini
- Batch processing: Process 50+ issues in parallel to amortize LLM calls
- Memory pool: Reuse successful configurations reduces total cost by 30%

**Quality Assurance**:
- Validate exit codes independently with 2+ test runners
- Manual review of first 50 instances to verify quality
- Monitor F1 score (0.99 target) on exit code extraction

**Supported Languages**:
- Python: pytest, unittest, nose
- Java: Maven, Gradle
- JavaScript/TypeScript: Jest, Mocha, npm test
- Others: Extensible via custom test runners

## Reference

- Unix exit codes: Standard convention (0=success, non-zero=failure) across all languages
- Docker layer caching: Significantly speeds up repeated builds for similar configurations
- Environment Memory Pool: Inspired by case-based reasoning; dramatically improves efficiency
