---
name: soft-instruction-defense
title: "Soft Instruction De-escalation Defense"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.21057"
keywords: [Security, Prompt Injection, Agent Safety, Sanitization]
description: "Defends tool-augmented LLM agents against prompt injection via iterative input sanitization. Multi-pass inspection detects malicious instructions in untrusted data, remediates them, and re-evaluates until clean or iteration limit reached. Raises attack barrier while maintaining agent usability."
---

# Soft Instruction De-escalation Defense: Protecting Agent Workflows

Tool-augmented agents process untrusted data from APIs, web responses, and user-provided documents. Prompt injection attacks embed malicious instructions within this data to hijack agent behavior.

SIC (Soft Instruction de-escalation) implements iterative sanitization loops that detect and remediate injected instructions while maintaining partial functionality, making attacks harder and more detectable.

## Core Concept

Key insight: **single-pass sanitization misses injections** that later steps expose. Iterative loops catch missed attacks by:
- Inspecting input for instruction-like content
- Remediating detected malicious instructions
- Re-evaluating sanitized output
- Halting if instructions persist after iterations

This raises the cost for attackers while preserving agent capability.

## Architecture Overview

- Instruction pattern detection (imperative verbs, command syntax)
- Remediation strategies (rewriting, masking, removal)
- Iterative sanitization loop with convergence checks
- Failsafe termination when malicious content persists

## Implementation Steps

Implement instruction detection that identifies suspicious patterns in untrusted data. This uses heuristics and ML to flag instruction-like content:

```python
class InstructionDetector:
    def __init__(self):
        # Patterns for common injection attacks
        self.imperative_verbs = [
            'ignore', 'forget', 'override', 'bypass', 'execute',
            'run', 'follow', 'replace', 'change', 'disable'
        ]

        # High-risk instruction patterns
        self.instruction_patterns = [
            r'ignore.*instructions',
            r'(from now on|henceforth)',
            r'new (task|instruction|mode)',
            r'system.*prompt',
        ]

    def detect_instructions(self, text):
        """Flag potentially malicious instructions in text."""
        detections = []

        # Check for imperative verbs
        tokens = text.lower().split()
        for i, token in enumerate(tokens):
            if token in self.imperative_verbs:
                # Check if next tokens form instruction-like pattern
                if i < len(tokens) - 1:
                    detections.append({
                        'type': 'imperative',
                        'position': i,
                        'span': ' '.join(tokens[i:min(i+4, len(tokens))])
                    })

        # Check for regex patterns
        for pattern in self.instruction_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detections.append({
                    'type': 'pattern',
                    'pattern': pattern,
                    'span': match.group()
                })

        return detections
```

Implement remediation strategies that remove or rewrite detected injections. Different strategies apply based on context:

```python
class InstructionRemediator:
    def remediate(self, text, detections, strategy='mask'):
        """Remove or rewrite detected instructions."""
        if strategy == 'mask':
            # Replace instruction content with placeholders
            remediated = text
            for detection in sorted(detections, key=lambda x: x.get('position', 0), reverse=True):
                span = detection['span']
                remediated = remediated.replace(span, '[REMOVED]')
            return remediated

        elif strategy == 'rewrite':
            # Rewrite instructions as non-imperative statements
            remediated = text
            for detection in detections:
                span = detection['span']
                # Convert "ignore X" to "X should be noted"
                rewritten = self._convert_to_passive(span)
                remediated = remediated.replace(span, rewritten)
            return remediated

        elif strategy == 'remove':
            # Delete instruction sentences
            remediated = text
            for detection in sorted(detections, key=lambda x: x.get('position', 0), reverse=True):
                span = detection['span']
                # Remove entire sentence containing instruction
                sentences = remediated.split('.')
                filtered = [s for s in sentences if span not in s]
                remediated = '.'.join(filtered)
            return remediated

        return text

    def _convert_to_passive(self, instruction):
        """Convert imperative to passive voice."""
        # Simple heuristic: add "It is noted that"
        return f"It is noted that {instruction.lower()}"
```

Implement the iterative sanitization loop that applies detection and remediation repeatedly:

```python
class IterativeSanitizer:
    def __init__(self, max_iterations=5):
        self.detector = InstructionDetector()
        self.remediator = InstructionRemediator()
        self.max_iterations = max_iterations

    def sanitize(self, untrusted_input):
        """Iteratively sanitize input until clean or limit reached."""
        current_text = untrusted_input
        iteration = 0
        sanitization_history = []

        while iteration < self.max_iterations:
            # Detect instructions in current text
            detections = self.detector.detect_instructions(current_text)

            if not detections:
                # No instructions found - text is clean
                return {
                    'status': 'clean',
                    'text': current_text,
                    'iterations': iteration,
                    'history': sanitization_history
                }

            # Remediate detected instructions
            remediated = self.remediator.remediate(current_text, detections)

            sanitization_history.append({
                'iteration': iteration,
                'detections_count': len(detections),
                'remediation_strategy': 'mask'
            })

            if remediated == current_text:
                # Remediation didn't change text - likely complex instruction
                return {
                    'status': 'suspicious',
                    'text': current_text,
                    'iterations': iteration,
                    'reason': f'{len(detections)} persistent instructions',
                    'history': sanitization_history
                }

            current_text = remediated
            iteration += 1

        # Exceeded iteration limit with suspicious content
        return {
            'status': 'failed_sanitization',
            'text': current_text,
            'iterations': iteration,
            'reason': f'Max iterations ({self.max_iterations}) reached',
            'history': sanitization_history
        }
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Max iterations | 3-5 (balance safety vs. latency) |
| Detection sensitivity | Medium (avoid false positives) |
| Remediation strategy | Mask or remove (safer than rewrite) |
| Suspicious content handling | Halt agent execution |

**When to use:**
- Tool-augmented agents processing untrusted data
- Systems handling user-provided documents or web scraping
- High-security applications where prompt injection risk is high
- Scenarios with unpredictable external input sources

**When NOT to use:**
- Fully controlled data sources (fine-tuning, internal APIs)
- Real-time systems with strict latency requirements (overhead ≈10-20ms per input)
- Natural language that uses imperatives legitimately

**Common pitfalls:**
- Detection rules too narrow (miss sophisticated injections)
- Remediation breaking legitimate instructions (false positives)
- Insufficient iterations (miss layered attacks)
- Not logging injection attempts (security visibility lost)

Reference: [Soft Instruction De-escalation Defense on arXiv](https://arxiv.org/abs/2510.21057)
