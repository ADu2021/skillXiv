---
name: agent-skills-security-analysis
title: "Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.10338"
keywords: [agent-security, skill-vetting, vulnerability-detection, threat-analysis, skill-marketplace]
description: "Empirically analyzes 31,132 agent skills to identify 14 distinct vulnerability patterns, finding 26.1% contain security flaws including data exfiltration, privilege escalation, and malicious intent risks that require mandatory vetting."
---

## Overview

Conduct security analysis of modular agent skills before deployment. Use multi-stage detection combining static analysis and LLM-based semantic classification to identify vulnerabilities such as data exfiltration, privilege escalation, prompt injection, and supply chain risks.

## When to Use

- When integrating third-party agent skills into your systems
- For skill marketplaces or skill package management
- To build automated vetting pipelines before skill deployment
- When auditing existing skill collections for vulnerabilities

## When NOT to Use

- For single-use, internally-developed skills
- In fully sandboxed environments with no skill communication
- For read-only skills with no side effects
- In low-risk applications where skill compromise has minimal impact

## Key Technical Components

### SkillScan Detection Framework

Implement multi-stage detection pipeline combining static and semantic analysis.

```python
# Multi-stage detection framework
class SkillScan:
    def __init__(self):
        self.static_analyzer = StaticAnalyzer()
        self.semantic_classifier = SemanticClassifier()

    def scan_skill(self, skill_code, skill_metadata):
        """Comprehensive vulnerability detection"""
        results = {
            "static_findings": self.static_analyzer.analyze(skill_code),
            "semantic_findings": self.semantic_classifier.classify(skill_code),
            "metadata_issues": self.check_metadata(skill_metadata),
            "vulnerability_patterns": []
        }

        # Consolidate findings
        results["vulnerability_patterns"] = self.consolidate_findings(results)
        results["risk_score"] = self.compute_risk_score(results)

        return results

    def consolidate_findings(self, findings):
        """Identify distinct vulnerability patterns"""
        patterns = []
        for finding in findings["static_findings"]:
            pattern = self.classify_pattern(finding)
            patterns.append(pattern)
        return list(set(patterns))
```

### Static Analysis Component

Detect vulnerabilities through code pattern matching.

```python
# Static analysis patterns
class StaticAnalyzer:
    VULNERABILITY_PATTERNS = {
        "credential_exposure": [
            r"api[_]?key\s*=",
            r"password\s*=",
            r"token\s*="
        ],
        "file_access_risk": [
            r"open\s*\(",
            r"read\s*file",
            r"write\s*file",
            r"os\.remove"
        ],
        "network_calls": [
            r"requests\.",
            r"urllib",
            r"socket\.",
            r"send.*http"
        ],
        "process_execution": [
            r"subprocess\.",
            r"os\.system",
            r"popen"
        ]
    }

    def analyze(self, code):
        """Find suspicious patterns in code"""
        findings = []
        for pattern_type, patterns in self.VULNERABILITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    findings.append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "match_count": len(matches),
                        "severity": self.estimate_severity(pattern_type)
                    })
        return findings

    def estimate_severity(self, pattern_type):
        """Assess severity of vulnerability pattern"""
        severity_map = {
            "credential_exposure": "critical",
            "process_execution": "high",
            "network_calls": "medium",
            "file_access_risk": "medium"
        }
        return severity_map.get(pattern_type, "low")
```

### LLM-Based Semantic Classification

Use language models to understand vulnerability intent.

```python
# Semantic vulnerability classification
class SemanticClassifier:
    def classify(self, code):
        """Identify vulnerability intent through semantic analysis"""
        # Classify malicious intent categories
        intent_categories = [
            "data_exfiltration",
            "privilege_escalation",
            "prompt_injection",
            "supply_chain_risk",
            "benign_risky_pattern"
        ]

        classifications = {}
        for category in intent_categories:
            prompt = f"""Analyze this code for {category} intent:
            {code}

            Is there evidence of {category}? (yes/no/unclear)
            Confidence: 0-1
            """
            result = self.llm_classify(prompt)
            classifications[category] = {
                "detected": result["answer"],
                "confidence": result["confidence"]
            }

        return classifications

    def llm_classify(self, prompt):
        """LLM-based semantic analysis"""
        # Would use actual LLM API
        pass
```

### Vulnerability Category Framework

Organize vulnerabilities into actionable categories.

```python
# Vulnerability categories with risk levels
class VulnerabilityCategory:
    CATEGORIES = {
        "data_exfiltration": {
            "description": "Attempt to send data outside system",
            "prevalence": 0.133,  # 13.3% of vulnerable skills
            "examples": ["send_logs", "collect_files", "api_exfil"]
        },
        "privilege_escalation": {
            "description": "Attempt to gain higher permissions",
            "prevalence": 0.118,  # 11.8%
            "examples": ["sudo_access", "admin_check", "permission_bypass"]
        },
        "malicious_intent": {
            "description": "Clear evidence of deliberate harm",
            "prevalence": 0.052,  # 5.2%
            "examples": ["backdoor", "ransomware_pattern", "botnet"]
        },
        "prompt_injection": {
            "description": "Vulnerability to LLM prompt injection",
            "prevalence": 0.045,
            "examples": ["unescaped_input", "eval_user_input"]
        },
        "supply_chain": {
            "description": "Risk to dependency management",
            "prevalence": 0.035,
            "examples": ["typosquatting", "dependency_confusion"]
        }
    }

    @staticmethod
    def get_risk_level(vulnerability):
        """Map vulnerability to risk level"""
        if vulnerability in ["data_exfiltration", "privilege_escalation", "malicious_intent"]:
            return "high"
        elif vulnerability in ["prompt_injection"]:
            return "medium"
        else:
            return "low"
```

### Risk Scoring System

Assign quantitative risk scores to skills.

```python
# Risk scoring
class RiskScorer:
    def compute_score(self, findings):
        """Compute overall risk score 0-1"""
        if not findings:
            return 0.0

        # Weighted scoring
        critical_count = len([f for f in findings if f.get("severity") == "critical"])
        high_count = len([f for f in findings if f.get("severity") == "high"])
        medium_count = len([f for f in findings if f.get("severity") == "medium"])

        score = (
            critical_count * 0.5 +
            high_count * 0.3 +
            medium_count * 0.1
        ) / len(findings)

        return min(1.0, score)

    def requires_vetting(self, risk_score):
        """Determine if skill requires manual review"""
        return risk_score > 0.3
```

### Skill Bundling Analysis

Identify that executable scripts increase vulnerability risk.

```python
# Script bundling risk factor
class BundlingAnalysis:
    def assess_bundling_risk(self, skill):
        """Check if skill includes executable scripts"""
        has_scripts = any(
            script_ext in skill["files"]
            for script_ext in [".py", ".sh", ".js", ".exe"]
        )

        if has_scripts:
            # Scripts are 2.12x more likely to contain vulnerabilities
            return {"has_scripts": True, "risk_multiplier": 2.12}
        else:
            return {"has_scripts": False, "risk_multiplier": 1.0}
```

## Performance Characteristics

- Detection coverage: 31,132 skills analyzed
- Vulnerability prevalence: 26.1% contain at least one vulnerability
- Detection precision: 86.7%
- Detection recall: 82.5%
- 14 distinct vulnerability patterns identified

## Deployment Recommendations

1. **Scan all incoming skills** before adding to marketplace
2. **Use tiered trust levels**: vetted, pending-review, blocked
3. **Implement capability-based permissions** to limit vulnerable skills
4. **Maintain signature database** of known vulnerable patterns
5. **Require security attestation** from skill publishers
6. **Archive and track** all security findings for auditing

## Vetting Checklist

- [ ] No direct credential storage
- [ ] No unescaped user input handling
- [ ] No suspicious network communications
- [ ] No privilege escalation patterns
- [ ] Clear description of permissions needed
- [ ] Author verified/trusted
- [ ] Source code transparency available

## References

- 26.1% of agent skills contain exploitable vulnerabilities
- Executable script bundling increases vulnerability risk 2.12x
- Multi-stage detection (static + semantic) required for comprehensive coverage
- Mandatory vetting essential before production deployment
