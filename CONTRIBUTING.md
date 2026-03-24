# Contributing to SkillXiv

Thanks for your interest in contributing! SkillXiv grows through community contributions — whether you're writing skills by hand, building extraction engines, or improving existing ones.

## What we're looking for

1. **New agent skills** extracted from arXiv papers (human-written or machine-extracted)
2. **New or improved paper-to-skill engines** (algorithms, prompts, pipelines, or evaluation methods)

## Skill folder structure

All skills live under `skills/`, organized by source:

```
skills/
├── human/                              # Human-written skills
├── skillxiv-v0.0.2-claude-opus-4.6/   # Skills from the v0.0.2 automated engine
└── <your-engine-name>/                 # Skills from your engine
```

**Routing rules:**

- If you wrote the skill by hand, it goes in `skills/human/`.
- If a machine/engine extracted it, it goes in `skills/<engine-name>/`, where `<engine-name>` identifies the extraction engine (e.g., `my-gpt4-extractor`, `skillxiv-v0.0.2-claude-opus-4.6`).

Each skill is a folder named with a descriptive kebab-case identifier:

```
skills/human/
└── adaptive-importance-ratio-clipping/
    ├── SKILL.md          # Required — the main skill file
    ├── scripts/          # Optional — executable code files
    │   └── core.py
    └── references/       # Optional — supplementary materials
        └── appendix.md
```

## SKILL.md format

Every SKILL.md must begin with a YAML frontmatter block. The full specification is defined in the [paper2skill engine](skills/paper2skill/SKILL.md) — please read it before contributing. Below is a summary of the required fields.

### Required YAML frontmatter

```yaml
---
name: meaningful-kebab-case-name
title: "Full Paper Title as Published"
version: 0.0.2
engine: human  # or your engine identifier
license: MIT
url: "https://arxiv.org/abs/XXXX.XXXXX"
keywords: [Keyword One, Keyword Two, Keyword Three]
description: "Plain-text description of what the skill does and when to use it. Under 1024 characters."
---
```

### Field requirements

| Field | Rules |
|-------|-------|
| `name` | Descriptive kebab-case. Must communicate the skill's purpose at a glance. **Never** use raw arXiv IDs like `2505-00212`. |
| `title` | The exact paper title as published on arXiv. |
| `version` | Skill spec version. Currently `0.0.2`. |
| `engine` | `human` for hand-written skills, or the engine identifier (e.g., `skillxiv-v0.0.2-claude-opus-4.6`). |
| `license` | Always `MIT`. |
| `url` | A verified, working arXiv URL (`https://arxiv.org/abs/XXXX.XXXXX`). No placeholders. |
| `keywords` | 5-10 keywords as an inline YAML list: `[A, B, C]`. **Never** use YAML block list syntax (`- item`). Title Case. |
| `description` | Under 1024 characters. Plain text only — no angle brackets (`<>`), no YAML block scalars (`>` or `\|`). Must describe both what the skill does and when to use it. Double-quoted string on one line. |

### SKILL.md body structure

After the frontmatter, the body should follow this general outline:

1. **Title and introduction** — one-line outcome statement, then 1-2 paragraphs on the problem in practical terms.
2. **Core concept** — the key insight in plain language.
3. **Architecture overview** — components and data flow described in bullet points (no ASCII art diagrams).
4. **Implementation** — numbered steps, each with a brief explanation and a language-labeled code block (10-40 lines showing the novel part).
5. **Practical guidance** — hyperparameter table, "when to use", "when NOT to use", common pitfalls.
6. **Reference** — paper title, arXiv link, and code repo URL if available.

See any skill in `skills/skillxiv-v0.0.2-claude-opus-4.6/` for a concrete example, and refer to the [paper2skill SKILL.md](skills/paper2skill/SKILL.md) for the full specification including code handling rules and quality checks.

## Contributing a skill

1. Fork this repository.
2. Create your skill folder under the appropriate source directory:
   - `skills/human/<your-skill-name>/` for hand-written skills
   - `skills/<your-engine>/` for engine-extracted skills (create the engine folder if new)
3. Add a `SKILL.md` with valid YAML frontmatter and body content following the format above.
4. Optionally add `scripts/` and `references/` subdirectories.
5. Open a pull request. In the PR description, include:
   - The arXiv paper URL
   - Whether the skill was human-written or machine-extracted (and which engine)
   - A brief note on what makes the paper's technique useful as a skill

## Contributing an engine

If you've built a paper-to-skill extraction engine:

1. Create a folder under `skills/` for your engine's outputs (e.g., `skills/my-engine-v1/`).
2. Include at least 5 sample skills extracted by your engine so reviewers can assess quality.
3. In your PR description, explain:
   - How the engine works (high-level)
   - What model(s) or tools it uses
   - How it compares to existing engines (quality, speed, coverage)
4. Optionally, contribute the engine code itself under `skills/paper2skill/` or a separate directory.

## Quality guidelines

Before submitting, check that your skill passes these tests:

- **Standalone test:** Can someone understand what to implement by reading only the skill (not the paper)?
- **Code review:** Do the code blocks run? Are they language-labeled? Are undefined variables avoided?
- **Trigger test:** Does the description cover at least 5 ways someone might search for this technique?
- **Depth check:** Does the skill go beyond a 2-sentence summary?
- **Practitioner test:** Does it include implementation steps, hyperparameters, and failure modes?

## Code of conduct

Be respectful, constructive, and focused on making research more accessible. We welcome contributors of all experience levels.
