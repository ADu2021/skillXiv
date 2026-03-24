# SkillXiv

**[skillxiv.org](https://skillxiv.org)**

SkillXiv is an open repository of agent skills extracted from arXiv papers. Each skill distills a paper's core technique into a structured, actionable format that any AI agent can use out-of-the-box — so practitioners can apply cutting-edge research without reading the full paper.

The project has three components:

1. **Open skill repository** — a growing collection of extracted skills from ML/AI papers, each following a standardized SKILL.md format with YAML metadata, implementation guidance, and code.
2. **Paper-to-skill engines** — automated pipelines that convert arXiv papers into agent skills. The architecture supports multiple engines, and we welcome new ones.
3. **Web frontend** — a browsable, searchable interface for discovering skills, inspired by [alphaXiv](https://www.alphaxiv.org/).

## Repository structure

```
skills/
├── skillxiv-v0.0.2-claude-opus-4.6/   # Skills extracted by the v0.0.2 engine
├── skillxiv-v0.0.1-claude-opus-4.6/   # Previous engine version (archived)
├── skillxiv-v0.0.0-claude-opus-4.6/   # Initial engine version (archived)
├── human/                              # Human-written skills
└── paper2skill/                        # The paper2skill extraction engine

frontend/                               # React/Vite static site (skillxiv.org)
skillxiv.config.json                    # Configuration for enabled skill sources
```

## Current stats

- **1,100+ skills** extracted from arXiv papers
- Covering ML, NLP, computer vision, reinforcement learning, systems, and more
- All skills licensed under MIT

## Getting started

Browse skills at [skillxiv.org](https://skillxiv.org), or clone this repo and explore locally:

```bash
git clone https://github.com/adu2021/skillxiv.git
cd skillxiv/skills/skillxiv-v0.0.2-claude-opus-4.6/
ls  # each folder is a skill
cat sparse-attention-training/SKILL.md  # read a skill
```

To build the frontend locally:

```bash
cd frontend
npm install
npm run build:index   # generates skills-index.json from skill folders
npm run dev           # starts dev server
```

## Contributing

We are actively looking for contributions in two areas:

1. **Extracted agent skills** — both human-written and machine-extracted skills are welcome. If you've read a paper and distilled its technique into a SKILL.md, we want it. If you've built a tool that does this automatically, even better.

2. **Better paper-to-skill engines** — improved extraction algorithms, higher-quality outputs, or entirely new engine approaches. Contributions can be new engines, improvements to existing ones, or evaluation results comparing engine quality.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guidelines on skill format, folder structure, and how to submit.

## License

All extracted skills are released under the MIT License.
