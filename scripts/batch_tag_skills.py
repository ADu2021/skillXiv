#!/usr/bin/env python3
"""Batch-tag all existing skills using an LLM.

Reads each SKILL.md, sends title + description + keywords to an LLM,
and writes back the `tags` field to the SKILL.md frontmatter.

Uses the dynamic tag registry (tags.json) — the LLM selects from existing tags
or proposes new ones at the same level of abstraction.

Requirements:
    pip install openai

Usage:
    # Tag all skills (skips already-tagged ones)
    python scripts/batch_tag_skills.py

    # Force re-tag everything
    python scripts/batch_tag_skills.py --force

    # Dry run (show tags without writing)
    python scripts/batch_tag_skills.py --dry-run

    # Tag a specific source only
    python scripts/batch_tag_skills.py --source skillxiv-v0.0.2-claude-opus-4.6

    # Use a specific model
    python scripts/batch_tag_skills.py --model qwen/qwen3-235b-a22b

Environment:
    OPENROUTER_API_KEY must be set (or pass --api-key directly).
"""
import os
import sys
import json
import re
import time
import argparse
import ast

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "skillxiv.config.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "tags.json")

def load_config(source_filter=None):
    """Load enabled sources from config, optionally filtering to one."""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    sources = [s for s in config['sources'] if s.get('enabled', False)]
    if source_filter:
        sources = [s for s in sources if s['id'] == source_filter]
    return sources

def load_tag_registry():
    """Load tag registry and return the list of tag names."""
    with open(TAGS_PATH, 'r') as f:
        registry = json.load(f)
    return registry

def save_tag_registry(registry):
    """Save updated tag registry."""
    with open(TAGS_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
        f.write('\n')

def parse_frontmatter(content):
    """Parse YAML frontmatter from SKILL.md content.
    Returns (meta_dict, frontmatter_text, body_text)."""
    meta = {}
    if not content.startswith('---'):
        return meta, '', content
    end = content.find('---', 3)
    if end == -1:
        return meta, '', content
    fm_text = content[3:end].strip()
    body = content[end+3:]
    for line in fm_text.split('\n'):
        if ':' in line:
            key = line[:line.index(':')].strip()
            val = line[line.index(':')+1:].strip().strip('"').strip("'")
            meta[key] = val
    return meta, fm_text, body

def has_tags(meta):
    """Check if skill already has tags."""
    tags_val = meta.get('tags', '').strip()
    if not tags_val or tags_val == '[]':
        return False
    return True

def write_tags_to_skill(filepath, tags):
    """Insert or update the tags field in a SKILL.md frontmatter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        return False

    end = content.find('---', 3)
    if end == -1:
        return False

    fm_text = content[3:end]
    body = content[end:]  # includes the closing ---

    # Format tags as inline YAML list
    tags_str = '[' + ', '.join(tags) + ']'

    # Remove existing tags line if present
    fm_lines = fm_text.strip().split('\n')
    fm_lines = [l for l in fm_lines if not l.strip().startswith('tags:')]

    # Insert tags after keywords line, or at end of frontmatter
    insert_idx = len(fm_lines)
    for i, line in enumerate(fm_lines):
        if line.strip().startswith('keywords:'):
            insert_idx = i + 1
            break

    fm_lines.insert(insert_idx, f'tags: {tags_str}')

    new_content = '---\n' + '\n'.join(fm_lines) + '\n' + body

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True

def tag_skill_with_llm(client, model, title, description, keywords, tag_names, allow_new=True):
    """Call OpenRouter API (OpenAI-compatible) to classify a skill into tags."""
    tag_list = '\n'.join(f'- {t}' for t in tag_names)

    new_tag_rules = ""
    if allow_new:
        new_tag_rules = """
5. If and ONLY if no existing tag fits at all, you may propose ONE new tag. But the bar is very high:
   - It must be a broad research AREA that could apply to 50+ papers (e.g. "Robotics", "Audio Processing")
   - NEVER create tags for techniques ("LoRA", "PPO"), specific problems ("Confidence Calibration"),
     subtopics ("Model Merging"), or niche areas ("Federated Learning")
   - Ask yourself: "Would a major ML conference have a track for this?" If not, don't create it.
   - When in doubt, pick the closest existing tag instead of creating a new one.
   - Put new tags in the "new" field. In 99% of cases, "new" should be empty."""
    else:
        new_tag_rules = """
5. Do NOT create any new tags. You MUST select only from the existing registry.
   If nothing fits perfectly, pick the closest match."""

    prompt = f"""You are tagging a research skill for an open skill library.

Current tag registry (you MUST select from these):
{tag_list}

Skill to tag:
- Title: {title}
- Description: {description}
- Keywords: {keywords}

Rules:
1. Select 1-3 tags from the registry. Every skill must get at least one tag.
2. ALWAYS use existing tags. The registry already covers most ML/AI research areas.
3. If a skill spans multiple areas, select 2-3 existing tags rather than creating a new narrow one.
4. Match to the BROADEST applicable tag. For example:
   - A paper about PPO clipping -> "Reinforcement Learning" (not a new "Policy Optimization" tag)
   - A paper about LoRA -> "Large Language Models" (not a new "Parameter Efficient Fine-tuning" tag)
   - A paper about calibration -> "AI Safety" or the most relevant area (not a new "Calibration" tag)
{new_tag_rules}

Return ONLY a JSON object, no other text:
{{"selected": ["Tag Name 1", "Tag Name 2"], "new": []}}"""

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        selected = result.get('selected', [])
        new = result.get('new', []) if allow_new else []
        return selected, new
    return [], []

def main():
    parser = argparse.ArgumentParser(description='Batch-tag skills using LLM')
    parser.add_argument('--force', action='store_true', help='Re-tag already-tagged skills')
    parser.add_argument('--dry-run', action='store_true', help='Show tags without writing')
    parser.add_argument('--source', type=str, default=None, help='Only tag skills from this source')
    parser.add_argument('--limit', type=int, default=None, help='Max skills to tag (for testing)')
    parser.add_argument('--model', type=str, default='qwen/qwen3-235b-a22b', help='Model to use (default: qwen/qwen3-235b-a22b)')
    parser.add_argument('--api-key', type=str, default=None, help='API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--no-new-tags', action='store_true', help='Only select from existing tags, never create new ones')
    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable or --api-key flag is required")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    model = args.model
    print(f"Using model: {model}")

    sources = load_config(args.source)
    registry = load_tag_registry()
    tag_names = [t['name'] for t in registry['tags']]

    total = 0
    tagged = 0
    skipped = 0
    new_tags_added = []

    for source in sources:
        source_path = os.path.join(PROJECT_ROOT, source['path'])
        if not os.path.isdir(source_path):
            print(f"WARNING: source path not found: {source_path}")
            continue

        skill_dirs = sorted(os.listdir(source_path))
        print(f"\nProcessing source: {source['id']} ({len(skill_dirs)} skill folders)")

        for skill_name in skill_dirs:
            if args.limit and tagged >= args.limit:
                break

            skill_path = os.path.join(source_path, skill_name, "SKILL.md")
            if not os.path.isfile(skill_path):
                continue

            total += 1

            with open(skill_path, 'r', encoding='utf-8') as f:
                content = f.read()

            meta, _, _ = parse_frontmatter(content)

            # Skip already-tagged unless --force
            if has_tags(meta) and not args.force:
                skipped += 1
                continue

            title = meta.get('title', meta.get('name', skill_name))
            description = meta.get('description', '')
            keywords = meta.get('keywords', '')

            if not description and not title:
                print(f"  SKIP {skill_name}: no title or description")
                skipped += 1
                continue

            try:
                allow_new = not args.no_new_tags
                selected, new = tag_skill_with_llm(client, model, title, description, keywords, tag_names, allow_new=allow_new)
                all_tags = selected + new

                if not all_tags:
                    print(f"  WARN {skill_name}: LLM returned no tags")
                    skipped += 1
                    continue

                # Register new tags
                for new_tag in new:
                    if new_tag.lower() not in [t.lower() for t in tag_names]:
                        slug = re.sub(r'[^a-z0-9]+', '-', new_tag.lower()).strip('-')
                        registry['tags'].append({"name": new_tag, "slug": slug})
                        tag_names.append(new_tag)
                        new_tags_added.append(new_tag)
                        print(f"  NEW TAG: {new_tag}")

                if args.dry_run:
                    print(f"  {skill_name}: {all_tags}")
                else:
                    write_tags_to_skill(skill_path, all_tags)
                    print(f"  TAGGED {skill_name}: {all_tags}")

                tagged += 1

                # Small delay to avoid rate limiting
                time.sleep(0.3)

            except Exception as e:
                print(f"  ERROR {skill_name}: {e}")

    # Save updated tag registry if new tags were added
    if new_tags_added and not args.dry_run:
        save_tag_registry(registry)
        print(f"\nNew tags added to registry: {new_tags_added}")

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:")
    print(f"  Total skills found: {total}")
    print(f"  Tagged: {tagged}")
    print(f"  Skipped: {skipped}")
    if new_tags_added:
        print(f"  New tags created: {len(new_tags_added)}")

if __name__ == '__main__':
    main()
