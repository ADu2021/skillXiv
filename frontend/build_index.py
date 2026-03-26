#!/usr/bin/env python3
"""Extract metadata from all SKILL.md files into a JSON index for the frontend.

Reads skillxiv.config.json to determine which engine sources are enabled.
Only processes skills from enabled sources.

Paths are resolved relative to the project root (parent of frontend/).
"""
import os
import json
import re
import ast
from collections import Counter

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up from frontend/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "skillxiv.config.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "tags.json")
OUTPUT_INDEX = os.path.join(SCRIPT_DIR, "public/skills-index.json")
OUTPUT_SKILLS_DIR = os.path.join(SCRIPT_DIR, "public/skills-data")

def load_config():
    """Load skillxiv.config.json and return list of enabled sources."""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    enabled = [s for s in config['sources'] if s.get('enabled', False)]
    print(f"Config loaded: {len(config['sources'])} sources, {len(enabled)} enabled")
    for s in config['sources']:
        status = "ENABLED" if s.get('enabled') else "disabled"
        print(f"  [{status}] {s['id']}")
    return enabled

def load_tag_registry():
    """Load the tag registry and build an alias lookup map."""
    if not os.path.isfile(TAGS_PATH):
        print("WARNING: tags.json not found, tags will be empty")
        return {}, {}
    with open(TAGS_PATH, 'r') as f:
        registry = json.load(f)
    # Map: canonical name (lowercase) -> { name, slug }
    tag_lookup = {}
    # Map: alias (lowercase) -> canonical name
    alias_lookup = {}
    for t in registry.get('tags', []):
        canonical = t['name'].lower()
        tag_lookup[canonical] = t
        alias_lookup[canonical] = t['name']
        for alias in t.get('aliases', []):
            alias_lookup[alias.lower()] = t['name']
    return tag_lookup, alias_lookup

def parse_tags(raw, alias_lookup):
    """Parse a tags field from frontmatter into a normalized list of tag names.
    Resolves aliases to canonical names using the tag registry."""
    if not raw:
        return []
    raw = raw.strip()
    tags = []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            tags = [str(k).strip() for k in parsed if str(k).strip()]
    except Exception:
        # Handle [A, B, C] format without quotes
        raw = raw.strip("[]")
        tags = [k.strip() for k in raw.split(",") if k.strip()]

    # Resolve aliases to canonical names
    resolved = []
    for tag in tags:
        canonical = alias_lookup.get(tag.lower(), tag)
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved

def parse_skill_md(filepath, alias_lookup):
    """Parse a SKILL.md file and extract metadata + full content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse YAML frontmatter
    meta = {}
    if content.startswith('---'):
        end = content.find('---', 3)
        if end != -1:
            frontmatter = content[3:end].strip()
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key = line[:line.index(':')].strip()
                    val = line[line.index(':')+1:].strip().strip('"').strip("'")
                    meta[key] = val
            body = content[end+3:].strip()
        else:
            body = content
    else:
        body = content

    # Extract paper title: prefer 'title' from frontmatter, fall back to first H1/H2
    paper_title = meta.get('title', '')
    if not paper_title:
        h_match = re.search(r'^#{1,2}\s+(.+)$', body, re.MULTILINE)
        if h_match:
            paper_title = h_match.group(1).strip()

    # Parse tags into a proper array, resolving aliases
    tags = parse_tags(meta.get('tags', ''), alias_lookup)

    return {
        'name': meta.get('name', ''),
        'engine': meta.get('engine', ''),
        'description': meta.get('description', ''),
        'paperTitle': paper_title,
        'url': meta.get('url', ''),
        'keywords': meta.get('keywords', ''),
        'tags': tags,
        'content': body
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)
    os.makedirs(OUTPUT_SKILLS_DIR, exist_ok=True)

    sources = load_config()
    tag_lookup, alias_lookup = load_tag_registry()
    index = []
    tag_counter = Counter()

    for source in sources:
        source_path = os.path.join(PROJECT_ROOT, source['path'])
        if not os.path.isdir(source_path):
            print(f"WARNING: source path not found: {source_path}")
            continue

        skill_dirs = sorted(os.listdir(source_path))
        source_count = 0

        for skill_name in skill_dirs:
            skill_path = os.path.join(source_path, skill_name, "SKILL.md")
            if not os.path.isfile(skill_path):
                continue

            try:
                data = parse_skill_md(skill_path, alias_lookup)

                # Count tags for globalTags
                for tag in data['tags']:
                    tag_counter[tag] += 1

                # Save full content as individual JSON file
                skill_json = {
                    'name': data['name'],
                    'engine': data['engine'],
                    'description': data['description'],
                    'paperTitle': data['paperTitle'],
                    'url': data['url'],
                    'keywords': data['keywords'],
                    'tags': data['tags'],
                    'source': source['id'],
                    'content': data['content']
                }
                with open(os.path.join(OUTPUT_SKILLS_DIR, f"{skill_name}.json"), 'w') as f:
                    json.dump(skill_json, f)

                # Add to index (without full content to keep index small)
                index.append({
                    'id': skill_name,
                    'name': data['name'],
                    'description': data['description'],
                    'paperTitle': data['paperTitle'],
                    'engine': data['engine'],
                    'url': data['url'],
                    'keywords': data['keywords'],
                    'tags': data['tags'],
                    'source': source['id']
                })
                source_count += 1
            except Exception as e:
                print(f"Error processing {skill_name}: {e}")

        print(f"Source '{source['id']}': indexed {source_count} skills")

    # Build globalTags sorted by count (descending)
    global_tags = [
        {"name": name, "slug": tag_lookup[name.lower()]["slug"] if name.lower() in tag_lookup else to_slug(name), "count": count}
        for name, count in tag_counter.most_common()
    ]

    # Write index with globalTags metadata
    output = {
        "skills": index,
        "globalTags": global_tags
    }
    with open(OUTPUT_INDEX, 'w') as f:
        json.dump(output, f)

    # Output sources metadata for frontend source filter
    sources_meta = [
        {'id': s['id'], 'label': s['label']}
        for s in sources
    ]
    sources_meta_path = os.path.join(SCRIPT_DIR, "public/sources-meta.json")
    with open(sources_meta_path, 'w') as f:
        json.dump(sources_meta, f)

    print(f"\nTotal: {len(index)} skills indexed")
    print(f"Tags: {len(global_tags)} unique tags")
    if global_tags:
        top = ', '.join(t['name'] + ' (' + str(t['count']) + ')' for t in global_tags[:10])
        print(f"Top tags: {top}")
    print(f"Sources metadata: {len(sources_meta)} sources written")
    print(f"Index size: {os.path.getsize(OUTPUT_INDEX) / 1024:.1f} KB")
    print(f"Sources metadata: {len(sources_meta)} sources written")

def to_slug(name):
    """Convert a tag name to a URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

if __name__ == '__main__':
    main()
