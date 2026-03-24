---
name: prompt-orchestration-markup
title: "Prompt Orchestration Markup Language (POML)"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.13948
keywords: [prompt-engineering, markup-language, structured-prompts, templating, developer-tools]
description: "Use component-based markup with CSS-like styling to structure complex prompts, integrate diverse data types, and separate content from formatting for maintainable, version-control-friendly LLM applications."
---

# Prompt Orchestration Markup Language (POML)

## Core Concept

Modern LLM applications require increasingly complex prompts with roles, instructions, examples, and diverse data (documents, tables, images). Traditional string concatenation is fragile, format-sensitive, and hard to version control.

POML provides structured markup for composing prompts declaratively, with CSS-like styling to separate content from formatting. This makes prompts maintainable, reusable, and compatible with standard developer tools.

## Architecture Overview

- **Component-Based Structure**: Define roles, tasks, examples as reusable components
- **CSS-Like Styling**: Separate content formatting from structure
- **Data Integration**: First-class support for documents, tables, images
- **Templating System**: Dynamic prompts with variables and conditionals
- **Developer Tooling**: IDE support, version control, SDKs
- **Format Normalization**: Reduces sensitivity to whitespace and formatting

## Implementation Steps

### 1. Define POML Schema and Parser

```python
import xml.etree.ElementTree as ET
from typing import Dict, List, Any

class POMLComponent:
    """Base component in POML"""
    def __init__(self, component_type: str, attributes: Dict[str, str] = None):
        self.type = component_type
        self.attributes = attributes or {}
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def to_text(self, styling_rules: Dict[str, Dict] = None) -> str:
        """Convert component to text using styling rules"""
        raise NotImplementedError

class POMLParser:
    """Parse POML markup"""
    def __init__(self):
        self.styling_rules = {}

    def parse(self, poml_string: str) -> POMLComponent:
        """Parse POML XML string"""
        root = ET.fromstring(poml_string)
        return self._parse_element(root)

    def _parse_element(self, element) -> POMLComponent:
        """Recursively parse XML element"""
        component = POMLComponent(
            element.tag,
            dict(element.attrib)
        )

        # Parse children
        for child_elem in element:
            component.add_child(self._parse_element(child_elem))

        # Text content
        if element.text and element.text.strip():
            component.text = element.text.strip()

        return component
```

### 2. Create POML Component Types

```python
class RoleComponent(POMLComponent):
    """POML <role> component"""
    def __init__(self, role_name: str, description: str = ""):
        super().__init__("role", {"name": role_name})
        self.description = description

    def to_text(self, styling: Dict = None) -> str:
        style = (styling or {}).get('role', {})
        prefix = style.get('prefix', 'Role: ')
        return f"{prefix}{self.attributes['name']}\n{self.description}\n"

class InstructionComponent(POMLComponent):
    """POML <instructions> component"""
    def __init__(self, instructions: List[str]):
        super().__init__("instructions")
        self.instructions = instructions

    def to_text(self, styling: Dict = None) -> str:
        style = (styling or {}).get('instructions', {})
        format_type = style.get('format', 'numbered')

        if format_type == 'numbered':
            return "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(self.instructions))
        elif format_type == 'bullets':
            return "\n".join(f"- {inst}" for inst in self.instructions)
        else:
            return "\n".join(self.instructions)

class ExampleComponent(POMLComponent):
    """POML <example> component"""
    def __init__(self, input_text: str, output_text: str):
        super().__init__("example")
        self.input = input_text
        self.output = output_text

    def to_text(self, styling: Dict = None) -> str:
        style = (styling or {}).get('example', {})
        input_label = style.get('input_label', 'Input:')
        output_label = style.get('output_label', 'Output:')

        return f"{input_label}\n{self.input}\n\n{output_label}\n{self.output}\n"

class DataComponent(POMLComponent):
    """POML <data> component for tables, documents"""
    def __init__(self, data_type: str, content: str):
        super().__init__("data", {"type": data_type})
        self.content = content

    def to_text(self, styling: Dict = None) -> str:
        data_type = self.attributes['type']

        if data_type == 'table':
            return self._format_table(self.content)
        elif data_type == 'document':
            return self.content
        elif data_type == 'code':
            return f"```\n{self.content}\n```\n"
        else:
            return self.content

    def _format_table(self, content: str) -> str:
        """Format table with consistent spacing"""
        # Parse CSV-like content
        rows = [row.split('|') for row in content.strip().split('\n')]
        col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

        formatted = []
        for i, row in enumerate(rows):
            padded = [cell.ljust(col_widths[j]) for j, cell in enumerate(row)]
            formatted.append(' | '.join(padded))
            if i == 0:
                formatted.append('-' * len(formatted[0]))

        return '\n'.join(formatted)
```

### 3. Define CSS-Like Styling System

```python
class POMLStylesheet:
    """CSS-like styling for POML components"""
    def __init__(self):
        self.rules = {}

    def add_rule(self, component_type: str, properties: Dict[str, str]):
        """Add styling rule for component type"""
        self.rules[component_type] = properties

    def parse_from_string(self, css_like_string: str):
        """Parse CSS-like string"""
        import re
        # Simple parser: component { property: value; }
        pattern = r'(\w+)\s*\{([^}]+)\}'

        for match in re.finditer(pattern, css_like_string):
            component = match.group(1)
            properties_str = match.group(2)

            properties = {}
            for prop in properties_str.split(';'):
                if ':' in prop:
                    key, value = prop.split(':')
                    properties[key.strip()] = value.strip()

            self.rules[component] = properties

    def get_styling(self) -> Dict:
        """Get styling dict for components"""
        return self.rules

# Example usage
style_str = """
role {
    prefix: "System Role: ";
    padding: "1 line";
}

instructions {
    format: "numbered";
    prefix: "Follow these steps:";
}

example {
    input_label: "Query:";
    output_label: "Response:";
    border: "line";
}
"""

stylesheet = POMLStylesheet()
stylesheet.parse_from_string(style_str)
```

### 4. Build Templating System

```python
from string import Template

class POMLTemplate:
    """Templating system for dynamic POML"""
    def __init__(self, poml_string: str):
        self.template = Template(poml_string)

    def render(self, **variables) -> str:
        """Render template with variables"""
        return self.template.substitute(variables)

    def safe_render(self, **variables) -> str:
        """Render with defaults for missing variables"""
        return self.template.safe_substitute(variables)

# Example
template_str = """
<prompt>
<role name="$role_name">$role_description</role>
<task>$task_description</task>
<examples>
#for example in $examples
<example input="$example.input" output="$example.output"/>
#end
</examples>
</prompt>
"""

template = POMLTemplate(template_str)
# rendered = template.render(
#     role_name="Technical Writer",
#     role_description="You are expert...",
#     task_description="Summarize...",
#     examples=[...]
# )
```

### 5. Create POML Builder API

```python
class POMLBuilder:
    """Fluent API for building POML"""
    def __init__(self):
        self.components = []
        self.stylesheet = POMLStylesheet()

    def add_role(self, name: str, description: str = ""):
        """Add role component"""
        self.components.append(RoleComponent(name, description))
        return self

    def add_instructions(self, instructions: List[str], format_type: str = "numbered"):
        """Add instructions component"""
        comp = InstructionComponent(instructions)
        self.stylesheet.add_rule("instructions", {"format": format_type})
        self.components.append(comp)
        return self

    def add_example(self, input_text: str, output_text: str):
        """Add example component"""
        self.components.append(ExampleComponent(input_text, output_text))
        return self

    def add_data(self, data_type: str, content: str):
        """Add data component"""
        self.components.append(DataComponent(data_type, content))
        return self

    def build(self) -> str:
        """Compile to final prompt"""
        styling = self.stylesheet.get_styling()
        result = ""
        for comp in self.components:
            result += comp.to_text(styling) + "\n"
        return result

# Usage example
builder = POMLBuilder()
prompt = (builder
    .add_role("Technical Assistant", "You explain concepts clearly")
    .add_instructions([
        "Use simple language",
        "Provide examples",
        "Ask clarifying questions"
    ], format_type="numbered")
    .add_example(
        input_text="What is recursion?",
        output_text="Recursion is when a function calls itself..."
    )
    .build())
```

### 6. Developer Tools Integration

```python
class POMLProject:
    """Manage POML files and components"""
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.prompts = {}

    def load_prompt(self, name: str) -> str:
        """Load POML prompt from file"""
        import os
        filepath = os.path.join(self.project_dir, f"{name}.poml")
        with open(filepath, 'r') as f:
            poml_content = f.read()
        return poml_content

    def save_prompt(self, name: str, poml_content: str):
        """Save POML prompt to file"""
        import os
        filepath = os.path.join(self.project_dir, f"{name}.poml")
        os.makedirs(self.project_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(poml_content)

    def render_prompt(self, name: str, **variables) -> str:
        """Load, render, and return prompt"""
        poml_content = self.load_prompt(name)
        parser = POMLParser()
        component = parser.parse(poml_content)

        # Apply templating if needed
        template = POMLTemplate(poml_content)
        rendered = template.safe_render(**variables)

        return rendered

# Version control friendly: save as text files
project = POMLProject("./prompts")
project.save_prompt("summarization", """
<prompt>
<role name="Summarizer">You create concise summaries</role>
<task>Summarize the following text in 1-2 sentences</task>
<data type="document">$content</data>
</prompt>
""")
```

## Practical Guidance

- **Component Granularity**: Break prompts into role, task, examples, data
- **Styling Consistency**: Use stylesheets to enforce formatting across prompts
- **Version Control**: POML files are text-based, git-friendly
- **Reusability**: Build component libraries for common patterns
- **Testing**: Test prompt variations using templating

## Reference

POML (2508.13948): https://arxiv.org/abs/2508.13948

Structure complex prompts using component-based markup with CSS-like styling, improving maintainability, version control, and reducing format sensitivity across LLM applications.
