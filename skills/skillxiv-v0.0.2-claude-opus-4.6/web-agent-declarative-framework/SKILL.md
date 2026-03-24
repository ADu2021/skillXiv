---
name: web-agent-declarative-framework
title: "Building the Web for Agents: A Declarative Framework for Agentic Web Interactions"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.11287"
keywords: [Web Agents, Declarative APIs, Tool Specification, Machine-Readable Contracts, Agent Integration]
description: "Enable safe and efficient AI agent interactions with websites via declarative HTML extensions—define agent-executable tools and context using <tool> and <context> tags instead of relying on brittle UI inference."
---

# Build Agent-Ready Websites with Declarative Tool Specifications

Web agents typically infer capabilities by analyzing human UI—button labels, form structures, navigation patterns. This approach is fragile: agents misinterpret intent, trigger unintended actions, and fail gracefully on novel UI layouts. The Agentic Web framework (VOIX) shifts responsibility to developers: specify what agents can do explicitly via declarative HTML extensions.

Rather than agents guessing "what this button does," developers declare tools through `<tool>` and `<context>` tags. This creates a machine-readable contract for agent behavior, enabling safe, predictable, and auditable agent interactions while preserving user privacy and control.

## Core Concept

Current agent-web integration suffers from two problems:

1. **UI Brittleness**: Agents infer capabilities from human-centric UI elements (button text, form labels); changes to UI break agents
2. **Intent Ambiguity**: UI elements designed for humans (e.g., "Submit") don't clearly specify agent-executable intent

VOIX solves this with two declarative mechanisms:

- **`<tool>` tags**: Developers explicitly declare actions agents can invoke, specifying inputs, outputs, and side effects
- **`<context>` tags**: Specify relevant state information agents need to make decisions (user preferences, session state, current form data)

Developers build the website normally; agents see enhanced, machine-readable augmentations. A hackathon study showed participants regardless of experience could rapidly build functional agent applications.

## Architecture Overview

- **`<tool>` Element**: Declarative action specification with name, description, parameters (type, required, validation), output format
- **`<context>` Element**: State declaration with visibility scope (user-visible, agent-only, public); prevents oversharing sensitive data
- **Tool Registry**: Browser-side index of available tools per page; agents query registry rather than inferring from DOM
- **Privacy Layer**: Developers control which context is exposed to agents vs. humans; decouples conversational state from website state
- **Validation & Execution**: Agents receive declaratively-specified parameters; framework validates before execution

## Implementation Steps

**Step 1: Define Tools.** Markup website actions with declarative tool specifications.

```html
<!-- Simple search tool -->
<tool name="search_products"
      description="Search product catalog by keyword"
      parameters="query:string(required), category:string(optional), limit:number(default=10)">
  <!-- Tool definition embedded; framework routes agent requests here -->
  <action method="POST" endpoint="/api/search">
    <param name="q" source="query"/>
    <param name="cat" source="category"/>
    <param name="limit" source="limit"/>
  </action>
  <output format="json" schema="product_list"/>
</tool>

<!-- Complex tool with validation -->
<tool name="add_to_cart"
      description="Add item to shopping cart"
      parameters="product_id:string(required,pattern='p_[0-9]+'), quantity:number(required,min=1,max=100)">
  <permission scope="user"/>  <!-- Requires user confirmation -->
  <action method="POST" endpoint="/cart/add">
    <param name="pid" source="product_id"/>
    <param name="qty" source="quantity"/>
  </action>
  <output format="json" schema="cart_status"/>
  <side_effect type="user_notification">Cart updated</side_effect>
</tool>
```

**Step 2: Declare Context.** Specify application state relevant to agent decision-making.

```html
<!-- User context: agent needs to know user's preferences -->
<context name="user_prefs" scope="agent">
  <field name="preferred_category">electronics</field>
  <field name="budget_max">500</field>
  <field name="shipping_address">123 Main St, City</field>
</context>

<!-- Session context: agent state for conversation continuity -->
<context name="session" scope="agent">
  <field name="conversation_id">sess_12345</field>
  <field name="previous_queries">last 5 user queries</field>
  <field name="current_task">help_choose_laptop</field>
</context>

<!-- Public context: information agents can expose to humans -->
<context name="product_listing" scope="public">
  <field name="featured_products">list of top 10 products</field>
  <field name="sale_items">current promotions</field>
</context>
```

**Step 3: Build Tool Registry.** Parse declarative specs and index tools for agent access.

```python
class ToolRegistry:
    def __init__(self, html_content):
        self.tools = {}
        self.context = {}
        self.parse_html(html_content)

    def parse_html(self, html):
        """Extract <tool> and <context> declarations from HTML."""
        soup = BeautifulSoup(html, 'html.parser')

        # Parse tools
        for tool_elem in soup.find_all('tool'):
            tool_name = tool_elem.get('name')
            description = tool_elem.get('description')
            param_spec = tool_elem.get('parameters')

            self.tools[tool_name] = {
                'description': description,
                'params': self.parse_parameters(param_spec),
                'endpoint': tool_elem.find('action').get('endpoint'),
                'method': tool_elem.find('action').get('method'),
                'requires_permission': bool(tool_elem.find('permission')),
                'output_schema': tool_elem.find('output').get('schema')
            }

        # Parse context
        for ctx_elem in soup.find_all('context'):
            ctx_name = ctx_elem.get('name')
            scope = ctx_elem.get('scope')

            fields = {}
            for field in ctx_elem.find_all('field'):
                fields[field.get('name')] = field.text

            self.context[ctx_name] = {
                'scope': scope,
                'fields': fields
            }

    def parse_parameters(self, param_spec):
        """Parse parameter string: 'name:type(constraints), ...'"""
        params = {}
        for param_str in param_spec.split(','):
            # Parse: "query:string(required)" or "limit:number(default=10)"
            parts = param_str.strip().split(':')
            name = parts[0]
            type_and_constraints = parts[1] if len(parts) > 1 else "string"

            params[name] = self.parse_type_spec(type_and_constraints)

        return params

    def get_available_tools(self):
        """Return list of tools available to agents."""
        return list(self.tools.keys())

    def get_tool_spec(self, tool_name):
        """Retrieve full specification for a tool."""
        return self.tools.get(tool_name)

    def validate_tool_call(self, tool_name, args):
        """Validate agent tool call against declared spec."""
        tool_spec = self.get_tool_spec(tool_name)
        if not tool_spec:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Validate each parameter
        for param_name, param_spec in tool_spec['params'].items():
            if param_spec['required'] and param_name not in args:
                raise ValueError(f"Missing required parameter: {param_name}")

            if param_name in args:
                # Type and constraint checking
                self.validate_parameter(args[param_name], param_spec)

        return True
```

**Step 4: Agent-Server Integration.** Agents query registry and execute validated tool calls.

```python
@app.route('/agent/tools', methods=['GET'])
def list_tools():
    """Agent queries available tools."""
    return {
        'tools': registry.get_available_tools(),
        'context': registry.context
    }

@app.route('/agent/execute', methods=['POST'])
def execute_tool():
    """Execute agent tool call with validation."""
    data = request.json
    tool_name = data['tool']
    args = data['args']

    # Validate against spec
    try:
        registry.validate_tool_call(tool_name, args)
    except ValueError as e:
        return {'error': str(e)}, 400

    # Execute
    tool_spec = registry.get_tool_spec(tool_name)
    response = requests.request(
        method=tool_spec['method'],
        url=tool_spec['endpoint'],
        json=args
    )

    return response.json()
```

## Practical Guidance

**When to Use:** Web applications requiring agent integration (customer service bots, shopping assistants, workflow automation). Use whenever you want agents to interact with your site predictably.

**Declarative Best Practices:**
- `<tool>` descriptions: be explicit about side effects (e.g., "Permanently deletes user account")
- Parameter specs: use meaningful type constraints (min/max, pattern regex) to reduce validation burden on agents
- Context scopes: carefully separate agent-only sensitive data from public-facing information
- Permissions: mark tools requiring user confirmation (purchases, deletions) with `<permission>`

**Pitfalls:**
- Over-specification (too many parameters) confuses agents; keep tools focused and composable
- Vague descriptions lead to misuse; test descriptions with agents in adversarial scenarios
- Leaking sensitive context (passwords, tokens) via scope="agent"; audit context declarations
- Breaking spec contracts during deployment; version your schema and communicate changes clearly

**When NOT to Use:** Static websites with no agent interaction needs; internal tools not exposed to agents.

**Integration:** Pairs naturally with any agent framework (ReAct, LangChain, custom). Can be layered atop existing websites without major refactoring.

---
Reference: https://arxiv.org/abs/2511.11287
