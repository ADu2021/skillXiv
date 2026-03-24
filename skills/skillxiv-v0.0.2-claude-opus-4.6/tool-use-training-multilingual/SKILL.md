---
name: tool-use-training-multilingual
title: "Teaching a Language Model to Speak the Language of Tools"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.23394"
keywords: [Tool Use, Function Calling, Multilingual Models, Instruction Tuning, Language Model Adaptation]
description: "Enable non-English language models to reliably generate function calls and tool invocations through continued training on bilingual function-calling datasets. Achieves 28% improvement in parsing accuracy while preserving language understanding."
---

# Tool-Use Training: Enabling Function Calls in Non-English Language Models

Most language models with reliable tool-use capabilities are English-dominant, leaving multilingual and low-resource language models unable to make clean function calls. When you ask a Bulgarian or Japanese model to use a tool, it often responds with verbose natural language explanations instead of parsable structured output. This forces developers to use English models regardless of their language requirements, limiting tool-augmented AI to English-speaking users.

The core problem is that tool-use training is typically English-centric. Models learn to generate function calls by being fine-tuned on English function-calling examples, which creates a language-specific skill. Teaching a model to reliably generate tool calls in another language requires explicitly training it on function-calling examples in that language, paired with English to show the relationship between natural language requests and structured outputs.

## Core Concept

Tool-use training works by treating function calls as a special output format that the model must learn to produce reliably. The key insight is:

1. Function calls are a structured language with precise syntax requirements
2. Non-English models can learn this structure through exposure to bilingual examples
3. Clean, parsable output (proper JSON, proper argument order) matters more than natural language explanations
4. Continued training on domain-specific function-calling data, even with relatively small datasets, significantly improves accuracy

The approach uses bilingual training data where the same functional intent appears in both English and the target language, showing the model that "call this function" should produce the same structured output regardless of input language.

## Architecture Overview

Tool-use training leverages the model's existing architecture without modification. The training pipeline consists of:

- **Bilingual Dataset**: Function-calling examples with English and target-language prompts paired with identical function call outputs
- **Continued Training Phase**: Fine-tune the base model on function-calling data with standard language modeling loss
- **Instruction Tuning Integration**: Mix function-calling examples with general instruction examples to preserve general capabilities
- **Parsing Validation**: Post-generation filtering that enforces valid JSON/structured output format

## Implementation

**Step 1: Create bilingual function-calling dataset**

Build training data where the same function call appears in both English and the target language. Start with existing MCP (Model Context Protocol) specifications or tool definitions.

```python
def create_bilingual_function_examples(tools_config, target_language='bg',
                                      examples_per_tool=5):
    """
    Generate bilingual function-calling examples from tool definitions.
    Creates training pairs showing the same function call in multiple languages.

    tools_config: dict with tool definitions (name, description, parameters)
    """
    translator = load_translator('en', target_language)
    examples = []

    for tool in tools_config['tools']:
        tool_name = tool['name']
        tool_description = tool['description']
        parameters = tool['function']['parameters']['properties']

        # Generate natural language prompts in English
        en_prompts = generate_prompts_for_tool(tool, num_prompts=examples_per_tool)

        for en_prompt in en_prompts:
            # Translate to target language
            tgt_prompt = translator.translate(en_prompt)

            # Generate the correct function call (deterministic from the prompt)
            function_call = extract_function_call_from_prompt(en_prompt, tool)

            # Add both language versions
            examples.append({
                'language': 'en',
                'prompt': en_prompt,
                'function_call': function_call,
                'tool': tool_name
            })
            examples.append({
                'language': target_language,
                'prompt': tgt_prompt,
                'function_call': function_call,
                'tool': tool_name
            })

    return examples

def generate_prompts_for_tool(tool, num_prompts=5):
    """
    Generate diverse natural language prompts for a tool.
    Each prompt should elicit a function call when processed.
    """
    prompts = []
    tool_name = tool['name']
    description = tool['description']
    parameters = tool['function']['parameters']['properties']

    param_names = list(parameters.keys())

    # Template 1: Direct command
    prompts.append(f"Use the {tool_name} function to {description}")

    # Template 2: Question asking for action
    prompts.append(f"Can you {description}?")

    # Template 3: Task request
    prompts.append(f"Please {description}")

    # Template 4-5: Specific parameter requests (vary which parameters)
    for i in range(2):
        selected_params = param_names[:len(param_names)//2]
        param_str = ', '.join(selected_params)
        prompts.append(f"Use {tool_name} with {param_str} to {description}")

    return prompts[:num_prompts]
```

**Step 2: Format training examples with special tokens for tool calls**

Structure the training data so the model learns to produce clean function call syntax in a consistent format.

```python
def format_tool_training_example(prompt, function_call, tool_name):
    """
    Format a training example with clear delimiters for function calls.
    This teaches the model when to switch to structured output mode.
    """

    # Standard format: prompt -> <TOOL_CALL>function_call</TOOL_CALL>
    # Explicit markers help the model learn the boundary between natural language and structure

    formatted = f"""User: {prompt}
Assistant: <TOOL_CALL>{function_call}</TOOL_CALL>"""

    return formatted
```

**Step 3: Train on mixed function-calling and instruction data**

Combine function-calling examples with general instruction data to maintain general capabilities while improving tool use.

```python
def train_tool_use_model(base_model, tool_examples, instruction_examples,
                        num_epochs=3, learning_rate=2e-5,
                        tool_example_ratio=0.3):
    """
    Fine-tune a base model on mixed tool-use and instruction data.
    Tool examples teach function calling; instruction examples maintain general abilities.
    """
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)
    total_examples = tool_examples + instruction_examples
    num_tool_batches = int(len(tool_examples) * tool_example_ratio)
    num_instruction_batches = len(instruction_examples) - num_tool_batches

    for epoch in range(num_epochs):
        # Shuffle and interleave examples
        random.shuffle(total_examples)

        # Group by type for better batch composition
        tool_batches = create_batches(tool_examples, batch_size=8)
        instruction_batches = create_batches(instruction_examples, batch_size=8)

        # Interleave: 30% tool examples, 70% instruction examples
        interleaved = []
        for i in range(max(len(tool_batches), len(instruction_batches))):
            if i < len(tool_batches):
                interleaved.append(('tool', tool_batches[i]))
            if i < len(instruction_batches):
                interleaved.append(('instruction', instruction_batches[i]))

        for batch_type, batch in interleaved:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = base_model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            optimizer.step()

            if batch_type == 'tool':
                print(f"Tool batch loss: {loss.item():.4f}")
            else:
                print(f"Instruction batch loss: {loss.item():.4f}")

    return base_model
```

**Step 4: Validate function-call parsing accuracy**

Test whether the model generates valid, parsable function calls in both languages.

```python
def evaluate_tool_use_accuracy(model, tokenizer, test_examples, language):
    """
    Measure how often the model generates valid, parsable function calls.
    Parsing accuracy is the key metric: the call must be valid JSON/structured format.
    """
    valid_calls = 0
    correct_calls = 0

    for example in test_examples:
        prompt = example['prompt']
        expected_call = example['function_call']

        # Generate output
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_length=256, temperature=0.1)
        generated_text = tokenizer.decode(output_ids[0])

        # Extract function call from generated text
        extracted_call = extract_function_call_text(generated_text)

        # Validate parsing
        is_valid = False
        try:
            parsed_call = json.loads(extracted_call)
            is_valid = True
        except json.JSONDecodeError:
            is_valid = False

        if is_valid:
            valid_calls += 1

            # Check if the call matches expected
            if parsed_call == json.loads(expected_call):
                correct_calls += 1

    parsing_accuracy = valid_calls / len(test_examples)
    correctness_accuracy = correct_calls / len(test_examples)

    return {
        'language': language,
        'parsing_accuracy': parsing_accuracy,
        'correctness_accuracy': correctness_accuracy,
        'total_examples': len(test_examples)
    }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| Learning rate | 2e-5 to 5e-5 | Lower than general fine-tuning due to small dataset |
| Batch size | 8-16 | Small batches help with diverse tool examples |
| Function-calling data ratio | 20-40% | Balance against catastrophic forgetting |
| Number of examples per tool | 5-20 | 10-15 is usually sufficient for 95%+ accuracy |
| Training epochs | 2-4 | Often converges in 2-3 epochs |
| Temperature for inference | 0.1-0.3 | Low temperature ensures consistent function call format |

**When to use tool-use training:**
- You need function-calling capability in non-English languages
- You have specific tool definitions or APIs you want to expose
- You need strict parsing requirements (must generate valid JSON/structured output)
- You want to preserve general language abilities while adding tool use

**When NOT to use tool-use training:**
- Your model already has reliable English tool-use (use prompt engineering instead)
- You don't need structured outputs (natural language descriptions suffice)
- You have unlimited access to larger multilingual models (just use them)
- Your tools change frequently (training overhead not justified)

**Common pitfalls:**
- **Catastrophic forgetting**: If the model stops answering general questions, increase the instruction data ratio from 70% to 80-90%.
- **Inconsistent function format**: Use explicit delimiters (<TOOL_CALL> tags) and enforce temperature=0.1 during inference.
- **Poor multilingual transfer**: The model might not generalize function-calling to the target language. Use truly bilingual examples, not translated ones.
- **Parsing failures**: Many generated calls might be syntactically invalid. Add validation loss that penalizes malformed JSON.

## Reference

Teaching a Language Model to Speak the Language of Tools
https://arxiv.org/abs/2506.23394