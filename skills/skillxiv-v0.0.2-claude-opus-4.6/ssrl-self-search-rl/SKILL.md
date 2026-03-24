---
name: ssrl-self-search-rl
title: "SSRL: Self-Search Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.10874
keywords: [reinforcement-learning, self-search, knowledge-retrieval, question-answering, structured-prompting]
description: "Enable LLMs to perform internal knowledge search using structured prompting and rule-based rewards, reducing reliance on external search while maintaining accuracy and reducing hallucination."
---

# SSRL: Self-Search Reinforcement Learning

## Core Concept

Most question-answering systems rely on external search engines or retrieval databases. Self-Search Reinforcement Learning (SSRL) teaches LLMs to leverage their internal knowledge more effectively through structured prompting and iterative refinement.

The key innovation is using format-based and rule-based rewards to guide the model through internal knowledge search without external tools, reducing hallucination while maintaining or improving answer quality. The system treats search as a reinforcement learning problem where the model learns to structure and refine its internal search process.

## Architecture Overview

- **Structured Prompting Framework**: Guides the model to decompose questions into search steps
- **Self-Search Process**: Repeated sampling and refinement of responses to "search" internal knowledge
- **Format-Based Rewards**: Reward signals based on output format conformance (structured output, proper reasoning chains)
- **Rule-Based Rewards**: Domain-specific correctness checks without external evaluation
- **Iterative Refinement**: Multiple rounds of generation with feedback to improve answer quality
- **Graceful Fallback**: Integrates with external search when internal knowledge is insufficient

## Implementation Steps

### 1. Design Structured Prompting Schema

Define a structured format that guides the model's internal search process.

```python
def create_search_prompt(question, search_instructions=None):
    """
    Create a structured prompt that guides internal knowledge search
    """
    if search_instructions is None:
        search_instructions = """
Please answer this question by:
1. [IDENTIFY_CONCEPTS]: List key concepts to search for
2. [SEARCH_MEMORY]: Internally search your knowledge for relevant information
3. [GATHER_EVIDENCE]: Compile evidence from multiple knowledge sources
4. [SYNTHESIZE]: Combine evidence into a coherent answer
5. [VERIFY]: Double-check the answer for consistency

Format your response with these exact labels.
"""

    prompt = f"""{search_instructions}

Question: {question}

Let's work through this step by step:
[IDENTIFY_CONCEPTS]: """

    return prompt

# Example usage
question = "What is the capital of France?"
prompt = create_search_prompt(question)
print(prompt)
```

### 2. Implement Structured Response Parsing

Parse the model's output to extract structured components.

```python
import re

class StructuredResponse:
    def __init__(self, full_response):
        self.full_response = full_response
        self.components = self._parse_components()

    def _parse_components(self):
        """Extract structured components from response"""
        components = {
            'concepts': self._extract_section('[IDENTIFY_CONCEPTS]', '[SEARCH_MEMORY]'),
            'search': self._extract_section('[SEARCH_MEMORY]', '[GATHER_EVIDENCE]'),
            'evidence': self._extract_section('[GATHER_EVIDENCE]', '[SYNTHESIZE]'),
            'synthesis': self._extract_section('[SYNTHESIZE]', '[VERIFY]'),
            'verification': self._extract_section('[VERIFY]', None),
        }
        return components

    def _extract_section(self, start_marker, end_marker):
        """Extract text between two markers"""
        start_idx = self.full_response.find(start_marker)
        if start_idx == -1:
            return ""

        if end_marker:
            end_idx = self.full_response.find(end_marker)
            if end_idx == -1:
                return self.full_response[start_idx + len(start_marker):]
        else:
            end_idx = len(self.full_response)

        return self.full_response[start_idx + len(start_marker):end_idx].strip()

    def get_final_answer(self):
        """Extract final answer from verification section"""
        verification = self.components['verification']
        # Look for answer pattern
        match = re.search(r'(?:Answer|Final Answer):\s*(.+?)(?:\n|$)', verification)
        if match:
            return match.group(1).strip()
        return verification.split('\n')[0].strip() if verification else ""
```

### 3. Define Format-Based Rewards

Create reward functions that check whether the response follows the structured format.

```python
def compute_format_reward(response, required_sections=None):
    """
    Reward based on format compliance.
    Returns higher scores for well-structured responses.
    """
    if required_sections is None:
        required_sections = [
            '[IDENTIFY_CONCEPTS]',
            '[SEARCH_MEMORY]',
            '[GATHER_EVIDENCE]',
            '[SYNTHESIZE]',
            '[VERIFY]'
        ]

    format_score = 0.0
    max_score = len(required_sections)

    # Check each section is present and non-empty
    for section in required_sections:
        if section in response:
            # Extract section content
            start_idx = response.find(section)
            if start_idx != -1:
                # Section exists and has content
                section_content = response[start_idx + len(section):]
                if len(section_content.strip()) > 10:  # Non-trivial content
                    format_score += 1.0

    format_score = format_score / max_score if max_score > 0 else 0.0
    return format_score

def compute_length_reward(response, min_length=100, max_length=2000):
    """
    Reward well-length responses (not too short, not too long).
    Prevents both lazy and verbose responses.
    """
    length = len(response)

    if length < min_length:
        return 0.2  # Too short
    elif length > max_length:
        return 0.6  # Too long but acceptable
    else:
        return 1.0  # Ideal length
```

### 4. Implement Rule-Based Verification

Define domain-specific rules for evaluating answer quality.

```python
class RuleBasedVerifier:
    """
    Verify answers using domain-specific rules
    """
    def __init__(self, domain='general'):
        self.domain = domain
        self.rules = self._load_rules(domain)

    def _load_rules(self, domain):
        """Load verification rules for specific domain"""
        rules = {
            'factual': {
                'name_consistency': self._check_name_consistency,
                'logical_coherence': self._check_logical_coherence,
                'no_contradictions': self._check_no_contradictions,
            },
            'qa': {
                'answers_question': self._check_answers_question,
                'evidence_cited': self._check_evidence_cited,
            }
        }
        return rules.get(domain, rules['factual'])

    def verify(self, response, question, gold_answer=None):
        """
        Apply rules and compute verification score
        """
        scores = {}
        for rule_name, rule_fn in self.rules.items():
            try:
                score = rule_fn(response, question, gold_answer)
                scores[rule_name] = score
            except Exception:
                scores[rule_name] = 0.0

        # Average all rule scores
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        return avg_score, scores

    def _check_name_consistency(self, response, question, gold_answer):
        """Check if entities are referred to consistently"""
        # Extract named entities
        entities = self._extract_entities(response)
        # Check for inconsistent references (Paris vs PARIS)
        return 0.9 if len(set(entities)) < len(entities) else 1.0

    def _check_logical_coherence(self, response, question, gold_answer):
        """Check if response makes logical sense"""
        sentences = response.split('.')
        if len(sentences) < 3:
            return 0.5  # Too few sentences
        return 1.0

    def _check_no_contradictions(self, response, question, gold_answer):
        """Check for internal contradictions"""
        # Simple heuristic: look for "but" followed by opposite statements
        if ' but ' in response.lower():
            # More complex response, less likely to be wrong
            return 0.9
        return 1.0

    def _check_answers_question(self, response, question, gold_answer):
        """Check if response directly answers the question"""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        overlap = len(question_words & response_words) / len(question_words)
        return min(overlap, 1.0)

    def _check_evidence_cited(self, response, question, gold_answer):
        """Check if response cites evidence"""
        evidence_markers = ['according to', 'based on', 'evidence shows', 'research indicates']
        has_evidence = any(marker in response.lower() for marker in evidence_markers)
        return 1.0 if has_evidence else 0.5

    def _extract_entities(self, text):
        """Simple entity extraction"""
        import nltk
        sentences = nltk.sent_tokenize(text)
        entities = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            for token, pos in pos_tags:
                if pos in ['NNP', 'NNPS']:  # Proper nouns
                    entities.append(token.lower())
        return entities
```

### 5. Implement Self-Search RL Training Loop

Use rewards to guide model refinement through multiple iterations.

```python
def train_ssrl(model, questions, gold_answers, num_iterations=3):
    """
    Train model using self-search with rule-based rewards
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    verifier = RuleBasedVerifier(domain='qa')

    for iteration in range(num_iterations):
        print(f"\n=== Self-Search Iteration {iteration + 1} ===")

        total_loss = 0.0
        total_reward = 0.0

        for question, gold_answer in zip(questions, gold_answers):
            # 1. Generate structured response
            prompt = create_search_prompt(question)
            response = model.generate(prompt, max_length=1024)

            # 2. Parse structured response
            parsed = StructuredResponse(response)
            final_answer = parsed.get_final_answer()

            # 3. Compute rewards
            format_score = compute_format_reward(response)
            length_score = compute_length_reward(response)
            rule_score, rule_details = verifier.verify(response, question, gold_answer)

            # Combined reward
            total_score = 0.5 * format_score + 0.2 * length_score + 0.3 * rule_score

            # 4. Compute policy gradient loss
            log_prob = model.compute_log_prob(response)

            # Advantage: how much better than average
            # (Using running mean for baseline)
            advantage = total_score - 0.5  # Assume average is 0.5

            loss = -log_prob * advantage

            # 5. Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_reward += total_score

            if len(questions) > 0 and questions.index(question) % 10 == 0:
                print(f"Question {questions.index(question)}: Reward={total_score:.3f}, "
                      f"Format={format_score:.2f}, Rule={rule_score:.2f}")

        avg_reward = total_reward / len(questions)
        avg_loss = total_loss / len(questions)
        print(f"Iteration {iteration}: Avg Loss={avg_loss:.4f}, Avg Reward={avg_reward:.3f}")
```

### 6. Inference with Fallback to External Search

Use the trained model for QA with graceful fallback to search if needed.

```python
def answer_with_self_search(model, question, search_engine=None, confidence_threshold=0.7):
    """
    Answer question using self-search, with optional fallback to search.
    """
    # 1. Try internal self-search
    prompt = create_search_prompt(question)
    response = model.generate(prompt, max_length=1024)

    parsed = StructuredResponse(response)
    final_answer = parsed.get_final_answer()

    # 2. Verify answer quality
    verifier = RuleBasedVerifier(domain='qa')
    confidence, details = verifier.verify(response, question)

    print(f"Self-search confidence: {confidence:.3f}")
    print(f"Answer: {final_answer}")

    # 3. Fallback to external search if needed
    if confidence < confidence_threshold and search_engine is not None:
        print(f"Confidence too low ({confidence:.3f}), falling back to search...")
        search_results = search_engine.search(question)
        external_answer = search_results[0]['content'] if search_results else "No results"
        return external_answer
    else:
        return final_answer
```

## Practical Guidance

### Hyperparameters & Configuration

- **Search Iterations**: 1-3 refinement passes per question (more = higher quality but slower)
- **Format Sections**: 5-7 structured sections guide but not constrain reasoning
- **Verification Threshold**: 0.6-0.8 for triggering external search fallback
- **Length Bounds**: min_length=100 tokens, max_length=2000 tokens
- **Learning Rate**: 2e-5 to 1e-4 (conservative to preserve knowledge)

### When to Use SSRL

- You want to reduce reliance on external search for question-answering
- Hallucination is a concern and structured reasoning helps
- You have domain-specific verification rules available
- You need interpretable reasoning traces
- Inference can afford multi-step generation

### When NOT to Use SSRL

- You need single-hop factual lookups (external search is faster)
- Your domain lacks verifiable rules or ground truth answers
- Latency is critical (SSRL adds multiple generation passes)
- Your model's knowledge is severely limited (no self-search help)
- You need real-time information (SSRL uses static knowledge)

### Common Pitfalls

1. **Weak Verification Rules**: If rules don't catch errors, model trains on wrong rewards. Start with simple rules and validate.
2. **Circular Reasoning**: Model might generate plausible-sounding but incorrect answers. Combine format + rule verification.
3. **Over-Reliance on Format**: Structure alone doesn't guarantee correctness. Weight rule-based rewards appropriately.
4. **Insufficient Data**: Need multiple questions per domain to learn patterns. At least 100+ examples recommended.
5. **No Baseline Comparison**: Always compare against pure LLM or search-based baseline to validate improvement.

## Reference

SSRL (2508.10874): https://arxiv.org/abs/2508.10874

Teach LLMs to leverage internal knowledge through structured self-search and rule-based verification, reducing hallucination and external search dependency while maintaining answer quality through iterative refinement.
