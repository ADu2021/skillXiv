---
name: comorag-cognitive-memory-rag
title: "ComoRAG: Cognitive Memory-Organized RAG for Long Narrative Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.10419
keywords: [retrieval-augmented-generation, long-context, narrative-reasoning, memory-workspace, iterative-reasoning]
description: "Iteratively retrieve and reason over long narratives using a dynamic memory workspace that integrates retrieved facts into a shared context for complex multi-hop reasoning."
---

# ComoRAG: Cognitive Memory-Organized RAG for Long Narrative Reasoning

## Core Concept

Traditional RAG systems retrieve relevant passages once and generate answers. For long narratives (200K+ tokens), this fails because complex questions require tracking entities, relationships, and events across the entire text. Readers build mental models of stories by revisiting information.

ComoRAG mimics human narrative comprehension through iterative reasoning with a dynamic memory workspace. Each iteration generates exploratory queries, retrieves supporting evidence, and integrates findings into a growing memory structure. This multi-hop approach handles questions requiring global narrative understanding.

## Architecture Overview

- **Dynamic Memory Workspace**: Evolving context that accumulates relevant facts from iterations
- **Exploratory Query Generation**: Create diverse queries to discover different narrative aspects
- **Iterative Retrieval**: Retrieve evidence repeatedly, refining understanding with each cycle
- **Memory Integration**: Incorporate retrieved facts into shared workspace to build comprehensive narrative model
- **Reasoning Cycles**: Multiple passes over narrative with different focus areas
- **Long-Context Handling**: Designed for 200K+ token narratives with complex plot structures

## Implementation Steps

### 1. Initialize Memory Workspace

Create a dynamic memory structure that accumulates information across reasoning cycles.

```python
from collections import defaultdict
import json

class MemoryWorkspace:
    """
    Dynamic memory workspace for narrative reasoning
    Stores entities, relationships, events discovered during reasoning
    """
    def __init__(self, max_memory_size=10000):
        self.max_memory_size = max_memory_size

        # Memory structures
        self.entities = {}  # entity_id -> {name, type, mentions}
        self.relationships = []  # List of (entity1, relation, entity2)
        self.events = []  # List of {time, action, participants}
        self.facts = []  # List of {fact, evidence_span, confidence}

        self.memory_tokens = 0

    def add_entity(self, entity_name, entity_type='PERSON'):
        """Register an entity in memory"""
        entity_id = len(self.entities)

        self.entities[entity_id] = {
            'name': entity_name,
            'type': entity_type,
            'mentions': [],
            'properties': {}
        }

        return entity_id

    def add_relationship(self, entity_id1, relation, entity_id2):
        """Add a relationship between entities"""
        self.relationships.append({
            'from': entity_id1,
            'relation': relation,
            'to': entity_id2
        })

    def add_event(self, time_point, action, participants, location=None):
        """Add an event to memory"""
        self.events.append({
            'time': time_point,
            'action': action,
            'participants': participants,
            'location': location
        })

    def add_fact(self, fact_text, evidence_span, confidence=0.9):
        """Add a fact with supporting evidence"""
        self.facts.append({
            'text': fact_text,
            'evidence': evidence_span,
            'confidence': confidence,
            'iteration': len(self.facts)
        })

        self.memory_tokens += len(fact_text.split())

    def get_memory_context(self, max_tokens=2000):
        """
        Serialize memory into context for model
        """
        context = "# Memory Workspace\n\n"

        # Entities
        if self.entities:
            context += "## Entities\n"
            for entity_id, entity in self.entities.items():
                context += f"- {entity['name']} ({entity['type']})\n"

        # Key relationships
        if self.relationships:
            context += "\n## Key Relationships\n"
            for rel in self.relationships[:10]:  # Top 10
                e1 = self.entities[rel['from']]['name']
                e2 = self.entities[rel['to']]['name']
                context += f"- {e1} {rel['relation']} {e2}\n"

        # Timeline of events
        if self.events:
            context += "\n## Timeline\n"
            for event in self.events[:10]:  # Recent events
                context += f"- [{event['time']}] {event['action']} "
                context += f"({', '.join(event['participants'])})\n"

        # Key facts
        if self.facts:
            context += "\n## Key Facts\n"
            for fact in self.facts[-5:]:  # Last 5 facts
                context += f"- {fact['text']}\n"

        return context[:max_tokens]  # Truncate to limit size

    def merge_with_iteration_memory(self, iteration_results):
        """
        Integrate results from a reasoning iteration
        """
        for entity in iteration_results.get('entities', []):
            if entity['name'] not in [e['name'] for e in self.entities.values()]:
                self.add_entity(entity['name'], entity.get('type', 'UNKNOWN'))

        for rel in iteration_results.get('relationships', []):
            self.add_relationship(rel['from'], rel['relation'], rel['to'])

        for event in iteration_results.get('events', []):
            self.add_event(event['time'], event['action'], event['participants'])

        for fact in iteration_results.get('facts', []):
            self.add_fact(fact['text'], fact['evidence'])
```

### 2. Implement Exploratory Query Generation

Generate diverse queries to explore different aspects of the narrative.

```python
class ExploratoryQueryGenerator:
    """
    Generate exploratory queries for iterative retrieval
    """
    def __init__(self, llm_model):
        self.llm = llm_model
        self.query_templates = [
            "What are the main characters and their relationships?",
            "What are the key events in chronological order?",
            "What conflicts or tensions exist between characters?",
            "What is the motivation for each character's actions?",
            "What are the major plot twists or surprises?",
            "How do settings change throughout the narrative?",
            "What consequences follow from key decisions?",
            "What themes or lessons emerge from the story?"
        ]

    def generate_exploration_queries(self, question, memory_workspace, iteration=0):
        """
        Generate queries tailored to exploring aspects relevant to the question
        """
        # Start with template queries
        if iteration == 0:
            queries = [
                f"What is directly relevant to: {question}?"
            ] + self.query_templates[:3]
        else:
            # Later iterations: generate based on memory gaps
            memory_context = memory_workspace.get_memory_context()

            prompt = f"""Based on this memory workspace and the original question,
what additional aspects should we explore?

Original Question: {question}

Current Memory:
{memory_context}

Generate 3 specific exploratory queries to fill gaps in understanding.
Focus on information NOT yet in memory."""

            response = self.llm.generate(prompt, max_length=200)
            queries = self._parse_queries(response)

        return queries

    def _parse_queries(self, response_text):
        """Extract individual queries from LLM response"""
        import re
        # Extract numbered or bulleted items
        query_pattern = r'[\d\-\*]\.\s*(.+?)(?:\n|$)'
        queries = re.findall(query_pattern, response_text)
        return queries[:3]  # Limit to 3 per iteration
```

### 3. Implement Iterative Retrieval

Retrieve evidence based on exploratory queries.

```python
from typing import List

class IterativeRetriever:
    """
    Retrieve evidence iteratively, updating based on memory state
    """
    def __init__(self, corpus_embedder, vectorstore):
        self.embedder = corpus_embedder
        self.vectorstore = vectorstore

    def retrieve_for_queries(self, queries: List[str], narrative_doc: str,
                            memory_workspace=None, k=3):
        """
        Retrieve evidence for each query
        """
        retrieved_passages = []

        for query in queries:
            # Optionally contextualize query with memory
            if memory_workspace:
                contextualized_query = self._contextualize_query(
                    query, memory_workspace
                )
            else:
                contextualized_query = query

            # Retrieve top-k passages
            query_embedding = self.embedder.encode(contextualized_query)
            passages = self.vectorstore.search(
                query_embedding,
                k=k,
                narrative_id=None
            )

            for passage in passages:
                retrieved_passages.append({
                    'query': query,
                    'passage': passage['text'],
                    'span': passage['span'],
                    'relevance': passage['score']
                })

        return retrieved_passages

    def _contextualize_query(self, query, memory_workspace):
        """
        Enhance query with context from memory
        """
        memory_context = memory_workspace.get_memory_context(max_tokens=500)

        contextualized = f"""Given this narrative context:
{memory_context}

Please provide information about: {query}"""

        return contextualized
```

### 4. Extract and Integrate New Information

Parse retrieved passages to extract structured information.

```python
class InformationExtractor:
    """
    Extract entities, relationships, and facts from retrieved passages
    """
    def __init__(self, llm_model):
        self.llm = llm_model

    def extract_from_passage(self, passage):
        """
        Extract structured information from a passage
        """
        prompt = f"""Extract the following from this narrative passage:
- Named entities (persons, places, things)
- Relationships between entities
- Events and their participants
- Key facts

Passage:
{passage}

Return as JSON with keys: entities, relationships, events, facts"""

        response = self.llm.generate(prompt, max_length=300)
        extracted = self._parse_json_response(response)

        return extracted

    def _parse_json_response(self, response_text):
        """Parse JSON from LLM response"""
        import json
        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except:
            # Fallback: extract JSON-like structure manually
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    return {
                        'entities': [],
                        'relationships': [],
                        'events': [],
                        'facts': []
                    }
            else:
                return {
                    'entities': [],
                    'relationships': [],
                    'events': [],
                    'facts': []
                }
```

### 5. Main Iterative Reasoning Loop

Implement the core iterative reasoning cycles.

```python
class ComoRAGReasoner:
    """
    Main ComoRAG reasoning loop
    """
    def __init__(self, llm_model, corpus_embedder, vectorstore):
        self.llm = llm_model
        self.query_generator = ExploratoryQueryGenerator(llm_model)
        self.retriever = IterativeRetriever(corpus_embedder, vectorstore)
        self.extractor = InformationExtractor(llm_model)
        self.memory = MemoryWorkspace()

    def reason_over_narrative(self, question, narrative_doc, num_iterations=3):
        """
        Iteratively reason over narrative to answer question
        """
        print(f"Question: {question}\n")

        for iteration in range(num_iterations):
            print(f"=== Iteration {iteration + 1} ===")

            # 1. Generate exploratory queries
            queries = self.query_generator.generate_exploration_queries(
                question, self.memory, iteration
            )
            print(f"Queries: {queries}")

            # 2. Retrieve relevant passages
            retrieved = self.retriever.retrieve_for_queries(
                queries, narrative_doc, self.memory, k=3
            )
            print(f"Retrieved {len(retrieved)} passages")

            # 3. Extract information from passages
            iteration_info = {
                'entities': [],
                'relationships': [],
                'events': [],
                'facts': []
            }

            for item in retrieved:
                extracted = self.extractor.extract_from_passage(item['passage'])

                # Merge into iteration info
                iteration_info['entities'].extend(extracted.get('entities', []))
                iteration_info['relationships'].extend(
                    extracted.get('relationships', [])
                )
                iteration_info['events'].extend(extracted.get('events', []))
                iteration_info['facts'].extend(extracted.get('facts', []))

            # 4. Integrate into memory workspace
            self.memory.merge_with_iteration_memory(iteration_info)
            print(f"Memory updated: {len(self.memory.entities)} entities, "
                  f"{len(self.memory.facts)} facts\n")

        # 5. Generate final answer based on accumulated memory
        final_answer = self._generate_answer(question)
        return final_answer

    def _generate_answer(self, question):
        """
        Generate final answer using accumulated memory
        """
        memory_context = self.memory.get_memory_context()

        prompt = f"""Based on this narrative understanding, answer the question:

Question: {question}

Memory/Understanding:
{memory_context}

Provide a comprehensive answer supported by the narrative."""

        answer = self.llm.generate(prompt, max_length=500)
        return answer
```

### 6. Integration and Usage

Use ComoRAG for long-narrative question-answering.

```python
def answer_long_narrative_question(narrative_doc, question, num_iterations=3):
    """
    Answer questions about long narratives using ComoRAG
    """
    # Initialize components
    llm = load_llm_model("gpt-4-turbo")
    embedder = load_embedder("all-MiniLM-L6-v2")
    vectorstore = build_vectorstore(narrative_doc)

    # Create reasoner
    reasoner = ComoRAGReasoner(llm, embedder, vectorstore)

    # Iteratively reason
    answer = reasoner.reason_over_narrative(
        question,
        narrative_doc,
        num_iterations=num_iterations
    )

    return answer

# Example usage
narrative = "..."  # 200K+ token document
question = "What is the relationship between Character A and Character B?"
answer = answer_long_narrative_question(narrative, question, num_iterations=3)
print(f"Answer: {answer}")
```

## Practical Guidance

### Hyperparameters & Configuration

- **Iterations**: 2-4 (more = better understanding but slower)
- **Retrieval k**: 2-5 passages per query (3 is good balance)
- **Memory Size**: 2000-5000 tokens (grow dynamically)
- **Query Count**: 2-3 per iteration (more queries = better coverage)
- **Entity Tracking**: Keep top 20-50 entities in workspace

### When to Use ComoRAG

- Documents are long (> 50K tokens) with complex narratives
- Questions require multi-hop reasoning across narrative
- You need to track entities and relationships over long spans
- Traditional single-pass RAG fails on complex questions
- You can afford multiple retrieval and reasoning passes

### When NOT to Use ComoRAG

- Documents are short (< 10K tokens) — standard RAG sufficient
- Questions are simple fact-lookups (no multi-hop needed)
- Latency is critical (iterative approach is slower)
- Computational resources are very limited
- You need real-time answers (multi-iteration takes time)

### Common Pitfalls

1. **Too Many Iterations**: Diminishing returns after 3-4 iterations. More doesn't always help.
2. **Poor Query Generation**: If exploratory queries aren't diverse, memory updates are redundant. Vary query templates.
3. **Memory Explosion**: Track memory size; prune less-relevant facts if it grows too large.
4. **Retrieval Collapse**: If all iterations retrieve the same passages, change retrieval strategy (e.g., add diversity).
5. **No Baseline Comparison**: Always compare against standard RAG to ensure iterative approach helps.

## Reference

ComoRAG (2508.10419): https://arxiv.org/abs/2508.10419

Iteratively retrieve and reason over long narratives using a cognitive memory workspace, achieving 11% improvements on 200K+ token narrative reasoning benchmarks through multi-hop exploration and fact integration.
