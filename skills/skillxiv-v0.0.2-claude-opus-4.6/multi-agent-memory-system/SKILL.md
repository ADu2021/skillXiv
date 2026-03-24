---
name: multi-agent-memory-system
title: "MIRIX: Multi-Agent Memory System for LLM-Based Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07957"
keywords: [Multi-Agent Systems, Memory Management, LLM Agents, Semantic Memory, Episodic Memory, Knowledge Retrieval]
description: "Build persistent, structured memory systems for LLM agents that remember user context across sessions, organize information semantically, and retrieve relevant knowledge automatically before responding. Achieves 35% accuracy gains over RAG baselines with 99.9% smaller storage overhead."
---

# Multi-Agent Memory System: Persistent Knowledge Management for Contextual Agents

Traditional LLM-based agents struggle with true long-term memory, treating each conversation as isolated and lacking user context. This creates agents that cannot genuinely personalize responses or learn from interaction history. MIRIX solves this by organizing agent memory into specialized, managed components that work together—semantic knowledge about entities, episodic events with timestamps, procedural task instructions, and secure storage for sensitive information—enabling agents to automatically retrieve and inject relevant context into every response.

When building agents that need to remember user preferences, learn from past interactions, or maintain consistent understanding of complex domains, a unifying memory architecture prevents fragmented context and enables sophisticated reasoning grounded in accumulated experience. Multi-agent coordination ensures memory operations scale without bottlenecking—different agents handle different memory types in parallel, much like how human cognition separates fact recall from skill execution.

## Core Concept

MIRIX implements a six-tier memory hierarchy managed by specialized agents. Rather than storing everything in a flat text index, it separates concerns: episodic memory timestamps specific events and interactions; semantic memory captures reusable facts and entity relationships; procedural memory stores task instructions and patterns; resource memory manages external documents; core memory holds frequently-accessed summaries; and the knowledge vault secures sensitive information. Before responding to any query, the system generates potential relevant topics, automatically retrieves matching memories from each tier, and injects this context into the system prompt. An orchestrating Meta Memory Manager coordinates eight total agents—one per memory type plus a chat interface agent—preventing conflicts and ensuring consistent memory operations.

## Architecture Overview

- **Episodic Memory Agent**: Stores timestamped events, interactions, and user activities with temporal context for sequence-aware retrieval
- **Semantic Memory Agent**: Organizes facts, entities, relationships, and concepts using hierarchical knowledge representation
- **Procedural Memory Agent**: Manages task workflows, instructions, domain-specific patterns, and execution strategies
- **Resource Memory Agent**: Indexes external documents, files, and attachments with efficient document retrieval
- **Core Memory Agent**: Maintains summaries of high-frequency information, user preferences, and essential context
- **Knowledge Vault Agent**: Secures sensitive information (credentials, personal data) with access control and encryption
- **Chat Agent**: Interfaces with users and formats responses based on memory augmentation
- **Meta Memory Manager**: Orchestrates all agents, ensures consistency, handles conflicts, and manages memory lifecycle

## Implementation

This example demonstrates building a persistent conversation agent with automatic context injection. The system generates relevant topics before each response, retrieves memories across all tiers, and augments the system prompt.

```python
# Memory system initialization with multi-agent coordination
class MemorySystem:
    def __init__(self):
        self.episodic = EpisodicMemory()  # Timestamped events
        self.semantic = SemanticMemory()  # Facts and relationships
        self.procedural = ProceduralMemory()  # Task patterns
        self.resource = ResourceMemory()  # Documents/files
        self.core = CoreMemory()  # High-frequency summaries
        self.vault = KnowledgeVault()  # Sensitive data
        self.meta_manager = MetaMemoryManager(self)

    def store_interaction(self, user_id, user_query, agent_response, context_data):
        """Store interaction across multiple memory tiers with metadata."""
        timestamp = datetime.now()
        interaction_id = str(uuid.uuid4())

        # Episodic: timestamp the interaction
        self.episodic.add(
            interaction_id=interaction_id,
            user_id=user_id,
            query=user_query,
            response=agent_response,
            timestamp=timestamp,
            metadata=context_data
        )

        # Extract and store semantic facts from interaction
        entities = extract_entities(user_query, agent_response)
        for entity in entities:
            self.semantic.add_entity(entity_name=entity.name,
                                    entity_type=entity.type,
                                    relationships=entity.relations)

        # Store any procedural patterns mentioned
        procedures = extract_procedures(agent_response)
        for proc in procedures:
            self.procedural.add(procedure_name=proc.name,
                               steps=proc.steps,
                               context=proc.context)

        return interaction_id
```

This example shows the active retrieval mechanism that automatically injects context before generating responses. The system identifies high-uncertainty topics and pulls memories from all tiers.

```python
def augment_with_retrieved_context(self, user_id, current_query):
    """Generate relevant topics and retrieve matching memories across all tiers."""

    # Generate candidate topics from query (what might be relevant?)
    candidate_topics = self.meta_manager.generate_topics(current_query)

    # Retrieve from each memory tier
    retrieved_context = {}

    # Episodic: recent interactions on this topic
    episodic_results = self.episodic.retrieve(
        user_id=user_id,
        topics=candidate_topics,
        time_window_days=30,
        limit=5
    )
    retrieved_context['episodic'] = episodic_results

    # Semantic: relevant facts and entities
    semantic_results = self.semantic.retrieve(
        topics=candidate_topics,
        entity_types=['person', 'project', 'concept'],
        relevance_threshold=0.7,
        limit=10
    )
    retrieved_context['semantic'] = semantic_results

    # Procedural: relevant task patterns
    procedural_results = self.procedural.retrieve(
        topics=candidate_topics,
        user_context=self.core.get_user_profile(user_id),
        limit=5
    )
    retrieved_context['procedural'] = procedural_results

    # Resource: relevant documents
    resource_results = self.resource.retrieve(
        topics=candidate_topics,
        user_id=user_id,
        limit=3
    )
    retrieved_context['resource'] = resource_results

    # Build augmented system prompt
    augmented_system_prompt = self._build_augmented_prompt(
        base_prompt="You are a helpful assistant.",
        retrieved_context=retrieved_context,
        user_profile=self.core.get_user_profile(user_id)
    )

    return augmented_system_prompt
```

This example demonstrates secure sensitive data storage and access patterns. The vault enforces access control and audit logging.

```python
def store_sensitive_info(self, user_id, info_type, data, access_policy):
    """Store sensitive information with encryption and access control."""

    encrypted_data = self.vault.encrypt(
        data=data,
        user_id=user_id,
        key=self._get_user_key(user_id)
    )

    self.vault.add(
        info_type=info_type,  # 'credentials', 'personal_data', etc.
        encrypted_data=encrypted_data,
        user_id=user_id,
        access_policy=access_policy,
        audit_log=True
    )

def retrieve_sensitive_info(self, user_id, info_type, requester_agent):
    """Retrieve sensitive data only if access policy permits."""

    # Check access policy
    if not self.vault.check_access(user_id, info_type, requester_agent):
        raise PermissionError(f"Agent {requester_agent} cannot access {info_type}")

    # Log access for audit
    self.vault.log_access(user_id, info_type, requester_agent)

    encrypted_data = self.vault.get(user_id, info_type)
    return self.vault.decrypt(encrypted_data, self._get_user_key(user_id))
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Episodic retention period | 90 days | Balance memory coverage vs. storage |
| Semantic entity limit | 1000 per user | Prevent knowledge graph explosion |
| Topic generation count | 5-10 | Coverage without over-retrieval |
| Relevance threshold | 0.7 | Filter low-confidence matches |
| Core memory summary interval | 7 days | Compress old episodic data |
| Vault encryption | AES-256-GCM | Industry-standard security |

**When to use:** Multi-agent memory systems excel for personal assistants, domain-specific agents managing complex user relationships, educational systems tracking student progress, and applications requiring genuine personalization. Use when your agent needs to reference past interactions, learn user preferences over time, or maintain sophisticated context.

**When NOT to use:** Avoid this architecture for stateless query-response systems, single-conversation chatbots, or domains where user privacy regulations (like GDPR) prohibit long-term behavioral storage. Don't implement full memory systems if your queries never depend on historical context. Simpler prompt engineering or session-scoped caching may suffice.

**Common pitfalls:** Storing redundant information across multiple memory tiers wastes space—extract and normalize facts. Not implementing decay/cleanup causes stale memories to dominate retrieval. Ignoring access control in the knowledge vault leaks sensitive data. Over-aggressive topic generation retrieves irrelevant memories, degrading response quality. Failing to version memories makes debugging user-reported inconsistencies impossible.

## Reference

Liang, Z., Song, L., Yang, L., et al. (2025). MIRIX: Multi-Agent Memory System for LLM-Based Agents. arXiv preprint arXiv:2507.07957. https://arxiv.org/abs/2507.07957
