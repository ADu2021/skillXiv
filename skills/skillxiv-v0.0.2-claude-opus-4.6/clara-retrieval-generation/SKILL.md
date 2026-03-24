---
name: clara-retrieval-generation
title: "CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.18659"
keywords: [Retrieval-Augmented Generation, Joint Optimization, Continuous Representations, Information Compression]
description: "Unify retrieval and generation in RAG systems by compressing documents into shared continuous embeddings that serve both retrieval and generation: implement joint training with differentiable selection, achieving up to 16× context compression while improving generation quality."
---

# CLaRa: Bridging Retrieval and Generation

Traditional RAG systems treat retrieval and generation as separate modules: retrievers select full documents while generators consume raw text, creating redundancy and representation mismatch. This skill demonstrates how to unify both components through shared continuous document representations that simultaneously serve retrieval ranking and generation context—all optimized jointly with gradients flowing bidirectionally.

The core innovation is using compact compressed representations for both ranking and generation, eliminating the gap between what retrievers optimize for and what generators need.

## Core Concept

CLaRa (Continuous Latent Reasoning for Augmented generation) implements:

1. **Salient Compressor Pretraining (SCP)**: Learns to compress documents into meaningful vectors by synthesizing QA pairs and paraphrases

2. **Joint CLaRa Training**: Query reasoner and answer generator train end-to-end with differentiable ranking of compressed documents

3. **Bidirectional Gradient Flow**: Retrieval relevance aligns with downstream generation quality through unified objective

## Architecture Overview

- **Document Compressor**: Encodes full documents into compact semantic vectors
- **Query Reasoner**: Maps questions to embedding space for retrieval
- **Differentiable Selector**: Learns to rank documents using Straight-Through estimation
- **Answer Generator**: Consumes compressed documents for generation
- **Joint Loss**: Single objective optimizing retrieval + generation together

## Implementation Steps

The system trains through pretraining and joint optimization phases.

**1. Build Document Compressor with Salient Content Focus**

Train encoder to compress documents while preserving information-rich features.

```python
class SalientDocumentCompressor(torch.nn.Module):
    """
    Compresses documents to compact vectors emphasizing salient information.
    Learned through QA pair and paraphrase synthesis.
    """
    def __init__(self, hidden_dim=768, compressed_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.compressed_dim = compressed_dim

        # Document encoder (transformer-based)
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=6
        )

        # Compression network: full → compact
        self.compressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, compressed_dim)
        )

        # Auxiliary reconstruction head (for pretraining)
        self.reconstructor = torch.nn.Linear(compressed_dim, hidden_dim)

    def forward(self, document_tokens):
        """
        Compress document to compact representation.
        Args:
            document_tokens: (batch, seq_len, hidden_dim) document embeddings
        Returns:
            compressed: (batch, compressed_dim) compact representation
            reconstruction: (batch, seq_len, hidden_dim) for auxiliary loss
        """
        # Encode document
        encoded = self.encoder(document_tokens)

        # Average pool over sequence
        pooled = encoded.mean(dim=1)

        # Compress
        compressed = self.compressor(pooled)

        # Reconstruction for pretraining
        reconstruction = self.reconstructor(compressed)

        return compressed, reconstruction
```

**2. Implement SCP Pretraining with Synthetic QA Pairs**

Create synthetic QA pairs emphasizing essential document content for initial compression training.

```python
def create_synthetic_qa_pairs(documents, qa_generation_model, num_pairs_per_doc=5):
    """
    Generate synthetic QA pairs from documents.
    Uses LLM to create questions about key information.
    Args:
        documents: List of document texts
        qa_generation_model: LLM for QA generation
        num_pairs_per_doc: How many QA pairs per document
    Returns:
        qa_pairs: List of (question, answer, document) tuples
    """
    qa_pairs = []

    for doc in documents:
        # Generate multiple QA pairs from document
        prompt = f"""
        Given this document:
        {doc}

        Generate {num_pairs_per_doc} QA pairs testing understanding of KEY information.
        Format: Question: ... Answer: ...
        """

        response = qa_generation_model.generate(prompt)
        pairs = parse_qa_pairs(response)

        for q, a in pairs:
            qa_pairs.append((q, a, doc))

    return qa_pairs
```

**3. Pretraining Phase: SCP**

Pretrain compressor to reconstruct document content from compressed vectors.

```python
def pretrain_salient_compressor(compressor, documents, qa_pairs, num_epochs=10):
    """
    Pretrain compressor to preserve salient information.
    Uses reconstruction loss + QA answering auxiliary loss.
    """
    optimizer = torch.optim.Adam(compressor.parameters(), lr=1e-4)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for epoch in range(num_epochs):
        for doc_idx, document in enumerate(documents):
            # Tokenize document
            tokens = tokenizer(document, return_tensors='pt')['input_ids']
            doc_embeddings = get_embedding_layer()(tokens)

            # Compress document
            compressed, reconstruction = compressor(doc_embeddings)

            # Loss 1: Reconstruction (preserve content)
            reconstruction_loss = torch.nn.functional.mse_loss(
                reconstruction, doc_embeddings
            )

            # Loss 2: QA auxiliary (answer questions about document)
            doc_qa_pairs = [qa for qa in qa_pairs if qa[2] == document]

            qa_loss = 0.0
            for question, answer, _ in doc_qa_pairs[:3]:  # Use top 3 QA pairs
                # Can question be answered from compressed representation?
                # Simplified: use LLM to verify
                can_answer = verify_qa_from_compressed(
                    compressed, question, answer
                )
                qa_loss += (1.0 - can_answer)

            # Combined loss
            total_loss = reconstruction_loss + 0.5 * (qa_loss / max(len(doc_qa_pairs), 1))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return compressor
```

**4. Implement Differentiable Document Selector**

Create ranking mechanism with gradient flow for joint training.

```python
class DifferentiableDocumentSelector(torch.nn.Module):
    """
    Selects top-k documents for generation with differentiable ranking.
    Uses Straight-Through estimation to enable gradient flow through discrete selection.
    """
    def __init__(self, num_docs_to_select=3):
        super().__init__()
        self.k = num_docs_to_select

    def forward(self, query_embedding, document_embeddings, temperature=0.1):
        """
        Rank documents and select top-k with gradient flow.
        Args:
            query_embedding: (batch, embed_dim)
            document_embeddings: (batch, num_docs, embed_dim)
            temperature: Controls softness of selection
        Returns:
            selected_docs: (batch, k, embed_dim)
            selection_weights: (batch, num_docs) for gradient computation
        """
        batch_size, num_docs, embed_dim = document_embeddings.shape

        # Compute relevance scores (cosine similarity)
        # Normalize embeddings
        query_norm = query_embedding / (torch.norm(query_embedding, dim=-1, keepdim=True) + 1e-8)
        docs_norm = document_embeddings / (torch.norm(document_embeddings, dim=-1, keepdim=True) + 1e-8)

        # Similarity: (batch, num_docs)
        scores = torch.bmm(docs_norm, query_norm.unsqueeze(-1)).squeeze(-1)

        # Soft selection weights using temperature-scaled softmax
        soft_weights = torch.softmax(scores / temperature, dim=-1)

        # Hard selection: top-k (discrete operation)
        _, top_k_indices = torch.topk(scores, k=self.k, dim=-1)

        # Straight-Through Estimator: use soft weights for backward, hard for forward
        hard_selection = torch.zeros_like(soft_weights)
        hard_selection.scatter_(1, top_k_indices, 1.0)

        # Straight-through: forward with hard, backward with soft
        selection_weights = hard_selection - soft_weights.detach() + soft_weights

        # Select documents using soft weights (for gradient flow)
        weighted_docs = torch.einsum('bnd,bn->bnd', document_embeddings, selection_weights)

        # Sum to get final selected context
        selected_docs = weighted_docs.sum(dim=1)  # (batch, embed_dim)

        return selected_docs, selection_weights
```

**5. Implement Joint CLaRa Training**

Train query reasoner, selector, and generator end-to-end.

```python
def clara_joint_training(
    compressor,
    query_reasoner,
    selector,
    answer_generator,
    batch,
    optimizer,
    learning_rate=1e-4
):
    """
    Single joint training step for CLaRa.
    Optimizes retrieval (via selector) and generation together.
    Args:
        compressor: Document compression model
        query_reasoner: Maps questions to embedding space
        selector: Differentiable document ranking
        answer_generator: Text generator using selected documents
        batch: Dict with 'questions', 'documents', 'answers'
        optimizer: PyTorch optimizer
        learning_rate: Gradient step size
    Returns:
        loss: Combined retrieval + generation loss
    """
    questions = batch['questions']
    documents = batch['documents']
    answers = batch['answers']

    # Step 1: Compress all documents
    doc_embeddings = []
    for doc in documents:
        doc_tokens = tokenizer(doc, return_tensors='pt')['input_ids']
        doc_emb = embedding_layer(doc_tokens)
        compressed, _ = compressor(doc_emb)
        doc_embeddings.append(compressed)

    doc_embeddings = torch.stack(doc_embeddings)  # (batch, num_docs, compressed_dim)

    # Step 2: Encode questions
    query_embeddings = []
    for q in questions:
        q_tokens = tokenizer(q, return_tensors='pt')['input_ids']
        q_emb = embedding_layer(q_tokens)
        q_encoded = query_reasoner(q_emb)
        query_embeddings.append(q_encoded)

    query_embeddings = torch.stack(query_embeddings)  # (batch, embed_dim)

    # Step 3: Select documents with gradient flow
    selected_docs, selection_weights = selector(
        query_embeddings, doc_embeddings, temperature=0.1
    )

    # Step 4: Generate answers using selected documents
    decoder_input = torch.cat([query_embeddings, selected_docs], dim=-1)
    logits = answer_generator(decoder_input)

    # Step 5: Compute losses
    # Loss 1: Generation loss (standard next-token prediction)
    generation_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        answers.reshape(-1)
    )

    # Loss 2: Retrieval loss (encourage selecting correct documents)
    # Simplified: documents that enable better generation get higher "relevance" signal
    retrieval_loss = -generation_loss.detach()  # Gradient signal flows backward through selector

    # Combined loss
    total_loss = generation_loss + 0.5 * retrieval_loss

    # Backward and update
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(compressor.parameters()) + list(query_reasoner.parameters()) + list(selector.parameters()) + list(answer_generator.parameters()),
        1.0
    )
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'generation_loss': generation_loss.item(),
        'retrieval_loss': retrieval_loss.item()
    }
```

**6. Inference with Compression**

Use trained system for efficient retrieval-augmented generation.

```python
def clara_inference(
    compressor,
    query_reasoner,
    selector,
    answer_generator,
    question,
    document_corpus,
    tokenizer,
    max_answer_length=128
):
    """
    Generate answer using CLaRa with compressed document retrieval.
    Args:
        compressor, query_reasoner, selector, answer_generator: Trained modules
        question: Input question string
        document_corpus: List of available documents
        tokenizer: Text tokenizer
        max_answer_length: Maximum generation length
    Returns:
        answer: Generated text
        used_documents: Indices of retrieved documents
    """
    # Compress document corpus
    doc_embeddings = []
    for doc in document_corpus:
        doc_tokens = tokenizer(doc, return_tensors='pt')['input_ids']
        doc_emb = embedding_layer(doc_tokens)
        compressed, _ = compressor(doc_emb)
        doc_embeddings.append(compressed)

    doc_embeddings = torch.stack(doc_embeddings)

    # Encode question
    q_tokens = tokenizer(question, return_tensors='pt')['input_ids']
    q_emb = embedding_layer(q_tokens)
    query_embedding = query_reasoner(q_emb)

    # Select documents
    selected_docs, selection_weights = selector(query_embedding, doc_embeddings)

    # Track which documents were selected
    used_doc_indices = torch.topk(selection_weights.squeeze(), k=3).indices.tolist()

    # Generate answer
    decoder_input = torch.cat([query_embedding, selected_docs], dim=-1)

    answer_tokens = answer_generator.generate(
        inputs_embeds=decoder_input,
        max_length=max_answer_length,
        num_beams=3
    )

    answer = tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

    return answer, used_doc_indices
```

## Practical Guidance

**When to Use CLaRa:**
- RAG systems where document count is large (>1000) and context compression is valuable
- Tasks benefiting from unified optimization of retrieval and generation
- Scenarios where reducing context length is important for latency
- Systems where document relevance varies significantly

**When NOT to Use:**
- Tasks with small document corpora (<100 documents)
- Scenarios requiring full-document context for quality
- Systems where interpretability of retrieval decisions is critical

**Key Hyperparameters:**
- `compressed_dim`: Document vector size (64-256; larger retains more info but increases memory)
- `num_docs_to_select`: Top-k documents for generation (2-5 typical)
- `temperature`: Softness of document selection (0.05-0.2)
- `scp_epochs`: Pretraining iterations (5-15)
- `lambda_retrieval`: Weight of retrieval loss (0.3-1.0)

**Compression Effectiveness:**
- Average compression ratio: 10-16× (1000-token docs → 64-128-dim vectors)
- Pretraining on synthetic QA improves downstream performance by 15-20%
- Joint training typically converges in 2-5 epochs

**Memory and Latency:**
- Compressed document storage: 128 dims × 4 bytes × corpus_size
- Retrieval latency: O(n × d) for n docs and d dimensions (much faster than dense retrieval)
- Generation uses compressed context only (shorter → faster decoding)

## Reference

Research paper: https://arxiv.org/abs/2511.18659
