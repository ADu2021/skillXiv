---
name: saffron-safety-scaling
title: "Saffron-1: Inference Scaling Paradigm for LLM Safety Assurance"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.06444"
keywords: [llm-safety, inference-scaling, reward-models, tree-search, adversarial-robustness]
description: "Implement Saffron's multifurcation reward model approach to achieve efficient inference-time safety scaling, improving robustness against prompt injection attacks while reducing computational overhead."
---

# Saffron-1: Inference Scaling Paradigm for LLM Safety Assurance

## Core Concept

Saffron-1 addresses a critical limitation in LLM safety: conventional inference scaling techniques fail to improve safety robustness effectively. The key insight is that frequent process reward model calls create computational overhead that undermines efficiency gains. Saffron replaces traditional single-output reward models with multifurcation reward models that predict rewards for all vocabulary tokens in a single forward pass, reducing required model evaluations from K calls to 1. Combined with conservative exploration constraints and efficient KV caching, Saffron achieves significantly lower attack success rates while maintaining computational efficiency.

## Architecture Overview

- **Multifurcation Reward Model (MRM)**: Predicts rewards for all vocabulary tokens simultaneously
- **Token-Level Supervision**: Partial supervision from pre-computed corpus rewards, not requiring full annotation
- **Conservative Exploration**: Restricts search to previously-seen tokens only
- **Trie-Based KV Caching**: Shares key-value caches across sequences with common prefixes
- **Safety Dataset**: Safety4M annotation with Llama Guard 3 on HH-RLHF corpus
- **Tree Search Integration**: Compatible with beam search and other inference algorithms

## Implementation

### Step 1: Create Multifurcation Reward Model

Build a reward model that predicts rewards for all tokens simultaneously:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class MultifurcationRewardModel(nn.Module):
    """
    Reward model that outputs reward for each vocabulary token.
    Single forward pass predicts reward[vocab_size] instead of K separate calls.
    """

    def __init__(self, base_model_name="meta-llama/Llama-2-7b-hf", vocab_size=32000):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.hidden_dim = self.base_model.config.hidden_size
        self.vocab_size = vocab_size

        # Reward head: projects hidden state to per-token rewards
        self.reward_head = nn.Linear(self.hidden_dim, vocab_size)

        # Optional: LoRA adaptation for efficiency
        self.use_lora = True
        if self.use_lora:
            self.apply_lora()

    def apply_lora(self, r=8, lora_alpha=16):
        """Apply LoRA to reduce parameters"""
        from peft import get_peft_model, LoraConfig

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.base_model = get_peft_model(self.base_model, lora_config)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass: single call outputs all token rewards.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            rewards: (batch_size, vocab_size) - reward for each token
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden state from final token
        last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
        final_hidden = last_hidden[:, -1, :]  # (batch_size, hidden_dim)

        # Compute reward for each vocabulary token
        rewards = self.reward_head(final_hidden)  # (batch_size, vocab_size)

        return rewards
```

### Step 2: Create Safety4M Training Dataset

Build the token-level supervision dataset:

```python
class Safety4MDataset:
    """
    Token-level safety rewards from HH-RLHF corpus.
    4 million tokens with pre-computed Llama Guard 3 annotations.
    """

    def __init__(self, corpus_path, reward_model_checkpoint, num_samples=4_000_000):
        self.corpus_path = corpus_path
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        self.num_samples = num_samples

        # Load Llama Guard 3 for labeling
        self.labeler = load_llama_guard_3()

    def create_token_level_rewards(self, texts):
        """
        Create token-level reward supervision.

        For each prefix in corpus, compute safety score and assign to next token.
        """
        dataset = []

        for text in texts:
            tokens = self.tokenizer.encode(text)

            # Compute reward (safety score) for each position
            for t in range(1, len(tokens)):
                prefix = self.tokenizer.decode(tokens[:t])
                next_token = tokens[t]

                # Llama Guard 3 safety score
                safety_score = self.labeler.compute_safety(prefix)

                # Assign reward to next token
                dataset.append({
                    'prefix_ids': torch.tensor(tokens[:t]),
                    'next_token': next_token,
                    'reward': safety_score  # Range: [0, 1]
                })

        return dataset

    def collate_batch(self, samples):
        """Batch with padding"""
        prefix_ids = torch.nn.utils.rnn.pad_sequence(
            [s['prefix_ids'] for s in samples],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        next_tokens = torch.tensor([s['next_token'] for s in samples])
        rewards = torch.tensor([s['reward'] for s in samples])

        return {
            'prefix_ids': prefix_ids,
            'next_tokens': next_tokens,
            'rewards': rewards
        }

def train_multifurcation_reward_model(model, dataset, epochs=3, lr=1e-5):
    """Train MRM with token-level supervision"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataset:
            prefix_ids = batch['prefix_ids'].to('cuda')
            next_tokens = batch['next_tokens'].to('cuda')
            rewards = batch['rewards'].to('cuda')

            optimizer.zero_grad()

            # Forward: get reward for each vocabulary token
            predicted_rewards = model(input_ids=prefix_ids)  # (batch_size, vocab_size)

            # Extract predicted reward for the actual next token
            batch_size = next_tokens.shape[0]
            predicted_next_rewards = predicted_rewards[
                torch.arange(batch_size), next_tokens
            ]  # (batch_size,)

            # MSE loss between predicted and ground truth rewards
            loss = torch.nn.functional.mse_loss(predicted_next_rewards, rewards)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataset):.4f}")
```

### Step 3: Implement Tree Search with Conservative Exploration

Build tree search that constrains exploration to safe token space:

```python
class SafeTreeSearch:
    def __init__(self, model, reward_model, tokenizer, beam_size=4):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.beam_size = beam_size

        # Track seen tokens for conservative exploration
        self.seen_tokens = set(range(1000))  # Common tokens

    def search(self, input_ids, max_depth=50, attack_success_threshold=0.5):
        """
        Tree search with safety-aware exploration.

        Restricts to seen tokens only to prevent unreliable predictions on
        out-of-distribution vocabulary.
        """
        # Initialize beam with starting sequence
        beam = [(input_ids, 0.0)]  # (sequence, accumulated_reward)

        for depth in range(max_depth):
            candidates = []

            for sequence, acc_reward in beam:
                # Get reward for all tokens
                with torch.no_grad():
                    rewards = self.reward_model(input_ids=sequence)  # (1, vocab_size)

                # Conservative exploration: only consider seen tokens
                safe_mask = torch.zeros(self.reward_model.vocab_size)
                for token_id in self.seen_tokens:
                    safe_mask[token_id] = 1.0

                # Mask out unseen tokens
                masked_rewards = rewards[0] * safe_mask - 1e9 * (1 - safe_mask)

                # Get top tokens
                top_rewards, top_tokens = torch.topk(masked_rewards, self.beam_size)

                # Create new sequences
                for reward, token_id in zip(top_rewards, top_tokens):
                    new_sequence = torch.cat([
                        sequence,
                        torch.tensor([[token_id.item()]])
                    ], dim=1)

                    new_reward = acc_reward + reward.item()
                    candidates.append((new_sequence, new_reward))

            # Keep top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_size]

            # Early stopping if attack success rate is too high
            attack_rate = self.estimate_attack_success_rate(beam)
            if attack_rate > attack_success_threshold:
                break

        # Return best sequence
        best_sequence, best_reward = beam[0]
        return best_sequence, best_reward

    def estimate_attack_success_rate(self, sequences):
        """Estimate likelihood of adversarial success"""
        # Use Llama Guard 3 or similar detector
        attack_count = 0

        for sequence, _ in sequences:
            text = self.tokenizer.decode(sequence[0])
            is_unsafe = self.reward_model.predict_unsafe(text)
            if is_unsafe:
                attack_count += 1

        return attack_count / len(sequences)
```

### Step 4: Implement Trie-Based KV Caching

Optimize memory usage for sequences with shared prefixes:

```python
class TrieKVCache:
    """
    Efficient KV caching for tree search using trie structure.

    Shares key-value caches across sequences with common prefixes.
    """

    def __init__(self):
        self.trie = {}
        self.cache = {}

    def get_cache(self, prefix_ids):
        """Retrieve cached KV for prefix"""
        key = tuple(prefix_ids.tolist())

        if key in self.cache:
            return self.cache[key]
        return None

    def store_cache(self, prefix_ids, kv_cache):
        """Store KV cache for prefix"""
        key = tuple(prefix_ids.tolist())
        self.cache[key] = kv_cache

    def forward_with_cache(self, model, input_ids, past_key_values=None):
        """Forward pass reusing cached KVs"""
        # Check if we have cached KVs for a prefix
        cached_kv = self.get_cache(input_ids[:, :-1])

        if cached_kv is not None:
            # Use cached KVs for all but last token
            outputs = model(
                input_ids=input_ids[:, -1:],
                past_key_values=cached_kv,
                use_cache=True
            )
        else:
            # Full forward pass
            outputs = model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True
            )

        # Store cache for future use
        self.store_cache(input_ids, outputs.past_key_values)

        return outputs
```

## Practical Guidance

- **Multifurcation Benefit**: Reduces reward model calls from K (beam size) to 1, dramatically improving efficiency
- **Token Supervision**: Uses existing corpus annotations; no need for full vocabulary labeling
- **Conservative Exploration**: Essential for safety; prevents searching unreliable token space
- **Attack Success Rate**: Typical improvements: 40.9% ASR (Saffron) vs. 58.2% (Best-of-N)
- **Training Data**: Safety4M dataset with Llama Guard 3 labels; can use other safety labelers
- **Beam Size**: 4-8 typical; larger beams increase safety but reduce speed
- **Computational Cost**: Trie-based caching reduces memory by 40-60% compared to naive tree search
- **Integration**: Works with existing LLMs without architectural changes

## Reference

- Multifurcation reward models parallelize vocabulary scoring while reducing forward passes
- Token-level supervision enables efficient training on unlabeled corpus data
- Conservative exploration fundamentally prevents out-of-distribution reward exploitation
- Trie-based caching is standard technique in sequence generation but particularly valuable for safety scaling
