---
name: visplay-self-evolving-vlms
title: "VisPlay: Self-Evolving VLMs from Images via Dual-Role GRPO"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.15661"
keywords: [Vision-Language Models, Self-Evolution, GRPO, Automatic Curriculum, Unlabeled Images]
description: "Enable VLMs to self-improve from unlabeled images via dual-role framework—questioner generates challenging visual questions while reasoner answers them, trained jointly with GRPO using difficulty and diversity rewards."
---

# Enable Vision-Language Models to Self-Improve from Unlabeled Images

Vision-language models typically require supervised fine-tuning on annotated data. VisPlay breaks this dependency via a **dual-role self-evolution framework**: the model splits into an Image-Conditioned Questioner (generates visual questions) and a Multimodal Reasoner (answers them), trained jointly with GRPO. The system autonomously creates an automatic curriculum from unlabeled images, progressively increasing difficulty.

No annotations needed—the questioner and reasoner co-evolve, with the questioner learning to generate harder questions as the reasoner improves, creating a self-reinforcing loop of increasing capability.

## Core Concept

Supervised learning for VLMs requires annotated data (image-question-answer triples), which is expensive. VisPlay observes that annotators follow a natural curriculum: they ask easier questions first, then harder ones as models improve. VisPlay automates this:

1. **Questioner Role**: Given an image, generate a challenging visual question about it
2. **Reasoner Role**: Answer the question using both image and question context

Both roles start as the same base model. During training:
- Questioner learns to generate questions of optimal difficulty (50% confidence for reasoner)
- Reasoner learns to answer increasingly difficult questions

They're trained jointly with GRPO, using composite rewards (uncertainty reward, diversity, format constraints). This creates an automatic curriculum without human annotation.

## Architecture Overview

- **Dual-Role Architecture**: Single base model splits into Questioner and Reasoner for alternating training
- **Questioner Rewards**: Uncertainty (target 50% reasoner confidence), diversity (penalize redundancy), format (enforce structure)
- **Reasoner Rewards**: Binary (correct/incorrect) based on pseudo-labels from majority voting on reasoner outputs
- **GRPO Training**: Group Relative Policy Optimization for stable reward signal without separate value network
- **Curriculum Learning**: Automatic difficulty progression as reasoner improves, questioner escalates

## Implementation Steps

**Step 1: Define Questioner and Reasoner Roles.** Split model into two heads.

```python
import torch
import torch.nn as nn

class DualRoleVLM(nn.Module):
    """
    Vision-language model with questioner and reasoner roles.
    Both use shared backbone; different heads for generation type.
    """
    def __init__(self, base_model_name='clip-vit-base', hidden_dim=768):
        super().__init__()

        # Shared vision-language backbone
        self.backbone = load_vision_language_model(base_model_name)

        # Role-specific heads
        self.questioner_head = nn.Linear(hidden_dim, hidden_dim)
        self.reasoner_head = nn.Linear(hidden_dim, hidden_dim)

        # Shared decoder for generation
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(hidden_dim, nhead=8),
            num_layers=6
        )

        self.output_projection = nn.Linear(hidden_dim, 32000)  # Vocab size

    def forward_questioner(self, images):
        """
        Generate question given image.
        Returns: question tokens and generation logits
        """
        # Encode image
        img_features = self.backbone.encode_image(images)  # (batch, hidden_dim)

        # Route through questioner head
        questioner_features = self.questioner_head(img_features)

        # Generate question via autoregressive decoding
        questions, logits = self._generate_sequence(
            questioner_features,
            max_length=20,
            prompt="Question: "
        )

        return questions, logits

    def forward_reasoner(self, images, questions):
        """
        Answer question given image and question.
        Returns: answer tokens and generation logits
        """
        # Encode image and question
        img_features = self.backbone.encode_image(images)
        question_tokens = self.backbone.encode_text(questions)

        # Route through reasoner head
        reasoner_features = self.reasoner_head(img_features)

        # Concatenate with question
        combined_features = torch.cat([reasoner_features, question_tokens], dim=1)

        # Generate answer
        answers, logits = self._generate_sequence(
            combined_features,
            max_length=30,
            prompt="Answer: "
        )

        return answers, logits

    def _generate_sequence(self, features, max_length, prompt=""):
        """
        Autoregressive generation helper.
        """
        batch_size = features.shape[0]
        tokens = []
        logits_list = []

        # Decode autoregressively
        for _ in range(max_length):
            # Decoder forward pass
            output = self.decoder(features.unsqueeze(0))  # (1, batch, hidden)
            output_logits = self.output_projection(output[-1])  # (batch, vocab)

            # Sample token (greedy or sampling)
            next_token = output_logits.argmax(dim=-1)
            tokens.append(next_token)
            logits_list.append(output_logits)

            # Check for end-of-sequence
            if (next_token == self.get_eos_token()).all():
                break

        return torch.stack(tokens, dim=1), torch.stack(logits_list, dim=1)
```

**Step 2: Define Reward Functions for Questioner and Reasoner.**

```python
class DualRoleRewardFunction:
    def __init__(self):
        self.difficulty_history = []

    def reward_questioner(self, question, reasoner_confidence, previous_confidence):
        """
        Reward questioner for generating questions of optimal difficulty.
        Optimal = 50% reasoner confidence (neither too easy nor too hard)
        """
        # Uncertainty reward: target 50% confidence
        uncertainty_reward = 1.0 - 2 * abs(0.5 - reasoner_confidence)

        # Difficulty progression reward
        # Bonus if difficulty increased (shows curriculum learning)
        difficulty_increase = reasoner_confidence - previous_confidence
        progression_reward = 0.5 if difficulty_increase < 0 else 0.0  # Negative = harder

        # Format/structure reward (enforced via constraints)
        format_reward = 0.5  # Assume format is valid; penalize invalid separately

        total_reward = 0.6 * uncertainty_reward + 0.3 * progression_reward + 0.1 * format_reward
        return total_reward

    def reward_reasoner(self, answer, ground_truth, confidence):
        """
        Reward reasoner for correct answers.
        Use majority-voted pseudo-labels as ground truth.
        """
        is_correct = self._evaluate_answer(answer, ground_truth)

        if is_correct:
            # Bonus for high confidence when correct
            return 1.0 + 0.2 * confidence
        else:
            # Penalty, worse with high confidence (overconfidence)
            return -1.0 - 0.2 * confidence

    def _evaluate_answer(self, answer, ground_truth):
        """
        Simple evaluation: exact match or semantic similarity.
        For unlabeled data, use majority voting on k reasoner outputs.
        """
        # Placeholder: use BERT embedding similarity
        return answer.lower().strip() == ground_truth.lower().strip()

    def get_difficulty_score(self, confidences):
        """Track difficulty progression over time."""
        avg_confidence = sum(confidences) / len(confidences)
        self.difficulty_history.append(avg_confidence)
        return avg_confidence
```

**Step 3: GRPO Training Loop.** Train questioner and reasoner jointly.

```python
def train_dual_role_with_grpo(
    model, image_dataloader, num_iterations=10000, lr=1e-5
):
    """
    Train vision-language model with dual-role GRPO.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    reward_fn = DualRoleRewardFunction()

    for iteration in range(num_iterations):
        for images in image_dataloader:
            batch_size = images.shape[0]

            # ===== Questioner Phase =====
            # Generate questions
            questions, questioner_logits = model.forward_questioner(images)

            # ===== Reasoner Phase =====
            # Answer questions to get rewards
            answers, reasoner_logits = model.forward_reasoner(images, questions)

            # Compute reasoner confidence (softmax on output logits)
            reasoner_confidence = torch.softmax(reasoner_logits, dim=-1).max(dim=-1).values.mean()

            # ===== Pseudo-Labeling =====
            # For unlabeled data, use majority voting on multiple reasoner runs
            pseudo_labels = self._majority_vote_labels(
                model, images, questions, num_runs=5
            )

            # ===== Reward Computation =====
            questioner_rewards = []
            reasoner_rewards = []

            for i in range(batch_size):
                # Questioner reward (difficulty calibration)
                q_reward = reward_fn.reward_questioner(
                    questions[i],
                    reasoner_confidence.item(),
                    prev_confidence=0.5  # Running average
                )
                questioner_rewards.append(q_reward)

                # Reasoner reward (answer correctness)
                r_reward = reward_fn.reward_reasoner(
                    answers[i],
                    pseudo_labels[i],
                    confidence=reasoner_confidence.item()
                )
                reasoner_rewards.append(r_reward)

            questioner_rewards = torch.tensor(questioner_rewards)
            reasoner_rewards = torch.tensor(reasoner_rewards)

            # ===== GRPO: Group Relative Policy Optimization =====
            # Compute advantages as relative performance within group
            q_baseline = questioner_rewards.mean()
            q_advantages = (questioner_rewards - q_baseline) / (questioner_rewards.std() + 1e-8)

            r_baseline = reasoner_rewards.mean()
            r_advantages = (reasoner_rewards - r_baseline) / (reasoner_rewards.std() + 1e-8)

            # Policy gradient losses
            q_log_probs = torch.log_softmax(questioner_logits, dim=-1).mean(dim=1)
            q_loss = -(q_log_probs * q_advantages).mean()

            r_log_probs = torch.log_softmax(reasoner_logits, dim=-1).mean(dim=1)
            r_loss = -(r_log_probs * r_advantages).mean()

            total_loss = q_loss + r_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if iteration % 100 == 0:
                print(f"Iter {iteration}: q_loss={q_loss:.4f}, r_loss={r_loss:.4f}, "
                      f"confidence={reasoner_confidence:.3f}")

    def _majority_vote_labels(self, model, images, questions, num_runs=5):
        """Pseudo-label generation via majority voting."""
        all_answers = []

        for _ in range(num_runs):
            answers, _ = model.forward_reasoner(images, questions)
            all_answers.append(answers)

        # Majority vote
        labels = []
        for i in range(images.shape[0]):
            answers_for_sample = [ans[i] for ans in all_answers]
            majority_answer = self._find_majority(answers_for_sample)
            labels.append(majority_answer)

        return labels
```

## Practical Guidance

**When to Use:** Self-improvement of VLMs when labeled data is scarce or expensive. Use for domains with many unlabeled images (web, internal datasets).

**Hyperparameters:**
- Uncertainty target: 50% confidence optimal; adjust to 40-60% range for harder curriculum
- Group size for GRPO: batch size ≥ 32; larger batches stabilize advantage estimation
- Confidence thresholds for reasoner: 0.25–0.75 for filtering moderately-confident examples; avoids learning from noise

**Pitfalls:**
- **Pseudo-label quality**: Majority voting with small k (< 5 runs) is noisy; use k ≥ 5, consider semantic clustering
- **Question degeneracy**: Questioner may converge to trivial questions; use diversity penalty aggressively
- **Confidence calibration**: Model confidence may not correlate with correctness; validate on labeled data periodically
- **Convergence oscillation**: Questioner and reasoner can oscillate; use EMA for stability

**When NOT to Use:** Tasks with abundant labeled data (supervised fine-tuning is simpler); domains where pseudo-labels are unreliable.

**Integration**: Works with any VLM backbone (CLIP, LLaVA, GPT-4V-style models); no architectural changes needed.

---
Reference: https://arxiv.org/abs/2511.15661
