---
name: progressive-citation-grounded-dialogue
title: "Progressive Training for Citation-Grounded Dialogue: Reducing Hallucination in English-Hindi LLMs"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.18911"
keywords: [Citation-Grounded Generation, Hallucination Reduction, Dialogue Systems, Multilingual Models, Structured Generation]
description: "Eliminate hallucination via four-stage progressive training: multilingual adaptation → English dialogue SFT → bilingual SFT → GRPO alignment. Achieve 0.0% hallucination rate for encoder-decoder models using structured citation markers and knowledge-source attribution, with automatic transfer of citation format across languages."
---

# Progressive Citation-Grounded Dialogue

## Ranked Findings

**Finding 1 (Highest Impact): Citation-Grounded SFT Eliminates Hallucination**
- Stages 2+ achieve 0.0% hallucination (encoder-decoder models under NLI-based evaluation)
- Structured format (citations to source passages) forces models to ground outputs
- No downstream GRPO alignment needed for hallucination elimination

**Finding 2: Encoder-Decoder Architecture Superior to Decoder-Only**
- Flan-T5 models: consistent 0.0% hallucination
- LLaMA-3.2-1B: language-selective citation failure (English works, Hindi fails)
- Mistral: larger decoders (7B) partially recover but never reach encoder-decoder parity
- Implication: model architecture matters as much as training procedure

**Finding 3: Smaller Models Can Match Larger Ones with Structured Training**
- Flan-T5-250M reaches English metrics of Flan-T5-780M after Stage 2
- Structured citation task requires <250M capacity if properly formatted
- Training recipe supersedes raw model size for hallucination control

**Finding 4: GRPO Alignment Provides Marginal Gains**
- Well-tuned SFT already achieves strong performance
- GRPO adds <2% improvement on citation-F1 and factuality scores
- Resource cost (reward modeling, RL training) not justified for this task
- Implication: structured generation tasks may not need RL fine-tuning

**Finding 5: Cross-Lingual Citation Transfer is Automatic**
- Models trained on English citation format automatically apply format to Hindi
- No need for language-specific instruction tuning for citation structure
- Citation formatting is learned as abstract pattern, not language-dependent

## Decision Checklist for Practitioners

1. **Choose Encoder-Decoder Foundation:** Start with encoder-decoder models (Flan-T5, mBART)
   - ✓ Encoder-decoder models exhibit 0.0% hallucination
   - ✗ Avoid decoder-only for critical applications unless scaling to 7B+

2. **Prepare Citation-Grounded Data:** Format SFT data with explicit source attributions
   - Format: "[Response text] [CITE: source_passage_id, source_passage_id]"
   - Verify source passages are included in context; no floating citations
   - Target: 1,000+ citation-grounded training examples minimum

3. **Run Stage 2 (English Dialogue SFT) First:**
   - Monitor for generation collapse (if occurs, will appear at Stage 2)
   - Verify 0.0% hallucination before proceeding to multilingual training
   - If collapse happens: reduce learning rate, increase training steps

4. **Add Stage 3 (Bilingual SFT) with Weighted Sampling:**
   - 60% target-language (Hindi), 40% English to prevent English forgetting
   - Verify citation format transfers to target language automatically
   - Accept minor performance drop in target language; citation format is preserved

5. **Skip Stage 4 (GRPO) Unless Additional Metrics Required:**
   - Hallucination already eliminated by Stage 2
   - GRPO only helpful if optimizing citation coverage (citing every relevant passage)
   - For factuality-only tasks, SFT sufficient

6. **Evaluation Strategy:**
   - Primary: NLI-based hallucination detection (0.0% target)
   - Secondary: Citation-F1 (precision and recall of citations)
   - Tertiary: BLEU, ROUGE (downstream task quality)
   - Sample evaluation: 100+ conversations with human verification of facts

## Conditions for Success

**Necessary Conditions:**

1. **Source Knowledge Must Be Available:** If knowledge sources aren't in model context, citation instructions won't work; system needs explicit passages to cite
2. **Structured Format Compliance:** Models must follow citation marker format (e.g., "[CITE: 1, 2]"); test format consistency during training
3. **Sufficient Training Data:** Minimum 1,000 citation-grounded dialogues; below this, models revert to ungrounded generation
4. **Encoder-Decoder Architecture:** Decoder-only models struggle with structured citation; if using decoder-only, scale to 7B+ and accept partial failure modes

**Sufficient Conditions (Optional Enhancements):**

1. **Multilingual Pretraining:** mBART or multilingual BERT helps Stage 1; monolingual models work but need more adaptation data
2. **GRPO Alignment:** Not necessary for hallucination elimination, but improves citation precision (cite only most relevant passages)
3. **Explicit Causal Grounding:** Use Integrated Gradients or occlusion analysis to verify citations actually influence model outputs (not just format compliance)

## Notable Phenomena

**Phenomenon 1: Generation Collapse and Recovery**
- Flan-T5-XL experienced complete generation failure at Stage 2 (outputting only "[CITE: X]")
- Recovered fully during Stage 3 bilingual training (bilingual context restabilized representations)
- Implication: citation format can destabilize some models; multilingual exposure helps recovery

**Phenomenon 2: Decoder-Only Dissociation**
- Larger decoder-only models (LLaMA-3.2-7B, Mistral-7B) learn citation FORMAT without causal grounding
- Models output correct citations but base responses on other factors, not cited passages
- Contrast: Encoder-decoder models inherently ground via cross-attention; citations reflect actual decision-making
- Implication: architecture enables or disables genuine grounding, not just format learning

**Phenomenon 3: Cross-Lingual Transfer Without Explicit Instruction**
- Models never see Hindi citation examples (Stage 2 English-only); automatically apply format to Hindi
- Transfer happens in Stage 3 without additional citation-specific instructions
- Suggests citation formatting learned as abstract structural pattern
- Opportunity: extend to zero-shot languages (untested in this work)

## Training Pipeline Details

**Stage 1 (Multilingual Adaptation):**
- English-Hindi parallel corpus (existing public datasets)
- Masked language modeling or back-translation
- Duration: 1-2 epochs

**Stage 2 (English Dialogue SFT):**
- Citation-grounded dialogues with English source passages
- Supervised fine-tuning with cross-entropy loss
- Monitor for generation collapse; reduce learning rate if detected
- Duration: 5-10 epochs or until validation plateau

**Stage 3 (Bilingual SFT):**
- Extend to Hindi with weighted sampling (60% Hindi, 40% English)
- Same citation format as Stage 2
- Higher learning rate than Stage 2 to prevent catastrophic forgetting
- Duration: 3-5 epochs

**Stage 4 (GRPO, Optional):**
- Group Relative Policy Optimization with citation-aware rewards
- Reward: bonus for citations covering diverse source passages, penalty for over-citing
- Only if optimizing for citation coverage/diversity
- Duration: 1-3 epochs

## Practical Implications

- **Hallucination Elimination at Scale:** Citation-grounded training transfers to longer conversations and complex topics
- **Multilingual Scalability:** Approach tested on English-Hindi; architecture generalizes to other language pairs
- **Production Deployment:** 0.0% hallucination under evaluation metrics suggests production-ready factuality
- **User Trust:** Explicit citations enable users to verify facts and trace reasoning
