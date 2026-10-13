# Fine-tuning: results (2026-10-13)

## Setup

**Distillation.** Each model is trained on the labels of the two LLM annotators, keeping only the reviews
where both agree.
- Data: **6,796 training and 754 validation** reviews, from 14 other productivity apps. Notion and the
  D2 dev set are excluded.
- Labels: Qwen3.8-27B and Gemma-4-31B agreed on 95.5% of the reviews.
- Heads: multi-task, with 3-class sentiment (class-weighted cross-entropy) plus a binary complaint head.
- Training: 4 epochs, bf16 autocast with fp32 master weights. Base models use lr 2e-5; large models
  use 1e-5 (RoBERTa) or 8e-6 (DeBERTa, XLM-R).
- Seeds: 3 per model. The best epoch is chosen on validation, never on Notion data.
- Compute: 21 runs on 6 shared L40S GPUs, about 50 minutes of wall time.
- Scripts:
  - `research/finetune.py` and `research/server/run_finetune.sh` for training;
  - `research/eval_finetuned.py` for evaluation, which uses the same metrics as the off-the-shelf
    comparison.

## Results (D1-human = the 100 human labels; D1-LLM = 294 LLM-consensus labels)

| system | val F1 | D1-human macro-F1 [95% CI] | acc | MCC | neg R/P | complaint F1 (human) | D1-LLM macro-F1 | D2-LLM neg P@0.30 |
|---|---|---|---|---|---|---|---|---|
| *off-the-shelf cardiffnlp (previous winner)* | – | *0.686 [0.57, 0.80]* | *0.84* | *0.70* | *0.90/0.87* | *– (no head)* | *0.693* | – |
| **cardiff-roberta-base, fine-tuned** | 0.907 | **0.763** [0.58, 0.89] | 0.86 | 0.74 | 0.90/0.91 | **0.96** | 0.893 | 0.90 |
| deberta-v3-large, fine-tuned | 0.926 | 0.762 [0.59, 0.88] | 0.86 | 0.73 | 0.91/0.90 | 0.95 | 0.940 | 0.91 |
| ensemble top-2 by val | 0.918 | 0.762 [0.59, 0.88] | 0.86 | 0.73 | 0.91/0.90 | 0.96 | 0.937 | 0.91 |
| roberta-large, fine-tuned | 0.918 | 0.755 [0.58, 0.88] | 0.85 | 0.72 | 0.90/0.90 | 0.95 | 0.937 | 0.91 |
| deberta-v3-base, fine-tuned | 0.925 | 0.751 [0.59, 0.87] | 0.86 | 0.74 | 0.90/0.91 | 0.95 | 0.853 | 0.90 |
| ensemble top-3 by val | 0.927 | 0.743 [0.58, 0.87] | 0.85 | 0.72 | 0.90/0.90 | 0.95 | 0.903 | 0.91 |
| clapai-xlmr-base, fine-tuned | 0.896 | 0.735 [0.57, 0.86] | 0.84 | 0.70 | 0.88/0.91 | 0.95 | 0.895 | 0.91 |
| xlmr-large, fine-tuned | 0.900 | 0.732 [0.58, 0.85] | 0.85 | 0.72 | 0.91/0.90 | 0.96 | 0.871 | 0.93 |
| mdeberta-v3-base, fine-tuned | 0.879 | 0.675 [0.56, 0.83] | 0.87 | 0.75 | 0.95/0.90 | 0.95 | 0.819 | 0.90 |
| *LLM reference (Qwen/Gemma)* | – | *0.807* | *0.90* | *0.81* | *0.95/0.92* | *0.86* | – | – |

Fine-tuned rows use seed ensembles. For single seeds, D1-human macro-F1 is mean ± sd over 3 seeds:

| Model | Mean ± sd |
|---|---|
| cardiff-roberta-base | 0.768 ± 0.009 |
| roberta-large | 0.758 ± 0.014 |
| deberta-v3-base | 0.755 ± 0.013 |
| deberta-v3-large | 0.746 ± 0.031 |
| xlmr-large | 0.736 ± 0.011 |
| clapai-xlmr-base | 0.735 ± 0.007 |
| mdeberta-v3-base | 0.710 ± 0.076 |

## Significance and where the gain comes from

**cardiffnlp off-the-shelf vs fine-tuned:**

| Set | Accuracy | Neutral F1 | McNemar |
|---|---|---|---|
| D1-LLM | 0.914 → 0.962 | 0.25 → 0.75 | **p = 0.003 (significant)** |
| D1-human | 0.84 → 0.86 | 0.31 → 0.55 | p = 0.69. The direction is the same, but with n = 100 the test has little power. |

**Larger models vs fine-tuned cardiffnlp:** not significant (DeBERTa-v3-large: p = 1.0 on D1-human, 0.18 on
D1-LLM). The larger models have 2.8–4.5× as many parameters and gain nothing measurable on human labels.

**Ensembles:** no gain (0.762 vs 0.763). Not adopted.

**Complaint head:**
- F1 0.96 against the human complaint labels. That is higher than the LLMs (0.86), which were trained
  as the teacher on other apps.
- It catches **11 of 11** complaints hidden inside *positive* reviews, which a sentiment label cannot
  express.

**Threshold check** with proper labels: negative precision at P(neg) ≥ 0.30 on D2-LLM is 0.90–0.93, so it
**passes** the 0.75 floor. The earlier failure came from noise in the star labels.

**Non-English** (5 reviews in D1, so only indicative):

| Model | Correct |
|---|---|
| off-the-shelf cardiffnlp | 0/5 |
| cardiffnlp, fine-tuned | 3/5 |
| clapAI, fine-tuned | **5/5** |

## Decision

- **English: `cardiffnlp/twitter-roberta-base-sentiment-latest` fine-tuned for multi-task**, seed 1, chosen on
  validation F1 (0.899).
  - It is significantly better than off-the-shelf on D1-LLM, and better in the same direction on
    D1-human.
  - It adds the complaint head.
  - By the pre-registered tie-break it is the smallest of the equally good models (125M parameters,
    about 30 ms per review on CPU).
- **Non-English: `clapAI/roberta-base-multilingual-sentiment` fine-tuned for multi-task**, seed 0, chosen on
  validation F1 (0.914).
- **Not adopted:** the larger models and the ensembles (no significant gain), and mDeBERTa (unstable across
  seeds, sd 0.076).
