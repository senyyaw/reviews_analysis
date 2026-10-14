# Sentiment model comparison: results (2026-10-13)

Produced by `research/sentiment_bakeoff.py`. Raw numbers are in `sentiment_bakeoff.json`. Protocol: README, Sentiment.

### D1-human (n=100)
| model | macro-F1 [95% CI] | acc | MCC | neg R / P (argmax) | neg R / P (@0.30) | complaint R (@0.30) | McNemar p vs best |
|---|---|---|---|---|---|---|---|
| llm:qwen3.8-27b | 0.807 [0.61, 0.92] | 0.90 | 0.81 | 0.95 / 0.92 | 0.95 / 0.92 | 0.82 | 0.146 |
| llm:gemma4-31b | 0.807 [0.61, 0.92] | 0.90 | 0.81 | 0.95 / 0.92 | 0.95 / 0.92 | 0.82 | 0.146 |
| cardiffnlp-roberta | 0.686 [0.57, 0.80] | 0.84 | 0.70 | 0.90 / 0.87 | 0.90 / 0.87 | 0.82 | 1.000 |
| ensemble | 0.654 [0.55, 0.76] | 0.79 | 0.63 | 0.86 / 0.89 | 0.93 / 0.84 | 0.84 | 0.332 |
| cardiffnlp-xlmt | 0.609 [0.51, 0.73] | 0.81 | 0.64 | 0.98 / 0.81 | 0.98 / 0.80 | 0.92 | 0.581 |
| clapai-xlmr | 0.596 [0.51, 0.70] | 0.78 | 0.60 | 0.81 / 0.90 | 0.83 / 0.86 | 0.74 | 0.146 |
| nlptown-mbert | 0.583 [0.51, 0.67] | 0.76 | 0.56 | 0.84 / 0.83 | 0.90 / 0.84 | 0.81 | 0.077 |
| vader | 0.527 [0.43, 0.63] | 0.68 | 0.48 | 0.62 / 0.92 | 0.62 / 0.92 | 0.51 | 0.004 |
| tfidf-logreg | 0.518 [0.44, 0.60] | 0.66 | 0.44 | 0.69 / 0.87 | 0.86 / 0.85 | 0.78 | 0.001 |
| always-negative | 0.245 [0.22, 0.27] | 0.58 | 0.00 | 1.00 / 0.58 | 1.00 / 0.58 | 1.00 | 0.000 |

### D1-LLM (n=294)
| model | macro-F1 [95% CI] | acc | MCC | neg R / P (argmax) | neg R / P (@0.30) | complaint R (@0.30) | McNemar p vs best |
|---|---|---|---|---|---|---|---|
| llm:qwen3.8-27b | 1.000 [1.00, 1.00] | 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | – | 0.000 |
| llm:gemma4-31b | 1.000 [1.00, 1.00] | 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | – | 0.000 |
| cardiffnlp-roberta | 0.693 [0.61, 0.78] | 0.91 | 0.83 | 0.95 / 0.94 | 0.95 / 0.94 | – | 1.000 |
| clapai-xlmr | 0.682 [0.62, 0.78] | 0.93 | 0.86 | 0.92 / 0.97 | 0.94 / 0.96 | – | 0.377 |
| cardiffnlp-xlmt | 0.647 [0.58, 0.75] | 0.88 | 0.78 | 0.97 / 0.87 | 0.97 / 0.85 | – | 0.215 |
| ensemble | 0.640 [0.60, 0.70] | 0.88 | 0.79 | 0.90 / 0.95 | 0.97 / 0.92 | – | 0.324 |
| nlptown-mbert | 0.608 [0.59, 0.63] | 0.88 | 0.77 | 0.90 / 0.92 | 0.94 / 0.90 | – | 0.188 |
| tfidf-logreg | 0.568 [0.54, 0.59] | 0.80 | 0.65 | 0.78 / 0.92 | 0.90 / 0.89 | – | 0.000 |
| vader | 0.510 [0.45, 0.57] | 0.68 | 0.47 | 0.51 / 0.92 | 0.51 / 0.92 | – | 0.000 |
| always-negative | 0.232 [0.21, 0.25] | 0.53 | 0.00 | 1.00 / 0.53 | 1.00 / 0.53 | – | 0.000 |

### D2 (n=250)
| model | macro-F1 [95% CI] | acc | MCC | neg R / P (argmax) | neg R / P (@0.30) | complaint R (@0.30) | McNemar p vs best |
|---|---|---|---|---|---|---|---|
| ensemble | 0.622 [0.56, 0.68] | 0.70 | 0.52 | 0.83 / 0.65 | 0.89 / 0.63 | – |  |
| tfidf-logreg | 0.602 [0.54, 0.66] | 0.67 | 0.47 | 0.75 / 0.68 | 0.84 / 0.64 | – |  |
| nlptown-mbert | 0.589 [0.53, 0.65] | 0.68 | 0.50 | 0.85 / 0.64 | 0.90 / 0.63 | – |  |
| clapai-xlmr | 0.578 [0.52, 0.64] | 0.71 | 0.54 | 0.86 / 0.68 | 0.92 / 0.64 | – |  |
| cardiffnlp-roberta | 0.578 [0.52, 0.64] | 0.68 | 0.49 | 0.87 / 0.64 | 0.87 / 0.61 | – |  |
| cardiffnlp-xlmt | 0.532 [0.48, 0.59] | 0.65 | 0.46 | 0.90 / 0.58 | 0.93 / 0.57 | – |  |
| vader | 0.411 [0.36, 0.46] | 0.51 | 0.22 | 0.41 / 0.59 | 0.41 / 0.59 | – |  |
| always-negative | 0.190 [0.17, 0.21] | 0.40 | 0.00 | 1.00 / 0.40 | 1.00 / 0.40 | – |  |


## Additional checks

**Decision threshold (P(neg) ≥ 0.30) on D2.**
- For cardiffnlp, negative precision stays at 0.58–0.66 for every threshold from 0.2 to 0.7.
- All the "false" flags are 3★ and 4★ reviews, which often contain complaints. The low precision is
  noise in the star labels, not an effect of the threshold. So the planned 0.75 floor cannot be met
  by any threshold on star-labelled data.
- On the human-labelled D1, negative precision is 0.87.
- The 0.30 rule changes **0 of 294** cardiffnlp decisions on D1. Its probabilities are peaked: the
  median top probability is 0.90.
- Decision: keep 0.30, as pre-registered. It is harmless for this model and matters for flatter
  models (TF-IDF: negative recall 0.69 → 0.86).

**Non-English reviews:** the 34 in the pool, scored against star-proxy labels.

| Model | Accuracy |
|---|---|
| cardiffnlp-roberta (English-only) | **0.18**. It predicts neutral for 28 of 34. |
| cardiffnlp-xlmt | 0.76 |
| nlptown-mbert | 0.76 |
| **clapai-xlmr** | **0.82** |

On the 6 non-English reviews in D1: cardiffnlp 0/6, XLM-T 4/6.

## Decision (pre-registered rule: best macro-F1 on D1-human; if not significantly different, the simplest)

- **English → `cardiffnlp/twitter-roberta-base-sentiment-latest`.**
  - Best deployable model on both D1 sets: macro-F1 0.686 on D1-human, 0.693 on D1-LLM.
  - Accuracy 0.84 / 0.91, MCC 0.70 / 0.83.
  - Negative recall 0.90, precision 0.87.
  - It is the smallest transformer here (125M parameters) and runs at about 30 ms per review on CPU.
  - Against XLM-T, clapAI and nlptown the McNemar p-values are 0.58 / 0.15 / 0.08, so the gap is not
    significant. The size tie-break also favours it.
- **Non-English → `clapAI/roberta-base-multilingual-sentiment`** (Apache-2.0). Best on the
  non-English reviews; XLM-T has no licence stated.
- **Not adopted:**
  - The ensemble (TF-IDF + nlptown, chosen on D2) scored 0.654 on D1, worse than cardiffnlp alone.
    This is a dev-set choice that did not transfer.
  - VADER and TF-IDF are significantly worse (p = 0.004 / 0.001).
- **Gap to the LLM reference:** 0.807 vs 0.686 macro-F1. Most of the gap is neutral (7 reviews) and
  mixed "love it, but…" reviews.
- **Complaint recall of "negative" as a complaint detector:** 0.82. Complaints inside positive
  reviews are missed (11 of 35 positive reviews contain one). They have to come from the theme
  step, or from a dedicated complaint model (see the distillation experiment).
