# Design: App Store Review Analysis API

Status: **draft for discussion**. Decisions marked *(open)* still need sign-off.
Target app for the sample report: **Notion: Notes, Tasks, AI**. App Store ID `1232780281`.

---

## 0. Decisions so far

| Area | Decision | Why (short) |
|---|---|---|
| App | Notion | The rating mix is balanced: in the 100 most recent US reviews, 38% are 1★ and 29% are 5★. That gives a fair test of 3-class sentiment and of "what people like". It also has varied themes. Fitbit was ~90% negative and dominated by one story. |
| Sample A | **300 random** written reviews from the **last 12 months**, English-speaking storefronts | Gives the overall picture: likes, dislikes, theme impact. 300 is about 25–30% of the window's population, so the margin of error is about ±4.8 pp. |
| Sample B | **100 most recent** written reviews, English storefronts | Shows current problems. It is every review in the most recent window, not a random sample, so there is no sampling error within that window. |
| Data source | Apple web JSON reviews endpoint (primary). Public RSS feed as fallback. iTunes Lookup to validate the app ID. | The RSS feed has served empty, cached pages since about 2026-10-08. The web endpoint allows random access by offset to every written review. |
| Constraints | Free, offline, open models only. **No paid APIs. No LLM in v1.** | User requirement. An LLM layer is listed as future work. |
| Sentiment | Chosen by a **model comparison** of off-the-shelf models. An ensemble and a fine-tuned model are adopted **only if they win by more than the margin of error**. | Each choice is backed by our own results, not popularity. |
| Language | **Decided.** Every review gets language ID. Sentiment and themes use multilingual models on the original text. Keywords use English text, with non-English reviews machine-translated (always on: 34 reviews in the pool). See §4 and §5.6. | Supported by the literature; details below. |
| Hosting | Vercel Hobby, native Python, ONNX int8 model. Fallback: report on GitHub Pages plus local Docker. | Free and needs no card. Google Cloud, Heroku and Koyeb need a card. HF Spaces Docker is now paid. |

---

## 1. Architecture

```
              ┌─────────────────────────── offline / GPU server ───────────────────────────┐
              │  model comparison (research/) ── fine-tune experiment ── export ONNX int8  │
              └───────────────────────────────────────┬────────────────────────────────────┘
                                                      │ model artefact
 ┌──────────┐   ┌─────────┐   ┌────────────┐   ┌──────▼───────────────────────┐   ┌──────────┐
 │1. Collect│──▶│ 2. Store│──▶│3.Preprocess│──▶│4.Metrics 5.Sentiment         │──▶│7.Insights│
 │ web JSON │   │ SQLite  │   │  2 tracks  │   │6.Keywords + themes           │   │ A vs B   │
 │ (+RSS)   │   │snapshot │   │ + lang ID  │   └──────────────────────────────┘   └────┬─────┘
 └──────────┘   └─────────┘   └────────────┘                                          ▼
                                   8. REST API (FastAPI) ──▶ 9. HTML report + charts ──▶ 10. Deploy
```

The **collection** is the central idea. It is an immutable snapshot of reviews gathered with known
parameters: app, storefronts, window, strategy, count and seed. All metrics and insights are computed
for one collection, so every number in the report can be traced to its exact input data.

---

## 2. Step 1: Data collection

### 2.1 Sources (verified live 2026-10-13)

| Source | Use | Notes |
|---|---|---|
| `itunes.apple.com/lookup?id=` | Validate the app; get its name and listing rating | Reliable. Unknown ID → `resultCount: 0`. Non-numeric ID → 400. |
| `apps.apple.com/api/apps/v1/catalog/{cc}/apps/{id}/reviews?platform=web&limit=20&offset=N&sort=recent` | **Primary** review source | No token needed. Max 20 per call. Any offset works. 404 code 40403 past the end. Heavy 429s ("API capacity exceeded", no `Retry-After`). **No app-version or vote fields.** |
| `itunes.apple.com/{cc}/rss/customerreviews/...` | Fallback | Has `im:version` and votes, but serves empty cached pages since ~2026-10-08. An empty page never means "end of data". |

Every existing Python scraper library tested broken, so we write our own small `httpx` client.

ToS note: the web endpoint is undocumented, `robots.txt` disallows `/api/*`, and Apple's terms prohibit
automated scraping. We access it politely: 1 request every 3–5 s and backoff on 429. The README states
this openly.

### 2.2 Sampling

**Sample A: random, 12 months, n=300**

1. For each storefront (`us, gb, ca, au, in, ie, nz`), find `N_cc`, the number of written reviews in
   the window. `sort=recent` returns reviews in descending date order, so this is a binary search over
   offsets for the cutoff date, about 10–15 calls per storefront. The result is cached.
2. Allocate the 300 across storefronts in proportion to `N_cc`, with largest-remainder rounding.
3. Draw distinct offsets in `[0, N_cc)` with `random.Random(seed)`. Fetch the pages of 20 that contain
   them, and keep exactly the drawn positions. This is a **true simple random sample within each
   storefront**.
4. Dedupe by review ID. Store the seed, `N_cc`, the population total and the strategy with the
   collection.

**Sample B: most recent, n=100**

1. For each storefront, take the most recent pages and merge them by date.
2. Keep the 100 newest overall.
3. Record the date span covered. US alone gives 100 reviews over about 2.5 months; pooling storefronts
   shortens the span to roughly 6–8 weeks.

The two samples **can overlap**, because B's window lies inside A's 12 months. We keep the overlap and
flag it. In the A-vs-B comparison (§7.3) we compare B against **A minus the B window**, so we are
comparing recent reviews with older ones.

### 2.3 Error handling

| Case | Detection | API response |
|---|---|---|
| Non-numeric or invalid ID format | pydantic validation | 422 |
| Unknown app, or app not in storefront | Lookup `resultCount: 0` | 404 |
| Unknown storefront | Checked against a local ISO list (also Apple 400 code 40009) | 422 |
| No reviews in window | `N = 0` | 200, empty collection plus a `warnings[]` entry |
| Fewer reviews than requested | `N < count` | 200, returns all, with `available` and a warning |
| 429 / 5xx / timeout | HTTP status or exception | Exponential backoff with jitter, 10 s per call, overall time budget. Then 503, keeping any partial results. |
| Malformed JSON, HTML error page, missing fields | parse or pydantic error | Retry, then skip the item and count it in `skipped` |
| Empty RSS page (fallback) | No `entry` | Retry once. Never treated as end of data. |

**The snapshot is committed to the repo**: a SQLite export or Parquet file in `data/snapshots/`. The demo,
tests and report then work even when Apple throttles us or changes the endpoint.

---

## 3. Step 2: Storage (SQLite)

```
apps(app_id PK, name, seller, listing_avg_rating, listing_rating_count, fetched_at)
reviews(review_id PK, app_id, storefront, date, rating, title, body, user_name,
        is_edited, developer_response, source, raw_json, fetched_at)
collections(collection_id PK, app_id, strategy ['random'|'recent'], window_start, window_end,
            storefronts, requested, collected, seed, population_json, warnings_json, created_at)
collection_reviews(collection_id, review_id)          -- many-to-many: A and B may share reviews
review_analysis(review_id, model_version, lang, sentiment, p_neg, p_neu, p_pos,
                rating_text_mismatch, themes_json)     -- computed once and cached
```

- Re-collecting is **idempotent** because rows are upserted on `review_id`.
- `raw_json` keeps Apple's original payload, for the raw-download endpoint and for auditing.
- On Vercel the filesystem is read-only except `/tmp`. The deployed app ships the snapshot DB read-only
  and treats new collections as a temporary cache in `/tmp`. This is documented.

---

## 4. Step 3: Preprocessing (two tracks). **Built**

Code: `src/review_analysis/preprocessing/`. Command: `review-analysis preprocess --snapshot <dir> [--translate]`.
Outputs: `processed.jsonl`, `preprocessing_report.json` and `translations.json` (a cache, so translation
never runs at serve time).

**Stage 0: data checks.**
- An exact duplicate is the same author **and** the same normalised text. Only the earliest copy is kept.
- The same text from different authors is legitimate and kept.
- Near-duplicates (character 3–5-gram TF-IDF cosine ≥ 0.90, for texts of at least 5 words) are flagged
  and grouped, not removed.
- Reviews under 3 words, and emoji-only reviews, are flagged. They count for ratings and sentiment, but
  contribute no keywords.

**Stage 1: shared normalisation.** It repairs and never removes content:
- `title + ". " + body`;
- `html.unescape`, `ftfy`, NFKC;
- zero-width characters are removed, but ZWJ is kept so multi-part emoji survive;
- URLs and emails become `<URL>` / `<EMAIL>`;
- line breaks become sentence ends.

**Language ID** (lingua, 20 plausible candidate languages):
- Latin-script text below 0.90 confidence goes to English if it contains an unambiguous English word.
  Otherwise, under 4 words it becomes `und`.
- Non-Latin scripts are accepted as detected.
- On the pool this fixed "iPad. Laggy" (detected as Ganda) and "One star. Won't open" (detected as
  Dutch).

**Sentences:** spaCy's rule-based sentencizer, plus a fix that splits on CJK 。！？.

**Track A (model input):** shared normalisation only. Case, punctuation, emoji, numbers and negations
are kept. Any model-specific formatting happens inside the sentiment step.

**Translation**, for Track B only:
- Model: `facebook/m2m100_1.2B` (MIT), **sentence by sentence**, beam 4, `no_repeat_ngram_size=4`.
  - Sentence-level input fixes the model dropping titles.
  - The repetition block fixes loops; the 418M model produced an Arabic review as "I don't know how
    to use it" ×10.
- Original text and translation are both stored, with `translated=true`. Quotes always show the
  original.
- Known residue: small artefacts in very short titles (a leading "and"), and a few idioms. Keywords
  need ≥ 3 reviews, which filters most of these out.

**Track B (keywords):**
- Lowercase, then a small synonym map (ipad, iphone, macos, offline, wifi, darkmode, ai), then collapse
  elongations.
- spaCy `en_core_web_sm` lemmas.
- Remove punctuation, emoji, placeholders and spaCy stopwords, **keeping negations and "too"**.
- Numbers are kept. No stemming and no spelling correction.
- Tokens are kept per sentence, so phrases never cross a sentence boundary.

**Domain stopwords** come from the data: words with document frequency ≥ 8% in both the 4–5★ and the
1–2★ groups, and a ratio ≤ 1.5, plus `notion`, `app`, `application`.
- On the pool: app, application, feature, like, need, note, notion, thing, use, want, work.
- They are dropped as **single-word** keywords, but kept inside phrases (`is_informative`), so
  "not work" and "need offline" survive.

**Results on the pool (1,143 reviews):**

| Measure | Value |
|---|---|
| English | 1,105 |
| Translated | 34 (es 14, zh 10, fr 3, ru 3, ko 2, ar 1, tr 1) |
| Undetermined | 4 |
| Short | 10 |
| Exact duplicates | 0 |
| Near-duplicate groups | 0 |

**Tests:** `tests/test_preprocessing.py`, using real reviews from the pool.

---

## 5. Step 4–5: Metrics and sentiment

### 5.1 Metrics (for each collection)

- Mean rating with a 95% CI (t-interval, with a finite-population correction for sample A).
- Rating distribution as counts and percentages, with Wilson CIs.
- Breakdown by storefront and by month.
- **Listing rating vs written-review rating.** The listing shows 4.78★ over 90k ratings, while written
  reviews average about 2.7★. The likely reason is selection bias: in-app prompts collect star-only
  ratings, which skew positive, while people who bother to write often have a complaint. Reported as a
  caveat.
- Review length, and the share of reviews with a developer response.

### 5.2 Candidate sentiment models (all offline)

| Model | Classes | Trained on | Languages | Notes |
|---|---|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3 | 124M tweets (TimeLMs) + TweetEval | EN | Only candidate with a published human-labelled app-review evaluation: macro-F1 0.725 (to verify) |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 1–5★ → 3 | Product reviews | en, de, fr, nl, it, es | Closer domain; MIT licence |
| `clapAI/roberta-base-multilingual-sentiment` | 3 | 3.9M rows, 27 domains incl. app reviews | multi | 2025; low adoption; self-reported F1 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 3 | Multilingual tweets | 8 langs | No licence stated |
| TF-IDF + logistic regression (trained on stars from other apps) | 3 | — | EN | Classic baseline (arXiv 2605.22003) |
| VADER | 3 | Lexicon | EN | Floor. It failed badly on the Fitbit pilot. |

Excluded: `tabularisai` (CC-BY-NC licence); `siebert` (positive/negative only and always overconfident, so
it cannot produce a neutral class); ModernBERT (too slow on CPU).

### 5.3 Comparison protocol (runs on the GPU server, in `research/`)

**Data:**
- **D1, the test set:** 300 Notion reviews drawn at random (seed 2026) from the pool **outside**
  samples A and B, in `data/labels/D1_test_set.json`. It is never used for training, tuning or theme
  building.
  - **Two labels per review**, following `research/labelling_guide.md` (approved):
    - **sentiment**: positive, neutral or negative, decided by the test "would they recommend the app
      as it is now?";
    - **complaint**: yes or no, "mentions anything broken, missing, too expensive or worth changing".
  - The complaint label measures the "catch complaints" priority (§5.4) directly.
  - Labels come from the text only, with stars hidden. Non-English reviews are shown with their
    translation.
  - **Human:** 100 reviews, **stratified by star** (21/21/21/17/20), so neutral and mixed cases have
    enough examples. Estimates are reweighted by D1's star shares for in-the-wild figures. The human
    labels blind with `research/label_tool.py`, a local page that saves after every review and also
    records an "unsure" flag.
  - **LLM annotators:** all 300 reviews, labelled by two models from different families, both run
    locally with vLLM 0.31 on one L40S. The guide is the system prompt, at temperature 0 with a fixed
    seed and thinking off. The JSON schema is enforced, the revision is pinned, and logprobs give a
    confidence.
    - `Qwen/Qwen3.8-27B-FP8` @017b9c7af6 (Apache-2.0);
    - `RedHatAI/gemma-4-31B-it-FP8-dynamic` @d4ab4f579d (Apache-2.0).
  - **Agreement**, computed by `research/annotator_agreement.py`:
    - human vs each LLM: Cohen's κ with bootstrap CIs, plus star-reweighted agreement;
    - LLM vs LLM on all 300 reviews. Their disagreements are the cases to adjudicate.
  - The headline metrics use the human 100. The LLM-labelled 300 are a secondary set, used only if
    human-LLM sentiment κ ≥ 0.8.
- **D2, the dev set:** 250 reviews stratified by star (50 per star) from 5–7 other productivity apps,
  with star ratings as labels. It is used only to check the decision threshold (§5.4).
- **D3, optional external check:** Lin et al. 2018, 341 Google Play review sentences labelled by two
  humans. It has only 25 neutral examples, and no licence is stated, so it is downloaded at evaluation
  time and never committed. Skip it if the data or licence is unclear.

**Metrics:**
- Macro-F1 (the selection metric).
- Per-class precision, recall and F1.
- MCC (Matthews correlation coefficient) and Cohen's κ against stars.
- **Negative recall at a negative-precision floor of ≥ 0.90.**
- Confusion matrices.
- An "always-negative" baseline row in every table.
- Bootstrap 95% CIs. McNemar's test for paired comparisons.
- Latency, RAM, artefact size, and agreement between fp32 and int8 ONNX.

### 5.4 Negative-class priority

A missed complaint is lost to the insights, while a false alarm costs little. So the rule is:
- predict `negative` if `P(neg) ≥ 0.30`, otherwise take the argmax of the remaining classes. This
  corresponds to treating a missed complaint as about 2.3× as costly as a false flag:
  `t = c_FP / (c_FP + c_FN)` (Elkan 2001).
- **Safety check on D2 only:** if negative precision falls below about 0.75, raise `t`. The change is
  documented.
- Report negative recall and precision at `t = 0.30` and at the default argmax, side by side.

Negative recall is always reported next to MCC and macro-F1. Otherwise "always negative" would look
like the best model.

### 5.5 Ensemble and fine-tuning (experiments with an adoption rule)

- **Ensemble:** average the probabilities of the two best models. Adopt it only if it beats the best
  single model significantly on D1 (McNemar). Otherwise report "tested, no significant gain". The
  ensemble gains claimed in arXiv 2605.22003 appear in no results table there.
- **Fine-tune:** continue training the best off-the-shelf model on about 10k reviews from **other**
  productivity apps. Notion is never included, to prevent leakage. There are two label variants:
  - **(i) star labels:** free, but about 20% are noisy;
  - **(ii) LLM labels (distillation):** the same local LLM labels the 10k reviews, which gives cleaner
    labels.

  Training is on the 48 GB GPU and takes tens of minutes. The same adoption rule on D1 applies. D1 is
  never used for training.

### 5.6 Language strategy (decided)

Options from the literature:
1. **One multilingual model.** XLM-R is "very competitive with strong monolingual models" (Conneau et
   al. 2020). For well-resourced languages the gap from monolingual models is small (Rust et al. 2021,
   sentiment: EN 91.6 monolingual vs 89.8 mBERT).
2. **A monolingual model per language, routed by language ID.** The accuracy gain is small for these
   languages. The cost is a separate test set, model and label scheme for each language.
3. **Translate-test**: machine-translate to English, then use the English model. Artetxe et al. 2023
   find it underrated in general, but on *review sentiment* (MARC) it scored below a multilingual model.
   It also adds a translation model and its errors.

**Decision:**
- **Language ID** (`lingua`) runs on every review and is stored.
- **Sentiment.** The comparison picks one of two setups:
  - (a) one multilingual model for all reviews;
  - (b) a router: the best English model for English reviews, the multilingual model for the rest.

  If the multilingual model is not significantly worse on English, we choose (a): simpler, one artefact.
- **Themes:** a multilingual sentence-embedding model, e.g. `paraphrase-multilingual-MiniLM-L12-v2` or
  `multilingual-e5-small`. Non-English sentences map onto the same English-named themes with no
  translation.
- **Keywords:** Track 2 is English-only. Non-English reviews are machine-translated to English first.
  - Translator: offline `opus-mt-*-en` or `m2m100_418M`, with licences to verify. Not NLLB, which is
    CC-BY-NC.
  - It is a pluggable step, enabled when non-English is at least 5% of the collection.
  - Translated reviews are flagged, and quotes always show the original text plus the translation.
  - It runs locally or in Docker only, because of Vercel's 500 MB limit; the deployed app serves a
    pre-processed snapshot.
- Rejected:
  - one model per language (cost too high for few reviews);
  - translate-test for *sentiment* (worse on review sentiment, Artetxe et al. 2023).

---

## 6. Step 6: Keywords and phrases in negative reviews

- **Main method: Fightin' Words** (Monroe et al. 2008), a weighted log-odds ratio with an informative
  Dirichlet prior.
  - Compare **negative vs non-negative** reviews on Track 2 n-grams.
  - Output a z-score for each term, plus **document frequency** ("14 / 112 negative reviews"), so one
  rant cannot dominate.
  - Show terms with |z| > 1.96 and document frequency ≥ 3.
- **Why not raw counts or TF-IDF:** raw counts surface "app" and "page", which are common everywhere.
  TF-IDF on a small corpus is noisy and does not contrast the groups.
- **Driver analysis** (the Artefact approach):
  - Fit L2 logistic regression predicting *low rating* from Track 2 n-grams.
  - Remove generic sentiment words first (`bad, great, love, hate, terrible …`), so that **product
    words** surface.
  - Report the top coefficients, plus SHAP as an option. This answers "which *things* drive low
    ratings".
- **Secondary:** YAKE, as a model-free check. **Skipped:** RAKE (it splits at stopwords and loses
  "not"); word clouds (they show no quantities).

---

## 7. Step 7: Themes and insights

### 7.1 Building the theme taxonomy

Notion has no established ground-truth theme list, so we build it from the data:

1. **Discovery:**
   - Embed negative-review sentences with a multilingual embedding model (§5.6). The specific model
     is picked by the theme-assignment check in §7.1 step 3.
   - Cluster them with agglomerative clustering (cosine distance), choosing k by silhouette score.
   - Label each cluster with c-TF-IDF terms and the sentences nearest its centre.
   - Embeddings group by **topic, not polarity**: "I love coffee" vs "I hate coffee" scores 0.79
     (DaCosta). So we cluster **within negative sentences only**, or show each cluster broken down by
     sentiment.
2. **Freeze the taxonomy.**
   - A human names and merges the clusters into about 8–12 themes. Expected candidates, to be confirmed
     by the data: performance/size, offline/sync, iPad/keyboard, mobile feature parity, editor bugs,
     AI features, pricing/plans, review prompts, UI complexity.
   - Each theme gets a description, seed terms or regexes, and example sentences.
   - It is committed as `themes.yaml`.
3. **Assignment:**
   - Rules first, for precision and easy explanation.
   - Then embedding similarity to the theme descriptions, above a threshold.
   - Validated by hand-labelling about 40 sentences, reporting precision and recall for each theme.

Each review also gets an orthogonal **review type** (Maalej & Nabil 2015): bug report, feature request,
user experience, or rating only. For example, "should be added is a color wheel" is a feature request.

### 7.2 Theme metrics

For each theme:
- `n`, and prevalence among negative reviews (Wilson CI).
- Mean rating.
- **Rating impact** = mean rating of reviews mentioning the theme − overall mean (Artefact).
- **Priority** = prevalence × (−impact).
- Top distinctive phrases.
- 2–3 **extractive** quotes with review IDs. They are copied verbatim, so nothing is invented.

This gives positive themes too ("what people like"): themes with positive impact.

### 7.3 Persistent, emerging and fading issues (A vs B)

- Compare the prevalence of each theme in B vs (A minus the B window) with a two-proportion test.
- Labels:
  - **emerging:** significantly higher in B;
  - **persistent:** high in both;
  - **fading:** significantly lower in B;
  - **not significant:** the difference is within noise.
- With n of about 100 vs 250, a difference must be roughly 12–14 pp or more to count. The report says
  this explicitly.

### 7.4 Recommendations

- A template dictionary maps theme → action, filled with the numbers and quotes.
- Example: *"Performance & app size: 18% of negative reviews (CI 12–25%), impact −1.4★, persistent.
  Quote: '680mb is crazy'. Suggest: reduce bundle size; lazy-load offline data."*
- These are deterministic, with no LLM.

---

## 8. Step 8: REST API (FastAPI)

```
GET  /health                                   liveness
GET  /ready                                    model loaded?
GET  /v1/apps/{app_id}                         lookup: name, listing rating, version
POST /v1/apps/{app_id}/collections             body {strategy: random|recent, count, window_days,
                                               storefronts[], seed} → 201 {collection_id, ...}
GET  /v1/collections/{id}                      parameters, population sizes, warnings
GET  /v1/collections/{id}/reviews?format=json|csv   raw reviews download (streamed)
GET  /v1/collections/{id}/metrics              §5.1
GET  /v1/collections/{id}/insights             sentiment, negative phrases, drivers, themes, recommendations
GET  /v1/compare?base={idA}&target={idB}       §7.3 persistent / emerging / fading
GET  /v1/collections/{id}/report               HTML report with charts
```

- Errors use the RFC 9457 problem+json format.
- Collection runs synchronously within the request, which takes about 1–3 minutes because of
  throttling. This is documented. A job queue is listed as future work.
- `/docs` (OpenAPI) is used in the video demo.

---

## 9. Step 9: Report and charts

Server-rendered with Jinja2 and **Chart.js**:
1. Rating distribution (A and B side by side).
2. Sentiment distribution, plus sentiment × star heatmap (shows mismatches).
3. Top negative phrases: Fightin' Words z-scores, annotated with document frequency.
4. Theme priority bubble chart (prevalence × impact), plus stacked sentiment for each theme.
5. Likes vs dislikes: themes with positive vs negative impact.
6. Emerging / persistent / fading chart (A vs B).
7. Monthly volume and % negative over 12 months.

The static export goes to GitHub Pages.

---

## 10. Step 10: Deployment and tooling

- **Vercel Hobby, native Python:**
  - onnxruntime + tokenizers. **No torch at runtime.** The int8 model is about 126 MB, and the total
    bundle is about 290 MB, under the 500 MB limit.
  - The read-only snapshot DB ships with the deployment.
  - First request after idle is slow (cold start).
- **Fallback:** local Docker + GitHub Pages report + optional Cloudflare quick tunnel for a live demo.
- **Tooling:**
  - uv with a committed lockfile.
  - ruff, mypy (strict on `src/`).
  - pytest + respx (Apple responses replayed from fixtures).
  - GitHub Actions running lint, types, tests and the Docker build.

---

## 11. Repository layout

```
src/review_analysis/
  config.py
  sources/      lookup.py  apple_web.py  apple_rss.py  base.py
  sampling.py   storage.py  preprocessing.py
  metrics.py    sentiment.py  keywords.py  themes.py  insights.py  compare.py
  api/          main.py  routes.py  schemas.py
  report/       templates/  render.py
research/       model comparison, fine-tune, labelling tool, results  (run on GPU server)
data/           snapshots/  labels/  themes.yaml
tests/          fixtures/  unit + API tests
DESIGN.md  README.md  REPORT (generated)  pyproject.toml  uv.lock  Dockerfile  vercel.json
```

## 12. Build order

1. Collector + sampling + storage, with the snapshot committed. **Checkpoint: real data in hand.**
2. Preprocessing + metrics.
3. Labelling D1 (user) + model comparison on GPU → sentiment model chosen.
4. Keywords + theme discovery → `themes.yaml` frozen with the user.
5. Insights + A-vs-B comparison.
6. API + tests.
7. Report + charts.
8. ONNX export + Vercel deploy + Pages.
9. README, sample report, video demo.

## 13. Limitations (stated in the README)

- Written reviews are not representative of all users: the listing shows 4.78★, while written reviews
  average about 2.7★.
- The web endpoint is undocumented and fragile, and accessing it conflicts with Apple's ToS/robots. The
  snapshot mitigates this.
- Using star ratings as sentiment labels is noisy: they disagree with the text in about 20% of reviews.
- English only for keywords and themes.
- Small samples: only large differences between A and B are detectable.
- No app version is available from the web endpoint, so trends are by date.

## 14. Future improvements

- Optional LLM layer, local open-weight or paid API:
  - second annotator;
  - routing uncertain or star-mismatched reviews to it;
  - summaries grounded in citations and verified by exact quote matching.
- Aspect-based sentiment. A multilingual themes pipeline.
- Async job queue for collection. Scheduled re-collection with trend tracking.

## 15. References (to verify before citing in the README)

Monroe et al. 2008 (Fightin' Words) · Conneau et al. 2020 (XLM-R) · Rust et al. 2021 (tokenizers) ·
Artetxe et al. 2023 (translate-test) · Loureiro et al. 2022 (TimeLMs) · Barbieri et al. 2020 (TweetEval) ·
Hartmann et al. 2023 (lexicon vs transfer learning) · Maalej & Nabil 2015 (review types) ·
Elkan 2001 (cost-sensitive thresholds) · Chicco & Jurman 2020 (MCC) · Malik & Bilal 2024 (PeerJ CS survey) ·
Shanto et al. 2026 (arXiv 2605.22003) · Artefact blog (Morin 2021) · DaCosta 2026 (TF-IDF vs SBERT) ·
Hossain et al. 2026 (arXiv 2609.19151; date inconsistency to check).
