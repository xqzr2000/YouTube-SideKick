# Study log: the data science in SideKick

The [dev log](devlog.md) records *what was built*, session by session. This log records *what
data science was used*: one entry per technique, with what it is, how SideKick used it, the
evidence it produced, the lesson, and where to read the code. It's written for study: read an
entry, open the file, rerun the notebook, try the exercise.

Entries are grouped by the stage of a data-science project they belong to, not by date. Each
ends with the session it came from (S1–S11, see the dev log). New entries are added at the end
of every session.

**Contents**

1. [Framing the problem](#1-framing-the-problem)
2. [Collecting data](#2-collecting-data)
3. [Exploring data](#3-exploring-data)
4. [Labels](#4-labels)
5. [Text representations](#5-text-representations)
6. [Evaluating with few labels](#6-evaluating-with-few-labels)
7. [Thresholds and decision rules](#7-thresholds-and-decision-rules)
8. [Feature engineering](#8-feature-engineering)
9. [Classical NLP](#9-classical-nlp)
10. [Retrieval and ranking](#10-retrieval-and-ranking)
11. [Performance as a measured quantity](#11-performance-as-a-measured-quantity)
12. [Offline evaluation of a ranker](#12-offline-evaluation-of-a-ranker)
13. [Learning to rank (M6c)](#13-learning-to-rank-m6c)
14. [Behaviour analysis from logs (M7)](#14-behaviour-analysis-from-logs-m7)
15. [Coming next (M7)](#15-coming-next-m7)
16. [Skills checklist](#16-skills-checklist)

---

## 1. Framing the problem

### 1.1 Filter first, then rank
- **What:** split "recommend videos" into cheap hard filters, then soft filters, then a ranker,
  each with its own error measure.
- **In SideKick:** Stage A (cheap fields), Stage B (fetched pages), Stage C (text scores), then
  the v0 score and MMR. Every stage logs one decision per video with all its reasons.
- **Lesson:** a pipeline you can measure stage by stage is debuggable. When the comedy context
  lost three good clips, the logs showed it was Stage C, on one profile line.
- **Where:** `src/sidekick/filters/`, `src/sidekick/recommend/`. (S1, S4)

### 1.2 Choose the error you can afford
- **What:** decide upfront whether false positives or false negatives cost more.
- **In SideKick:** soft filters are **precision-first**. Dropping a good video is worse than
  letting a mediocre one through, because the ranker can still push the mediocre one down.
- **Lesson:** the threshold rule follows from this choice (section 7.1), not the other way
  round. (S5)

### 1.3 Rank by your own signal, not engagement
- **What:** borrowed from Karpathy's HackerTube: rank by a trust signal rather than views.
- **In SideKick:** the profile, 👎 notes and clicks are the signal; views count for 5 % of v0.
  **Recurrence** (how many lists surfaced a video) is a cheap, engagement-free relevance signal.
- **Where:** `recommend/score.py` (`recurrence`, `persistence`, `views_part`). (S1, S6)

### 1.4 Know what kind of learning problem you have
- **What:** positives are rare clicks, negatives are rare 👎s, and most impressions are
  unlabeled. That's **positive–unlabeled (PU) learning**, not ordinary classification.
- **In SideKick:** clicks stay an online check only, because a click also depends on the card's
  position (position bias). Hand labels are the training data. (S1, S8)

---

## 2. Collecting data

### 2.1 Scrape structured data, not HTML
- **What:** YouTube pages embed JSON (`ytInitialPlayerResponse`, `ytInitialData`); parsing that
  is more reliable than parsing HTML.
- **In SideKick:** the parsers look values up by **key name, not fixed path**, so they survive
  small layout moves. When YouTube switched up-next to `lockupViewModel`, a `--skeleton` dump of
  the real structure found it.
- **Lesson:** inspect real data before trusting a parser. 0 of 20 up-next items parsed until the
  real page was looked at.
- **Where:** `src/sidekick/parse/`, `just parse-check --skeleton`. (S3)

### 2.2 Keep raw data so you can replay
- **What:** store what you collected in a form you can reprocess when the parser or features
  change.
- **In SideKick:** a gzip **raw log** of the extracted JSON blobs (10× smaller than HTML), kept
  90 days. Earlier drops were replayed from cache after rule changes (#20, #21) without new
  requests.
- **Where:** `src/sidekick/store/rawlog.py`. (S3)

### 2.3 Log every decision, not just the output
- **What:** **data provenance**: for each output, record what went in and why.
- **In SideKick:** SQLite tables `computes`, `candidates`, `decisions` (stage, kept, reasons,
  score, features as JSON), `impressions`, `feedback`, `labels`, `profile_versions` (hashed).
- **Lesson:** most later analyses (E3 replay, P@10, the labeling queues) needed no new data,
  because the logs already held it.
- **Where:** `src/sidekick/store/db.py`. (S4)

### 2.4 Synthetic data for tests and public reproducibility
- **What:** a generated sample with the real data's structure but no private content.
- **In SideKick:** `data/sample/` (four topics, raw-log format) runs every notebook in CI.
  Real pages stay in `~/.sidekick/`.
- **Lesson:** the synthetic sample goes through **the same code path** as real data, so the
  test checks the real pipeline.
- **Where:** `scripts/make_sample.py`, `explore._sample_store`. (S3)

### 2.5 Collect politely, and measure the cost of collecting
- **What:** rate limits, jitter, back-off, and a page cap per compute.
- **In SideKick:** 1–2 s jitter, 60 pages per compute, 30 minutes of back-off on a bot page. E3
  was done from logs instead of a 350-request sweep (section 11.3). (S3, S6)

---

## 3. Exploring data

### 3.1 Funnel analysis
- **What:** count items through each stage, and why they drop.
- **In SideKick:** NB02 shows candidates → Stage A → Stage B per compute, with drop reasons by
  stage (a video can carry several).
- **Where:** `notebooks/02_filters.py`. (S4)

### 3.2 Look at the borderline cases
- **What:** the items just over or under a threshold are where a rule is wrong.
- **In SideKick:**
  - McElreath's 2 h lecture was dropped by `max_duration: 90m`, so the limit became 150m.
  - "MIT 6.S191: Reinforcement Learning" was dropped as German on the title alone, so language
    drops moved to Stage B (title + description).
- **Lesson:** a single false removal you can name beats an aggregate rate when the data is
  small. (S4, S5)

### 3.3 Look at distributions before choosing a threshold
- **What:** plot scores first. Raw e5 cosines bunch between 0.74 and 0.89, so a fixed threshold
  on them is fragile (section 5.4).
- **Where:** `notebooks/03_representations.py`. (S5)

---

## 4. Labels

### 4.1 Contextual labels
- **What:** a label whose meaning depends on a context.
- **In SideKick:** *while watching **this** video, would you want **that** one next?* A
  stand-up clip is good after stand-up and bad after a lecture.
- **Lesson:** the context is part of the label's key (`compute_id → current video`). (S5)

### 4.2 Separate questions get separate labels
- **What:** don't mix two questions in one label.
- **In SideKick:** the **seed** label (does the current video match my Interests?) is E2's
  ground truth. The **candidate** label is E1's and the ranker's. The first seed pass answered
  "yes" for everything until the Interests lines were shown with the question. (S5)

### 4.3 Blind labeling
- **What:** hide the model's output from the labeler, so labels don't drift toward it.
- **In SideKick:** filter decisions, scores and ranks are hidden. SideKick's and YouTube's top
  10s were labeled mixed and unmarked for E4. (S5, S7)

### 4.4 Label noise
- **What:** the same labeler isn't perfectly consistent.
- **In SideKick:** "ads" got bad in one session and meh in another, and a Dry Bar special with
  the same app pitch got good once and bad once. So the promo threshold wasn't tightened on
  those labels. (S5)

### 4.5 Labeling tools follow the metric
- **What:** what you label is decided by the metric you'll compute.
- **In SideKick:** NDCG of a re-ranker needs **whole pools** labeled, so the M6 app finishes one
  current video's pool at a time.
- **Where:** `src/sidekick/labeling.py`, `notebooks/05_labeling.py`, and the full study note:
  [labeling-app-guide.md](labeling-app-guide.md). (S8)

---

## 5. Text representations

### 5.1 TF-IDF
- **What:** sparse word and bigram vectors, weighted by how rare a term is.
- **In SideKick:** fitted on our own videos plus the profile lines. It has no stemming, so
  "chemometric" ≠ "chemometrics", and it scores 0 for synonyms.
- **Where:** `src/sidekick/embed/tfidf.py`. (S5)

### 5.2 LSA
- **What:** truncated SVD of TF-IDF into about 64 dimensions, to catch co-occurrence.
- **In SideKick:** worse than chance on Don't-show-me with 40 labels (AUC 0.22–0.38). With little
  data, dimensionality reduction can blur exactly the words that matter. (S5)

### 5.3 Sentence embeddings
- **What:** a pretrained encoder (`multilingual-e5-small`, int8 ONNX, 118 MB) gives dense
  384-dimension vectors.
- **In SideKick:** it caught "Bayesian Statistics" ↔ "Statistik für Anfänger: Bayes-Theorem"
  (0.88). The `query:` / `passage:` prefixes follow e5's training convention.
- **Where:** `src/sidekick/embed/e5.py`. (S5)

### 5.4 Anisotropy and mean centering
- **What:** embedding vectors crowd into a narrow cone, so all cosines look similar.
  Subtracting the mean vector spreads them out.
- **In SideKick:** centering tripled the interest-gate gap (0.03 → 0.11), but **hurt**
  video-to-video similarity (−0.06 AUC).
- **Lesson:** decide per comparison. Centered vectors are used against profile lines, raw
  vectors between videos. (S5)

### 5.5 Embeddings can match the wrong concept
- **What:** an embedding matches everything in a phrase, not the part you meant.
- **In SideKick:** "political late-night clips" matched *late-night comedy* as much as
  *politics* and dropped plain stand-up. Rewording it to "political commentary and election
  news" fixed it with no code change.
- **Lesson:** one concept per profile line. (S5)

### 5.6 Cross-lingual retrieval
- **What:** a multilingual encoder puts texts in different languages that mean the same thing
  near each other.
- **In SideKick:** for a Japanese video, the nearest **English** titles already in the database
  vote for query words. A Japanese Bayesian-statistics talk produced the query "Bayesian
  Statistics", with no translation model.
- **Where:** `recommend/queries.py` (`english_round1`). (S7, S8)

---

## 6. Evaluating with few labels

### 6.1 AUC instead of accuracy
- **What:** the probability that a random positive scores above a random negative. It needs no
  threshold and is fair with unbalanced classes.
- **In SideKick:** E1 compared representations by "Don't-show AUC" (bad > good) and "seed-sim
  AUC" (good > rest).
- **Where:** `src/sidekick/embed/evaluate.py`, `reports/E1-representations.md`. (S5)

### 6.2 Bootstrap confidence intervals
- **What:** resample the labels with replacement many times to see how much a metric moves.
- **In SideKick:** every AUC carries a 95 % interval. With 10 bad and 17 good labels, the
  standard error was about ±0.09, so "q vs qp" was a tie. (S5)

### 6.3 Results move when the data covers new contexts
- **What:** a metric measured on one kind of data can change sharply on another.
- **In SideKick:** with 40 labels, e5 led on Don't-show (0.81). With 95 labels that included news
  contexts, e5 fell to 0.65 and TF-IDF rose to 0.81. While watching news, news is wanted, so
  "Don't show me political news" depends on context.
- **Lesson:** small label sets from few contexts overstate certainty. Collect labels across
  contexts before choosing a model. (S5)

### 6.4 Resample the independent unit
- **What:** bootstrap over the units that are independent, not over items inside them.
- **In SideKick:** E4 resamples **seeds**, not videos, because videos labeled in one context and
  one sitting are correlated. That gave +17 points [+6, +29] for SideKick over YouTube's up-next.
- **Where:** `src/sidekick/recommend/evaluate.py`, `reports/E4-ranker.md`. (S7)

---

## 7. Thresholds and decision rules

### 7.1 Precision-first thresholds from labels
- **What:** set a drop threshold just above the highest-scoring **good** label, so no labeled
  good video would be dropped.
- **In SideKick:** `just tune` sets threshold = top good score + 0.01. It also reports how many
  **unlabeled** survivors each threshold would drop. That report flagged a comedy false removal
  before it happened.
- **Where:** `src/sidekick/tune.py`, `filters/thresholds.py`. (S5)

### 7.2 Floors against small-sample overfitting
- **What:** a minimum value for a learned threshold.
- **In SideKick:** with 17 good labels, a signal no good video triggers would get a threshold
  near 0 and drop many unlabeled videos. Floors (Don't-show 0.20, clickbait 0.65) cap how
  aggressive a small label set can make a filter. (S5)

### 7.3 Exclude labels that answer a different question
- **What:** drop labels from the threshold fit when the rule doesn't apply to them.
- **In SideKick:** in a news context the Don't-show line no longer decides alone. So `just tune`
  ignores news-context labels for that threshold; otherwise they'd push it to about 0.36 and
  switch the filter off. (S6)

### 7.4 A gate between two modes
- **What:** a threshold on similarity that switches behaviour.
- **In SideKick:** the **interest gate**: if the current video matches an Interests line
  (τ = midpoint between the lowest "yes" seed and the highest "no" seed), queries blend in your
  interests; otherwise SideKick only filters.
- **Where:** `reports/E2-interest-gate.md`. (S5)

---

## 8. Feature engineering

### 8.1 Interpretable hand-weighted scores before learned ones
- **What:** combine a few named parts by hand, log every part, and learn the weights later.
- **In SideKick:**
  - **clickbait:** the share of SHOUTED words, the `!`/`?` count, any emoji, and a bait-phrase
    list, weighted 0.4 / 0.2 / 0.2 / 0.4 and capped at 1;
  - **promo:** how many groups of promotional phrases (sponsor reads and the like) appear in the
    description (sponsor reads, discount codes, …), divided by 3. It catches what SponsorBlock
    misses.

  Both are logged per video for M6.
- **Where:** `src/sidekick/features/text_signals.py`. (S5)

### 8.2 A feature found in label reasons
- **What:** read the free-text reasons for patterns.
- **In SideKick:** "good" mostly meant the **same series**, and "meh" mostly meant "drifted from
  the course". That became the topic anchor's `series` part and the lecture-context rules. (S5,
  S6)

### 8.3 Rank-normalize scores on different scales
- **What:** convert scores to percentiles within the candidate pool.
- **In SideKick:** e5 cosines (0.74–0.98) and the TF-IDF fallback live on different scales.
  Percentiles make them comparable without tuning.
- **Where:** `recommend/score.py` (`percentiles`). (S6)

### 8.4 Script detection
- **What:** classify a text's writing system by Unicode character names. No model needed.
- **In SideKick:** a title mostly in kana, kanji or Hangul is dropped before any fetch. Those
  characters weigh 3× a Latin letter (one carries a syllable or word), so a Japanese song title
  with a romanized artist name still counts as Japanese.
- **Where:** `src/sidekick/features/script.py`. (S7)

### 8.5 Language ID on short text
- **What:** fastText `lid.176` gives a language and a confidence.
- **In SideKick:** on titles alone it is weak (an MIT lecture was guessed German); with the
  description most confidences reach 0.85–0.97. Drops require ≥ 0.80 on title + description,
  and music is exempt. (S4, S5)

---

## 9. Classical NLP

### 9.1 Boilerplate removal by repetition
- **What:** text that repeats across one source's documents is template, not content.
- **In SideKick:** description lines that appear on 2+ of a channel's videos, and title words in
  ≥ 50 % of its titles, are removed. Template words are kept as `series`. CTV News's whole
  description turned out to be boilerplate.
- **Where:** `src/sidekick/topic/text.py`. (S6)

### 9.2 NER and POS with spaCy
- **What:** named entities (people, places, organisations) and part-of-speech tags from a small
  model (`en_core_web_sm`, 13 MB).
- **In SideKick:**
  - headlines break the parser ("collapse" read as the head noun), so keyphrases come from runs
    of ADJ/NOUN/PROPN tags instead of noun chunks;
  - Title Case confuses NER ("Kimmel Breaks Down" → PERSON), so entities are matched at token
    level, as one signal among several.
- **Where:** `src/sidekick/topic/nlp.py`. (S6)

### 9.3 IDF weighting and weighted fields
- **What:** weight terms by rarity and by where they appear.
- **In SideKick:** the headline counts 1×, the first description sentence 0.6×, keywords 0.3×.
  (S6)

---

## 10. Retrieval and ranking

### 10.1 Query generation from a document
- **What:** turn a document into a search query: keyphrases, places, and its own English words
  for non-English text.
- **Where:** `src/sidekick/topic/__init__.py` (`query`), `recommend/queries.py`. (S6)

### 10.2 Rocchio pseudo-relevance feedback
- **What:** expand the query with terms from the best results so far.
- **In SideKick:** rounds 2–3 only run while fewer than 10 videos survive and under a 60 s
  deadline. On real seeds they never triggered (26–30 survivors after round 1). (S6, S7)

### 10.3 Snowball sampling
- **What:** grow the pool from neighbours of good items.
- **In SideKick:** the up-next lists of pages already fetched, so it costs no extra requests.
  (S6)

### 10.4 MMR diversity with group caps
- **What:** Maximal Marginal Relevance picks the next item by λ · relevance − (1 − λ) ·
  similarity to the items already picked.
- **In SideKick:**
  - λ = 0.75, on centered vectors;
  - caps: the same course ≤ 5 in a lecture context, the same performer ≤ 2 elsewhere;
  - MMR fought the course cap until same-course lectures stopped counting as redundant (#27).
  - **Finding (S8):** in a one-topic pool, MMR promoted an off-topic "binaural beats" video over
    three on-topic ones, because novelty is cheapest for an outlier. Fix planned: only let MMR
    choose among videos close to the top relevance.
- **Where:** `recommend/diversity.py`. (S6, S8)

### 10.5 Spend an expensive budget where it matters
- **What:** a cheap pre-score decides which candidates get the expensive step.
- **In SideKick:** enrichment (about 2.6 s per page) goes to the top 25 by `0.60 topic +
  0.25 recurrence + 0.15 position`. (S6)

---

## 11. Performance as a measured quantity

### 11.1 Regression of time on pages
- **What:** ordinary least squares, `seconds = a + b · pages`.
- **In SideKick:** 1.5 s + 2.69 s per page, R² 0.98 (39 computes). Time is almost all page
  fetches, so the line predicts any compute.
- **Where:** `src/sidekick/e3.py`, `reports/E3-timing.md`. (S6, S8)

### 11.2 Prediction interval, not confidence interval
- **What:** a confidence interval bounds the **mean**. A prediction interval bounds **one new
  observation**, and it is wider.
- **In SideKick:** "every compute under 90 s" is a promise about single computes, so the budget
  is set where the one-sided 95 % **prediction** bound stays under 90 s. That gives an
  enrichment budget of 25, not the spec's 50. (S6)

### 11.3 Counterfactual replay from logs
- **What:** answer "what if the budget were smaller?" from what was logged, instead of rerunning.
- **In SideKick:** replaying logged computes at budgets 10–30 showed which shown or good-labeled
  videos would have gone unenriched. Three cold runs checked the line, instead of a 12-run
  sweep of about 350 requests. (S6)

### 11.4 Confounders in a comparison
- **What:** a difference between two groups can come from something other than the factor you
  compare.
- **In SideKick:** 44 % of panel cards were unenriched on the Chromebook against about 10 % on
  WSL2. The cause wasn't the machine but the Chromebook's fresh, empty page cache. (S8)

---

## 12. Offline evaluation of a ranker

### 12.1 Precision@10 against a real baseline
- **What:** of the top 10, the share labeled good. Compare against what users already get.
- **In SideKick:** SideKick 66 % good / 14 % bad, YouTube's first 10 up-next 49 % / 31 %, on 7
  seeds with blind, mixed labels. Ahead on 5 seeds, tied on 2, never behind.
- **Where:** `reports/E4-ranker.md` part 1. (S7)

### 12.2 Don't tune on the evaluation set
- **What:** hand-tuning weights on the same few cases you report on overfits.
- **In SideKick:** after E4 part 1, the v0 weights were **not** hand-tuned on 7 seeds. Weights
  will be learned in M6 with a proper split. (S7)

### 12.3 NDCG@10 with graded labels
- **What:** Discounted Cumulative Gain adds up each result's gain, discounted by position:
  DCG@k = Σ gainᵢ / log₂(i + 1) for i = 1…k. NDCG divides by the DCG of the best possible
  order of the same items, so 1.0 means a perfect order on any pool.
- **In SideKick:** gains good 2, meh 1, bad 0 (linear; the common alternative 2^g − 1 gives
  3 / 1 / 0 and weighs "good" more). Computed per pool, then averaged. A pool with nothing
  good or meh has no ideal order and is skipped.
- **Lesson:** P@10 only asks "how many good ones in the top 10"; NDCG also cares *where* they
  are, and gives partial credit for meh.
- **Where:** `src/sidekick/rank/metrics.py`, `just eval`. (S10)

### 12.4 Incomplete labels: condensed lists
- **What:** when some items in a pool are unlabeled, rank the whole pool, then drop the
  unlabeled items and score what's left in the same order (Sakai, 2007). An unlabeled item
  neither helps nor hurts.
- **In SideKick:** this is why the labeling app finishes whole pools. With complete pools
  nothing is dropped. `--min-coverage 0.9` lets nearly complete pools in.
- **Lesson:** treating unlabeled as "bad" punishes a new ranker for finding things nobody
  labeled; that bias grows exactly when the new ranker differs most from the old one. (S10)

### 12.5 Paired comparisons
- **What:** when two rankers are scored on the same pools, compare them pool by pool (the
  difference per pool), not by their two separate means. Paired differences cancel out how
  easy each pool is.
- **In SideKick:** `paired_diff` bootstraps the per-pool differences ("v0 shipped − v0 score
  only" shows what MMR and the caps cost or add).
- **Where:** `rank/metrics.py` (`paired_diff`). (S10)

### 12.6 Baselines that bracket the result
- **What:** report a floor and simple alternatives next to the system: a random order (what
  luck gives), single features (what one signal gives), and the current system.
- **In SideKick:** random, topic only, seed similarity only and interest only, next to v0.
  If "topic only" matched v0, the other v0 parts would be adding nothing.
- **Pitfall found:** sorting by a missing feature with a stable sort kept the logged order,
  which *is* v0's order, so the "single feature" baselines looked as good as v0. Starting
  from a fixed shuffle fixed it. A baseline must not see the system's own output. (S10)

### 12.7 Feature screening, and why it isn't evaluation
- **What:** one AUC per feature (good vs rest) on all labeled rows, to see which logged
  signals carry information before training anything.
- **In SideKick:** printed by `just eval`, sorted by distance from 0.5. An AUC below 0.5 means
  higher values go with worse videos (clickbait, Don't-show similarity).
- **Lesson:** pooled screening mixes current videos, so a feature that mostly identifies
  *which* current video a row came from can look strong. The real test is on held-out
  current videos (13). (S10)

### 12.8 Replaying a decision step offline
- **What:** to compare alternatives to a step the system already took, redo that step from
  what was logged, then check that the replay reproduces the logged output before trusting it.
- **In SideKick:** the "pick 10" step (MMR + caps) is replayed from logged scores, groups and
  cached e5 vectors. `just eval --learn` first prints how often the replay matches the logged
  top 10.
- **Where:** `src/sidekick/rank/rerank.py`. (S10)

### 12.9 Diversity that can't promote outliers
- **What:** MMR trades relevance for novelty with λ. In a pool that is mostly one topic, the
  most novel item is the off-topic one. A **window** limits each pick to items within δ of the
  best remaining score: variety can reorder close calls, not rescue weak ones.
- **In SideKick:** `mmr_select(window=δ)`; δ = 0 is score order with caps, no window is plain
  MMR. δ is chosen inside each training fold, never on the test video. (S10)

---

## 13. Learning to rank (M6c)

### 13.1 Pointwise vs listwise
- **What:** a **pointwise** model scores each item alone (here: logistic regression for
  P(good)). A **listwise** model is trained on whole lists to optimize a ranking metric (here:
  LightGBM `lambdarank`, which weights each pair of items by how much swapping them would
  change NDCG).
- **In SideKick:** both, on the same 31 features. `label_gain=[0, 1, 2]` makes LightGBM optimize
  the same linear-gain NDCG that `just eval` reports.
- **Where:** `src/sidekick/rank/learn.py`. (S10)

### 13.2 Grouped cross-validation: leave one group out
- **What:** split by the unit that is independent, not by row. Here a group is one current
  video: its candidates were labeled in one context and one sitting.
- **In SideKick:** 13 folds, each trained on 12 current videos and tested on the 13th; the
  interval is a bootstrap over the 13 held-out results. A time split (older train, newer test)
  is the second check. Leave-one-out became the main test because a 30 % time split would test
  on only 4 current videos.
- **Lesson:** a random row split would put candidates of the same current video on both sides
  and overstate how well the model generalizes to a *new* video. (S10)

### 13.3 Nested tuning
- **What:** any setting chosen by looking at results (a threshold, a window, a penalty) must be
  chosen inside each training fold, or the test score is optimistic.
- **In SideKick:** the MMR window is picked per fold on its 12 training pools, and the chosen
  values are printed per fold. Model settings (C = 0.3, shallow trees) were fixed in advance,
  not tuned on these labels. (S10)

### 13.4 Missing values
- **What:** "not measured" is information too. Impute with the training median *and* add an
  "is missing" indicator column; tree models (LightGBM) handle missing values natively.
- **In SideKick:** Stage B facts are missing for videos that weren't enriched; the indicator
  lets the model learn whether that itself matters. (S10)

### 13.5 Reading what a model uses
- **What:** standardized logistic-regression coefficients ("one SD more of this feature →
  this much more log-odds of good") and LightGBM's share of split gain, each averaged over the
  folds with its spread.
- **Caveats:** correlated features share or swap credit (topic vs seed similarity); gain
  shares say what the trees split on, not the direction; a feature that is constant within a
  pool can't change the order within it, whatever its weight. (S10)

### 13.6 A negative result is a result
- **What:** report it when the learned model doesn't beat the baseline on held-out data.
- **In SideKick (S10):** without caps, LightGBM reached NDCG@10 0.878 and logistic regression
  0.874, against v0's score alone at 0.895, on 13 held-out current videos; the time split
  agrees. With 13 contexts, 31 features are too many to learn reliably, and v0's hand-set
  weights already encode most of the signal (similarity and topic).
- **Lesson:** more labeled *contexts*, not more rows per context, is what would change this.

### 13.7 Correlated features swap credit
- **In SideKick:** logistic regression gave `seed` (the percentile) +0.81 and `seed_sim` (the
  raw cosine behind it) −0.27. Two near-duplicate features split the effect, and one can even
  flip sign. Read coefficients as a group, or drop duplicates before interpreting. (S10)

### 13.8 Item labels can't see list-level goals
- **What:** a label on one item ("would I want this next?") says nothing about the list as a
  whole. Variety is a property of the list, so NDCG on item labels is biased *against*
  diversity by construction.
- **In SideKick:** the diversity caps cost 0.070 NDCG [0.014, 0.140], and MMR on top of them
  nothing (+0.003). Whether ten good clips of one comedian make a worse list is a product
  decision. The cap-grid report shows **groups in the top 10** and the **largest group** next
  to NDCG, so the trade-off is visible instead of hidden. (S10)

### 13.9 Prefer the simpler option when it's nearly as good
- **What:** when choosing among settings, take the most conservative one within a small
  tolerance of the best (the idea behind the "one standard error rule" in model selection).
- **In SideKick:** in each leave-one-out round, the cap setting chosen is the **tightest**
  whose training NDCG is within 0.01 of the best. Variety stays unless loosening the caps
  clearly pays. (S10)

---

## 14. Behaviour analysis from logs (M7)

### 14.1 Define the measure before looking: stickiness vs spread
- **What:** "YouTube keeps recommending it" could mean two different things. It could be
  **sticky** (the same video comes back for the same context) or **broad** (it shows up across
  unrelated contexts). Each gets its own measure, decided before reading the numbers.
- **In SideKick (E6):** stickiness = share still there in the next compute of the same current
  video. Spread = the number of distinct current videos a video appeared under. The answer was
  breadth: unwanted videos were *less* sticky (32 % vs 64 %) but spread more (13 % vs 8 % under
  ≥ 2 current videos).
- **Where:** `src/sidekick/e6.py`, `reports/E6-persistence.md`. (S11)

### 14.2 Rates with a cluster bootstrap
- **What:** when items come in groups (sightings under one current video), resample whole
  groups and pool their counts, rather than resampling items.
- **In SideKick:** `cluster_rate` resamples current videos; the signed-in rate (63 sightings
  from 3 computes) gets a wide interval, [15, 70] %. Resampling the 63 sightings would have
  pretended there was far more evidence. (S11)

### 14.3 Look for the confounder
- **What:** when two groups differ, ask what else differs between them.
- **In SideKick:** unwanted videos were a week old at median and wanted ones almost two years,
  with more views. Freshness and popularity, not position, separate them, which points to a
  context-free "new and popular" signal in YouTube's lists. It also explains why `age_days_log`
  helped in E4's feature screen. (S11)

### 14.4 A label is about a context, not an item
- **In SideKick:** MIT OpenCourseWare and 3Blue1Brown are unwanted in some contexts and wanted
  in others. A channel-level rule must stay context-aware, or it would drop the lectures you
  most want. (S11)

### 14.5 Error analysis: follow one bad pool to its cause
- **What:** when a metric looks fine on average, open the worst case and trace each bad item to
  the step that let it in.
- **In SideKick:** the CS336 pool was 22 of 33 bad, yet its ranking was decent (NDCG 0.80 vs
  0.29 random). Listing the pool with each candidate's source showed that 19 of the bad videos
  came from one search, `Architectures`, a word that means buildings without its course
  context. The fix went to query construction, not to the ranker.
- **Lesson:** ranking metrics measure the order *within* the pool; a separate check is needed
  for what gets *into* the pool (candidate generation, "recall"). `just eval --queries` is that
  check for searches. (S11)

### 14.6 Averages dilute rules that only apply sometimes
- **What:** a setting that can only matter in one subgroup should be judged on that subgroup.
  Averaging over everything adds zeros from the cases it can't affect.
- **In SideKick:** the course cap only binds in lecture pools. Over all 16 pools it cost
  +0.007 NDCG (under the 0.01 tolerance); per lecture pool it was about 0.02. (S11)

### 14.7 Pre-registration and the garden of forking paths
- **What:** if you change how a result will be judged *after* seeing which way it points,
  almost any result can be made to look like a win. Fix the rule, the subgroup and the stopping
  point in writing before the run that decides.
- **In SideKick:** instead of switching to course cap 8 on the spot, the rule for the next run
  was written into the repo first (`reports/E4-ranker.md` addendum; `just eval --caps` prints
  it): lecture pools only, the same nested choice, decided at ≥ 8 lecture pools. (S11)

---

## 15. Coming next (M7)

Planned techniques, to be written up here as they're used:

- **Online check of the M6 change** (no performer cap): label a few new pools and compare
  NDCG and the bad share with the offline estimate; watch 👎 notes for "too repetitive".
- **Retrying learned rankers** once about 30 current videos are labeled.
- **Position bias** in clicks, and why clicks stay an online check (M6).
- **Taste drift:** a time-decay half-life for feedback and profile lines (E5, M7).
- **Model card:** what the ranker is for, its data, metrics and limits (M7).

---

## 16. Skills checklist

| Skill | Where you practised it | Status |
|---|---|---|
| Web data collection, schema design, provenance logging | 2.1–2.5 | done |
| Exploratory analysis in notebooks (marimo) | 3, NB01–NB03 | done |
| Designing a labeling task and tool | 4, NB05, [guide](labeling-app-guide.md) | done |
| Sparse vs dense text representations | 5.1–5.4 | done |
| Evaluation with small samples (AUC, bootstrap, cluster bootstrap) | 6 | done |
| Threshold selection under a precision constraint | 7 | done |
| Feature engineering (interpretable scores, normalization) | 8 | done |
| Classical NLP (NER, POS, IDF, boilerplate) | 9 | done |
| Information retrieval (query generation, Rocchio, MMR) | 10 | done |
| Regression and prediction intervals for an engineering budget | 11 | done |
| Offline ranking evaluation (P@10, NDCG, paired bootstrap, baselines) | 12 | done |
| Learning to rank, grouped CV, nested tuning | 13 | done (M6c) |
| PU data, position bias | 15 | M7 |
| Persistence analysis from logs (cluster bootstrap, confounders) | 14 | done (E6) |
| Drift analysis, model cards | 15 | M7 |
