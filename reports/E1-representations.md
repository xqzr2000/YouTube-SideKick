# E1 — Which text representation should the filters use?

*M3 · 2026-11-04/05 · data: one user's hand labels on one machine (private; aggregate numbers only)
· reproduce with `just sim --eval --md` and NB03 (`notebooks/03_representations.py`)*

## Question

SideKick compares texts in two places: video ↔ Don't-show-me line (Stage C soft filter) and
video ↔ the video being watched (a ranking feature from M4). Which representation of a video's
text separates what the user wants from what they don't, and does mean-centering help?

## Setup

- **Video text** (same input for every representation): title · channel · keywords · first
  1000 characters of the description, with URLs and timestamps removed.
- **Representations:** `tfidf` (word + bigram, fitted on our own logged videos and the profile
  lines), `lsa` (TF-IDF → ≤ 64-dimension SVD), `e5-small-q` (`multilingual-e5-small`, int8 ONNX,
  `query:` prefix on every text), `e5-small-qp` (same model, `passage:` prefix on videos,
  `query:` on profile lines).
- **Variants:** *raw* cosine, and *centered* (the mean of all video vectors subtracted, then
  re-normalized), the standard fix for anisotropic embeddings: raw e5 cosines bunch up at
  0.74–0.89 for almost any pair.
- **Labels:** `just label`, asked *"while watching this, would you want that shown next?"* with
  filter decisions hidden. 95 candidate labels (42 good / 31 meh / 22 bad) over 7 current
  videos: three on-interest lectures/talks (Bayesian statistics, reinforcement learning,
  chemometrics), stand-up comedy, local news, a cooking show and a jazz concert.
- **Metrics** (threshold-free, so they don't depend on a tuned cut-off):
  - *AUC don't-show*: does max similarity to the Don't-show-me lines rank *bad* above *good*?
  - *r0*: share of *bad* videos scoring above every *good* one, which is what a precision-first
    threshold could catch.
  - *AUC seed-sim*: does similarity to the current video rank *good* above *meh + bad*?
  - 95 % bootstrap intervals (each class resampled separately, 2000 draws, fixed seed).

## Results

Final run, 95 labels:

| representation | variant | AUC don't-show [95 % CI] | r0 | AUC seed-sim [95 % CI] |
|---|---|---|---|---|
| tfidf | raw | 0.58 [0.45, 0.72] | 0.05 | 0.78 [0.68, 0.86] |
| tfidf | centered | **0.81 [0.68, 0.91]** | 0.05 | 0.78 [0.69, 0.87] |
| lsa | raw | 0.71 [0.59, 0.83] | 0.00 | 0.75 [0.65, 0.85] |
| lsa | centered | 0.78 [0.66, 0.89] | 0.00 | 0.75 [0.65, 0.85] |
| e5-small-q | raw | 0.61 [0.46, 0.76] | 0.00 | 0.76 [0.66, 0.85] |
| e5-small-q | centered | 0.60 [0.45, 0.75] | 0.00 | 0.78 [0.69, 0.87] |
| e5-small-qp | raw | 0.66 [0.51, 0.79] | 0.00 | 0.77 [0.66, 0.85] |
| e5-small-qp | centered | 0.65 [0.51, 0.78] | 0.00 | 0.78 [0.68, 0.86] |

An earlier run with 40 labels from the three lecture contexts and comedy only (no intervals
yet) favored e5: Don't-show AUC 0.81 for e5-qp centered vs 0.73 for TF-IDF centered and 0.38
for LSA centered. Seed-sim AUC was 0.82 for e5-qp raw vs 0.75 for TF-IDF raw.

## Findings

1. **Similarity to the current video: no representation wins.** All eight variants land at
   0.75–0.78 with heavily overlapping intervals. TF-IDF matches e5 because "good" mostly means
   *same course or series*, and series share words: course codes, lecturer names, channel
   names. For M4, keep both e5 (raw) and TF-IDF seed similarity as ranking features and let the
   ranker weigh them.
2. **Don't-show-me depends on context, and the ranking of representations flipped once new
   contexts were added.** With science and comedy seeds only, e5 centered was best. After
   adding a news seed, e5 dropped to 0.65 and TF-IDF centered rose to 0.81. The intervals
   overlap only slightly, but this is one run on one user's labels. The cause is visible in the
   data: while watching local news, the user labeled a public-broadcaster news episode and a
   national report on trade policy as **good**, and both score 0.33–0.35 on the line "political
   commentary and election news", among the highest Don't-show scores of any video. That also
   explains `r0 = 0` for e5: the highest-scoring *good* video is above every *bad* one. A
   context-free line can't express *"not unless I'm already watching news"*.
3. **Centering matters for profile lines, not for video ↔ video.** It tripled the gate's margin
   (see E2) and lifted TF-IDF and LSA on Don't-show-me, but it doesn't change seed similarity
   (±0.02).
4. **Wording matters more than the model.** The first Don't-show line, "political late-night
   clips", mixed a topic (politics) and a genre (late-night comedy). e5 matched the genre and
   dropped three non-political stand-up clips. Rewording to one concept ("political commentary
   and election news") removed those false drops without any code change.
5. **The two e5 prefix variants are indistinguishable** (≤ 0.05 apart everywhere, overlapping
   intervals). **LSA is never the best**, and on 40 labels it was worse than chance on
   Don't-show-me.

## Decision

- **No switch yet.** Stage C keeps e5-small-qp centered for Don't-show-me. With the reworded line it
  drops only political/news items in the lecture and comedy contexts. TF-IDF
  centered's higher AUC rests largely on the one news context, so it doesn't justify switching
  on its own.
- **Don't re-tune Don't-show-me on the current labels.** The precision-first rule ("just above
  the top good score") would move the threshold to ≈ 0.36 because of the two good news
  videos. That would turn the filter off almost everywhere (most political clips score
  0.24–0.41). Thresholds stay as written before the news labels.
- **Next experiment (E1b): context-relative Don't-show.** Score a candidate by how much *more*
  it matches a Don't-show line than the current video does (`sim(candidate, line) −
  max(0, sim(current, line))`), or suspend a line when the current video itself matches it.
  This tests the user's apparent meaning: don't *push* this on me, but if I chose to watch it,
  related videos are fine. Compare both against the absolute score, with AUC computed per
  context, before M4's ranker uses either.

## Threats to validity

- One labeler, 95 labels, 7 contexts. The intervals are wide, and a single new context moved
  the ranking.
- Labels are contextual and somewhat noisy. "ads" is used for both *meh* and *bad*, and one
  political clip was labeled bad with the reason "ads".
- The same labels set the thresholds (`just tune`) and are used here; there is no held-out set
  yet. M6 will add time-based splits.
- `bad` mixes reasons (politics, influencer, ads, off-topic) that the Don't-show lines were never
  meant to cover, so AUC don't-show understates how well the lines catch what they describe.
  Per-reason breakdowns are a cheap next step.
