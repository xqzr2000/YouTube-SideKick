# How to build a labeling app (a data-science study note)

Labels are the most expensive data in most projects, and the tool you collect them with shapes
what they mean. This note walks through how SideKick's labeling app (`notebooks/05_labeling.py`,
`just label-app`) was designed and built, so you can build one for any project.

It has three parts:

1. **Design**: the decisions that make labels useful, before any code.
2. **Build**: marimo's reactive model, then a 150-line working app you can run and change
   ([`notebooks/examples/mini_labeler.py`](../notebooks/examples/mini_labeler.py)).
3. **Use**: testing the app, checking label quality, and exercises.

Ready-made tools exist (Label Studio, Prodigy, Argilla) and are the right choice for large
teams and standard tasks (boxes on images, spans in text). A small custom app wins when the
question depends on context (here: *which video were you watching?*), the data is private,
and you want the queue to follow your own evaluation. Building one also teaches you the parts
that the ready-made tools decide for you.

---

## Part 1: Design

### 1. Start from the metric, not from the data

Ask first: *what number will these labels compute?* The answer decides what to label.

| You want to measure | You need labels on |
|---|---|
| Precision of a filter's removals | a sample of what the filter **removed** |
| False-removal rate | removals too, mostly the borderline ones |
| Precision@10 of a ranker | that ranker's **top 10** |
| NDCG@10 of a *new* ranker, offline | the **whole candidate pool**, because the new ranker can move any candidate into its top 10 |
| A classifier's quality | a **random** sample, or a stratified one with known weights |

SideKick learned this the hard way. `just label` (M3) spread labels across every current
video, which was right for scoring text representations (E1). For M6, a re-ranker is
evaluated by re-ordering logged pools, and an unlabeled video in its top 10 is a hole in the
metric. So the app labels **one pool completely before moving on** (`sidekick/labeling.py`,
`next_item`).

**Rule:** write down the metric and its formula, then check that every term the formula needs
will have a label.

### 2. Ask one question, and make it answerable in seconds

- **One question per screen.** SideKick asks: *while watching the current video, would you
  want this one shown next?*
- **Make the context part of the question.** A stand-up clip is good after another stand-up
  clip and bad after a statistics lecture. A label without its context is noise. SideKick
  stores labels per (current video, candidate).
- **A small graded scale beats a binary one** for ranking: good / meh / bad becomes 2 / 1 / 0
  for NDCG. Five levels slow you down and blur the middle.
- **Separate questions get separate labels.** "Does this video match my Interests?" (the seed
  question) and "do I want this next?" are different, and mixing them was a real mistake early
  on: every seed got "yes" until the Interests lines were shown with the question.
- **An optional free-text reason** costs a few seconds and pays back later: reasons become a
  taxonomy ("politics", "drifted from the course", "influencer") and point at missing
  features.

### 3. Don't let the model steer the labeler

If the labeler can see what the model thinks, the labels drift toward the model and the
evaluation becomes circular. Things to hide:

- **scores, ranks and filter decisions** (`labeling.card()` returns none of them, and a test
  checks that);
- **whether a video was shown**: mix shown and not-shown videos in one queue;
- **which system produced it**: for the E4 comparison, SideKick's top 10 and YouTube's top 10
  were labeled mixed and unmarked;
- **the order**: shuffle, but with a **fixed seed** (`random.Random(f"{seed}:{queue}")`). A
  fresh shuffle on every rerun makes the screen jump, and a sorted order leaks rank.

### 4. Store answers so you can audit them

- **Append-only rows**: `label_id, item, context, label, reason, session, ts`. Never update in
  place. The newest row per (context, item) is the current label.
- **Keep a session id and a timestamp.** They let you split train and test by time, compare
  sessions, and spot a tired late-night session.
- **Undo means deleting a row this session created**, never editing old ones.
- **Keep the database outside the repo** if the content is personal (SideKick:
  `~/.sidekick/`), and open it per read or write so other programs aren't kept waiting.

### 5. Ergonomics are data quality

- **The fewest clicks:** pick a label, maybe type a reason, then Save. Skip is a label option,
  not a separate flow.
- **Undo** and **progress** ("16 left for this video", a table of finished pools). People label
  more carefully when a mistake is cheap and the end is visible.
- **Show the reasons used before,** so the wording stays consistent across sessions.
- **Everything is saved immediately,** so a session can stop at any point.

---

## Part 2: Build

### The architecture: logic module plus a thin UI

```
sidekick/labeling.py          notebooks/05_labeling.py
(pure Python, tested)         (marimo: layout and clicks only)
- ranked_pools / filtered     - pickers (data, queue, current video)
- next_item (queue order)     - card + form
- save / labeled / progress   - save cell, undo button
- card (what to show)         - progress table
```

Keep every decision (what to ask next, what counts as done) in plain functions that take a
database and return data. Then:

- you can unit-test them without a browser (`tests/test_labeling.py`);
- the CLI, the notebook and a future web page can share them;
- the notebook stays short enough to read.

### marimo in five ideas

marimo notebooks are Python files where each cell is a function.

1. **Cells form a graph.** A cell runs again when a variable it *uses* changes. The variables
   it *returns* are what other cells can use. No hidden state from running cells out of order.
2. **UI elements are values.** `mo.ui.dropdown(...)` shown on screen; when you change it, every
   cell that reads `dropdown.value` reruns. The last expression of a cell is what it displays.
3. **A form collects inputs and submits once.** `mo.md("{label} {reason}").batch(label=...,
   reason=...).form()` gives `form.value` = `None` until Save, then a dict. That's what you want
   for labeling: no half-finished answer ever reaches the database.
4. **`mo.state` is for "something happened".** `get, set = mo.state(0)`. Calling `set(...)`
   reruns the cells that call `get()`. A labeling app uses it as a *tick*: after a save, bump
   the tick, and the queue cell re-reads the database.
5. **The self-loop rule.** marimo does **not** rerun the cell that called the setter. If a
   button's callback changes state, a display of that state in the *same* cell won't update.
   Put the display in its own cell. (SideKick hit this: the "saved this session" counter didn't
   change after Undo.)

### The data flow

```
            ┌───────────── tick (mo.state) ◄────────────┐
            ▼                                            │
   queue cell: read DB → next item                       │
            ▼                                            │
   view cell: card + a NEW empty form for this item      │
            ▼                                            │
   save cell: if form.value → write DB ──────────────────┘
```

The loop closes without any manual "next" logic. Saving changes the database, the tick makes
the queue cell look again, the next item gets a fresh form, and the save cell sees an empty
form and does nothing. Skip works the same way: add the item to a `skip` state instead of
writing a label.

### The minimal app

[`notebooks/examples/mini_labeler.py`](../notebooks/examples/mini_labeler.py) is the whole
pattern in about 150 lines, with no SideKick code. It labels a hard-coded list into
`labels.db` in the current folder.

Run it:

```bash
cd ~/YouTube-SideKick
uv run --group nb marimo edit notebooks/examples/mini_labeler.py   # see and change the code
uv run --group nb marimo run notebooks/examples/mini_labeler.py    # the app view only
```

Its eight cells, in order:

| Cell | What it does | marimo idea |
|---|---|---|
| imports | `marimo`, `sqlite3`, `random`, `datetime` | returned names are shared |
| data + storage | `ITEMS`, one fixed shuffle (`ORDER`), `connect`, `labeled_ids`, `save`, `delete` | plain functions, no UI |
| state | `tick` and `history` (label ids for undo) | `mo.state` |
| queue | `get_tick()`, read labeled ids, first unlabeled item | reruns on every tick |
| view | the item, a radio + text form | form value is `None` until Save |
| save | if the form has a label: save, push the id, bump the tick | setter → rerun the queue |
| undo | a button whose callback deletes the last id and bumps the tick | `on_click` |
| counter | "N saved this session" | its own cell (self-loop rule) |

The key part, the save cell:

```python
@app.cell
def _(form, item, save, set_history, set_tick):
    if form is not None and form.value and form.value["label"]:
        _lid = save(item, form.value["label"], form.value["reason"])
        set_history(lambda h: (*h, _lid))
        set_tick(lambda t: t + 1)
    return
```

Two details matter:

- `set_tick(lambda t: t + 1)` passes a function, so the save cell doesn't need to read
  `get_tick()`. If it did, it would depend on the tick and run in a loop.
- Names starting with `_` (`_lid`) are private to the cell. marimo requires every other name to
  be defined in exactly one cell.

### From the minimal app to SideKick's

| Minimal app | SideKick (`labeling.py` + NB05) | Why |
|---|---|---|
| `ITEMS` list | `ranked_pools` / `filtered_pools` from logged decisions | the queue follows the metric (Part 1, §1) |
| one shuffle of all items | a fixed shuffle per current video and queue | stable and blind, per context |
| item id | (current video, candidate) via `compute_id` | the label depends on context |
| no context shown | "While watching ..." and the seed question first | the question includes the context |
| text only | midpoint thumbnail, channel, duration, views, age, description | enough to judge in seconds |
| — | skip state, reasons used before, progress table | ergonomics (Part 1, §5) |
| `labels.db` | `~/.sidekick/sidekick.db`, opened per read/write | private data; `just dev` runs alongside |
| — | a `sample` source with stand-in rankings | CI and demos without private data |

---

## Part 3: Use

### Testing the app

Three layers, cheapest first:

1. **Unit tests on the logic** with a small hand-built database: "the seed question comes
   first", "a whole pool before the next seed", "Shorts never appear in the filtered queue",
   "rerunning doesn't reshuffle", "undo puts the item back" (`tests/test_labeling.py`).
2. **A smoke test of the notebook**: `app.run()` executes every cell once with empty forms.
   It catches import errors and broken cell wiring (`test_nb05_runs_on_sample`, and
   `test_mini_labeler_runs` for the example).
3. **A real browser, once per change to the UI.** Start `marimo run --headless`, then click
   through with Playwright and check the database. This is how the undo-counter bug was found:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    page = p.chromium.launch().new_page()  # `playwright install chromium` once
    page.goto("http://localhost:2718")
    page.wait_for_timeout(8000)  # let marimo run the cells
    page.get_by_role("radio", name="good").first.check(force=True)
    page.get_by_placeholder("reason (optional)").fill("relevant")
    page.get_by_role("button", name="Save").click()
    # then: SELECT * FROM labels, and check the row is there
```

### Checking the labels themselves

A labeling app is a measuring instrument, so check its error.

- **Balance:** counts per label and per context. 90 % "good" means the queue is too easy or
  the question too lenient.
- **Self-agreement:** a week later, relabel about 10 % of the items blind and compute Cohen's
  kappa (`sklearn.metrics.cohen_kappa_score`). Below about 0.6, the question or the scale is
  unclear. With one labeler, this is the honest error bar on every metric built on the labels.
- **Reasons as data:** group the reasons. A frequent reason with no matching feature
  ("influencer") is a feature to build.
- **Time per label:** the `ts` gaps. Very fast runs at the end of a session are worth a second
  look.
- **How many labels:** plot the metric's bootstrap interval against the number of labeled
  contexts. Stop when it stops shrinking enough to change a decision. SideKick resamples
  **contexts (seeds), not videos**, because videos watched in one context aren't independent.

### Exercises

1. Run the minimal app, label all six items, then open `labels.db` with `sqlite3` or pandas
   and compute the label counts.
2. Add a **skip** option to the minimal app without writing a row (hint: a `mo.state` holding a
   set of skipped ids, read in the queue cell).
3. Add a **context**: label pairs (query, item) instead of items, and make the queue finish one
   query before the next.
4. **Active learning:** train a logistic regression on the labels so far and put the items
   with predicted probability nearest 0.5 first. Compare how fast accuracy rises against a
   random order.
5. **Self-agreement:** add a mode that re-asks 10 % of labeled items blind, then compute
   Cohen's kappa between the two passes.
6. Read `sidekick/labeling.py` and its tests. Find where the queue hides rank and where a
   Short is kept out, and change the shuffle seed to see that the order changes but stays
   stable.
