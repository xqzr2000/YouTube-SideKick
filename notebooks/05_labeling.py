import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="SideKick labeling")


@app.cell
def _():
    from contextlib import contextmanager, nullcontext
    from datetime import UTC, datetime

    import marimo as mo

    from sidekick import labeling as L
    from sidekick.explore import LABELING_SOURCES, load_labeling
    from sidekick.store.db import Store

    return (
        L,
        LABELING_SOURCES,
        Store,
        UTC,
        contextmanager,
        datetime,
        load_labeling,
        mo,
        nullcontext,
    )


@app.cell
def _(mo):
    mo.md("""
    # NB05 — Labeling app

    One question per video: **while watching the current video, would you want this one shown
    next?** good / meh / bad, with an optional short reason. Answers are saved at once.

    - **ranked** (default): every video that reached SideKick's ranking step for a current
      video, shown or not. M6 compares rankers by re-ranking these pools, so the app finishes
      one current video's pool before moving to the next.
    - **filtered**: videos the filters dropped, to measure false removals.

    Scores, ranks and filter decisions are hidden on purpose, so they can't steer the label,
    and the order is shuffled. For a current video without a seed label, the app first asks
    whether it matches your Interests (yes / partly / no).

    `just label-app` serves this page at `http://localhost:2718`. The `sample` source is a
    synthetic replay whose labels vanish when the app stops.
    """)
    return


@app.cell
def _(LABELING_SOURCES, L, mo):
    source = mo.ui.dropdown(options=list(LABELING_SOURCES), value="auto", label="data")
    queue = mo.ui.dropdown(options=list(L.QUEUES), value="ranked", label="queue")
    mo.hstack([source, queue], justify="start")
    return queue, source


@app.cell
def _(Store, UTC, contextmanager, datetime, load_labeling, nullcontext, source):
    data = load_labeling(source.value)
    session = "app-" + datetime.now(UTC).strftime("%Y%m%dT%H%MZ")

    @contextmanager
    def db():
        """A fresh connection per read or write for the real database, so the backend is never
        kept waiting; the in-memory sample store stays open."""
        if data["store"] is not None:
            with nullcontext(data["store"]) as s:
                yield s
        else:
            with Store(data["db_path"]) as s:
                yield s

    return data, db, session


@app.cell
def _(mo):
    get_tick, set_tick = mo.state(0)  # bumped after every save, skip or undo
    get_skip, set_skip = mo.state(frozenset())  # (seed, video) pairs skipped this session
    get_history, set_history = mo.state(())  # label ids saved this session, for undo
    return get_history, get_skip, get_tick, set_history, set_skip, set_tick


@app.cell
def _(L, db, mo):
    with db() as _s:
        _seeds = L.progress(_s)
    seed_pick = mo.ui.dropdown(
        options={"next unfinished (newest first)": None}
        | {f"{p.title[:70]} ({p.seed_id})": p.seed_id for p in _seeds},
        value="next unfinished (newest first)",
        label="current video",
    )
    seed_pick
    return (seed_pick,)


@app.cell
def _(L, db, get_skip, get_tick, queue, seed_pick):
    get_tick()
    with db() as _s:
        item = L.next_item(_s, queue.value, seed=seed_pick.value, skip=get_skip())
        seed_card = L.card(_s, item.seed_id) if item else None
        video_card = L.card(_s, item.video_id) if item else None
        left = L.remaining(_s, queue.value, item.seed_id) if item else 0
        reasons = L.recent_reasons(_s)
    return item, left, reasons, seed_card, video_card


@app.cell
def _(data, item, left, mo, reasons, seed_card, video_card):
    def _video(c, width):
        return mo.hstack(
            [
                mo.image(src=c["thumbnail"], width=width, rounded=True),
                mo.md(
                    f"**[{c['title']}]({c['url']})**  \n"
                    f"{c['channel']} · {c['duration']} · {c['views']} views · {c['age']}"
                    + (f"\n\n<small>{c['description']}</small>" if c["description"] else "")
                ),
            ],
            justify="start",
            align="start",
            widths=[0, 1],
        )

    form = None
    if item is None:
        view = mo.callout(
            mo.md(
                "Nothing left in this queue. Pick another current video or queue, or use "
                "the panel on more videos."
            ),
            kind="success",
        )
    elif item.kind == "seed":
        form = (
            mo.md("{label}")
            .batch(
                label=mo.ui.radio(
                    options={"yes": "good", "partly": "meh", "no": "bad", "skip": "skip"},
                    inline=True,
                )
            )
            .form(submit_button_label="Save", clear_on_submit=True)
        )
        _interests = " · ".join(data["interests"]) or "(none in profile.md)"
        view = mo.vstack(
            [
                mo.md("### Current video: does it match one of your Interests?"),
                mo.md(f"<small>Not *do I like it*. Interests: {_interests}</small>"),
                _video(seed_card, 240),
                form,
            ]
        )
    else:
        form = (
            mo.md("{label}\n\n{reason}")
            .batch(
                label=mo.ui.radio(
                    options={"good": "good", "meh": "meh", "bad": "bad", "skip": "skip"},
                    inline=True,
                ),
                reason=mo.ui.text(placeholder="reason (optional)", full_width=True),
            )
            .form(submit_button_label="Save", clear_on_submit=True)
        )
        view = mo.vstack(
            [
                mo.md(
                    f"<small>While watching</small> **{seed_card['title']}** "
                    f"<small>({seed_card['channel']}) · {left} left for this video</small>"
                ),
                mo.md("### Would you want this shown next?"),
                _video(video_card, 320),
                form,
                mo.md("<small>Reasons used before: " + " · ".join(reasons) + "</small>")
                if reasons
                else mo.md(""),
            ]
        )
    view
    return (form,)


@app.cell
def _(L, db, form, item, session, set_history, set_skip, set_tick):
    _v = form.value if form is not None else None
    if _v and _v.get("label"):
        if _v["label"] == "skip":
            set_skip(lambda s: s | {(item.seed_id, item.video_id)})
        else:
            with db() as _s:
                _lid = L.save(_s, item, _v["label"], reason=_v.get("reason") or "", session=session)
            set_history(lambda h: (*h, _lid))
        set_tick(lambda t: t + 1)
    return


@app.cell
def _(db, get_history, mo, set_history, set_tick):
    def _undo(_):
        h = get_history()
        if not h:
            return
        with db() as s:
            s.delete_label(h[-1])
        set_history(h[:-1])
        set_tick(lambda t: t + 1)

    undo = mo.ui.button(label="Undo last label", on_click=_undo)
    return (undo,)


@app.cell
def _(get_history, mo, undo):
    # its own cell: marimo doesn't rerun the cell whose button changed the state
    mo.hstack(
        [undo, mo.md(f"<small>{len(get_history())} saved this session</small>")], justify="start"
    )
    return


@app.cell
def _(L, db, get_tick, mo):
    get_tick()
    with db() as _s:
        _rows = L.progress(_s)
    _seed_word = {"good": "yes", "meh": "partly", "bad": "no", None: "–"}
    mo.vstack(
        [
            mo.md("### Progress"),
            mo.ui.table(
                [
                    {
                        "current video": p.title[:60],
                        "panel": "yes" if p.panel else "",
                        "Interests?": _seed_word[p.seed_label],
                        "ranked labeled": f"{p.ranked_labeled}/{p.ranked}",
                        "shown labeled": f"{p.shown_labeled}/{p.shown}",
                        "filtered labeled": f"{p.filtered_labeled}/{p.filtered}",
                        "good/meh/bad": "{good}/{meh}/{bad}".format(**p.counts),
                    }
                    for p in _rows
                ],
                selection=None,
                pagination=False,
            )
            if _rows
            else mo.md("No ranked computes yet: use the panel or `just recommend <id>`."),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
