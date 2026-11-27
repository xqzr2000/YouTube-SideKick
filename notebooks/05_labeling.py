import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="SideKick labeling")


@app.cell
def _():
    import html
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
        html,
        load_labeling,
        mo,
        nullcontext,
    )


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("## SideKick labeling"),
            mo.accordion(
                {
                    "How this works": mo.md("""
    One question per video: **while watching the current video, would you want this one
    shown next?** Type or pick a reason if you like (optional), then tap **good**, **meh**,
    **bad** or **skip**. Each tap is saved at once; **Undo** takes back the last one.

    - **ranked** (default): every video that reached SideKick's ranking step for a current
      video, shown or not. M6 compares rankers by re-ranking these pools, so the app finishes
      one current video's pool before moving to the next.
    - **filtered**: videos the filters dropped, to measure false removals.
    - **relabel** (E5): 50 labels from your first week of labeling, asked again without
      your old answer, to see whether your taste moved. Answer as you would today. These go
      to a separate table, so nothing else changes.

    Scores, ranks and filter decisions are hidden on purpose, so they can't steer the label,
    and the order is shuffled. For a current video without a seed label, the app first asks
    whether it matches your Interests (yes / partly / no).

    `just label-app` serves this page at `http://localhost:2718`; `just label-app-lan` also
    serves it to a phone or the Chromebook on your home network. The `sample` source is a
    synthetic replay whose labels vanish when the app stops.
    """)
                }
            ),
        ]
    )
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
def _(html, mo):
    def video_card_html(c, width):
        """Thumbnail beside the text on a wide screen, above it on a phone (flex-wrap)."""
        e = html.escape
        desc = (
            f'<div style="font-size:0.85em;opacity:0.8;margin-top:6px">{e(c["description"])}</div>'
            if c["description"]
            else ""
        )
        return mo.Html(
            '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start">'
            f'<img src="{e(c["thumbnail"])}" alt="" '
            f'style="width:{width}px;max-width:100%;border-radius:8px">'
            '<div style="flex:1 1 240px;min-width:0">'
            f'<a href="{e(c["url"])}" target="_blank" rel="noopener" '
            f'style="font-weight:600;font-size:1.05em">{e(c["title"])}</a>'
            f'<div style="margin-top:4px">{e(c["channel"])} · {e(c["duration"])} · '
            f"{e(c['views'])} views · {e(c['age'])}</div>{desc}</div></div>"
        )

    return (video_card_html,)


@app.cell
def _(
    L,
    data,
    db,
    item,
    left,
    mo,
    queue,
    reasons,
    seed_card,
    session,
    set_history,
    set_skip,
    set_tick,
    video_card,
    video_card_html,
):
    # One tap per answer: type or pick a reason first (optional), then tap a label button.
    # The buttons are made fresh for each item, so their callbacks always label that item.
    reason_text = mo.ui.text(placeholder="reason (optional)", full_width=True)
    reason_pick = mo.ui.dropdown(options=reasons, label="or a reason used before", full_width=True)
    _answered = []  # a second tap on the same item does nothing

    def _answer(label):
        def _on_click(_):
            if _answered:
                return
            _answered.append(label)
            if label == "skip":
                set_skip(lambda s: s | {(item.seed_id, item.video_id)})
            else:
                _reason = (reason_text.value or "").strip() or (reason_pick.value or "")
                with db() as s:
                    lid = L.save(s, item, label, reason=_reason, session=session)
                set_history(lambda h: (*h, (item.kind, lid)))
            set_tick(lambda t: t + 1)

        return _on_click

    def _buttons(names):
        kinds = {"good": "success", "bad": "danger"}
        return mo.hstack(
            [
                mo.ui.button(
                    label=shown,
                    kind=kinds.get(label, "neutral"),
                    full_width=True,
                    on_click=_answer(label),
                )
                for shown, label in names
            ],
            widths="equal",
            gap=0.5,
        )

    if item is None:
        view = mo.callout(
            mo.md(
                "Relabel sample done: `just e5` shows how your answers compare."
                if queue.value == "relabel"
                else "Nothing left in this queue. Pick another current video or queue, or use "
                "the panel on more videos."
            ),
            kind="success",
        )
    elif item.kind == "seed":
        _interests = " · ".join(data["interests"]) or "(none in profile.md)"
        view = mo.vstack(
            [
                mo.md("### Current video: does it match one of your Interests?"),
                mo.md(f"<small>Not *do I like it*. Interests: {_interests}</small>"),
                video_card_html(seed_card, 240),
                _buttons([("yes", "good"), ("partly", "meh"), ("no", "bad"), ("skip", "skip")]),
            ]
        )
    else:
        _left = "in the relabel sample" if item.kind == "relabel" else "for this video"
        view = mo.vstack(
            [
                mo.md(
                    f"<small>While watching</small> **{seed_card['title']}** "
                    f"<small>({seed_card['channel']}) · {left} left {_left}</small>"
                ),
                mo.md("### Would you want this shown next?"),
                video_card_html(video_card, 320),
                reason_text,
                reason_pick if reasons else mo.md(""),
                _buttons([(n, n) for n in ("good", "meh", "bad", "skip")]),
            ]
        )
    view
    return


@app.cell
def _(L, db, get_history, mo, set_history, set_tick):
    def _undo(_):
        h = get_history()
        if not h:
            return
        _kind, _id = h[-1]
        with db() as s:
            L.undo(s, _kind, _id)
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
    mo.accordion(
        {
            "Progress": mo.ui.table(
                [
                    {
                        "current video": p.title[:50],
                        "panel": "yes" if p.panel else "",
                        "Interests?": _seed_word[p.seed_label],
                        "ranked": f"{p.ranked_labeled}/{p.ranked}",
                        "shown": f"{p.shown_labeled}/{p.shown}",
                        "filtered": f"{p.filtered_labeled}/{p.filtered}",
                        "g/m/b": "{good}/{meh}/{bad}".format(**p.counts),
                    }
                    for p in _rows
                ],
                selection=None,
                pagination=False,
            )
            if _rows
            else mo.md("No ranked computes yet: use the panel or `just recommend <id>`.")
        }
    )
    return


if __name__ == "__main__":
    app.run()
