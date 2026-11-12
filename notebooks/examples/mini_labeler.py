import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import random
    import sqlite3
    from datetime import UTC, datetime

    import marimo as mo

    return UTC, datetime, mo, os, random, sqlite3


@app.cell
def _(UTC, datetime, os, random, sqlite3):
    # 1. The data: replace ITEMS with your own (id, text) pairs.
    ITEMS = [
        ("v01", "17. Bayesian Statistics (MIT lecture)"),
        ("v02", "Bayes theorem, the geometry of changing beliefs"),
        ("v03", "Late-night host roasts the president"),
        ("v04", "Principal component analysis for spectra"),
        ("v05", "I tried every pizza in New York"),
        ("v06", "Q-learning explained in 10 minutes"),
    ]
    ORDER = [item_id for item_id, _ in ITEMS]
    random.Random(42).shuffle(ORDER)  # one fixed shuffle of the whole list
    TEXT = dict(ITEMS)

    # 2. The storage: an append-only table, one row per answer.
    DB = os.environ.get("MINI_LABELS_DB", "labels.db")

    def connect():
        con = sqlite3.connect(DB)
        con.execute(
            "CREATE TABLE IF NOT EXISTS labels "
            "(label_id INTEGER PRIMARY KEY, item_id TEXT, label TEXT, reason TEXT, ts TEXT)"
        )
        return con

    def labeled_ids():
        con = connect()
        ids = {r[0] for r in con.execute("SELECT item_id FROM labels")}
        con.close()
        return ids

    def save(item_id, label, reason):
        con = connect()
        cur = con.execute(
            "INSERT INTO labels (item_id, label, reason, ts) VALUES (?, ?, ?, ?)",
            (item_id, label, reason.strip() or None, datetime.now(UTC).isoformat()),
        )
        con.commit()
        con.close()
        return cur.lastrowid

    def delete(label_id):
        con = connect()
        con.execute("DELETE FROM labels WHERE label_id = ?", (label_id,))
        con.commit()
        con.close()

    return ORDER, TEXT, delete, labeled_ids, save


@app.cell
def _(mo):
    # 3. The app's memory: a counter that says "something changed, look again",
    #    and the ids saved in this session (for undo).
    get_tick, set_tick = mo.state(0)
    get_history, set_history = mo.state(())
    return get_history, get_tick, set_history, set_tick


@app.cell
def _(ORDER, get_tick, labeled_ids):
    # 4. The queue: re-read from the database whenever the tick changes.
    get_tick()
    done = labeled_ids()
    queue = [i for i in ORDER if i not in done]
    item = queue[0] if queue else None
    return item, queue


@app.cell
def _(TEXT, item, mo, queue):
    # 5. The screen: one item, one form. A new item means a new, empty form.
    form = None
    if item is None:
        view = mo.callout(mo.md("All done."), kind="success")
    else:
        form = (
            mo.md("{label}\n\n{reason}")
            .batch(
                label=mo.ui.radio(options=["good", "meh", "bad"], inline=True),
                reason=mo.ui.text(placeholder="reason (optional)", full_width=True),
            )
            .form(submit_button_label="Save")
        )
        view = mo.vstack(
            [
                mo.md(f"**{len(queue)} left.** Would you want this next?"),
                mo.md(f"### {TEXT[item]}"),
                form,
            ]
        )
    view
    return (form,)


@app.cell
def _(form, item, save, set_history, set_tick):
    # 6. The action: runs when the form is submitted, saves, then bumps the tick.
    if form is not None and form.value and form.value["label"]:
        _lid = save(item, form.value["label"], form.value["reason"])
        set_history(lambda h: (*h, _lid))
        set_tick(lambda t: t + 1)
    return


@app.cell
def _(delete, get_history, mo, set_history, set_tick):
    # 7. Undo: a button with a callback.
    def _undo(_):
        h = get_history()
        if h:
            delete(h[-1])
            set_history(h[:-1])
            set_tick(lambda t: t + 1)

    undo = mo.ui.button(label="Undo", on_click=_undo)
    return (undo,)


@app.cell
def _(get_history, mo, undo):
    # Its own cell: marimo doesn't rerun the cell whose button changed the state.
    mo.hstack([undo, mo.md(f"{len(get_history())} saved this session")], justify="start")
    return


if __name__ == "__main__":
    app.run()
