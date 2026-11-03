import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sidekick.explore import SOURCES, load_candidates

    return SOURCES, load_candidates, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # NB01 — Candidate pools

    What does YouTube hand us before any filtering? For each page fetched (a watch page's
    **up-next** list, or a **search** results page) this notebook looks at:

    1. how many candidates each page gives, and how many are Shorts / live / upcoming
    2. durations, view counts and ages, up-next vs search
    3. which channels dominate
    4. overlap and recurrence: how often the same video shows up in several pages
       (the HackerTube-style recurrence feature)

    Data: `auto` uses your raw log, then `just record` fixtures, then the synthetic sample.
    """)
    return


@app.cell
def _(SOURCES, mo):
    source = mo.ui.dropdown(options=list(SOURCES), value="auto", label="data source")
    source
    return (source,)


@app.cell
def _(load_candidates, mo, pd, source):
    loaded = load_candidates(source.value)
    df = pd.DataFrame(loaded["candidates"])
    seeds = pd.DataFrame(loaded["seeds"])
    mo.md(
        f"**Loaded `{loaded['source']}`:** {loaded['pages']} pages, {len(df)} candidates, "
        f"{df['video_id'].nunique() if len(df) else 0} distinct videos, {len(seeds)} seed videos."
    )
    return (df,)


@app.cell
def _(df, mo):
    per_page = (
        df.groupby(["page_kind", "page_key"])
        .agg(
            candidates=("video_id", "size"),
            live=("is_live", "sum"),
            upcoming=("is_upcoming", "sum"),
            shorts=("is_short", "sum"),
            median_dur_min=("duration_s", lambda s: round(s.median() / 60, 1)),
            median_views=("views", "median"),
        )
        .reset_index()
    )
    mo.vstack([mo.md("## 1. Per page"), mo.ui.table(per_page, selection=None)])
    return


@app.cell
def _(df, mo, np, plt):
    clean = df[~(df.is_live | df.is_upcoming | df.is_short)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for kind, color in (("watch", "#3b6ea8"), ("search", "#d08c2b")):
        part = clean[clean.page_kind == kind]
        label = "up-next" if kind == "watch" else "search"
        axes[0].hist(part.duration_s.dropna() / 60, bins=20, alpha=0.6, color=color, label=label)
        axes[1].hist(np.log10(part.views.dropna().clip(lower=1)), bins=20, alpha=0.6, color=color)
        axes[2].hist(part.age_days.dropna() / 365, bins=20, alpha=0.6, color=color)
    axes[0].set(title="Duration (min)", ylabel="videos")
    axes[1].set(title="Views (log10)")
    axes[2].set(title="Age (years, approx.)")
    axes[0].legend()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    mo.vstack(
        [
            mo.md("## 2. Distributions (excluding live, upcoming and Shorts)"),
            fig,
            mo.md(
                "Ages come from relative text (`9y ago`), so they are approximate; "
                "exact dates arrive with Stage B enrichment."
            ),
        ]
    )
    return


@app.cell
def _(df, mo):
    top = (
        df.groupby(["page_kind", "channel"])
        .size()
        .rename("videos")
        .reset_index()
        .sort_values(["page_kind", "videos"], ascending=[True, False])
        .groupby("page_kind")
        .head(8)
    )
    mo.vstack([mo.md("## 3. Top channels per source"), mo.ui.table(top, selection=None)])
    return


@app.cell
def _(df, mo, pd):
    upnext_ids = set(df[df.page_kind == "watch"].video_id)
    search_ids = set(df[df.page_kind == "search"].video_id)
    both = upnext_ids & search_ids
    recurrence = (
        df.groupby("video_id")
        .agg(
            pages=("page_key", "nunique"),
            sources=("page_kind", "nunique"),
            title=("title", "first"),
            channel=("channel", "first"),
        )
        .sort_values(["pages", "sources"], ascending=False)
    )
    dist = recurrence.pages.value_counts().sort_index().rename("videos")
    summary = pd.DataFrame(
        {
            "up-next ids": [len(upnext_ids)],
            "search ids": [len(search_ids)],
            "in both": [len(both)],
            "share of up-next also in search": [
                round(len(both) / len(upnext_ids), 3) if upnext_ids else None
            ],
        }
    )
    mo.vstack(
        [
            mo.md("## 4. Overlap and recurrence"),
            mo.ui.table(summary, selection=None),
            mo.md("Videos by number of pages they appeared on:"),
            mo.ui.table(dist.reset_index(), selection=None),
            mo.md("Most recurrent videos:"),
            mo.ui.table(recurrence.head(10).reset_index(), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    *Write findings here after running on real data (e.g. how many Shorts search pages
    carry, how much up-next and search overlap, which channels recur).*
    """)
    return


if __name__ == "__main__":
    app.run()
