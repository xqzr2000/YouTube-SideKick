import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from sidekick.explore import DECISION_SOURCES, load_decisions

    # Categorical slots 1-3 of the dataviz reference palette (validated light-mode, CVD-safe):
    # blue = kept / title-only, orange = dropped, aqua = Stage B / title + description.
    KEPT, DROPPED, STAGE_B = "#2a78d6", "#eb6834", "#1baf7a"

    def tidy(ax):
        from matplotlib.ticker import MaxNLocator

        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_ylabel() in ("distinct videos", "unique videos"):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)

    return DECISION_SOURCES, DROPPED, KEPT, STAGE_B, load_decisions, mo, pd, plt, tidy


@app.cell
def _(mo):
    mo.md("""
    # NB02 — What gets filtered, and why

    Every compute logs one decision per unique video and stage, with all the reasons it was
    dropped and the features it was judged on. This notebook reads those decisions:

    1. the funnel per compute: candidates → after Stage A → after Stage B
    2. drop reasons by stage (a video can carry several)
    3. durations against your `min_duration` / `max_duration` and the **borderline drops**,
       the first candidates for false removals
    4. language: title-only guesses (Stage A) against title + description (Stage B)
    5. sponsor share from SponsorBlock
    6. what still gets through, and which survivors keep coming back: targets for the
       Stage C soft filters

    Data: `auto` uses `~/.sidekick/sidekick.db` when it has decisions, otherwise it replays
    the synthetic sample through Stage A (no Stage B, so sections 4–5 stay empty).
    """)
    return


@app.cell
def _(DECISION_SOURCES, mo):
    source = mo.ui.dropdown(options=list(DECISION_SOURCES), value="auto", label="data source")
    source
    return (source,)


@app.cell
def _(load_decisions, mo, pd, source):
    loaded = load_decisions(source.value)
    df = pd.DataFrame(loaded["decisions"])
    computes = pd.DataFrame(loaded["computes"])
    profile = loaded["profile"]
    rules = profile.rules
    a = df[df.stage == "A"] if len(df) else df
    b = df[df.stage == "B"] if len(df) else df
    mo.md(
        f"**Loaded `{loaded['source']}`:** {len(computes)} computes, {len(df)} decisions "
        f"({len(a)} Stage A, {len(b)} Stage B), {df.video_id.nunique() if len(df) else 0} "
        f"distinct videos. Rules: duration {rules.min_duration_s}–{rules.max_duration_s} s, "
        f"languages {', '.join(rules.languages) or 'any'}."
    )
    return a, b, computes, df, rules


@app.cell
def _(DROPPED, KEPT, STAGE_B, computes, mo, plt, tidy):
    c = computes.copy()
    c["label"] = [
        f"#{i + 1} {ts[5:10]} {ts[11:16]}\n{v if isinstance(v, str) else '(search)'}"
        for i, (ts, v) in enumerate(zip(c.ts, c.current_video_id, strict=True))
    ]
    # computes without Stage B (sample, --no-enrich, older runs) get no B bar, not a copy of A
    c["kept_b"] = c.kept_b.astype(float)
    c["seconds"] = (c.duration_ms / 1000).round(1)
    fig1, ax1 = plt.subplots(figsize=(10, 3.2))
    x = range(len(c))
    w = 0.26
    ax1.bar([i - w for i in x], c.unique_candidates, w * 0.92, color=DROPPED, label="candidates")
    ax1.bar(list(x), c.kept_a, w * 0.92, color=KEPT, label="after Stage A")
    if c.kept_b.notna().any():
        ax1.bar([i + w for i in x], c.kept_b, w * 0.92, color=STAGE_B, label="after Stage B")
    ax1.set_xticks(list(x), c.label, fontsize=8)
    ax1.set(ylabel="unique videos")
    ax1.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0, 1.15))
    tidy(ax1)
    fig1.tight_layout()
    cols = [
        "compute_id",
        "trigger",
        "unique_candidates",
        "kept_a",
        "kept_b",
        "pages_fetched",
        "seconds",
    ]
    mo.vstack([mo.md("## 1. Funnel per compute"), fig1, mo.ui.table(c[cols], selection=None)])
    return


@app.cell
def _(df, mo, pd):
    dropped = df[~df.kept].copy()
    dropped["reason_list"] = dropped.reason.str.split(";")
    exploded = dropped.explode("reason_list")
    exploded["rule"] = exploded.reason_list.str.split(":").str[0]
    by_rule = (
        exploded.groupby(["stage", "rule"])
        .size()
        .rename("drops")
        .reset_index()
        .sort_values(["stage", "drops"], ascending=[True, False])
    )
    multi = int((dropped.reason_list.str.len() > 1).sum())
    detail = (
        dropped.groupby(["reason", "video_id"])
        .agg(
            computes=("compute_id", "nunique"),
            title=("title", "first"),
            channel=("channel", "first"),
        )
        .reset_index()
        .sort_values(["computes", "reason"], ascending=[False, True])
    )
    mo.vstack(
        [
            mo.md(
                f"## 2. Drop reasons\n\n{len(dropped)} drops; {multi} carry more than one "
                "reason (counted once per reason below)."
            ),
            mo.ui.table(by_rule, selection=None),
            mo.md("Each dropped video, with how many computes dropped it:"),
            mo.ui.table(detail if len(detail) else pd.DataFrame({"none": []}), selection=None),
        ]
    )
    return


@app.cell
def _(DROPPED, KEPT, a, mo, plt, rules, tidy):
    dur = a[a.f_duration_s.notna()].drop_duplicates("video_id", keep="last")
    fig3, ax3 = plt.subplots(figsize=(10, 3.2))
    # 3-minute bins line up with the default rules (3 m, 90 m), so no bin straddles a limit
    minutes = (dur.f_duration_s / 60).clip(upper=180)
    ax3.hist(
        [minutes[dur.kept], minutes[~dur.kept]],
        bins=list(range(0, 184, 3)),
        stacked=True,
        color=[KEPT, DROPPED],
        label=["kept", "dropped"],
        edgecolor="white",
        linewidth=1,
    )
    for limit, name in ((rules.min_duration_s, "min"), (rules.max_duration_s, "max")):
        if limit:
            ax3.axvline(limit / 60, color="#52514e", linewidth=1.2, linestyle="--")
            ax3.text(
                limit / 60,
                ax3.get_ylim()[1] * 0.95,
                f" {name} {limit // 60}m",
                fontsize=8,
                color="#52514e",
                va="top",
            )
    ax3.set(xlabel="duration (min, clipped at 180)", ylabel="distinct videos")
    ax3.legend(frameon=False)
    tidy(ax3)
    fig3.tight_layout()

    def near_limit(s):
        lo, hi = rules.min_duration_s, rules.max_duration_s
        return (lo is not None and lo * 0.5 <= s < lo) or (hi is not None and hi < s <= hi * 1.5)

    border = dur[(~dur.kept) & dur.f_duration_s.map(near_limit)]
    border = border.assign(minutes=(border.f_duration_s / 60).round(1))[
        ["video_id", "minutes", "reason", "channel", "title"]
    ]
    mo.vstack(
        [
            mo.md("## 3. Duration against the rules"),
            fig3,
            mo.md(
                "**Borderline drops** (within 50% of a limit). Look for good videos here; "
                "they feed the false-removal rate:"
            ),
            mo.ui.table(border, selection=None),
        ]
    )
    return


@app.cell
def _(KEPT, STAGE_B, a, b, mo, pd, plt, tidy):
    if "f_title_lang_conf" not in a or "f_lang_conf" not in b or not len(b):
        _out = mo.md(
            "## 4. Language\n\nNo Stage B language data yet "
            "(needs `just models` and an enriched compute)."
        )
    else:
        ta = a.dropna(subset=["f_title_lang_conf"]).drop_duplicates("video_id", keep="last")
        tb = b.dropna(subset=["f_lang_conf"]).drop_duplicates("video_id", keep="last")
        both = ta[["video_id", "title", "f_title_lang", "f_title_lang_conf"]].merge(
            tb[["video_id", "f_lang", "f_lang_conf"]], on="video_id"
        )
        fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(11, 3.4))
        bins4 = [i / 20 for i in range(21)]
        ax4.hist(
            ta.f_title_lang_conf,
            bins=bins4,
            alpha=0.8,
            color=KEPT,
            label="title only (A)",
            edgecolor="white",
        )
        ax4.hist(
            tb.f_lang_conf,
            bins=bins4,
            alpha=0.7,
            color=STAGE_B,
            label="title + description (B)",
            edgecolor="white",
        )
        ax4.axvline(0.8, color="#52514e", linestyle="--", linewidth=1.2)
        ax4.set(xlabel="fastText confidence", ylabel="distinct videos")
        ax4.legend(frameon=False, fontsize=8)
        ax5.scatter(
            both.f_title_lang_conf,
            both.f_lang_conf,
            s=36,
            color=STAGE_B,
            edgecolor="white",
            linewidth=1.5,
        )
        ax5.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1)
        ax5.set(xlabel="title only", ylabel="title + description", xlim=(0, 1), ylim=(0, 1))
        for ax in (ax4, ax5):
            tidy(ax)
        fig4.tight_layout()
        summary = pd.DataFrame(
            {
                "videos": [len(both)],
                "A below 0.80": [int((both.f_title_lang_conf < 0.8).sum())],
                "B below 0.80": [int((both.f_lang_conf < 0.8).sum())],
                "label changed A→B": [int((both.f_title_lang != both.f_lang).sum())],
            }
        )
        changed = both[both.f_title_lang != both.f_lang]
        _out = mo.vstack(
            [
                mo.md("## 4. Language: title only vs title + description"),
                fig4,
                mo.ui.table(summary, selection=None),
                mo.md("Videos whose language label changed between A and B:"),
                mo.ui.table(changed, selection=None),
            ]
        )
    _out
    return


@app.cell
def _(DROPPED, b, mo, plt, tidy):
    if "f_sponsor_frac" not in b or b.f_sponsor_frac.notna().sum() == 0:
        _out = mo.md("## 5. Sponsor share\n\nNo SponsorBlock data yet.")
    else:
        sp = b.dropna(subset=["f_sponsor_frac"]).drop_duplicates("video_id", keep="last").copy()
        sp["promo_pct"] = 100 * (sp.f_sponsor_frac + sp.f_selfpromo_frac.fillna(0))
        marked = sp[sp.promo_pct > 0]
        fig5, ax6 = plt.subplots(figsize=(10, 2.8))
        ax6.hist(
            marked.promo_pct,
            bins=[x / 2 for x in range(0, 101, 5)],
            color=DROPPED,
            edgecolor="white",
            linewidth=1,
        )
        ax6.set(
            xlabel="% of video marked sponsor + self-promo (videos with any marked)",
            ylabel="distinct videos",
            xlim=(0, 50),
        )
        tidy(ax6)
        fig5.tight_layout()
        top = (
            sp[sp.promo_pct > 0]
            .sort_values("promo_pct", ascending=False)[
                ["video_id", "promo_pct", "f_sponsor_segments", "channel", "title"]
            ]
            .round(1)
        )
        _out = mo.vstack(
            [
                mo.md(
                    f"## 5. Sponsor share\n\n{int((sp.promo_pct > 0).sum())} of {len(sp)} "
                    "enriched videos have any marked segment."
                ),
                fig5,
                mo.ui.table(top, selection=None),
            ]
        )
    _out
    return


@app.cell
def _(a, b, df, mo):
    final_stage = b if len(b) else a
    kept = final_stage[final_stage.kept]
    n_computes = df.compute_id.nunique()
    persistent = (
        kept.groupby("video_id")
        .agg(
            computes=("compute_id", "nunique"),
            channel=("channel", "first"),
            title=("title", "first"),
            sources=(
                "sources",
                lambda s: ",".join(sorted(set(",".join(s.dropna()).split(",")) - {""})),
            ),
            best_upnext_pos=("upnext_position", "min"),
        )
        .reset_index()
        .sort_values(["computes", "best_upnext_pos"], ascending=[False, True])
    )
    channels = (
        kept.groupby("channel")
        .video_id.nunique()
        .rename("videos")
        .sort_values(ascending=False)
        .reset_index()
    )
    mo.vstack(
        [
            mo.md(
                f"## 6. What gets through\n\nSurvivors of the last stage that ran, across "
                f"{n_computes} computes. Off-topic or unwanted items here are what Stage C "
                "(Don't-show-me similarity, clickbait) and the interest gate have to catch; the "
                "`computes` column is persistence (experiment E6)."
            ),
            mo.ui.table(persistent, selection=None),
            mo.md("Channels among survivors:"),
            mo.ui.table(channels.head(15), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    *Write findings here after running on real data (e.g. which rules fire most, borderline
    drops you'd have wanted, how much description text helps language ID, which unwanted
    videos survive and how often).*
    """)
    return


if __name__ == "__main__":
    app.run()
