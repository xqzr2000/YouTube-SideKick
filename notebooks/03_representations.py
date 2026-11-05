import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sidekick.embed.evaluate import markdown
    from sidekick.explore import REPRESENTATION_SOURCES, load_representations

    # Categorical slots 1-3 of the dataviz reference palette (validated light-mode, CVD-safe),
    # in fixed order: good = blue, bad = orange, meh = aqua. Raw = blue, centered = orange.
    GOOD, BAD, MEH = "#2a78d6", "#eb6834", "#1baf7a"
    LABEL_COLOR = {"good": GOOD, "bad": BAD, "meh": MEH}
    INK, MUTED, GRID = "#2b2b2a", "#6b6a66", "#e4e3df"

    def tidy(ax, grid_axis="x"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)

    return (
        BAD,
        GOOD,
        INK,
        LABEL_COLOR,
        MUTED,
        REPRESENTATION_SOURCES,
        load_representations,
        markdown,
        mo,
        np,
        pd,
        plt,
        tidy,
    )


@app.cell
def _(mo):
    mo.md("""
    # NB03 — Text representations against your labels (E1, E2)

    Which way of turning video text into vectors should the soft filters and the interest gate
    use? Each representation is scored against the hand labels from `just label`, raw and
    **centered** (the mean video vector subtracted, which spreads out e5's bunched-up cosines):

    1. **E1, Don't-show-me:** does similarity to your Don't-show-me lines rank the videos you
       labeled *bad* above the *good* ones? (AUC with a 95 % bootstrap interval, and `r0`: the
       share of bad videos a precision-first threshold could catch.)
    2. **E1, similarity to the current video:** does it rank *good* above *meh + bad*? Your
       labels say "good" mostly means the same series or a tight topic match.
    3. **E2, interest gate:** do the current videos you said match your Interests score above
       the ones you said don't, and where does τ sit between them?

    Data: `auto` uses `~/.sidekick/sidekick.db` when it has labels (with e5 if downloaded),
    otherwise the synthetic sample with synthetic labels (TF-IDF and LSA only).
    """)
    return


@app.cell
def _(REPRESENTATION_SOURCES, mo):
    source = mo.ui.dropdown(options=list(REPRESENTATION_SOURCES), value="auto", label="data source")
    source
    return (source,)


@app.cell
def _(load_representations, markdown, mo, source):
    loaded = load_representations(source.value)
    results = loaded["results"]
    th = loaded["thresholds"]
    titles = loaded["titles"]
    mo.md(
        f"**Loaded `{loaded['source']}`** ({loaded['n_labels']} label rows; thresholds: "
        f"{th.source}).\n\n" + "\n".join(markdown(results))
    )
    return results, th, titles


@app.cell
def _(mo):
    mo.md("""
    ## 1. AUC per representation (E1)

    Dots are the AUC, lines the 95 % bootstrap interval; the dashed line is chance (0.5).
    Overlapping intervals mean the labels can't tell two options apart yet.
    """)
    return


@app.cell
def _(BAD, GOOD, INK, MUTED, np, plt, results, tidy):
    def auc_figure():
        reps = list(dict.fromkeys(r.rep for r in results))
        fig, axes = plt.subplots(1, 2, figsize=(9, 0.55 * len(reps) + 1.4), sharey=True)
        metrics = [
            ("auc_dont", "auc_dont_ci", "Don't-show-me: bad above good"),
            ("auc_seed", "auc_seed_ci", "Current video: good above meh + bad"),
        ]
        for ax, (key, ci_key, title) in zip(axes, metrics, strict=True):
            for variant, color, dy in (("raw", GOOD, -0.12), ("centered", BAD, 0.12)):
                for i, rep in enumerate(reps):
                    r = next((x for x in results if x.rep == rep and x.variant == variant), None)
                    if r is None or np.isnan(getattr(r, key)):
                        continue
                    lo, hi = getattr(r, ci_key)
                    ax.plot(
                        [lo, hi], [i + dy, i + dy], color=color, linewidth=2, solid_capstyle="round"
                    )
                    ax.scatter(
                        getattr(r, key),
                        i + dy,
                        s=64,
                        color=color,
                        zorder=3,
                        edgecolors="white",
                        linewidths=2,
                        label=variant if i == 0 else None,
                    )
            ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("AUC")
            tidy(ax)
        axes[0].set_yticks(range(len(reps)), reps)
        axes[0].invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    auc_figure()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Interest gate (E2)

    Each dot is a current video, placed at its best similarity to your Interests and coloured
    by your seed label. A gate threshold works when every *yes* sits right of every *no*. The
    dashed line is the saved τ (only drawn where it applies: e5 centered).
    """)
    return


@app.cell
def _(BAD, GOOD, INK, MUTED, plt, results, th, tidy, titles):
    def gate_figure():
        rows = [r for r in results if r.gate]
        if not rows:
            return None
        fig, axes = plt.subplots(len(rows), 1, figsize=(9, 0.75 * len(rows) + 0.8))
        axes = [axes] if len(rows) == 1 else list(axes)
        for ax, r in zip(axes, rows, strict=True):
            for _seed, lab, score in r.gate:
                color = GOOD if lab == "good" else BAD if lab == "bad" else MUTED
                ax.scatter(score, 0, s=72, color=color, edgecolors="white", linewidths=2, zorder=3)
            if r.variant == "centered" and r.rep.startswith("e5"):
                tau = th.for_rep(r.rep)["gate"]
                ax.axvline(tau, color=MUTED, linestyle="--", linewidth=1)
                ax.text(tau, 0.35, f" τ {tau:.2f}", color=MUTED, fontsize=8)
            gap = "" if r.gate_gap != r.gate_gap else f"   gap {r.gate_gap:+.2f}"
            ax.set_title(f"{r.rep} · {r.variant}{gap}", fontsize=9, color=INK, loc="left")
            xs = [v for *_, v in r.gate]
            lo, hi = min(xs), max(xs)
            pad = max((hi - lo) * 0.15, 0.05)  # all-zero TF-IDF rows still get a sane axis
            ax.set_xlim(lo - pad, hi + pad)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
            ax.set_yticks([])
            ax.set_ylim(-0.6, 0.6)
            tidy(ax)
            ax.spines["left"].set_visible(False)
        axes[-1].set_xlabel("best similarity to an Interests line (own scale per row)")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=GOOD, markersize=8, label="yes"),
            plt.Line2D([], [], marker="o", linestyle="", color=BAD, markersize=8, label="no"),
        ]
        fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
        return fig

    gate_rows = [
        {
            "representation": r.rep,
            "variant": r.variant,
            "seed": titles.get(s, s)[:60],
            "label": {"good": "yes", "bad": "no", "meh": "partly"}[lab],
            "score": round(v, 3),
        }
        for r in results
        for s, lab, v in r.gate
    ]
    gate_figure()
    return (gate_rows,)


@app.cell
def _(gate_rows, mo, pd):
    mo.accordion({"gate scores as a table": mo.ui.table(pd.DataFrame(gate_rows))})
    return


@app.cell
def _(mo, results):
    options = [f"{r.rep} · {r.variant}" for r in results]
    default = next(
        (o for o in options if o == "e5-small-qp · centered"),
        next((o for o in options if o.endswith("centered")), options[0] if options else None),
    )
    pick = mo.ui.dropdown(options=options, value=default, label="representation")
    mo.md(f"""
    ## 3. Score distributions by label

    Pick a representation. Left: similarity to the Don't-show-me lines (the Stage C filter;
    the dashed line is the saved threshold where it applies). Right: similarity to the
    current video (a ranking feature from M4 on).

    {pick}
    """)
    return (pick,)


@app.cell
def _(LABEL_COLOR, MUTED, INK, np, pick, plt, results, th, tidy):
    def strips():
        r = next((x for x in results if f"{x.rep} · {x.variant}" == pick.value), None)
        if r is None or not r.scores:
            return None
        labels = [lab for lab in ("good", "meh", "bad") if any(s[2] == lab for s in r.scores)]
        fig, axes = plt.subplots(1, 2, figsize=(9, 0.6 * len(labels) + 1.4), sharey=True)
        rng = np.random.default_rng(0)
        for ax, idx, title in (
            (axes[0], 3, "Don't-show-me similarity"),
            (axes[1], 4, "similarity to the current video"),
        ):
            for i, lab in enumerate(labels):
                xs = [s[idx] for s in r.scores if s[2] == lab and s[idx] is not None]
                ys = i + rng.uniform(-0.18, 0.18, size=len(xs))
                ax.scatter(
                    xs,
                    ys,
                    s=40,
                    color=LABEL_COLOR[lab],
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=3,
                    label=f"{lab} ({len(xs)})",
                )
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            tidy(ax)
        if r.variant == "centered":
            t = th.for_rep(r.rep)["dont_show"]
            axes[0].axvline(t, color=MUTED, linestyle="--", linewidth=1)
            axes[0].text(
                t,
                1.0,
                f" threshold {t:.2f}",
                color=MUTED,
                fontsize=8,
                va="bottom",
                transform=axes[0].get_xaxis_transform(),
            )
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, frameon=False, loc="upper center", ncol=len(names), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
        return fig

    strips()
    return


@app.cell
def _(mo, pd, pick, results, titles):
    _r = next((x for x in results if f"{x.rep} · {x.variant}" == pick.value), None)
    _rows = [
        {
            "label": lab,
            "dont_show": None if d is None else round(d, 3),
            "seed_sim": round(sv, 3),
            "title": titles.get(v, v)[:70],
            "current video": titles.get(s, s)[:40],
        }
        for s, v, lab, d, sv in (_r.scores if _r else [])
    ]
    _df = pd.DataFrame(_rows)
    if len(_df):
        _df = _df.sort_values("dont_show", ascending=False, na_position="last")
    mo.accordion({"labeled videos, highest Don't-show-me first": mo.ui.table(_df)})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes

    *Write what you see here: which representation wins, where the intervals overlap, which
    labeled videos sit on the wrong side of a threshold and why.*
    """)
    return


if __name__ == "__main__":
    app.run()
