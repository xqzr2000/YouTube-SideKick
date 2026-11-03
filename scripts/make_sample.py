"""Regenerate the synthetic sample (``data/sample/pages.jsonl.gz``).

Everything is made up: ids, titles, channels and counts. The JSON mimics the shape of real
pages as of Nov 2026 (up-next as ``lockupViewModel`` with bare counts, search as
``videoRenderer``), so the sample replays through the same parser as the raw log.

Run: ``uv run python scripts/make_sample.py``
"""

from __future__ import annotations

import gzip
import json
import random
import string
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "pages.jsonl.gz"
rng = random.Random(42)

TOPICS = {
    "bayes": ["Bayesian inference", "Priors explained", "MCMC by hand", "Credible intervals"],
    "rl": ["Q-learning", "Policy gradients", "Double DQN", "Reward shaping"],
    "chem": ["PCA for spectra", "PLS regression", "GC-MS peak picking", "Baseline correction"],
    "comedy": ["Stand-up set", "Crowd work", "Road stories", "Family bits"],
}
CHANNELS = {
    "bayes": ["Lecture Hall A", "Stats Workshop", "Open Course Lab"],
    "rl": ["Agent Lab", "Lecture Hall A", "Deep Notes"],
    "chem": ["Chemometrics Corner", "Open Course Lab"],
    "comedy": ["Clean Comedy Stage", "Laugh Factory Sim", "Open Mic Archive"],
}


def vid() -> str:
    return "".join(rng.choices(string.ascii_letters + string.digits + "-_", k=11))


def compact(n: int) -> str:
    for div, suf in ((1_000_000, "M"), (1_000, "K")):
        if n >= div:
            v = n / div
            return f"{v:.1f}{suf}" if v < 10 else f"{v:.0f}{suf}"
    return str(n)


def clock(s: int) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def age() -> tuple[str, str]:
    unit, long, hi = rng.choice([("y", "years", 9), ("mo", "months", 11), ("w", "weeks", 4)])
    n = rng.randint(1, hi)
    return f"{n}{unit} ago", f"{n} {long} ago"


def make_video(topic: str) -> dict:
    kind = rng.random()
    dur = rng.randint(180, 5400) if topic != "comedy" else rng.randint(240, 3000)
    return {
        "id": vid(),
        "title": f"{rng.choice(TOPICS[topic])} #{rng.randint(1, 40)} (synthetic)",
        "channel": rng.choice(CHANNELS[topic]),
        "dur": None if kind < 0.04 else (rng.randint(15, 59) if kind < 0.08 else dur),
        "views": int(10 ** rng.uniform(2.5, 7)),
        "live": kind < 0.04,
        "short": 0.04 <= kind < 0.08,
        "age": age(),
    }


def lockup(v: dict) -> dict:
    badge = "LIVE" if v["live"] else clock(v["dur"])
    style = "THUMBNAIL_OVERLAY_BADGE_STYLE_LIVE" if v["live"] else "DEFAULT"
    row2 = (
        [{"text": {"content": f"{compact(v['views'] // 100)} watching"}}]
        if v["live"]
        else [
            {
                "text": {"content": compact(v["views"])},
                "accessibilityLabel": f"{v['views']:,} views",
            },
            {"text": {"content": v["age"][0]}, "accessibilityLabel": v["age"][1]},
        ]
    )
    lk = {
        "contentImage": {
            "thumbnailViewModel": {
                "overlays": [
                    {
                        "thumbnailOverlayBadgeViewModel": {
                            "thumbnailBadges": [
                                {"thumbnailBadgeViewModel": {"text": badge, "badgeStyle": style}}
                            ]
                        }
                    }
                ]
            }
        },
        "metadata": {
            "lockupMetadataViewModel": {
                "title": {"content": v["title"]},
                "metadata": {
                    "contentMetadataViewModel": {
                        "metadataRows": [
                            {"metadataParts": [{"text": {"content": v["channel"]}}]},
                            {"metadataParts": row2},
                        ]
                    }
                },
            }
        },
        "contentId": v["id"],
        "contentType": "LOCKUP_CONTENT_TYPE_VIDEO",
    }
    if v["short"]:
        lk["rendererContext"] = {"url": f"/shorts/{v['id']}"}
    return {"lockupViewModel": lk}


def video_renderer(v: dict) -> dict:
    r = {
        "videoId": v["id"],
        "title": {"runs": [{"text": v["title"]}]},
        "ownerText": {"runs": [{"text": v["channel"]}]},
        "publishedTimeText": {"simpleText": v["age"][1]},
        "viewCountText": {"simpleText": f"{v['views']:,} views"},
        "thumbnailOverlays": [],
    }
    if v["live"]:
        r["viewCountText"] = {"simpleText": f"{v['views'] // 100:,} watching"}
        r["thumbnailOverlays"].append({"thumbnailOverlayTimeStatusRenderer": {"style": "LIVE"}})
    else:
        r["lengthText"] = {"simpleText": clock(v["dur"])}
    return {"videoRenderer": r}


def main() -> None:
    pool = {t: [make_video(t) for _ in range(45)] for t in TOPICS}
    records = []
    ts = "2026-11-03T08:{:02d}:00+00:00"
    n = 0
    for topic in TOPICS:
        seed = rng.choice([v for v in pool[topic] if not (v["live"] or v["short"])])
        # Up-next: mostly same topic, a few from others (as on real pages).
        same = rng.sample([v for v in pool[topic] if v is not seed], 16)
        other = rng.sample([v for t in TOPICS if t != topic for v in pool[t]], 4)
        upnext = same + other
        rng.shuffle(upnext)
        player = {
            "videoDetails": {
                "videoId": seed["id"],
                "title": seed["title"],
                "author": seed["channel"],
                "channelId": "UCsynthetic",
                "lengthSeconds": str(seed["dur"]),
                "viewCount": str(seed["views"]),
                "keywords": [topic, "synthetic"],
                "shortDescription": f"Synthetic {topic} video for the public sample.",
            },
            "microformat": {
                "playerMicroformatRenderer": {
                    "publishDate": "2024-01-15T10:00:00-08:00",
                    "category": "Comedy" if topic == "comedy" else "Education",
                }
            },
        }
        data = {"secondaryResults": {"results": [lockup(v) for v in upnext]}}
        records.append(
            {
                "ts": ts.format(n),
                "kind": "watch",
                "key": seed["id"],
                "data": {"player": player, "data": data},
            }
        )
        n += 1
        # One search per topic, overlapping partly with that topic's up-next.
        results = rng.sample(same, 6) + rng.sample(pool[topic], 14)
        seen, uniq = set(), []
        for v in results:
            if v["id"] not in seen:
                seen.add(v["id"])
                uniq.append(v)
        sdata = {
            "contents": [video_renderer(v) for v in uniq]
            + [{"reelItemRenderer": {"videoId": vid()}} for _ in range(rng.randint(5, 20))]
        }
        records.append(
            {
                "ts": ts.format(n),
                "kind": "search",
                "key": f"{topic} synthetic query",
                "data": {"data": sdata},
            }
        )
        n += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for rec in records:
            gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode())
    print(f"wrote {len(records)} pages to {OUT}")


if __name__ == "__main__":
    main()
