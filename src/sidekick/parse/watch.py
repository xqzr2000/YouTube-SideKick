"""Parse a signed-out or signed-in watch page: exact metadata plus the up-next list."""

from __future__ import annotations

from sidekick.models import VideoMeta, WatchPage
from sidekick.parse.lockup import candidate_from_lockup
from sidekick.parse.renderers import candidate_from_renderer
from sidekick.parse.util import extract_json, find_all, parse_count


def parse_meta(player_response: dict) -> VideoMeta | None:
    vd = player_response.get("videoDetails")
    if not isinstance(vd, dict) or not vd.get("videoId"):
        return None
    micro = next(find_all(player_response, "playerMicroformatRenderer"), {})
    length = vd.get("lengthSeconds")
    views = vd.get("viewCount")
    return VideoMeta(
        video_id=str(vd["videoId"]),
        title=str(vd.get("title", "")),
        channel=str(vd.get("author", "")),
        channel_id=str(vd.get("channelId", "")),
        description=str(vd.get("shortDescription", "")),
        keywords=[str(k) for k in vd.get("keywords", [])],
        duration_s=int(length) if str(length).isdigit() else None,
        views=parse_count(str(views)) if views is not None else None,
        publish_date=str(micro.get("publishDate") or micro.get("uploadDate") or ""),
        category=str(micro.get("category", "")),
        is_live=bool(vd.get("isLive") or (vd.get("isLiveContent") and not length)),
        is_upcoming=bool(vd.get("isUpcoming")),
    )


def extract_watch_blobs(html: str) -> dict:
    """The two JSON blobs a watch page carries; this is what the raw log stores."""
    return {
        "player": extract_json(html, "ytInitialPlayerResponse"),
        "data": extract_json(html, "ytInitialData"),
    }


def parse_watch(html: str) -> WatchPage:
    blobs = extract_watch_blobs(html)
    return parse_watch_data(blobs["player"], blobs["data"])


def parse_watch_data(player: dict | None, data: dict | None) -> WatchPage:
    meta = parse_meta(player) if isinstance(player, dict) else None

    upnext = []
    nonvideo = 0
    if isinstance(data, dict):
        secondary = next(find_all(data, "secondaryResults"), None)
        if secondary is not None:
            seen: set[str] = set()
            # Older layout (compactVideoRenderer) and newer layout (lockupViewModel).
            items = [
                candidate_from_renderer(r) for r in find_all(secondary, "compactVideoRenderer")
            ]
            for lk in find_all(secondary, "lockupViewModel"):
                cand = candidate_from_lockup(lk)
                if cand is None:
                    nonvideo += 1
                items.append(cand)
            for cand in items:
                if cand and cand.video_id not in seen:
                    seen.add(cand.video_id)
                    cand.source = "upnext"
                    upnext.append(cand)

    return WatchPage(meta=meta, upnext=upnext, nonvideo_skipped=nonvideo)
