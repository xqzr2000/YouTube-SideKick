"""``just profile``: show how SideKick reads your ``profile.md``."""

from __future__ import annotations

import sys

from sidekick.config import get_paths
from sidekick.profile.store import load_profile


def _dur(s: int | None) -> str:
    if s is None:
        return "-"
    m, sec = divmod(s, 60)
    return f"{m}m{sec:02d}s" if sec else f"{m}m"


def main() -> int:
    paths = get_paths()
    p = load_profile(paths)
    r = p.rules
    print(f"profile:  {paths.profile}")
    print(f"hash:     {p.hash[:12]}")
    print(f"\ninterests ({len(p.interests)}):")
    for i in p.interests:
        print(f"  - {i}")
    print("\nrules:")
    print(f"  languages:        {', '.join(r.languages) or '(any)'}")
    print(f"  duration:         {_dur(r.min_duration_s)} .. {_dur(r.max_duration_s)}")
    print(f"  exclude_channels: {', '.join(r.exclude_channels) or '-'}")
    print(f"  exclude_keywords: {', '.join(r.exclude_keywords) or '-'}")
    print(f"\ndon't show me ({len(p.dont_show)}):")
    for d in p.dont_show:
        extra = f"   [{d.date}; {d.video}]" if d.date else ""
        print(f"  - {d.text}{extra}")
    if p.warnings:
        print(f"\nwarnings ({len(p.warnings)}):")
        for w in p.warnings:
            print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
