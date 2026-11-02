"""Paths for SideKick's private data folder.

Everything private (profile, database, raw logs, models) lives outside the repo in
``~/.sidekick`` by default. Override with the ``SIDEKICK_HOME`` environment variable.
On both dev machines (WSL2 and Crostini) this folder lives inside the Linux filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

ENV_VAR = "SIDEKICK_HOME"
DEFAULT_HOME = "~/.sidekick"


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def profile(self) -> Path:
        return self.home / "profile.md"

    @property
    def db(self) -> Path:
        return self.home / "sidekick.db"

    @property
    def raw_dir(self) -> Path:
        return self.home / "raw"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"


def get_paths() -> Paths:
    """Resolve the private folder from SIDEKICK_HOME (or ~/.sidekick)."""
    return Paths(home=Path(os.environ.get(ENV_VAR, DEFAULT_HOME)).expanduser().resolve())


def ensure_home(paths: Paths | None = None) -> Paths:
    """Create the private folder layout and a starter profile.md. Never overwrites."""
    paths = paths or get_paths()
    for d in (paths.home, paths.raw_dir, paths.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not paths.profile.exists():
        template = resources.files("sidekick").joinpath("templates/profile.md")
        paths.profile.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return paths


if __name__ == "__main__":
    p = ensure_home()
    print(f"SideKick home: {p.home}")
    print(f"profile:       {p.profile}")
