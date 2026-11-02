from sidekick.config import ENV_VAR, ensure_home, get_paths


def test_home_defaults_and_override(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "sk"))
    paths = get_paths()
    assert paths.home == (tmp_path / "sk").resolve()
    assert paths.db.name == "sidekick.db"


def test_ensure_home_creates_layout_and_keeps_profile(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "sk"))
    paths = ensure_home()
    assert paths.raw_dir.is_dir() and paths.models_dir.is_dir()
    assert "## Interests" in paths.profile.read_text(encoding="utf-8")

    paths.profile.write_text("## Interests\n- mine\n", encoding="utf-8")
    ensure_home()
    assert paths.profile.read_text(encoding="utf-8") == "## Interests\n- mine\n"
