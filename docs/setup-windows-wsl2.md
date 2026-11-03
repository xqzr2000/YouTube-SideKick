# Setup: Windows 11 + Chrome + WSL2 (primary dev box)

Chrome runs on Windows. The backend runs inside WSL2 (Ubuntu). Everything private lives in the
WSL2 home at `~/.sidekick/`. WSL2 is never linked to GitHub with credentials here: you only
`git clone`/`git pull` from the public repo.

## 1. Check the WSL2 environment

In Ubuntu (WSL2):

```bash
free -h; nproc; lscpu | grep 'Model name'; df -h ~
cat /etc/wsl.conf; ps -p 1 -o comm=     # "systemd" means systemd is on
```

In Windows PowerShell: `wsl -l -v` (VERSION should be 2).

If WSL2 uses too much RAM, cap it in `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=6GB
```

then run `wsl --shutdown` from PowerShell.

## 2. Install tools (Ubuntu)

```bash
sudo apt update && sudo apt install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv (installs Python 3.12 on demand)
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
# open a new shell so ~/.local/bin is on PATH
```

## 3. Clone inside WSL2 (not under /mnt/c)

```bash
cd ~ && git clone https://github.com/xqzr2000/YouTube-SideKick.git
cd YouTube-SideKick
just sync && just test
just init        # creates ~/.sidekick/ with a starter profile.md (never overwrites)
```

To get updates later: `git pull`.

## 4. Chrome-to-WSL2 reachability test

```bash
just dev
```

Open `http://localhost:8000/health` in **Windows Chrome**. Expected: `{"status":"ok",...}`.

If it doesn't load: Windows 11 normally forwards WSL2 `localhost` automatically. If it doesn't, add
to `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

then `wsl --shutdown` and retry. Record which setup worked here.

## Notes
- Keep the repo and `~/.sidekick/` in the WSL2 filesystem for fast I/O and safe SQLite locking.
- The extension reads its backend URL from settings (default `http://localhost:8000`).

## 5. Record real pages and check the parser (M1)

Real pages stay in the private folder (`~/.sidekick/fixtures/`), never in the repo.

```bash
just record <video id or URL> <another> --search "some query"   # 1-2 s delay between requests
just parse-check ~/.sidekick/fixtures/*.html
```

`parse-check` prints which fields were extracted and which are missing (no page text beyond
short titles), so its output is safe to paste back when the parser needs fixing. Set
`SIDEKICK_USER_AGENT` to override the default browser-like User-Agent.

## 6. Fetch candidates and explore them (M1)

```bash
just fetch <video id or URL>                     # watch page + up-next table, raw log written
just fetch <id> --search "query" --check-thumbs  # add searches; check midpoint thumbnails
just nb                                          # NB01 in marimo (opens a browser URL)
```

`just fetch` waits 1-2 s between requests, stops at 60 pages, and on a consent/bot page or
HTTP 429 refuses further fetches for 30 minutes (`~/.sidekick/blocked_until`; delete the file
to reset). `just nb` prints a `http://localhost:2718/...` URL to open in Windows Chrome; the
notebook reads your raw log, then `just record` fixtures, then the synthetic sample.
