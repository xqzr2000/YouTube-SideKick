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

## 7. The Chrome extension: SideKick's Selection (M5)

The extension is built inside WSL2 and loaded into Windows Chrome as an unpacked extension.

**Once: Node.js 22+ in WSL2**

```bash
sudo apt update && sudo apt install -y nodejs npm
node --version        # needs v22 or newer
```

If apt gives an older Node, install it with nvm instead:
`curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash`, open a new
terminal, then `nvm install 22`.

**Build and load**

```bash
just ext-install      # once, and again after package changes
just ext-win          # builds and copies it to C:\Users\<you>\SideKick-extension
```

1. In Windows Chrome, open `chrome://extensions` and switch on **Developer mode** (top right).
2. Click **Load unpacked** and pick the folder `just ext-win` printed.
3. Pin "YouTube SideKick" in the toolbar if you like. The toolbar icon and the right-click
   menu **SideKick's Selection** both open the side panel.
4. **Reload any YouTube tab that was already open,** so the extension can read its up-next
   list. Without a reload the panel still works, using only the signed-out list the backend
   fetches.

**Use it.** Keep `just dev` running in a WSL2 terminal. Open a YouTube video, then open the
panel: it shows progress (about 70 s cold, 20–40 s warm), then 10 cards.

- **Click a card:** the video plays in the same tab, and the panel recomputes for it.
- **👎:** asks for a note. The note is appended to `~/.sidekick/profile.md` and applies from
  the next refresh.
- **Settings:** the link at the bottom of the panel; the default backend is
  `http://localhost:8000`.

**After pulling changes,** run `just ext-win` again, then click the reload icon on the
SideKick card in `chrome://extensions`. Backend changes only need `just dev` (it reloads by
itself).
