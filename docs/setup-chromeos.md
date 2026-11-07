# Setup: ChromeOS + Chrome + Crostini (secondary)

Chrome runs on ChromeOS. The backend runs in the Crostini Linux VM. The private folder is
`~/.sidekick/` inside Crostini. Data is not shared with the Windows machine.

This setup was checked on 2026-11-06 with a zip download of the repo (no git):

| | |
|---|---|
| VM | 6.3 GiB RAM, no swap, 8 CPUs, x86_64, 40 GB disk |
| Tests | all pass |
| Backend URL | `http://localhost:8000` works |
| Panel compute | under 90 s |

## 1. Turn on Linux and check the VM

Turn it on under Settings → About ChromeOS → Developers → Linux development environment.
Give it at least 15 GB of disk. Then, in the Terminal app:

```bash
free -h; nproc; df -h ~; uname -m
```

## 2. Install the tools

```bash
sudo apt update && sudo apt install -y curl unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install rust-just          # the `just` command, from PyPI
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
```

Open a new terminal, then:

```bash
nvm install 22                     # Debian's own nodejs is too old for the extension build
node --version; just --version; uv --version
```

## 3. Get the code (zip, no git)

```bash
cd ~
curl -L -o sidekick.zip https://github.com/xqzr2000/YouTube-SideKick/archive/refs/heads/main.zip
unzip -q sidekick.zip && mv YouTube-SideKick-main YouTube-SideKick && rm sidekick.zip
cd ~/YouTube-SideKick
```

To update later, download a fresh zip and replace the folder. `~/.sidekick/` is not touched.
With git installed, `git clone https://github.com/xqzr2000/YouTube-SideKick.git` works too
(then `git pull` to update).

## 4. Backend, models, profile

```bash
just sync && just test
just init                          # ~/.sidekick/ with a starter profile.md
just models                        # one-time downloads, ~135 MB
nano ~/.sidekick/profile.md        # same Interests / Rules / Don't show me as the other machine
just profile                       # check how it was read
```

## 5. Chrome-to-Crostini reachability

```bash
just dev                           # leave it running
```

In ChromeOS Chrome, open `http://localhost:8000/health`. If that fails, try
`http://penguin.linux.test:8000/health` and use that URL in the extension's Settings.

## 6. The Chrome extension (M5)

In a second terminal tab:

```bash
cd ~/YouTube-SideKick
just ext-install && just ext-test
just ext-build
rm -rf ~/SideKick-extension && cp -r extension/.output/chrome-mv3 ~/SideKick-extension
```

The build lands in `.output`, a hidden folder the file picker doesn't show, hence the copy.

In ChromeOS Chrome, open `chrome://extensions`, switch on **Developer mode**, click **Load
unpacked**, and choose **Linux files › SideKick-extension**. After a rebuild, repeat the `cp`
line and click the reload icon on the SideKick card. Use the extension as described in the
Windows guide, section 7.
