# Setup: ChromeOS + Chrome + Crostini (secondary)

Chrome runs on ChromeOS. The backend runs in the Crostini Linux VM. The private folder is
`~/.sidekick/` inside Crostini. Data is not shared with the Windows machine.

## 1. Check the VM

```bash
free -h; nproc; df -h ~
```

Note how much RAM the VM gets (the project assumes a shared 8 GB machine).

## 2. Install tools and clone

Same as the Windows/WSL2 guide: install `git`, `uv` and `just`, then:

```bash
git clone https://github.com/xqzr2000/YouTube-SideKick.git
cd YouTube-SideKick
just sync && just test && just init
```

## 3. Chrome-to-Crostini reachability test

```bash
just dev
```

In ChromeOS Chrome, open `http://localhost:8000/health`. If that fails, try
`http://penguin.linux.test:8000/health`. Record which hostname works and set it as the
extension's backend URL.
