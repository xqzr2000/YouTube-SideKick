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

## 4. The Chrome extension (M5)

Install Node.js 22+ in Crostini (`sudo apt install -y nodejs npm`, or nvm if apt's is older),
then:

```bash
just ext-install && just ext-build
```

In ChromeOS Chrome, open `chrome://extensions`, switch on **Developer mode**, click **Load
unpacked**, and choose `Linux files › YouTube-SideKick › extension › .output › chrome-mv3`.
In the extension's **Settings**, set the backend URL to the hostname that worked in step 3.
After a rebuild, click the reload icon on the SideKick card. The other steps are the same as in
the Windows guide, section 7.
