// The side panel: runs a compute when it opens, shows progress, then 10 cards.
// Click a card → the video plays in the same tab, the click is logged, and the panel
// recomputes for that video. 👎 → a note is required; it goes to profile.md and applies from
// the next compute (this list doesn't change).
import './style.css';
import { Api, BackendError, type Card, type Job, type Trigger, waitForJob } from '@/lib/api';
import { DEFAULT_BACKEND, formatDuration } from '@/lib/format';
import { type TabItem, videoIdFromUrl } from '@/lib/upnext';

interface PageInfo {
  tabId?: number;
  videoId: string | null;
  upnext: TabItem[];
}

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = el<HTMLElement>('status');
const cardsEl = el<HTMLOListElement>('cards');
const currentEl = el<HTMLParagraphElement>('current');

let run = 0; // bumped by every new compute; older polling loops stop updating the panel
let tabId: number | undefined;

const sleep = (ms: number) => new Promise<void>((res) => setTimeout(res, ms));

async function backend(): Promise<Api> {
  const { backendUrl } = await browser.storage.local.get('backendUrl');
  return new Api((backendUrl as string | undefined) || DEFAULT_BACKEND);
}

async function askTab(id: number): Promise<{ videoId: string | null; upnext: TabItem[] } | null> {
  try {
    return await browser.tabs.sendMessage(id, { type: 'sidekick:read-page' });
  } catch {
    return null; // no content script yet (tab opened before the extension was loaded)
  }
}

async function readActiveTab(): Promise<PageInfo> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  const videoId = videoIdFromUrl(tab?.url);
  if (tab?.id == null || !videoId) {
    const { lastVideo } = await browser.storage.local.get('lastVideo');
    return { tabId: tab?.id, videoId: (lastVideo as string | undefined) ?? null, upnext: [] };
  }
  const r = await askTab(tab.id);
  return { tabId: tab.id, videoId: r?.videoId ?? videoId, upnext: r?.upnext ?? [] };
}

/** After navigating, wait for the new page's up-next to render (up to ~12 s). */
async function readAfterNavigation(id: number, videoId: string): Promise<TabItem[]> {
  let best: TabItem[] = [];
  for (let i = 0; i < 12; i++) {
    await sleep(1000);
    const r = await askTab(id);
    if (r?.videoId !== videoId) continue;
    if (r.upnext.length > best.length) best = r.upnext;
    if (best.length >= 10 || (i >= 6 && best.length > 0)) break;
  }
  return best;
}

function setStatus(html: Node[] | string, cls = ''): void {
  statusEl.className = `status ${cls}`.trim();
  statusEl.replaceChildren(...(typeof html === 'string' ? [document.createTextNode(html)] : html));
}

function node<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls = '',
  text = '',
): HTMLElementTagNameMap[K] {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text) n.textContent = text;
  return n;
}

function renderProgress(j: Job): void {
  const head = node(
    'p',
    'progress-head',
    j.status === 'queued'
      ? 'Waiting for the previous compute to finish…'
      : `Working… ${Math.round(j.elapsed_s ?? 0)} s`,
  );
  const list = node('ul', 'steps');
  for (const s of j.steps) list.append(node('li', '', s.text));
  setStatus([head, list], 'busy');
}

function renderJob(j: Job): void {
  if (j.current_title) currentEl.textContent = `Next to: ${j.current_title}`;
  switch (j.status) {
    case 'queued':
    case 'running':
      cardsEl.replaceChildren();
      renderProgress(j);
      return;
    case 'done':
      setStatus(j.cards.length ? '' : 'No videos survived the filters this time. Try Refresh.');
      renderCards(j);
      return;
    case 'empty':
    case 'none':
      currentEl.textContent = '';
      cardsEl.replaceChildren();
      setStatus('Open a YouTube video, then SideKick picks the next ten.');
      return;
    case 'blocked':
      setStatus(`YouTube asked for a pause: ${j.error ?? ''} Try again later.`, 'error');
      return;
    case 'error':
      setStatus(`The compute failed: ${j.error ?? 'unknown error'}`, 'error');
      return;
    case 'superseded':
      return;
  }
}

function renderCards(j: Job): void {
  cardsEl.replaceChildren(...j.cards.map((c) => cardNode(c, j)));
}

function cardNode(c: Card, j: Job): HTMLLIElement {
  const li = node('li', 'card');
  const open = node('button', 'open');
  open.type = 'button';
  open.title = c.title;
  const img = node('img', 'thumb');
  img.src = c.thumbnail;
  img.alt = '';
  img.loading = 'lazy';
  const meta = node('span', 'meta');
  meta.append(
    node('span', 'title', c.title),
    node('span', 'sub', [c.channel, formatDuration(c.duration_s)].filter(Boolean).join(' · ')),
  );
  open.append(img, meta);
  open.addEventListener('click', () => void openCard(c, j));

  const down = node('button', 'down', '👎');
  down.type = 'button';
  down.title = "Don't show me videos like this (a note is required)";
  const form = noteForm(c, j, li);
  down.addEventListener('click', () => {
    form.hidden = !form.hidden;
    li.classList.toggle('noting', !form.hidden);
    if (!form.hidden) form.querySelector('textarea')?.focus();
  });
  li.append(open, down, form);
  return li;
}

function noteForm(c: Card, j: Job, li: HTMLLIElement): HTMLFormElement {
  const form = node('form', 'note');
  form.hidden = true;
  const ta = node('textarea');
  ta.placeholder = 'Why not this one? (required, saved to your profile)';
  ta.rows = 2;
  const save = node('button', '', 'Save');
  save.type = 'submit';
  save.disabled = true;
  const cancel = node('button', 'link', 'Cancel');
  cancel.type = 'button';
  const msg = node('span', 'msg');
  form.append(ta, save, cancel, msg);
  ta.addEventListener('input', () => (save.disabled = !ta.value.trim()));
  cancel.addEventListener('click', () => {
    form.hidden = true;
    ta.value = '';
    save.disabled = true;
    li.classList.remove('noting');
  });
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const note = ta.value.trim();
    if (!note) return;
    save.disabled = true;
    try {
      await (await backend()).feedback('downvote', c.video_id, j.compute_id, note);
      li.classList.remove('noting');
      li.classList.add('downvoted');
      form.replaceChildren(node('span', 'msg', 'Saved to your profile. Applies from the next refresh.'));
    } catch (err) {
      msg.textContent = err instanceof Error ? err.message : String(err);
      save.disabled = false;
    }
  });
  return form;
}

async function openCard(c: Card, j: Job): Promise<void> {
  const my = ++run;
  const api = await backend();
  api.feedback('click', c.video_id, j.compute_id).catch(() => {});
  if (tabId == null) tabId = (await browser.tabs.query({ active: true, currentWindow: true }))[0]?.id;
  if (tabId == null) return;
  await browser.tabs.update(tabId, { url: `https://www.youtube.com/watch?v=${c.video_id}` });
  currentEl.textContent = `Next to: ${c.title}`;
  cardsEl.replaceChildren();
  setStatus('Opening the video…', 'busy');
  const upnext = await readAfterNavigation(tabId, c.video_id);
  if (my !== run) return;
  await compute('click', { tabId, videoId: c.video_id, upnext });
}

async function compute(trigger: Trigger, given?: PageInfo): Promise<void> {
  const my = ++run;
  try {
    const page = given ?? (await readActiveTab());
    if (page.tabId != null) tabId = page.tabId;
    if (my !== run) return;
    setStatus('Starting…', 'busy');
    const api = await backend();
    const first = await api.startCompute(page.videoId, page.upnext, trigger);
    await waitForJob(api, first, (j) => my === run && renderJob(j), {
      cancelled: () => my !== run,
    });
  } catch (e) {
    if (my !== run) return;
    setStatus(e instanceof BackendError ? e.message : `Unexpected error: ${String(e)}`, 'error');
  }
}

el<HTMLButtonElement>('refresh').addEventListener('click', () => void compute('refresh'));
el<HTMLButtonElement>('settings').addEventListener('click', () => browser.runtime.openOptionsPage());
browser.runtime.onMessage.addListener((msg) => {
  if ((msg as { type?: string })?.type === 'sidekick:open') void compute('open');
});

void compute('open');
