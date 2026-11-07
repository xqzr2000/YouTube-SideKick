// Reads the up-next list from a YouTube watch page (signed in, as the user sees it).
//
// YouTube changes its markup often, so this avoids exact class names where it can: it walks the
// links to /watch?v=… inside the right-hand column, takes the nearest item container, and
// classifies that item's visible text pieces by shape ("12:34" is a duration, "… views" is a
// view count, "… ago" is an age). The backend parses the numbers; this only collects text.

export interface TabItem {
  video_id: string;
  title: string;
  channel: string;
  duration: string;
  views: string;
  age: string;
  badges: string[];
}

const ID_RE = /^[A-Za-z0-9_-]{11}$/;
const ITEM_SELECTOR = [
  'yt-lockup-view-model',
  'ytd-compact-video-renderer',
  'ytd-compact-radio-renderer',
  'ytd-compact-playlist-renderer',
  'ytd-rich-item-renderer',
].join(', ');
const SCOPE_SELECTOR = 'ytd-watch-next-secondary-results-renderer, #secondary, #related';
const DURATION_RE = /^\d{1,2}(:\d{2}){1,2}$/;
const VIEWS_RE = /(views?|watching)$/i;
const AGE_RE = /\bago$/i;
const BADGES = ['LIVE', 'SHORTS', 'PREMIERE', 'UPCOMING', 'NEW', '4K', 'CC'];

/** The video id of a watch URL (``/watch?v=…``), or null. */
export function videoIdFromUrl(url: string | undefined | null): string | null {
  if (!url) return null;
  try {
    const u = new URL(url, 'https://www.youtube.com');
    if (!/(^|\.)youtube\.com$/.test(u.hostname) || u.pathname !== '/watch') return null;
    const v = u.searchParams.get('v');
    return v && ID_RE.test(v) ? v : null;
  } catch {
    return null;
  }
}

function texts(el: Element): string[] {
  const out: string[] = [];
  const walker = el.ownerDocument.createTreeWalker(el, 4 /* NodeFilter.SHOW_TEXT */);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const raw = (n.textContent ?? '').replace(/\s+/g, ' ').trim();
    for (const t of raw.split(/\s*[•·]\s*/)) {
      if (t && out[out.length - 1] !== t) out.push(t);
    }
  }
  return out;
}

function titleOf(item: Element, link: HTMLAnchorElement): string {
  const el = item.querySelector('#video-title, h3, [title]');
  const t = el?.getAttribute('title') || el?.textContent || link.getAttribute('title') || '';
  return t.replace(/\s+/g, ' ').trim();
}

/** Up-next videos in page order (no Shorts, no playlists or mixes, no duplicates). */
export function extractUpnext(root: ParentNode, currentId: string | null, max = 40): TabItem[] {
  const scope = root.querySelector(SCOPE_SELECTOR) ?? root;
  const seen = new Set<string>(currentId ? [currentId] : []);
  const out: TabItem[] = [];
  for (const link of Array.from(scope.querySelectorAll<HTMLAnchorElement>('a[href*="/watch?v="]'))) {
    const href = link.getAttribute('href') ?? '';
    if (/[?&]list=/.test(href)) continue; // a playlist or mix, not a single video
    const id = videoIdFromUrl(href);
    if (!id || seen.has(id)) continue;
    const item = link.closest(ITEM_SELECTOR) ?? link.parentElement;
    if (!item || item.closest('ytd-reel-shelf-renderer, ytm-shorts-lockup-view-model')) continue;
    seen.add(id);
    const title = titleOf(item, link);
    const rec: TabItem = { video_id: id, title, channel: '', duration: '', views: '', age: '', badges: [] };
    for (const t of texts(item)) {
      if (t === title) continue;
      if (!rec.duration && DURATION_RE.test(t)) rec.duration = t;
      else if (BADGES.includes(t.toUpperCase())) rec.badges.push(t);
      else if (!rec.views && VIEWS_RE.test(t)) rec.views = t;
      else if (!rec.age && (AGE_RE.test(t) || /^(streamed|premiere)/i.test(t))) rec.age = t;
      else if (!rec.channel && !/^\d/.test(t)) rec.channel = t;
    }
    out.push(rec);
    if (out.length >= max) break;
  }
  return out;
}
