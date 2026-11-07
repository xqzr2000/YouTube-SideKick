/** 4325 → "1:12:05", 754 → "12:34", null → "". */
export function formatDuration(s: number | null | undefined): string {
  if (s == null || s < 0) return '';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const pad = (n: number) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** Strip a trailing slash so paths can be appended. */
export function normalizeBackendUrl(url: string): string {
  return url.trim().replace(/\/+$/, '') || DEFAULT_BACKEND;
}

export const DEFAULT_BACKEND = 'http://localhost:8000';
