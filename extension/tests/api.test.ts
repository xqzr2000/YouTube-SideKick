import { describe, expect, it, vi } from 'vitest';
import { Api, BackendError, type Job, waitForJob } from '../lib/api';
import { formatDuration, normalizeBackendUrl } from '../lib/format';

const job = (status: Job['status'], extra: Partial<Job> = {}): Job => ({
  job_id: 'j1',
  video_id: 'bFZ-0FH5hfs',
  status,
  steps: [],
  cards: [],
  ...extra,
});

function fakeFetch(responses: Array<{ status?: number; body: unknown } | Error>) {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const r = responses.shift();
    if (!r) throw new Error('no more responses');
    if (r instanceof Error) throw r;
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  });
  return { fn: fn as unknown as typeof fetch, calls };
}

describe('Api', () => {
  it('posts a compute with the tab list and trigger', async () => {
    const f = fakeFetch([{ body: job('running') }]);
    const api = new Api('http://localhost:8000/', f.fn);
    const up = [{ video_id: 'AAAAAAAAAA1', title: 't', channel: 'c', duration: '1:00', views: '', age: '', badges: [] }];
    const j = await api.startCompute('bFZ-0FH5hfs', up, 'open');
    expect(j.status).toBe('running');
    expect(f.calls[0]!.url).toBe('http://localhost:8000/computes');
    expect(JSON.parse(String(f.calls[0]!.init?.body))).toEqual({ video_id: 'bFZ-0FH5hfs', upnext: up, trigger: 'open' });
  });

  it('explains an unreachable backend and passes on API errors', async () => {
    const down = new Api('http://localhost:8000', fakeFetch([new TypeError('Failed to fetch')]).fn);
    await expect(down.health()).rejects.toThrow(/just dev/);
    const bad = new Api('http://localhost:8000', fakeFetch([{ status: 422, body: { detail: 'a 👎 needs a note' } }]).fn);
    await expect(bad.feedback('downvote', 'AAAAAAAAAA1', null, '')).rejects.toBeInstanceOf(BackendError);
  });
});

describe('waitForJob', () => {
  const noSleep = async () => {};

  it('polls until done and reports every update', async () => {
    const f = fakeFetch([{ body: job('running', { steps: [{ key: 'r0', text: 'Up-next', seconds: 3 }] }) }, { body: job('done') }]);
    const seen: string[] = [];
    const end = await waitForJob(new Api('http://x', f.fn), job('queued'), (j) => seen.push(j.status), { sleep: noSleep });
    expect(end.status).toBe('done');
    expect(seen).toEqual(['queued', 'running', 'done']);
  });

  it('stops on superseded or when cancelled', async () => {
    const f = fakeFetch([{ body: job('superseded') }]);
    expect((await waitForJob(new Api('http://x', f.fn), job('queued'), () => {}, { sleep: noSleep })).status).toBe('superseded');
    let n = 0;
    const g = fakeFetch([{ body: job('running') }, { body: job('running') }]);
    const end = await waitForJob(new Api('http://x', g.fn), job('running'), () => n++, { sleep: noSleep, cancelled: () => n >= 2 });
    expect(end.status).toBe('running');
  });

  it('returns at once for an empty panel', async () => {
    const end = await waitForJob(new Api('http://x', fakeFetch([]).fn), job('empty', { job_id: null }), () => {});
    expect(end.status).toBe('empty');
  });
});

describe('format', () => {
  it('formats durations and URLs', () => {
    expect(formatDuration(4325)).toBe('1:12:05');
    expect(formatDuration(754)).toBe('12:34');
    expect(formatDuration(null)).toBe('');
    expect(normalizeBackendUrl(' http://penguin.linux.test:8000/ ')).toBe('http://penguin.linux.test:8000');
    expect(normalizeBackendUrl('')).toBe('http://localhost:8000');
  });
});
