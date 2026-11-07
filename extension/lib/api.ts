// Client for the SideKick backend (FastAPI in WSL2 or Crostini).
import type { TabItem } from './upnext';
import { normalizeBackendUrl } from './format';

export interface Step {
  key: string;
  text: string;
  seconds: number;
}

export interface Card {
  video_id: string;
  title: string;
  channel: string;
  duration_s: number | null;
  thumbnail: string;
  rank: number;
}

export type JobStatus =
  | 'queued'
  | 'running'
  | 'done'
  | 'blocked'
  | 'error'
  | 'superseded'
  | 'empty'
  | 'none';

export interface Job {
  job_id: string | null;
  video_id: string | null;
  trigger?: string;
  status: JobStatus;
  elapsed_s?: number;
  steps: Step[];
  compute_id?: string | null;
  current_title?: string;
  mode?: string | null;
  cards: Card[];
  error?: string | null;
}

export type Trigger = 'open' | 'refresh' | 'click';

export class BackendError extends Error {}

type FetchFn = typeof fetch;

export class Api {
  readonly base: string;
  constructor(
    base: string,
    private readonly fetchFn: FetchFn = (...a) => fetch(...a),
  ) {
    this.base = normalizeBackendUrl(base);
  }

  private async call<T>(path: string, init?: RequestInit): Promise<T> {
    let r: Response;
    try {
      r = await this.fetchFn(`${this.base}${path}`, {
        ...init,
        headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
      });
    } catch {
      throw new BackendError(`Can't reach the backend at ${this.base}. Is \`just dev\` running?`);
    }
    if (!r.ok) {
      let detail = `${r.status}`;
      try {
        const j = await r.json();
        detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j);
      } catch {
        /* keep the status code */
      }
      throw new BackendError(detail);
    }
    return (await r.json()) as T;
  }

  health() {
    return this.call<{ status: string; version: string }>('/health');
  }

  startCompute(videoId: string | null, upnext: TabItem[], trigger: Trigger) {
    return this.call<Job>('/computes', {
      method: 'POST',
      body: JSON.stringify({ video_id: videoId, upnext, trigger }),
    });
  }

  job(jobId: string) {
    return this.call<Job>(`/computes/${jobId}`);
  }

  feedback(type: 'click' | 'downvote', videoId: string, computeId?: string | null, note?: string) {
    return this.call<{ ok: boolean; profile_line: string | null }>('/feedback', {
      method: 'POST',
      body: JSON.stringify({ type, video_id: videoId, compute_id: computeId ?? null, note }),
    });
  }
}

const FINAL: JobStatus[] = ['done', 'blocked', 'error', 'empty', 'none'];

/** Poll a job until it ends. A superseded job is followed to the latest one for the same panel. */
export async function waitForJob(
  api: Api,
  first: Job,
  onUpdate: (j: Job) => void,
  opts: { intervalMs?: number; sleep?: (ms: number) => Promise<void>; cancelled?: () => boolean } = {},
): Promise<Job> {
  const sleep = opts.sleep ?? ((ms) => new Promise<void>((res) => setTimeout(res, ms)));
  let j = first;
  onUpdate(j);
  while (!FINAL.includes(j.status) && j.job_id) {
    if (opts.cancelled?.()) return j;
    await sleep(opts.intervalMs ?? 1500);
    if (opts.cancelled?.()) return j;
    j = await api.job(j.job_id);
    onUpdate(j);
    if (j.status === 'superseded') return j;
  }
  return j;
}
