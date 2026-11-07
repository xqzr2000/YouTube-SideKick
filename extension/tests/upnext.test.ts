import { describe, expect, it } from 'vitest';
import { extractUpnext, videoIdFromUrl } from '../lib/upnext';

// Synthetic markup in the two shapes YouTube has used for up-next (no real page content).
const LOCKUP = `
<div id="secondary"><ytd-watch-next-secondary-results-renderer>
  <yt-lockup-view-model>
    <a href="/watch?v=AAAAAAAAAA1"><div><badge-shape><div>1:12:05</div></badge-shape></div></a>
    <h3 title="18. Bayesian Statistics (cont.)"><a href="/watch?v=AAAAAAAAAA1"><span>18. Bayesian Statistics (cont.)</span></a></h3>
    <div><span>MIT OpenCourseWare</span></div>
    <div><span>120K views</span><span> • </span><span>9 years ago</span></div>
  </yt-lockup-view-model>
  <yt-lockup-view-model>
    <a href="/watch?v=AAAAAAAAAA2"><div><badge-shape><div>LIVE</div></badge-shape></div></a>
    <h3 title="Live now"><a href="/watch?v=AAAAAAAAAA2">Live now</a></h3>
    <div><span>Some Channel</span></div><div><span>1.2K watching</span></div>
  </yt-lockup-view-model>
  <yt-lockup-view-model>
    <a href="/watch?v=AAAAAAAAAA3&list=RDAAAAAAAAAA3"><h3 title="Mix - Jazz">Mix - Jazz</h3></a>
  </yt-lockup-view-model>
  <ytd-reel-shelf-renderer><a href="/watch?v=AAAAAAAAAA4">a short</a></ytd-reel-shelf-renderer>
  <a href="/shorts/AAAAAAAAAA5">another short</a>
  <yt-lockup-view-model>
    <a href="/watch?v=CURRENT0001">the video being watched</a>
  </yt-lockup-view-model>
  <yt-lockup-view-model>
    <a href="/watch?v=AAAAAAAAAA6"><span>12:34</span></a>
    <h3 title="Credible intervals"><a href="/watch?v=AAAAAAAAAA6">Credible intervals</a></h3>
    <div><span>Stats Hub • 3.4M views • 2 months ago</span></div>
  </yt-lockup-view-model>
  <yt-lockup-view-model><a href="/watch?v=AAAAAAAAAA1">duplicate</a></yt-lockup-view-model>
</ytd-watch-next-secondary-results-renderer></div>
<div id="primary"><a href="/watch?v=AAAAAAAAAA7">a link outside the up-next column</a></div>`;

const COMPACT = `
<div id="related">
  <ytd-compact-video-renderer>
    <a id="thumbnail" href="/watch?v=BBBBBBBBBB1"><ytd-thumbnail-overlay-time-status-renderer><span id="text"> 9:05 </span></ytd-thumbnail-overlay-time-status-renderer></a>
    <a href="/watch?v=BBBBBBBBBB1"><span id="video-title" title="Double DQN explained">Double DQN explained</span></a>
    <ytd-channel-name><yt-formatted-string id="text">RL Lab</yt-formatted-string></ytd-channel-name>
    <div id="metadata-line"><span>45K views</span><span>Streamed 1 year ago</span></div>
  </ytd-compact-video-renderer>
</div>`;

function doc(html: string): Document {
  document.body.innerHTML = html;
  return document;
}

describe('extractUpnext', () => {
  it('reads the lockup layout in page order', () => {
    const items = extractUpnext(doc(LOCKUP), 'CURRENT0001');
    expect(items.map((i) => i.video_id)).toEqual(['AAAAAAAAAA1', 'AAAAAAAAAA2', 'AAAAAAAAAA6']);
    expect(items[0]).toEqual({
      video_id: 'AAAAAAAAAA1',
      title: '18. Bayesian Statistics (cont.)',
      channel: 'MIT OpenCourseWare',
      duration: '1:12:05',
      views: '120K views',
      age: '9 years ago',
      badges: [],
    });
    expect(items[1]!.badges).toEqual(['LIVE']);
    expect(items[1]!.views).toBe('1.2K watching');
    expect(items[2]).toMatchObject({ channel: 'Stats Hub', views: '3.4M views', age: '2 months ago', duration: '12:34' });
  });

  it('reads the older compact layout', () => {
    const [item] = extractUpnext(doc(COMPACT), null);
    expect(item).toMatchObject({
      video_id: 'BBBBBBBBBB1',
      title: 'Double DQN explained',
      channel: 'RL Lab',
      duration: '9:05',
      views: '45K views',
      age: 'Streamed 1 year ago',
    });
  });

  it('stops at max', () => {
    expect(extractUpnext(doc(LOCKUP), null, 2)).toHaveLength(2);
  });
});

describe('videoIdFromUrl', () => {
  it.each([
    ['https://www.youtube.com/watch?v=bFZ-0FH5hfs&t=10s', 'bFZ-0FH5hfs'],
    ['/watch?v=bFZ-0FH5hfs', 'bFZ-0FH5hfs'],
    ['https://www.youtube.com/shorts/bFZ-0FH5hfs', null],
    ['https://www.youtube.com/', null],
    ['https://example.com/watch?v=bFZ-0FH5hfs', null],
    ['https://www.youtube.com/watch?v=short', null],
    [undefined, null],
  ])('%s → %s', (url, id) => {
    expect(videoIdFromUrl(url)).toBe(id);
  });
});
