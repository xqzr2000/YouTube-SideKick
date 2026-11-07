import { extractUpnext, videoIdFromUrl } from '@/lib/upnext';

// Runs on youtube.com and answers the side panel: which video is open, and its up-next list
// exactly as the signed-in user sees it. Nothing is sent anywhere from here.
export default defineContentScript({
  matches: ['https://www.youtube.com/*'],
  main() {
    browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      if ((msg as { type?: string })?.type !== 'sidekick:read-page') return;
      const videoId = videoIdFromUrl(location.href);
      sendResponse({ videoId, upnext: videoId ? extractUpnext(document, videoId) : [] });
    });
  },
});
