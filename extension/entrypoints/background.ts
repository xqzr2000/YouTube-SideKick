import { videoIdFromUrl } from '@/lib/upnext';

// Right-click menu → "SideKick's Selection" opens the side panel (so does the toolbar icon).
// Opening the panel triggers a compute; the panel page does that itself when it loads, and
// this message makes an already-open panel recompute too.
export default defineBackground(() => {
  browser.runtime.onInstalled.addListener(() => {
    browser.contextMenus.create({
      id: 'sidekick',
      title: "SideKick's Selection",
      contexts: ['all'],
    });
  });
  browser.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});

  browser.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId !== 'sidekick' || tab?.windowId == null) return;
    // must run inside the click's user gesture, so no await before it
    browser.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
    browser.runtime.sendMessage({ type: 'sidekick:open' }).catch(() => {});
  });

  // "Off a watch page, use the last watched video": remember every watch page a tab shows.
  browser.tabs.onUpdated.addListener((_tabId, change) => {
    const v = videoIdFromUrl(change.url);
    if (v) browser.storage.local.set({ lastVideo: v });
  });
});
