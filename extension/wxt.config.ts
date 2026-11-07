import { defineConfig } from 'wxt';

// One MV3 build for Chrome on Windows 11 and ChromeOS; only the backend URL differs (Settings).
export default defineConfig({
  manifest: {
    name: 'YouTube SideKick',
    description: "SideKick's Selection: 10 cleaner next videos for the YouTube tab, from your own backend.",
    permissions: ['sidePanel', 'contextMenus', 'storage', 'tabs'],
    host_permissions: [
      'https://www.youtube.com/*',
      'http://localhost/*',
      'http://127.0.0.1/*',
      'http://penguin.linux.test/*',
    ],
    action: { default_title: "SideKick's Selection" },
  },
});
