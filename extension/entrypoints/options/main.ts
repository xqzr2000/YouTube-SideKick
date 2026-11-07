import { Api } from '@/lib/api';
import { DEFAULT_BACKEND, normalizeBackendUrl } from '@/lib/format';

const input = document.getElementById('url') as HTMLInputElement;
const result = document.getElementById('result') as HTMLSpanElement;

browser.storage.local.get('backendUrl').then(({ backendUrl }) => {
  input.value = (backendUrl as string | undefined) || DEFAULT_BACKEND;
});

document.getElementById('save')!.addEventListener('click', async () => {
  const url = normalizeBackendUrl(input.value);
  input.value = url;
  await browser.storage.local.set({ backendUrl: url });
  result.textContent = 'Saved.';
});

document.getElementById('test')!.addEventListener('click', async () => {
  result.textContent = 'Testing…';
  try {
    const h = await new Api(input.value).health();
    result.textContent = `OK, backend ${h.version}`;
  } catch (e) {
    result.textContent = e instanceof Error ? e.message : String(e);
  }
});
