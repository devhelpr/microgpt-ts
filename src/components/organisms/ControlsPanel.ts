import type { LocaleStrings } from '../../i18n/types';

export function controlsPanelHtml(t: Pick<LocaleStrings, 'controls' | 'defaultDataset'>): string {
  return `
    <div class="panel p-4 lg:col-span-1">
      <p class="panel-title">${t.controls.panelTitle}</p>
      <div class="mt-4 flex flex-wrap gap-2">
        <button id="startBtn" class="rounded-lg bg-neon px-4 py-2 text-sm font-semibold text-black transition hover:brightness-110">${t.controls.start}</button>
        <button id="pauseBtn" class="rounded-lg border border-white/20 px-4 py-2 text-sm font-semibold text-white/90 hover:bg-white/10">${t.controls.pause}</button>
        <button id="resetBtn" class="rounded-lg border border-coral/50 px-4 py-2 text-sm font-semibold text-coral hover:bg-coral/10">${t.controls.reset}</button>
      </div>

      <label class="mt-4 block text-sm text-white/70">${t.controls.datasetLabel}</label>
      <textarea id="dataset" class="mono mt-2 h-44 w-full rounded-xl border border-white/15 bg-black/30 p-3 text-sm text-white/90 focus:border-neon focus:outline-none">${t.defaultDataset}</textarea>

      <div class="mt-4 grid grid-cols-2 gap-3">
        <label class="text-sm text-white/70">${t.controls.maxSteps}
          <input id="maxSteps" type="number" value="1200" min="50" step="50" class="mono mt-1 w-full rounded-lg border border-white/15 bg-black/30 p-2 text-sm" />
        </label>
        <label class="text-sm text-white/70">${t.controls.evalEvery}
          <input id="evalEvery" type="number" value="24" min="5" step="1" class="mono mt-1 w-full rounded-lg border border-white/15 bg-black/30 p-2 text-sm" />
        </label>
      </div>

      <div class="mt-4 space-y-2 text-xs text-white/60">
        <div id="dataStats"></div>
        <div>${t.controls.tipReset}</div>
      </div>
    </div>
  `;
}
