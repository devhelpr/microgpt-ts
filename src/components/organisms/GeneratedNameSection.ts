import type { LocaleStrings } from '../../i18n/types';

export function generatedNameSectionHtml(t: Pick<LocaleStrings, 'generatedName'>): string {
  return `
    <section class="mb-4">
      <div class="panel p-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="panel-title mb-0">${t.generatedName.panelTitle}</p>
          <div class="flex items-center gap-2">
            <button type="button" id="generateInfoBtn" class="rounded-lg border border-white/20 p-1.5 text-white/60 hover:bg-white/10 hover:text-white transition focus:outline-none focus:ring-2 focus:ring-neon/50" aria-label="${t.generatedName.infoBtn}">
              <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </button>
            <button id="sampleBtn" class="rounded-lg border border-butter/50 px-4 py-2 text-sm font-semibold text-butter hover:bg-butter/10">${t.generatedName.generate}</button>
          </div>
        </div>
        <pre id="sample" class="mono mt-3 min-h-14 whitespace-pre-wrap rounded-xl border border-white/10 bg-black/25 p-3 text-lg text-neon"></pre>
      </div>
    </section>
  `;
}
