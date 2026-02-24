import type { LocaleStrings } from '../../i18n/types';

export function generatedNameSectionHtml(t: Pick<LocaleStrings, 'generatedName'>): string {
  return `
    <section class="mb-4">
      <div class="panel p-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="panel-title mb-0">${t.generatedName.panelTitle}</p>
          <button id="sampleBtn" class="rounded-lg border border-butter/50 px-4 py-2 text-sm font-semibold text-butter hover:bg-butter/10">${t.generatedName.generate}</button>
        </div>
        <pre id="sample" class="mono mt-3 min-h-14 whitespace-pre-wrap rounded-xl border border-white/10 bg-black/25 p-3 text-lg text-neon"></pre>
      </div>
    </section>
  `;
}
