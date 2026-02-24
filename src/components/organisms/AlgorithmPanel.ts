import type { LocaleStrings } from '../../i18n/types';

export function algorithmPanelHtml(
  t: Pick<LocaleStrings, 'algorithm'>,
  flowNodesHtml: string,
): string {
  return `
    <div class="panel p-4 lg:col-span-2">
      <div class="flex items-center justify-between">
        <p class="panel-title">${t.algorithm.panelTitle}</p>
        <div class="flex items-center gap-2 flex-wrap">
          <span class="text-xs text-white/50">${t.algorithm.review}</span>
          <select id="iterationSelect" class="mono rounded-lg border border-white/15 bg-black/30 px-2 py-1.5 text-xs text-white/90 focus:border-neon focus:outline-none">
            <option value="live">${t.algorithm.live}</option>
          </select>
          <span id="iterationStepLabel" class="text-xs text-white/50"></span>
        </div>
      </div>
      <div class="mt-2 flex flex-wrap items-center gap-2">
        <button type="button" id="showTransformerDiagramBtn" class="rounded-lg border border-neon/50 px-3 py-1.5 text-xs font-semibold text-neon transition hover:bg-neon/15 focus:outline-none focus:ring-2 focus:ring-neon/50">${t.algorithm.viewTransformerDiagram}</button>
      </div>
      <div id="flowGrid" class="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">${flowNodesHtml}</div>
      <div id="flowDetail" class="mt-3 rounded-lg border border-neon/25 bg-neon/10 p-3 text-sm text-neon"></div>
      <div id="flowVisual" class="mt-3 rounded-xl border border-white/10 bg-black/30 p-3"></div>
    </div>
  `;
}
