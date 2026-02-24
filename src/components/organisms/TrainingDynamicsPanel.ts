import type { LocaleStrings } from '../../i18n/types';
import { statRowHtml } from '../molecules/StatRow';

export function trainingDynamicsPanelHtml(t: Pick<LocaleStrings, 'trainingDynamics'>): string {
  return `
    <div class="panel p-4 lg:col-span-2">
      <div class="flex items-center justify-between">
        <p class="panel-title">${t.trainingDynamics.panelTitle}</p>
        <div class="flex items-center gap-2">
          <button type="button" id="trainingDynamicsInfoBtn" class="rounded-lg border border-white/20 p-1.5 text-white/60 hover:bg-white/10 hover:text-white transition focus:outline-none focus:ring-2 focus:ring-neon/50" aria-label="${t.trainingDynamics.explainBtn}">
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          </button>
          <div id="statusPill" class="rounded-full border border-neon/40 bg-neon/10 px-3 py-1 text-xs text-neon">${t.trainingDynamics.statusIdle}</div>
        </div>
      </div>

      ${statRowHtml(t)}

      <div class="mt-4 rounded-2xl border border-white/10 bg-black/30 p-3">
        <canvas id="lossChart" class="h-56 w-full"></canvas>
        <div class="mt-2 h-2 overflow-hidden rounded-full bg-white/10"><div id="progressBar" class="meter-fill h-full w-0 transition-all duration-300"></div></div>
      </div>
    </div>
  `;
}
