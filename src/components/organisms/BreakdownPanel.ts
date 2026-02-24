import type { LocaleStrings } from '../../i18n/types';

export function breakdownPanelHtml(t: Pick<LocaleStrings, 'breakdown'>): string {
  return `
    <div class="panel p-4">
      <p class="panel-title" id="breakdownTitle">${t.breakdown.panelTitle}</p>
      <div class="mt-3 space-y-2 text-sm">
        <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.contextIds}</span> <span id="traceContext" class="mono text-white/90"></span></div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.contextTokens}</span> <span id="traceTokens" class="mono text-white/90"></span></div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.target}</span> <span id="traceTarget" class="mono text-butter"></span></div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.predicted}</span> <span id="tracePred" class="mono text-neon"></span></div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.learningRate}</span> <span id="traceLr" class="mono text-white/90"></span></div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.gradientNorm}</span> <span id="traceGrad" class="mono text-coral"></span></div>
      </div>
    </div>
  `;
}
