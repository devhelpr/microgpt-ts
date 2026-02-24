import type { FlowStage } from '../../types';
import type { LocaleStrings } from '../../i18n/types';

export function flowNodeHtml(stage: FlowStage, index: number, t: Pick<LocaleStrings, 'aria'>): string {
  return `
    <article id="flow-${stage.id}" class="flow-node relative cursor-pointer rounded-xl border border-white/10 bg-black/25 p-3 pr-9 transition hover:border-white/25 hover:bg-black/40" data-stage-index="${index}" role="button" tabindex="0">
      <button type="button" class="flow-info-btn absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full border border-white/20 text-white/60 transition hover:border-neon/50 hover:bg-neon/15 hover:text-neon focus:outline-none focus:ring-2 focus:ring-neon/50" data-stage-index="${index}" aria-label="${t.aria.learnMoreAbout} ${stage.title}">
        <svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
      </button>
      <p class="mono text-xs text-white/45">${String(index + 1).padStart(2, '0')}</p>
      <h4 class="mt-1 text-sm font-semibold text-white">${stage.title}</h4>
      <p class="mt-1 text-xs text-white/65">${stage.description}</p>
    </article>
  `;
}
