import type { FlowStage } from '../../types';
import type { LocaleStrings } from '../../i18n/types';
import type { IllustrationLabels } from '../../i18n/types';

export function stepExplainerDialogHtml(
  stage: FlowStage,
  t: Pick<LocaleStrings, 'aria' | 'explainerBodies' | 'illustrationLabels'>,
  buildIllustrationSvg: (stageId: string, labels: IllustrationLabels) => string,
): string {
  const illo = buildIllustrationSvg(stage.id, t.illustrationLabels);
  const body = t.explainerBodies[stage.id] ?? stage.description;
  return `<dialog id="dialog-${stage.id}" class="explainer-dialog rounded-2xl border border-white/15 bg-slate/95 p-0 shadow-2xl backdrop:bg-black/60" aria-labelledby="dialog-title-${stage.id}">
  <div class="explainer-dialog-content max-h-[85vh] overflow-y-auto p-6 text-center">
    <div class="flex items-center justify-center gap-4 relative">
      <h2 id="dialog-title-${stage.id}" class="text-xl font-bold text-white text-center">${stage.title}</h2>
      <button type="button" class="dialog-close absolute right-0 top-0 rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white" aria-label="${t.aria.close}">✕</button>
    </div>
    <div class="mt-4 text-base text-white/85 leading-relaxed text-center [&_code]:rounded [&_code]:bg-black/30 [&_code]:px-1 [&_code]:font-mono [&_code]:text-neon">${body}</div>
    ${illo ? `<div class="mt-6 flex justify-center explainer-illo-wrapper">${illo}</div>` : ''}
    <p class="mt-4 text-sm text-white/50 text-center">${stage.description}</p>
  </div>
</dialog>`;
}
