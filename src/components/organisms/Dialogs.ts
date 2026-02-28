import type { LocaleStrings } from '../../i18n/types';
import type { FlowStage } from '../../types';
import { stepExplainerDialogHtml } from '../molecules/ExplainerDialog';
import type { IllustrationLabels } from '../../i18n/types';

export function trainingDynamicsDialogHtml(t: Pick<LocaleStrings, 'dialogs' | 'aria'>): string {
  const d = t.dialogs.trainingDynamics;
  return `
    <dialog id="dialog-training-dynamics" class="training-dynamics-dialog rounded-2xl border border-white/15 bg-slate/98 p-0 shadow-2xl backdrop:bg-black/70" aria-labelledby="dialog-training-dynamics-title" aria-modal="true">
      <div class="max-h-[90vh] overflow-y-auto p-6">
        <div class="flex items-start justify-between gap-4">
          <h2 id="dialog-training-dynamics-title" class="text-xl font-bold text-white">${d.title}</h2>
          <button type="button" class="dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white transition" aria-label="${t.aria.close}">✕</button>
        </div>
        <div class="mt-4 space-y-4 text-base text-white/85 leading-relaxed [&_strong]:text-neon">
          <p><strong>${d.whatGraphShows}</strong><br/>${d.whatGraphShowsBody}</p>
          <p><strong>${d.spikesMean}</strong><br/>${d.spikesMeanBody}</p>
          <p><strong>${d.numbersAboveGraph}</strong><br/>${d.numbersAboveGraphBody}</p>
          <p class="text-sm text-white/55">${d.lowerLossNote}</p>
        </div>
      </div>
    </dialog>
  `;
}

export function transformerDialogHtml(t: Pick<LocaleStrings, 'dialogs' | 'aria'>): string {
  const d = t.dialogs.transformer;
  return `
    <dialog id="dialog-transformer" class="transformer-dialog rounded-2xl border border-white/15 bg-slate/98 p-0 shadow-2xl backdrop:bg-black/70" aria-labelledby="dialog-transformer-title" aria-modal="true">
      <div class="transformer-dialog-content max-h-[90vh] overflow-y-auto p-6">
        <div class="flex items-start justify-between gap-4">
          <h2 id="dialog-transformer-title" class="text-xl font-bold text-white">${d.title}</h2>
          <button type="button" class="dialog-close transformer-dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white transition" aria-label="${t.aria.close}">✕</button>
        </div>
        <p class="mt-2 text-base text-white/75 leading-relaxed">${d.description}</p>
        <p class="mt-3 text-base text-white/75 leading-relaxed">${d.intro}</p>
        <div id="transformerDiagramContainer" class="mt-6 flex justify-center overflow-auto rounded-xl border border-white/10 bg-black/30 p-6 min-h-[420px] min-w-0"></div>
        <p class="mt-4 text-sm text-white/55">${d.oneForwardPassNote}</p>
      </div>
    </dialog>
  `;
}

export function stageDialogsHtml(
  stages: FlowStage[],
  t: Pick<LocaleStrings, 'aria' | 'explainerBodies' | 'illustrationLabels'>,
  buildIllustrationSvg: (stageId: string, labels: IllustrationLabels) => string,
): string {
  return stages.map((s) => stepExplainerDialogHtml(s, t, buildIllustrationSvg)).join('');
}
