import type { LocaleCode } from '../i18n';
import { setLocale } from '../i18n';
import type { LocaleStrings } from '../i18n/types';
import type { DOMRefs } from '../types';
import type { AppState } from './state';
import type { FlowStage } from '../types';
import { render } from './render';

export function registerEvents(
  app: HTMLElement,
  state: AppState,
  refs: DOMRefs,
  t: LocaleStrings,
  flowStages: FlowStage[],
  onTrainLoop: () => void,
  onReset: () => void,
  renderDiagram: (container: HTMLElement) => void,
  destroyDiagram: (container: HTMLElement) => void,
  diagramErrorMsg: string,
): void {
  refs.startBtn.addEventListener('click', () => {
    // If the user has changed maxSteps/evalEvery before the first run,
    // recreate the trainer with the new settings on first start.
    if (!state.running && state.trainer.step === 0) {
      const desiredMaxSteps = Math.max(50, Number(refs.maxSteps.value) || 1200);
      const desiredEvalEvery = Math.max(5, Number(refs.evalEvery.value) || 24);
      if (
        state.trainer.maxSteps !== desiredMaxSteps ||
        state.trainer.evalEvery !== desiredEvalEvery
      ) {
        onReset();
      }
    }
    void onTrainLoop();
  });

  refs.pauseBtn.addEventListener('click', () => {
    state.running = false;
    render(state, refs, t);
  });

  refs.resetBtn.addEventListener('click', () => onReset());

  refs.sampleBtn.addEventListener('click', () => {
    state.manualSample = state.trainer.generate(42);
    render(state, refs, t);
  });

  refs.localeSelect.addEventListener('change', (e) => {
    const select = e.target as HTMLSelectElement;
    const code = select.value as LocaleCode;
    if (code !== 'en' && code !== 'nl') return;
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('locale', code);
      setLocale(code);
      window.location.reload();
    }
  });

  refs.trainingDynamicsInfoBtn?.addEventListener('click', () => {
    refs.dialogTrainingDynamics?.showModal();
  });

  refs.showTransformerDiagramBtn.addEventListener('click', () => {
    const container = document.getElementById('transformerDiagramContainer');
    if (!container) return;
    container.innerHTML = '';
    refs.dialogTransformer.showModal();
    try {
      renderDiagram(container);
    } catch {
      container.innerHTML = `<p class="text-sm text-coral">${diagramErrorMsg}</p>`;
    }
  });

  refs.dialogTransformer.addEventListener('close', () => {
    const container = document.getElementById('transformerDiagramContainer');
    if (container) destroyDiagram(container);
  });

  refs.iterationSelect.addEventListener('change', () => {
    state.selectedIterationKey = refs.iterationSelect.value;
    const snap =
      state.selectedIterationKey !== 'live'
        ? state.iterationHistory.find((s) => `step-${s.step}` === state.selectedIterationKey)
        : null;
    refs.iterationStepLabel.textContent = snap
      ? t.iterationSelect.trainDevLabel
          .replace('{trainLoss}', snap.trainLoss.toFixed(4))
          .replace('{devLoss}', snap.devLoss.toFixed(4))
      : '';
    render(state, refs, t);
  });

  refs.flowGrid.addEventListener('click', (e) => {
    const infoBtn = (e.target as HTMLElement).closest('.flow-info-btn');
    if (infoBtn) {
      e.preventDefault();
      e.stopPropagation();
      const idx = parseInt((infoBtn as HTMLElement).dataset.stageIndex ?? '-1', 10);
      if (idx >= 0 && idx < flowStages.length) {
        const stage = flowStages[idx];
        const dialog = app.querySelector<HTMLDialogElement>(`#dialog-${stage.id}`);
        if (dialog) dialog.showModal();
      }
      return;
    }
    const target = (e.target as HTMLElement).closest('[data-stage-index]');
    if (!target) return;
    const idx = parseInt((target as HTMLElement).dataset.stageIndex ?? '-1', 10);
    if (idx >= 0 && idx < flowStages.length) {
      state.selectedStageIndex = state.selectedStageIndex === idx ? null : idx;
      render(state, refs, t);
    }
  });

  refs.flowGrid.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const target = (e.target as HTMLElement).closest('[data-stage-index]');
    if (!target) return;
    e.preventDefault();
    const idx = parseInt((target as HTMLElement).dataset.stageIndex ?? '-1', 10);
    if (idx >= 0 && idx < flowStages.length) {
      state.selectedStageIndex = state.selectedStageIndex === idx ? null : idx;
      render(state, refs, t);
    }
  });

  app.addEventListener('click', (e) => {
    const closeBtn = (e.target as HTMLElement).closest('.dialog-close');
    if (closeBtn) {
      const dialog = closeBtn.closest('dialog');
      if (dialog instanceof HTMLDialogElement) dialog.close();
    }
  });
  app.addEventListener(
    'cancel',
    (e) => {
      if ((e.target as HTMLDialogElement)?.tagName === 'DIALOG') (e.target as HTMLDialogElement).close();
    },
    true,
  );

  if (typeof window !== 'undefined') {
    window.addEventListener('resize', () => render(state, refs, t));
  }
}
