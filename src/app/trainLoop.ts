import type { LocaleStrings } from '../i18n/types';
import type { DOMRefs } from '../types';
import type { AppState } from './state';
import { MAX_ITERATION_HISTORY } from './state';
import { cloneTrace } from '../lib/cloneTrace';
import { updateIterationSelectOptions, render } from './render';

export async function trainLoop(
  state: AppState,
  refs: DOMRefs,
  t: LocaleStrings,
): Promise<void> {
  if (state.running) return;
  state.running = true;
  render(state, refs, t);

  while (state.running && state.trainer.step < state.trainer.maxSteps) {
    state.trainer.trainStep();
    state.iterationHistory.push({
      step: state.trainer.step,
      trace: cloneTrace(state.trainer.lastTrace),
      trainLoss: state.trainer.trainLoss,
      devLoss: state.trainer.devLoss,
      batchLoss: state.trainer.latestBatchLoss,
    });
    if (state.iterationHistory.length > MAX_ITERATION_HISTORY) state.iterationHistory.shift();
    updateIterationSelectOptions(state, refs, t);
    render(state, refs, t);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  state.running = false;
  render(state, refs, t);
}

export function resetTrainer(
  state: AppState,
  refs: DOMRefs,
  t: LocaleStrings,
  createTrainer: (datasetText: string, config: { blockSize: number; nEmbd: number; maxSteps: number; evalEvery: number; seed: number }) => AppState['trainer'],
): void {
  state.running = false;
  state.phaseCursor = 0;
  state.manualSample = '';
  state.iterationHistory = [];
  state.selectedIterationKey = 'live';
  state.selectedStageIndex = null;
  state.trainer = createTrainer(refs.dataset.value, {
    blockSize: 16,
    nEmbd: 16,
    maxSteps: Math.max(50, Number(refs.maxSteps.value) || 1200),
    evalEvery: Math.max(5, Number(refs.evalEvery.value) || 24),
    seed: 1337,
  });
  updateIterationSelectOptions(state, refs, t);
  render(state, refs, t);
}
