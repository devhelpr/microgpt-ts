import type { LocaleStrings } from '../i18n/types';
import type { DOMRefs } from '../types';
import type { AppState } from './state';
import { MAX_ITERATION_HISTORY } from './state';
import { drawLossChart } from '../lib/chart';
import { stageVisualHtml } from '../components/organisms/StageVisual';
import { tokenBarRow } from '../components/atoms/TokenBarRow';
import type { FlowStage } from '../types';

export function getDisplayTrace(state: AppState): import('../types').StepTrace {
  if (state.selectedIterationKey === 'live') return state.trainer.lastTrace;
  const stepNum = parseInt(state.selectedIterationKey.replace(/^step-/, ''), 10);
  const snap = state.iterationHistory.find((s) => s.step === stepNum);
  return snap ? snap.trace : state.trainer.lastTrace;
}

export function updateIterationSelectOptions(
  state: AppState,
  refs: DOMRefs,
  t: Pick<LocaleStrings, 'algorithm' | 'iterationSelect'>,
): void {
  refs.iterationSelect.innerHTML = `<option value="live">${t.algorithm.live}</option>`;
  const start = Math.max(0, state.iterationHistory.length - MAX_ITERATION_HISTORY);
  for (let i = state.iterationHistory.length - 1; i >= start; i--) {
    const s = state.iterationHistory[i];
    const opt = document.createElement('option');
    opt.value = `step-${s.step}`;
    opt.textContent = t.iterationSelect.stepLabel.replace('{n}', String(s.step));
    refs.iterationSelect.appendChild(opt);
  }
  refs.iterationSelect.value =
    state.selectedIterationKey === 'live' ||
    !state.iterationHistory.some((s) => `step-${s.step}` === state.selectedIterationKey)
      ? 'live'
      : state.selectedIterationKey;
  const snap =
    state.selectedIterationKey !== 'live'
      ? state.iterationHistory.find((s) => `step-${s.step}` === state.selectedIterationKey)
      : null;
  refs.iterationStepLabel.textContent = snap
    ? t.iterationSelect.trainDevLabel
        .replace('{trainLoss}', snap.trainLoss.toFixed(4))
        .replace('{devLoss}', snap.devLoss.toFixed(4))
    : '';
}

export function renderFlow(
  state: AppState,
  refs: DOMRefs,
  flowStages: FlowStage[],
  t: LocaleStrings,
): void {
  const activeIdx =
    state.selectedStageIndex !== null
      ? state.selectedStageIndex
      : state.running
        ? state.phaseCursor
        : flowStages.length - 1;
  const trace = getDisplayTrace(state);
  flowStages.forEach((stage, i) => {
    const el = refs.flowGrid.querySelector<HTMLElement>(`#flow-${stage.id}`);
    if (!el) return;
    el.classList.toggle('active', i === activeIdx);
    el.classList.toggle('flow-node-selected', state.selectedStageIndex === i);
  });
  const active = flowStages[activeIdx] ?? flowStages[0];
  refs.flowDetail.textContent = `${active.title}: ${active.description}`;
  refs.flowVisual.innerHTML = stageVisualHtml(active.id, trace, state.trainer, t);
}

export function render(state: AppState, refs: DOMRefs, t: LocaleStrings): void {
  const trainer = state.trainer;
  refs.step.textContent = `${trainer.step}/${trainer.maxSteps}`;
  refs.batchLoss.textContent = trainer.latestBatchLoss.toFixed(4);
  refs.trainLoss.textContent = trainer.trainLoss.toFixed(4);
  refs.devLoss.textContent = trainer.devLoss.toFixed(4);
  refs.sample.textContent =
    state.manualSample || trainer.sample || t.samplePlaceholder;

  refs.dataStats.textContent = t.dataStatsTemplate
    .replace('{words}', String(refs.dataset.value.split(/\r?\n/).filter(Boolean).length))
    .replace('{vocab}', String(trainer.vocabSize))
    .replace('{train}', String(trainer.trainSize))
    .replace('{dev}', String(trainer.devSize))
    .replace('{test}', String(trainer.testSize));

  refs.progressBar.style.width = `${Math.min(100, (trainer.step / trainer.maxSteps) * 100)}%`;
  refs.statusPill.textContent = state.running
    ? t.trainingDynamics.statusTraining
    : trainer.step >= trainer.maxSteps
      ? t.trainingDynamics.statusCompleted
      : t.trainingDynamics.statusIdle;

  const showStepDetails = !state.running;
  const breakdownPanel = refs.breakdownTitle.closest('.panel');
  const flowSection = refs.flowGrid.closest('.panel');
  const tokenBarsPanel = refs.tokenBars.closest('.panel');
  if (breakdownPanel) breakdownPanel.classList.toggle('hidden', !showStepDetails);
  if (flowSection) flowSection.querySelector('#flowDetail')?.classList.toggle('hidden', !showStepDetails);
  if (flowSection) flowSection.querySelector('#flowVisual')?.classList.toggle('hidden', !showStepDetails);
  if (tokenBarsPanel) tokenBarsPanel.classList.toggle('hidden', !showStepDetails);

  if (showStepDetails) {
    refs.breakdownTitle.textContent =
      state.selectedIterationKey === 'live'
        ? t.breakdown.panelTitle
        : t.breakdown.stepBreakdown.replace('{n}', state.selectedIterationKey.replace(/^step-/, ''));

    const displayTrace = getDisplayTrace(state);
    const pairedContext = displayTrace.context.map((id, i) => ({
      id,
      token: displayTrace.contextTokens[i],
    }));
    const visibleContext = pairedContext.filter((p) => p.token !== 'BOS');
    const visibleIds = visibleContext.map((p) => p.id);
    const visibleTokens = visibleContext.map((p) => p.token);

    // Human‑readable context text (just the characters, no BOS)
    const contextText = visibleTokens.join('');
    refs.traceContextText.textContent = contextText || t.breakdown.noContextText;

    // Numeric IDs and token symbols, excluding BOS for clarity
    refs.traceContext.textContent = `[${visibleIds.join(', ')}]`;
    refs.traceTokens.textContent = `[${visibleTokens.join(', ')}]`;
    refs.traceTarget.textContent = `${displayTrace.targetToken} (${displayTrace.targetIndex})`;
    refs.tracePred.textContent = displayTrace.predictedToken;
    refs.traceLr.textContent = displayTrace.lr.toFixed(4);
    refs.traceGrad.textContent = displayTrace.gradNorm.toFixed(6);

    refs.tokenBars.innerHTML = displayTrace.top
      .map((row) => {
        const width = Math.max(6, Math.round(row.prob * 100));
        return tokenBarRow(row.token, row.prob, width);
      })
      .join('');

    const flowStages = t.flowStages as FlowStage[];
    renderFlow(state, refs, flowStages, t);
  }

  drawLossChart(refs.lossChart, trainer.losses.slice(-300), t);
}
