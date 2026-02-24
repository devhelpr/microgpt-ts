import type { DOMRefs } from '../types';

export function queryRefs(root: HTMLElement, missingError: string): DOMRefs {
  const get = <T extends HTMLElement>(selector: string): T => {
    const el = root.querySelector<T>(selector);
    if (!el) throw new Error(missingError);
    return el;
  };

  const trainingDynamicsInfoBtn = root.querySelector<HTMLButtonElement>('#trainingDynamicsInfoBtn');
  const dialogTrainingDynamics = root.querySelector<HTMLDialogElement>('#dialog-training-dynamics');

  return {
    dataset: get<HTMLTextAreaElement>('#dataset'),
    maxSteps: get<HTMLInputElement>('#maxSteps'),
    evalEvery: get<HTMLInputElement>('#evalEvery'),
    startBtn: get<HTMLButtonElement>('#startBtn'),
    pauseBtn: get<HTMLButtonElement>('#pauseBtn'),
    resetBtn: get<HTMLButtonElement>('#resetBtn'),
    sampleBtn: get<HTMLButtonElement>('#sampleBtn'),
    step: get<HTMLElement>('#step'),
    batchLoss: get<HTMLElement>('#batchLoss'),
    trainLoss: get<HTMLElement>('#trainLoss'),
    devLoss: get<HTMLElement>('#devLoss'),
    sample: get<HTMLElement>('#sample'),
    tokenBars: get<HTMLElement>('#tokenBars'),
    progressBar: get<HTMLElement>('#progressBar'),
    statusPill: get<HTMLElement>('#statusPill'),
    lossChart: get<HTMLCanvasElement>('#lossChart'),
    flowGrid: get<HTMLElement>('#flowGrid'),
    flowDetail: get<HTMLElement>('#flowDetail'),
    flowVisual: get<HTMLElement>('#flowVisual'),
    iterationSelect: get<HTMLSelectElement>('#iterationSelect'),
    iterationStepLabel: get<HTMLElement>('#iterationStepLabel'),
    dataStats: get<HTMLElement>('#dataStats'),
    traceContext: get<HTMLElement>('#traceContext'),
    traceTokens: get<HTMLElement>('#traceTokens'),
    traceTarget: get<HTMLElement>('#traceTarget'),
    tracePred: get<HTMLElement>('#tracePred'),
    traceLr: get<HTMLElement>('#traceLr'),
    traceGrad: get<HTMLElement>('#traceGrad'),
    breakdownTitle: get<HTMLElement>('#breakdownTitle'),
    showTransformerDiagramBtn: get<HTMLButtonElement>('#showTransformerDiagramBtn'),
    dialogTransformer: get<HTMLDialogElement>('#dialog-transformer'),
    trainingDynamicsInfoBtn,
    dialogTrainingDynamics,
    localeSelect: get<HTMLSelectElement>('#localeSelect'),
  };
}
