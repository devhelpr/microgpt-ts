import type { StepTrace } from '../../microgpt';
import type { LocaleStrings } from '../i18n/types';

export type { StepTrace };

/** Re-export flow stage shape from locale. */
export type FlowStage = LocaleStrings['flowStages'][number];

export type IterationSnapshot = {
  step: number;
  trace: StepTrace;
  trainLoss: number;
  devLoss: number;
  batchLoss: number;
};

/** All DOM elements required by the app. Used for dependency injection and tests. */
export interface DOMRefs {
  dataset: HTMLTextAreaElement;
  maxSteps: HTMLInputElement;
  evalEvery: HTMLInputElement;
  startBtn: HTMLButtonElement;
  pauseBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  sampleBtn: HTMLButtonElement;
  step: HTMLElement;
  batchLoss: HTMLElement;
  trainLoss: HTMLElement;
  devLoss: HTMLElement;
  sample: HTMLElement;
  tokenBars: HTMLElement;
  progressBar: HTMLElement;
  statusPill: HTMLElement;
  lossChart: HTMLCanvasElement;
  flowGrid: HTMLElement;
  flowDetail: HTMLElement;
  flowVisual: HTMLElement;
  iterationSelect: HTMLSelectElement;
  iterationStepLabel: HTMLElement;
  dataStats: HTMLElement;
  traceContextText: HTMLElement;
  traceContext: HTMLElement;
  traceTokens: HTMLElement;
  traceTarget: HTMLElement;
  tracePred: HTMLElement;
  traceLr: HTMLElement;
  traceGrad: HTMLElement;
  breakdownTitle: HTMLElement;
  showTransformerDiagramBtn: HTMLButtonElement;
  dialogTransformer: HTMLDialogElement;
  trainingDynamicsInfoBtn: HTMLButtonElement | null;
  dialogTrainingDynamics: HTMLDialogElement | null;
  localeSelect: HTMLSelectElement;
}

/** Trainer shape needed for stage visual (dataset split). */
export interface TrainerSizes {
  trainSize: number;
  devSize: number;
  testSize: number;
}
