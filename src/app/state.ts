import type { createMicroGptTrainer } from '../../microgpt';
import type { IterationSnapshot } from '../types';

export const MAX_ITERATION_HISTORY = 50;

export type MicroGptTrainer = ReturnType<typeof createMicroGptTrainer>;

/** Mutable app state. Trainer is replaced on reset. */
export interface AppState {
  trainer: MicroGptTrainer;
  manualSample: string;
  running: boolean;
  phaseCursor: number;
  iterationHistory: IterationSnapshot[];
  selectedStageIndex: number | null;
  selectedIterationKey: 'live' | string;
}

export function createInitialState(trainer: MicroGptTrainer): AppState {
  return {
    trainer,
    manualSample: '',
    running: false,
    phaseCursor: 0,
    iterationHistory: [],
    selectedStageIndex: null,
    selectedIterationKey: 'live',
  };
}
