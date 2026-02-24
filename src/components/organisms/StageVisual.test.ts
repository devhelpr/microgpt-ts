import { describe, it, expect } from 'vitest';
import { stageVisualHtml } from './StageVisual';
import type { StepTrace } from '../../types';
import type { TrainerSizes } from '../../types';

const mockTrace: StepTrace = {
  context: [1, 2],
  contextTokens: ['a', 'b'],
  targetIndex: 2,
  targetToken: 'c',
  predictedToken: 'c',
  loss: 0.5,
  lr: 0.01,
  gradNorm: 0.1,
  top: [{ token: 'c', prob: 0.9 }],
  tokenEmbedding: [0.1, 0.2],
  positionEmbedding: [0.3, 0.4],
  summedEmbedding: [0.5, 0.6],
  mlpOut: [],
  lnOut: [0.7, 0.8],
  logits: [1, 2, 3],
  targetProb: 0.9,
};

const mockTrainer: TrainerSizes = {
  trainSize: 80,
  devSize: 10,
  testSize: 10,
};

const mockT = {
  stageVisual: {
    dataset: {
      splitDescription: 'Split description',
      train: 'Train',
      dev: 'Dev',
      test: 'Test',
    },
    encode: {
      description: 'Encode description',
      contextTokensToIds: 'Tokens to IDs',
      target: 'Target',
    },
    context: { description: 'Context description' },
    forward: {
      attentionHead0: 'Attention head 0',
      tokenEmbedding: 'Token emb',
      positionEmbedding: 'Pos emb',
      sumEmbedding: 'Sum emb',
      preHeadAfterMlp: 'Pre-head',
      preHeadBeforeLmHead: 'Pre-head before LM',
    },
    softmax: { description: 'Softmax description' },
    loss: {
      description: 'Loss description',
      predictedTarget: 'Predicted {pred}, target {target}',
    },
    backprop: {
      description: 'Backprop description',
      gradientNorm: 'Gradient norm',
    },
    update: {
      description: 'Update description',
      formula: 'θ = θ - lr * ∇L',
      lrStepMagnitude: 'lr={lr} | avg step magnitude≈{delta}',
    },
  },
  vectorBars: { first8Dims: '(first 8)' },
};

describe('stageVisualHtml', () => {
  it('encode stage includes context tokens and target', () => {
    const html = stageVisualHtml('encode', mockTrace, mockTrainer, mockT);
    expect(html).toContain('a');
    expect(html).toContain('b');
    expect(html).toContain('c');
    expect(html).toContain('Target');
    expect(html).toContain('Encode description');
  });

  it('loss stage includes loss value and predictedTarget text', () => {
    const html = stageVisualHtml('loss', mockTrace, mockTrainer, mockT);
    expect(html).toContain('0.5000');
    expect(html).toContain('Predicted c, target c');
    expect(html).toContain('Loss description');
  });

  it('dataset stage includes train/dev/test sizes and split description', () => {
    const html = stageVisualHtml('dataset', mockTrace, mockTrainer, mockT);
    expect(html).toContain('80');
    expect(html).toContain('10');
    expect(html).toContain('Split description');
    expect(html).toContain('Train');
    expect(html).toContain('Dev');
    expect(html).toContain('Test');
  });

  it('context stage includes tokens and target', () => {
    const html = stageVisualHtml('context', mockTrace, mockTrainer, mockT);
    expect(html).toContain('Context description');
    expect(html).toContain('c');
  });
});
