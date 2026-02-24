import { describe, it, expect } from 'vitest';
import { cloneTrace } from './cloneTrace';
import type { StepTrace } from '../../microgpt';

function makeTrace(overrides: Partial<StepTrace> = {}): StepTrace {
  return {
    context: [1, 2, 3],
    contextTokens: ['a', 'b', 'c'],
    targetIndex: 2,
    targetToken: 'c',
    predictedToken: 'c',
    loss: 0.5,
    lr: 0.01,
    gradNorm: 0.1,
    top: [{ token: 'c', prob: 0.8 }],
    tokenEmbedding: [0.1, 0.2],
    positionEmbedding: [0.3, 0.4],
    summedEmbedding: [0.5, 0.6],
    mlpOut: [],
    lnOut: [0.7, 0.8],
    logits: [1, 2, 3],
    targetProb: 0.8,
    attentionWeights: [0.5, 0.5],
    ...overrides,
  };
}

describe('cloneTrace', () => {
  it('returns a deep copy with same values', () => {
    const tr = makeTrace();
    const cloned = cloneTrace(tr);
    expect(cloned).not.toBe(tr);
    expect(cloned.context).toEqual(tr.context);
    expect(cloned.context).not.toBe(tr.context);
    expect(cloned.contextTokens).not.toBe(tr.contextTokens);
    expect(cloned.top).not.toBe(tr.top);
    expect(cloned.top[0]).not.toBe(tr.top[0]);
    expect(cloned.tokenEmbedding).not.toBe(tr.tokenEmbedding);
    expect(cloned.attentionWeights).not.toBe(tr.attentionWeights);
  });

  it('modifying cloned arrays does not affect original', () => {
    const tr = makeTrace();
    const cloned = cloneTrace(tr);
    cloned.context.push(99);
    cloned.contextTokens[0] = 'x';
    cloned.top[0].prob = 0;
    cloned.tokenEmbedding[0] = 999;
    expect(tr.context).toHaveLength(3);
    expect(tr.contextTokens[0]).toBe('a');
    expect(tr.top[0].prob).toBe(0.8);
    expect(tr.tokenEmbedding[0]).toBe(0.1);
  });

  it('handles optional attentionWeights', () => {
    const tr = makeTrace({ attentionWeights: undefined });
    const cloned = cloneTrace(tr);
    expect(cloned.attentionWeights).toBeUndefined();
  });

  it('handles empty mlpOut', () => {
    const tr = makeTrace({ mlpOut: [] });
    const cloned = cloneTrace(tr);
    expect(cloned.mlpOut).toEqual([]);
    expect(cloned.mlpOut).not.toBe(tr.mlpOut);
  });
});
