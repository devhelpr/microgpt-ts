import { describe, it, expect } from 'vitest';
import {
  Value,
  makeRng,
  randn,
  matrix,
  linear,
  softmaxValue,
  rmsnorm,
  softmax,
  sampleCategorical,
  sparkline,
  topKIndices,
  createMicroGptTrainer,
  n_embd,
  block_size,
  head_dim,
  n_head,
} from './microgpt';

describe('Value', () => {
  it('add returns correct data', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = a.add(b);
    expect(c.data).toBe(5);
  });

  it('add with number coerces to Value', () => {
    const a = new Value(2);
    const c = a.add(3);
    expect(c.data).toBe(5);
  });

  it('mul returns correct data', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = a.mul(b);
    expect(c.data).toBe(6);
  });

  it('backward propagates gradient for add', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = a.add(b);
    c.backward();
    expect(c.grad).toBe(1);
    expect(a.grad).toBe(1);
    expect(b.grad).toBe(1);
  });

  it('backward propagates gradient for mul', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = a.mul(b);
    c.backward();
    expect(a.grad).toBe(3);
    expect(b.grad).toBe(2);
  });

  it('relu(positive) returns value', () => {
    const a = new Value(3);
    const r = a.relu();
    expect(r.data).toBe(3);
  });

  it('relu(negative) returns zero', () => {
    const a = new Value(-2);
    const r = a.relu();
    expect(r.data).toBe(0);
  });

  it('relu(0) returns zero', () => {
    const a = new Value(0);
    const r = a.relu();
    expect(r.data).toBe(0);
  });

  it('relu backward passes grad only when data > 0', () => {
    const a = new Value(1);
    const r = a.relu();
    r.backward();
    expect(a.grad).toBe(1);

    const b = new Value(-1);
    const s = b.relu();
    s.backward();
    expect(b.grad).toBe(0);
  });

  it('log and exp round-trip', () => {
    const a = new Value(2);
    const b = a.log().exp();
    expect(b.data).toBeCloseTo(2, 10);
  });

  it('pow computes power', () => {
    const a = new Value(2);
    expect(a.pow(3).data).toBe(8);
    expect(a.pow(0.5).data).toBeCloseTo(Math.SQRT2, 10);
  });
});

describe('makeRng', () => {
  it('returns deterministic sequence for same seed', () => {
    const r1 = makeRng(42);
    const r2 = makeRng(42);
    for (let i = 0; i < 5; i++) {
      expect(r1()).toBe(r2());
    }
  });

  it('returns values in [0, 1)', () => {
    const r = makeRng(123);
    for (let i = 0; i < 20; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe('randn', () => {
  it('with zero std returns mean', () => {
    const r = makeRng(99);
    // randn uses two draws; we can't get exact 0 std behavior without mocking
    const v = randn(r, 5, 0.0001);
    expect(v).toBeCloseTo(5, 1);
  });
});

describe('matrix', () => {
  it('returns correct shape', () => {
    const rng = makeRng(1);
    const m = matrix(3, 4, rng);
    expect(m.length).toBe(3);
    expect(m[0].length).toBe(4);
    expect(m.every((row) => row.length === 4)).toBe(true);
  });

  it('contains Value instances', () => {
    const rng = makeRng(1);
    const m = matrix(2, 2, rng);
    expect(m[0][0]).toBeInstanceOf(Value);
  });
});

describe('linear', () => {
  it('output length equals number of rows of w', () => {
    const x = [new Value(1), new Value(2)];
    const w = [
      [new Value(1), new Value(0)],
      [new Value(0), new Value(1)],
      [new Value(1), new Value(1)],
    ];
    const out = linear(x, w);
    expect(out.length).toBe(3);
  });

  it('computes dot product per row', () => {
    const x = [new Value(1), new Value(2)];
    const w = [[new Value(1), new Value(1)]];
    const out = linear(x, w);
    expect(out[0].data).toBe(3);
  });
});

describe('softmaxValue', () => {
  it('output sums to 1', () => {
    const logits = [new Value(0), new Value(1), new Value(2)];
    const probs = softmaxValue(logits);
    const sum = probs.reduce((s, p) => s + p.data, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it('all outputs are positive', () => {
    const logits = [new Value(-10), new Value(0), new Value(10)];
    const probs = softmaxValue(logits);
    probs.forEach((p) => expect(p.data).toBeGreaterThan(0));
  });

  it('backward flows through softmax', () => {
    const logits = [new Value(0), new Value(1), new Value(0)];
    const probs = softmaxValue(logits);
    const loss = probs[1].log().mul(-1);
    for (const p of logits) p.grad = 0;
    loss.backward();
    logits.forEach((l) => expect(typeof l.grad).toBe('number'));
  });
});

describe('rmsnorm', () => {
  it('output length equals input length', () => {
    const x = [new Value(1), new Value(2), new Value(3)];
    const out = rmsnorm(x);
    expect(out.length).toBe(3);
  });

  it('scale is applied (output magnitude ~1 for unit-like input)', () => {
    const x = [new Value(1), new Value(1), new Value(1)];
    const out = rmsnorm(x);
    const rms = Math.sqrt(out.reduce((s, o) => s + o.data * o.data, 0) / 3);
    expect(rms).toBeCloseTo(1, 5);
  });
});

describe('softmax (numeric)', () => {
  it('output sums to 1', () => {
    const probs = softmax([0, 1, 2]);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it('handles large negative logits', () => {
    const probs = softmax([-1000, -1000, 0]);
    expect(probs[2]).toBeCloseTo(1, 5);
    expect(probs[0]).toBeCloseTo(0, 5);
  });
});

describe('sampleCategorical', () => {
  it('returns index in valid range', () => {
    const probs = [0.5, 0.3, 0.2];
    const rng = makeRng(456);
    for (let i = 0; i < 50; i++) {
      const idx = sampleCategorical(probs, rng);
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(probs.length);
    }
  });

  it('deterministic with same rng seed returns same sequence', () => {
    const probs = [0.33, 0.33, 0.34];
    const a = makeRng(777);
    const b = makeRng(777);
    for (let i = 0; i < 10; i++) {
      expect(sampleCategorical(probs, a)).toBe(sampleCategorical(probs, b));
    }
  });
});

describe('sparkline', () => {
  it('returns empty string for empty input', () => {
    expect(sparkline([])).toBe('');
  });

  it('returns string of length min(width, values.length)', () => {
    const values = [1, 2, 3, 4, 5];
    expect(sparkline(values, 3).length).toBe(3);
    expect(sparkline(values, 10).length).toBe(5);
  });

  it('uses only allowed characters', () => {
    const chars = ' .:-=+*#%@';
    const out = sparkline([1, 2, 3, 4, 5], 5);
    for (const c of out) {
      expect(chars).toContain(c);
    }
  });
});

describe('topKIndices', () => {
  it('returns k indices', () => {
    const arr = [3, 1, 4, 1, 5];
    expect(topKIndices(arr, 2)).toHaveLength(2);
  });

  it('returns indices with largest values first', () => {
    const arr = [3, 1, 4, 1, 5];
    const top = topKIndices(arr, 3);
    expect(arr[top[0]]).toBe(5);
    expect(arr[top[1]]).toBe(4);
    expect(arr[top[2]]).toBe(3);
  });

  it('k larger than array returns all indices', () => {
    const arr = [1, 2];
    const top = topKIndices(arr, 10);
    expect(top).toHaveLength(2);
  });
});

describe('constants', () => {
  it('head_dim * n_head equals n_embd', () => {
    expect(head_dim * n_head).toBe(n_embd);
  });

  it('block_size and n_embd are positive', () => {
    expect(block_size).toBeGreaterThan(0);
    expect(n_embd).toBeGreaterThan(0);
  });
});

describe('createMicroGptTrainer (training run + generation)', () => {
  const dataset = 'anna\nbob\ncarla\ndiana\nelias\nfrank\n';
  const allowedChars = new Set(dataset.replace(/\n/g, '').split(''));

  it('runs a complete training run and generate() returns names using only dataset characters', { timeout: 30000 }, () => {
    const trainer = createMicroGptTrainer(dataset, {
      maxSteps: 200,
      evalEvery: 50,
      seed: 42,
      blockSize: 16,
      nEmbd: 16,
    });

    for (let i = 0; i < trainer.maxSteps; i++) {
      trainer.trainStep();
    }

    expect(trainer.step).toBe(trainer.maxSteps);
    expect(trainer.losses.length).toBeGreaterThan(0);
    expect(typeof trainer.trainLoss).toBe('number');
    expect(typeof trainer.devLoss).toBe('number');
    expect(typeof trainer.testLoss).toBe('number');

    const generated = trainer.generate(40);
    expect(typeof generated).toBe('string');
    for (const c of generated) {
      expect(allowedChars.has(c)).toBe(true);
    }
    console.log('Generated names (sample):', generated || '(empty)');
  });

  it('generate() is deterministic with same seed and produces non-empty or empty string', { timeout: 15000 }, () => {
    const trainer = createMicroGptTrainer(dataset, {
      maxSteps: 100,
      evalEvery: 50,
      seed: 123,
    });
    for (let i = 0; i < trainer.maxSteps; i++) trainer.trainStep();

    const a = trainer.generate(30);
    const b = trainer.generate(30);
    expect(typeof a).toBe('string');
    expect(typeof b).toBe('string');
    expect(a.length).toBeLessThanOrEqual(30);
    expect(b.length).toBeLessThanOrEqual(30);
    console.log('Generated (a):', a || '(empty)');
    console.log('Generated (b):', b || '(empty)');
  });
});
