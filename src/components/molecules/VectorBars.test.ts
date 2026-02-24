import { describe, it, expect } from 'vitest';
import { vectorBars } from './VectorBars';

const mockT = {
  vectorBars: {
    first8Dims: '(first 8 dims)',
  },
};

describe('vectorBars', () => {
  it('includes label and first8Dims text', () => {
    const html = vectorBars('Token embedding', [0.1, 0.2], 'bg-neon/70', mockT);
    expect(html).toContain('Token embedding');
    expect(html).toContain('(first 8 dims)');
  });

  it('renders at most 8 value rows', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const html = vectorBars('Test', values, 'bg-neon/70', mockT);
    const rowMatches = html.match(/text-white\/70">[-\d.]+<\/span>/g);
    expect(rowMatches?.length ?? 0).toBe(8);
  });

  it('uses provided colorClass for bars', () => {
    const html = vectorBars('Test', [1], 'bg-butter/70', mockT);
    expect(html).toContain('bg-butter/70');
  });

  it('handles empty values with fallback maxAbs', () => {
    const html = vectorBars('Empty', [], 'bg-neon/70', mockT);
    expect(html).toContain('Empty');
    expect(html).toContain('(first 8 dims)');
  });
});
