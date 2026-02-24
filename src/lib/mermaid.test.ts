import { describe, it, expect } from 'vitest';
import { getTransformerMermaidCode } from './mermaid';

const mockT = {
  mermaid: {
    whichCharacter: 'Which character?',
    whichPosition: 'Which position?',
    turnCharIntoVector: 'Turn char into vector',
    addPositionAsVector: 'Add position as vector',
    combineBoth: 'Combine both',
    stabilizeScale: 'Stabilize scale',
    transformerBlock: 'Transformer block × {n}',
    stabilize: 'Stabilize',
    attentionMixContext: 'Attention mix context',
    addShortcut: 'Add shortcut',
    smallFeedForward: 'Small feed-forward',
    predictNextChar: 'Predict next char',
    scoresForEachChar: 'Scores for each char',
  },
};

describe('getTransformerMermaidCode', () => {
  it('returns flowchart TD structure', () => {
    const code = getTransformerMermaidCode(mockT);
    expect(code).toContain('flowchart TD');
  });

  it('includes all mock node labels', () => {
    const code = getTransformerMermaidCode(mockT);
    expect(code).toContain('Which character?');
    expect(code).toContain('Which position?');
    expect(code).toContain('Turn char into vector');
    expect(code).toContain('Add position as vector');
    expect(code).toContain('Combine both');
    expect(code).toContain('Stabilize scale');
    expect(code).toContain('Transformer block');
    expect(code).toContain('Predict next char');
    expect(code).toContain('Scores for each char');
  });

  it('includes subgraph with direction TB', () => {
    const code = getTransformerMermaidCode(mockT);
    expect(code).toContain('subgraph');
    expect(code).toContain('direction TB');
  });

  it('replaces {n} in transformerBlock with layer number', () => {
    const code = getTransformerMermaidCode(mockT);
    expect(code).toMatch(/Transformer block × \d+/);
  });
});
