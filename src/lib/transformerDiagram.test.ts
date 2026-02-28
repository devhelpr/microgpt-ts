import { describe, it, expect } from 'vitest';
import { buildElements } from './transformerDiagram';

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

describe('buildElements', () => {
  it('returns expected node count', () => {
    const { nodes } = buildElements(mockT);
    expect(nodes).toHaveLength(15);
  });

  it('returns expected edge count', () => {
    const { edges } = buildElements(mockT);
    expect(edges).toHaveLength(13);
  });

  it('includes all mock labels in nodes', () => {
    const { nodes } = buildElements(mockT);
    const labels = nodes.map((n) => n.data.label);
    expect(labels).toContain('Which character?');
    expect(labels).toContain('Which position?');
    expect(labels).toContain('Turn char into vector');
    expect(labels).toContain('Combine both');
    expect(labels).toContain('Stabilize');
    expect(labels).toContain('Scores for each char');
  });

  it('replaces {n} in transformerBlock with layer number', () => {
    const { nodes } = buildElements(mockT);
    const tbNode = nodes.find((n) => n.data.id === 'TB');
    expect(tbNode).toBeDefined();
    expect(tbNode!.data.label).toMatch(/Transformer block × \d+/);
  });

  it('creates compound structure with TB as parent of G1–G6', () => {
    const { nodes } = buildElements(mockT);
    const children = nodes.filter((n) => n.data.parent === 'TB');
    expect(children).toHaveLength(6);
    expect(children.map((c) => c.data.id).sort()).toEqual(['G1', 'G2', 'G3', 'G4', 'G5', 'G6']);
  });

  it('creates edges in correct order', () => {
    const { edges } = buildElements(mockT);
    const edgeIds = edges.map((e) => e.data.id);
    expect(edgeIds).toContain('A-B');
    expect(edgeIds).toContain('E-F');
    expect(edgeIds).toContain('F-G1');
    expect(edgeIds).toContain('G5-G6');
    expect(edgeIds).toContain('G6-H');
    expect(edgeIds).toContain('H-I');
  });
});
