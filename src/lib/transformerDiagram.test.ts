import { describe, it, expect } from 'vitest';
import { buildElements } from './transformerDiagram';

const mockExplainers = {
  A: 'Explainer for A',
  B: 'Explainer for B',
  C: 'Explainer for C',
  D: 'Explainer for D',
  E: 'Explainer for E',
  F: 'Explainer for F',
  TB: 'Explainer for TB',
  G1: 'Explainer for G1',
  G2: 'Explainer for G2',
  G3: 'Explainer for G3',
  G4: 'Explainer for G4',
  G5: 'Explainer for G5',
  G6: 'Explainer for G6',
  H: 'Explainer for H',
  I: 'Explainer for I',
};

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
  transformerDiagramExplainers: mockExplainers,
};

describe('buildElements', () => {
  it('returns expected node count', () => {
    const { nodes } = buildElements(mockT);
    expect(nodes).toHaveLength(16);
  });

  it('returns expected edge count', () => {
    const { edges } = buildElements(mockT);
    expect(edges).toHaveLength(14);
  });

  it('includes all mock labels and explainers in nodes', () => {
    const { nodes } = buildElements(mockT);
    const labels = nodes.map((n) => n.data.label);
    expect(labels.some((l) => l.includes('Which character?'))).toBe(true);
    expect(labels.some((l) => l.includes('Which position?'))).toBe(true);
    expect(labels.some((l) => l.includes('Turn char into vector'))).toBe(true);
    expect(labels.some((l) => l.includes('Combine both'))).toBe(true);
    expect(labels.some((l) => l.includes('Stabilize'))).toBe(true);
    expect(labels.some((l) => l.includes('Scores for each char'))).toBe(true);
    const nodeA = nodes.find((n) => n.data.id === 'A');
    expect(nodeA?.data.explainer).toBe('Explainer for A');
  });

  it('replaces {n} in transformerBlock with layer number', () => {
    const { nodes } = buildElements(mockT);
    const tbTitleNode = nodes.find((n) => n.data.id === 'TB_TITLE');
    expect(tbTitleNode).toBeDefined();
    expect(tbTitleNode!.data.label).toMatch(/Transformer block × \d+.*/);
  });

  it('creates compound structure with TB as parent of TB_TITLE and G1–G6', () => {
    const { nodes } = buildElements(mockT);
    const children = nodes.filter((n) => n.data.parent === 'TB');
    expect(children).toHaveLength(7);
    expect(children.map((c) => c.data.id).sort()).toEqual(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'TB_TITLE']);
  });

  it('creates edges in correct order', () => {
    const { edges } = buildElements(mockT);
    const edgeIds = edges.map((e) => e.data.id);
    expect(edgeIds).toContain('A-B');
    expect(edgeIds).toContain('E-F');
    expect(edgeIds).toContain('F-TB_TITLE');
    expect(edgeIds).toContain('TB_TITLE-G1');
    expect(edgeIds).toContain('G5-G6');
    expect(edgeIds).toContain('G6-H');
    expect(edgeIds).toContain('H-I');
  });
});
