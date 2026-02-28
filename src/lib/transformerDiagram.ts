import cytoscape, { type Core } from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { n_layer } from '../../microgpt';
import type { LocaleStrings } from '../i18n/types';

cytoscape.use(dagre);

const ACCENT = '#44f2d9';
const BG_DARK = '#0f172a';
const NODE_BG = '#1e293b';
const TEXT = '#e2e8f0';
const BORDER = '#64748b';

/** Build Cytoscape elements from locale labels. Same structure as the original Mermaid flowchart. */
export function buildElements(t: Pick<LocaleStrings, 'mermaid'>) {
  const M = t.mermaid;
  const tbLabel = M.transformerBlock.replace('{n}', String(n_layer));

  return {
    nodes: [
      { data: { id: 'A', label: M.whichCharacter } },
      { data: { id: 'B', label: M.turnCharIntoVector } },
      { data: { id: 'C', label: M.whichPosition } },
      { data: { id: 'D', label: M.addPositionAsVector } },
      { data: { id: 'E', label: M.combineBoth } },
      { data: { id: 'F', label: M.stabilizeScale } },
      { data: { id: 'TB', label: tbLabel } },
      { data: { id: 'G1', label: M.stabilize, parent: 'TB' } },
      { data: { id: 'G2', label: M.attentionMixContext, parent: 'TB' } },
      { data: { id: 'G3', label: M.addShortcut, parent: 'TB' } },
      { data: { id: 'G4', label: M.stabilize, parent: 'TB' } },
      { data: { id: 'G5', label: M.smallFeedForward, parent: 'TB' } },
      { data: { id: 'G6', label: M.addShortcut, parent: 'TB' } },
      { data: { id: 'H', label: M.predictNextChar } },
      { data: { id: 'I', label: M.scoresForEachChar } },
    ],
    edges: [
      { data: { id: 'A-B', source: 'A', target: 'B' } },
      { data: { id: 'C-D', source: 'C', target: 'D' } },
      { data: { id: 'B-E', source: 'B', target: 'E' } },
      { data: { id: 'D-E', source: 'D', target: 'E' } },
      { data: { id: 'E-F', source: 'E', target: 'F' } },
      { data: { id: 'F-G1', source: 'F', target: 'G1' } },
      { data: { id: 'G1-G2', source: 'G1', target: 'G2' } },
      { data: { id: 'G2-G3', source: 'G2', target: 'G3' } },
      { data: { id: 'G3-G4', source: 'G3', target: 'G4' } },
      { data: { id: 'G4-G5', source: 'G4', target: 'G5' } },
      { data: { id: 'G5-G6', source: 'G5', target: 'G6' } },
      { data: { id: 'G6-H', source: 'G6', target: 'H' } },
      { data: { id: 'H-I', source: 'H', target: 'I' } },
    ],
  };
}

let flowAnimationFrame: number | null = null;

function startEdgeFlowAnimation(cy: Core) {
  let offset = 0;
  const animate = () => {
    offset = (offset + 2) % 20;
    cy.edges().style('line-dash-offset', String(-offset));
    flowAnimationFrame = requestAnimationFrame(animate);
  };
  flowAnimationFrame = requestAnimationFrame(animate);
}

function stopEdgeFlowAnimation() {
  if (flowAnimationFrame != null) {
    cancelAnimationFrame(flowAnimationFrame);
    flowAnimationFrame = null;
  }
}

/**
 * Renders the transformer architecture diagram into the given container.
 * Supports animation, interaction, and full control over styling.
 * Call destroyTransformerDiagram(container) before re-rendering to clean up.
 */
export function renderTransformerDiagram(
  container: HTMLElement,
  t: Pick<LocaleStrings, 'mermaid'>,
): void {
  const existing = (container as HTMLElement & { _cy?: Core })._cy;
  if (existing) {
    stopEdgeFlowAnimation();
    existing.destroy();
    delete (container as HTMLElement & { _cy?: Core })._cy;
  }

  const elements = buildElements(t);

  const cy = cytoscape({
    container,
    elements,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Cytoscape style types are overly strict
    style: [
      {
        selector: 'node',
        style: {
          'background-color': NODE_BG,
          'border-width': 1,
          'border-color': ACCENT,
          'border-opacity': 0.8,
          color: TEXT,
          'font-family': 'IBM Plex Mono, monospace',
          'font-size': '15px',
          label: 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'text-wrap': 'wrap',
          'text-max-width': '280px',
          width: 'label',
          height: 'label',
          shape: 'round-rectangle',
          padding: '16px',
          'transition-property': 'border-color border-width background-color',
          'transition-duration': 200,
        },
      },
      {
        selector: 'node:parent',
        style: {
          'background-color': BG_DARK,
          'border-color': BORDER,
          'border-width': 1,
          'border-style': 'dashed',
          'text-valign': 'top',
          padding: '20px',
        },
      },
      {
        selector: 'node:hover',
        style: {
          'border-width': 2,
          'border-color': ACCENT,
          'background-color': '#334155',
        },
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': ACCENT,
          'line-color': ACCENT,
          'line-opacity': 0.9,
          width: 1.5,
          'line-dash-pattern': [6, 4] as [number, number],
          'line-dash-offset': 0,
          'transition-property': 'line-opacity',
          'transition-duration': 200,
        },
      },
      {
        selector: 'edge:hover',
        style: {
          'line-opacity': 1,
          width: 2,
        },
      },
    ],
    layout: {
      name: 'dagre',
      rankDir: 'TB' as const,
      nodeSep: 50,
      rankSep: 60,
      padding: 40,
      fit: true,
      animate: true,
      animationDuration: 400,
      nodeDimensionsIncludeLabels: true,
    } as cytoscape.LayoutOptions,
    minZoom: 0.4,
    maxZoom: 3,
    wheelSensitivity: 0.3,
  });

  (container as HTMLElement & { _cy?: Core })._cy = cy;

  cy.one('layoutstop', () => {
    cy.fit(undefined, 40);
    const zoom = cy.zoom();
    if (zoom < 1) cy.zoom(Math.min(1.15, 1 / zoom));
  });

  startEdgeFlowAnimation(cy);
}

/**
 * Destroys the diagram instance and stops animations.
 * Call before re-rendering or when the dialog closes to avoid memory leaks.
 */
export function destroyTransformerDiagram(container: HTMLElement): void {
  const existing = (container as HTMLElement & { _cy?: Core })._cy;
  if (existing) {
    stopEdgeFlowAnimation();
    existing.destroy();
    delete (container as HTMLElement & { _cy?: Core })._cy;
  }
}
