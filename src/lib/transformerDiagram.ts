import cytoscape, { type Core } from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { n_layer } from '../../microgpt';
import type { LocaleStrings } from '../i18n/types';

cytoscape.use(dagre);

const INFO_ICON = '\u24D8'; // ⓘ

const ACCENT = '#44f2d9';
const BG_DARK = '#0f172a';
const NODE_BG = '#1e293b';
const TEXT = '#e2e8f0';
const BORDER = '#64748b';

type TransformerDiagramLocale = Pick<LocaleStrings, 'mermaid' | 'transformerDiagramExplainers'>;

/** Build Cytoscape elements from locale labels. Same structure as the original Mermaid flowchart. */
export function buildElements(t: TransformerDiagramLocale) {
  const M = t.mermaid;
  const E = t.transformerDiagramExplainers;
  const tbLabel = M.transformerBlock.replace('{n}', String(n_layer));

  return {
    nodes: [
      { data: { id: 'A', label: `${M.whichCharacter} ${INFO_ICON}`, explainer: E.A } },
      { data: { id: 'B', label: `${M.turnCharIntoVector} ${INFO_ICON}`, explainer: E.B } },
      { data: { id: 'C', label: `${M.whichPosition} ${INFO_ICON}`, explainer: E.C } },
      { data: { id: 'D', label: `${M.addPositionAsVector} ${INFO_ICON}`, explainer: E.D } },
      { data: { id: 'E', label: `${M.combineBoth} ${INFO_ICON}`, explainer: E.E } },
      { data: { id: 'F', label: `${M.stabilizeScale} ${INFO_ICON}`, explainer: E.F } },
      { data: { id: 'TB', label: '' } },
      { data: { id: 'TB_TITLE', label: `${tbLabel} ${INFO_ICON}`, explainer: E.TB, parent: 'TB' } },
      { data: { id: 'G1', label: `${M.stabilize} ${INFO_ICON}`, explainer: E.G1, parent: 'TB' } },
      { data: { id: 'G2', label: `${M.attentionMixContext} ${INFO_ICON}`, explainer: E.G2, parent: 'TB' } },
      { data: { id: 'G3', label: `${M.addShortcut} ${INFO_ICON}`, explainer: E.G3, parent: 'TB' } },
      { data: { id: 'G4', label: `${M.stabilize} ${INFO_ICON}`, explainer: E.G4, parent: 'TB' } },
      { data: { id: 'G5', label: `${M.smallFeedForward} ${INFO_ICON}`, explainer: E.G5, parent: 'TB' } },
      { data: { id: 'G6', label: `${M.addShortcut} ${INFO_ICON}`, explainer: E.G6, parent: 'TB' } },
      { data: { id: 'H', label: `${M.predictNextChar} ${INFO_ICON}`, explainer: E.H } },
      { data: { id: 'I', label: `${M.scoresForEachChar} ${INFO_ICON}`, explainer: E.I } },
    ],
    edges: [
      { data: { id: 'A-B', source: 'A', target: 'B' } },
      { data: { id: 'C-D', source: 'C', target: 'D' } },
      { data: { id: 'B-E', source: 'B', target: 'E' } },
      { data: { id: 'D-E', source: 'D', target: 'E' } },
      { data: { id: 'E-F', source: 'E', target: 'F' } },
      { data: { id: 'F-TB_TITLE', source: 'F', target: 'TB_TITLE' } },
      { data: { id: 'TB_TITLE-G1', source: 'TB_TITLE', target: 'G1' } },
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

function createPopup(container: HTMLElement): { backdrop: HTMLDivElement; popup: HTMLDivElement } {
  const backdrop = document.createElement('div');
  backdrop.className = 'transformer-diagram-popup-backdrop';
  backdrop.setAttribute('aria-hidden', 'true');
  backdrop.hidden = true;

  const popup = document.createElement('div');
  popup.className = 'transformer-diagram-popup';
  popup.setAttribute('role', 'dialog');
  popup.setAttribute('aria-modal', 'true');
  popup.setAttribute('aria-labelledby', 'transformer-diagram-popup-content');

  const content = document.createElement('p');
  content.id = 'transformer-diagram-popup-content';
  content.className = 'transformer-diagram-popup-content';
  popup.appendChild(content);

  backdrop.appendChild(popup);
  container.appendChild(backdrop);

  return { backdrop, popup };
}

function showPopup(backdrop: HTMLDivElement, popup: HTMLDivElement, text: string): void {
  const content = popup.querySelector('.transformer-diagram-popup-content');
  if (content) content.textContent = text;
  backdrop.hidden = false;
}

function hidePopup(backdrop: HTMLDivElement): void {
  backdrop.hidden = true;
}

/**
 * Renders the transformer architecture diagram into the given container.
 * Supports animation, interaction, and full control over styling.
 * Call destroyTransformerDiagram(container) before re-rendering to clean up.
 */
export function renderTransformerDiagram(
  container: HTMLElement,
  t: TransformerDiagramLocale,
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
      animate: false,
      nodeDimensionsIncludeLabels: true,
    } as cytoscape.LayoutOptions,
    minZoom: 0.4,
    maxZoom: 3,
    wheelSensitivity: 0.3,
  });

  (container as HTMLElement & { _cy?: Core })._cy = cy;

  const { backdrop, popup } = createPopup(container);

  cy.on('tap', 'node', (evt) => {
    const explainer = evt.target.data('explainer');
    if (explainer) showPopup(backdrop, popup, explainer);
  });

  backdrop.addEventListener('click', (e) => {
    if (e.target === backdrop) hidePopup(backdrop);
  });

  cy.on('tap', (evt) => {
    if (evt.target === cy) hidePopup(backdrop);
  });

  cy.one('layoutstop', () => {
    const padding = 40;
    cy.fit(undefined, padding);
    const zoom = cy.zoom();
    if (zoom < 1) cy.zoom(Math.min(1.15, 1 / zoom));
    const extent = cy.extent();
    const pan = cy.pan();
    const zoomFinal = cy.zoom();
    const topRendered = extent.y1 * zoomFinal + pan.y;
    if (topRendered !== padding) {
      cy.panBy({ x: 0, y: padding - topRendered });
    }
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
