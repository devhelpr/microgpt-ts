import { n_layer } from '../../microgpt';
import type { LocaleStrings } from '../i18n/types';

/** Plain-English Mermaid flowchart so the diagram is understandable without ML jargon. */
export function getTransformerMermaidCode(t: Pick<LocaleStrings, 'mermaid'>): string {
  const M = t.mermaid;
  return `flowchart TD
  A["${M.whichCharacter}"]
  C["${M.whichPosition}"]
  A --> B["${M.turnCharIntoVector}"]
  C --> D["${M.addPositionAsVector}"]
  B --> E["${M.combineBoth}"]
  D --> E
  E --> F["${M.stabilizeScale}"]
  F --> TB
  subgraph TB["${M.transformerBlock.replace('{n}', String(n_layer))}"]
    direction TB
    G1["${M.stabilize}"]
    G2["${M.attentionMixContext}"]
    G3["${M.addShortcut}"]
    G4["${M.stabilize}"]
    G5["${M.smallFeedForward}"]
    G6["${M.addShortcut}"]
    G1 --> G2 --> G3 --> G4 --> G5 --> G6
  end
  TB --> H["${M.predictNextChar}"]
  H --> I["${M.scoresForEachChar}"]`;
}

export const mermaidThemeConfig = {
  startOnLoad: false,
  theme: 'base' as const,
  themeVariables: {
    darkMode: true,
    background: '#0f172a',
    primaryColor: '#1e293b',
    primaryTextColor: '#e2e8f0',
    primaryBorderColor: '#44f2d9',
    lineColor: '#44f2d9',
    secondaryColor: '#0f172a',
    tertiaryColor: '#1e293b',
    tertiaryBorderColor: '#64748b',
    nodeBorder: '#44f2d9',
    clusterBkg: '#0f172a',
    clusterBorder: '#475569',
    titleColor: '#44f2d9',
    edgeLabelBackground: '#1e293b',
    nodeTextColor: '#e2e8f0',
    textColor: '#94a3b8',
    mainBkg: '#1e293b',
    border1: '#44f2d9',
    border2: '#64748b',
    arrowheadColor: '#44f2d9',
    fontFamily: 'IBM Plex Mono, monospace',
  },
  flowchart: {
    curve: 'basis' as const,
    padding: 20,
    nodeSpacing: 50,
    rankSpacing: 40,
    useMaxWidth: true,
    htmlLabels: true,
  },
};
