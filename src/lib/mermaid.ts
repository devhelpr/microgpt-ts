import { n_layer } from '../../microgpt';
import type { LocaleStrings } from '../i18n/types';

/**
 * Insert <br/> at word boundaries only for long labels (avoids clipping on very long text).
 * Breaks at spaces or after hyphens (e.g. "feed-forward" → "feed-<br/>forward").
 */
function wrapLabel(s: string, maxChars = 20): string {
  if (s.length <= maxChars) return s;
  const midpoint = Math.ceil(s.length / 2);
  const lastSpace = s.lastIndexOf(' ', midpoint);
  const firstSpace = s.indexOf(' ', midpoint);
  const lastHyphen = s.lastIndexOf('-', midpoint);
  const firstHyphen = s.indexOf('-', midpoint);

  let first: string;
  let rest: string;
  if (lastSpace > 0 || firstSpace > 0) {
    const i = lastSpace > 0 ? lastSpace : firstSpace;
    first = s.slice(0, i);
    rest = s.slice(i + 1);
  } else if (lastHyphen >= 0 || firstHyphen >= 0) {
    const i = lastHyphen >= 0 ? lastHyphen : firstHyphen;
    first = s.slice(0, i + 1);
    rest = s.slice(i + 1);
  } else {
    first = s.slice(0, midpoint);
    rest = s.slice(midpoint);
  }
  if (!rest) return first;
  return rest.length > maxChars ? `${first}<br/>${wrapLabel(rest, maxChars)}` : `${first}<br/>${rest}`;
}

/** Plain-English Mermaid flowchart so the diagram is understandable without ML jargon. */
export function getTransformerMermaidCode(t: Pick<LocaleStrings, 'mermaid'>): string {
  const M = t.mermaid;
  return `flowchart TD
  A["${wrapLabel(M.whichCharacter)}"]
  C["${wrapLabel(M.whichPosition)}"]
  A --> B["${wrapLabel(M.turnCharIntoVector)}"]
  C --> D["${wrapLabel(M.addPositionAsVector)}"]
  B --> E["${wrapLabel(M.combineBoth)}"]
  D --> E
  E --> F["${wrapLabel(M.stabilizeScale)}"]
  F --> TB
  subgraph TB["${wrapLabel(M.transformerBlock.replace('{n}', String(n_layer)))}"]
    direction TB
    G1["${wrapLabel(M.stabilize)}"]
    G2["${wrapLabel(M.attentionMixContext)}"]
    G3["${wrapLabel(M.addShortcut)}"]
    G4["${wrapLabel(M.stabilize)}"]
    G5["${wrapLabel(M.smallFeedForward)}"]
    G6["${wrapLabel(M.addShortcut)}"]
    G1 --> G2 --> G3 --> G4 --> G5 --> G6
  end
  TB --> H["${wrapLabel(M.predictNextChar)}"]
  H --> I["${wrapLabel(M.scoresForEachChar)}"]`;
}

export const mermaidThemeConfig = {
  startOnLoad: false,
  theme: 'base' as const,
  themeVariables: {
    darkMode: true,
    fontSize: '17px',
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
    padding: 30,
    nodeSpacing: 50,
    rankSpacing: 40,
    useMaxWidth: true,
    htmlLabels: true,
    wrappingWidth: 280,
  },
};
