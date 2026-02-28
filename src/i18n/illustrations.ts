import type { IllustrationLabels } from './types';

/** Build stage explainer SVG from locale labels (for i18n). */
export function buildIllustrationSvg(stageId: string, labels: IllustrationLabels): string {
  const markerId = `arr-${stageId}`;
  const arrowMarker = `<defs><marker id="${markerId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="currentColor" opacity="0.95"/></marker></defs>`;
  const strokeArrow = `stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.7" marker-end="url(#${markerId})"`;
  const strokeLine = 'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.5"';
  const L = labels;
  const svgs: Record<string, string> = {
    dataset: `<svg viewBox="0 0 300 120" class="explainer-illo w-full max-w-sm mx-auto h-32 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="16" width="72" height="26" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="48" width="72" height="26" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="80" width="72" height="26" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="44" y="34" text-anchor="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.train}</text>
      <text x="44" y="66" text-anchor="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.dev}</text>
      <text x="44" y="98" text-anchor="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.test}</text>
      <path d="M88 59 Q 130 59 165 59" ${strokeLine}/>
      <path d="M165 59 L195 59" ${strokeArrow}/>
      <rect x="200" y="38" width="92" height="44" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="246" y="58" text-anchor="middle" fill="currentColor" font-size="10" font-weight="500">${L.dataset.names}</text>
      <text x="246" y="72" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.85">${L.dataset.namesPerLine}</text>
    </svg>`,
    encode: `<svg viewBox="0 0 300 100" class="explainer-illo w-full max-w-sm mx-auto h-28 text-butter/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="58" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="104" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="150" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="32" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">${L.encode.charA}</text>
      <text x="78" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">${L.encode.charN}</text>
      <text x="124" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">${L.encode.charN}</text>
      <text x="170" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">${L.encode.charA}</text>
      <path d="M198 44 L228 44" ${strokeArrow}/>
      <rect x="232" y="32" width="28" height="24" rx="6" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <rect x="264" y="32" width="28" height="24" rx="6" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="246" y="49" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">${L.encode.id0}</text>
      <text x="278" y="49" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">${L.encode.id1}</text>
    </svg>`,
    context: `<svg viewBox="0 0 320 88" class="explainer-illo w-full max-w-sm mx-auto h-24 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="58" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="104" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="150" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="196" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <path d="M242 41 L272 41" ${strokeArrow}/>
      <rect x="276" y="18" width="36" height="46" rx="8" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="294" y="42" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">${L.context.next}</text>
      <text x="31" y="76" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.7">${L.context.pos0}</text>
      <text x="215" y="76" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.7">${L.context.block}</text>
    </svg>`,
    forward: `<svg viewBox="0 0 320 128" class="explainer-illo w-full max-w-sm mx-auto h-32 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="12" width="52" height="26" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="44" width="52" height="26" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <path d="M68 25 L92 25" ${strokeLine}/>
      <path d="M68 57 L92 57" ${strokeLine}/>
      <rect x="96" y="8" width="48" height="56" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="120" y="38" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">${L.forward.embed}</text>
      <path d="M152 36 L182 36" ${strokeArrow}/>
      <rect x="186" y="20" width="52" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="212" y="40" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.forward.attn}</text>
      <path d="M246 36 L276 36" ${strokeArrow}/>
      <rect x="280" y="20" width="32" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="296" y="40" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">${L.forward.mlp}</text>
      <path d="M296 60 L296 82" ${strokeLine}/>
      <rect x="268" y="86" width="56" height="28" rx="8" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="296" y="104" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">${L.forward.logits}</text>
    </svg>`,
    softmax: `<svg viewBox="0 0 300 112" class="explainer-illo w-full max-w-sm mx-auto h-28 text-butter/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="16" width="88" height="56" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="56" y="42" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.softmax.logits}</text>
      <text x="56" y="58" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.8">${L.softmax.raw}</text>
      <path d="M108 44 L152 44" ${strokeArrow}/>
      <text x="130" y="38" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.9">${L.softmax.expSum}</text>
      <rect x="164" y="12" width="124" height="72" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="176" y="28" width="72" height="10" rx="4" fill="currentColor" fill-opacity="0.35"/>
      <rect x="176" y="44" width="52" height="10" rx="4" fill="currentColor" fill-opacity="0.25"/>
      <rect x="176" y="60" width="62" height="10" rx="4" fill="currentColor" fill-opacity="0.3"/>
      <rect x="176" y="76" width="44" height="10" rx="4" fill="currentColor" fill-opacity="0.2"/>
      <text x="226" y="96" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.85">${L.softmax.probsSum1}</text>
    </svg>`,
    loss: `<svg viewBox="0 0 300 96" class="explainer-illo w-full max-w-sm mx-auto h-26 text-coral/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="20" width="100" height="56" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="62" y="48" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.loss.pTarget}</text>
      <text x="62" y="64" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.85">${L.loss.probability}</text>
      <path d="M120 48 L168 48" ${strokeArrow}/>
      <text x="144" y="42" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.9">${L.loss.negLog}</text>
      <rect x="180" y="20" width="108" height="56" rx="8" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="234" y="48" text-anchor="middle" fill="currentColor" font-size="14" font-weight="700">${L.loss.L}</text>
      <text x="234" y="64" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.9">${L.loss.loss}</text>
    </svg>`,
    backprop: `<svg viewBox="0 0 300 96" class="explainer-illo w-full max-w-sm mx-auto h-26 text-coral/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="24" width="56" height="48" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="36" y="52" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.backprop.params}</text>
      <path d="M72 48 L112 48" ${strokeArrow}/>
      <rect x="116" y="32" width="68" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="150" y="52" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">${L.backprop.gradLabel}</text>
      <path d="M192 48 L232 48" ${strokeArrow}/>
      <rect x="236" y="24" width="56" height="48" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="264" y="52" text-anchor="middle" fill="currentColor" font-size="12" font-weight="700">${L.backprop.L}</text>
      <text x="150" y="84" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.75">${L.backprop.backward}</text>
    </svg>`,
    update: `<svg viewBox="0 0 360 100" class="explainer-illo w-full max-w-sm mx-auto h-28 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="22" width="64" height="56" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="40" y="42" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.update.param}</text>
      <text x="40" y="58" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.8">+ ${L.update.grad}</text>
      <path d="M76 50 L98 50" ${strokeArrow}/>
      <rect x="102" y="8" width="156" height="84" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="180" y="28" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.update.adam}</text>
      <text x="180" y="48" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.9">${L.update.adamMomentum}</text>
      <text x="180" y="64" text-anchor="middle" fill="currentColor" font-size="8" font-family="monospace" opacity="0.9">${L.update.adamUpdate}</text>
      <path d="M262 50 L284 50" ${strokeArrow}/>
      <rect x="288" y="22" width="64" height="56" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="320" y="50" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">${L.update.updated}</text>
    </svg>`,
  };
  return svgs[stageId] ?? '';
}
