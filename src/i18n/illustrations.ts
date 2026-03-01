import type { IllustrationLabels } from './types';

/** Build stage explainer SVG from locale labels (for i18n). */
export function buildIllustrationSvg(stageId: string, labels: IllustrationLabels): string {
  const markerId = `arr-${stageId}`;
  const arrowMarker = `<defs><marker id="${markerId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="currentColor" opacity="0.95"/></marker></defs>`;
  const strokeArrow = `stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.7" marker-end="url(#${markerId})"`;
  const strokeLine = 'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.5"';
  const L = labels;
  const svgs: Record<string, string> = {
    dataset: `<svg viewBox="0 0 300 130" class="explainer-illo w-full mx-auto text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="16" width="80" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="52" width="80" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="88" width="80" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="48" y="32" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.train}</text>
      <text x="48" y="68" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.dev}</text>
      <text x="48" y="104" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.test}</text>
      <path d="M96 65 Q 138 65 173 65" ${strokeLine}/>
      <path d="M173 65 L203 65" ${strokeArrow}/>
      <rect x="208" y="42" width="84" height="52" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="250" y="60" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="500">${L.dataset.names}</text>
      <text x="250" y="76" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.dataset.namesPerLine}</text>
    </svg>`,
    encode: `<svg viewBox="0 0 340 100" class="explainer-illo w-full mx-auto text-butter/90" aria-hidden="true">${arrowMarker}
      <rect x="10" y="24" width="48" height="44" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="62" y="24" width="48" height="44" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="114" y="24" width="48" height="44" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="166" y="24" width="48" height="44" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="34" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="13" font-family="monospace">${L.encode.charA}</text>
      <text x="86" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="13" font-family="monospace">${L.encode.charN}</text>
      <text x="138" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="13" font-family="monospace">${L.encode.charN}</text>
      <text x="190" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="13" font-family="monospace">${L.encode.charA}</text>
      <path d="M218 46 L248 46" ${strokeArrow}/>
      <rect x="252" y="32" width="32" height="32" rx="6" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <rect x="288" y="32" width="32" height="32" rx="6" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="268" y="48" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.encode.id0}</text>
      <text x="304" y="48" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.encode.id1}</text>
    </svg>`,
    context: `<svg viewBox="0 0 340 100" class="explainer-illo w-full mx-auto text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="10" y="20" width="44" height="44" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="58" y="20" width="44" height="44" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="106" y="20" width="44" height="44" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="154" y="20" width="44" height="44" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="202" y="20" width="44" height="44" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <path d="M254 42 L282 42" ${strokeArrow}/>
      <rect x="286" y="16" width="40" height="52" rx="8" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="306" y="42" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.context.next}</text>
      <text x="32" y="78" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.context.pos0}</text>
      <text x="224" y="78" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.context.block}</text>
    </svg>`,
    forward: `<svg viewBox="0 0 500 140" class="explainer-illo w-full mx-auto text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="20" width="70" height="78" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.5"/>
      <text x="43" y="52" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="12" font-weight="600">${L.forward.tokens}</text>
      <text x="43" y="68" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">ids</text>
      <path d="M82 59 L104 59" ${strokeArrow}/>
      <rect x="108" y="14" width="74" height="90" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="145" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="12" font-weight="600">${L.forward.embed}</text>
      <text x="145" y="64" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.forward.embedHint}</text>
      <path d="M186 59 L208 59" ${strokeArrow}/>
      <rect x="212" y="14" width="74" height="90" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="249" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="12" font-weight="600">${L.forward.attn}</text>
      <text x="249" y="64" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.forward.attnHint}</text>
      <path d="M290 59 L312 59" ${strokeArrow}/>
      <rect x="316" y="14" width="74" height="90" rx="8" fill="currentColor" fill-opacity="0.14" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="353" y="46" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="12" font-weight="600">${L.forward.mlp}</text>
      <text x="353" y="64" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.forward.mlpHint}</text>
      <path d="M394 59 L416 59" ${strokeArrow}/>
      <rect x="420" y="20" width="56" height="78" rx="8" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="448" y="50" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="12" font-weight="600">${L.forward.logits}</text>
      <text x="448" y="68" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.forward.logitsHint}</text>
    </svg>`,
    softmax: `<svg viewBox="0 0 300 120" class="explainer-illo w-full mx-auto text-butter/90" aria-hidden="true">${arrowMarker}
      <rect x="10" y="16" width="96" height="64" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="58" y="38" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.softmax.logits}</text>
      <text x="58" y="54" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.softmax.raw}</text>
      <path d="M112 48 L156 48" ${strokeArrow}/>
      <text x="134" y="42" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.softmax.expSum}</text>
      <rect x="168" y="10" width="124" height="80" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="180" y="28" width="72" height="12" rx="4" fill="currentColor" fill-opacity="0.35"/>
      <rect x="180" y="44" width="52" height="12" rx="4" fill="currentColor" fill-opacity="0.25"/>
      <rect x="180" y="60" width="62" height="12" rx="4" fill="currentColor" fill-opacity="0.3"/>
      <rect x="180" y="76" width="44" height="12" rx="4" fill="currentColor" fill-opacity="0.2"/>
      <text x="230" y="100" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.softmax.probsSum1}</text>
    </svg>`,
    loss: `<svg viewBox="0 0 300 100" class="explainer-illo w-full mx-auto text-coral/90" aria-hidden="true">${arrowMarker}
      <rect x="10" y="18" width="108" height="64" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="64" y="42" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.loss.pTarget}</text>
      <text x="64" y="58" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.loss.probability}</text>
      <path d="M126 50 L174 50" ${strokeArrow}/>
      <text x="150" y="44" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.loss.negLog}</text>
      <rect x="182" y="18" width="108" height="64" rx="8" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="236" y="42" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="15" font-weight="700">${L.loss.L}</text>
      <text x="236" y="58" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.loss.loss}</text>
    </svg>`,
    backprop: `<svg viewBox="0 0 300 100" class="explainer-illo w-full mx-auto text-coral/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="20" width="64" height="56" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="40" y="48" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.backprop.params}</text>
      <path d="M76 48 L116 48" ${strokeArrow}/>
      <rect x="120" y="28" width="76" height="40" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="158" y="48" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.backprop.gradLabel}</text>
      <path d="M200 48 L240 48" ${strokeArrow}/>
      <rect x="244" y="20" width="64" height="56" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="276" y="48" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="13" font-weight="700">${L.backprop.L}</text>
      <text x="158" y="88" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.backprop.backward}</text>
    </svg>`,
    update: `<svg viewBox="0 0 400 130" class="explainer-illo w-full mx-auto text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="20" width="76" height="72" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="46" y="48" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.update.param}</text>
      <text x="46" y="66" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">+ ${L.update.grad}</text>
      <path d="M88 56 L110 56" ${strokeArrow}/>
      <rect x="114" y="6" width="172" height="108" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="200" y="32" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="12" font-weight="600">${L.update.adam}</text>
      <text x="200" y="54" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" opacity="0.95">${L.update.adamMomentum}</text>
      <text x="200" y="78" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="10" font-family="monospace" opacity="0.95">${L.update.adamUpdate}</text>
      <path d="M290 56 L312 56" ${strokeArrow}/>
      <rect x="316" y="20" width="76" height="72" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="354" y="56" text-anchor="middle" dominant-baseline="middle" fill="currentColor" font-size="11" font-weight="600">${L.update.updated}</text>
    </svg>`,
  };
  return svgs[stageId] ?? '';
}
