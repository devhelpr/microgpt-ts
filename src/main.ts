import './style.css';
import mermaid from 'mermaid';
import { createMicroGptTrainer, n_layer, n_head, type StepTrace } from '../microgpt';

const MAX_ITERATION_HISTORY = 50;

function cloneTrace(t: StepTrace): StepTrace {
  return {
    context: t.context.slice(),
    contextTokens: t.contextTokens.slice(),
    targetIndex: t.targetIndex,
    targetToken: t.targetToken,
    predictedToken: t.predictedToken,
    loss: t.loss,
    lr: t.lr,
    gradNorm: t.gradNorm,
    top: t.top.map((x) => ({ token: x.token, prob: x.prob })),
    tokenEmbedding: t.tokenEmbedding.slice(),
    positionEmbedding: t.positionEmbedding.slice(),
    summedEmbedding: t.summedEmbedding.slice(),
    mlpOut: t.mlpOut?.slice() ?? [],
    lnOut: t.lnOut.slice(),
    logits: t.logits.slice(),
    targetProb: t.targetProb,
    attentionWeights: t.attentionWeights?.slice(),
  };
}

type FlowStage = {
  id: string;
  title: string;
  description: string;
};

const FLOW_STAGES: FlowStage[] = [
  {
    id: 'dataset',
    title: '1. Dataset',
    description: 'Read names and split into train/dev/test.',
  },
  {
    id: 'encode',
    title: '2. Encoding',
    description: 'Map characters to integer token IDs (BOS = end-of-sequence).',
  },
  {
    id: 'context',
    title: '3. Context Window',
    description: 'Fixed block_size context; each position predicts the next token.',
  },
  {
    id: 'forward',
    title: '4. Forward Pass',
    description: 'Token + position embedding → RMSNorm → Attention (Q,K,V, multi-head) → residual → RMSNorm → MLP (ReLU) → residual → lm_head → logits.',
  },
  {
    id: 'softmax',
    title: '5. Softmax',
    description: 'Convert logits to probabilities over next characters.',
  },
  {
    id: 'loss',
    title: '6. Loss',
    description: 'Cross-entropy compares predicted distribution vs target token.',
  },
  {
    id: 'backprop',
    title: '7. Backprop',
    description: 'Autograd computes gradients for each parameter.',
  },
  {
    id: 'update',
    title: '8. Update',
    description: 'Adam: m = β1·m + (1−β1)·g, v = β2·v + (1−β2)·g²; param -= lr·m_hat/(√v_hat + ε).',
  },
];

const defaultDataset = ['anna', 'bob', 'carla', 'diana', 'elias', 'frank', 'lucas', 'mila', 'nora'].join('\n');

type IterationSnapshot = {
  step: number;
  trace: StepTrace;
  trainLoss: number;
  devLoss: number;
  batchLoss: number;
};

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App root not found');

function stepExplainerDialogHtml(stage: FlowStage): string {
  const markerId = `arr-${stage.id}`;
  const arrowMarker = `<defs><marker id="${markerId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="currentColor" opacity="0.95"/></marker></defs>`;
  const strokeArrow = `stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.7" marker-end="url(#${markerId})"`;
  const strokeLine = 'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.5"';
  const illustrations: Record<string, string> = {
    dataset: `<svg viewBox="0 0 300 120" class="explainer-illo w-full max-w-sm mx-auto h-32 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="16" width="72" height="26" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="48" width="72" height="26" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="80" width="72" height="26" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="44" y="34" text-anchor="middle" fill="currentColor" font-size="11" font-weight="500">train</text>
      <text x="44" y="66" text-anchor="middle" fill="currentColor" font-size="11" font-weight="500">dev</text>
      <text x="44" y="98" text-anchor="middle" fill="currentColor" font-size="11" font-weight="500">test</text>
      <path d="M88 59 Q 130 59 165 59" ${strokeLine}/>
      <path d="M165 59 L195 59" ${strokeArrow}/>
      <rect x="200" y="38" width="92" height="44" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="246" y="58" text-anchor="middle" fill="currentColor" font-size="10" font-weight="500">names</text>
      <text x="246" y="72" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.85">1 per line</text>
    </svg>`,
    encode: `<svg viewBox="0 0 300 100" class="explainer-illo w-full max-w-sm mx-auto h-28 text-butter/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="58" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="104" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="150" y="28" width="40" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="32" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">a</text>
      <text x="78" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">n</text>
      <text x="124" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">n</text>
      <text x="170" y="50" text-anchor="middle" fill="currentColor" font-size="14" font-family="monospace">a</text>
      <path d="M198 44 L228 44" ${strokeArrow}/>
      <rect x="232" y="32" width="28" height="24" rx="6" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <rect x="264" y="32" width="28" height="24" rx="6" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="246" y="49" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">0</text>
      <text x="278" y="49" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">1</text>
    </svg>`,
    context: `<svg viewBox="0 0 320 88" class="explainer-illo w-full max-w-sm mx-auto h-24 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="58" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="104" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="150" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="196" y="22" width="38" height="38" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <path d="M242 41 L272 41" ${strokeArrow}/>
      <rect x="276" y="18" width="36" height="46" rx="8" fill="currentColor" fill-opacity="0.2" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="294" y="42" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">next?</text>
      <text x="31" y="76" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.7">pos 0</text>
      <text x="215" y="76" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.7">block</text>
    </svg>`,
    forward: `<svg viewBox="0 0 320 128" class="explainer-illo w-full max-w-sm mx-auto h-32 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="12" width="52" height="26" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <rect x="8" y="44" width="52" height="26" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <path d="M68 25 L92 25" ${strokeLine}/>
      <path d="M68 57 L92 57" ${strokeLine}/>
      <rect x="96" y="8" width="48" height="56" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="120" y="38" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">+ embed</text>
      <path d="M152 36 L182 36" ${strokeArrow}/>
      <rect x="186" y="20" width="52" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="212" y="40" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Attn</text>
      <path d="M246 36 L276 36" ${strokeArrow}/>
      <rect x="280" y="20" width="32" height="32" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="296" y="40" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">MLP</text>
      <path d="M296 60 L296 82" ${strokeLine}/>
      <rect x="268" y="86" width="56" height="28" rx="8" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="296" y="104" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">logits</text>
    </svg>`,
    softmax: `<svg viewBox="0 0 300 112" class="explainer-illo w-full max-w-sm mx-auto h-28 text-butter/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="16" width="88" height="56" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="56" y="42" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">logits</text>
      <text x="56" y="58" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.8">(raw)</text>
      <path d="M108 44 L152 44" ${strokeArrow}/>
      <text x="130" y="38" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.9">exp / Σ</text>
      <rect x="164" y="12" width="124" height="72" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <rect x="176" y="28" width="72" height="10" rx="4" fill="currentColor" fill-opacity="0.35"/>
      <rect x="176" y="44" width="52" height="10" rx="4" fill="currentColor" fill-opacity="0.25"/>
      <rect x="176" y="60" width="62" height="10" rx="4" fill="currentColor" fill-opacity="0.3"/>
      <rect x="176" y="76" width="44" height="10" rx="4" fill="currentColor" fill-opacity="0.2"/>
      <text x="226" y="96" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.85">probs Σ = 1</text>
    </svg>`,
    loss: `<svg viewBox="0 0 300 96" class="explainer-illo w-full max-w-sm mx-auto h-26 text-coral/90" aria-hidden="true">${arrowMarker}
      <rect x="12" y="20" width="100" height="56" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="62" y="48" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">p(target)</text>
      <text x="62" y="64" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.85">probability</text>
      <path d="M120 48 L168 48" ${strokeArrow}/>
      <text x="144" y="42" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.9">−log(·)</text>
      <rect x="180" y="20" width="108" height="56" rx="8" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>
      <text x="234" y="48" text-anchor="middle" fill="currentColor" font-size="14" font-weight="700">L</text>
      <text x="234" y="64" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.9">loss</text>
    </svg>`,
    backprop: `<svg viewBox="0 0 300 96" class="explainer-illo w-full max-w-sm mx-auto h-26 text-coral/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="24" width="56" height="48" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="36" y="52" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">params</text>
      <path d="M72 48 L112 48" ${strokeArrow}/>
      <rect x="116" y="32" width="68" height="32" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="150" y="52" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">∂L∕∂</text>
      <path d="M192 48 L232 48" ${strokeArrow}/>
      <rect x="236" y="24" width="56" height="48" rx="8" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="264" y="52" text-anchor="middle" fill="currentColor" font-size="12" font-weight="700">L</text>
      <text x="150" y="84" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.75">backward</text>
    </svg>`,
    update: `<svg viewBox="0 0 300 118" class="explainer-illo w-full max-w-sm mx-auto h-28 text-neon/90" aria-hidden="true">${arrowMarker}
      <rect x="8" y="28" width="72" height="40" rx="8" fill="currentColor" fill-opacity="0.1" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5"/>
      <text x="44" y="52" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">param</text>
      <path d="M88 48 L132 48" ${strokeArrow}/>
      <rect x="136" y="12" width="128" height="56" rx="8" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
      <text x="200" y="34" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">Adam</text>
      <text x="200" y="50" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.85">m,v ← grad · param −= lr·m̂/√v̂</text>
      <path d="M264 76 L264 92" ${strokeLine}/>
      <rect x="218" y="96" width="92" height="20" rx="8" fill="currentColor" fill-opacity="0.15" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5"/>
      <text x="264" y="110" text-anchor="middle" fill="currentColor" font-size="9" font-weight="600">updated</text>
    </svg>`,
  };
  const bodies: Record<string, string> = {
    dataset: `We start with a list of names (one per line). That list is split into three parts: <strong>train</strong>, <strong>dev</strong>, and <strong>test</strong>. The model learns only from the train set. The dev set tells us how well it’s doing while we train (e.g. the “dev loss” you see). The test set is kept aside until the end to measure final performance. Usually we use about 80% for training, 10% for dev, and 10% for test.`,
    encode: `The model can’t work with letters directly—it needs numbers. So every character (like <code>a</code>, <code>n</code>) is turned into a number called a <strong>token ID</strong>. A special token marks the start of a name (<strong>BOS</strong> = beginning of sequence). Later, an “embedding” layer turns these IDs into vectors of numbers the model actually uses. Think of it as: letters → ID numbers → rich number vectors.`,
    context: `The model doesn’t see the whole name at once. It only looks at a fixed number of previous characters—that’s the <strong>context window</strong> (or block). At each step it tries to guess the <strong>next</strong> character. For example, if the context is <code>a, n, n</code>, the target might be <code>a</code> (for “anna”). Then we slide by one character and repeat. So the same name produces many small “predict the next character” tasks.`,
    forward: `This is one full pass through the network. We take the token IDs and their positions, turn them into vectors (embeddings), and add them. Then we run that through: a normalization step (<strong>RMSNorm</strong>), <strong>attention</strong> (so each position can look at the others), a small “MLP” block, and finally a head that outputs one score per possible next character—those scores are the <strong>logits</strong>. No learning happens here; we’re just computing the model’s current prediction.`,
    softmax: `The network outputs raw scores (logits). We need probabilities: “how likely is each character to come next?” <strong>Softmax</strong> does that. It turns the logits into numbers between 0 and 1 that add up to 1 (like a proper probability distribution). The formula is: each probability = exp(logit) divided by the sum of exp of all logits. Training tries to make the probability of the correct next character as high as possible.`,
    loss: `We need one number that says “how wrong was the prediction?” That’s the <strong>loss</strong>. Here we use <strong>cross-entropy</strong>: <code>L = -log(probability we gave to the correct character)</code>. If the model was confident and right, that probability is high and the loss is low. If it was wrong or unsure, the loss is higher. Training aims to make this loss smaller over time.`,
    backprop: `After we have the loss, we need to know how to change every weight in the network to reduce it. <strong>Backpropagation</strong> does that: it works backward from the loss through every layer (attention, MLP, embeddings) and computes a <strong>gradient</strong> for each parameter. The gradient tells us the direction and (roughly) how much to adjust. The “gradient norm” you see is a single number summarizing how big those gradients are.`,
    update: `We have gradients; now we actually change the weights. We use <strong>Adam</strong>, a popular optimizer. It keeps a little “memory” (momentum) and “spread” (variance) per parameter, then updates each weight using the learning rate and those terms: <code>param -= lr · m_hat / (√v_hat + ε)</code>. The learning rate often gets smaller over training (e.g. linear decay), so steps are smaller near the end.`,
  };
  const illo = illustrations[stage.id] ?? '';
  const body = bodies[stage.id] ?? stage.description;
  return `<dialog id="dialog-${stage.id}" class="explainer-dialog rounded-2xl border border-white/15 bg-slate/95 p-0 shadow-2xl backdrop:bg-black/60" aria-labelledby="dialog-title-${stage.id}">
  <div class="explainer-dialog-content max-h-[85vh] overflow-y-auto p-6">
    <div class="flex items-start justify-between gap-4">
      <h2 id="dialog-title-${stage.id}" class="text-xl font-bold text-white">${stage.title}</h2>
      <button type="button" class="dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white" aria-label="Close">✕</button>
    </div>
    <div class="mt-4 text-sm text-white/85 leading-relaxed [&_code]:rounded [&_code]:bg-black/30 [&_code]:px-1 [&_code]:font-mono [&_code]:text-neon">${body}</div>
    ${illo ? `<div class="mt-6 flex justify-center">${illo}</div>` : ''}
    <p class="mt-4 text-xs text-white/50">${stage.description}</p>
  </div>
</dialog>`;
}

const flowNodesHtml = FLOW_STAGES.map(
  (s, i) => `
    <article id="flow-${s.id}" class="flow-node relative cursor-pointer rounded-xl border border-white/10 bg-black/25 p-3 pr-9 transition hover:border-white/25 hover:bg-black/40" data-stage-index="${i}" role="button" tabindex="0">
      <button type="button" class="flow-info-btn absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full border border-white/20 text-white/60 transition hover:border-neon/50 hover:bg-neon/15 hover:text-neon focus:outline-none focus:ring-2 focus:ring-neon/50" data-stage-index="${i}" aria-label="Learn more about ${s.title}">
        <svg class="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
      </button>
      <p class="mono text-xs text-white/45">${String(i + 1).padStart(2, '0')}</p>
      <h4 class="mt-1 text-sm font-semibold text-white">${s.title}</h4>
      <p class="mt-1 text-xs text-white/65">${s.description}</p>
    </article>
  `,
).join('');

app.innerHTML = `
  <main class="mx-auto max-w-7xl px-4 py-8 md:px-8">
    <header class="relative mb-6 overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate to-[#1a1533] p-6 shadow-glow md:p-8">
      <div class="absolute -right-12 -top-12 h-44 w-44 rounded-full bg-coral/20 blur-3xl"></div>
      <div class="absolute -left-8 bottom-0 h-36 w-36 rounded-full bg-neon/20 blur-2xl"></div>
      <p class="panel-title">microgpt in browser</p>
      <h1 class="mt-2 text-3xl font-bold leading-tight md:text-5xl">Visualize the full algorithm, live</h1>
      <p class="mt-4 max-w-3xl text-sm text-white/75 md:text-base">This dashboard explains each phase of microGPT and updates it with real values while training: context tokens, probabilities, loss, gradient norm, and SGD updates.</p>
    </header>

    <section class="mb-4">
      <div class="panel p-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="panel-title mb-0">Generated name</p>
          <button id="sampleBtn" class="rounded-lg border border-butter/50 px-4 py-2 text-sm font-semibold text-butter hover:bg-butter/10">Generate</button>
        </div>
        <pre id="sample" class="mono mt-3 min-h-14 whitespace-pre-wrap rounded-xl border border-white/10 bg-black/25 p-3 text-lg text-neon"></pre>
      </div>
    </section>

    <section class="mb-4 grid gap-4 lg:grid-cols-3">
      <div class="panel p-4 lg:col-span-1">
        <p class="panel-title">Controls</p>
        <label class="mt-4 block text-sm text-white/70">Dataset (1 name per line)</label>
        <textarea id="dataset" class="mono mt-2 h-44 w-full rounded-xl border border-white/15 bg-black/30 p-3 text-sm text-white/90 focus:border-neon focus:outline-none">${defaultDataset}</textarea>

        <div class="mt-4 grid grid-cols-2 gap-3">
          <label class="text-sm text-white/70">Max steps
            <input id="maxSteps" type="number" value="1200" min="50" step="50" class="mono mt-1 w-full rounded-lg border border-white/15 bg-black/30 p-2 text-sm" />
          </label>
          <label class="text-sm text-white/70">Eval every
            <input id="evalEvery" type="number" value="24" min="5" step="1" class="mono mt-1 w-full rounded-lg border border-white/15 bg-black/30 p-2 text-sm" />
          </label>
        </div>

        <div class="mt-4 flex flex-wrap gap-2">
          <button id="startBtn" class="rounded-lg bg-neon px-4 py-2 text-sm font-semibold text-black transition hover:brightness-110">Start</button>
          <button id="pauseBtn" class="rounded-lg border border-white/20 px-4 py-2 text-sm font-semibold text-white/90 hover:bg-white/10">Pause</button>
          <button id="resetBtn" class="rounded-lg border border-coral/50 px-4 py-2 text-sm font-semibold text-coral hover:bg-coral/10">Reset</button>
        </div>

        <div class="mt-4 space-y-2 text-xs text-white/60">
          <div id="dataStats"></div>
          <div>Tip: reset after changing dataset/hyperparameters.</div>
        </div>
      </div>

      <div class="panel p-4 lg:col-span-2">
        <div class="flex items-center justify-between">
          <p class="panel-title">Training Dynamics</p>
          <div id="statusPill" class="rounded-full border border-neon/40 bg-neon/10 px-3 py-1 text-xs text-neon">idle</div>
        </div>

        <div class="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">Step</div><div id="step" class="mono mt-1 text-xl">0</div></div>
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">Batch Loss</div><div id="batchLoss" class="mono mt-1 text-xl">0.0000</div></div>
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">Train Loss</div><div id="trainLoss" class="mono mt-1 text-xl">0.0000</div></div>
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">Dev Loss</div><div id="devLoss" class="mono mt-1 text-xl">0.0000</div></div>
        </div>

        <div class="mt-4 rounded-2xl border border-white/10 bg-black/30 p-3">
          <canvas id="lossChart" class="h-56 w-full"></canvas>
          <div class="mt-2 h-2 overflow-hidden rounded-full bg-white/10"><div id="progressBar" class="meter-fill h-full w-0 transition-all duration-300"></div></div>
        </div>
      </div>
    </section>

    <section class="mb-4 grid gap-4 lg:grid-cols-3">
      <div class="panel p-4 lg:col-span-2">
        <div class="flex items-center justify-between">
          <p class="panel-title">How The Algorithm Works</p>
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs text-white/50">Review:</span>
            <select id="iterationSelect" class="mono rounded-lg border border-white/15 bg-black/30 px-2 py-1.5 text-xs text-white/90 focus:border-neon focus:outline-none">
              <option value="live">Live</option>
            </select>
            <span id="iterationStepLabel" class="text-xs text-white/50"></span>
          </div>
        </div>
        <div class="mt-2 flex flex-wrap items-center gap-2">
          <button type="button" id="showTransformerDiagramBtn" class="rounded-lg border border-neon/50 px-3 py-1.5 text-xs font-semibold text-neon transition hover:bg-neon/15 focus:outline-none focus:ring-2 focus:ring-neon/50">View transformer diagram</button>
        </div>
        <div id="flowGrid" class="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">${flowNodesHtml}</div>
        <div id="flowDetail" class="mt-3 rounded-lg border border-neon/25 bg-neon/10 p-3 text-sm text-neon"></div>
        <div id="flowVisual" class="mt-3 rounded-xl border border-white/10 bg-black/30 p-3"></div>
      </div>

      <div class="panel p-4">
        <p class="panel-title" id="breakdownTitle">Current Step Breakdown</p>
        <div class="mt-3 space-y-2 text-sm">
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">Context IDs:</span> <span id="traceContext" class="mono text-white/90"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">Context tokens:</span> <span id="traceTokens" class="mono text-white/90"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">Target:</span> <span id="traceTarget" class="mono text-butter"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">Predicted:</span> <span id="tracePred" class="mono text-neon"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">Learning rate:</span> <span id="traceLr" class="mono text-white/90"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">Gradient norm:</span> <span id="traceGrad" class="mono text-coral"></span></div>
        </div>
      </div>
    </section>

    <section class="grid gap-4 lg:grid-cols-1">
      <div class="panel p-4">
        <p class="panel-title">Probabilities (Current Step)</p>
        <div id="tokenBars" class="mt-3 space-y-2"></div>
      </div>
    </section>
    <dialog id="dialog-transformer" class="transformer-dialog rounded-2xl border border-white/15 bg-slate/98 p-0 shadow-2xl backdrop:bg-black/70" aria-labelledby="dialog-transformer-title" aria-modal="true">
      <div class="transformer-dialog-content max-h-[90vh] overflow-y-auto p-6">
        <div class="flex items-start justify-between gap-4">
          <h2 id="dialog-transformer-title" class="text-xl font-bold text-white">How the model sees one step</h2>
          <button type="button" class="dialog-close transformer-dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white transition" aria-label="Close">✕</button>
        </div>
        <p class="mt-2 text-sm text-white/75 leading-relaxed">Data flows <strong class="text-neon">top to bottom</strong>. We start with &ldquo;which character?&rdquo; and &ldquo;where in the sequence?&rdquo;, turn them into vectors, combine and normalize, then run the transformer block (attention + small MLP). Finally we get scores for the next character.</p>
        <div id="transformerDiagramContainer" class="mt-6 flex justify-center rounded-xl border border-white/10 bg-black/30 p-6 min-h-[420px]"></div>
        <p class="mt-4 text-xs text-white/55">This is one &ldquo;forward pass&rdquo; for a single position; the same structure repeats for each character the model predicts.</p>
      </div>
    </dialog>
    ${FLOW_STAGES.map((s) => stepExplainerDialogHtml(s)).join('')}
  </main>
`;

const datasetEl = document.querySelector<HTMLTextAreaElement>('#dataset');
const maxStepsEl = document.querySelector<HTMLInputElement>('#maxSteps');
const evalEveryEl = document.querySelector<HTMLInputElement>('#evalEvery');
const startBtn = document.querySelector<HTMLButtonElement>('#startBtn');
const pauseBtn = document.querySelector<HTMLButtonElement>('#pauseBtn');
const resetBtn = document.querySelector<HTMLButtonElement>('#resetBtn');
const sampleBtn = document.querySelector<HTMLButtonElement>('#sampleBtn');
const stepEl = document.querySelector<HTMLElement>('#step');
const batchLossEl = document.querySelector<HTMLElement>('#batchLoss');
const trainLossEl = document.querySelector<HTMLElement>('#trainLoss');
const devLossEl = document.querySelector<HTMLElement>('#devLoss');
const sampleEl = document.querySelector<HTMLElement>('#sample');
const tokenBarsEl = document.querySelector<HTMLElement>('#tokenBars');
const progressBarEl = document.querySelector<HTMLElement>('#progressBar');
const statusPillEl = document.querySelector<HTMLElement>('#statusPill');
const chartCanvas = document.querySelector<HTMLCanvasElement>('#lossChart');
const flowGridEl = document.querySelector<HTMLElement>('#flowGrid');
const flowDetailEl = document.querySelector<HTMLElement>('#flowDetail');
const flowVisualEl = document.querySelector<HTMLElement>('#flowVisual');
const iterationSelectEl = document.querySelector<HTMLSelectElement>('#iterationSelect');
const iterationStepLabelEl = document.querySelector<HTMLElement>('#iterationStepLabel');
const dataStatsEl = document.querySelector<HTMLElement>('#dataStats');
const traceContextEl = document.querySelector<HTMLElement>('#traceContext');
const traceTokensEl = document.querySelector<HTMLElement>('#traceTokens');
const traceTargetEl = document.querySelector<HTMLElement>('#traceTarget');
const tracePredEl = document.querySelector<HTMLElement>('#tracePred');
const traceLrEl = document.querySelector<HTMLElement>('#traceLr');
const traceGradEl = document.querySelector<HTMLElement>('#traceGrad');
const breakdownTitleEl = document.querySelector<HTMLElement>('#breakdownTitle');
const showTransformerDiagramBtn = document.querySelector<HTMLButtonElement>('#showTransformerDiagramBtn');
const dialogTransformer = document.querySelector<HTMLDialogElement>('#dialog-transformer');

if (
  !datasetEl ||
  !maxStepsEl ||
  !evalEveryEl ||
  !startBtn ||
  !pauseBtn ||
  !resetBtn ||
  !sampleBtn ||
  !stepEl ||
  !batchLossEl ||
  !trainLossEl ||
  !devLossEl ||
  !sampleEl ||
  !tokenBarsEl ||
  !progressBarEl ||
  !statusPillEl ||
  !chartCanvas ||
  !flowGridEl ||
  !flowDetailEl ||
  !flowVisualEl ||
  !iterationSelectEl ||
  !iterationStepLabelEl ||
  !dataStatsEl ||
  !traceContextEl ||
  !traceTokensEl ||
  !traceTargetEl ||
  !tracePredEl ||
  !traceLrEl ||
  !traceGradEl ||
  !breakdownTitleEl ||
  !showTransformerDiagramBtn ||
  !dialogTransformer
) {
  throw new Error('Missing required UI element');
}

let trainer = createMicroGptTrainer(datasetEl!.value, {
  blockSize: 16,
  nEmbd: 16,
  maxSteps: Number(maxStepsEl!.value),
  evalEvery: Number(evalEveryEl!.value),
  seed: 1337,
});
let manualSample = '';

let running = false;
let phaseCursor = 0;

let iterationHistory: IterationSnapshot[] = [];
let selectedStageIndex: number | null = null;
let selectedIterationKey: 'live' | string = 'live';

function getDisplayTrace(): StepTrace {
  if (selectedIterationKey === 'live') return trainer.lastTrace;
  const stepNum = parseInt(selectedIterationKey.replace(/^step-/, ''), 10);
  const snap = iterationHistory.find((s) => s.step === stepNum);
  return snap ? snap.trace : trainer.lastTrace;
}

function updateIterationSelectOptions(): void {
  if (!iterationSelectEl || !iterationStepLabelEl) return;
  iterationSelectEl.innerHTML = '<option value="live">Live</option>';
  const start = Math.max(0, iterationHistory.length - MAX_ITERATION_HISTORY);
  for (let i = iterationHistory.length - 1; i >= start; i--) {
    const s = iterationHistory[i];
    const opt = document.createElement('option');
    opt.value = `step-${s.step}`;
    opt.textContent = `Step ${s.step}`;
    iterationSelectEl.appendChild(opt);
  }
  iterationSelectEl.value = selectedIterationKey === 'live' || !iterationHistory.some((s) => `step-${s.step}` === selectedIterationKey) ? 'live' : selectedIterationKey;
  const snap = selectedIterationKey !== 'live' ? iterationHistory.find((s) => `step-${s.step}` === selectedIterationKey) : null;
  iterationStepLabelEl.textContent = snap ? `train=${snap.trainLoss.toFixed(4)} dev=${snap.devLoss.toFixed(4)}` : '';
}

function drawLossChart(losses: number[]): void {
  if (!chartCanvas) return;
  const rect = chartCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(300, Math.floor(rect.width));
  const height = 220;
  chartCanvas.width = width * dpr;
  chartCanvas.height = height * dpr;

  const ctx = chartCanvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, 'rgba(68, 242, 217, 0.12)');
  bg.addColorStop(1, 'rgba(7, 11, 20, 0.0)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  if (losses.length < 2) return;

  const xPad = 12;
  const yPad = 14;
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = max - min || 1;

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#44f2d9';
  ctx.beginPath();

  losses.forEach((loss, i) => {
    const x = xPad + (i / (losses.length - 1)) * (width - xPad * 2);
    const y = yPad + ((max - loss) / range) * (height - yPad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.65)';
  ctx.font = '11px IBM Plex Mono';
  ctx.fillText(`min ${min.toFixed(3)}`, 12, height - 8);
  ctx.fillText(`max ${max.toFixed(3)}`, width - 82, height - 8);
}

function vectorBars(label: string, values: number[], colorClass = 'bg-neon/70'): string {
  const slice = values.slice(0, 8);
  const maxAbs = Math.max(...slice.map((v) => Math.abs(v)), 1e-6);
  const bars = slice
    .map((v) => {
      const w = Math.max(4, Math.round((Math.abs(v) / maxAbs) * 100));
      return `
        <div class="grid grid-cols-[36px_1fr_56px] items-center gap-2 text-xs">
          <span class="mono text-white/50">${v >= 0 ? '+' : '-'}</span>
          <div class="h-2 overflow-hidden rounded bg-white/10"><div class="h-full ${colorClass}" style="width:${w}%"></div></div>
          <span class="mono text-right text-white/70">${v.toFixed(3)}</span>
        </div>
      `;
    })
    .join('');
  return `
    <div class="rounded-lg border border-white/10 bg-black/25 p-2">
      <div class="mb-2 text-xs text-white/60">${label} (first 8 dims)</div>
      <div class="space-y-1">${bars}</div>
    </div>
  `;
}

function stageVisualHtml(stageId: string, trace: StepTrace): string {
  const t = trace;
  if (stageId === 'dataset') {
    const total = Math.max(1, trainer.trainSize + trainer.devSize + trainer.testSize);
    const trainW = Math.round((trainer.trainSize / total) * 100);
    const devW = Math.round((trainer.devSize / total) * 100);
    const testW = Math.round((trainer.testSize / total) * 100);
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">Split of docs (names) used for train/dev/test.</div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mb-1 text-xs text-white/55">train ${trainer.trainSize}</div>
          <div class="h-2 rounded bg-white/10"><div class="h-full rounded bg-neon/70" style="width:${trainW}%"></div></div>
        </div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mb-1 text-xs text-white/55">dev ${trainer.devSize}</div>
          <div class="h-2 rounded bg-white/10"><div class="h-full rounded bg-butter/70" style="width:${devW}%"></div></div>
        </div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mb-1 text-xs text-white/55">test ${trainer.testSize}</div>
          <div class="h-2 rounded bg-white/10"><div class="h-full rounded bg-coral/70" style="width:${testW}%"></div></div>
        </div>
      </div>
    `;
  }

  if (stageId === 'encode') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">Characters become token IDs before training.</div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mono text-xs text-white/60">context tokens -> ids</div>
          <div class="mt-2 flex flex-wrap gap-2">
            ${t.contextTokens
              .map((tok, i) => `<span class="mono rounded border border-white/15 px-2 py-1 text-xs">${tok} -> ${t.context[i]}</span>`)
              .join('')}
          </div>
        </div>
        <div class="mono text-xs text-butter">target: ${t.targetToken} -> ${t.targetIndex}</div>
      </div>
    `;
  }

  if (stageId === 'context') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">Sliding context window predicts next token.</div>
        <div class="flex items-center gap-2">
          ${t.contextTokens.map((tok) => `<div class="mono rounded-lg border border-white/15 bg-black/25 px-3 py-2 text-sm">${tok}</div>`).join('')}
          <div class="text-neon">-></div>
          <div class="mono rounded-lg border border-butter/30 bg-butter/10 px-3 py-2 text-sm text-butter">${t.targetToken}</div>
        </div>
      </div>
    `;
  }

  if (stageId === 'forward') {
    const attnHtml =
      t.attentionWeights && t.attentionWeights.length > 0
        ? `
      <div class="rounded-lg border border-white/10 bg-black/25 p-2">
        <div class="mb-2 text-xs text-white/60">Attention (head 0) over context positions</div>
        <div class="flex flex-wrap gap-1">
          ${t.attentionWeights
            .map(
              (w, i) =>
                `<div class="h-4 w-6 rounded bg-neon/70" style="opacity:${Math.max(0.2, w)}" title="pos ${i}: ${w.toFixed(3)}"></div>`,
            )
            .join('')}
        </div>
      </div>
    `
        : '';
    return `
      <div class="grid gap-2 md:grid-cols-2">
        ${vectorBars('token embedding', t.tokenEmbedding, 'bg-neon/70')}
        ${vectorBars('position embedding', t.positionEmbedding, 'bg-butter/70')}
        ${vectorBars('sum embedding', t.summedEmbedding, 'bg-cyan-300/70')}
        ${vectorBars('pre-head (after RMSNorm + MLP block)', t.lnOut, 'bg-indigo-300/70')}
        ${attnHtml}
      </div>
    `;
  }

  if (stageId === 'softmax') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">Logits converted into normalized probabilities.</div>
        ${t.top
          .map((row) => {
            const w = Math.max(6, Math.round(row.prob * 100));
            return `
              <div class="grid grid-cols-[40px_1fr_56px] items-center gap-2">
                <span class="mono text-white/70">${row.token}</span>
                <div class="h-2 rounded bg-white/10"><div class="meter-fill h-full" style="width:${w}%"></div></div>
                <span class="mono text-right text-white/70">${row.prob.toFixed(3)}</span>
              </div>
            `;
          })
          .join('')}
      </div>
    `;
  }

  if (stageId === 'loss') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">Cross-entropy for the true next token.</div>
        <div class="mono rounded-lg border border-white/10 bg-black/25 p-2 text-xs">
          L = -log(p(target)) = -log(${Math.max(1e-9, t.targetProb).toFixed(4)}) = ${t.loss.toFixed(4)}
        </div>
        <div class="text-xs text-white/60">Predicted ${t.predictedToken}, target ${t.targetToken}</div>
      </div>
    `;
  }

  if (stageId === 'backprop') {
    const pct = Math.min(100, Math.round(t.gradNorm * 200));
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">Backward pass computes gradients through the graph.</div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mono text-xs text-white/60">gradient norm ${t.gradNorm.toFixed(6)}</div>
          <div class="mt-2 h-2 rounded bg-white/10"><div class="h-full rounded bg-coral/70" style="width:${pct}%"></div></div>
        </div>
      </div>
    `;
  }

  const delta = t.lr * t.gradNorm;
  return `
    <div class="space-y-2 text-sm">
      <div class="text-white/70">Adam: m = β1·m + (1−β1)·g, v = β2·v + (1−β2)·g²; param -= lr·m_hat/(√v_hat + ε).</div>
      <div class="mono rounded-lg border border-white/10 bg-black/25 p-2 text-xs">
        param -= lr * m_hat / (sqrt(v_hat) + ε)
      </div>
      <div class="mono text-xs text-white/70">lr=${t.lr.toFixed(4)} | avg step magnitude≈${delta.toExponential(2)}</div>
      ${vectorBars('pre-head (before lm_head)', t.lnOut, 'bg-emerald-300/70')}
    </div>
  `;
}

/** Plain-English Mermaid flowchart so the diagram is understandable without ML jargon. */
function getTransformerMermaidCode(): string {
  return `flowchart TD
  A["Which character?"]
  C["Which position?"]
  A --> B["Turn character into a vector"]
  C --> D["Add position as a vector"]
  B --> E["Combine both"]
  D --> E
  E --> F["Stabilize scale"]
  F --> TB
  subgraph TB["Transformer block × ${n_layer}"]
    direction TB
    G1["Stabilize"]
    G2["Attention: mix with context"]
    G3["Add shortcut"]
    G4["Stabilize"]
    G5["Small feed-forward"]
    G6["Add shortcut"]
    G1 --> G2 --> G3 --> G4 --> G5 --> G6
  end
  TB --> H["Predict next character"]
  H --> I["Scores for each character"]`;
}

function renderFlow(): void {
  const activeIdx =
    selectedStageIndex !== null
      ? selectedStageIndex
      : running
        ? phaseCursor
        : FLOW_STAGES.length - 1;
  const trace = getDisplayTrace();
  FLOW_STAGES.forEach((stage, i) => {
    const el = document.querySelector<HTMLElement>(`#flow-${stage.id}`);
    if (!el) return;
    el.classList.toggle('active', i === activeIdx);
    el.classList.toggle('flow-node-selected', selectedStageIndex === i);
  });
  const active = FLOW_STAGES[activeIdx] ?? FLOW_STAGES[0];
  flowDetailEl!.textContent = `${active.title}: ${active.description}`;
  flowVisualEl!.innerHTML = stageVisualHtml(active.id, trace);
}

function render(): void {
  stepEl!.textContent = `${trainer.step}/${trainer.maxSteps}`;
  batchLossEl!.textContent = trainer.latestBatchLoss.toFixed(4);
  trainLossEl!.textContent = trainer.trainLoss.toFixed(4);
  devLossEl!.textContent = trainer.devLoss.toFixed(4);
  sampleEl!.textContent = manualSample || trainer.sample || '...';

  dataStatsEl!.textContent = `words=${datasetEl!.value.split(/\r?\n/).filter(Boolean).length} | vocab=${trainer.vocabSize} | train/dev/test=${trainer.trainSize}/${trainer.devSize}/${trainer.testSize}`;

  progressBarEl!.style.width = `${Math.min(100, (trainer.step / trainer.maxSteps) * 100)}%`;
  statusPillEl!.textContent = running ? 'training' : trainer.step >= trainer.maxSteps ? 'completed' : 'idle';

  const showStepDetails = !running;
  const breakdownPanel = breakdownTitleEl!.closest('.panel');
  const flowSection = flowGridEl!.closest('.panel');
  const tokenBarsPanel = tokenBarsEl!.closest('.panel');
  if (breakdownPanel) breakdownPanel.classList.toggle('hidden', !showStepDetails);
  if (flowSection) flowSection.querySelector('#flowDetail')?.classList.toggle('hidden', !showStepDetails);
  if (flowSection) flowSection.querySelector('#flowVisual')?.classList.toggle('hidden', !showStepDetails);
  if (tokenBarsPanel) tokenBarsPanel.classList.toggle('hidden', !showStepDetails);

  if (showStepDetails) {
    breakdownTitleEl!.textContent = selectedIterationKey === 'live' ? 'Current Step Breakdown' : `Step ${selectedIterationKey.replace(/^step-/, '')} Breakdown`;

    const displayTrace = getDisplayTrace();
    traceContextEl!.textContent = `[${displayTrace.context.join(', ')}]`;
    traceTokensEl!.textContent = `[${displayTrace.contextTokens.join(', ')}]`;
    traceTargetEl!.textContent = `${displayTrace.targetToken} (${displayTrace.targetIndex})`;
    tracePredEl!.textContent = displayTrace.predictedToken;
    traceLrEl!.textContent = displayTrace.lr.toFixed(4);
    traceGradEl!.textContent = displayTrace.gradNorm.toFixed(6);

    tokenBarsEl!.innerHTML = displayTrace.top
      .map((t) => {
        const width = Math.max(6, Math.round(t.prob * 100));
        return `
        <div class="grid grid-cols-[40px_1fr_52px] items-center gap-2 text-sm">
          <span class="mono text-white/70">${t.token}</span>
          <div class="h-2 overflow-hidden rounded bg-white/10"><div class="meter-fill h-full" style="width:${width}%"></div></div>
          <span class="mono text-right text-white/70">${t.prob.toFixed(3)}</span>
        </div>
      `;
      })
      .join('');

    renderFlow();
  }

  drawLossChart(trainer.losses.slice(-300));
}

async function trainLoop(): Promise<void> {
  if (running) return;
  running = true;
  render();

  while (running && trainer.step < trainer.maxSteps) {
    trainer.trainStep();
    iterationHistory.push({
      step: trainer.step,
      trace: cloneTrace(trainer.lastTrace),
      trainLoss: trainer.trainLoss,
      devLoss: trainer.devLoss,
      batchLoss: trainer.latestBatchLoss,
    });
    if (iterationHistory.length > MAX_ITERATION_HISTORY) iterationHistory.shift();
    updateIterationSelectOptions();
    render();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  running = false;
  render();
}

function resetTrainer(): void {
  running = false;
  phaseCursor = 0;
  manualSample = '';
  iterationHistory = [];
  selectedIterationKey = 'live';
  selectedStageIndex = null;
  trainer = createMicroGptTrainer(datasetEl!.value, {
    blockSize: 16,
    nEmbd: 16,
    maxSteps: Math.max(50, Number(maxStepsEl!.value) || 1200),
    evalEvery: Math.max(5, Number(evalEveryEl!.value) || 24),
    seed: 1337,
  });
  updateIterationSelectOptions();
  render();
}

startBtn.addEventListener('click', () => {
  void trainLoop();
});

pauseBtn.addEventListener('click', () => {
  running = false;
  render();
});

resetBtn.addEventListener('click', () => {
  resetTrainer();
});

sampleBtn.addEventListener('click', () => {
  manualSample = trainer.generate(42);
  render();
});

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
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
    curve: 'basis',
    padding: 20,
    nodeSpacing: 50,
    rankSpacing: 40,
    useMaxWidth: true,
    htmlLabels: true,
  },
});

showTransformerDiagramBtn!.addEventListener('click', async () => {
  const container = document.getElementById('transformerDiagramContainer');
  if (!container) return;
  container.innerHTML = `<div class="mermaid">${getTransformerMermaidCode()}</div>`;
  dialogTransformer!.showModal();
  try {
    await mermaid.run({ nodes: container.querySelectorAll('.mermaid') });
  } catch (err) {
    container.innerHTML = `<p class="text-sm text-coral">Diagram could not be drawn. Try refreshing.</p>`;
  }
});

iterationSelectEl!.addEventListener('change', () => {
  selectedIterationKey = iterationSelectEl!.value;
  const snap = selectedIterationKey !== 'live' ? iterationHistory.find((s) => `step-${s.step}` === selectedIterationKey) : null;
  iterationStepLabelEl!.textContent = snap ? `train=${snap.trainLoss.toFixed(4)} dev=${snap.devLoss.toFixed(4)}` : '';
  render();
});

flowGridEl!.addEventListener('click', (e) => {
  const infoBtn = (e.target as HTMLElement).closest('.flow-info-btn');
  if (infoBtn) {
    e.preventDefault();
    e.stopPropagation();
    const idx = parseInt((infoBtn as HTMLElement).dataset.stageIndex ?? '-1', 10);
    if (idx >= 0 && idx < FLOW_STAGES.length) {
      const stage = FLOW_STAGES[idx];
      const dialog = document.querySelector<HTMLDialogElement>(`#dialog-${stage.id}`);
      if (dialog) dialog.showModal();
    }
    return;
  }
  const target = (e.target as HTMLElement).closest('[data-stage-index]');
  if (!target) return;
  const idx = parseInt((target as HTMLElement).dataset.stageIndex ?? '-1', 10);
  if (idx >= 0 && idx < FLOW_STAGES.length) {
    selectedStageIndex = selectedStageIndex === idx ? null : idx;
    render();
  }
});

flowGridEl!.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const target = (e.target as HTMLElement).closest('[data-stage-index]');
  if (!target) return;
  e.preventDefault();
  const idx = parseInt((target as HTMLElement).dataset.stageIndex ?? '-1', 10);
  if (idx >= 0 && idx < FLOW_STAGES.length) {
    selectedStageIndex = selectedStageIndex === idx ? null : idx;
    render();
  }
});

// Close explainer dialogs: close button and backdrop click (native)
app.addEventListener('click', (e) => {
  const closeBtn = (e.target as HTMLElement).closest('.dialog-close');
  if (closeBtn) {
    const dialog = closeBtn.closest('dialog');
    if (dialog instanceof HTMLDialogElement) dialog.close();
  }
});
app.addEventListener('cancel', (e) => {
  if ((e.target as HTMLDialogElement)?.tagName === 'DIALOG') (e.target as HTMLDialogElement).close();
}, true);

window.addEventListener('resize', () => render());

render();
