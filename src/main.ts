import './style.css';
import { createMicroGptTrainer, type StepTrace } from '../microgpt';

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
  const illustrations: Record<string, string> = {
    dataset: `<svg viewBox="0 0 240 120" class="w-full max-w-sm mx-auto h-32 text-neon/80" aria-hidden="true">
      <rect x="10" y="20" width="70" height="24" rx="4" fill="currentColor" opacity="0.25"/>
      <rect x="10" y="50" width="70" height="24" rx="4" fill="currentColor" opacity="0.25"/>
      <rect x="10" y="80" width="70" height="24" rx="4" fill="currentColor" opacity="0.25"/>
      <text x="45" y="37" text-anchor="middle" fill="currentColor" font-size="10" font-family="system-ui">train</text>
      <text x="45" y="67" text-anchor="middle" fill="currentColor" font-size="10">dev</text>
      <text x="45" y="97" text-anchor="middle" fill="currentColor" font-size="10">test</text>
      <path d="M90 60 L150 60" stroke="currentColor" stroke-width="2" opacity="0.6"/>
      <polygon points="145,55 155,60 145,65" fill="currentColor" opacity="0.6"/>
      <rect x="160" y="40" width="70" height="40" rx="4" fill="currentColor" opacity="0.15" stroke="currentColor" stroke-width="1"/>
      <text x="195" y="65" text-anchor="middle" fill="currentColor" font-size="9">names</text>
      <text x="195" y="78" text-anchor="middle" fill="currentColor" font-size="9">1 per line</text>
    </svg>`,
    encode: `<svg viewBox="0 0 260 100" class="w-full max-w-sm mx-auto h-28 text-butter/90" aria-hidden="true">
      <rect x="10" y="30" width="36" height="28" rx="3" fill="currentColor" opacity="0.3"/>
      <text x="28" y="48" text-anchor="middle" fill="currentColor" font-size="12" font-family="monospace">a</text>
      <rect x="52" y="30" width="36" height="28" rx="3" fill="currentColor" opacity="0.3"/>
      <text x="70" y="48" text-anchor="middle" fill="currentColor" font-size="12">n</text>
      <rect x="94" y="30" width="36" height="28" rx="3" fill="currentColor" opacity="0.3"/>
      <text x="112" y="48" text-anchor="middle" fill="currentColor" font-size="12">n</text>
      <rect x="136" y="30" width="36" height="28" rx="3" fill="currentColor" opacity="0.3"/>
      <text x="154" y="48" text-anchor="middle" fill="currentColor" font-size="12">a</text>
      <path d="M80 70 L120 70" stroke="currentColor" stroke-width="1.5" opacity="0.7"/>
      <polygon points="115,65 125,70 115,75" fill="currentColor" opacity="0.7"/>
      <rect x="130" y="58" width="24" height="24" rx="2" fill="currentColor" opacity="0.2"/>
      <text x="142" y="74" text-anchor="middle" fill="currentColor" font-size="10">0</text>
      <rect x="158" y="58" width="24" height="24" rx="2" fill="currentColor" opacity="0.2"/>
      <text x="170" y="74" text-anchor="middle" fill="currentColor" font-size="10">1</text>
      <rect x="186" y="58" width="24" height="24" rx="2" fill="currentColor" opacity="0.2"/>
      <text x="198" y="74" text-anchor="middle" fill="currentColor" font-size="10">1</text>
      <rect x="214" y="58" width="24" height="24" rx="2" fill="currentColor" opacity="0.2"/>
      <text x="226" y="74" text-anchor="middle" fill="currentColor" font-size="10">0</text>
    </svg>`,
    context: `<svg viewBox="0 0 320 90" class="w-full max-w-sm mx-auto h-24 text-neon/80" aria-hidden="true">
      <rect x="8" y="25" width="32" height="32" rx="4" fill="currentColor" opacity="0.2"/>
      <rect x="44" y="25" width="32" height="32" rx="4" fill="currentColor" opacity="0.2"/>
      <rect x="80" y="25" width="32" height="32" rx="4" fill="currentColor" opacity="0.2"/>
      <rect x="116" y="25" width="32" height="32" rx="4" fill="currentColor" opacity="0.2"/>
      <rect x="152" y="25" width="32" height="32" rx="4" fill="currentColor" opacity="0.2"/>
      <line x1="188" y1="41" x2="228" y2="41" stroke="currentColor" stroke-width="2" opacity="0.8"/>
      <polygon points="223,36 233,41 223,46" fill="currentColor" opacity="0.8"/>
      <rect x="238" y="25" width="40" height="32" rx="4" fill="currentColor" opacity="0.4" stroke="currentColor" stroke-width="1.5"/>
      <text x="258" y="45" text-anchor="middle" fill="currentColor" font-size="9">next?</text>
      <text x="20" y="78" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.8">pos 0</text>
      <text x="128" y="78" text-anchor="middle" fill="currentColor" font-size="8" opacity="0.8">… block_size</text>
    </svg>`,
    forward: `<svg viewBox="0 0 280 130" class="w-full max-w-sm mx-auto h-32 text-neon/80" aria-hidden="true">
      <rect x="20" y="10" width="50" height="22" rx="3" fill="currentColor" opacity="0.2"/>
      <text x="45" y="25" text-anchor="middle" fill="currentColor" font-size="9">tokens</text>
      <rect x="20" y="38" width="50" height="22" rx="3" fill="currentColor" opacity="0.2"/>
      <text x="45" y="53" text-anchor="middle" fill="currentColor" font-size="9">pos</text>
      <path d="M70 21 L95 21" stroke="currentColor" stroke-width="1" opacity="0.6"/>
      <path d="M70 49 L95 49" stroke="currentColor" stroke-width="1" opacity="0.6"/>
      <rect x="95" y="8" width="42" height="52" rx="4" fill="currentColor" opacity="0.15"/>
      <text x="116" y="35" text-anchor="middle" fill="currentColor" font-size="8">+</text>
      <text x="116" y="50" text-anchor="middle" fill="currentColor" font-size="8">embed</text>
      <path d="M137 34 L162 34" stroke="currentColor" stroke-width="1" opacity="0.6"/>
      <rect x="162" y="18" width="50" height="32" rx="3" fill="currentColor" opacity="0.2"/>
      <text x="187" y="38" text-anchor="middle" fill="currentColor" font-size="8">Attn</text>
      <path d="M212 34 L237 34" stroke="currentColor" stroke-width="1" opacity="0.6"/>
      <rect x="237" y="18" width="36" height="32" rx="3" fill="currentColor" opacity="0.25"/>
      <text x="255" y="38" text-anchor="middle" fill="currentColor" font-size="8">MLP</text>
      <path d="M255 50 L255 70" stroke="currentColor" stroke-width="1" opacity="0.6"/>
      <rect x="235" y="70" width="40" height="24" rx="3" fill="currentColor" opacity="0.3"/>
      <text x="255" y="86" text-anchor="middle" fill="currentColor" font-size="8">logits</text>
    </svg>`,
    softmax: `<svg viewBox="0 0 240 110" class="w-full max-w-sm mx-auto h-28 text-butter/90" aria-hidden="true">
      <rect x="20" y="20" width="80" height="70" rx="4" fill="currentColor" opacity="0.15"/>
      <text x="60" y="42" text-anchor="middle" fill="currentColor" font-size="9">logits</text>
      <text x="60" y="58" text-anchor="middle" fill="currentColor" font-size="8">(raw)</text>
      <path d="M100 55 L140 55" stroke="currentColor" stroke-width="1.5" opacity="0.7"/>
      <text x="120" y="50" text-anchor="middle" fill="currentColor" font-size="8">exp / Σ</text>
      <polygon points="136,50 146,55 136,60" fill="currentColor" opacity="0.7"/>
      <rect x="150" y="15" width="80" height="80" rx="4" fill="currentColor" opacity="0.2"/>
      <rect x="158" y="28" width="64" height="8" rx="2" fill="currentColor" opacity="0.5"/>
      <rect x="158" y="42" width="48" height="8" rx="2" fill="currentColor" opacity="0.4"/>
      <rect x="158" y="56" width="56" height="8" rx="2" fill="currentColor" opacity="0.6"/>
      <rect x="158" y="70" width="40" height="8" rx="2" fill="currentColor" opacity="0.3"/>
      <text x="190" y="100" text-anchor="middle" fill="currentColor" font-size="9">probs Σ=1</text>
    </svg>`,
    loss: `<svg viewBox="0 0 260 100" class="w-full max-w-sm mx-auto h-26 text-coral/90" aria-hidden="true">
      <rect x="20" y="25" width="100" height="50" rx="4" fill="currentColor" opacity="0.15"/>
      <text x="70" y="48" text-anchor="middle" fill="currentColor" font-size="9">p(target)</text>
      <text x="70" y="62" text-anchor="middle" fill="currentColor" font-size="8">probability</text>
      <path d="M120 50 L160 50" stroke="currentColor" stroke-width="1.5" opacity="0.7"/>
      <text x="140" y="42" text-anchor="middle" fill="currentColor" font-size="8">-log(·)</text>
      <polygon points="156,45 166,50 156,55" fill="currentColor" opacity="0.7"/>
      <rect x="170" y="25" width="70" height="50" rx="4" fill="currentColor" opacity="0.25"/>
      <text x="205" y="52" text-anchor="middle" fill="currentColor" font-size="10">L</text>
      <text x="205" y="68" text-anchor="middle" fill="currentColor" font-size="8">loss</text>
    </svg>`,
    backprop: `<svg viewBox="0 0 280 100" class="w-full max-w-sm mx-auto h-26 text-coral/90" aria-hidden="true">
      <rect x="200" y="30" width="60" height="40" rx="4" fill="currentColor" opacity="0.2"/>
      <text x="230" y="55" text-anchor="middle" fill="currentColor" font-size="9">L</text>
      <path d="M200 50 L150 50" stroke="currentColor" stroke-width="2" opacity="0.6"/>
      <polygon points="155,45 165,50 155,55" fill="currentColor" opacity="0.6"/>
      <rect x="100" y="30" width="50" height="40" rx="4" fill="currentColor" opacity="0.2"/>
      <text x="125" y="55" text-anchor="middle" fill="currentColor" font-size="8">∂L/∂</text>
      <path d="M100 50 L50 50" stroke="currentColor" stroke-width="2" opacity="0.6"/>
      <polygon points="55,45 65,50 55,55" fill="currentColor" opacity="0.6"/>
      <rect x="10" y="30" width="45" height="40" rx="4" fill="currentColor" opacity="0.25"/>
      <text x="32" y="55" text-anchor="middle" fill="currentColor" font-size="8">params</text>
      <text x="140" y="88" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.8">backward pass</text>
    </svg>`,
    update: `<svg viewBox="0 0 280 110" class="w-full max-w-sm mx-auto h-28 text-neon/80" aria-hidden="true">
      <rect x="20" y="25" width="70" height="35" rx="4" fill="currentColor" opacity="0.2"/>
      <text x="55" y="47" text-anchor="middle" fill="currentColor" font-size="9">param</text>
      <path d="M90 42 L125 42" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
      <polygon points="120,37 130,42 120,47" fill="currentColor" opacity="0.6"/>
      <rect x="130" y="15" width="120" height="55" rx="4" fill="currentColor" opacity="0.12"/>
      <text x="190" y="35" text-anchor="middle" fill="currentColor" font-size="8">Adam: m,v ← gradients</text>
      <text x="190" y="50" text-anchor="middle" fill="currentColor" font-size="8">param -= lr · m_hat / (√v_hat + ε)</text>
      <path d="M250 70 L250 90" stroke="currentColor" stroke-width="1" opacity="0.5"/>
      <polygon points="245,85 250,92 255,85" fill="currentColor" opacity="0.5"/>
      <rect x="200" y="90" width="100" height="18" rx="3" fill="currentColor" opacity="0.2"/>
      <text x="250" y="102" text-anchor="middle" fill="currentColor" font-size="8">updated</text>
    </svg>`,
  };
  const bodies: Record<string, string> = {
    dataset: `Names (one per line) are loaded and split into <strong>train</strong>, <strong>dev</strong>, and <strong>test</strong> sets. Training uses the train set to update weights; dev is used to monitor generalization (e.g. loss); test is held out for final evaluation. A typical split is ~80% train, ~10% dev, ~10% test.`,
    encode: `Each character is mapped to a unique integer <strong>token ID</strong>. A special <strong>BOS</strong> (beginning-of-sequence) token marks the start. The model only sees integers; the embedding layer turns them into continuous vectors.`,
    context: `A fixed <strong>block size</strong> defines how many previous tokens the model can attend to. For each position in the context, the task is to predict the <strong>next token</strong>. So context <code>[a,n,n]</code> predicts <code>a</code>; sliding one step gives <code>[n,n,a]</code> → predict next, and so on.`,
    forward: `Input tokens and positions are embedded and summed. Then: <strong>RMSNorm</strong> → <strong>Attention</strong> (Q, K, V, multi-head) → residual → RMSNorm → <strong>MLP</strong> (linear → ReLU → linear) → residual → <strong>lm_head</strong> → <strong>logits</strong> (one score per vocab token).`,
    softmax: `Logits are converted to a probability distribution over the next token: <code>p_i = exp(logit_i) / Σ exp(logit_j)</code>. All probabilities sum to 1. The model is trained to assign high probability to the correct next character.`,
    loss: `<strong>Cross-entropy</strong> measures how well the predicted distribution matches the target: <code>L = -log(p(target))</code>. Lower loss means the model assigned higher probability to the correct token. We minimize L by gradient descent.`,
    backprop: `<strong>Backpropagation</strong> computes the gradient of the loss with respect to every parameter. The chain rule flows gradients backward through the graph (attention, MLP, embeddings). The gradient norm indicates how large the updates will be.`,
    update: `<strong>Adam</strong> keeps per-parameter momentum (m) and variance (v). Parameters are updated with bias-corrected m and v: <code>param -= lr · m_hat / (√v_hat + ε)</code>. Learning rate is often decayed over steps (e.g. linear decay to 0).`,
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
          <button id="sampleBtn" class="rounded-lg border border-butter/50 px-4 py-2 text-sm font-semibold text-butter hover:bg-butter/10">Generate</button>
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

    <section class="grid gap-4 lg:grid-cols-2">
      <div class="panel p-4">
        <div class="flex items-center justify-between">
          <p class="panel-title">Sample Output</p>
          <span class="rounded-full border border-white/20 px-2 py-1 text-xs text-white/60">live</span>
        </div>
        <pre id="sample" class="mono mt-3 min-h-20 whitespace-pre-wrap rounded-xl border border-white/10 bg-black/25 p-3 text-sm text-neon"></pre>
      </div>

      <div class="panel p-4">
        <p class="panel-title">Probabilities (Current Step)</p>
        <div id="tokenBars" class="mt-3 space-y-2"></div>
      </div>
    </section>
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
  !breakdownTitleEl
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
const FLOW_FRAME_DELAY_MS = 90;

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

  breakdownTitleEl!.textContent = selectedIterationKey === 'live' ? 'Current Step Breakdown' : `Step ${selectedIterationKey.replace(/^step-/, '')} Breakdown`;

  const displayTrace = getDisplayTrace();
  traceContextEl!.textContent = `[${displayTrace.context.join(', ')}]`;
  traceTokensEl!.textContent = `[${displayTrace.contextTokens.join(', ')}]`;
  traceTargetEl!.textContent = `${displayTrace.targetToken} (${displayTrace.targetIndex})`;
  tracePredEl!.textContent = displayTrace.predictedToken;
  traceLrEl!.textContent = displayTrace.lr.toFixed(4);
  traceGradEl!.textContent = displayTrace.gradNorm.toFixed(6);

  progressBarEl!.style.width = `${Math.min(100, (trainer.step / trainer.maxSteps) * 100)}%`;
  statusPillEl!.textContent = running ? 'training' : trainer.step >= trainer.maxSteps ? 'completed' : 'idle';

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
    for (let stage = 0; stage < FLOW_STAGES.length; stage += 1) {
      if (!running) break;
      phaseCursor = stage;
      render();
      await new Promise((resolve) => setTimeout(resolve, FLOW_FRAME_DELAY_MS));
    }
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
