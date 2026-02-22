import './style.css';
import { createMicroGptTrainer } from '../microgpt';

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

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('App root not found');

const flowNodesHtml = FLOW_STAGES.map(
  (s, i) => `
    <article id="flow-${s.id}" class="flow-node rounded-xl border border-white/10 bg-black/25 p-3">
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
          <span class="rounded-full border border-white/20 px-2 py-1 text-xs text-white/60">live stage</span>
        </div>
        <div id="flowGrid" class="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">${flowNodesHtml}</div>
        <div id="flowDetail" class="mt-3 rounded-lg border border-neon/25 bg-neon/10 p-3 text-sm text-neon"></div>
        <div id="flowVisual" class="mt-3 rounded-xl border border-white/10 bg-black/30 p-3"></div>
      </div>

      <div class="panel p-4">
        <p class="panel-title">Current Step Breakdown</p>
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
const dataStatsEl = document.querySelector<HTMLElement>('#dataStats');
const traceContextEl = document.querySelector<HTMLElement>('#traceContext');
const traceTokensEl = document.querySelector<HTMLElement>('#traceTokens');
const traceTargetEl = document.querySelector<HTMLElement>('#traceTarget');
const tracePredEl = document.querySelector<HTMLElement>('#tracePred');
const traceLrEl = document.querySelector<HTMLElement>('#traceLr');
const traceGradEl = document.querySelector<HTMLElement>('#traceGrad');

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
  !dataStatsEl ||
  !traceContextEl ||
  !traceTokensEl ||
  !traceTargetEl ||
  !tracePredEl ||
  !traceLrEl ||
  !traceGradEl
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

function stageVisualHtml(stageId: string): string {
  const t = trainer.lastTrace;
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

function renderFlow(activeIdx: number): void {
  FLOW_STAGES.forEach((stage, i) => {
    const el = document.querySelector<HTMLElement>(`#flow-${stage.id}`);
    if (!el) return;
    el.classList.toggle('active', i === activeIdx);
  });
  const active = FLOW_STAGES[activeIdx] ?? FLOW_STAGES[0];
  flowDetailEl!.textContent = `${active.title}: ${active.description}`;
  flowVisualEl!.innerHTML = stageVisualHtml(active.id);
}

function render(): void {
  stepEl!.textContent = `${trainer.step}/${trainer.maxSteps}`;
  batchLossEl!.textContent = trainer.latestBatchLoss.toFixed(4);
  trainLossEl!.textContent = trainer.trainLoss.toFixed(4);
  devLossEl!.textContent = trainer.devLoss.toFixed(4);
  sampleEl!.textContent = manualSample || trainer.sample || '...';

  dataStatsEl!.textContent = `words=${datasetEl!.value.split(/\r?\n/).filter(Boolean).length} | vocab=${trainer.vocabSize} | train/dev/test=${trainer.trainSize}/${trainer.devSize}/${trainer.testSize}`;

  traceContextEl!.textContent = `[${trainer.lastTrace.context.join(', ')}]`;
  traceTokensEl!.textContent = `[${trainer.lastTrace.contextTokens.join(', ')}]`;
  traceTargetEl!.textContent = `${trainer.lastTrace.targetToken} (${trainer.lastTrace.targetIndex})`;
  tracePredEl!.textContent = trainer.lastTrace.predictedToken;
  traceLrEl!.textContent = trainer.lastTrace.lr.toFixed(4);
  traceGradEl!.textContent = trainer.lastTrace.gradNorm.toFixed(6);

  progressBarEl!.style.width = `${Math.min(100, (trainer.step / trainer.maxSteps) * 100)}%`;
  statusPillEl!.textContent = running ? 'training' : trainer.step >= trainer.maxSteps ? 'completed' : 'idle';

  tokenBarsEl!.innerHTML = trainer.lastTrace.top
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

  renderFlow(running ? phaseCursor : trainer.step > 0 ? FLOW_STAGES.length - 1 : 0);
  drawLossChart(trainer.losses.slice(-300));
}

async function trainLoop(): Promise<void> {
  if (running) return;
  running = true;
  render();

  while (running && trainer.step < trainer.maxSteps) {
    trainer.trainStep();
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
  trainer = createMicroGptTrainer(datasetEl!.value, {
    blockSize: 16,
    nEmbd: 16,
    maxSteps: Math.max(50, Number(maxStepsEl!.value) || 1200),
    evalEvery: Math.max(5, Number(evalEveryEl!.value) || 24),
    seed: 1337,
  });
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

window.addEventListener('resize', () => render());

render();
