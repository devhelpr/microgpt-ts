import './style.css';
import mermaid from 'mermaid';
import { createMicroGptTrainer, n_layer, n_head, type StepTrace } from '../microgpt';
import { getLocale, setLocale, getLocaleCode, type LocaleCode, buildIllustrationSvg } from './i18n';

const MAX_ITERATION_HISTORY = 50;

let t = getLocale();

function cloneTrace(tr: StepTrace): StepTrace {
  return {
    context: tr.context.slice(),
    contextTokens: tr.contextTokens.slice(),
    targetIndex: tr.targetIndex,
    targetToken: tr.targetToken,
    predictedToken: tr.predictedToken,
    loss: tr.loss,
    lr: tr.lr,
    gradNorm: tr.gradNorm,
    top: tr.top.map((x) => ({ token: x.token, prob: x.prob })),
    tokenEmbedding: tr.tokenEmbedding.slice(),
    positionEmbedding: tr.positionEmbedding.slice(),
    summedEmbedding: tr.summedEmbedding.slice(),
    mlpOut: tr.mlpOut?.slice() ?? [],
    lnOut: tr.lnOut.slice(),
    logits: tr.logits.slice(),
    targetProb: tr.targetProb,
    attentionWeights: tr.attentionWeights?.slice(),
  };
}

const FLOW_STAGES = t.flowStages;
type FlowStage = (typeof FLOW_STAGES)[number];

type IterationSnapshot = {
  step: number;
  trace: StepTrace;
  trainLoss: number;
  devLoss: number;
  batchLoss: number;
};

// Initialise locale from persisted preference (if any) before building UI.
const storedLocale = (typeof window !== 'undefined' ? window.localStorage.getItem('locale') : null) as LocaleCode | null;
if (storedLocale === 'en' || storedLocale === 'nl') {
  setLocale(storedLocale);
}
t = getLocale();
const localeCode = getLocaleCode();

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error(t.errors.appRootNotFound);

function stepExplainerDialogHtml(stage: FlowStage): string {
  const illo = buildIllustrationSvg(stage.id, t.illustrationLabels);
  const body = t.explainerBodies[stage.id] ?? stage.description;
  return `<dialog id="dialog-${stage.id}" class="explainer-dialog rounded-2xl border border-white/15 bg-slate/95 p-0 shadow-2xl backdrop:bg-black/60" aria-labelledby="dialog-title-${stage.id}">
  <div class="explainer-dialog-content max-h-[85vh] overflow-y-auto p-6">
    <div class="flex items-start justify-between gap-4">
      <h2 id="dialog-title-${stage.id}" class="text-xl font-bold text-white">${stage.title}</h2>
      <button type="button" class="dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white" aria-label="${t.aria.close}">✕</button>
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
      <button type="button" class="flow-info-btn absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full border border-white/20 text-white/60 transition hover:border-neon/50 hover:bg-neon/15 hover:text-neon focus:outline-none focus:ring-2 focus:ring-neon/50" data-stage-index="${i}" aria-label="${t.aria.learnMoreAbout} ${s.title}">
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
      <p class="panel-title">${t.header.panelTitle}</p>
      <div class="mt-2 flex flex-wrap items-center justify-between gap-3">
        <h1 class="text-3xl font-bold leading-tight md:text-5xl">${t.header.title}</h1>
        <div class="flex items-center gap-2 rounded-2xl border border-white/15 bg-black/30 px-3 py-1.5 text-xs text-white/70">
          <span>${t.languages.label}</span>
          <select
            id="localeSelect"
            class="relative z-10 rounded-lg border border-white/20 bg-black/60 px-2 py-1 text-xs text-white/90 cursor-pointer pointer-events-auto focus:border-neon focus:outline-none"
          >
            <option value="en" ${localeCode === 'en' ? 'selected' : ''}>${t.languages.en}</option>
            <option value="nl" ${localeCode === 'nl' ? 'selected' : ''}>${t.languages.nl}</option>
          </select>
        </div>
      </div>
      <p class="mt-4 max-w-3xl text-sm text-white/75 md:text-base">${t.header.description}</p>
    </header>

    <section class="mb-4">
      <div class="panel p-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <p class="panel-title mb-0">${t.generatedName.panelTitle}</p>
          <button id="sampleBtn" class="rounded-lg border border-butter/50 px-4 py-2 text-sm font-semibold text-butter hover:bg-butter/10">${t.generatedName.generate}</button>
        </div>
        <pre id="sample" class="mono mt-3 min-h-14 whitespace-pre-wrap rounded-xl border border-white/10 bg-black/25 p-3 text-lg text-neon"></pre>
      </div>
    </section>

    <section class="mb-4 grid gap-4 lg:grid-cols-3">
      <div class="panel p-4 lg:col-span-1">
        <p class="panel-title">${t.controls.panelTitle}</p>
        <label class="mt-4 block text-sm text-white/70">${t.controls.datasetLabel}</label>
        <textarea id="dataset" class="mono mt-2 h-44 w-full rounded-xl border border-white/15 bg-black/30 p-3 text-sm text-white/90 focus:border-neon focus:outline-none">${t.defaultDataset}</textarea>

        <div class="mt-4 grid grid-cols-2 gap-3">
          <label class="text-sm text-white/70">${t.controls.maxSteps}
            <input id="maxSteps" type="number" value="1200" min="50" step="50" class="mono mt-1 w-full rounded-lg border border-white/15 bg-black/30 p-2 text-sm" />
          </label>
          <label class="text-sm text-white/70">${t.controls.evalEvery}
            <input id="evalEvery" type="number" value="24" min="5" step="1" class="mono mt-1 w-full rounded-lg border border-white/15 bg-black/30 p-2 text-sm" />
          </label>
        </div>

        <div class="mt-4 flex flex-wrap gap-2">
          <button id="startBtn" class="rounded-lg bg-neon px-4 py-2 text-sm font-semibold text-black transition hover:brightness-110">${t.controls.start}</button>
          <button id="pauseBtn" class="rounded-lg border border-white/20 px-4 py-2 text-sm font-semibold text-white/90 hover:bg-white/10">${t.controls.pause}</button>
          <button id="resetBtn" class="rounded-lg border border-coral/50 px-4 py-2 text-sm font-semibold text-coral hover:bg-coral/10">${t.controls.reset}</button>
        </div>

        <div class="mt-4 space-y-2 text-xs text-white/60">
          <div id="dataStats"></div>
          <div>${t.controls.tipReset}</div>
        </div>
      </div>

      <div class="panel p-4 lg:col-span-2">
        <div class="flex items-center justify-between">
          <p class="panel-title">${t.trainingDynamics.panelTitle}</p>
          <div class="flex items-center gap-2">
            <button type="button" id="trainingDynamicsInfoBtn" class="rounded-lg border border-white/20 p-1.5 text-white/60 hover:bg-white/10 hover:text-white transition focus:outline-none focus:ring-2 focus:ring-neon/50" aria-label="${t.trainingDynamics.explainBtn}">
              <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </button>
            <div id="statusPill" class="rounded-full border border-neon/40 bg-neon/10 px-3 py-1 text-xs text-neon">${t.trainingDynamics.statusIdle}</div>
          </div>
        </div>

        <div class="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">${t.trainingDynamics.step}</div><div id="step" class="mono mt-1 text-xl">0</div></div>
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">${t.trainingDynamics.batchLoss}</div><div id="batchLoss" class="mono mt-1 text-xl">0.0000</div></div>
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">${t.trainingDynamics.trainLoss}</div><div id="trainLoss" class="mono mt-1 text-xl">0.0000</div></div>
          <div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">${t.trainingDynamics.devLoss}</div><div id="devLoss" class="mono mt-1 text-xl">0.0000</div></div>
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
          <p class="panel-title">${t.algorithm.panelTitle}</p>
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs text-white/50">${t.algorithm.review}</span>
            <select id="iterationSelect" class="mono rounded-lg border border-white/15 bg-black/30 px-2 py-1.5 text-xs text-white/90 focus:border-neon focus:outline-none">
              <option value="live">${t.algorithm.live}</option>
            </select>
            <span id="iterationStepLabel" class="text-xs text-white/50"></span>
          </div>
        </div>
        <div class="mt-2 flex flex-wrap items-center gap-2">
          <button type="button" id="showTransformerDiagramBtn" class="rounded-lg border border-neon/50 px-3 py-1.5 text-xs font-semibold text-neon transition hover:bg-neon/15 focus:outline-none focus:ring-2 focus:ring-neon/50">${t.algorithm.viewTransformerDiagram}</button>
        </div>
        <div id="flowGrid" class="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-4">${flowNodesHtml}</div>
        <div id="flowDetail" class="mt-3 rounded-lg border border-neon/25 bg-neon/10 p-3 text-sm text-neon"></div>
        <div id="flowVisual" class="mt-3 rounded-xl border border-white/10 bg-black/30 p-3"></div>
      </div>

      <div class="panel p-4">
        <p class="panel-title" id="breakdownTitle">${t.breakdown.panelTitle}</p>
        <div class="mt-3 space-y-2 text-sm">
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.contextIds}</span> <span id="traceContext" class="mono text-white/90"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.contextTokens}</span> <span id="traceTokens" class="mono text-white/90"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.target}</span> <span id="traceTarget" class="mono text-butter"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.predicted}</span> <span id="tracePred" class="mono text-neon"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.learningRate}</span> <span id="traceLr" class="mono text-white/90"></span></div>
          <div class="rounded-lg border border-white/10 bg-black/25 p-2"><span class="text-white/60">${t.breakdown.gradientNorm}</span> <span id="traceGrad" class="mono text-coral"></span></div>
        </div>
      </div>
    </section>

    <section class="grid gap-4 lg:grid-cols-1">
      <div class="panel p-4">
        <p class="panel-title">${t.probabilities.panelTitle}</p>
        <div id="tokenBars" class="mt-3 space-y-2"></div>
      </div>
    </section>
    <dialog id="dialog-training-dynamics" class="rounded-2xl border border-white/15 bg-slate/98 p-0 shadow-2xl backdrop:bg-black/70 max-w-lg" aria-labelledby="dialog-training-dynamics-title" aria-modal="true">
      <div class="max-h-[90vh] overflow-y-auto p-6">
        <div class="flex items-start justify-between gap-4">
          <h2 id="dialog-training-dynamics-title" class="text-xl font-bold text-white">${t.dialogs.trainingDynamics.title}</h2>
          <button type="button" class="dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white transition" aria-label="${t.aria.close}">✕</button>
        </div>
        <div class="mt-4 space-y-4 text-sm text-white/85 leading-relaxed [&_strong]:text-neon">
          <p><strong>${t.dialogs.trainingDynamics.whatGraphShows}</strong><br/>${t.dialogs.trainingDynamics.whatGraphShowsBody}</p>
          <p><strong>${t.dialogs.trainingDynamics.spikesMean}</strong><br/>${t.dialogs.trainingDynamics.spikesMeanBody}</p>
          <p><strong>${t.dialogs.trainingDynamics.numbersAboveGraph}</strong><br/>${t.dialogs.trainingDynamics.numbersAboveGraphBody}</p>
          <p class="text-xs text-white/55">${t.dialogs.trainingDynamics.lowerLossNote}</p>
        </div>
      </div>
    </dialog>
    <dialog id="dialog-transformer" class="transformer-dialog rounded-2xl border border-white/15 bg-slate/98 p-0 shadow-2xl backdrop:bg-black/70" aria-labelledby="dialog-transformer-title" aria-modal="true">
      <div class="transformer-dialog-content max-h-[90vh] overflow-y-auto p-6">
        <div class="flex items-start justify-between gap-4">
          <h2 id="dialog-transformer-title" class="text-xl font-bold text-white">${t.dialogs.transformer.title}</h2>
          <button type="button" class="dialog-close transformer-dialog-close rounded-lg border border-white/20 p-2 text-white/80 hover:bg-white/10 hover:text-white transition" aria-label="${t.aria.close}">✕</button>
        </div>
        <p class="mt-2 text-sm text-white/75 leading-relaxed">${t.dialogs.transformer.intro}</p>
        <div id="transformerDiagramContainer" class="mt-6 flex justify-center rounded-xl border border-white/10 bg-black/30 p-6 min-h-[420px]"></div>
        <p class="mt-4 text-xs text-white/55">${t.dialogs.transformer.oneForwardPassNote}</p>
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
const trainingDynamicsInfoBtn = document.querySelector<HTMLButtonElement>('#trainingDynamicsInfoBtn');
const dialogTrainingDynamics = document.querySelector<HTMLDialogElement>('#dialog-training-dynamics');
const localeSelectEl = document.querySelector<HTMLSelectElement>('#localeSelect');

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
  !dialogTransformer ||
  !localeSelectEl
) {
  throw new Error(t.errors.missingUiElement);
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
  iterationSelectEl.innerHTML = `<option value="live">${t.algorithm.live}</option>`;
  const start = Math.max(0, iterationHistory.length - MAX_ITERATION_HISTORY);
  for (let i = iterationHistory.length - 1; i >= start; i--) {
    const s = iterationHistory[i];
    const opt = document.createElement('option');
    opt.value = `step-${s.step}`;
    opt.textContent = t.iterationSelect.stepLabel.replace('{n}', String(s.step));
    iterationSelectEl.appendChild(opt);
  }
  iterationSelectEl.value = selectedIterationKey === 'live' || !iterationHistory.some((s) => `step-${s.step}` === selectedIterationKey) ? 'live' : selectedIterationKey;
  const snap = selectedIterationKey !== 'live' ? iterationHistory.find((s) => `step-${s.step}` === selectedIterationKey) : null;
  iterationStepLabelEl.textContent = snap ? t.iterationSelect.trainDevLabel.replace('{trainLoss}', snap.trainLoss.toFixed(4)).replace('{devLoss}', snap.devLoss.toFixed(4)) : '';
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
  ctx.fillText(`${t.chart.min} ${min.toFixed(3)}`, 12, height - 8);
  ctx.fillText(`${t.chart.max} ${max.toFixed(3)}`, width - 82, height - 8);
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
      <div class="mb-2 text-xs text-white/60">${label} ${t.vectorBars.first8Dims}</div>
      <div class="space-y-1">${bars}</div>
    </div>
  `;
}

function stageVisualHtml(stageId: string, trace: StepTrace): string {
  const tr = trace;
  if (stageId === 'dataset') {
    const total = Math.max(1, trainer.trainSize + trainer.devSize + trainer.testSize);
    const trainW = Math.round((trainer.trainSize / total) * 100);
    const devW = Math.round((trainer.devSize / total) * 100);
    const testW = Math.round((trainer.testSize / total) * 100);
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">${t.stageVisual.dataset.splitDescription}</div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mb-1 text-xs text-white/55">${t.stageVisual.dataset.train} ${trainer.trainSize}</div>
          <div class="h-2 rounded bg-white/10"><div class="h-full rounded bg-neon/70" style="width:${trainW}%"></div></div>
        </div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mb-1 text-xs text-white/55">${t.stageVisual.dataset.dev} ${trainer.devSize}</div>
          <div class="h-2 rounded bg-white/10"><div class="h-full rounded bg-butter/70" style="width:${devW}%"></div></div>
        </div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mb-1 text-xs text-white/55">${t.stageVisual.dataset.test} ${trainer.testSize}</div>
          <div class="h-2 rounded bg-white/10"><div class="h-full rounded bg-coral/70" style="width:${testW}%"></div></div>
        </div>
      </div>
    `;
  }

  if (stageId === 'encode') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">${t.stageVisual.encode.description}</div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mono text-xs text-white/60">${t.stageVisual.encode.contextTokensToIds}</div>
          <div class="mt-2 flex flex-wrap gap-2">
            ${tr.contextTokens
              .map((tok, i) => `<span class="mono rounded border border-white/15 px-2 py-1 text-xs">${tok} -> ${tr.context[i]}</span>`)
              .join('')}
          </div>
        </div>
        <div class="mono text-xs text-butter">${t.stageVisual.encode.target} ${tr.targetToken} -> ${tr.targetIndex}</div>
      </div>
    `;
  }

  if (stageId === 'context') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">${t.stageVisual.context.description}</div>
        <div class="flex items-center gap-2">
          ${tr.contextTokens.map((tok) => `<div class="mono rounded-lg border border-white/15 bg-black/25 px-3 py-2 text-sm">${tok}</div>`).join('')}
          <div class="text-neon">-></div>
          <div class="mono rounded-lg border border-butter/30 bg-butter/10 px-3 py-2 text-sm text-butter">${tr.targetToken}</div>
        </div>
      </div>
    `;
  }

  if (stageId === 'forward') {
    const attnHtml =
      tr.attentionWeights && tr.attentionWeights.length > 0
        ? `
      <div class="rounded-lg border border-white/10 bg-black/25 p-2">
        <div class="mb-2 text-xs text-white/60">${t.stageVisual.forward.attentionHead0}</div>
        <div class="flex flex-wrap gap-1">
          ${tr.attentionWeights!
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
        ${vectorBars(t.stageVisual.forward.tokenEmbedding, tr.tokenEmbedding, 'bg-neon/70')}
        ${vectorBars(t.stageVisual.forward.positionEmbedding, tr.positionEmbedding, 'bg-butter/70')}
        ${vectorBars(t.stageVisual.forward.sumEmbedding, tr.summedEmbedding, 'bg-cyan-300/70')}
        ${vectorBars(t.stageVisual.forward.preHeadAfterMlp, tr.lnOut, 'bg-indigo-300/70')}
        ${attnHtml}
      </div>
    `;
  }

  if (stageId === 'softmax') {
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">${t.stageVisual.softmax.description}</div>
        ${tr.top
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
        <div class="text-white/70">${t.stageVisual.loss.description}</div>
        <div class="mono rounded-lg border border-white/10 bg-black/25 p-2 text-xs">
          L = -log(p(target)) = -log(${Math.max(1e-9, tr.targetProb).toFixed(4)}) = ${tr.loss.toFixed(4)}
        </div>
        <div class="text-xs text-white/60">${t.stageVisual.loss.predictedTarget.replace('{pred}', tr.predictedToken).replace('{target}', tr.targetToken)}</div>
      </div>
    `;
  }

  if (stageId === 'backprop') {
    const pct = Math.min(100, Math.round(tr.gradNorm * 200));
    return `
      <div class="space-y-2 text-sm">
        <div class="text-white/70">${t.stageVisual.backprop.description}</div>
        <div class="rounded-lg border border-white/10 bg-black/25 p-2">
          <div class="mono text-xs text-white/60">${t.stageVisual.backprop.gradientNorm} ${tr.gradNorm.toFixed(6)}</div>
          <div class="mt-2 h-2 rounded bg-white/10"><div class="h-full rounded bg-coral/70" style="width:${pct}%"></div></div>
        </div>
      </div>
    `;
  }

  const delta = tr.lr * tr.gradNorm;
  return `
    <div class="space-y-2 text-sm">
      <div class="text-white/70">${t.stageVisual.update.description}</div>
      <div class="mono rounded-lg border border-white/10 bg-black/25 p-2 text-xs">
        ${t.stageVisual.update.formula}
      </div>
      <div class="mono text-xs text-white/70">${t.stageVisual.update.lrStepMagnitude.replace('{lr}', tr.lr.toFixed(4)).replace('{delta}', delta.toExponential(2))}</div>
      ${vectorBars(t.stageVisual.forward.preHeadBeforeLmHead, tr.lnOut, 'bg-emerald-300/70')}
    </div>
  `;
}

/** Plain-English Mermaid flowchart so the diagram is understandable without ML jargon. */
function getTransformerMermaidCode(): string {
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
  sampleEl!.textContent = manualSample || trainer.sample || t.samplePlaceholder;

  dataStatsEl!.textContent = t.dataStatsTemplate.replace('{words}', String(datasetEl!.value.split(/\r?\n/).filter(Boolean).length)).replace('{vocab}', String(trainer.vocabSize)).replace('{train}', String(trainer.trainSize)).replace('{dev}', String(trainer.devSize)).replace('{test}', String(trainer.testSize));

  progressBarEl!.style.width = `${Math.min(100, (trainer.step / trainer.maxSteps) * 100)}%`;
  statusPillEl!.textContent = running ? t.trainingDynamics.statusTraining : trainer.step >= trainer.maxSteps ? t.trainingDynamics.statusCompleted : t.trainingDynamics.statusIdle;

  const showStepDetails = !running;
  const breakdownPanel = breakdownTitleEl!.closest('.panel');
  const flowSection = flowGridEl!.closest('.panel');
  const tokenBarsPanel = tokenBarsEl!.closest('.panel');
  if (breakdownPanel) breakdownPanel.classList.toggle('hidden', !showStepDetails);
  if (flowSection) flowSection.querySelector('#flowDetail')?.classList.toggle('hidden', !showStepDetails);
  if (flowSection) flowSection.querySelector('#flowVisual')?.classList.toggle('hidden', !showStepDetails);
  if (tokenBarsPanel) tokenBarsPanel.classList.toggle('hidden', !showStepDetails);

  if (showStepDetails) {
    breakdownTitleEl!.textContent = selectedIterationKey === 'live' ? t.breakdown.panelTitle : t.breakdown.stepBreakdown.replace('{n}', selectedIterationKey.replace(/^step-/, ''));

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

localeSelectEl.addEventListener('change', (e) => {
  const select = e.target as HTMLSelectElement;
  const code = select.value as LocaleCode;
  if (code !== 'en' && code !== 'nl') return;
  window.localStorage.setItem('locale', code);
  setLocale(code);
  window.location.reload();
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

trainingDynamicsInfoBtn?.addEventListener('click', () => {
  dialogTrainingDynamics?.showModal();
});

showTransformerDiagramBtn!.addEventListener('click', async () => {
  const container = document.getElementById('transformerDiagramContainer');
  if (!container) return;
  container.innerHTML = `<div class="mermaid">${getTransformerMermaidCode()}</div>`;
  dialogTransformer!.showModal();
  try {
    await mermaid.run({ nodes: container.querySelectorAll('.mermaid') });
  } catch (err) {
    container.innerHTML = `<p class="text-sm text-coral">${t.errors.diagramRenderFailed}</p>`;
  }
});

iterationSelectEl!.addEventListener('change', () => {
  selectedIterationKey = iterationSelectEl!.value;
  const snap = selectedIterationKey !== 'live' ? iterationHistory.find((s) => `step-${s.step}` === selectedIterationKey) : null;
  iterationStepLabelEl!.textContent = snap ? t.iterationSelect.trainDevLabel.replace('{trainLoss}', snap.trainLoss.toFixed(4)).replace('{devLoss}', snap.devLoss.toFixed(4)) : '';
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
