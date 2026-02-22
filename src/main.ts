import './style.css';

type Split = { X: number[][]; Y: number[] };

type TrainerConfig = {
  datasetText: string;
  blockSize: number;
  nEmbd: number;
  maxSteps: number;
  evalEvery: number;
  seed: number;
};

type StepTrace = {
  context: number[];
  contextTokens: string[];
  targetIndex: number;
  targetToken: string;
  predictedToken: string;
  loss: number;
  lr: number;
  gradNorm: number;
  top: Array<{ token: string; prob: number }>;
  tokenEmbedding: number[];
  positionEmbedding: number[];
  summedEmbedding: number[];
  mlpOut: number[];
  lnOut: number[];
  logits: number[];
  targetProb: number;
};

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
    description: 'Map characters to integer token IDs.',
  },
  {
    id: 'context',
    title: '3. Context Window',
    description: 'Build fixed-size context, e.g. [.,.,a] -> next token.',
  },
  {
    id: 'forward',
    title: '4. Forward Pass',
    description: 'Embedding + MLP + LayerNorm + Linear head produce logits.',
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
    description: 'Apply SGD step: param = param - lr * grad.',
  },
];

class Value {
  data: number;
  grad: number;
  private prev: Set<Value>;
  private backwardFn: () => void;

  constructor(data: number, prev: Value[] = [], backwardFn: () => void = () => {}) {
    this.data = data;
    this.grad = 0;
    this.prev = new Set(prev);
    this.backwardFn = backwardFn;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], () => {
      this.grad += out.grad;
      o.grad += out.grad;
    });
    return out;
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.mul(-1));
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    });
    return out;
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  pow(power: number): Value {
    const out = new Value(Math.pow(this.data, power), [this], () => {
      this.grad += power * Math.pow(this.data, power - 1) * out.grad;
    });
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], () => {
      this.grad += (1 - t * t) * out.grad;
    });
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], () => {
      this.grad += e * out.grad;
    });
    return out;
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], () => {
      this.grad += (1 / this.data) * out.grad;
    });
    return out;
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v.prev) build(p);
      topo.push(v);
    };

    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i -= 1) {
      topo[i].backwardFn();
    }
  }
}

function makeRng(seed = 1337): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}

function randn(rng: () => number, mean = 0, std = 1): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z0;
}

class Neuron {
  w: Value[];
  b: Value;
  nonlin: boolean;

  constructor(nin: number, rng: () => number, nonlin = true) {
    this.w = Array.from({ length: nin }, () => new Value(randn(rng, 0, 1)));
    this.b = new Value(0);
    this.nonlin = nonlin;
  }

  call(x: Value[]): Value {
    let act = this.b;
    for (let i = 0; i < x.length; i += 1) {
      act = act.add(this.w[i].mul(x[i]));
    }
    return this.nonlin ? act.tanh() : act;
  }

  callNumeric(x: number[]): number {
    let act = this.b.data;
    for (let i = 0; i < x.length; i += 1) {
      act += this.w[i].data * x[i];
    }
    return this.nonlin ? Math.tanh(act) : act;
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

class Layer {
  neurons: Neuron[];

  constructor(nin: number, nout: number, rng: () => number, nonlin = true) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, rng, nonlin));
  }

  call(x: Value[]): Value[] {
    return this.neurons.map((n) => n.call(x));
  }

  callNumeric(x: number[]): number[] {
    return this.neurons.map((n) => n.callNumeric(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

class MLP {
  layers: Layer[];

  constructor(nin: number, nouts: number[], rng: () => number) {
    const sz = [nin, ...nouts];
    this.layers = [];
    for (let i = 0; i < nouts.length; i += 1) {
      this.layers.push(new Layer(sz[i], sz[i + 1], rng, i !== nouts.length - 1));
    }
  }

  call(x: Value[]): Value[] {
    let out = x;
    for (const l of this.layers) out = l.call(out);
    return out;
  }

  callNumeric(x: number[]): number[] {
    let out = x;
    for (const l of this.layers) out = l.callNumeric(out);
    return out;
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

class LayerNorm {
  gamma: Value[];
  beta: Value[];
  eps: number;

  constructor(dim: number, eps = 1e-5) {
    this.gamma = Array.from({ length: dim }, () => new Value(1));
    this.beta = Array.from({ length: dim }, () => new Value(0));
    this.eps = eps;
  }

  call(x: Value[]): Value[] {
    const n = x.length;
    let mean = new Value(0);
    for (const xi of x) mean = mean.add(xi);
    mean = mean.div(n);

    let variance = new Value(0);
    for (const xi of x) variance = variance.add(xi.sub(mean).pow(2));
    variance = variance.div(n);

    const std = variance.add(this.eps).pow(0.5);
    return x.map((xi, i) => xi.sub(mean).div(std).mul(this.gamma[i]).add(this.beta[i]));
  }

  callNumeric(x: number[]): number[] {
    const n = x.length;
    const mean = x.reduce((a, b) => a + b, 0) / n;
    const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance + this.eps);
    return x.map((xi, i) => ((xi - mean) / std) * this.gamma[i].data + this.beta[i].data);
  }

  parameters(): Value[] {
    return [...this.gamma, ...this.beta];
  }
}

class Linear {
  w: Value[][];
  b: Value[];

  constructor(fanIn: number, fanOut: number, rng: () => number) {
    const std = 1 / Math.sqrt(fanIn);
    this.w = Array.from({ length: fanOut }, () =>
      Array.from({ length: fanIn }, () => new Value(randn(rng, 0, std))),
    );
    this.b = Array.from({ length: fanOut }, () => new Value(0));
  }

  call(x: Value[]): Value[] {
    return this.w.map((row, o) => {
      let v = this.b[o];
      for (let i = 0; i < x.length; i += 1) v = v.add(row[i].mul(x[i]));
      return v;
    });
  }

  callNumeric(x: number[]): number[] {
    return this.w.map((row, o) => {
      let v = this.b[o].data;
      for (let i = 0; i < x.length; i += 1) v += row[i].data * x[i];
      return v;
    });
  }

  parameters(): Value[] {
    return [...this.w.flat(), ...this.b];
  }
}

class BigramLanguageModel {
  wte: number[][];
  wpe: number[][];
  mlp: MLP;
  ln: LayerNorm;
  lmHead: Linear;

  constructor(vocabSize: number, blockSize: number, nEmbd: number, rng: () => number) {
    this.wte = Array.from({ length: vocabSize }, () =>
      Array.from({ length: nEmbd }, () => randn(rng, 0, 0.2)),
    );
    this.wpe = Array.from({ length: blockSize }, () =>
      Array.from({ length: nEmbd }, () => randn(rng, 0, 0.2)),
    );
    this.mlp = new MLP(nEmbd, [nEmbd], rng);
    this.ln = new LayerNorm(nEmbd);
    this.lmHead = new Linear(nEmbd, vocabSize, rng);
  }

  call(idx: number[]): Value[] {
    let last: Value[] = [];
    for (let pos = 0; pos < idx.length; pos += 1) {
      const x = this.wte[idx[pos]].map((v, i) => new Value(v + this.wpe[pos][i]));
      last = this.ln.call(this.mlp.call(x));
    }
    return this.lmHead.call(last);
  }

  debugForward(idx: number[]): {
    tokenEmbedding: number[];
    positionEmbedding: number[];
    summedEmbedding: number[];
    mlpOut: number[];
    lnOut: number[];
    logits: number[];
  } {
    const pos = idx.length - 1;
    const tokenEmbedding = [...this.wte[idx[pos]]];
    const positionEmbedding = [...this.wpe[pos]];
    const summedEmbedding = tokenEmbedding.map((v, i) => v + positionEmbedding[i]);
    const mlpOut = this.mlp.callNumeric(summedEmbedding);
    const lnOut = this.ln.callNumeric(mlpOut);
    const logits = this.lmHead.callNumeric(lnOut);
    return { tokenEmbedding, positionEmbedding, summedEmbedding, mlpOut, lnOut, logits };
  }

  parameters(): Value[] {
    return [...this.mlp.parameters(), ...this.ln.parameters(), ...this.lmHead.parameters()];
  }
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function crossEntropy(logits: Value[], target: number): Value {
  const exps = logits.map((v) => v.exp());
  let sumExp = new Value(0);
  for (const e of exps) sumExp = sumExp.add(e);
  const probs = exps.map((e) => e.div(sumExp));
  return probs[target].log().mul(-1);
}

function sampleCategorical(probs: number[], rng: () => number): number {
  const r = rng();
  let c = 0;
  for (let i = 0; i < probs.length; i += 1) {
    c += probs[i];
    if (r <= c) return i;
  }
  return probs.length - 1;
}

function topKIndices(arr: number[], k: number): number[] {
  return [...arr.keys()].sort((a, b) => arr[b] - arr[a]).slice(0, k);
}

class Trainer {
  private rng: () => number;
  private model: BigramLanguageModel;
  private parameters: Value[];
  private words: string[];
  private train: Split;
  private dev: Split;
  private test: Split;
  private stoi: Map<string, number>;
  private itos: Map<number, string>;

  step = 0;
  maxSteps: number;
  evalEvery: number;
  losses: number[] = [];
  latestBatchLoss = 0;
  trainLoss = 0;
  devLoss = 0;
  testLoss = 0;
  sample = '';
  topTokens: Array<{ token: string; prob: number }> = [];
  vocabSize = 0;
  trainSize = 0;
  devSize = 0;
  testSize = 0;
  lastTrace: StepTrace = {
    context: [],
    contextTokens: [],
    targetIndex: 0,
    targetToken: '.',
    predictedToken: '.',
    loss: 0,
    lr: 0,
    gradNorm: 0,
    top: [],
    tokenEmbedding: [],
    positionEmbedding: [],
    summedEmbedding: [],
    mlpOut: [],
    lnOut: [],
    logits: [],
    targetProb: 0,
  };

  constructor(cfg: TrainerConfig) {
    this.rng = makeRng(cfg.seed);
    this.maxSteps = cfg.maxSteps;
    this.evalEvery = cfg.evalEvery;

    this.words = cfg.datasetText
      .split(/\r?\n/)
      .map((w) => w.trim())
      .filter((w) => w.length > 0);

    if (this.words.length < 2) {
      this.words = ['anna', 'bob', 'carla', 'diana', 'elias', 'frank'];
    }

    const chars = Array.from(new Set(this.words.join(''))).sort();
    this.stoi = new Map<string, number>();
    this.itos = new Map<number, string>();

    chars.forEach((ch, i) => {
      this.stoi.set(ch, i + 1);
      this.itos.set(i + 1, ch);
    });

    this.stoi.set('.', 0);
    this.itos.set(0, '.');

    this.vocabSize = this.stoi.size;
    this.model = new BigramLanguageModel(this.vocabSize, cfg.blockSize, cfg.nEmbd, this.rng);
    this.parameters = this.model.parameters();

    const shuffled = this.shuffle(this.words);
    const n1 = Math.max(1, Math.floor(shuffled.length * 0.8));
    const n2 = Math.max(n1 + 1, Math.floor(shuffled.length * 0.9));

    const trainWords = shuffled.slice(0, n1);
    const devWords = shuffled.slice(n1, Math.min(n2, shuffled.length));
    const testWords = shuffled.slice(Math.min(n2, shuffled.length));

    this.train = this.buildDataset(trainWords, cfg.blockSize);
    this.dev = this.buildDataset(devWords.length > 0 ? devWords : trainWords, cfg.blockSize);
    this.test = this.buildDataset(testWords.length > 0 ? testWords : trainWords, cfg.blockSize);

    this.trainSize = this.train.X.length;
    this.devSize = this.dev.X.length;
    this.testSize = this.test.X.length;

    this.evaluate();
    this.seedInitialTrace();
  }

  private shuffle<T>(arr: T[]): T[] {
    const out = [...arr];
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = Math.floor(this.rng() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  private encode(s: string): number[] {
    return s.split('').map((c) => this.stoi.get(c) ?? 0);
  }

  private tokenOf(ix: number): string {
    return this.itos.get(ix) ?? '?';
  }

  private decode(ids: number[]): string {
    return ids.map((i) => this.tokenOf(i)).join('');
  }

  private buildDataset(words: string[], blockSize: number): Split {
    const X: number[][] = [];
    const Y: number[] = [];

    for (const w of words) {
      let context = Array(blockSize).fill(0);
      for (const ix of this.encode(`${w}.`)) {
        X.push([...context]);
        Y.push(ix);
        context = [...context.slice(1), ix];
      }
    }

    return { X, Y };
  }

  private sampleExample(split: Split): { x: number[]; y: number } {
    const i = Math.floor(this.rng() * split.X.length);
    return { x: split.X[i], y: split.Y[i] };
  }

  private estimateLoss(split: Split, batches = 96): number {
    const n = Math.min(split.X.length, batches);
    let total = 0;
    for (let i = 0; i < n; i += 1) {
      const { x, y } = this.sampleExample(split);
      total += crossEntropy(this.model.call(x), y).data;
    }
    return total / n;
  }

  private seedInitialTrace(): void {
    const { x, y } = this.sampleExample(this.train);
    const logits = this.model.call(x);
    const debug = this.model.debugForward(x);
    const probs = softmax(debug.logits);
    const maxIdx = probs.reduce((best, p, i) => (p > probs[best] ? i : best), 0);
    this.lastTrace = {
      context: [...x],
      contextTokens: x.map((i) => this.tokenOf(i)),
      targetIndex: y,
      targetToken: this.tokenOf(y),
      predictedToken: this.tokenOf(maxIdx),
      loss: crossEntropy(logits, y).data,
      lr: 0,
      gradNorm: 0,
      top: topKIndices(probs, Math.min(6, probs.length)).map((idx) => ({
        token: this.tokenOf(idx),
        prob: probs[idx],
      })),
      tokenEmbedding: debug.tokenEmbedding,
      positionEmbedding: debug.positionEmbedding,
      summedEmbedding: debug.summedEmbedding,
      mlpOut: debug.mlpOut,
      lnOut: debug.lnOut,
      logits: debug.logits,
      targetProb: probs[y] ?? 0,
    };
  }

  generate(maxTokens = 26): string {
    const blockSize = this.train.X[0].length;
    let context = Array(blockSize).fill(0);
    const out: number[] = [];
    for (let i = 0; i < maxTokens; i += 1) {
      const probs = softmax(this.model.call(context).map((v) => v.data));
      const ix = sampleCategorical(probs, this.rng);
      out.push(ix);
      context = [...context.slice(1), ix];
      if (ix === 0) break;
    }
    return this.decode(out);
  }

  private evaluate(): void {
    this.trainLoss = this.estimateLoss(this.train);
    this.devLoss = this.estimateLoss(this.dev);
    this.testLoss = this.estimateLoss(this.test);
    this.sample = this.generate(36);

    const probe = Array(this.train.X[0].length).fill(0);
    const probs = softmax(this.model.call(probe).map((v) => v.data));
    this.topTokens = topKIndices(probs, Math.min(8, probs.length)).map((idx) => ({
      token: this.tokenOf(idx),
      prob: probs[idx],
    }));
  }

  trainStep(): void {
    if (this.step >= this.maxSteps) return;

    const { x, y } = this.sampleExample(this.train);
    const logits = this.model.call(x);
    const debug = this.model.debugForward(x);
    const probs = softmax(debug.logits);
    const loss = crossEntropy(logits, y);

    for (const p of this.parameters) p.grad = 0;
    loss.backward();

    const gradSq = this.parameters.reduce((sum, p) => sum + p.grad * p.grad, 0);
    const gradNorm = Math.sqrt(gradSq / this.parameters.length);

    const lr = this.step < 100 ? 0.05 : 0.02;
    for (const p of this.parameters) p.data -= lr * p.grad;

    const predIdx = probs.reduce((best, p, i) => (p > probs[best] ? i : best), 0);
    this.lastTrace = {
      context: [...x],
      contextTokens: x.map((i) => this.tokenOf(i)),
      targetIndex: y,
      targetToken: this.tokenOf(y),
      predictedToken: this.tokenOf(predIdx),
      loss: loss.data,
      lr,
      gradNorm,
      top: topKIndices(probs, Math.min(6, probs.length)).map((idx) => ({
        token: this.tokenOf(idx),
        prob: probs[idx],
      })),
      tokenEmbedding: debug.tokenEmbedding,
      positionEmbedding: debug.positionEmbedding,
      summedEmbedding: debug.summedEmbedding,
      mlpOut: debug.mlpOut,
      lnOut: debug.lnOut,
      logits: debug.logits,
      targetProb: probs[y] ?? 0,
    };

    this.latestBatchLoss = loss.data;
    this.losses.push(loss.data);
    this.step += 1;

    if (this.step % this.evalEvery === 0 || this.step === this.maxSteps) {
      this.evaluate();
    }
  }
}

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

let trainer = new Trainer({
  datasetText: datasetEl.value,
  blockSize: 3,
  nEmbd: 10,
  maxSteps: Number(maxStepsEl.value),
  evalEvery: Number(evalEveryEl.value),
  seed: 1337,
});

let running = false;
let phaseCursor = 0;
const FLOW_FRAME_DELAY_MS = 90;

function drawLossChart(losses: number[]): void {
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
        <div class="text-white/70">Split of context windows used for optimization and evaluation.</div>
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
    return `
      <div class="grid gap-2 md:grid-cols-2">
        ${vectorBars('token embedding', t.tokenEmbedding, 'bg-neon/70')}
        ${vectorBars('position embedding', t.positionEmbedding, 'bg-butter/70')}
        ${vectorBars('sum embedding', t.summedEmbedding, 'bg-cyan-300/70')}
        ${vectorBars('MLP output', t.mlpOut, 'bg-indigo-300/70')}
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
      <div class="text-white/70">SGD update nudges each parameter using its gradient.</div>
      <div class="mono rounded-lg border border-white/10 bg-black/25 p-2 text-xs">
        param = param - lr * grad
      </div>
      <div class="mono text-xs text-white/70">lr=${t.lr.toFixed(4)} | avg step magnitude≈${delta.toExponential(2)}</div>
      ${vectorBars('layernorm output (fed into head)', t.lnOut, 'bg-emerald-300/70')}
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
  flowDetailEl.textContent = `${active.title}: ${active.description}`;
  flowVisualEl.innerHTML = stageVisualHtml(active.id);
}

function render(): void {
  stepEl.textContent = `${trainer.step}/${trainer.maxSteps}`;
  batchLossEl.textContent = trainer.latestBatchLoss.toFixed(4);
  trainLossEl.textContent = trainer.trainLoss.toFixed(4);
  devLossEl.textContent = trainer.devLoss.toFixed(4);
  sampleEl.textContent = trainer.sample || '...';

  dataStatsEl.textContent = `words=${datasetEl.value.split(/\r?\n/).filter(Boolean).length} | vocab=${trainer.vocabSize} | train/dev/test windows=${trainer.trainSize}/${trainer.devSize}/${trainer.testSize}`;

  traceContextEl.textContent = `[${trainer.lastTrace.context.join(', ')}]`;
  traceTokensEl.textContent = `[${trainer.lastTrace.contextTokens.join(', ')}]`;
  traceTargetEl.textContent = `${trainer.lastTrace.targetToken} (${trainer.lastTrace.targetIndex})`;
  tracePredEl.textContent = trainer.lastTrace.predictedToken;
  traceLrEl.textContent = trainer.lastTrace.lr.toFixed(4);
  traceGradEl.textContent = trainer.lastTrace.gradNorm.toFixed(6);

  progressBarEl.style.width = `${Math.min(100, (trainer.step / trainer.maxSteps) * 100)}%`;
  statusPillEl.textContent = running ? 'training' : trainer.step >= trainer.maxSteps ? 'completed' : 'idle';

  tokenBarsEl.innerHTML = trainer.lastTrace.top
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
  trainer = new Trainer({
    datasetText: datasetEl.value,
    blockSize: 3,
    nEmbd: 10,
    maxSteps: Math.max(50, Number(maxStepsEl.value) || 1200),
    evalEvery: Math.max(5, Number(evalEveryEl.value) || 24),
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
  trainer.sample = trainer.generate(42);
  render();
});

window.addEventListener('resize', () => render());

render();
