import fs from 'node:fs';

class Value {
  data: number;
  grad: number;
  private _prev: Set<Value>;
  private _backward: () => void;

  constructor(data: number, prev: Value[] = [], backward: () => void = () => {}) {
    this.data = data;
    this.grad = 0;
    this._prev = new Set(prev);
    this._backward = backward;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], () => {
      this.grad += 1 * out.grad;
      o.grad += 1 * out.grad;
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

    const buildTopo = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._prev) buildTopo(child);
      topo.push(v);
    };

    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
}

function randn(rng: () => number, mean = 0, std = 1): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + std * z0;
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
    for (let i = 0; i < x.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    return this.nonlin ? act.tanh() : act;
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

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

class MLP {
  layers: Layer[];

  constructor(nin: number, nouts: number[], rng: () => number) {
    const sz = [nin, ...nouts];
    this.layers = [];
    for (let i = 0; i < nouts.length; i++) {
      this.layers.push(new Layer(sz[i], sz[i + 1], rng, i !== nouts.length - 1));
    }
  }

  call(x: Value[]): Value[] {
    let out = x;
    for (const layer of this.layers) {
      out = layer.call(out);
    }
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
    for (const xi of x) {
      variance = variance.add(xi.sub(mean).pow(2));
    }
    variance = variance.div(n);
    const std = variance.add(this.eps).pow(0.5);

    const out: Value[] = [];
    for (let i = 0; i < n; i++) {
      out.push(x[i].sub(mean).div(std).mul(this.gamma[i]).add(this.beta[i]));
    }
    return out;
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
    const out: Value[] = [];
    for (let o = 0; o < this.w.length; o++) {
      let v = this.b[o];
      for (let i = 0; i < x.length; i++) {
        v = v.add(this.w[o][i].mul(x[i]));
      }
      out.push(v);
    }
    return out;
  }

  parameters(): Value[] {
    return [...this.w.flat(), ...this.b];
  }
}

class BigramLanguageModel {
  vocabSize: number;
  blockSize: number;
  nEmbd: number;
  wte: number[][];
  wpe: number[][];
  mlp: MLP;
  ln: LayerNorm;
  lmHead: Linear;

  constructor(vocabSize: number, blockSize: number, nEmbd: number, rng: () => number) {
    this.vocabSize = vocabSize;
    this.blockSize = blockSize;
    this.nEmbd = nEmbd;
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
    const t = idx.length;
    let lastHidden: Value[] = [];

    for (let pos = 0; pos < t; pos++) {
      const tokEmb = this.wte[idx[pos]];
      const posEmb = this.wpe[pos];
      const x = tokEmb.map((v, i) => new Value(v + posEmb[i]));
      lastHidden = this.ln.call(this.mlp.call(x));
    }

    return this.lmHead.call(lastHidden);
  }

  parameters(): Value[] {
    return [...this.mlp.parameters(), ...this.ln.parameters(), ...this.lmHead.parameters()];
  }
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
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
  for (let i = 0; i < probs.length; i++) {
    c += probs[i];
    if (r <= c) return i;
  }
  return probs.length - 1;
}

function sparkline(values: number[], width = 64): string {
  if (values.length === 0) return '';
  const chars = ' .:-=+*#%@';
  const slice = values.slice(Math.max(0, values.length - width));
  const min = Math.min(...slice);
  const max = Math.max(...slice);
  const range = max - min || 1;
  return slice
    .map((v) => {
      const norm = (v - min) / range;
      const idx = Math.min(chars.length - 1, Math.floor(norm * (chars.length - 1)));
      return chars[idx];
    })
    .join('');
}

function topKIndices(arr: number[], k: number): number[] {
  return [...arr.keys()].sort((a, b) => arr[b] - arr[a]).slice(0, k);
}

const rng = makeRng(1337);

const defaultText = `anna\nbob\ncarla\ndiana\nelias\nfrank\n`; 
const text = fs.existsSync('input.txt') ? fs.readFileSync('input.txt', 'utf8') : defaultText;
const words = text
  .split(/\r?\n/)
  .map((w) => w.trim())
  .filter((w) => w.length > 0);

if (words.length < 2) {
  throw new Error('Geen geldige dataset gevonden. Voeg meerdere regels toe aan input.txt');
}

const chars = Array.from(new Set(text.replace(/\r/g, '').split(''))).sort();
const stoi = new Map<string, number>();
const itos = new Map<number, string>();
chars.forEach((ch, i) => {
  stoi.set(ch, i + 1);
  itos.set(i + 1, ch);
});
stoi.set('.', 0);
itos.set(0, '.');

const vocabSize = stoi.size;

function encode(s: string): number[] {
  return s.split('').map((c) => stoi.get(c) ?? 0);
}

function decode(ids: number[]): string {
  return ids.map((i) => itos.get(i) ?? '?').join('');
}

function buildDataset(ws: string[], blockSize: number): { X: number[][]; Y: number[] } {
  const X: number[][] = [];
  const Y: number[] = [];

  for (const w of ws) {
    let context = Array(blockSize).fill(0);
    const encoded = encode(w + '.');
    for (const ch of encoded) {
      X.push([...context]);
      Y.push(ch);
      context = [...context.slice(1), ch];
    }
  }

  return { X, Y };
}

function randomShuffle<T>(arr: T[], rngFn: () => number): T[] {
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rngFn() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const shuffled = randomShuffle(words, rng);
const n1 = Math.floor(0.8 * shuffled.length);
const n2 = Math.floor(0.9 * shuffled.length);
const trainWords = shuffled.slice(0, n1);
const devWords = shuffled.slice(n1, n2);
const testWords = shuffled.slice(n2);

const blockSize = 3;
const nEmbd = 10;
const model = new BigramLanguageModel(vocabSize, blockSize, nEmbd, rng);
const parameters = model.parameters();

const train = buildDataset(trainWords, blockSize);
const dev = buildDataset(devWords.length ? devWords : trainWords, blockSize);
const test = buildDataset(testWords.length ? testWords : trainWords, blockSize);

function sampleExample(split: { X: number[][]; Y: number[] }): { x: number[]; y: number } {
  const i = Math.floor(rng() * split.X.length);
  return { x: split.X[i], y: split.Y[i] };
}

function estimateLoss(split: { X: number[][]; Y: number[] }, batches = 128): number {
  let total = 0;
  const n = Math.min(batches, split.X.length);
  for (let i = 0; i < n; i++) {
    const { x, y } = sampleExample(split);
    const logits = model.call(x);
    const loss = crossEntropy(logits, y);
    total += loss.data;
  }
  return total / n;
}

function generate(maxTokens = 40): string {
  let context = Array(blockSize).fill(0);
  const out: number[] = [];

  for (let i = 0; i < maxTokens; i++) {
    const logits = model.call(context);
    const probs = softmax(logits.map((v) => v.data));
    const ix = sampleCategorical(probs, rng);
    context = [...context.slice(1), ix];
    out.push(ix);
    if (ix === 0) break;
  }

  return decode(out);
}

const maxSteps = 1000;
const evalEvery = 25;
const history: number[] = [];

for (let step = 0; step < maxSteps; step++) {
  const { x, y } = sampleExample(train);
  const logits = model.call(x);
  const loss = crossEntropy(logits, y);

  for (const p of parameters) p.grad = 0;
  loss.backward();

  const lr = step < 100 ? 0.05 : 0.02;
  for (const p of parameters) {
    p.data -= lr * p.grad;
  }

  history.push(loss.data);

  if (step % evalEvery === 0 || step === maxSteps - 1) {
    const trainLoss = estimateLoss(train, 200);
    const devLoss = estimateLoss(dev, 200);
    const testLoss = estimateLoss(test, 200);

    const probe = Array(blockSize).fill(0);
    const probeLogits = model.call(probe).map((v) => v.data);
    const probeProbs = softmax(probeLogits);
    const top = topKIndices(probeProbs, Math.min(6, probeProbs.length));

    process.stdout.write('\x1Bc');
    console.log('microgpt.ts (TypeScript port + live visual)');
    console.log(`step ${step + 1}/${maxSteps}`);
    console.log(`loss train=${trainLoss.toFixed(4)} dev=${devLoss.toFixed(4)} test=${testLoss.toFixed(4)}`);
    console.log(`batch loss: ${loss.data.toFixed(4)}`);
    console.log('loss history:');
    console.log(sparkline(history, 80));
    console.log('sample:');
    console.log(generate(60));
    console.log('next-char probs for context "...":');
    for (const idx of top) {
      const label = itos.get(idx) ?? '?';
      const p = probeProbs[idx];
      const bar = '#'.repeat(Math.max(1, Math.round(p * 50)));
      console.log(`${label.padEnd(2)} ${bar} ${p.toFixed(3)}`);
    }
  }
}

console.log('\nFinal samples:');
for (let i = 0; i < 12; i++) {
  console.log(generate(60));
}
