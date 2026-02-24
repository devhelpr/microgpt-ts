// Browser-safe core: no top-level Node or fs. CLI uses run-microgpt.ts.

export class Value {
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

  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this], () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
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

export function randn(rng: () => number, mean = 0, std = 1): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + std * z0;
}

export function makeRng(seed = 1337): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}

// --- Model constants (defaults; overridden per-trainer via config) ---
export const n_layer = 1;
export const n_embd_default = 16;
export const block_size_default = 16;
export const n_head = 4;
// Backward-compatible names for tests and consumers that expect fixed constants
export const n_embd = n_embd_default;
export const block_size = block_size_default;
export const head_dim = Math.floor(n_embd / n_head);

export function matrix(
  nout: number,
  nin: number,
  rng: () => number,
  std = 0.08,
): Value[][] {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(randn(rng, 0, std))),
  );
}

export function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((wo) => wo.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), new Value(0)));
}

export function softmaxValue(logits: Value[]): Value[] {
  const maxVal = logits.reduce((m, v) => (v.data > m ? v.data : m), -Infinity);
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((s, e) => s.add(e), new Value(0));
  return exps.map((e) => e.div(total));
}

export function rmsnorm(x: Value[], eps = 1e-5): Value[] {
  const n = x.length;
  let ms = new Value(0);
  for (const xi of x) ms = ms.add(xi.mul(xi));
  ms = ms.div(n);
  const scale = ms.add(eps).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function sampleCategorical(probs: number[], rng: () => number): number {
  const r = rng();
  let c = 0;
  for (let i = 0; i < probs.length; i++) {
    c += probs[i];
    if (r <= c) return i;
  }
  return probs.length - 1;
}

export function sparkline(values: number[], width = 64): string {
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

export function topKIndices(arr: number[], k: number): number[] {
  return [...arr.keys()].sort((a, b) => arr[b] - arr[a]).slice(0, k);
}

// --- StepTrace for UI (matches main.ts expectations) ---
export type StepTrace = {
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
  attentionWeights?: number[];
};

export type MicroGptConfig = {
  blockSize?: number;
  nEmbd?: number;
  maxSteps?: number;
  evalEvery?: number;
  seed?: number;
};

function randomShuffle<T>(arr: T[], rngFn: () => number): T[] {
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rngFn() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const defaultLearningRate = 0.01;
const beta1 = 0.85;
const beta2 = 0.99;
const eps_adam = 1e-8;
const defaultTemperature = 0.5;

function emptyTrace(itos: Map<number, string>, BOS: number): StepTrace {
  const tokenStr = (idx: number) => (idx === BOS ? 'BOS' : itos.get(idx) ?? '?');
  return {
    context: [],
    contextTokens: [],
    targetIndex: 0,
    targetToken: 'BOS',
    predictedToken: 'BOS',
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
}

export function createMicroGptTrainer(
  datasetText: string,
  config?: MicroGptConfig,
): {
  trainStep: () => void;
  get losses(): number[];
  get step(): number;
  get maxSteps(): number;
  get evalEvery(): number;
  get trainLoss(): number;
  get devLoss(): number;
  get testLoss(): number;
  get sample(): string;
  get topTokens(): Array<{ token: string; prob: number }>;
  get lastTrace(): StepTrace;
  get vocabSize(): number;
  get trainSize(): number;
  get devSize(): number;
  get testSize(): number;
  get latestBatchLoss(): number;
  generate: (maxTokens?: number, temp?: number) => string;
} {
  const block_size = config?.blockSize ?? block_size_default;
  const n_embd = config?.nEmbd ?? n_embd_default;
  const maxSteps = config?.maxSteps ?? 1000;
  const evalEvery = config?.evalEvery ?? 25;
  const seed = config?.seed ?? 42;
  const head_dim = Math.floor(n_embd / n_head);

  const docs = datasetText
    .split(/\r?\n/)
    .map((line: string) => line.trim())
    .filter((line: string) => line.length > 0);

  if (docs.length < 2) {
    throw new Error('Dataset must have at least 2 non-empty lines.');
  }

  const uchars: string[] = Array.from(new Set(docs.join('').split(''))).sort();
  const BOS = uchars.length;
  const vocab_size = uchars.length + 1;

  const itos = new Map<number, string>();
  uchars.forEach((ch, i) => itos.set(i, ch));
  itos.set(BOS, 'BOS');

  function tokenStr(idx: number): string {
    return idx === BOS ? 'BOS' : uchars[idx] ?? '?';
  }

  function encodeDoc(doc: string): number[] {
    return [BOS, ...doc.split('').map((c) => uchars.indexOf(c)), BOS];
  }

  const rngShuffle = makeRng(seed);
  const shuffled = randomShuffle([...docs], rngShuffle);
  const trainDocs = shuffled;
  const n1 = Math.floor(0.8 * shuffled.length);
  const n2 = Math.floor(0.9 * shuffled.length);
  const trainWords = shuffled.slice(0, n1);
  const devWords = shuffled.slice(n1, n2);
  const testWords = shuffled.slice(n2);

  const rngParams = makeRng(seed + 1);
  const rng = makeRng(seed + 2);

  const state_dict: Record<string, Value[][]> = {};
  state_dict['wte'] = matrix(vocab_size, n_embd, rngParams);
  state_dict['wpe'] = matrix(block_size, n_embd, rngParams);
  state_dict['lm_head'] = matrix(vocab_size, n_embd, rngParams);
  for (let i = 0; i < n_layer; i++) {
    state_dict[`layer${i}.attn_wq`] = matrix(n_embd, n_embd, rngParams);
    state_dict[`layer${i}.attn_wk`] = matrix(n_embd, n_embd, rngParams);
    state_dict[`layer${i}.attn_wv`] = matrix(n_embd, n_embd, rngParams);
    state_dict[`layer${i}.attn_wo`] = matrix(n_embd, n_embd, rngParams);
    state_dict[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd, rngParams);
    state_dict[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd, rngParams);
  }

  const params: Value[] = [];
  for (const mat of Object.values(state_dict) as Value[][][]) {
    for (const row of mat) for (const p of row) params.push(p);
  }

  const m: number[] = Array(params.length).fill(0);
  const v: number[] = Array(params.length).fill(0);

  function gpt(
    token_id: number,
    pos_id: number,
    keys: Value[][][],
    values: Value[][][],
  ): Value[] {
    const tok_emb = state_dict['wte'][token_id];
    const pos_emb = state_dict['wpe'][pos_id];
    let x = tok_emb.map((t, i) => t.add(pos_emb[i]));
    x = rmsnorm(x);

    for (let li = 0; li < n_layer; li++) {
      const x_residual = x;
      x = rmsnorm(x);
      const q = linear(x, state_dict[`layer${li}.attn_wq`]);
      const k = linear(x, state_dict[`layer${li}.attn_wk`]);
      const v = linear(x, state_dict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);

      const x_attn: Value[] = [];
      for (let h = 0; h < n_head; h++) {
        const hs = h * head_dim;
        const q_h = q.slice(hs, hs + head_dim);
        const k_h = keys[li].map((ki) => ki.slice(hs, hs + head_dim));
        const v_h = values[li].map((vi) => vi.slice(hs, hs + head_dim));
        const attn_logits = k_h.map((k_t) =>
          q_h.reduce((s, qj, j) => s.add(qj.mul(k_t[j])), new Value(0)).div(Math.pow(head_dim, 0.5)),
        );
        const attn_weights = softmaxValue(attn_logits);
        for (let j = 0; j < head_dim; j++) {
          let head_out = new Value(0);
          for (let t = 0; t < v_h.length; t++) {
            head_out = head_out.add(attn_weights[t].mul(v_h[t][j]));
          }
          x_attn.push(head_out);
        }
      }
      x = linear(x_attn, state_dict[`layer${li}.attn_wo`]).map((a, i) => a.add(x_residual[i]));

      const x_residual2 = x;
      x = rmsnorm(x);
      x = linear(x, state_dict[`layer${li}.mlp_fc1`]).map((xi) => xi.relu());
      x = linear(x, state_dict[`layer${li}.mlp_fc2`]).map((a, i) => a.add(x_residual2[i]));
    }

    return linear(x, state_dict['lm_head']);
  }

  type GptTraceResult = {
    logits: Value[];
    tokenEmbedding: number[];
    positionEmbedding: number[];
    summedEmbedding: number[];
    lnOut: number[];
    mlpOut: number[];
    attentionWeights: number[];
  };

  function gptWithTrace(
    token_id: number,
    pos_id: number,
    keys: Value[][][],
    values: Value[][][],
  ): GptTraceResult {
    const tok_emb = state_dict['wte'][token_id];
    const pos_emb = state_dict['wpe'][pos_id];
    let x = tok_emb.map((t, i) => t.add(pos_emb[i]));
    const summedEmbedding = x.map((v) => v.data);
    x = rmsnorm(x);

    let attentionWeights: number[] = [];
    for (let li = 0; li < n_layer; li++) {
      const x_residual = x;
      x = rmsnorm(x);
      const q = linear(x, state_dict[`layer${li}.attn_wq`]);
      const k = linear(x, state_dict[`layer${li}.attn_wk`]);
      const v = linear(x, state_dict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);

      const x_attn: Value[] = [];
      for (let h = 0; h < n_head; h++) {
        const hs = h * head_dim;
        const q_h = q.slice(hs, hs + head_dim);
        const k_h = keys[li].map((ki) => ki.slice(hs, hs + head_dim));
        const v_h = values[li].map((vi) => vi.slice(hs, hs + head_dim));
        const attn_logits = k_h.map((k_t) =>
          q_h.reduce((s, qj, j) => s.add(qj.mul(k_t[j])), new Value(0)).div(Math.pow(head_dim, 0.5)),
        );
        const attn_weights = softmaxValue(attn_logits);
        if (h === 0) attentionWeights = attn_weights.map((w) => w.data);
        for (let j = 0; j < head_dim; j++) {
          let head_out = new Value(0);
          for (let t = 0; t < v_h.length; t++) {
            head_out = head_out.add(attn_weights[t].mul(v_h[t][j]));
          }
          x_attn.push(head_out);
        }
      }
      x = linear(x_attn, state_dict[`layer${li}.attn_wo`]).map((a, i) => a.add(x_residual[i]));

      const x_residual2 = x;
      x = rmsnorm(x);
      x = linear(x, state_dict[`layer${li}.mlp_fc1`]).map((xi) => xi.relu());
      x = linear(x, state_dict[`layer${li}.mlp_fc2`]).map((a, i) => a.add(x_residual2[i]));
    }
    const preHead = x.map((v) => v.data);
    const logits = linear(x, state_dict['lm_head']);

    return {
      logits,
      tokenEmbedding: tok_emb.map((v) => v.data),
      positionEmbedding: pos_emb.map((v) => v.data),
      summedEmbedding,
      lnOut: preHead,
      mlpOut: preHead,
      attentionWeights,
    };
  }

  function estimateLoss(split: string[], batches = 128): number {
    let total = 0;
    let count = 0;
    const n = Math.min(batches, split.length);
    for (let i = 0; i < n; i++) {
      const doc = split[i];
      const tokens = encodeDoc(doc);
      const seqLen = Math.min(block_size, tokens.length - 1);
      if (seqLen < 1) continue;
      const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
      const values: Value[][][] = Array.from({ length: n_layer }, () => []);
      let docLoss = 0;
      for (let pos_id = 0; pos_id < seqLen; pos_id++) {
        const token_id = tokens[pos_id];
        const target_id = tokens[pos_id + 1];
        const logits = gpt(token_id, pos_id, keys, values);
        const probs = softmaxValue(logits);
        docLoss += -Math.log(probs[target_id].data + 1e-10);
      }
      total += docLoss / seqLen;
      count += 1;
    }
    return count > 0 ? total / count : 0;
  }

  function generate(maxTokens = 60, temp = defaultTemperature): string {
    const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const values: Value[][][] = Array.from({ length: n_layer }, () => []);
    let token_id = BOS;
    const sample: string[] = [];

    for (let pos_id = 0; pos_id < block_size; pos_id++) {
      const logits = gpt(token_id, pos_id, keys, values);
      const scaled = logits.map((l) => l.data / temp);
      const probs = softmax(scaled);
      token_id = sampleCategorical(probs, rng);
      if (token_id === BOS) break;
      sample.push(uchars[token_id]);
      if (sample.length >= maxTokens) break;
    }
    return sample.join('');
  }

  let step = 0;
  const losses: number[] = [];
  let trainLoss = 0;
  let devLoss = 0;
  let testLoss = 0;
  let sampleStr = '';
  let topTokensList: Array<{ token: string; prob: number }> = [];
  let latestBatchLoss = 0;
  let lastTrace: StepTrace = emptyTrace(itos, BOS);

  function evaluate(): void {
    trainLoss = estimateLoss(trainWords, 200);
    devLoss = estimateLoss(devWords.length ? devWords : trainWords, 200);
    testLoss = estimateLoss(testWords.length ? testWords : trainWords, 200);
    sampleStr = generate(36);

    const probeKeys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const probeValues: Value[][][] = Array.from({ length: n_layer }, () => []);
    const probeLogits = gpt(BOS, 0, probeKeys, probeValues);
    const probeProbsArr = softmaxValue(probeLogits).map((p) => p.data);
    const top = topKIndices(probeProbsArr, Math.min(8, probeProbsArr.length));
    topTokensList = top.map((idx) => ({
      token: tokenStr(idx),
      prob: probeProbsArr[idx],
    }));
  }

  evaluate();
  // Seed initial trace with first doc, first position
  (function seedTrace() {
    const doc = trainDocs[0];
    const tokens = encodeDoc(doc);
    if (tokens.length < 2) return;
    const traceKeys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const traceValues: Value[][][] = Array.from({ length: n_layer }, () => []);
    const res = gptWithTrace(tokens[0], 0, traceKeys, traceValues);
    const probs = softmaxValue(res.logits).map((p) => p.data);
    const targetId = tokens[1];
    const predId = probs.reduce((best, p, i) => (p > probs[best] ? i : best), 0);
    lastTrace = {
      context: [tokens[0]],
      contextTokens: [tokenStr(tokens[0])],
      targetIndex: targetId,
      targetToken: tokenStr(targetId),
      predictedToken: tokenStr(predId),
      loss: 0,
      lr: defaultLearningRate,
      gradNorm: 0,
      top: topKIndices(probs, Math.min(6, probs.length)).map((idx) => ({
        token: tokenStr(idx),
        prob: probs[idx],
      })),
      tokenEmbedding: res.tokenEmbedding,
      positionEmbedding: res.positionEmbedding,
      summedEmbedding: res.summedEmbedding,
      mlpOut: res.mlpOut,
      lnOut: res.lnOut,
      logits: res.logits.map((v) => v.data),
      targetProb: probs[targetId] ?? 0,
      attentionWeights: res.attentionWeights,
    };
  })();

  function trainStep(): void {
    if (step >= maxSteps) return;

    const doc = trainDocs[step % trainDocs.length];
    const tokens = encodeDoc(doc);
    const n = Math.min(block_size, tokens.length - 1);

    if (n < 1) {
      step += 1;
      losses.push(0);
      if (step % evalEvery === 0) evaluate();
      return;
    }

    const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const values: Value[][][] = Array.from({ length: n_layer }, () => []);

    const losses_vals: Value[] = [];
    for (let pos_id = 0; pos_id < n; pos_id++) {
      const token_id = tokens[pos_id];
      const target_id = tokens[pos_id + 1];
      const logits = gpt(token_id, pos_id, keys, values);
      const probs = softmaxValue(logits);
      losses_vals.push(probs[target_id].log().mul(-1));
    }

    let loss = losses_vals[0];
    for (let i = 1; i < losses_vals.length; i++) loss = loss.add(losses_vals[i]);
    loss = loss.div(n);

    for (const p of params) p.grad = 0;
    loss.backward();

    const gradSq = params.reduce((sum, p) => sum + p.grad * p.grad, 0);
    const gradNorm = Math.sqrt(gradSq / Math.max(1, params.length));

    const lr_t = defaultLearningRate * (1 - step / maxSteps);
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad;
      const m_hat = m[i] / (1 - Math.pow(beta1, step + 1));
      const v_hat = v[i] / (1 - Math.pow(beta2, step + 1));
      p.data -= (lr_t * m_hat) / (Math.sqrt(v_hat) + eps_adam);
      p.grad = 0;
    }

    latestBatchLoss = loss.data;
    losses.push(loss.data);
    step += 1;

    const traceKeys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const traceValues: Value[][][] = Array.from({ length: n_layer }, () => []);
    const traceRes = gptWithTrace(tokens[0], 0, traceKeys, traceValues);
    const traceProbs = softmaxValue(traceRes.logits).map((p) => p.data);
    const targetId = tokens[1];
    const predId = traceProbs.reduce((best, p, i) => (p > traceProbs[best] ? i : best), 0);

    lastTrace = {
      context: tokens.slice(0, Math.min(n + 1, tokens.length)),
      contextTokens: tokens.slice(0, Math.min(n + 1, tokens.length)).map(tokenStr),
      targetIndex: targetId,
      targetToken: tokenStr(targetId),
      predictedToken: tokenStr(predId),
      loss: loss.data,
      lr: lr_t,
      gradNorm,
      top: topKIndices(traceProbs, Math.min(6, traceProbs.length)).map((idx) => ({
        token: tokenStr(idx),
        prob: traceProbs[idx],
      })),
      tokenEmbedding: traceRes.tokenEmbedding,
      positionEmbedding: traceRes.positionEmbedding,
      summedEmbedding: traceRes.summedEmbedding,
      mlpOut: traceRes.mlpOut,
      lnOut: traceRes.lnOut,
      logits: traceRes.logits.map((v) => v.data),
      targetProb: traceProbs[targetId] ?? 0,
      attentionWeights: traceRes.attentionWeights,
    };

    if (step % evalEvery === 0 || step === maxSteps) {
      evaluate();
    }
  }

  const trainSize = trainWords.length;
  const devSize = devWords.length;
  const testSize = testWords.length;

  return {
    trainStep,
    get losses() {
      return losses;
    },
    get step() {
      return step;
    },
    get maxSteps() {
      return maxSteps;
    },
    get evalEvery() {
      return evalEvery;
    },
    get trainLoss() {
      return trainLoss;
    },
    get devLoss() {
      return devLoss;
    },
    get testLoss() {
      return testLoss;
    },
    get sample() {
      return sampleStr;
    },
    get topTokens() {
      return topTokensList;
    },
    get lastTrace() {
      return lastTrace;
    },
    get vocabSize() {
      return vocab_size;
    },
    get trainSize() {
      return trainSize;
    },
    get devSize() {
      return devSize;
    },
    get testSize() {
      return testSize;
    },
    get latestBatchLoss() {
      return latestBatchLoss;
    },
    generate,
  };
}
