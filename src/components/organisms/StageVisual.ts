import type { StepTrace } from '../../types';
import type { TrainerSizes } from '../../types';
import type { LocaleStrings } from '../../i18n/types';
import { vectorBars } from '../molecules/VectorBars';

export function stageVisualHtml(
  stageId: string,
  tr: StepTrace,
  trainer: TrainerSizes,
  t: Pick<LocaleStrings, 'stageVisual' | 'vectorBars'>,
): string {
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
        ${vectorBars(t.stageVisual.forward.tokenEmbedding, tr.tokenEmbedding, 'bg-neon/70', t)}
        ${vectorBars(t.stageVisual.forward.positionEmbedding, tr.positionEmbedding, 'bg-butter/70', t)}
        ${vectorBars(t.stageVisual.forward.sumEmbedding, tr.summedEmbedding, 'bg-cyan-300/70', t)}
        ${vectorBars(t.stageVisual.forward.preHeadAfterMlp, tr.lnOut, 'bg-indigo-300/70', t)}
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
      ${vectorBars(t.stageVisual.forward.preHeadBeforeLmHead, tr.lnOut, 'bg-emerald-300/70', t)}
    </div>
  `;
}
