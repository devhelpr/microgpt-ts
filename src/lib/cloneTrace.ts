import type { StepTrace } from '../../microgpt';

export function cloneTrace(tr: StepTrace): StepTrace {
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
