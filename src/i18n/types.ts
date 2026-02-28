/**
 * Shape of a locale for multilanguage support.
 * Add new locales (e.g. de.ts) that satisfy this interface.
 */
export type FlowStageEntry = {
  id: string;
  title: string;
  description: string;
};

export type IllustrationLabels = {
  dataset: { train: string; dev: string; test: string; names: string; namesPerLine: string };
  encode: { charA: string; charN: string; id0: string; id1: string };
  context: { next: string; pos0: string; block: string };
  forward: {
    embed: string;
    attn: string;
    mlp: string;
    logits: string;
  };
  softmax: { logits: string; raw: string; expSum: string; probsSum1: string };
  loss: { pTarget: string; probability: string; negLog: string; L: string; loss: string };
  backprop: { params: string; gradLabel: string; L: string; backward: string };
  update: { param: string; adam: string; adamFormula: string; updated: string };
};

export interface LocaleStrings {
  errors: {
    appRootNotFound: string;
    missingUiElement: string;
    diagramRenderFailed: string;
  };
  defaultDataset: string;
  flowStages: FlowStageEntry[];
  illustrationLabels: IllustrationLabels;
  explainerBodies: Record<string, string>;
  aria: {
    close: string;
    learnMoreAbout: string; // "Learn more about {title}" - title is interpolated
    explainTrainingDynamics: string;
  };
  header: {
    panelTitle: string;
    title: string;
    subtitle: string; // HTML: link to microgpt + author credit
    description: string;
  };
  generatedName: {
    panelTitle: string;
    generate: string;
    infoBtn: string; // aria-label for info icon
  };
  controls: {
    panelTitle: string;
    datasetLabel: string;
    maxSteps: string;
    evalEvery: string;
    start: string;
    pause: string;
    reset: string;
    tipReset: string;
  };
  trainingDynamics: {
    panelTitle: string;
    explainBtn: string;
    statusIdle: string;
    statusTraining: string;
    statusCompleted: string;
    step: string;
    batchLoss: string;
    trainLoss: string;
    devLoss: string;
  };
  algorithm: {
    panelTitle: string;
    review: string;
    live: string;
    viewTransformerDiagram: string;
  };
  breakdown: {
    panelTitle: string;
    stepBreakdown: string; // "Step {n} Breakdown"
    contextText: string;
    noContextText: string;
    contextIds: string;
    contextTokens: string;
    contextTokensBosHint: string; // e.g. "Showing only characters; internal markers hidden"
    target: string;
    predicted: string;
    learningRate: string;
    gradientNorm: string;
  };
  probabilities: {
    panelTitle: string;
    explainer: string;
  };
  languages: {
    label: string;
    en: string;
    nl: string;
  };
  iterationSelect: {
    stepLabel: string; // "Step {n}"
    trainDevLabel: string; // "train={trainLoss} dev={devLoss}"
  };
  chart: {
    min: string;
    max: string;
  };
  vectorBars: {
    first8Dims: string;
  };
  stageVisual: {
    dataset: {
      splitDescription: string;
      train: string;
      dev: string;
      test: string;
    };
    encode: {
      description: string;
      contextTokensToIds: string;
      target: string;
    };
    context: {
      description: string;
    };
    forward: {
      attentionHead0: string;
      tokenEmbedding: string;
      positionEmbedding: string;
      sumEmbedding: string;
      preHeadAfterMlp: string;
      preHeadBeforeLmHead: string;
    };
    softmax: {
      description: string;
    };
    loss: {
      description: string;
      predictedTarget: string; // "Predicted {pred}, target {target}"
    };
    backprop: {
      description: string;
      gradientNorm: string;
    };
    update: {
      description: string;
      formula: string;
      lrStepMagnitude: string; // "lr={lr} | avg step magnitude≈{delta}"
    };
  };
  mermaid: {
    whichCharacter: string;
    whichPosition: string;
    turnCharIntoVector: string;
    addPositionAsVector: string;
    combineBoth: string;
    stabilizeScale: string;
    transformerBlock: string; // "Transformer block × {n}"
    stabilize: string;
    attentionMixContext: string;
    addShortcut: string;
    smallFeedForward: string;
    predictNextChar: string;
    scoresForEachChar: string;
  };
  transformerDiagramExplainers: {
    A: string;
    B: string;
    C: string;
    D: string;
    E: string;
    F: string;
    TB: string;
    G1: string;
    G2: string;
    G3: string;
    G4: string;
    G5: string;
    G6: string;
    H: string;
    I: string;
  };
  dialogs: {
    trainingDynamics: {
      title: string;
      whatGraphShows: string;
      whatGraphShowsBody: string;
      spikesMean: string;
      spikesMeanBody: string;
      numbersAboveGraph: string;
      numbersAboveGraphBody: string;
      lowerLossNote: string;
    };
    transformer: {
      title: string;
      description: string;
      intro: string;
      diagramHint: string;
      oneForwardPassNote: string;
    };
    generatedName: {
      title: string;
      whatItDoes: string;
      whatItDoesBody: string;
      howItWorks: string;
      howItWorksBody: string;
      lastGeneratedLabel: string;
      lastGeneratedEmpty: string;
    };
  };
  samplePlaceholder: string; // "..." when no sample yet
  dataStatsTemplate: string; // "words={words} | vocab={vocab} | train/dev/test={train}/{dev}/{test}"
}
