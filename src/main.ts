import './style.css';
import mermaid from 'mermaid';
import { createMicroGptTrainer } from '../microgpt';
import { getLocale, setLocale, getLocaleCode, type LocaleCode, buildIllustrationSvg } from './i18n';
import { queryRefs } from './dom/refs';
import { pageTemplateHtml } from './components/template/PageTemplate';
import { flowNodeHtml } from './components/molecules/FlowNode';
import { createInitialState } from './app/state';
import { trainLoop, resetTrainer } from './app/trainLoop';
import { render } from './app/render';
import { registerEvents } from './app/events';
import { getTransformerMermaidCode, mermaidThemeConfig } from './lib/mermaid';

// Initialise locale from persisted preference (if any) before reading locale strings.
const storedLocale = (typeof window !== 'undefined' ? window.localStorage.getItem('locale') : null) as LocaleCode | null;
if (storedLocale === 'en' || storedLocale === 'nl') {
  setLocale(storedLocale);
}

const t = getLocale();
const localeCode = getLocaleCode();
const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error(t.errors.appRootNotFound);

const flowStages = t.flowStages;
const flowNodesHtml = flowStages
  .map((stage, i) => flowNodeHtml(stage, i, t))
  .join('');

app.innerHTML = pageTemplateHtml(t, localeCode, flowNodesHtml, buildIllustrationSvg);

const refs = queryRefs(app, t.errors.missingUiElement);

const trainer = createMicroGptTrainer(refs.dataset.value, {
  blockSize: 16,
  nEmbd: 16,
  maxSteps: Math.max(50, Number(refs.maxSteps.value) || 1200),
  evalEvery: Math.max(5, Number(refs.evalEvery.value) || 24),
  seed: 1337,
});

const state = createInitialState(trainer);

mermaid.initialize(mermaidThemeConfig);

registerEvents(
  app,
  state,
  refs,
  t,
  flowStages,
  () => trainLoop(state, refs, t),
  () => resetTrainer(state, refs, t, createMicroGptTrainer),
  () => getTransformerMermaidCode(t),
  async (container) => {
    await mermaid.run({ nodes: container.querySelectorAll('.mermaid') });
  },
  t.errors.diagramRenderFailed,
);

render(state, refs, t);
