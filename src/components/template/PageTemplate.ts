import type { LocaleStrings } from '../../i18n/types';
import type { LocaleCode } from '../../i18n';
import {
  headerHtml,
  generatedNameSectionHtml,
  controlsPanelHtml,
  trainingDynamicsPanelHtml,
  algorithmPanelHtml,
  breakdownPanelHtml,
  probabilitiesPanelHtml,
  trainingDynamicsDialogHtml,
  transformerDialogHtml,
  stageDialogsHtml,
} from '../organisms';
import type { FlowStage } from '../../types';
import type { IllustrationLabels } from '../../i18n/types';

export function pageTemplateHtml(
  t: LocaleStrings,
  localeCode: LocaleCode,
  flowNodesHtml: string,
  buildIllustrationSvg: (stageId: string, labels: IllustrationLabels) => string,
): string {
  const stages = t.flowStages as FlowStage[];
  const stageDialogs = stageDialogsHtml(stages, t, buildIllustrationSvg);
  return `
  <main class="mx-auto max-w-7xl px-4 py-8 md:px-8">
    ${headerHtml(t, localeCode)}
    ${generatedNameSectionHtml(t)}
    <section class="mb-4 grid gap-4 lg:grid-cols-3">
      ${controlsPanelHtml(t)}
      ${trainingDynamicsPanelHtml(t)}
    </section>
    <section class="mb-4 grid gap-4 lg:grid-cols-3">
      ${algorithmPanelHtml(t, flowNodesHtml)}
      ${breakdownPanelHtml(t)}
    </section>
    <section class="grid gap-4 lg:grid-cols-1">
      ${probabilitiesPanelHtml(t)}
    </section>
    ${trainingDynamicsDialogHtml(t)}
    ${transformerDialogHtml(t)}
    ${stageDialogs}
  </main>
`;
}
