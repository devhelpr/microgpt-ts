import type { LocaleStrings } from '../../i18n/types';

export function probabilitiesPanelHtml(t: Pick<LocaleStrings, 'probabilities'>): string {
  return `
    <div class="panel p-4">
      <p class="panel-title">${t.probabilities.panelTitle}</p>
      <div id="tokenBars" class="mt-3 space-y-2"></div>
    </div>
  `;
}
