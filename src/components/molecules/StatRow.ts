import type { LocaleStrings } from '../../i18n/types';
import { statBox } from '../atoms/StatBox';

export function statRowHtml(t: Pick<LocaleStrings, 'trainingDynamics'>): string {
  const td = t.trainingDynamics;
  return `
    <div class="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
      ${statBox(td.step, '0', 'step')}
      ${statBox(td.batchLoss, '0.0000', 'batchLoss')}
      ${statBox(td.trainLoss, '0.0000', 'trainLoss')}
      ${statBox(td.devLoss, '0.0000', 'devLoss')}
    </div>
  `;
}
