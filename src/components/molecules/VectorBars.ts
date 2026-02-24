import type { LocaleStrings } from '../../i18n/types';
import { vectorBarRow } from '../atoms/VectorBarRow';

export function vectorBars(
  label: string,
  values: number[],
  colorClass: string,
  t: Pick<LocaleStrings, 'vectorBars'>,
): string {
  const slice = values.slice(0, 8);
  const maxAbs = Math.max(...slice.map((v) => Math.abs(v)), 1e-6);
  const bars = slice.map((v, i) => vectorBarRow(v, i, maxAbs, colorClass)).join('');
  return `
    <div class="rounded-lg border border-white/10 bg-black/25 p-2">
      <div class="mb-2 text-xs text-white/60">${label} ${t.vectorBars.first8Dims}</div>
      <div class="space-y-1">${bars}</div>
    </div>
  `;
}
