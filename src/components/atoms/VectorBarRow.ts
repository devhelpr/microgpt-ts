/**
 * Single row in a vector bar chart: sign, bar (width %), numeric value.
 */
export function vectorBarRow(value: number, _index: number, maxAbs: number, colorClass: string): string {
  const w = Math.max(4, Math.round((Math.abs(value) / maxAbs) * 100));
  return `
    <div class="grid grid-cols-[36px_1fr_56px] items-center gap-2 text-xs">
      <span class="mono text-white/50">${value >= 0 ? '+' : '-'}</span>
      <div class="h-2 overflow-hidden rounded bg-white/10"><div class="h-full ${colorClass}" style="width:${w}%"></div></div>
      <span class="mono text-right text-white/70">${value.toFixed(3)}</span>
    </div>
  `;
}
