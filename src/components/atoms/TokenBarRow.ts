/**
 * Single token probability row: token label, bar (width %), probability value.
 */
export function tokenBarRow(token: string, prob: number, widthPct: number): string {
  const w = Math.max(6, Math.round(widthPct));
  return `
    <div class="grid grid-cols-[40px_1fr_52px] items-center gap-2 text-sm">
      <span class="mono text-white/70">${token}</span>
      <div class="h-2 overflow-hidden rounded bg-white/10"><div class="meter-fill h-full" style="width:${w}%"></div></div>
      <span class="mono text-right text-white/70">${prob.toFixed(3)}</span>
    </div>
  `;
}
