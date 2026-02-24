/**
 * Single stat box (label + value). Used in training dynamics stat row.
 */
export function statBox(label: string, value: string, id?: string): string {
  const valueEl = id
    ? `<div id="${id}" class="mono mt-1 text-xl">${value}</div>`
    : `<div class="mono mt-1 text-xl">${value}</div>`;
  return `<div class="rounded-xl border border-white/10 bg-black/25 p-3"><div class="text-xs text-white/50">${label}</div>${valueEl}</div>`;
}
