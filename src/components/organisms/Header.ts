import type { LocaleStrings } from '../../i18n/types';
import type { LocaleCode } from '../../i18n';

export function headerHtml(t: Pick<LocaleStrings, 'header' | 'languages'>, localeCode: LocaleCode): string {
  return `
    <header class="relative mb-6 overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate to-[#1a1533] p-6 shadow-glow md:p-8">
      <div class="absolute -right-12 -top-12 h-44 w-44 rounded-full bg-coral/20 blur-3xl"></div>
      <div class="absolute -left-8 bottom-0 h-36 w-36 rounded-full bg-neon/20 blur-2xl"></div>
      <p class="panel-title">${t.header.panelTitle}</p>
      <div class="mt-2 flex flex-wrap items-center justify-between gap-3">
        <h1 class="text-3xl font-bold leading-tight md:text-5xl">${t.header.title}</h1>
        <div class="flex items-center gap-2 rounded-2xl border border-white/15 bg-black/30 px-3 py-1.5 text-xs text-white/70">
          <span>${t.languages.label}</span>
          <select
            id="localeSelect"
            class="relative z-10 rounded-lg border border-white/20 bg-black/60 px-2 py-1 text-xs text-white/90 cursor-pointer pointer-events-auto focus:border-neon focus:outline-none"
          >
            <option value="en" ${localeCode === 'en' ? 'selected' : ''}>${t.languages.en}</option>
            <option value="nl" ${localeCode === 'nl' ? 'selected' : ''}>${t.languages.nl}</option>
          </select>
        </div>
      </div>
      <p class="mt-2 max-w-3xl text-sm text-white/70 md:text-base">${t.header.subtitle}</p>
      <p class="mt-4 max-w-3xl text-sm text-white/75 md:text-base">${t.header.description}</p>
    </header>
  `;
}
