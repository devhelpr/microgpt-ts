/**
 * i18n / locale entry point.
 * Default locale is English. To add another language:
 * 1. Create e.g. src/i18n/de.ts implementing LocaleStrings
 * 2. Export it and add a locale key (e.g. currentLocale = en or de based on user/settings)
 */
export type { LocaleStrings, FlowStageEntry, IllustrationLabels } from './types';
export { en } from './en';
export { nl } from './nl';
export { buildIllustrationSvg } from './illustrations';

import { en } from './en';
import { nl } from './nl';
import type { LocaleStrings } from './types';

export type LocaleCode = 'en' | 'nl';

let currentLocale: LocaleStrings = en;
let currentCode: LocaleCode = 'en';

export function setLocale(code: LocaleCode): void {
  if (code === 'nl') {
    currentLocale = nl;
    currentCode = 'nl';
  } else {
    currentLocale = en;
    currentCode = 'en';
  }
}

export function getLocale(): LocaleStrings {
  return currentLocale;
}

export function getLocaleCode(): LocaleCode {
  return currentCode;
}
