/**
 * i18n / locale entry point.
 * Default locale is English. To add another language:
 * 1. Create e.g. src/i18n/de.ts implementing LocaleStrings
 * 2. Export it and add a locale key (e.g. currentLocale = en or de based on user/settings)
 */
export type { LocaleStrings, FlowStageEntry, IllustrationLabels } from './types';
export { en } from './en';
export { buildIllustrationSvg } from './illustrations';

import { en } from './en';
import type { LocaleStrings } from './types';

/** Currently active locale. Replace with a setter/locale detector when adding multilanguage support. */
export const currentLocale: LocaleStrings = en;

/** Shortcut for current locale strings (e.g. t.header.title). Use currentLocale when you need the full object. */
export const t = currentLocale;
