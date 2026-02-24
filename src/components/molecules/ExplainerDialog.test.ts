import { describe, it, expect } from 'vitest';
import { stepExplainerDialogHtml } from './ExplainerDialog';
import type { FlowStage } from '../../types';

const stage: FlowStage = {
  id: 'encode',
  title: 'Encode',
  description: 'Turn text into token IDs',
};

const mockT = {
  aria: { close: 'Close', learnMoreAbout: 'Learn more about', explainTrainingDynamics: 'Explain' },
  explainerBodies: { encode: 'Body about encoding.' },
  illustrationLabels: {} as import('../../i18n/types').IllustrationLabels,
};

function mockBuildSvg(_stageId: string, _labels: import('../../i18n/types').IllustrationLabels): string {
  return '<svg>mock</svg>';
}

function mockBuildSvgEmpty(): string {
  return '';
}

describe('stepExplainerDialogHtml', () => {
  it('includes dialog with correct id', () => {
    const html = stepExplainerDialogHtml(stage, mockT, mockBuildSvgEmpty);
    expect(html).toContain('id="dialog-encode"');
    expect(html).toContain('<dialog');
  });

  it('includes stage title and close button', () => {
    const html = stepExplainerDialogHtml(stage, mockT, mockBuildSvgEmpty);
    expect(html).toContain('Encode');
    expect(html).toContain('dialog-close');
    expect(html).toContain('Close');
  });

  it('uses explainerBodies when present', () => {
    const html = stepExplainerDialogHtml(stage, mockT, mockBuildSvgEmpty);
    expect(html).toContain('Body about encoding.');
  });

  it('falls back to stage description when explainerBodies missing', () => {
    const tNoBody = { ...mockT, explainerBodies: {} as Record<string, string> };
    const html = stepExplainerDialogHtml(stage, tNoBody, mockBuildSvgEmpty);
    expect(html).toContain('Turn text into token IDs');
  });

  it('includes illustration when buildIllustrationSvg returns non-empty', () => {
    const html = stepExplainerDialogHtml(stage, mockT, mockBuildSvg);
    expect(html).toContain('<svg>mock</svg>');
  });

  it('omits illustration div when buildIllustrationSvg returns empty', () => {
    const html = stepExplainerDialogHtml(stage, mockT, mockBuildSvgEmpty);
    expect(html).not.toContain('flex justify-center');
  });
});
