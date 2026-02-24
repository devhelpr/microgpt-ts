import { describe, it, expect } from 'vitest';
import { flowNodeHtml } from './FlowNode';
import type { FlowStage } from '../../types';

const stage: FlowStage = {
  id: 'loss',
  title: 'Loss',
  description: 'Compute negative log probability',
};

const mockT = {
  aria: { close: 'Close', learnMoreAbout: 'Learn more about', explainTrainingDynamics: 'Explain' },
};

describe('flowNodeHtml', () => {
  it('includes article with id and data-stage-index', () => {
    const html = flowNodeHtml(stage, 5, mockT);
    expect(html).toContain('id="flow-loss"');
    expect(html).toContain('data-stage-index="5"');
  });

  it('includes stage title and description', () => {
    const html = flowNodeHtml(stage, 0, mockT);
    expect(html).toContain('Loss');
    expect(html).toContain('Compute negative log probability');
  });

  it('includes info button with aria-label', () => {
    const html = flowNodeHtml(stage, 0, mockT);
    expect(html).toContain('flow-info-btn');
    expect(html).toContain('Learn more about');
    expect(html).toContain('Loss');
  });

  it('pads step number with leading zero', () => {
    const html = flowNodeHtml(stage, 0, mockT);
    expect(html).toContain('01');
    const html9 = flowNodeHtml(stage, 8, mockT);
    expect(html9).toContain('09');
  });
});
