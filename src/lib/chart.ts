import type { LocaleStrings } from '../i18n/types';

const CHART_HEIGHT = 220;

export function drawLossChart(
  canvas: HTMLCanvasElement | null,
  losses: number[],
  t: Pick<LocaleStrings, 'chart'>,
): void {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const width = Math.max(300, Math.floor(rect.width));
  const height = CHART_HEIGHT;
  canvas.width = width * dpr;
  canvas.height = height * dpr;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);

  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, 'rgba(68, 242, 217, 0.12)');
  bg.addColorStop(1, 'rgba(7, 11, 20, 0.0)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  if (losses.length < 2) return;

  const xPad = 12;
  const yPad = 14;
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = max - min || 1;

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#44f2d9';
  ctx.beginPath();

  losses.forEach((loss, i) => {
    const x = xPad + (i / (losses.length - 1)) * (width - xPad * 2);
    const y = yPad + ((max - loss) / range) * (height - yPad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.65)';
  ctx.font = '11px IBM Plex Mono';
  ctx.fillText(`${t.chart.min} ${min.toFixed(3)}`, 12, height - 8);
  ctx.fillText(`${t.chart.max} ${max.toFixed(3)}`, width - 82, height - 8);
}
