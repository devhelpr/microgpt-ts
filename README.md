# microgpt TypeScript

TypeScript-port van Karpathy's `microgpt.py` met:

- CLI training (`microgpt.ts`)
- Browser UI (Vite + Vanilla TS + Tailwind)

## Browser UI run

```bash
npm install
npm run dev
```

Open daarna de lokale Vite URL (meestal `http://localhost:5173`).

## Browser features

- Live loss chart (canvas)
- Train/dev/test metrics
- Progress bar
- Live samples
- Top-k token probability bars
- Dataset en hyperparameter controls

## CLI run

```bash
node --experimental-strip-types microgpt.ts
```

CLI toont een live terminal-dashboard met loss, samples en kansverdeling.
