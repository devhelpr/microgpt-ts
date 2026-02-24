# microgpt TypeScript

**TypeScript port of [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) by [Andrej Karpathy](https://karpathy.github.io/).**

Features:

- CLI training (`microgpt.ts`)
- Browser UI (Vite + Vanilla TS + Tailwind)

## Browser UI run

```bash
npm install
npm run dev
```

Then open the local Vite URL (usually `http://localhost:5173`).

## Browser features

- Live loss chart (canvas)
- Train/dev/test metrics
- Progress bar
- Live samples
- Top-k token probability bars
- Full algorithm flow visualizer (dataset -> encoding -> context -> forward -> softmax -> loss -> backprop -> update)
- Live step breakdown (context ids/tokens, target vs prediction, learning rate, gradient norm)
- Dataset and hyperparameter controls

## CLI run

```bash
node --experimental-strip-types microgpt.ts
```

The CLI shows a live terminal dashboard with loss, samples, and probability distribution.

## Python run (microgpt.py)

The Python version reads names from `input.txt` in the **project root** (not in the same directory as the script). You can run the script from any directory:

```bash
python python/microgpt.py
```

The script looks for `input.txt` automatically in the directory above `python/`. If the file is missing, a sample dataset is downloaded.
