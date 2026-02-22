/**
 * Node CLI entry: reads input.txt, runs createMicroGptTrainer, and logs
 * the same training output as the original microgpt main(). Do not
 * import this from the browser app.
 */
import fs from 'node:fs';
import { createMicroGptTrainer, sparkline } from './microgpt';

const defaultText = `anna\nbob\ncarla\ndiana\nelias\nfrank\n`;
const text = fs.existsSync('input.txt') ? fs.readFileSync('input.txt', 'utf8') : defaultText;

const trainer = createMicroGptTrainer(text, {
  maxSteps: 1000,
  evalEvery: 25,
  seed: 42,
});

for (let step = 0; step < trainer.maxSteps; step++) {
  trainer.trainStep();
  if (
    step % trainer.evalEvery === 0 ||
    step === trainer.maxSteps - 1 ||
    trainer.step === trainer.maxSteps
  ) {
    process.stdout.write('\x1Bc');
    console.log('microgpt.ts (TypeScript port + live visual)');
    console.log(`step ${trainer.step}/${trainer.maxSteps}`);
    console.log(
      `loss train=${trainer.trainLoss.toFixed(4)} dev=${trainer.devLoss.toFixed(4)} test=${trainer.testLoss.toFixed(4)}`,
    );
    console.log(`batch loss: ${trainer.latestBatchLoss.toFixed(4)}`);
    console.log('loss history:');
    console.log(sparkline(trainer.losses, 80));
    console.log('sample:');
    console.log(trainer.generate(60));
    console.log('next-char probs for context "...":');
    for (const t of trainer.topTokens.slice(0, 6)) {
      const bar = '#'.repeat(Math.max(1, Math.round(t.prob * 50)));
      console.log(`${t.token.padEnd(2)} ${bar} ${t.prob.toFixed(3)}`);
    }
  }
}

console.log('\nFinal samples:');
for (let i = 0; i < 12; i++) {
  console.log(trainer.generate(60));
}
