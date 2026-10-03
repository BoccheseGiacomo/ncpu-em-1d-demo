// Checks nca.js against PyTorch reference data written by tools/export_model.py.
//   node tests/verify.mjs          exact replays + sampled accuracies
//   node tests/verify.mjs --quick  exact replays only
//
// 1. Replay: the same firing masks PyTorch drew must give the same trajectory
//    (float tolerance) and the same decoded tape at every step.
// 2. Accuracy: with the browser's own PRNG, sampled semantic accuracy must
//    agree with PyTorch's within sampling noise (two-proportion z-test).

import { readFileSync } from "node:fs";
import {
  decodeSingle,
  loadModel,
  rollout,
  semanticCorrect,
  taskTarget,
  tapeSymbols,
  value,
} from "../nca.js";

const root = new URL("../", import.meta.url);
const read = (path) => JSON.parse(readFileSync(new URL(path, root), "utf8"));
const doc = read("model.json");
const fixtures = read("tests/fixtures.json");
const model = loadModel(doc);
const quick = process.argv.includes("--quick");
const TOLERANCE = 1e-4;
const MAX_Z = 4;
let failures = 0;

function fail(message) {
  failures++;
  console.log(`  FAIL ${message}`);
}

if (fixtures.model_md5 !== doc.source.md5) fail("fixtures were exported from a different model");
console.log(`model ${doc.source.checkpoint} (${doc.source.md5.slice(0, 10)}), torch ${fixtures.torch_version}`);

console.log("\nReplay with PyTorch's firing masks");
let worst = 0;
for (const fixture of fixtures.replay) {
  if (taskTarget(fixture.task, fixture.input) !== fixture.target) {
    fail(`${fixture.name}: task definition differs from Python`);
  }
  const result = rollout(model, {
    task: fixture.task,
    input: fixture.input,
    tapeSlots: fixture.tape_slots,
    steps: fixture.steps,
    masks: fixture.masks,
  });
  let maxError = 0;
  let symbolMismatches = 0;
  for (let t = 0; t <= fixture.steps; t++) {
    let expected = "";
    for (let x = 0; x < fixture.tape_slots; x++) {
      const reference = fixture.io[t][x];
      maxError = Math.max(maxError, Math.abs(value(result, t, model.io, x) - reference));
      expected += reference > model.threshold ? "1" : reference < -model.threshold ? "0" : "B";
    }
    if (tapeSymbols(result, t) !== expected) symbolMismatches++;
  }
  fixture.final_state.forEach((row, c) =>
    row.forEach((reference, x) => {
      maxError = Math.max(maxError, Math.abs(value(result, fixture.steps, c, x) - reference));
    }),
  );
  worst = Math.max(worst, maxError);
  const final = tapeSymbols(result, fixture.steps);
  const correct = semanticCorrect(final, fixture.target) ? "correct" : "wrong";
  console.log(
    `  ${fixture.name.padEnd(48)} steps=${String(fixture.steps).padStart(3)} ` +
      `max|Δ|=${maxError.toExponential(1)} final=${decodeSingle(final).output || "<empty>"} (${correct})`,
  );
  if (maxError > TOLERANCE) fail(`${fixture.name}: max error ${maxError} > ${TOLERANCE}`);
  if (symbolMismatches) fail(`${fixture.name}: decoded tape differs at ${symbolMismatches} steps`);
  if (final !== fixture.decoded) fail(`${fixture.name}: final tape ${final} != ${fixture.decoded}`);
}
console.log(`  worst max|Δ| = ${worst.toExponential(2)} (tolerance ${TOLERANCE})`);

function zScore(a, b, n) {
  // Laplace-smoothed proportions keep the test finite at 0% and 100%.
  const p = (a * n + 1) / (n + 2);
  const q = (b * n + 1) / (n + 2);
  return Math.abs(p - q) / Math.sqrt((p * (1 - p) + q * (1 - q)) / n);
}

if (!quick) {
  console.log("\nSampled accuracy with the browser PRNG (window mean / final step)");
  for (const [caseIndex, fixture] of fixtures.accuracy.entries()) {
    const start = fixture.free_steps + 1;
    const steps = fixture.free_steps + fixture.supervision_steps;
    let windowSum = 0;
    let finalSum = 0;
    fixture.inputs.forEach((input, index) => {
      const target = taskTarget(fixture.task, input);
      const result = rollout(model, {
        task: fixture.task,
        input,
        tapeSlots: fixture.tape_slots,
        steps,
        seed: caseIndex * 100003 + index,
      });
      let correctSteps = 0;
      for (let t = start; t <= steps; t++) correctSteps += semanticCorrect(tapeSymbols(result, t), target);
      windowSum += correctSteps / (steps - start + 1);
      finalSum += semanticCorrect(tapeSymbols(result, steps), target);
    });
    const n = fixture.inputs.length;
    const js = { window: windowSum / n, final: finalSum / n };
    const z = Math.max(
      zScore(js.window, fixture.python_window_semantic, n),
      zScore(js.final, fixture.python_final_semantic, n),
    );
    const percent = (x) => `${(100 * x).toFixed(2)}%`.padStart(8);
    console.log(
      `  ${fixture.name.padEnd(34)} n=${n} js ${percent(js.window)} /${percent(js.final)}  ` +
        `torch ${percent(fixture.python_window_semantic)} /${percent(fixture.python_final_semantic)}  z=${z.toFixed(2)}`,
    );
    if (z > MAX_Z) fail(`${fixture.name}: accuracies differ beyond sampling noise (z=${z.toFixed(2)})`);
  }
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
