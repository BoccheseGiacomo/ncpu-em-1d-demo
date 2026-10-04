# NCPU-EM-1D demo

An interactive browser demo of a frozen
[NCPU-EM-1D](https://ncpu.pages.dev/) model: a one-dimensional neural cellular
automaton that transforms binary strings: reverse, copy, shifts, Gray encoding,
prefix XOR and increment. One small local rule
is applied to every tape cell over and over. A learned per-task program selects
which transformation emerges.

Everything runs client-side in plain JavaScript. There is no build step and no
dependencies.

## What you can do

- Pick a task and type a binary string. You can also change the tape size, the
  number of time steps, and the firing seed. By default the number of time
  steps equals the length of a training rollout for that tape: about 10 per cell
  for this model.
- Read the decoded output at the last time step, whether it is correct, and the
  step from which it stays correct. A badge marks inputs beyond the training
  range.
- Inspect **a channel over time** as a space-time diagram. It shows the I/O
  channel by default, and any program or hidden channel can be selected. You
  can also see **all channels** at any step: use the slider, play, or click the
  diagram.
- Compare with the model's **measured accuracy** per task: in distribution
  (ID), and 3 or 5 bits beyond the training maximum (OOD +3, OOD +5). The row of
  the selected task is highlighted.
- Share a run: the URL hash stores the task, input, seed, and any manual
  settings.

Limits: up to 64 tape cells and 3,000 steps.

## Files

| File | Role |
|---|---|
| `index.html`, `style.css`, `app.js` | The page |
| `nca.js` | Inference engine (pure ES module, no DOM). It mirrors `NeuralCellularAutomaton._step` |
| `model.json` | Frozen weights, configuration, and provenance |
| `results.json` | Accuracies measured offline in PyTorch (written by hand), tied to the model's md5 |
| `tools/export_model.py` | Exports `model.json` and `tests/fixtures.json` from a checkpoint |
| `tests/verify.mjs` | Checks `nca.js` against PyTorch |

## Faithfulness

`model.json` stores the fused perception/projection filters exactly as
`model.hidden_filters()` computes them. Weights use 9 significant digits, which
represent every float32 exactly.

Firing is stochastic, as in training: each cell updates with probability
`fire_rate` per step. The browser draws its masks from a seeded PRNG
(mulberry32). It is statistically equivalent to PyTorch's RNG, but not the same
stream.

`node tests/verify.mjs` checks two things:

1. **Exact replay.** The exporter recovers the exact firing masks PyTorch drew
   (and self-checks them against `forward()`). Replaying those masks in
   `nca.js` must reproduce every trajectory within 1e-4, with identical decoded
   tapes at every step. The current worst difference is about 1e-5.
2. **Sampled accuracy.** With the browser PRNG, semantic accuracy on the same
   300 inputs per case must match PyTorch within sampling noise (z ≤ 4). Cases
   cover every task, both at the training maximum and at the first
   extrapolation test case.

Use `node tests/verify.mjs --quick` to run the replays only.

## Run locally

`fetch` needs HTTP, so serve the folder rather than opening the file directly:

```bash
python -m http.server 8137
```

Then open <http://localhost:8137>.

## Swap in another checkpoint

The page, engine, and tests are driven by `model.json`: the task list,
channels, radius, program layout, activation, gate, and timing defaults. Any
checkpoint of the `ncpu-computer-1d-v2` format works, including ones with more
tasks. Export it with an environment that has PyTorch, then re-run the checks:

```bash
python tools/export_model.py --project <path/to/ncpu-em-1d> --checkpoint checkpoints/<run>/best.pt
```

```bash
node tests/verify.mjs
```

Then update `results.json` with the new model's md5 and measured accuracies.
The page shows the accuracy table only when its md5 matches `model.json`, so a
stale table is hidden rather than shown for the wrong model. Finally, bump the
`?v=` tag in `index.html` (see below).

## Current model

`checkpoints/reverse_curriculum_1d_prefix4_mutable_p5_c5_6k_learned_mutable_trials_5/best.pt`
(md5 `267e94df72…`): 6,000 updates and 8,719 parameters. It has 11 channels:
5 program, 1 I/O and 5 hidden. Each task's program covers 4 cells at the left
end of the tape and can change during a run. Training used 4 free steps per
tape cell, then ×1.5 supervised steps, on 7 tasks: reverse (weight 4), copy,
shift left, shift right, Gray encode, prefix XOR and increment.

Semantic accuracy, measured in PyTorch at the training schedule:

| Task | ID | OOD +3 (19 bits) | OOD +5 (21 bits) |
|---|---:|---:|---:|
| reverse | 99.74% | 71.24% | 39.78% |
| copy | 100.00% | 85.94% | 65.43% |
| shift left | 100.00% | 97.70% | 86.47% |
| shift right | 99.99% | 98.00% | 94.53% |
| Gray encode | 100.00% | 92.79% | 77.12% |
| prefix XOR | 99.99% | 84.01% | 58.95% |
| increment | 100.00% | 98.38% | 87.95% |
| mean | 99.96% | 89.72% | 72.89% |

Running 1.2–1.7× longer than the training schedule does not change these
numbers beyond sampling noise.

## Deploy on GitHub Pages

Push this repository, then go to Settings → Pages → Deploy from a branch →
`main` / root. The `.nojekyll` file makes Pages serve the files as they are.

Pages lets browsers cache each file for 10 minutes. When you change any file,
including `model.json`, bump the `?v=` tag in `index.html` (it appears twice).
`app.js` passes that tag on to `nca.js` and `model.json`, so a browser always
loads one consistent set of files.
