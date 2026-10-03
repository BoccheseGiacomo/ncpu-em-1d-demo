# NCPU-EM-1D demo

An interactive browser demo of a frozen
[NCPU-EM-1D](https://ncpu.pages.dev/) model: a one-dimensional neural cellular
automaton that reverses, copies, and shifts binary strings. One small local rule
is applied to every tape cell over and over. A learned per-task program selects
which transformation emerges.

Everything runs client-side in plain JavaScript. There is no build step and no
dependencies.

## What you can do

- Pick a task and type a binary string. You can also change the tape size, the
  number of time steps, and the firing seed. By default the number of time
  steps equals the length of a training rollout for that tape: about 5 per cell
  for this model.
- Read the decoded output at the last time step, whether it is correct, and the
  step from which it stays correct. A badge marks inputs beyond the training
  range.
- Inspect the **I/O channel over time** (a space-time diagram), and **all
  channels** at any step. Use the slider, play, or click the diagram.
- Share a run: the URL hash stores the task, input, seed, and any manual
  settings.

Limits: up to 64 tape cells and 3,000 steps.

## Files

| File | Role |
|---|---|
| `index.html`, `style.css`, `app.js` | The page |
| `nca.js` | Inference engine (pure ES module, no DOM). It mirrors `NeuralCellularAutomaton._step` |
| `model.json` | Frozen weights, configuration, and provenance |
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
   tapes at every step. The current worst difference is about 2e-5.
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

## Current model

`checkpoints/preserved_run/best.pt` (md5 `3cc33eac80…`): 6,000 updates,
10,131 parameters, 2.0 free steps per tape cell, with tasks reverse, copy,
shift left and shift right. Inside the training range (inputs ≤ 16 bits,
tapes ≤ 24 cells) validation accuracy is ≥ 99.7% on every task. Beyond it,
copy and the shifts still hold, but reverse collapses (about 12% at 19 bits).

## Deploy on GitHub Pages

Push this repository, then go to Settings → Pages → Deploy from a branch →
`main` / root. The `.nojekyll` file makes Pages serve the files as they are.
