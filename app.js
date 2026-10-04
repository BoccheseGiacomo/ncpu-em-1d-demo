// The engine and model load with this script's version tag (set in index.html),
// so the page never combines files from different deploys.
const VERSION = new URL(import.meta.url).search;
const {
  TASKS,
  decodeSingle,
  loadModel,
  outputLengthBound,
  rollout,
  semanticCorrect,
  taskTarget,
  tapeSymbols,
  value,
} = await import(`./nca.js${VERSION}`);

const MAX_TAPE = 64;
const MAX_STEPS = 3000;
const BUSY_WORK = 12000; // tape cells × steps above which a "computing…" indicator is shown
const NEGATIVE = [59, 76, 192];
const NEUTRAL = [245, 245, 245];
const POSITIVE = [180, 4, 38];

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(
  [
    "modelInfo", "task", "input", "random", "examples", "tape", "tapeAuto", "steps",
    "stepsAuto", "stepsHint", "seed", "reseed", "error", "badge", "target", "output",
    "verdict", "raw", "settle", "readoutNote", "channel", "spacetime", "spacetimeHover", "channels",
    "channelsHover", "time", "play", "stepLabel", "stepDecoded", "stepVerdict", "busy", "busyText", "accuracyPanel", "accuracyTable", "accuracyNotes",
    "programRole", "programPlacement", "taskHint", "taskList", "taskExampleInput", "async",
    "asyncLabel", "asyncHint", "firingText",
    "provenance",
  ].map((id) => [id, $(id)]),
);

let model;
let training;
let run = null; // { task, input, tape, steps, seed, result, target }
let step = 0;
let playing = false;
let scheduled = 0;

const roundHalfUp = (x) => Math.floor(x + 0.5);

function color(v) {
  const t = Math.max(-1, Math.min(1, v));
  const end = t < 0 ? NEGATIVE : POSITIVE;
  const a = Math.abs(t);
  return NEUTRAL.map((n, i) => Math.round(n + a * (end[i] - n)));
}

function channelName(c) {
  const { program_channels: P, io_channel: io } = model.doc.config;
  if (c < P) return `P${c}`;
  if (c === io) return "I/O";
  return `C${c - io - 1}`;
}

function channelDescription(c) {
  const { program_channels: P, io_channel: io } = model.doc.config;
  if (c < P) return `${channelName(c)} · program`;
  if (c === io) return "I/O · input/output";
  return `${channelName(c)} · hidden`;
}

const shownChannel = () => Number(ui.channel.value);

// --- Parameters ---

// --- Task descriptions (examples are computed with the same functions as the targets) ---

const EXAMPLE_INPUT = "10110";
const x = (i) => `<var>x</var><sub>${i}</sub>`;
const y = (i) => `<var>y</var><sub>${i}</sub>`;
const TASK_INFO = {
  copy: { formula: `${y("i")} = ${x("i")}`, text: "leave the string unchanged." },
  bit_not: { formula: `${y("i")} = ¬${x("i")}`, text: "flip every bit." },
  reverse: { formula: `${y("i")} = ${x("n+1−i")}`, text: "read the string backwards." },
  reverse_not: { formula: `${y("i")} = ¬${x("n+1−i")}`, text: "reverse, then flip every bit." },
  shift_left_zero: {
    formula: `${y("i")} = ${x("i+1")}, &nbsp;${y("n")} = 0`,
    text: "move every bit one place left; the first bit drops off and a 0 enters on the right.",
  },
  shift_right_zero: {
    formula: `${y("1")} = 0, &nbsp;${y("i")} = ${x("i−1")}`,
    text: "move every bit one place right; a 0 enters on the left and the last bit drops off.",
  },
  gray_encode: {
    formula: `${y("1")} = ${x("1")}, &nbsp;${y("i")} = ${x("i−1")} ⊕ ${x("i")}`,
    text: "binary to Gray code: each bit marks where neighbouring input bits differ.",
  },
  prefix_xor: {
    formula: `${y("i")} = ${x("1")} ⊕ ${x("2")} ⊕ … ⊕ ${x("i")}`,
    text: "running parity of the bits so far (this is also Gray decoding).",
  },
  increment: {
    formula: `<var>y</var> = (<var>x</var> + 1) mod 2<sup><var>n</var></sup>`,
    text: "add 1 to the binary number (most significant bit first); the width stays n, so 11…1 wraps to 00…0.",
  },
  parity: {
    formula: `<var>y</var> = ${x("1")} ⊕ … ⊕ ${x("n")}`,
    text: "a single bit: 1 when the input has an odd number of 1s.",
  },
  append_0: { formula: `<var>y</var> = <var>x</var>0`, text: "add a 0 at the end." },
  append_1: { formula: `<var>y</var> = <var>x</var>1`, text: "add a 1 at the end." },
};

function exampleOf(task) {
  return `${EXAMPLE_INPUT} → ${taskTarget(task, EXAMPLE_INPUT)}`;
}

function renderTasks() {
  ui.taskExampleInput.textContent = EXAMPLE_INPUT;
  const items = [];
  for (const task of model.tasks) {
    const info = TASK_INFO[task];
    const name = element("dt", TASKS[task].label);
    name.dataset.task = task;
    const formula = document.createElement("dd");
    formula.className = "formula";
    formula.innerHTML = info.formula;
    const example = document.createElement("dd");
    example.className = "example";
    example.append(element("code", exampleOf(task)), ` — ${info.text}`);
    items.push(name, formula, example);
  }
  ui.taskList.replaceChildren(...items);
}

function showTask() {
  const task = ui.task.value;
  ui.taskHint.innerHTML = TASK_INFO[task].formula;
  ui.taskHint.append("  ·  ", element("code", exampleOf(task)));
  for (const name of ui.taskList.querySelectorAll("dt")) {
    name.classList.toggle("current", name.dataset.task === task);
  }
}

// Asynchronous firing (as trained) or every cell at every step.
function showFiring() {
  const trainedAsync = model.fireRate < 1;
  const asynchronous = ui.async.checked;
  ui.seed.disabled = ui.reseed.disabled = !asynchronous;
  ui.asyncLabel.textContent = trainedAsync
    ? `asynchronous firing (p = ${model.fireRate})`
    : "asynchronous firing (not used by this model)";
  ui.asyncHint.textContent = asynchronous
    ? `Each cell updates with probability ${model.fireRate} per step, as in training; the seed fixes which cells fire.`
    : trainedAsync
      ? "Off: every cell updates at every step, so the run is deterministic. The model was trained " +
        "with random firing; see the table for both modes."
      : "Every cell updates at every step, as in training.";
  ui.firingText.textContent = trainedAsync
    ? "Training used asynchronous updates: at each step a random subset of cells fires. By default " +
      "this page runs the model synchronously (every cell at every step), which is deterministic; " +
      "switch on asynchronous firing to run it as trained, with the seed fixing which cells fire."
    : "Every cell updates at every step, as in training.";
  if (ui.accuracyTable.dataset.modes) highlightAccuracy();
}

// The tape must hold at least one blank and the whole program.
const minTape = () => Math.max(2, model.L);

function autoTape(task, length) {
  const fit = Math.max(length + 1, outputLengthBound(task, length) + 1);
  // Reproduces the training pairs (3→5, 5→8, 7→11, 10→15, 13→20).
  return Math.min(MAX_TAPE, Math.max(minTape(), fit, length + Math.max(2, Math.ceil(length / 2))));
}

// The training schedule: free evolution followed by the supervised window,
// whose end is where training read the answer.
function autoSteps(tape) {
  const free = roundHalfUp(training.free_steps_per_tape_slot * tape);
  return free + roundHalfUp(training.supervision_ratio * free);
}

function readInteger(input) {
  const text = input.value.trim();
  return /^\d+$/.test(text) ? Number(text) : NaN;
}

function syncAutoFields() {
  const task = ui.task.value;
  showTask();
  showFiring();
  const input = ui.input.value.trim();
  ui.tape.disabled = ui.tapeAuto.checked;
  ui.steps.disabled = ui.stepsAuto.checked;
  if (ui.tapeAuto.checked && /^[01]*$/.test(input)) ui.tape.value = autoTape(task, input.length);
  const tape = readInteger(ui.tape);
  if (ui.stepsAuto.checked && Number.isInteger(tape)) ui.steps.value = autoSteps(tape);
}

function readParameters() {
  const task = ui.task.value;
  const input = ui.input.value.trim();
  const tape = readInteger(ui.tape);
  const steps = readInteger(ui.steps);
  const seed = readInteger(ui.seed);
  const invalid = (field, message) => ({ error: message, field });
  if (!/^[01]*$/.test(input)) return invalid(ui.input, "The input may contain only 0 and 1.");
  if (!Number.isInteger(tape) || tape < minTape() || tape > MAX_TAPE) {
    const reason = model.L > 2 ? ` (the task program needs ${model.L} cells)` : "";
    return invalid(ui.tape, `Tape cells must be a whole number from ${minTape()} to ${MAX_TAPE}${reason}.`);
  }
  if (input.length > tape - 1) {
    return invalid(ui.input, `The input must leave at least one blank cell (at most ${tape - 1} bits).`);
  }
  if (outputLengthBound(task, input.length) > tape - 1) {
    return invalid(ui.tape, "The output must leave at least one blank cell; use a longer tape.");
  }
  if (!Number.isInteger(steps) || steps < 1 || steps > MAX_STEPS) {
    return invalid(ui.steps, `Time steps must be a whole number from 1 to ${MAX_STEPS}.`);
  }
  if (!Number.isInteger(seed) || seed > 0xffffffff) return invalid(ui.seed, "The seed must be a whole number ≥ 0.");
  return { task, input, tape, steps, seed, asynchronous: ui.async.checked };
}

// --- URL state (shareable links) ---

function writeHash(p) {
  const params = new URLSearchParams({ task: p.task, input: p.input, seed: p.seed });
  if (!ui.tapeAuto.checked) params.set("tape", p.tape);
  if (!ui.stepsAuto.checked) params.set("steps", p.steps);
  if (p.asynchronous) params.set("async", "1");
  history.replaceState(null, "", `#${params}`);
}

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (model.tasks.includes(params.get("task"))) ui.task.value = params.get("task");
  if (params.has("input")) ui.input.value = params.get("input");
  if (params.has("seed")) ui.seed.value = params.get("seed");
  // A link fully determines the run: absent tape/steps means automatic.
  ui.tapeAuto.checked = !params.has("tape");
  if (params.has("tape")) ui.tape.value = params.get("tape");
  ui.stepsAuto.checked = !params.has("steps");
  // Synchronous by default; "async=1" runs the model with random firing, as trained.
  ui.async.checked = model.fireRate < 1 && params.has("async");
  if (params.has("steps")) ui.steps.value = params.get("steps");
}

// --- Running ---

function schedule(delay = 80) {
  clearTimeout(scheduled);
  syncAutoFields();
  scheduled = setTimeout(compute, delay);
}

function compute() {
  for (const field of [ui.input, ui.tape, ui.steps, ui.seed]) {
    field.removeAttribute("aria-invalid");
  }
  const p = readParameters();
  if (p.error) {
    setBusy(null);
    ui.error.textContent = p.error;
    p.field.setAttribute("aria-invalid", "true");
    return;
  }
  ui.error.textContent = "";
  if (p.tape * p.steps > BUSY_WORK) {
    // Long runs block the page briefly: show the indicator and let it paint first.
    setBusy("computing…");
    scheduled = setTimeout(() => simulate(p), 30);
  } else {
    simulate(p);
  }
}

function setBusy(text) {
  ui.busy.hidden = text === null;
  if (text !== null) ui.busyText.textContent = text;
  for (const element of [ui.settle.closest("dl"), ui.spacetime.closest(".panel"), ui.channels.closest(".panel")]) {
    element.classList.toggle("stale", text !== null);
  }
}

function simulate(p) {
  const result = rollout(model, {
    task: p.task,
    input: p.input,
    tapeSlots: p.tape,
    steps: p.steps,
    seed: p.seed,
    fireRate: p.asynchronous ? model.fireRate : 1,
  });
  run = { ...p, result, target: taskTarget(p.task, p.input) };
  writeHash(p);
  stop();
  ui.time.max = p.steps;
  showResult();
  setStep(p.steps);
  setBusy(null);
}

function showResult() {
  const { result, target, steps, input, tape } = run;
  const raw = tapeSymbols(result, steps);
  const decoded = decodeSingle(raw);
  const correct = semanticCorrect(raw, target);
  // Earliest step from which the output stays correct through the last step.
  let settled = steps + 1;
  while (settled > 0 && semanticCorrect(tapeSymbols(result, settled - 1), target)) settled--;

  ui.target.textContent = target || "<empty>";
  ui.output.textContent = decoded.output || "<empty>";
  ui.verdict.textContent = correct ? "✓ correct" : "✗ wrong";
  ui.verdict.className = `verdict ${correct ? "good" : "bad"}`;
  ui.raw.textContent = raw;
  ui.settle.textContent = correct
    ? `correct from step ${settled} through step ${steps}`
    : "not correct at the last step";
  ui.readoutNote.textContent =
    `Output is read at the last time step (${steps}). Cells above ` +
    `+${model.threshold} read as 1, below −${model.threshold} as 0, otherwise blank (B); ` +
    "the output ends at the first blank.";

  const reasons = [];
  if (input.length > training.input_max) reasons.push(`input ${input.length} > ${training.input_max} bits`);
  if (tape > training.tape_max) reasons.push(`tape ${tape} > ${training.tape_max} cells`);
  if (tape < training.tape_min) reasons.push(`tape ${tape} < ${training.tape_min} cells`);
  ui.badge.textContent = reasons.length ? `extrapolation: ${reasons.join(", ")}` : "within training range";
  ui.badge.className = `badge ${reasons.length ? "out" : "in"}`;
  highlightAccuracy();
}

function setStep(t) {
  step = Math.max(0, Math.min(run.steps, t));
  ui.time.value = step;
  ui.stepLabel.textContent = `${step} / ${run.steps}`;
  const raw = tapeSymbols(run.result, step);
  const correct = semanticCorrect(raw, run.target);
  ui.stepDecoded.textContent = decodeSingle(raw).output || "<empty>";
  ui.stepVerdict.textContent = correct ? "✓" : "✗";
  ui.stepVerdict.className = `verdict ${correct ? "good" : "bad"}`;
  drawSpacetime();
  drawChannels();
}

// --- Drawing ---

function prepare(canvas, height) {
  const width = canvas.parentElement.clientWidth;
  const ratio = window.devicePixelRatio || 1;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function spacetimeGeometry() {
  const rows = run.steps + 1;
  return { rows, height: Math.max(160, Math.min(560, rows * 4)) };
}

function drawSpacetime() {
  const { result, tape: T, steps } = run;
  const { rows, height } = spacetimeGeometry();
  const { context, width } = prepare(ui.spacetime, height);
  const channel = shownChannel();
  const image = new ImageData(T, rows);
  for (let t = 0; t <= steps; t++) {
    for (let x = 0; x < T; x++) {
      const [r, g, b] = color(value(result, t, channel, x));
      const i = 4 * (t * T + x);
      image.data[i] = r;
      image.data[i + 1] = g;
      image.data[i + 2] = b;
      image.data[i + 3] = 255;
    }
  }
  const bitmap = document.createElement("canvas");
  bitmap.width = T;
  bitmap.height = rows;
  bitmap.getContext("2d").putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(bitmap, 0, 0, width, height);

  // Current step marker.
  const y = (step + 0.5) * (height / rows);
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--text");
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(0, y);
  context.lineTo(width, y);
  context.stroke();
}

function spacetimeCell(event) {
  const rect = ui.spacetime.getBoundingClientRect();
  const { rows } = spacetimeGeometry();
  const t = Math.floor(((event.clientY - rect.top) / rect.height) * rows);
  const x = Math.floor(((event.clientX - rect.left) / rect.width) * run.tape);
  return { t: Math.max(0, Math.min(run.steps, t)), x };
}

const LABEL_WIDTH = 40;
const INDEX_HEIGHT = 16;

function channelsGeometry() {
  const width = ui.channels.parentElement.clientWidth;
  const cell = Math.max(4, Math.min(36, Math.floor((width - LABEL_WIDTH) / run.tape)));
  return { cell, height: INDEX_HEIGHT + (model.C + 1) * cell + 6 };
}

function drawChannels() {
  const { cell, height } = channelsGeometry();
  const { context } = prepare(ui.channels, height);
  const styles = getComputedStyle(document.documentElement);
  const text = styles.getPropertyValue("--text");
  const muted = styles.getPropertyValue("--muted");
  const gold = styles.getPropertyValue("--gold");
  const T = run.tape;
  context.font = `${Math.min(12, Math.max(9, cell * 0.4))}px ui-monospace, Consolas, monospace`;
  context.textBaseline = "middle";

  context.textAlign = "center";
  context.fillStyle = muted;
  const every = cell >= 18 ? 1 : cell >= 9 ? 5 : 10;
  for (let x = 0; x < T; x += every) {
    context.fillText(String(x), LABEL_WIDTH + (x + 0.5) * cell, INDEX_HEIGHT / 2);
  }
  for (let c = 0; c < model.C; c++) {
    const y = INDEX_HEIGHT + c * cell;
    context.textAlign = "right";
    context.fillStyle = c === model.io ? gold : text;
    context.fillText(channelName(c), LABEL_WIDTH - 6, y + cell / 2);
    for (let x = 0; x < T; x++) {
      const [r, g, b] = color(value(run.result, step, c, x));
      context.fillStyle = `rgb(${r},${g},${b})`;
      context.fillRect(LABEL_WIDTH + x * cell, y, cell, cell);
    }
  }
  if (cell >= 6) {
    context.strokeStyle = styles.getPropertyValue("--grid");
    context.lineWidth = 1;
    context.beginPath();
    for (let x = 0; x <= T; x++) {
      context.moveTo(LABEL_WIDTH + x * cell + 0.5, INDEX_HEIGHT);
      context.lineTo(LABEL_WIDTH + x * cell + 0.5, INDEX_HEIGHT + model.C * cell);
    }
    for (let c = 0; c <= model.C; c++) {
      context.moveTo(LABEL_WIDTH, INDEX_HEIGHT + c * cell + 0.5);
      context.lineTo(LABEL_WIDTH + T * cell, INDEX_HEIGHT + c * cell + 0.5);
    }
    context.stroke();
  }
  context.strokeStyle = gold;
  context.lineWidth = 2;
  context.strokeRect(LABEL_WIDTH, INDEX_HEIGHT + model.io * cell, T * cell, cell);

  // Readout row: the I/O channel quantized to symbols.
  const y = INDEX_HEIGHT + model.C * cell + 6;
  const raw = tapeSymbols(run.result, step);
  context.textAlign = "right";
  context.fillStyle = muted;
  context.fillText("read", LABEL_WIDTH - 6, y + cell / 2);
  context.textAlign = "center";
  if (cell >= 8) {
    for (let x = 0; x < T; x++) {
      context.fillStyle = raw[x] === "B" ? muted : text;
      context.fillText(raw[x], LABEL_WIDTH + (x + 0.5) * cell, y + cell / 2);
    }
  }
}

function channelsCell(event) {
  const rect = ui.channels.getBoundingClientRect();
  const { cell } = channelsGeometry();
  const x = Math.floor((event.clientX - rect.left - LABEL_WIDTH) / cell);
  const c = Math.floor((event.clientY - rect.top - INDEX_HEIGHT) / cell);
  return { x, c };
}

// --- Playback ---

function stop() {
  playing = false;
  ui.play.textContent = "▶";
  ui.play.setAttribute("aria-label", "Play");
}

function play() {
  if (step >= run.steps) setStep(0);
  playing = true;
  ui.play.textContent = "❚❚";
  ui.play.setAttribute("aria-label", "Pause");
  const rate = Math.max(20, run.steps / 8); // steps per second; a full run plays in ≤ 8 s
  let last = performance.now();
  let carry = 0;
  const tick = (now) => {
    if (!playing) return;
    carry += ((now - last) / 1000) * rate;
    last = now;
    const advance = Math.floor(carry);
    if (advance) {
      carry -= advance;
      setStep(step + advance);
    }
    if (step >= run.steps) return stop();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// --- Wiring ---

function describe() {
  const s = model.doc.source;
  const c = model.doc.config;
  ui.programRole.textContent =
    c.program_mode === "learned_mutable"
      ? "that start from the learned task program and then evolve like any other channel"
      : "that hold the learned task program and stay fixed during a run";
  ui.programPlacement.textContent =
    c.program_placement === "prefix"
      ? "the chosen task's program is written into the program channels of the first few " +
        "cells at the left end of the tape (the rest start at zero)"
      : "the chosen task's program is written into the program channels of every cell";
  ui.modelInfo.textContent =
    `${s.update.toLocaleString()} training updates · ${s.parameter_count.toLocaleString()} parameters · ` +
    `trained on inputs ≤ ${training.input_max} bits, tapes ${training.tape_min}–${training.tape_max} cells`;
  const perCell = training.free_steps_per_tape_slot * (1 + training.supervision_ratio);
  ui.stepsHint.textContent =
    `Auto: about ${Number(perCell.toFixed(2))} time steps per tape cell, the length of a training rollout.`;
  ui.provenance.textContent = `Checkpoint ${s.checkpoint} (md5 ${s.md5.slice(0, 10)}).`;
}

// Accuracies measured offline (results.json), shown only for the model they belong to.
async function loadResults() {
  try {
    const response = await fetch(`results.json${VERSION}`);
    if (!response.ok) return;
    const results = await response.json();
    if (results.model_md5 === model.doc.source.md5) renderAccuracy(results);
  } catch (error) {
    console.warn("results.json could not be loaded:", error);
  }
}

function element(tag, text, title) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (title) node.title = title;
  return node;
}

function renderAccuracy(results) {
  const row = (cells) => {
    const tr = document.createElement("tr");
    tr.append(...cells);
    return tr;
  };
  const format = (v) => `${v.toFixed(2)}%`;
  // One cell per (group, mode); cells carry their mode so the active one can be highlighted.
  const valueCells = (values) =>
    results.groups.flatMap((g) =>
      results.modes.map((m) => {
        const cell = element("td", format(values[g.key][m.key]));
        cell.dataset.mode = m.key;
        return cell;
      }),
    );
  const table = ui.accuracyTable;
  table.replaceChildren();
  const head = table.createTHead();
  const task = element("th", "Task");
  task.rowSpan = 2;
  head.append(
    row([
      task,
      ...results.groups.map((g) => {
        const cell = element("th", g.label, `${g.title}: ${g.detail}`);
        cell.colSpan = results.modes.length;
        cell.className = "group";
        return cell;
      }),
    ]),
    row(
      results.groups.flatMap(() =>
        results.modes.map((m) => {
          const cell = element("th", m.label, m.detail);
          cell.dataset.mode = m.key;
          return cell;
        }),
      ),
    ),
  );
  const body = table.createTBody();
  for (const name of model.tasks) {
    if (!results.rows[name]) continue;
    const tr = row([element("td", TASKS[name].label), ...valueCells(results.rows[name])]);
    tr.dataset.task = name;
    body.append(tr);
  }
  if (results.mean) table.createTFoot().append(row([element("td", "mean"), ...valueCells(results.mean)]));

  const note = (label, text) => {
    const item = document.createElement("li");
    item.append(element("strong", label), `: ${text}.`);
    return item;
  };
  ui.accuracyNotes.replaceChildren(
    element("li", `${results.metric[0].toUpperCase()}${results.metric.slice(1)}: the output is correct and followed by a blank.`),
    ...results.groups.map((g) => note(`${g.label}, ${g.title.toLowerCase()}`, g.detail)),
    ...results.modes.map((m) => note(m.label, m.detail)),
  );
  ui.accuracyTable.dataset.modes = JSON.stringify(
    Object.fromEntries(results.modes.map((m) => [m.key, m.asynchronous])),
  );
  ui.accuracyPanel.hidden = false;
  highlightAccuracy();
}

// Bold the selected task's row and the columns of the active firing mode.
function highlightAccuracy() {
  for (const tr of ui.accuracyTable.querySelectorAll("tbody tr")) {
    tr.classList.toggle("current", tr.dataset.task === ui.task.value);
  }
  const modes = JSON.parse(ui.accuracyTable.dataset.modes || "{}");
  for (const cell of ui.accuracyTable.querySelectorAll("[data-mode]")) {
    cell.classList.toggle("active-mode", modes[cell.dataset.mode] === ui.async.checked);
  }
}

function wire() {
  const fields = [ui.task, ui.input, ui.tape, ui.steps, ui.seed];
  for (const field of fields) field.addEventListener("input", () => schedule());
  ui.tapeAuto.addEventListener("change", () => schedule(0));
  ui.stepsAuto.addEventListener("change", () => schedule(0));
  ui.async.addEventListener("change", () => schedule(0));
  ui.examples.addEventListener("click", (event) => {
    const example = event.target.closest("button[data-example]");
    if (!example) return;
    ui.input.value = example.dataset.example;
    schedule(0);
  });
  ui.random.addEventListener("click", () => {
    const length = 1 + Math.floor(Math.random() * training.input_max);
    ui.input.value = Array.from({ length }, () => (Math.random() < 0.5 ? "0" : "1")).join("");
    schedule(0);
  });
  ui.reseed.addEventListener("click", () => {
    ui.seed.value = Math.floor(Math.random() * 1e6);
    schedule(0);
  });
  ui.time.addEventListener("input", () => {
    stop();
    setStep(Number(ui.time.value));
  });
  ui.play.addEventListener("click", () => (playing ? stop() : play()));
  ui.channel.addEventListener("change", () => run && drawSpacetime());

  ui.spacetime.addEventListener("pointerdown", (event) => {
    ui.spacetime.setPointerCapture(event.pointerId);
    stop();
    setStep(spacetimeCell(event).t);
  });
  ui.spacetime.addEventListener("pointermove", (event) => {
    if (!run) return;
    const { t, x } = spacetimeCell(event);
    if (event.buttons) setStep(t);
    ui.spacetimeHover.textContent =
      x >= 0 && x < run.tape
        ? `step ${t}, cell ${x}: ${channelName(shownChannel())} = ${value(run.result, t, shownChannel(), x).toFixed(3)}`
        : " ";
  });
  ui.spacetime.addEventListener("pointerleave", () => (ui.spacetimeHover.textContent = " "));
  ui.channels.addEventListener("pointermove", (event) => {
    if (!run) return;
    const { x, c } = channelsCell(event);
    const inside = x >= 0 && x < run.tape && c >= 0 && c < model.C;
    ui.channelsHover.textContent = inside
      ? `step ${step}, cell ${x}, ${channelName(c)} = ${value(run.result, step, c, x).toFixed(3)}`
      : " ";
  });
  ui.channels.addEventListener("pointerleave", () => (ui.channelsHover.textContent = " "));

  let width = 0;
  new ResizeObserver(() => {
    const current = ui.channels.parentElement.clientWidth;
    if (run && current !== width) {
      width = current;
      drawSpacetime();
      drawChannels();
    }
  }).observe(document.querySelector(".layout"));
  window.addEventListener("hashchange", () => {
    readHash();
    schedule(0);
  });
}

async function main() {
  try {
    const response = await fetch(`model.json${VERSION}`);
    if (!response.ok) throw new Error(`model.json: HTTP ${response.status}`);
    model = loadModel(await response.json());
  } catch (error) {
    document.body.classList.remove("loading");
    ui.busy.hidden = true;
    ui.modelInfo.textContent = `Could not load the model: ${error.message}`;
    ui.error.textContent = "The model failed to load. Serve this folder over HTTP (see README).";
    return;
  }
  training = model.doc.training;
  for (const task of model.tasks) ui.task.add(new Option(TASKS[task].label, task));
  for (let c = 0; c < model.C; c++) ui.channel.add(new Option(channelDescription(c), c));
  ui.channel.value = model.io;
  ui.input.value = "11111100";
  ui.seed.value = 0;
  ui.async.disabled = model.fireRate >= 1;
  ui.tape.min = minTape();
  renderTasks();
  loadResults();
  readHash();
  describe();
  wire();
  document.body.classList.remove("loading");
  schedule(0); // the first run hides the "loading model…" indicator
}

main();
