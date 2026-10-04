// Inference engine for the NCPU-EM-1D neural cellular automaton.
// Pure functions with no DOM access, shared by the page and tests/verify.mjs.
// step() mirrors NeuralCellularAutomaton._step in ncpu_computer_1d/model.py.

export const FORMAT = "ncpu-em-1d-web-v1";
const SYMBOL_VALUES = { 0: -1, 1: 1, B: 0 };

// --- Tasks (ncpu_computer_1d/tasks.py: TASK_TRANSFORMS, output_length_bound) ---

const bitNot = (value) => value.replace(/[01]/g, (bit) => (bit === "0" ? "1" : "0"));
const reverse = (value) => [...value].reverse().join("");

export const TASKS = {
  copy: { label: "copy", transform: (value) => value },
  bit_not: { label: "bitwise NOT", transform: bitNot },
  reverse: { label: "reverse", transform: reverse },
  reverse_not: { label: "reverse + NOT", transform: (value) => bitNot(reverse(value)) },
  shift_left_zero: {
    label: "shift left (fill 0)",
    transform: (value) => (value ? value.slice(1) + "0" : ""),
  },
  shift_right_zero: {
    label: "shift right (fill 0)",
    transform: (value) => (value ? "0" + value.slice(0, -1) : ""),
  },
  gray_encode: {
    label: "Gray encode",
    transform: (value) =>
      value ? value[0] + [...value.slice(1)].map((bit, i) => (bit !== value[i] ? "1" : "0")).join("") : "",
  },
  prefix_xor: {
    label: "prefix XOR",
    transform: (value) => {
      let parity = 0;
      return [...value].map((bit) => (parity ^= Number(bit))).join("");
    },
  },
  increment: {
    label: "increment (wraps)",
    transform: (value) => {
      const bits = [...value];
      for (let i = bits.length - 1; i >= 0; i--) {
        if (bits[i] === "0") {
          bits[i] = "1";
          break;
        }
        bits[i] = "0";
      }
      return bits.join("");
    },
  },
  parity: {
    label: "parity",
    transform: (value) => ([...value].filter((bit) => bit === "1").length % 2 ? "1" : "0"),
  },
  append_0: { label: "append 0", transform: (value) => value + "0" },
  append_1: { label: "append 1", transform: (value) => value + "1" },
};

export function taskTarget(task, input) {
  if (!(task in TASKS)) throw new Error(`unknown task: ${task}`);
  if (!/^[01]*$/.test(input)) throw new Error("input must contain only 0 and 1");
  return TASKS[task].transform(input);
}

export function outputLengthBound(task, inputLength) {
  if (task === "parity") return 1;
  if (task === "append_0" || task === "append_1") return inputLength + 1;
  return inputLength;
}

// --- Model ---

function flat(values, length, name) {
  const array = Float32Array.from(values.flat(Infinity));
  if (array.length !== length) {
    throw new Error(`${name} has ${array.length} values, expected ${length}`);
  }
  return array;
}

export function loadModel(doc) {
  if (doc.format !== FORMAT) throw new Error(`unsupported model format: ${doc.format}`);
  const config = doc.config;
  const weights = doc.weights;
  const C = config.channels;
  const H = config.hidden_size;
  const K = 2 * config.radius + 1;
  const L = config.program_length;
  const P = config.program_channels;
  const gated = config.gate !== "none";
  const rows = gated ? 2 * C : C;
  for (const task of doc.tasks) {
    if (!(task in TASKS)) throw new Error(`task ${task} has no browser definition`);
  }
  if (!["relu", "softplus"].includes(config.activation)) {
    throw new Error(`unsupported activation: ${config.activation}`);
  }
  if (!["none", "linear", "sigmoid", "tanh", "relu"].includes(config.gate)) {
    throw new Error(`unsupported gate: ${config.gate}`);
  }
  return {
    doc,
    tasks: doc.tasks,
    C,
    H,
    K,
    L,
    P,
    rows,
    radius: config.radius,
    io: config.io_channel,
    placement: config.program_placement,
    mutable: Uint8Array.from(config.mutable, Number),
    activation: config.activation,
    gate: config.gate,
    fireRate: config.fire_rate,
    leak: config.state_leak,
    clip: config.max_abs_state,
    threshold: config.ternary_threshold,
    filters: flat(weights.hidden_filters, H * C * K, "hidden_filters"),
    hiddenBias: flat(weights.hidden_bias, H, "hidden_bias"),
    outWeight: flat(weights.output_weight, rows * H, "output_weight"),
    outBias: gated ? flat(weights.output_bias, rows, "output_bias") : null,
    programs: flat(weights.programs, doc.tasks.length * L * P, "programs"),
  };
}

export function taskIndex(model, task) {
  const index = model.tasks.indexOf(task);
  if (index < 0) throw new Error(`the model was not trained on task ${task}`);
  return index;
}

// State layout: Float32Array of [channel][cell], channel-major.
export function initialState(model, task, input, tapeSlots) {
  const T = tapeSlots;
  if (!Number.isInteger(T) || T < Math.max(1, model.L)) {
    throw new Error("tape must contain the complete program");
  }
  if (!/^[01B]*$/.test(input) || input.length > T) {
    throw new Error("input does not fit the tape");
  }
  const state = new Float32Array(model.C * T);
  const offset = taskIndex(model, task) * model.L * model.P;
  for (let x = 0; x < T; x++) {
    if (model.placement === "prefix" && x >= model.L) continue;
    const phase = x % model.L;
    for (let p = 0; p < model.P; p++) {
      state[p * T + x] = model.programs[offset + phase * model.P + p];
    }
  }
  for (let x = 0; x < input.length; x++) state[model.io * T + x] = SYMBOL_VALUES[input[x]];
  return state;
}

function activate(model, value) {
  if (model.activation === "relu") return value > 0 ? value : 0;
  return value > 20 ? value : Math.log1p(Math.exp(value)); // torch softplus, threshold 20
}

function gateValue(model, value) {
  switch (model.gate) {
    case "linear":
      return value;
    case "sigmoid":
      return 1 / (1 + Math.exp(-value));
    case "tanh":
      return Math.tanh(value);
    default:
      return value > 0 ? value : 0;
  }
}

export function createScratch(model) {
  return {
    neighborhood: new Float32Array(model.C * model.K),
    hidden: new Float32Array(model.H),
    output: new Float32Array(model.rows),
  };
}

// One update. mask[x] is 0 or 1 per cell (shared by all channels) or null.
export function step(model, state, T, mask, out, scratch = createScratch(model)) {
  const { C, H, K, radius, filters, hiddenBias, outWeight, outBias, mutable } = model;
  const CK = C * K;
  const { neighborhood, hidden, output } = scratch;
  for (let x = 0; x < T; x++) {
    // Fused perception + first projection: one convolution with zero padding.
    for (let c = 0; c < C; c++) {
      for (let k = 0; k < K; k++) {
        const source = x + k - radius;
        neighborhood[c * K + k] = source >= 0 && source < T ? state[c * T + source] : 0;
      }
    }
    for (let h = 0; h < H; h++) {
      let sum = hiddenBias[h];
      const base = h * CK;
      for (let j = 0; j < CK; j++) sum += filters[base + j] * neighborhood[j];
      hidden[h] = activate(model, sum);
    }
    const fires = mask === null || mask[x] === 1;
    for (let c = 0; c < C; c++) {
      const current = state[c * T + x];
      if (!mutable[c]) {
        out[c * T + x] = current;
        continue;
      }
      let updated = current;
      if (fires) {
        let delta = outBias ? outBias[c] : 0;
        for (let h = 0; h < H; h++) delta += outWeight[c * H + h] * hidden[h];
        if (outBias) {
          let gate = outBias[C + c];
          for (let h = 0; h < H; h++) gate += outWeight[(C + c) * H + h] * hidden[h];
          delta *= gateValue(model, gate);
        }
        if (model.leak) delta -= model.leak * current;
        updated = current + delta;
      }
      if (model.clip !== null) updated = Math.min(model.clip, Math.max(-model.clip, updated));
      out[c * T + x] = updated;
    }
  }
  return out;
}

// Seeded PRNG (mulberry32) for firing masks.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Full trajectory: states[t] for t = 0..steps, each [channel][cell].
// fireRate overrides the trained firing probability (1 = every cell, every step).
// masks (optional, for tests): one "0101..." string or Uint8Array per step.
export function rollout(
  model,
  { task, input, tapeSlots, steps, seed = 0, fireRate = model.fireRate, masks = null },
) {
  if (!Number.isInteger(steps) || steps < 0) throw new Error("steps must be a non-negative integer");
  const T = tapeSlots;
  const size = model.C * T;
  const states = new Float32Array((steps + 1) * size);
  states.set(initialState(model, task, input, T), 0);
  const random = mulberry32(seed);
  const mask = new Uint8Array(T);
  const scratch = createScratch(model);
  for (let t = 0; t < steps; t++) {
    let stepMask = null;
    if (masks) {
      const given = masks[t];
      for (let x = 0; x < T; x++) mask[x] = typeof given === "string" ? Number(given[x]) : given[x];
      stepMask = mask;
    } else if (fireRate < 1) {
      for (let x = 0; x < T; x++) mask[x] = random() < fireRate ? 1 : 0;
      stepMask = mask;
    }
    const current = states.subarray(t * size, (t + 1) * size);
    const next = states.subarray((t + 1) * size, (t + 2) * size);
    step(model, current, T, stepMask, next, scratch);
  }
  return { model, task, input, T, steps, states };
}

export function value(result, t, channel, x) {
  const { model, T } = result;
  return result.states[t * model.C * T + channel * T + x];
}

// --- Readout (ncpu_computer_1d/tape.py and tasks.semantic_correct) ---

export function symbol(model, v) {
  if (v > model.threshold) return "1";
  if (v < -model.threshold) return "0";
  return "B";
}

export function tapeSymbols(result, t) {
  let raw = "";
  for (let x = 0; x < result.T; x++) raw += symbol(result.model, value(result, t, result.model.io, x));
  return raw;
}

export function decodeSingle(raw) {
  const end = raw.indexOf("B");
  return end < 0 ? { output: raw, terminated: false } : { output: raw.slice(0, end), terminated: true };
}

// Correct output followed by one blank (when the tape has room); later cells are ignored.
export function semanticCorrect(raw, target) {
  const required = Math.min(target.length + 1, raw.length);
  for (let x = 0; x < required; x++) {
    if (raw[x] !== (x < target.length ? target[x] : "B")) return false;
  }
  return true;
}
