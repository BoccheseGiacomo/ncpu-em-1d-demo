"""Export a frozen NCPU-EM-1D checkpoint for the browser applet.

Writes two files:
  model.json           weights, configuration, and metadata used by nca.js
  tests/fixtures.json  PyTorch reference rollouts used by tests/verify.mjs

Usage (from this repository, with an environment that has PyTorch):
  python tools/export_model.py --project <path to ncpu-em-1d> [--checkpoint ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ACTIVATIONS = {"relu", "softplus"}
SUPPORTED_GATES = {"none", "linear", "sigmoid", "tanh", "relu"}


def rounded(values):
    # Nine significant digits identify every float32 exactly.
    if isinstance(values, dict):
        return {key: rounded(value) for key, value in values.items()}
    if isinstance(values, list):
        return [rounded(value) for value in values]
    return float(f"{values:.9g}")


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def random_bits(generator: random.Random, length: int) -> str:
    return "".join(generator.choice("01") for _ in range(length))


def masks_and_rollout(model, initial, steps, seed):
    """Run the real model and recover the exact firing masks it drew."""
    shape = (steps, initial.shape[0], 1, initial.shape[-1])
    torch.manual_seed(seed)
    if model.config.fire_rate < 1.0:
        masks = torch.rand(shape) < model.config.fire_rate
    else:
        masks = torch.ones(shape, dtype=torch.bool)
    torch.manual_seed(seed)
    rollout = model(initial, steps)
    # Self-check: replaying the recovered masks step by step must agree exactly.
    state = initial
    filters = model.hidden_filters()
    for index in range(steps):
        state = model._step(state, 0.0, filters, masks[index])
        if not torch.equal(state, rollout[:, index + 1]):
            raise AssertionError("recovered firing masks do not reproduce forward()")
    return masks, rollout


class Exporter:
    def __init__(self, project: Path, checkpoint_path: Path):
        sys.path.insert(0, str(project / "src"))
        import ncpu_computer_1d as package
        from ncpu_computer_1d.tasks import output_length_bound

        self.package = package
        self.output_length_bound = output_length_bound
        self.project = project
        self.checkpoint_path = checkpoint_path
        self.model, self.config, self.checkpoint = package.load_model(
            checkpoint_path, "cpu"
        )
        self.model.eval()
        model_config = self.config.model
        if model_config.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"unsupported activation: {model_config.activation}")
        if model_config.gate not in SUPPORTED_GATES:
            raise ValueError(f"unsupported gate: {model_config.gate}")
        training = self.config.training
        self.free_per_slot = training.free_steps_per_tape_slot
        self.supervision_ratio = training.supervision_ratio

    def timing(self, tape_slots: int, free_steps: int | None = None):
        if free_steps is None:
            free_steps = self.package.round_half_up(self.free_per_slot * tape_slots)
        supervision = self.package.round_half_up(self.supervision_ratio * free_steps)
        return free_steps, supervision

    def fit_tape(self, task: str, length: int, tape_slots: int) -> int:
        return max(
            tape_slots,
            self.output_length_bound(task, length) + 1,
            length + 1,
            self.config.model.program_length,
        )

    def training_range(self):
        training = self.config.training
        minimum = self.config.model.program_length
        tapes = [
            self.package.tape_bounds(training, base, minimum)
            for base in training.base_tape_slots
        ]
        inputs = [
            self.package.varied_bounds(base, training.input_variation)
            for base in training.base_input_max_lengths
        ]
        return {
            "updates": training.updates,
            "free_steps_per_tape_slot": training.free_steps_per_tape_slot,
            "supervision_ratio": training.supervision_ratio,
            "time_variation": training.time_variation,
            "base_tape_slots": list(training.base_tape_slots),
            "base_input_max_lengths": list(training.base_input_max_lengths),
            "tape_min": min(low for low, _ in tapes),
            "tape_max": max(high for _, high in tapes),
            "input_max": max(high for _, high in inputs),
        }

    def model_document(self):
        model, model_config = self.model, self.config.model
        validation = [
            entry
            for entry in self.checkpoint.get("history", [])
            if entry.get("validation_loss") is not None
        ]
        best = min(validation, key=lambda entry: entry["validation_loss"], default=None)
        output = model.rule.output
        with torch.no_grad():
            weights = {
                "hidden_filters": model.hidden_filters().tolist(),
                "hidden_bias": model.rule.hidden.bias.tolist(),
                "output_weight": output.weight[:, :, 0].tolist(),
                "output_bias": None if output.bias is None else output.bias.tolist(),
                "programs": model.programs.tolist(),
            }
        if weights["output_bias"] is None:
            del weights["output_bias"]
        return {
            "format": "ncpu-em-1d-web-v1",
            "source": {
                "checkpoint": self.checkpoint_path.relative_to(
                    self.project
                ).as_posix(),
                "md5": file_md5(self.checkpoint_path),
                "format_version": self.checkpoint["format_version"],
                "update": int(self.checkpoint["update"]),
                "parameter_count": model.parameter_count,
                "best_validation_update": None if best is None else best["update"],
                "best_validation_loss": None
                if best is None
                else float(best["validation_loss"]),
                "validation_accuracies": None
                if best is None
                else best["validation_accuracies"],
            },
            "tasks": list(model.task_names),
            "task_weights": [
                float(weight) for _, weight in self.checkpoint["task_specs"]
            ],
            "config": {
                "channels": model_config.channels,
                "program_channels": model_config.program_channels,
                "computation_channels": model_config.computation_channels,
                "io_channel": model_config.io_channel,
                "hidden_size": model_config.hidden_size,
                "radius": model_config.radius,
                "program_length": model_config.program_length,
                "program_placement": model_config.program_placement,
                "program_mode": model_config.program_mode,
                "mutable": model.mutable_mask.flatten().tolist(),
                "activation": model_config.activation,
                "gate": model_config.gate,
                "fire_rate": model_config.fire_rate,
                "state_leak": model_config.state_leak,
                "max_abs_state": model_config.max_abs_state,
                "ternary_threshold": self.package.TERNARY_THRESHOLD,
            },
            "training": self.training_range(),
            "test_cases": [
                {
                    "name": case.name,
                    "tape_slots": case.tape_slots,
                    "input_length": case.input_length,
                    "free_steps": case.free_steps,
                    "supervision_steps": case.supervision_steps,
                }
                for case in self.config.test_cases
            ],
            "weights": rounded(weights),
        }

    def replay_cases(self):
        """Exact rollouts, with their firing masks, for every task."""
        training = self.config.training
        test = self.config.test_cases[0]
        input_max = self.training_range()["input_max"]
        specs = []
        for task in self.model.task_names:
            specs += [
                (task, 0, 5, None, 0),
                (task, 2, 5, None, 0),
                (task, 7, 11, None, 0),
                (task, input_max, self.training_range()["tape_max"], None, 0),
                (task, test.input_length, test.tape_slots, test.free_steps, 0),
            ]
        # One rollout far beyond the supervision window checks long-run stability.
        specs.append(
            (
                self.model.task_names[0],
                training.base_input_max_lengths[-2],
                training.base_tape_slots[-2],
                None,
                400,
            )
        )
        cases = []
        for seed, (task, length, tape, free_steps, extra) in enumerate(specs, 100):
            symbols = random_bits(random.Random(seed), length)
            tape = self.fit_tape(task, length, tape)
            free_steps, supervision = self.timing(tape, free_steps)
            steps = free_steps + supervision + extra
            inputs = self.package.encode_strings((symbols,), tape)
            indices = torch.tensor([self.model.task_index(task)])
            with torch.no_grad():
                initial = self.model.initial_state(inputs, indices)
                masks, rollout = masks_and_rollout(self.model, initial, steps, seed)
            io = rollout[0, :, self.model.config.io_channel]
            cases.append(
                {
                    "name": f"{task}:{symbols or '<empty>'}@{tape}",
                    "task": task,
                    "input": symbols,
                    "target": self.package.task_target(task, symbols),
                    "tape_slots": tape,
                    "free_steps": free_steps,
                    "supervision_steps": supervision,
                    "steps": steps,
                    "masks": [
                        "".join("1" if bit else "0" for bit in step[0, 0].tolist())
                        for step in masks
                    ],
                    "io": rounded(io.tolist()),
                    "final_state": rounded(rollout[0, -1].tolist()),
                    "decoded": self.package.tensor_to_symbols(io[-1]),
                }
            )
        return cases

    def accuracy_cases(self, examples: int):
        """Sampled accuracies with PyTorch's own random firing, per task."""
        training = self.config.training
        test = self.config.test_cases[0]
        specs = []
        for task in self.model.task_names:
            specs += [
                (
                    task,
                    "training_max",
                    training.base_tape_slots[-1],
                    training.base_input_max_lengths[-1],
                    None,
                ),
                (task, test.name, test.tape_slots, test.input_length, test.free_steps),
            ]
        results = []
        for seed, (task, case, tape, length, free_steps) in enumerate(specs, 500):
            tape = self.fit_tape(task, length, tape)
            free_steps, supervision = self.timing(tape, free_steps)
            generator = random.Random(seed)
            inputs = [random_bits(generator, length) for _ in range(examples)]
            targets = [self.package.task_target(task, value) for value in inputs]
            encoded = self.package.encode_strings(inputs, tape)
            target_tape = self.package.encode_strings(targets, tape)
            lengths = torch.tensor([len(value) for value in targets])
            indices = torch.full((examples,), self.model.task_index(task))
            torch.manual_seed(seed)
            with torch.no_grad():
                rollout = self.model(
                    self.model.initial_state(encoded, indices),
                    free_steps + supervision,
                    io_only=True,
                )
            semantic = self.package.semantic_correct(
                self.package.quantize(rollout),
                target_tape.to(torch.int8),
                lengths,
                "single",
            )
            window = semantic[:, free_steps + 1 :].float()
            results.append(
                {
                    "name": f"{task}:{case}",
                    "task": task,
                    "tape_slots": tape,
                    "free_steps": free_steps,
                    "supervision_steps": supervision,
                    "inputs": inputs,
                    "python_window_semantic": float(window.mean()),
                    "python_final_semantic": float(window[:, -1].mean()),
                }
            )
            print(
                f"{task:<17} {case:<16} window={results[-1]['python_window_semantic']:.2%}"
                f" final={results[-1]['python_final_semantic']:.2%}"
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True, help="ncpu-em-1d path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/preserved_run/best.pt"),
        help="checkpoint path, relative to --project",
    )
    parser.add_argument("--accuracy-examples", type=int, default=300)
    args = parser.parse_args()
    project = args.project.resolve()
    exporter = Exporter(project, (project / args.checkpoint).resolve())

    document = exporter.model_document()
    (ROOT / "model.json").write_text(
        json.dumps(document, separators=(",", ":")), encoding="utf-8"
    )
    fixtures = {
        "model_md5": document["source"]["md5"],
        "torch_version": torch.__version__,
        "replay": exporter.replay_cases(),
        "accuracy": exporter.accuracy_cases(args.accuracy_examples),
    }
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "fixtures.json").write_text(
        json.dumps(fixtures, separators=(",", ":")), encoding="utf-8"
    )
    print(f"wrote model.json ({document['source']['parameter_count']:,} parameters)")
    print(f"wrote tests/fixtures.json ({len(fixtures['replay'])} replay cases)")


if __name__ == "__main__":
    main()
