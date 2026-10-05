"""Measure a checkpoint's accuracy in PyTorch and write results.json for the page.

Protocol (all at the model's training schedule: T free steps per tape cell,
then supervision_ratio x T supervised steps; semantic accuracy at the last step):
  ID      every string up to each base input maximum on its base tape,
          averaged over the base pairs (as in training validation)
  OOD +k  1,000 random strings of exactly (input_max + k) bits on a tape of
          (bits + 6) cells, where input_max is the largest training input

Usage (from this repository, with an environment that has PyTorch):
  python tools/measure_accuracy.py --project <path to ncpu-em-1d> --checkpoint ... \
      [--firing sync|async|both] [--ood 3 5 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "sync": "every cell updates at every step (fire rate 1.0)",
    "async": "random firing with the trained probability, as in training",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True, help="ncpu-em-1d path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="relative to --project")
    parser.add_argument("--firing", choices=("sync", "async", "both"), default="sync")
    parser.add_argument("--ood", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--examples", type=int, default=1000)
    args = parser.parse_args()

    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    import ncpu_computer_1d as package

    checkpoint_path = (project / args.checkpoint).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained, config, _ = package.load_model(checkpoint_path, device)
    training = config.training
    names = trained.task_names
    models = {}
    for mode in ("sync", "async") if args.firing == "both" else (args.firing,):
        if mode == "async":
            models[mode] = trained
        else:
            model = package.NeuralCellularAutomaton(replace(config.model, fire_rate=1.0), names)
            model.load_state_dict(trained.state_dict())
            models[mode] = model.to(device).eval()

    def schedule(tape):
        free = package.round_half_up(training.free_steps_per_tape_slot * tape)
        return free + package.round_half_up(training.supervision_ratio * free)

    @torch.no_grad()
    def accuracy(model, task, inputs, tape, seed):
        targets = [package.task_target(task, value) for value in inputs]
        encoded = package.encode_strings(inputs, tape).to(device)
        target_tape = package.encode_strings(targets, tape).to(device).to(torch.int8)
        lengths = torch.tensor([len(value) for value in targets], device=device)
        indices = torch.full((len(inputs),), model.task_index(task), device=device)
        steps = schedule(tape)
        torch.manual_seed(seed)
        final = model(model.initial_state(encoded, indices), steps, io_only=True, start_step=steps)
        correct = package.semantic_correct(package.quantize(final[:, -1]), target_tape, lengths, "single")
        return correct.float().sum().item()

    input_max = max(
        package.varied_bounds(base, training.input_variation)[1]
        for base in training.base_input_max_lengths
    )
    base_pairs = list(zip(training.base_tape_slots, training.base_input_max_lengths))
    groups = [
        {
            "key": "id",
            "label": "ID",
            "title": "In distribution",
            "detail": "every string up to "
            + ", ".join(str(length) for _, length in base_pairs)
            + " bits on "
            + ", ".join(str(tape) for tape, _ in base_pairs)
            + "-cell tapes, averaged over these pairs",
        }
    ] + [
        {
            "key": f"ood{k}",
            "label": f"OOD +{k}",
            "title": f"{k} bits beyond the training maximum",
            "detail": f"{args.examples:,} random {input_max + k}-bit strings on a "
            f"{input_max + k + 6}-cell tape",
        }
        for k in args.ood
    ]
    rows = {name: {group["key"]: {} for group in groups} for name in names}
    for mode, model in models.items():
        for name in names:
            scores = []
            for tape, length in base_pairs:
                inputs = list(package.binary_strings(length))
                correct = sum(
                    accuracy(model, name, inputs[i : i + 2048], tape, seed=1)
                    for i in range(0, len(inputs), 2048)
                )
                scores.append(correct / len(inputs))
            rows[name]["id"][mode] = round(100 * sum(scores) / len(scores), 2)
            for k in args.ood:
                bits = input_max + k
                generator = random.Random(1000 + k)
                inputs = ["".join(generator.choice("01") for _ in range(bits)) for _ in range(args.examples)]
                rows[name][f"ood{k}"][mode] = round(
                    100 * accuracy(model, name, inputs, bits + 6, seed=k) / args.examples, 2
                )
            print(f"{mode:<5} {name:<17} " + "  ".join(f"{g['label']} {rows[name][g['key']][mode]:6.2f}" for g in groups))

    document = {
        "model_md5": hashlib.md5(checkpoint_path.read_bytes()).hexdigest(),
        "metric": (
            "semantic accuracy (%) at the last step, measured in PyTorch with "
            "ncpu_computer_1d at the training schedule "
            f"({training.free_steps_per_tape_slot:g} free steps per tape cell, then "
            f"×{training.supervision_ratio:g})"
        ),
        "groups": groups,
        "modes": [
            {"key": mode, "label": mode, "asynchronous": mode == "async", "detail": MODES[mode]}
            for mode in models
        ],
        "rows": rows,
        "mean": {
            group["key"]: {
                mode: round(sum(rows[name][group["key"]][mode] for name in names) / len(names), 2)
                for mode in models
            }
            for group in groups
        },
    }
    (ROOT / "results.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote results.json")


if __name__ == "__main__":
    main()
