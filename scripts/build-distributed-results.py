#!/usr/bin/env python3
"""Build the compact data bundle used by the distributed-training blog plots.

The input is ``csv/runs.csv`` from the nanoTitan results repository.  Each
manifest entry points to a per-run ``scalars.csv`` file.  Only the benchmark
runs used to compare DP, PP, EP, and their compositions are included; the
learning-rate and DP bucket-size sweeps are intentionally left to their
specialised plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "assets"
    / "_draft"
    / "distributed"
    / "distributed-results-data.js"
)
START_STEP = 50

METRIC_COLUMNS = {
    "throughput": "train/tokens_per_second",
    "step_time": "time/step_time",
    "forward_time": "time/forward_completion_time",
}
MEMORY_COLUMNS = (
    "memory/pp_first_peak_allocated_gib",
    "memory/pp_middle_peak_allocated_gib",
    "memory/pp_last_peak_allocated_gib",
)
REQUIRED_COLUMNS = {
    "step",
    "occurrence",
    "train/tokens_per_step",
    *METRIC_COLUMNS.values(),
    *MEMORY_COLUMNS,
}
STRATEGY_ORDER = {
    "DP": 0,
    "EP": 1,
    "PP": 2,
    "DP + EP": 3,
    "DP + PP": 4,
    "PP + EP": 5,
    "DP + PP + EP": 6,
}
BUCKET_PATTERN = re.compile(r"^GPU-2_DP_BS-20__runtime-bucket_size-(?P<size>\d+)$")
LR_PATTERN = re.compile(r"^GPU-1_DP_BS-20__optim-lr-(?P<learning_rate>.+)$")

PP_2_PATTERN = re.compile(
    r"^GPU-2_PP_BS-42__runtime-pipeline_schedule-(?P<schedule>gpipe|1f1b)"
    r"__runtime-activation_checkpointing-(?P<checkpoint>true|false)"
    r"__runtime-num_microbatches-(?P<microbatches>\d+)$"
)
PP_4_PATTERN = re.compile(
    r"^GPU-4_PP_BS-21_1f1B_AC-ON__runtime-num_microbatches-(?P<microbatches>\d+)$"
)


@dataclass(frozen=True)
class RunIdentity:
    gpus: int
    strategy: str
    config: str
    short_config: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise nanoTitan benchmark runs for the blog visualisations."
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        required=True,
        help="Path to nanoTitan/csv/runs.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JavaScript data bundle.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=START_STEP,
        help="First training step included in steady-state summaries.",
    )
    return parser.parse_args(argv)


def parse_identity(run: str) -> RunIdentity | None:
    match = PP_2_PATTERN.fullmatch(run)
    if match:
        schedule = "GPipe" if match.group("schedule") == "gpipe" else "1F1B"
        checkpoint = "on" if match.group("checkpoint") == "true" else "off"
        microbatches = int(match.group("microbatches"))
        config = f"2×PP · {schedule} · AC {checkpoint} · M={microbatches}"
        short_config = f"{schedule} · AC {checkpoint} · M={microbatches}"
        return RunIdentity(2, "PP", config, short_config)

    match = PP_4_PATTERN.fullmatch(run)
    if match:
        microbatches = int(match.group("microbatches"))
        config = f"4×PP · 1F1B · AC on · M={microbatches}"
        short_config = f"1F1B · AC on · M={microbatches}"
        return RunIdentity(4, "PP", config, short_config)

    exact: dict[str, RunIdentity] = {
        "GPU=2_DP_BS=21": RunIdentity(2, "DP", "2×DP", "2×DP"),
        "GPU=2_EP_BS=21": RunIdentity(2, "EP", "2×EP", "2×EP"),
        "GPU=4_DP_BS=21": RunIdentity(4, "DP", "4×DP", "4×DP"),
        "GPU=4_EP_BS=21": RunIdentity(4, "EP", "4×EP", "4×EP"),
        "GPU=4_2DP-2PP_BS=21": RunIdentity(
            4, "DP + PP", "2 DP × 2 PP", "2DP×2PP"
        ),
        "GPU=4_EP-DP_BS=21": RunIdentity(
            4, "DP + EP", "2 DP × 2 EP", "2DP×2EP"
        ),
        "GPU=4_EP-PP_BS=21": RunIdentity(
            4, "PP + EP", "2 PP × 2 EP", "2PP×2EP"
        ),
        "GPU=8_DP_BS=5": RunIdentity(8, "DP", "8×DP", "8×DP"),
        "GPU=8_EP_BS=5": RunIdentity(8, "EP", "8×EP", "8×EP"),
        "GPU=8_PP_BS=5": RunIdentity(8, "PP", "8×PP", "8×PP"),
        "GPU=8_2DP-4EP_BS=5": RunIdentity(
            8, "DP + EP", "2 DP × 4 EP", "2DP×4EP"
        ),
        "GPU=8_4DP-2EP_BS=5": RunIdentity(
            8, "DP + EP", "4 DP × 2 EP", "4DP×2EP"
        ),
        "GPU=8_2DP-4PP_BS=5": RunIdentity(
            8, "DP + PP", "2 DP × 4 PP", "2DP×4PP"
        ),
        "GPU=8_4DP-2PP_BS=5": RunIdentity(
            8, "DP + PP", "4 DP × 2 PP", "4DP×2PP"
        ),
        "GPU=8_2EP-4PP_BS=5": RunIdentity(
            8, "PP + EP", "4 PP × 2 EP", "4PP×2EP"
        ),
        "GPU=8_4EP-2PP_BS=5": RunIdentity(
            8, "PP + EP", "2 PP × 4 EP", "2PP×4EP"
        ),
        "GPU=8_3D_BS=5": RunIdentity(
            8, "DP + PP + EP", "Three-axis composition", "3D"
        ),
    }
    return exact.get(run)


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite scalar value: {value!r}")
    return number


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarise(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "q1": quantile(values, 0.25),
        "q3": quantile(values, 0.75),
        "n": len(values),
    }


def latest_rows(path: Path, start_step: int) -> list[dict[str, str]]:
    by_step: dict[int, tuple[int, dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            step = int(row["step"])
            if step < start_step:
                continue
            occurrence = int(row["occurrence"])
            previous = by_step.get(step)
            if previous is None or occurrence >= previous[0]:
                by_step[step] = (occurrence, row)
    return [by_step[step][1] for step in sorted(by_step)]


def metric_values(rows: Iterable[dict[str, str]]) -> dict[str, list[float]]:
    values = {
        "throughput": [],
        "step_time": [],
        "forward_time": [],
        "non_forward_time": [],
        "peak_memory": [],
        "first_stage_memory": [],
        "middle_stage_memory": [],
        "last_stage_memory": [],
    }
    for row in rows:
        step_time = finite_float(row[METRIC_COLUMNS["step_time"]])
        forward_time = finite_float(row[METRIC_COLUMNS["forward_time"]])
        memory = max(finite_float(row[column]) for column in MEMORY_COLUMNS)
        values["throughput"].append(
            finite_float(row[METRIC_COLUMNS["throughput"]])
        )
        values["step_time"].append(step_time)
        values["forward_time"].append(forward_time)
        values["non_forward_time"].append(step_time - forward_time)
        values["peak_memory"].append(memory)
        values["first_stage_memory"].append(finite_float(row[MEMORY_COLUMNS[0]]))
        values["middle_stage_memory"].append(finite_float(row[MEMORY_COLUMNS[1]]))
        values["last_stage_memory"].append(finite_float(row[MEMORY_COLUMNS[2]]))
    return values


def hardware_for(gpus: int) -> dict[str, str | int]:
    if gpus == 8:
        return {"gpu": "RTX 3060", "experts": 16}
    return {"gpu": "RTX 3090", "experts": 20}


def build_benchmark_records(
    candidates: list[tuple[str, RunIdentity, list[dict[str, str]]]],
) -> list[dict[str, Any]]:
    common_end_by_gpu = {
        gpus: min(int(rows[-1]["step"]) for _, identity, rows in candidates if identity.gpus == gpus)
        for gpus in {identity.gpus for _, identity, _ in candidates}
    }
    records: list[dict[str, Any]] = []
    for run, identity, all_rows in candidates:
        common_end = common_end_by_gpu[identity.gpus]
        rows = [row for row in all_rows if int(row["step"]) <= common_end]
        values = metric_values(rows)
        tokens_per_step = statistics.median(
            finite_float(row["train/tokens_per_step"]) for row in rows
        )
        record = {
            "run": run,
            "gpus": identity.gpus,
            "strategy": identity.strategy,
            "config": identity.config,
            "shortConfig": identity.short_config,
            "hardware": hardware_for(identity.gpus),
            "tokensPerStep": tokens_per_step,
            "firstStep": int(rows[0]["step"]),
            "lastStep": int(rows[-1]["step"]),
            "metrics": {
                metric: summarise(metric_samples)
                for metric, metric_samples in values.items()
            },
        }
        records.append(record)
    records.sort(
        key=lambda record: (
            record["gpus"],
            STRATEGY_ORDER[record["strategy"]],
            record["config"],
        )
    )
    return records


def build_bucket_sweep(
    candidates: list[tuple[int, list[dict[str, str]]]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    common_steps = set.intersection(
        *(set(int(row["step"]) for row in rows) for _, rows in candidates)
    )
    by_size: dict[int, dict[int, dict[str, str]]] = {
        size: {int(row["step"]): row for row in rows if int(row["step"]) in common_steps}
        for size, rows in candidates
    }
    if 0 not in by_size:
        raise ValueError("bucket-size sweep is missing the 0 MiB baseline")
    baseline = {
        step: finite_float(row["time/step_time"])
        - finite_float(row["time/forward_completion_time"])
        for step, row in by_size[0].items()
    }
    output: list[dict[str, Any]] = []
    for size in sorted(by_size):
        rows = by_size[size]
        deltas = [
            1000
            * (
                finite_float(rows[step]["time/step_time"])
                - finite_float(rows[step]["time/forward_completion_time"])
                - baseline[step]
            )
            for step in sorted(common_steps)
        ]
        non_forward = [
            finite_float(rows[step]["time/step_time"])
            - finite_float(rows[step]["time/forward_completion_time"])
            for step in sorted(common_steps)
        ]
        throughput = [
            finite_float(rows[step]["train/tokens_per_second"])
            for step in sorted(common_steps)
        ]
        output.append(
            {
                "bucketMiB": size,
                "deltaNonForwardMs": summarise(deltas),
                "nonForwardTime": summarise(non_forward),
                "throughput": summarise(throughput),
                "firstStep": min(common_steps),
                "lastStep": max(common_steps),
            }
        )
    return output


def build_lr_sweep(
    candidates: list[tuple[float, list[dict[str, str]]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for learning_rate, rows in sorted(candidates):
        points = [
            {
                "tokens": finite_float(row["train/total_tokens_seen"]),
                "loss": finite_float(row["train/total_loss"]),
                "gradNorm": finite_float(row["train/grad_norm"]),
            }
            for row in rows
        ]
        output.append({"learningRate": learning_rate, "points": points})
    return output


def build_payload(runs_csv: Path, start_step: int) -> dict[str, Any]:
    base = runs_csv.parent
    benchmark_candidates: list[tuple[str, RunIdentity, list[dict[str, str]]]] = []
    bucket_candidates: list[tuple[int, list[dict[str, str]]]] = []
    lr_candidates: list[tuple[float, list[dict[str, str]]]] = []
    with runs_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for manifest_row in reader:
            if manifest_row.get("status") != "ok":
                continue
            run = manifest_row["run"]
            identity = parse_identity(run)
            scalar_path = base / Path(manifest_row["output_csv"])
            if identity is not None:
                rows = latest_rows(scalar_path, start_step)
                if not rows:
                    raise ValueError(
                        f"{run} has no scalar observations at or after step {start_step}"
                    )
                benchmark_candidates.append((run, identity, rows))
                continue
            bucket_match = BUCKET_PATTERN.fullmatch(run)
            if bucket_match:
                rows = latest_rows(scalar_path, start_step)
                bucket_candidates.append((int(bucket_match.group("size")), rows))
                continue
            lr_match = LR_PATTERN.fullmatch(run)
            if lr_match:
                rows = latest_rows(scalar_path, 0)
                lr_candidates.append((float(lr_match.group("learning_rate")), rows))

    if not benchmark_candidates:
        raise ValueError("no benchmark runs matched the expected naming schemes")
    if any(not rows for _, _, rows in benchmark_candidates):
        empty_run = next(run for run, _, rows in benchmark_candidates if not rows)
        raise ValueError(
            f"{empty_run} has no scalar observations at or after step {start_step}"
        )
    records = build_benchmark_records(benchmark_candidates)
    return {
        "startStep": start_step,
        "generatedFrom": runs_csv.name,
        "strategyOrder": list(STRATEGY_ORDER),
        "records": records,
        "bucketSweep": build_bucket_sweep(bucket_candidates),
        "lrSweep": build_lr_sweep(lr_candidates),
    }


def write_bundle(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    destination.write_text(
        f"window.distributedResults = Object.freeze({encoded});\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_csv = args.runs_csv.resolve()
    output = args.output.resolve()
    if not runs_csv.is_file():
        raise SystemExit(f"runs manifest not found: {runs_csv}")
    payload = build_payload(runs_csv, args.start_step)
    write_bundle(output, payload)
    gpu_counts = sorted({record["gpus"] for record in payload["records"]})
    print(
        f"Wrote {len(payload['records'])} benchmark summaries for GPU counts "
        f"{gpu_counts}, {len(payload['bucketSweep'])} bucket settings, and "
        f"{len(payload['lrSweep'])} learning rates to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
