from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


BASE_OBJECTIVES = (
    ("clip_score", "max"),
    ("latency_p50_ms", "min"),
    ("peak_allocated_mb", "min"),
)
ENERGY_OBJECTIVE = ("energy_joules_per_image", "min")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _objective_values(summary: dict[str, Any], include_energy: bool) -> dict[str, float]:
    try:
        values = {
            "clip_score": float(summary["quality_metrics"]["clip_score"]["mean"]),
            "latency_p50_ms": float(summary["latency_ms"]["p50"]),
            "peak_allocated_mb": float(summary["peak_allocated_mb"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing calibration CLIP, latency, or memory metric") from exc
    if include_energy:
        energy = summary.get("energy_joules_per_image")
        if energy is None:
            raise ValueError("energy requested but not measured")
        values["energy_joules_per_image"] = float(energy)
    return values


def dominates(
    left: dict[str, float], right: dict[str, float], objectives: Iterable[tuple[str, str]]
) -> bool:
    no_worse = True
    strictly_better = False
    for name, direction in objectives:
        if direction == "min":
            no_worse &= left[name] <= right[name]
            strictly_better |= left[name] < right[name]
        elif direction == "max":
            no_worse &= left[name] >= right[name]
            strictly_better |= left[name] > right[name]
        else:
            raise ValueError(f"unknown objective direction: {direction}")
    return bool(no_worse and strictly_better)


def nondominated(points: list[dict[str, Any]], objectives: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    objective_list = tuple(objectives)
    return [
        point
        for index, point in enumerate(points)
        if not any(
            dominates(other["objectives"], point["objectives"], objective_list)
            for other_index, other in enumerate(points)
            if other_index != index
        )
    ]


def construct_front(
    run_dirs: Iterable[str | Path], output_path: str | Path, include_energy: bool = False
) -> dict[str, Any]:
    objectives = list(BASE_OBJECTIVES)
    if include_energy:
        objectives.append(ENERGY_OBJECTIVE)
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir).resolve()
        try:
            manifest = _read_json(run_path / "manifest.json")
            summary = _read_json(run_path / "summary.json")
            config = _read_json(run_path / "config.json")
            if manifest.get("status") != "succeeded":
                raise ValueError(f"run status is {manifest.get('status')!r}")
            values = _objective_values(summary, include_energy)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            skipped.append({"run_dir": str(run_path), "reason": str(exc)})
            continue
        eligible.append(
            {
                "run_id": manifest["run_id"],
                "run_dir": str(run_path),
                "method": manifest["method"],
                "implementation_fidelity": manifest["implementation_fidelity"],
                "policy": {
                    "inference": config["inference"],
                    "deepcache": config.get("deepcache"),
                },
                "objectives": values,
            }
        )
    if not eligible:
        raise ValueError("no eligible scored runs were found")
    front_points = nondominated(eligible, objectives)
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objectives": [{"name": name, "direction": direction} for name, direction in objectives],
        "eligible_count": len(eligible),
        "nondominated_count": len(front_points),
        "points": sorted(front_points, key=lambda point: point["objectives"]["latency_p50_ms"]),
        "skipped": skipped,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fieldnames = ["run_id", "method", "implementation_fidelity"] + [name for name, _ in objectives]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in result["points"]:
            writer.writerow(
                {
                    "run_id": point["run_id"],
                    "method": point["method"],
                    "implementation_fidelity": point["implementation_fidelity"],
                    **point["objectives"],
                }
            )
    return result

