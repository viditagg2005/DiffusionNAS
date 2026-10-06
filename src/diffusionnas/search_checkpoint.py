"""Crash-safe candidate journal shared by the two search algorithms.

On resume, the optimizer replays its seeded decisions and looks up completed
candidate objectives here. This also reconstructs ECAD's population and RNG
state without serializing Python objects or rerunning completed GPU work.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from .provenance import cuda_device


SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Never expose a partially written checkpoint after a job interruption."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SearchPaused(RuntimeError):
    """The current allocation should end; resubmit against the same output."""

    def __init__(self, checkpoint_path: Path, completed: int, reason: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.completed = completed
        self.reason = reason
        super().__init__(f"{reason}; {completed} candidates checkpointed at {checkpoint_path}")


class SearchCheckpoint:
    """Append one durable record after each fully scored candidate."""

    def __init__(
        self,
        *,
        output_dir: Path,
        config: dict[str, Any],
        method: str,
        dry_run: bool,
        clip_model_id: str,
        score_device: str,
        batch_size: int,
        reference_point: dict[str, float],
        max_walltime_seconds: float | None = None,
        max_new_evaluations: int | None = None,
    ) -> None:
        if max_walltime_seconds is not None and max_walltime_seconds < 0:
            raise ValueError("max_walltime_seconds must be non-negative")
        if max_new_evaluations is not None and max_new_evaluations < 0:
            raise ValueError("max_new_evaluations must be non-negative")
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "checkpoint.json"
        self._started = time.monotonic()
        self._last_accounted = self._started
        self._deadline = None if max_walltime_seconds is None else self._started + max_walltime_seconds
        self._max_new_evaluations = max_new_evaluations
        self._new_evaluations = 0
        self.budget = config["budget"]
        protocol = {
            "checkpoint_schema_version": SCHEMA_VERSION,
            "method": method,
            "config": config,
            "dry_run": dry_run,
            "clip_model_id": clip_model_id,
            "score_device": score_device,
            "batch_size": batch_size,
            "reference_point": reference_point,
        }
        encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.gpu = None if dry_run else cuda_device()
        if not dry_run and self.gpu is None:
            raise RuntimeError("CUDA GPU is unavailable for a real search")

        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported checkpoint schema: {self.path}")
            if self.state.get("protocol_sha256") != self.fingerprint:
                raise ValueError(f"search config/protocol changed; refusing to reuse {self.path}")
            if self.state.get("gpu") != self.gpu:
                raise ValueError(f"GPU differs from checkpoint; refusing mixed-hardware timings in {self.path}")
            if not isinstance(self.state.get("evaluations"), list) or len(self.state["evaluations"]) > self.budget:
                raise ValueError(f"invalid evaluation journal: {self.path}")
            print(f"Resuming {method} search from {len(self.state['evaluations'])}/{self.budget} candidates")
            self.state["status"] = "running"
            self._save()
        else:
            runs_dir = self.output_dir / "runs"
            if any((self.output_dir / name).exists() for name in ("search_results.json", "pareto_front.json")) or (
                runs_dir.exists() and any(runs_dir.glob("*/manifest.json"))
            ):
                raise ValueError(f"existing search artifacts have no checkpoint; use a new output directory: {self.output_dir}")
            self.state = {
                "schema_version": SCHEMA_VERSION,
                "protocol_sha256": self.fingerprint,
                "search_method": method,
                "budget": self.budget,
                "gpu": self.gpu,
                "status": "running",
                "created_at": _now(),
                "updated_at": _now(),
                "elapsed_wall_seconds": 0.0,
                "evaluations": [],
            }
            self._save()

    @property
    def elapsed_wall_seconds(self) -> float:
        return float(self.state["elapsed_wall_seconds"]) + time.monotonic() - self._last_accounted

    def _save(self) -> None:
        current = time.monotonic()
        self.state["elapsed_wall_seconds"] += current - self._last_accounted
        self._last_accounted = current
        self.state["updated_at"] = _now()
        _atomic_json(self.path, self.state)

    def evaluate(
        self,
        candidate_id: int,
        chromosome: list[int],
        generation: int,
        evaluator: Callable[[], tuple[Path, dict[str, float]]],
    ) -> tuple[Path, dict[str, float]]:
        entries = self.state["evaluations"]
        if candidate_id < len(entries):
            record = entries[candidate_id]
            if record["candidate_id"] != candidate_id or record["chromosome"] != chromosome or record["generation"] != generation:
                raise ValueError(f"candidate {candidate_id} differs from checkpoint; refusing nondeterministic resume")
            run_dir = Path(record["run_dir"])
            if not run_dir.is_dir():
                raise ValueError(f"checkpointed run directory is missing: {run_dir}")
            return run_dir, dict(record["objectives"])
        if candidate_id != len(entries):
            raise ValueError(f"candidate {candidate_id} is out of checkpoint order")
        if self._max_new_evaluations is not None and self._new_evaluations >= self._max_new_evaluations:
            self._pause(len(entries), "evaluation limit reached")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self._pause(len(entries), "wall-time budget reached")

        run_dir, objectives = evaluator()
        if not run_dir.is_dir():
            raise ValueError(f"candidate run directory is missing: {run_dir}")
        entries.append({
            "candidate_id": candidate_id,
            "generation": generation,
            "chromosome": list(chromosome),
            "run_dir": str(run_dir.resolve()),
            "objectives": dict(objectives),
        })
        self._new_evaluations += 1
        self._save()
        return run_dir, objectives

    def _pause(self, completed: int, reason: str) -> None:
        self.state["status"] = "paused"
        self.state["pause_reason"] = reason
        self._save()
        raise SearchPaused(self.path, completed, reason)

    def mark_complete(self) -> None:
        if len(self.state["evaluations"]) != self.budget:
            raise ValueError("cannot mark an incomplete search as succeeded")
        self.state["status"] = "succeeded"
        self.state.pop("pause_reason", None)
        self._save()
