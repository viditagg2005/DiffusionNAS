from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .provenance import collect
from .schema import SCHEMA_VERSION


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_id(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{config['method']}_{digest}"


def execute(config: dict[str, Any], project_root: Path, dry_run: bool = False) -> Path:
    output_root = project_root / config.get("output_root", "artifacts/runs")
    run_dir = output_root / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = run_dir / "config.json"
    _write_json(resolved_config, config)

    fidelity = "adapted" if config["method"] == "autodiffusion" else "paper_implementation"
    if config["method"] == "baseline":
        fidelity = "reference"
    command = [
        sys.executable,
        "-m",
        "diffusionnas.sd15_worker",
        "--config",
        str(resolved_config.resolve()),
        "--output-dir",
        str(run_dir.resolve()),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": "dry_run" if dry_run else "running",
        "method": config["method"],
        "model_id": config["model"]["id"],
        "implementation_fidelity": fidelity,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "provenance": collect(project_root),
    }
    _write_json(run_dir / "manifest.json", manifest)

    if dry_run:
        (run_dir / "stdout.log").write_text("dry run: command not executed\n", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        _write_json(
            run_dir / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "dry_run",
                "method": config["method"],
                "sample_count": 0,
                "latency_ms": None,
                "peak_allocated_mb": None,
                "peak_reserved_mb": None,
                "quality_metrics": {},
                "energy_joules_per_image": None,
            },
        )
        return run_dir

    env = os.environ.copy()
    source_path = project_root / "src"
    deepcache_path = project_root / "third_party" / "DeepCache"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_path), str(deepcache_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(command, cwd=project_root, env=env, text=True, capture_output=True)
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    manifest["status"] = "succeeded" if completed.returncode == 0 else "failed"
    manifest["exit_code"] = completed.returncode
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "manifest.json", manifest)
    if completed.returncode:
        raise RuntimeError(f"run failed ({completed.returncode}); inspect {run_dir / 'stderr.log'}")
    return run_dir
