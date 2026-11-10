"""Command line for the v3 search: ``python -m diffusionnas.nas_cli --config C --output D --reference-run R``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nas_search import run_nas_search
from .search_checkpoint import SearchPaused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diffusionnas-nas")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-walltime-seconds", type=float)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        result = run_nas_search(config, Path(__file__).resolve().parents[2], args.output, args.reference_run,
                                dry_run=args.dry_run, score_device=args.device, batch_size=args.batch_size,
                                max_walltime_seconds=args.max_walltime_seconds)
    except SearchPaused as exc:
        print(json.dumps({"status": "paused", "completed": exc.completed, "reason": exc.reason}))
        return 75
    print(json.dumps({k: result[k] for k in ("search_method", "status", "evaluated_count", "final_hypervolume")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
