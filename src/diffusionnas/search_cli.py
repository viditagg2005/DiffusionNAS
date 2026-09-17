"""CLI entry point for DiffusionNAS policy search optimizers (Random Search and ECAD).

Allows executing search algorithms from the command line:
    python -m diffusionnas.search_cli random --config configs/random_search_sd15.json --dry-run
    python -m diffusionnas.search_cli ecad --config configs/ecad_sd15.json --dry-run
    python -m diffusionnas.search_cli run configs/random_search_sd15.json --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .ecad import run_ecad_search
from .quality import DEFAULT_CLIP_MODEL
from .random_search import run_random_search
from .search_space import validate_search_config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_search_config(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"search config file not found: {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {file_path}: {exc}") from exc
    validate_search_config(data)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diffusionnas-search", description="DiffusionNAS Policy Search Optimizers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: random
    rand_parser = subparsers.add_parser("random", help="Run uniform random search baseline")
    rand_parser.add_argument("--config", required=True, help="Path to search config JSON")
    rand_parser.add_argument("--output", help="Override output directory")
    rand_parser.add_argument("--dry-run", action="store_true", help="Simulate search without GPU compute")
    rand_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP scoring model")
    rand_parser.add_argument("--device", default="cuda", help="CLIP scoring device")
    rand_parser.add_argument("--batch-size", type=int, default=8, help="CLIP scoring batch size")

    # Subcommand: ecad
    ecad_parser = subparsers.add_parser("ecad", help="Run ECAD evolutionary search")
    ecad_parser.add_argument("--config", required=True, help="Path to search config JSON")
    ecad_parser.add_argument("--output", help="Override output directory")
    ecad_parser.add_argument("--dry-run", action="store_true", help="Simulate search without GPU compute")
    ecad_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP scoring model")
    ecad_parser.add_argument("--device", default="cuda", help="CLIP scoring device")
    ecad_parser.add_argument("--batch-size", type=int, default=8, help="CLIP scoring batch size")

    # Subcommand: run (auto-detects method from config)
    run_parser = subparsers.add_parser("run", help="Run search optimizer based on config search_method")
    run_parser.add_argument("config", help="Path to search config JSON")
    run_parser.add_argument("--output", help="Override output directory")
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate search without GPU compute")
    run_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP scoring model")
    run_parser.add_argument("--device", default="cuda", help="CLIP scoring device")
    run_parser.add_argument("--batch-size", type=int, default=8, help="CLIP scoring batch size")

    args = parser.parse_args(argv)

    try:
        config_file = args.config
        config = load_search_config(config_file)
        method = config.get("search_method") if args.command == "run" else args.command

        root = project_root()

        if method == "random":
            result = run_random_search(
                config=config,
                project_root=root,
                output_dir=args.output,
                dry_run=args.dry_run,
                clip_model_id=args.clip_model,
                score_device=args.device,
                batch_size=args.batch_size,
            )
            print(
                json.dumps(
                    {
                        "search_method": "random",
                        "status": result["status"],
                        "evaluations": result["evaluated_count"],
                        "nondominated_points": result["nondominated_count"],
                        "output_dir": args.output or config.get("output_root"),
                    },
                    indent=2,
                )
            )
            return 0

        if method == "ecad":
            result = run_ecad_search(
                config=config,
                project_root=root,
                output_dir=args.output,
                dry_run=args.dry_run,
                clip_model_id=args.clip_model,
                score_device=args.device,
                batch_size=args.batch_size,
            )
            print(
                json.dumps(
                    {
                        "search_method": "ecad",
                        "status": result["status"],
                        "generations": result["total_generations"],
                        "evaluations": result["evaluated_count"],
                        "nondominated_points": result["nondominated_count"],
                        "output_dir": args.output or config.get("output_root"),
                    },
                    indent=2,
                )
            )
            return 0

        raise ValueError(f"unknown search method: {method}")

    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

