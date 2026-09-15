from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .provenance import collect
from .holdout import validate_holdout
from .pareto import construct_front
from .quality import DEFAULT_CLIP_MODEL, discover_successful_runs, score_runs
from .runner import execute
from .schema import ConfigError, load_config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diffusionnas")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("config")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument("--dry-run", action="store_true")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("run_dirs", nargs="*")
    score_parser.add_argument("--runs-root")
    score_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    score_parser.add_argument("--device", default="cuda")
    score_parser.add_argument("--batch-size", type=int, default=8)
    pareto_parser = subparsers.add_parser("pareto")
    pareto_parser.add_argument("--runs-root", required=True)
    pareto_parser.add_argument("--output", required=True)
    pareto_parser.add_argument("--include-energy", action="store_true")
    holdout_parser = subparsers.add_parser("holdout")
    holdout_parser.add_argument("--front", required=True)
    holdout_parser.add_argument("--prompts", required=True)
    holdout_parser.add_argument("--seeds", required=True, help="comma-separated integer seeds")
    holdout_parser.add_argument("--output", required=True)
    holdout_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    holdout_parser.add_argument("--device", default="cuda")
    holdout_parser.add_argument("--batch-size", type=int, default=8)
    holdout_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("preflight")
    args = parser.parse_args(argv)

    try:
        if args.command == "preflight":
            info = collect(project_root())
            print(json.dumps(info, indent=2, sort_keys=True))
            packages = info["packages"]
            ok = all(packages.get(name) for name in ("torch", "diffusers", "transformers"))
            numpy_version = packages.get("numpy") or ""
            if numpy_version.startswith("2."):
                print("ERROR: NumPy 2.x is incompatible with the installed PyTorch; create the pinned environment.", file=sys.stderr)
                ok = False
            deepcache_source = project_root() / "third_party" / "DeepCache" / "DeepCache" / "__init__.py"
            if not deepcache_source.exists():
                print("ERROR: initialize the DeepCache Git submodule", file=sys.stderr)
                ok = False
            return 0 if ok else 2
        if args.command == "score":
            run_dirs = [Path(path) for path in args.run_dirs]
            if args.runs_root:
                run_dirs.extend(discover_successful_runs(args.runs_root))
            results = score_runs(
                run_dirs,
                model_id=args.clip_model,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0
        if args.command == "pareto":
            runs = discover_successful_runs(args.runs_root)
            result = construct_front(runs, args.output, include_energy=args.include_energy)
            print(json.dumps({"output": str(Path(args.output).resolve()), "points": result["nondominated_count"]}, indent=2))
            return 0
        if args.command == "holdout":
            try:
                seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            except ValueError as exc:
                raise ValueError("--seeds must be comma-separated integers") from exc
            result = validate_holdout(
                front_path=args.front,
                prompts_path=args.prompts,
                seeds=seeds,
                project_root=project_root(),
                output_dir=args.output,
                dry_run=args.dry_run,
                clip_model_id=args.clip_model,
                score_device=args.device,
                batch_size=args.batch_size,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        config = load_config(args.config)
        if args.command == "validate":
            print(f"valid: {args.config}")
            return 0
        run_dir = execute(config, project_root(), dry_run=args.dry_run)
        print(run_dir)
        return 0
    except (ConfigError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
