from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .provenance import collect
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
        config = load_config(args.config)
        if args.command == "validate":
            print(f"valid: {args.config}")
            return 0
        run_dir = execute(config, project_root(), dry_run=args.dry_run)
        print(run_dir)
        return 0
    except (ConfigError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
