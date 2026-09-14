from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any


def _git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(project_root: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "project_dirty": bool(_git_output(project_root, "status", "--porcelain")),
        "upstream": {
            "autodiffusion": _git_output(project_root / "third_party" / "AutoDiffusion", "rev-parse", "HEAD"),
            "deepcache": _git_output(project_root / "third_party" / "DeepCache", "rev-parse", "HEAD"),
        },
        "packages": {
            name: package_version(name)
            for name in ("torch", "numpy", "diffusers", "transformers", "accelerate", "DeepCache")
        },
    }

