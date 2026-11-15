"""Noise-aware final selection (NFS) of a few deployable configurations from a search archive.

Objective vectors are minimization vectors ``[-clip, -fidelity, latency_ms, unet_mb]``. A *box* is the
reference point of the hypervolume the deployer cares about: CLIP >= reference - 1, fidelity >= 75,
latency < factor x reference latency (relative, so a slower node does not move configurations out),
U-Net < 1,700 MB.

NFS (C4, ``configs/nas/c4_protocol.json``):
  1. screen: greedy hypervolume subset selection (HSSP) of ``n_screen`` in-box points of the archive front;
  2. re-measure the screened configurations on fresh prompts, in one job with its own reference;
  3. select: greedy HSSP of ``k`` from the fresh measurements only.
The naive control is step 3 applied to the archive's own (selection-biased) measurements, i.e. the first
``k`` points of the greedy order in step 1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

LATENCY_FACTOR = 1.18  # 600 ms / 507 ms, the pre-registered C2 box relative to its reference
FIDELITY_FLOOR = 75.0
UNET_MB_LIMIT = 1700.0
CLIP_MARGIN = 1.0


def vector(objectives: dict) -> list[float]:
    return [-objectives["clip_score"], -objectives["clip_image_fidelity"], objectives["latency_p50_ms"],
            objectives["unet_weight_mb"]]


def box(reference_clip: float, reference_latency_ms: float) -> np.ndarray:
    return np.array([-(reference_clip - CLIP_MARGIN), -FIDELITY_FLOOR, LATENCY_FACTOR * reference_latency_ms,
                     UNET_MB_LIMIT])


def in_box(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, len(ref))
    return (points < ref).all(axis=1)


def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    import moocore

    points = np.asarray(points, dtype=float).reshape(-1, len(ref))
    inside = points[in_box(points, ref)]
    return float(moocore.hypervolume(inside, ref=ref)) if len(inside) else 0.0


def greedy_hssp(points: np.ndarray, ref: np.ndarray, k: int) -> list[int]:
    """Indices of up to ``k`` in-box points, added one at a time by largest hypervolume gain.

    Out-of-box points are never chosen; ties break toward the lower index (deterministic). Points that
    add nothing (dominated by the chosen set) are still taken while slots remain, in index order.
    """
    points = np.asarray(points, dtype=float).reshape(-1, len(ref))
    candidates = [i for i in np.flatnonzero(in_box(points, ref))]
    chosen: list[int] = []
    current = 0.0
    while candidates and len(chosen) < k:
        gains = [hypervolume(points[chosen + [i]], ref) - current for i in candidates]
        best = candidates[int(np.argmax(gains))]
        chosen.append(int(best))
        candidates.remove(best)
        current = hypervolume(points[chosen], ref)
    return chosen


def nondominated_indices(points: Sequence[Sequence[float]]) -> list[int]:
    points = np.asarray(points, dtype=float)
    keep = []
    for i, p in enumerate(points):
        dominated = ((points <= p).all(axis=1) & (points < p).any(axis=1)).any()
        if not dominated:
            keep.append(i)
    return keep
