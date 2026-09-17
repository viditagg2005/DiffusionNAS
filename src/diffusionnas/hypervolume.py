"""Hypervolume indicator computation for multi-objective optimisation.

Provides exact 2-D and 3-D hypervolume calculation via sweep-line
algorithms.  The helpers normalise mixed-direction objectives (min/max)
to all-minimisation form before computing the dominated volume.
"""
from __future__ import annotations

from typing import Any, Iterable


# ---- internal helpers --------------------------------------------------

def _hypervolume_2d(points: list[list[float]], ref: list[float]) -> float:
    """Exact 2-D dominated hypervolume via sweep-line.

    *points* is a list of ``[x, y]`` vectors in **minimisation** form.
    *ref* is the ``[x, y]`` reference point (must dominate no point).
    """
    filtered = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
    if not filtered:
        return 0.0
    filtered.sort(key=lambda p: p[0])
    # Extract non-dominated staircase (y must be strictly improving)
    staircase: list[list[float]] = []
    best_y = ref[1]
    for p in filtered:
        if p[1] < best_y:
            staircase.append(p)
            best_y = p[1]
    volume = 0.0
    for i, p in enumerate(staircase):
        x_next = staircase[i + 1][0] if i + 1 < len(staircase) else ref[0]
        volume += (x_next - p[0]) * (ref[1] - p[1])
    return volume


def _hypervolume_3d(points: list[list[float]], ref: list[float]) -> float:
    """Exact 3-D dominated hypervolume via incremental slicing.

    Sorts points along the third coordinate and computes the 2-D
    hypervolume of the accumulated projection for each slice.
    """
    filtered = [p for p in points if all(p[i] < ref[i] for i in range(3))]
    if not filtered:
        return 0.0
    filtered.sort(key=lambda p: p[2])
    volume = 0.0
    active_2d: list[list[float]] = []
    for i, p in enumerate(filtered):
        active_2d.append([p[0], p[1]])
        z_next = filtered[i + 1][2] if i + 1 < len(filtered) else ref[2]
        hv_2d = _hypervolume_2d(active_2d, [ref[0], ref[1]])
        volume += hv_2d * (z_next - p[2])
    return volume


# ---- public API --------------------------------------------------------

def compute_hypervolume(
    points: list[dict[str, float]],
    reference_point: dict[str, float],
    objectives: Iterable[tuple[str, str]],
) -> float:
    """Compute the dominated hypervolume indicator.

    *points*
        List of dicts mapping objective names to values.
    *reference_point*
        Dict mapping objective names to worst-acceptable values.
    *objectives*
        Iterable of ``(name, direction)`` pairs where *direction* is
        ``"min"`` or ``"max"``.

    Returns the hypervolume dominated by *points* with respect to the
    reference point.  Handles mixed objective directions by converting
    maximisation objectives to minimisation (negation).
    """
    obj_list = list(objectives)
    ndim = len(obj_list)
    if ndim < 2 or ndim > 3:
        raise ValueError(f"hypervolume requires 2 or 3 objectives, got {ndim}")
    if not points:
        return 0.0

    def _convert(d: dict[str, float]) -> list[float]:
        row: list[float] = []
        for name, direction in obj_list:
            value = d[name]
            row.append(-value if direction == "max" else value)
        return row

    converted = [_convert(p) for p in points]
    ref = _convert(reference_point)

    if ndim == 2:
        return _hypervolume_2d(converted, ref)
    return _hypervolume_3d(converted, ref)


def default_reference_point(
    points: list[dict[str, float]],
    objectives: Iterable[tuple[str, str]],
    margin: float = 0.1,
) -> dict[str, float]:
    """Derive a reference point that is 1+*margin* × worse than the
    worst observed value for each objective.

    For minimisation objectives the reference is *above* the maximum
    observed value; for maximisation it is *below* the minimum.
    """
    obj_list = list(objectives)
    if not points:
        raise ValueError("need at least one point to derive a reference")
    ref: dict[str, float] = {}
    for name, direction in obj_list:
        values = [p[name] for p in points]
        if direction == "min":
            worst = max(values)
            ref[name] = worst * (1.0 + margin) if worst > 0 else worst - abs(worst) * margin - 1e-6
        else:
            worst = min(values)
            ref[name] = worst * (1.0 - margin) if worst > 0 else worst + abs(worst) * margin + 1e-6
        # Ensure ref is strictly worse than all points
        if direction == "min":
            ref[name] = max(ref[name], max(values) + 1e-9)
        else:
            ref[name] = min(ref[name], min(values) - 1e-9)
    return ref

