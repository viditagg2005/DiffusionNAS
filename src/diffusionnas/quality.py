from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class ImageRecord:
    prompt_index: int
    prompt: str
    seed: int
    image_path: Path


def image_records(run_dir: str | Path) -> list[ImageRecord]:
    run_path = Path(run_dir)
    config = _read_json(run_path / "config.json")
    records: list[ImageRecord] = []
    missing: list[Path] = []
    for prompt_index, prompt in enumerate(config["prompts"]):
        for seed in config["seeds"]:
            image_path = run_path / "images" / f"p{prompt_index:04d}_s{seed}.png"
            if image_path.exists():
                records.append(ImageRecord(prompt_index, prompt, seed, image_path))
            else:
                missing.append(image_path)
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise ValueError(f"run is missing {len(missing)} expected image(s): {preview}")
    if not records:
        raise ValueError(f"run has no scoreable images: {run_path}")
    return records


def summarize_clip_scores(scores: list[float], model_id: str) -> dict[str, Any]:
    if not scores:
        raise ValueError("cannot summarize an empty CLIP score list")
    ordered = sorted(scores)
    p10_index = max(0, int(0.1 * (len(ordered) - 1)))
    return {
        "definition": "100 * max(cosine_similarity(CLIP_image, CLIP_text), 0)",
        "model_id": model_id,
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "p10": ordered[p10_index],
        "population_std": statistics.pstdev(scores),
    }


class ClipScorer:
    """Load a frozen CLIP model once and score any number of benchmark runs."""

    def __init__(self, model_id: str = DEFAULT_CLIP_MODEL, device: str = "cuda", batch_size: int = 8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).eval().to(device)

    def score_run(self, run_dir: str | Path) -> dict[str, Any]:
        from PIL import Image

        run_path = Path(run_dir).resolve()
        records = image_records(run_path)
        rows: list[dict[str, Any]] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            images = [Image.open(record.image_path).convert("RGB") for record in batch]
            inputs = self.processor(
                text=[record.prompt for record in batch],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.inference_mode():
                image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                text_features = self.model.get_text_features(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores = (100.0 * (image_features * text_features).sum(dim=-1).clamp(min=0)).cpu().tolist()
            for record, score in zip(batch, scores):
                rows.append(
                    {
                        "schema_version": 1,
                        "prompt_index": record.prompt_index,
                        "seed": record.seed,
                        "image": str(record.image_path.relative_to(run_path)),
                        "clip_score": float(score),
                    }
                )
            for image in images:
                image.close()

        measurement_path = run_path / "quality_measurements.jsonl"
        measurement_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        clip_summary = summarize_clip_scores([row["clip_score"] for row in rows], self.model_id)
        summary = _read_json(run_path / "summary.json")
        summary.setdefault("quality_metrics", {})["clip_score"] = clip_summary
        _write_json(run_path / "summary.json", summary)

        manifest = _read_json(run_path / "manifest.json")
        manifest["quality_scoring"] = {
            "status": "succeeded",
            "model_id": self.model_id,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(run_path / "manifest.json", manifest)
        return clip_summary


def discover_successful_runs(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    discovered: list[Path] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest.get("status") == "succeeded":
            discovered.append(manifest_path.parent)
    return discovered


def score_runs(
    run_dirs: Iterable[str | Path],
    model_id: str = DEFAULT_CLIP_MODEL,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, dict[str, Any]]:
    paths = [Path(path) for path in run_dirs]
    if not paths:
        raise ValueError("no successful runs were selected for scoring")
    scorer = ClipScorer(model_id=model_id, device=device, batch_size=batch_size)
    return {str(path.resolve()): scorer.score_run(path) for path in paths}

