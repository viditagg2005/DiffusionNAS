#!/usr/bin/env python3
"""Build fixed, disjoint COCO caption splits for calibration and hold-out."""

from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "ee4c7a657443714cc448e12e6193c4d4616de8ad2382a07b8f702ee5e3c46d2b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/tylin/coco-caption/"
    "3a9afb2682141a03e1cdc02b0df6770d2c884f6f/annotations/captions_val2014.json"
)
LICENSE_URL = "https://cocodataset.org/#termsofuse"
SELECTION_SEED = 20260930
STRATA = ("simple", "relational", "detailed")
QUOTAS = {
    "calibration": {"simple": 16, "relational": 16, "detailed": 16},
    "holdout": {"simple": 42, "relational": 43, "detailed": 43},
}
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RELATION = re.compile(
    r"\b(?:and|with|beside|near|between|behind|under|next to|"
    r"in front of|while|surrounded by|on top of|alongside)\b",
    re.IGNORECASE,
)


def stratum(caption: str) -> str | None:
    count = len(WORDS.findall(caption))
    if 18 <= count <= 30:
        return "detailed"
    if 9 <= count <= 17 and RELATION.search(caption):
        return "relational"
    if 6 <= count <= 10 and not RELATION.search(caption):
        return "simple"
    return None


def build_splits(annotations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pools: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in STRATA}
    for annotation in sorted(annotations, key=lambda item: item["id"]):
        caption = " ".join(annotation["caption"].split())
        label = stratum(caption)
        image_id = annotation["image_id"]
        if label and image_id not in pools[label]:
            pools[label][image_id] = {
                "prompt": caption,
                "image_id": image_id,
                "annotation_id": annotation["id"],
                "stratum": label,
            }

    rng = random.Random(SELECTION_SEED)
    candidates = {}
    for label in STRATA:
        candidates[label] = sorted(pools[label].values(), key=lambda item: item["image_id"])
        rng.shuffle(candidates[label])

    used_images: set[int] = set()
    used_captions: set[str] = set()
    cursors = {label: 0 for label in STRATA}
    result = {}
    for split, quotas in QUOTAS.items():
        chosen: dict[str, list[dict[str, Any]]] = {label: [] for label in STRATA}
        for label in STRATA:
            while len(chosen[label]) < quotas[label]:
                if cursors[label] >= len(candidates[label]):
                    raise ValueError(f"insufficient unique captions in {label} stratum")
                item = candidates[label][cursors[label]]
                cursors[label] += 1
                key = item["prompt"].casefold()
                if item["image_id"] in used_images or key in used_captions:
                    continue
                chosen[label].append(item)
                used_images.add(item["image_id"])
                used_captions.add(key)
        prompts = [
            item
            for row in zip_longest(*(chosen[label] for label in STRATA))
            for item in row
            if item is not None
        ]
        result[split] = {
            "split": split,
            "source": "MS-COCO 2014 validation captions",
            "source_url": SOURCE_URL,
            "source_sha256": SOURCE_SHA256,
            "license": "CC BY 4.0 (COCO Consortium)",
            "license_url": LICENSE_URL,
            "selection_seed": SELECTION_SEED,
            "selection_rule": (
                "Simple: 6-10 words without relation marker; relational: 9-17 words "
                "with relation marker; detailed: 18-30 words. One caption per image; "
                "whitespace normalized."
            ),
            "stratum_counts": quotas,
            "prompts": prompts,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "prompts")
    args = parser.parse_args()

    raw = args.annotations.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != SOURCE_SHA256:
        parser.error(f"unexpected COCO annotation SHA-256: {actual_sha256}")
    source = json.loads(raw)
    splits = build_splits(source["annotations"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, value in splits.items():
        path = args.output_dir / f"{split}.json"
        content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite changed prompt file: {path}")
        path.write_text(content, encoding="utf-8")
        print(f"{path}: {len(value['prompts'])} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
