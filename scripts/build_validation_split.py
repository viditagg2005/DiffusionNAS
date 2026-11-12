#!/usr/bin/env python3
"""Build a fresh COCO validation split, image- and caption-disjoint from existing splits.

The original hold-out results have been inspected, so finalists from the v2
searches are tested on this untouched split. Same source, strata and quotas as
the hold-out; a different selection seed.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random

from build_coco_prompt_splits import LICENSE_URL, SOURCE_SHA256, SOURCE_URL, STRATA, stratum

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SEED = 20261010
QUOTAS = {"simple": 42, "relational": 43, "detailed": 43}
EXCLUDE = ("calibration.json", "holdout.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "prompts" / "validation.json")
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--quotas", default="42,43,43", help="simple,relational,detailed counts")
    parser.add_argument("--exclude", default=",".join(EXCLUDE), help="prompt files whose images/captions are excluded")
    args = parser.parse_args()
    quotas = dict(zip(STRATA, (int(x) for x in args.quotas.split(","))))
    exclude = [name for name in args.exclude.split(",") if name]

    raw = args.annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        parser.error("unexpected COCO annotation SHA-256")
    used_images: set[int] = set()
    used_captions: set[str] = set()
    for name in exclude:
        for item in json.loads((ROOT / "data" / "prompts" / name).read_text(encoding="utf-8"))["prompts"]:
            used_images.add(item["image_id"])
            used_captions.add(item["prompt"].casefold())

    pools: dict[str, dict[int, dict]] = {name: {} for name in STRATA}
    for annotation in sorted(json.loads(raw)["annotations"], key=lambda item: item["id"]):
        caption = " ".join(annotation["caption"].split())
        label = stratum(caption)
        image_id = annotation["image_id"]
        if label and image_id not in used_images and image_id not in pools[label]:
            pools[label][image_id] = {
                "prompt": caption,
                "image_id": image_id,
                "annotation_id": annotation["id"],
                "stratum": label,
            }

    rng = random.Random(args.selection_seed)
    chosen: dict[str, list[dict]] = {}
    taken_images: set[int] = set()
    for label in STRATA:
        candidates = sorted(pools[label].values(), key=lambda item: item["image_id"])
        rng.shuffle(candidates)
        chosen[label] = []
        for item in candidates:
            if len(chosen[label]) == quotas[label]:
                break
            if item["image_id"] in taken_images or item["prompt"].casefold() in used_captions:
                continue
            chosen[label].append(item)
            taken_images.add(item["image_id"])
            used_captions.add(item["prompt"].casefold())
    prompts = [item for row in zip_longest(*(chosen[label] for label in STRATA)) for item in row if item]
    value = {
        "split": args.split_name,
        "source": "MS-COCO 2014 validation captions",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "license": "CC BY 4.0 (COCO Consortium)",
        "license_url": LICENSE_URL,
        "selection_seed": args.selection_seed,
        "selection_rule": "Same strata as calibration/holdout; images and captions disjoint from both.",
        "excluded_splits": exclude,
        "stratum_counts": quotas,
        "prompts": prompts,
    }
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite changed prompt file: {args.output}")
    args.output.write_text(content, encoding="utf-8")
    print(f"{args.output}: {len(prompts)} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
