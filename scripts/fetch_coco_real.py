#!/usr/bin/env python3
"""Download real COCO val2014 images for KID/FID "vs real" (run on a login node with internet).

The set is the 128 validation-split images plus a seeded random sample of other
val2014 images, 5,000 in total by default: KID/FID against only 128 images would
rest on an unstable 2048-dimensional covariance. Resumable; images are resized to
512x512 (center crop) to match generated images. Output: data/coco_real/ (git-ignored).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_URL = ("https://raw.githubusercontent.com/tylin/coco-caption/"
                   "3a9afb2682141a03e1cdc02b0df6770d2c884f6f/annotations/captions_val2014.json")
ANNOTATIONS_SHA256 = "ee4c7a657443714cc448e12e6193c4d4616de8ad2382a07b8f702ee5e3c46d2b"
IMAGE_URL = "http://images.cocodataset.org/val2014/{}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "coco_real")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    raw = fetch(ANNOTATIONS_URL)
    if hashlib.sha256(raw).hexdigest() != ANNOTATIONS_SHA256:
        raise SystemExit("unexpected COCO annotation hash")
    files = {image["id"]: image["file_name"] for image in json.loads(raw)["images"]}
    validation = [p["image_id"] for p in json.loads((ROOT / "data/prompts/validation.json").read_text())["prompts"]]
    others = sorted(set(files) - set(validation))
    chosen = validation + random.Random(args.seed).sample(others, args.count - len(validation))

    def download(image_id: int) -> str:
        target = args.output / f"{image_id:012d}.png"
        if target.exists():
            return "cached"
        image = Image.open(io.BytesIO(fetch(IMAGE_URL.format(files[image_id])))).convert("RGB")
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        image.crop((left, top, left + side, top + side)).resize((512, 512), Image.BICUBIC).save(target)
        return "downloaded"

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(download, chosen))
    (args.output / "manifest.json").write_text(json.dumps(
        {"count": len(chosen), "seed": args.seed, "validation_images": len(validation), "image_ids": chosen}, indent=1))
    print(f"{results.count('downloaded')} downloaded, {results.count('cached')} cached -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
