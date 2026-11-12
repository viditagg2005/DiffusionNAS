"""Distribution-level image quality: FID and KID on Inception-v3 pool features.

Uses the standard FID Inception weights from ``pytorch-fid``. With the small
sets used here (hundreds of images), KID is the primary statistic because it is
unbiased in sample size; FID is reported for familiarity but is biased upward
for small sets, so compare FID only between policies scored on identical set
sizes.

Two references are supported:
- real COCO images for the evaluated captions ("vs real"), and
- the uncompressed 20-step reference model's images for the same prompts and
  seeds ("relative", the role rFID plays in Flexiffusion): how far a
  compressed policy's output distribution drifts from the original model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def inception_features(paths: list[Path], device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    import torch
    from PIL import Image
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(device)
    features = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start : start + batch_size]:
            try:
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            except Exception as exc:  # report the exact file instead of a bare decoder error
                raise RuntimeError(f"cannot read image {path}: {exc}") from exc
            batch.append(torch.from_numpy(array).permute(2, 0, 1))
        # pytorch-fid resizes to 299x299 internally and expects inputs in [0, 1].
        pixels = torch.stack(batch).to(device)
        with torch.inference_mode():
            pooled = model(pixels)[0].squeeze(-1).squeeze(-1)
        features.append(pooled.cpu().numpy().astype(np.float64))
    return np.concatenate(features)


def fid(left: np.ndarray, right: np.ndarray) -> float:
    from scipy import linalg

    mu1, mu2 = left.mean(axis=0), right.mean(axis=0)
    sigma1, sigma2 = np.cov(left, rowvar=False), np.cov(right, rowvar=False)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def kid(left: np.ndarray, right: np.ndarray, subsets: int = 100, subset_size: int | None = None, seed: int = 0) -> dict[str, float]:
    """Unbiased MMD^2 with the cubic polynomial kernel, averaged over random subsets (x1000)."""
    rng = np.random.default_rng(seed)
    size = min(subset_size or 1000, len(left), len(right))
    dim = left.shape[1]
    values = []
    for _ in range(subsets):
        x = left[rng.choice(len(left), size, replace=False)]
        y = right[rng.choice(len(right), size, replace=False)]
        kxx = (x @ x.T / dim + 1) ** 3
        kyy = (y @ y.T / dim + 1) ** 3
        kxy = (x @ y.T / dim + 1) ** 3
        mmd = (
            (kxx.sum() - np.trace(kxx)) / (size * (size - 1))
            + (kyy.sum() - np.trace(kyy)) / (size * (size - 1))
            - 2 * kxy.mean()
        )
        values.append(mmd)
    return {"mean_x1000": float(np.mean(values) * 1000), "std_x1000": float(np.std(values) * 1000), "subset_size": size}


def distribution_metrics(generated: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    return {"kid": kid(generated, reference), "fid": fid(generated, reference),
            "generated_count": len(generated), "reference_count": len(reference)}
