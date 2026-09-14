# DiffusionNAS

Training-free, standardized benchmarking of inference-time acceleration policies for Stable Diffusion 1.5.

## Upstream implementations

- `third_party/AutoDiffusion` pins the official AutoDiffusion repository.
- `third_party/DeepCache` pins the official DeepCache repository.

The wrappers do not modify or install into either upstream tree. AutoDiffusion is adapted to SD1.5 as explicit timestep-sequence search/application. This is marked `adapted`, because the upstream Stable Diffusion experiment uses SD-v1.4 and does not apply its architecture search to SD1.5. DeepCache is imported from its pinned submodule through the runner and is marked `paper_implementation`.

## Quick start

```bash
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r requirements-gpu.txt

diffusionnas preflight
diffusionnas validate configs/deepcache_sd15.json
diffusionnas run configs/deepcache_sd15.json --dry-run
diffusionnas run configs/deepcache_sd15.json
```

All runs create an immutable directory under `artifacts/runs/` containing resolved configuration, provenance, stdout/stderr, per-sample measurements, and a summary. Dry runs test configuration and logging without loading weights or requiring a GPU.

## Common protocol

- Model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Resolution: 512×512; batch size 1
- Identical prompts and seeds for all methods
- CUDA synchronization around end-to-end inference timing
- Peak CUDA allocated/reserved memory captured per sample
- One-time model load excluded; warm-ups recorded
- AutoDiffusion candidates use DPM-Solver++, the Diffusers 0.32 scheduler in this harness that accepts explicit timestep lists

The harness currently establishes execution and systems metrics. CLIP/KID/FID scoring should consume the saved images in a separate stage so metric computation is not included in inference latency.

## Commands

```bash
diffusionnas validate CONFIG.json
diffusionnas run CONFIG.json [--dry-run]
diffusionnas preflight
python3 -m unittest discover -s tests -v
```
