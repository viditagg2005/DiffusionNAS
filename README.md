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

## Calibration, Pareto front, and hold-out validation

Score every successful calibration run with one frozen CLIP model loaded once:

```bash
diffusionnas score --runs-root artifacts/calibration/runs \
  --clip-model openai/clip-vit-large-patch14 --device cuda --batch-size 8
```

Construct the calibration front. The default objectives are mean CLIP score (maximize), median end-to-end latency (minimize), and peak allocated GPU memory (minimize):

```bash
diffusionnas pareto --runs-root artifacts/calibration/runs \
  --output artifacts/calibration/pareto_front.json
```

Use `--include-energy` only when every candidate has a physical energy measurement. Missing quality/system metrics make a run ineligible and are reported in the front's `skipped` list.

Materialize and inspect a hold-out run plan without GPU work:

```bash
diffusionnas holdout \
  --front artifacts/calibration/pareto_front.json \
  --prompts data/prompts/holdout.json --seeds 101,202 \
  --output artifacts/holdout --dry-run
```

Remove `--dry-run` to rerun every calibration-front policy on untouched prompts, score the images, reconstruct the hold-out front, and report the retained policies plus Spearman rank correlation between calibration and hold-out CLIP scores.

## Commands

```bash
diffusionnas validate CONFIG.json
diffusionnas run CONFIG.json [--dry-run]
diffusionnas score RUN_DIR [RUN_DIR ...]
diffusionnas score --runs-root RUNS_ROOT
diffusionnas pareto --runs-root RUNS_ROOT --output FRONT.json
diffusionnas holdout --front FRONT.json --prompts PROMPTS.json --seeds 101,202 --output OUTPUT_DIR
diffusionnas preflight
python3 -m unittest discover -s tests -v
```
