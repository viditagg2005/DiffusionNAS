# Benchmark Plan Review — 2026-09-14

## Outcome

Keep Stable Diffusion 1.5 as the single matched model. The A100-side workload is feasible under the stated budget; target-edge feasibility must be measured later and an OOM is an infeasible policy, not a reason to silently change the model.

## Corrections made

1. The maintained Diffusers model ID is `stable-diffusion-v1-5/stable-diffusion-v1-5`; the older RunwayML ID in upstream examples is deprecated.
2. AutoDiffusion's repository includes a Stable Diffusion experiment, but it uses SD-v1.4 and searches timestep sequences. Its architecture-search implementation is not an SD1.5 implementation. Our SD1.5 result must therefore be labeled **adapted**, initially covering timestep search/application only.
3. DeepCache directly supports SD1.5 through its plug-and-play `DeepCacheSDHelper`; this is the closer paper-implementation reproduction.
4. Quality metrics must be computed outside the timed inference region. Otherwise CLIP/FID/KID overhead contaminates deployment latency.
5. FID at 1,000 samples is retained only as within-protocol descriptive evidence. KID is the safer primary finite-sample distribution metric.

## Implementation decisions

- Upstream repositories are pinned as Git submodules.
- One JSON configuration schema locks model, resolution, prompts, seeds, scheduler, precision, and method-specific controls.
- Every run stores implementation fidelity, upstream commits, package versions, raw stdout/stderr, per-sample measurements, and summary metrics.
- GPU tests and model downloads are separate from CPU-only configuration/dry-run tests.

## Current validation boundary

CPU-only unit and dry-run tests validate configuration and artifact logging. A real inference smoke test is pending a clean CUDA environment, DeepCache installation, and SD1.5 weight download. The current global Python environment is unsuitable because NumPy 2.2.6 is paired with a PyTorch build compiled against NumPy 1.x.

