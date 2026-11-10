# DiffusionNAS research plan and phase tracker

Single source of truth for what the study claims, how it is measured, and what is left. Older documents (`docs/calibration_results.md`, `experiment_results_summary.md`, `docs/benchmarking_plan.md`) are records of exploratory phases.

## Goal

A **training-free, multi-objective neural architecture search** for Stable Diffusion 1.5 (frozen `stable-diffusion-v1-5/stable-diffusion-v1-5`, 512×512, batch 1, fp16) that finds **compressed U-Net subnetworks plus inference policies**, and validates them against matched baselines on an untouched split.

**Done when** the validation report (`docs/final_report.md` and its published page) exists with every metric and figure below, and every claim is backed by a committed script and a result file.

## Search space (v3)

| Gene group | Choices | Effect |
| --- | --- | --- |
| Static unit removal | Subset of the 24 removable U-Net units (16 attention blocks, 8 channel-preserving ResNets), restricted to those the probe shows to be tolerable | Deletes weights: fewer parameters, less memory, fewer FLOPs |
| Weight precision | fp16, int8 weight-only (per-channel) | U-Net weight bytes −50%, peak VRAM −0.8 GB, ~+17–21% latency (measured locally) |
| Sampler and steps | DDIM / DPM-Solver++, 4–25 steps; guidance 5–12 | Effective compute |
| Execution reuse | DeepCache interval/branch, Flexiffusion per-segment full/partial/null steps | Effective compute (no weight change, cache adds memory) |

Up-block ResNets take concatenated skip connections and cannot be removed without retraining.

## Objectives and metrics

Search (calibration split, 48 COCO prompts × seeds 42, 123):
- maximize CLIP prompt alignment (CLIP-L/14, `100·cos`),
- maximize CLIP image fidelity to the 20-step DDIM reference image for the same prompt/seed,
- minimize p50 end-to-end latency, GFLOPs per image (FlopCounterMode: text encoder + U-Net steps + VAE), peak allocated VRAM, U-Net weight bytes.

Validation (128 untouched COCO prompts × seeds 303, 404): the above, plus KID/FID vs real COCO images and vs the reference model's images, black (safety-filtered) image count, U-Net parameters, and paired prompt-level bootstrap 95% CIs against the 20-step reference.

Interpretation rules: fidelity measures drift from the original model, not absolute quality (even int8 weights drift to ~92). FID on a few hundred images is biased; KID is primary. Latency is compared only within one GPU model, and across jobs only as speedup over a reference re-measured in the same job.

## Phases

| Phase | What | Where | Status |
| --- | --- | --- | --- |
| 0 | Harness, baselines (DDIM, DeepCache, AutoDiffusion timestep adaptation), calibration + hold-out benchmark, v1 searches, Flexiffusion reimplementation, v2 policy search, validation split, finalist rule | — | ✅ Done (exploratory; see CLAUDE.md) |
| 0b | Control benchmark (native DPM++, 18 Flexiffusion schedules) | cluster, submitted Oct 10 | ⏳ results not yet copied back (`artifacts/controls`, json/jsonl only) |
| 1 | Single-unit sensitivity probe (24 units, 6 groups, int8) | `scripts/run_probe.sh` (srun, ~45–60 min) | ⏳ ready to run |
| 2 | v3 search space from probe; random vs NSGA-II, 3 seeds, equal unique budgets | sbatch jobs < 24 h, one GPU each | ☐ |
| 3 | Lock finalists (committed rule) | CPU | ☐ |
| 4 | Validation: finalists + baselines (DDIM step sweep, native DPM++, DeepCache, Flexiffusion, BK-SDM external) in one allocation | sbatch | ☐ |
| 5 | Final report, Pareto/HV figures, docs, Notion | local | ☐ |

## Cluster run log

| Date | Job | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-29/30 | calibration + hold-out benchmark | `benchmark_array.sbatch`, `holdout.sbatch` | ✅ `docs/calibration_results.md` |
| 2026-10-06/08 | v1 random + ECAD-inspired searches | `random_search.sbatch`, `ecad_search.sbatch` | ✅ no algorithm difference |
| 2026-10-10 | control benchmark | `controls_benchmark.sbatch` | results pending copy-back |
| — | v2 search (`search_v2.sbatch`) | — | **superseded by v3; do not run** |

## Local smoke evidence (RTX 3070 Ti laptop, 2 prompts, not a result)

20-step reference 34,672 GFLOPs/image, U-Net 859.5 M params / 1,639 MB. int8: U-Net 821 MB, peak 3,256 → 2,443 MB, +17% latency, fidelity 91.7. Removing `mid.attn0`: −34.7 M params, fidelity 79.2. Removing all 64-px attention: −22% GFLOPs, −24% latency, only −24 MB weights, fidelity 73.9. Compute-heavy and parameter-heavy units differ, so FLOPs/latency and weight bytes are genuinely separate objectives.
