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
| 0b | Control benchmark (native DPM++, 18 Flexiffusion schedules) | cluster, Oct 10 | ✅ see Results |
| 1 | Single-unit sensitivity probe (24 units, 6 groups, int8) | `scripts/run_probe.sh` (srun) | ✅ see Results; 15 units kept (`configs/nas/units.json`) |
| 2 | v3 search: random vs NSGA-II, 150 unique candidates each, seeds 1–3 | `nas_search.sbatch` × 3 (~3.7 h each) | ✅ see Results |
| 3 | Lock finalists (`scripts/lock_nas_finalists.py`, committed before searches ran) | CPU | ✅ 10 finalists (`configs/generated/validation_v3/locked_finalists.json`) |
| 3b | Many-objective optimizer benchmark on surrogates (LHFiD, NSGA-III, NSGA-II, random; protocol `configs/nas/surrogate_benchmark.json`) + real confirmation | CPU + srun | ✅ benchmark (report §5.4); ⏳ confirmation in `run_followup.sh` |
| 4 | Validation: 43 runs (10 finalists, 12 architecture-ablation twins, DDIM/DPM++ sweeps, DeepCache, int8, 2 Flexiffusion, BK-SDM ×3, reference ×3) in one allocation | `validation_v3.sbatch` (job 9827) | ✅ generation, CLIP, fidelity (report §5.3); ⏳ KID/FID (job's final scoring step failed) and post-hoc BK-SDM in `run_followup.sh` |
| 5 | Final report, Pareto/HV figures, docs, Notion | local | ⏳ `docs/final_report.md` drafted with all results except §5.5 |

## Cluster run log

| Date | Job | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-29/30 | calibration + hold-out benchmark | `benchmark_array.sbatch`, `holdout.sbatch` | ✅ `docs/calibration_results.md` |
| 2026-10-06/08 | v1 random + ECAD-inspired searches | `random_search.sbatch`, `ecad_search.sbatch` | ✅ no algorithm difference |
| 2026-10-10 | control benchmark | `controls_benchmark.sbatch` | results pending copy-back |
| — | v2 search (`search_v2.sbatch`) | — | **superseded by v3; do not run** |
| 2026-11-10 | sensitivity probe | `scripts/run_probe.sh` (srun) | ✅ RTX 6000 Ada |
| 2026-11-10/11 | v3 searches, seeds 1–3 | `nas_search.sbatch` | ✅ RTX 6000 Ada, 900 candidates |
| 2026-11-12 | validation | `validation_v3.sbatch` (job 9827) | FAILED in the final KID/FID step after 1h46m; 43 runs generated and CLIP/fidelity scored |
| 2026-11-13 | follow-up session | `scripts/run_followup.sh` (srun) | running |

## Local smoke evidence (RTX 3070 Ti laptop, 2 prompts, not a result)

20-step reference 34,672 GFLOPs/image, U-Net 859.5 M params / 1,639 MB. int8: U-Net 821 MB, peak 3,256 → 2,443 MB, +17% latency, fidelity 91.7. Removing `mid.attn0`: −34.7 M params, fidelity 79.2. Removing all 64-px attention: −22% GFLOPs, −24% latency, only −24 MB weights, fidelity 73.9. Compute-heavy and parameter-heavy units differ, so FLOPs/latency and weight bytes are genuinely separate objectives.

## Results so far (RTX 6000 Ada, calibration split, 48 prompts × 2 seeds)

**Controls (Oct 10).** Native 4-step DPM-Solver++ (140.5 ms, CLIP 25.280) is indistinguishable from the AutoDiffusion timestep adaptation (140.3 ms, 25.264): the adapted timetable adds nothing. No uniform Flexiffusion schedule dominates DeepCache i3 b0 (236 ms, 26.157); the fastest (N7 b1 i2) is 245 ms at CLIP 25.93. Long Flexiffusion schedules (N9–11, branch 6, interval 2: 399–478 ms, CLIP 26.35–26.51) sit slightly above the 20-step baseline's CLIP at lower latency, within calibration noise.

**Sensitivity probe (Nov 10).** Reference: 505 ms, 34.7 TFLOPs/image, U-Net 859.5 M params / 1,639 MB, peak 3,271 MB.
- int8 weights: U-Net 1,639 → 821 MB, peak 3,271 → 2,438 MB, CLIP +0.04 [−0.26, +0.32], fidelity 95.9, latency +24% (628 ms). Memory compression with no measurable alignment loss.
- Safe single removals (CLIP CI reaches 0, fidelity ≥ 80): 15 units — `down0.attn0/1`, `down1.attn1`, `down1.res1`, `down2.attn0/1`, `down3.res0/1`, `mid.res1`, `up1.attn2`, `up2.attn0/1/2`, `up3.attn0/1`. Deep units save 59–66 MB each; 64-px attentions save 4.5% FLOPs each but only 5 MB.
- Harmful: `down0.res0` (−11.5 CLIP, 16 black images), `up3.attn2` (−5.1, 15 black), `down0.res1` (−3.8), `mid.attn0` (−2.0), `down2.res1` (−1.8), `up1.attn0/1`, `mid.res0`, `down1.attn0`.
- Removals interact: removing 5 individually-harmless 32-px attentions together costs −3.1 CLIP; all 16-px attentions −7.7. Hence a search over combinations, not greedy stacking.
- Latency responds weakly to removing low-resolution units on this GPU (batch 1 is not compute-bound there): −22% FLOPs from all 64-px attentions gives −18% latency, while deep-unit removals change latency by ±3%.

**Decision D021 (search objectives).** Search on 4 objectives (CLIP, fidelity, latency, U-Net MB); measure and report GFLOPs and peak VRAM for every candidate. With 6 objectives nearly every candidate is nondominated and NSGA-II selection degenerates; GFLOPs tracks latency and peak VRAM tracks weight bytes plus a cache term. Pre-registered hypervolume reference: CLIP 25.2 (reference − 1.0), fidelity 75, 600 ms, 1,700 MB. NSGA-II starts from random novel genomes (no hand-picked seeds), so its hypervolume gap over random search measures the optimizer only.

**v3 searches (Nov 10–11).** 6 searches × 150 unique candidates, all on the RTX 6000 Ada; job references 485 / 491 / 511 ms (5% between-job drift, so latency is normalized by each job's reference). 233 of 900 candidates pass CLIP ≥ 25.2 and fidelity ≥ 75. Hypervolume (pre-registered 4-D reference, latency normalized), NSGA-II minus random per seed: −0.78, +0.71, +2.64 (×10⁶); drift-free variant with GFLOPs in place of latency gives the same 2-of-3 pattern, and the seed where NSGA-II ran first is not the outlier, so job order does not explain it. **No consistent optimizer advantage at 150 real evaluations** (`scripts/analyze_search.py`, `docs/figures/search_hypervolume.png`). Speed comes mostly from fewer steps (6-step DPM-Solver++ with no edits: 180 ms, CLIP 26.29); the architecture genes mainly buy bytes (U-Net down to 696 MB, −58%, with int8 + 7 removed units at CLIP 25.39).

**Finalist lock (Nov 12).** 839 unique policies, 186 eligible, 82 on the 4-objective front, 10 locked; validation design (`scripts/prepare_validation_v3.py`) committed before the lock.
