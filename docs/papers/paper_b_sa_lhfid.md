# Surrogate-Assisted LHFiD for Expensive Many-Objective Neural Architecture Search of Diffusion Models

*Draft for an evolutionary-computation venue (GECCO / IEEE CEC / EMO). Numbers come from committed scripts; [C2] marks results pending the real-evaluation confirmation.*

## Abstract (draft)

Compressing a pretrained text-to-image diffusion model for deployment is a many-objective problem: prompt alignment, fidelity to the original model, latency, compute, memory and weight bytes conflict, and each candidate configuration costs about a minute of GPU time to evaluate on 96 images. We formulate training-free compression of Stable Diffusion 1.5 as a six-objective problem over U-Net unit removal, weight precision, sampler, step count, guidance and feature-caching schedules, and study evolutionary search under the budgets this cost allows. With 150 real evaluations, no multi- or many-objective evolutionary algorithm beats random search, and quality objectives measured on small calibration sets are too noisy to rank good candidates (split-half reliability 0.50). On surrogate benchmarks built from 900 real evaluations, LHFiD outperforms NSGA-III and NSGA-II at realistic budgets, increasingly so with more objectives, but only beyond roughly 336–840 evaluations. We therefore propose surrogate-assisted LHFiD (SA-LHFiD), which spends the real budget on candidates proposed by LHFiD running on an online surrogate. In a pre-registered simulation calibrated to the real problem, SA-LHFiD improves hypervolume by 26–42% over random search at 150 evaluations and beats surrogate-assisted NSGA-II, RVEA and (under one of two truth models) MOEA/D; surrogate assistance with Pareto-dominance selection does not beat random search. [C2: real-evaluation confirmation with 6 seeds per arm.] The best searched configuration halves U-Net weight bytes, reduces peak VRAM by 0.83 GB and FLOPs by 47%, runs 1.58× faster, and is statistically non-inferior in prompt alignment on 500 held-out prompts.

## 1. Introduction

- Deployment compression of diffusion models: distillation-based compact models (SnapFusion, BK-SDM, LDMOES) need training; training-free acceleration (DeepCache, Flexiffusion, AutoDiffusion, step reduction, post-training quantization) tunes one knob at a time.
- Joint training-free configuration is a genuinely many-objective, expensive and noisy problem: six measured objectives, about 40 s of GPU time per candidate, CLIP-based quality noisy at practical calibration sizes.
- Contributions:
  1. Formulation of training-free SD1.5 compression as a six-objective expensive optimization problem with real-hardware objectives, and a released benchmark (companion paper).
  2. Evidence that at real budgets selection scheme does not matter (no MOEA beats random at 150 evaluations) and why: noisy quality objectives and an evaluation-count crossover around 336–840.
  3. SA-LHFiD: surrogate-assisted search with LHFiD's localized high-fidelity dominance as the inner selector; pre-registered simulation and real-evaluation confirmation [C2].
  4. A validated application outcome: a training-free configuration that halves U-Net bytes and is non-inferior on 500 held-out prompts.

## 2. Related work

- Many-objective EAs: NSGA-III (Deb and Jain 2014), MOEA/D (Zhang and Li 2007), RVEA (Cheng et al. 2016), LHFiD (Saxena, Mittal, Kapoor and Deb, COIN 2021020; stabilization-timed nadir estimation, Saxena and Kapoor 2019).
- Surrogate-assisted (many-objective) optimization: ParEGO, K-RVEA, MOEA/D-EGO, CSEA; NAS surrogates (NAS-Bench-301).
- Diffusion efficiency: DeepCache, Flexiffusion, AutoDiffusion, ECAD, LDMOES, SnapFusion, BK-SDM, Q-Diffusion/PTQ4DM.
- Positioning: first application of hf-dominance-based selection inside a surrogate-assisted loop for an expensive real-hardware many-objective problem; first many-objective formulation of training-free diffusion compression with measured cost objectives.

## 3. Problem

- Model: Stable Diffusion 1.5, 512×512, batch 1, frozen weights.
- Decision variables (45 genes): method (DDIM/DPM-Solver++ baseline, DeepCache, Flexiffusion segment schedules), steps 4–25, guidance 5–12, DeepCache interval/branch, Flexiffusion N-segment and per-segment branch/interval, 15 removal bits (units selected by a single-unit sensitivity probe), fp16/int8 weight storage.
- Objectives: CLIP prompt alignment (max), CLIP image fidelity to the 20-step reference (max), p50 latency (min), GFLOPs (min), peak VRAM (min), U-Net weight bytes (min). Search uses M = 4 (alignment, fidelity, latency, bytes); benchmark uses M = 4 and 6.
- Cost and noise: ~40 s per candidate on an RTX 6000 Ada; CLIP reliability 0.50 in the quality-floor region; latency drift 5% between jobs (normalized by per-job reference).

## 4. Method: SA-LHFiD

1. Evaluate 50 random novel configurations.
2. For 5 rounds: fit an ExtraTrees surrogate per objective on all real observations; run LHFiD (Das–Dennis p = 5, 56 vectors, 30 generations, identical variation operators) on the surrogate excluding evaluated configurations; take the predicted-nondominated members of LHFiD's final population, choose 20 by per-objective best then farthest-point spread; evaluate them for real.
3. Objectives scaled once by fixed bounds (LHFiD's translation-only pre-nadir phase and hf weighted sums are not scale-invariant).
4. Variant (ablation): noise-aware hf-dominance treats quality differences below the measured noise SD as ties.

## 5. Experiments

### 5.1 Real searches at 150 evaluations
- Random vs NSGA-II, 3 seeds: NSGA-II ahead in 2 of 3 seeds (latency-HV and drift-free GFLOPs-HV); no consistent advantage.
- CLIP split-half reliability in the quality-floor region 0.50.

### 5.2 Surrogate benchmark (pre-registered, `configs/nas/surrogate_benchmark.json`)
- 5 independently fitted surrogates × 7 seeds; Wilcoxon + Holm; claims only if consistent across surrogates.
- 3,360 evaluations, M = 6, archive HV: LHFiD 0.258 > RVEA 0.252 ≈ MOEA/D 0.250 > NSGA-III 0.232 > NSGA-II 0.213 ≈ random 0.215.
- Final-population HV, M = 6: RVEA 0.209 ≈ MOEA/D 0.200 ≈ LHFiD 0.199 ≫ NSGA-III 0.139 > NSGA-II 0.105. LHFiD's lead over NSGA-III grows from +0.022 (M = 4) to +0.064 (M = 6).
- At 150 evaluations random search is ahead of every MOEA; the crossover is 336–840 evaluations.
- Real measurement of 25 optimizer-chosen points: MAE CLIP 0.32 (bias +0.16, not over-optimistic), fidelity 1.5, latency ~3%, GFLOPs ~2%, VRAM 15 MB, bytes 7 MB; 24/25 truly inside the reference box.

### 5.3 SA-LHFiD, simulation (pre-registered, `configs/nas/sa_simulation.json`)
- Truth = HGB or RF fitted on the 900 real evaluations (a family different from the internal surrogate); noise calibrated to reliabilities 0.50 / 0.90; HV on noise-free truth; 31 seeds; Mann–Whitney + Holm.
- HGB / RF truth medians: SA-LHFiD 0.415 / 0.338; SA-MOEA/D 0.374 / 0.320; SA-RVEA 0.369 / 0.314; LHFiD 0.333 / 0.263; SA-NSGA-II 0.303 / 0.257; random 0.292 / 0.268; NSGA-II 0.260 / 0.208.
- Noise-aware ties: no benefit.

### 5.4 SA-LHFiD, real evaluations [C2]
- Arms: SA-LHFiD, SA-NSGA-II, random, NSGA-II; 6 seeds each; primary 4-D HV with per-job latency normalization; robustness: GFLOPs-HV and HV without CLIP; Mann–Whitney + Holm.

### 5.5 Application outcome (C1, pre-registered)
- 500 held-out prompts: DDIM-10, guidance 10.5, int8 is non-inferior in CLIP (margin 0.5; difference −0.000, Holm p = 0.0012), with half the U-Net bytes, −0.83 GB peak VRAM, −47% FLOPs, 1.58× speed; it beats distilled BK-SDM-small at the same sampler and precision (+1.31 CLIP).

## 6. Discussion and limitations
- Training-free unit removal is not worth it once int8 is available (twins ablation, C1).
- Single GPU model; CLIP/Inception metrics only; int8 latency from a dequantize-per-call implementation; simulation conclusions conditional on calibrated truths, hence C2.
