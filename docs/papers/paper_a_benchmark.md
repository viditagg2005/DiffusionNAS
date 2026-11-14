# DiffNAS-Bench: A Real-Hardware Many-Objective Benchmark for Training-Free Diffusion Model Compression

*Draft for an AI venue (NeurIPS Datasets & Benchmarks / AutoML-Conf / ICLR workshop). Numbers from committed scripts; [C2] marks data still being added.*

## Abstract (draft)

Deploying text-to-image diffusion models under latency and memory limits requires choosing among many training-free knobs — sampler and step count, guidance, feature caching, segment-wise step skipping, weight precision and removal of U-Net blocks — whose effects interact and must be measured on real hardware. We release DiffNAS-Bench, a benchmark of [N ≈ 2,700 after C2; 839 now] real evaluated Stable Diffusion 1.5 configurations with six measured objectives (prompt alignment, fidelity to the original model, latency, GFLOPs, peak VRAM, weight bytes), a 45-gene search space, a fixed evaluation protocol, surrogate models with reported accuracy (leave-one-run-out Spearman 0.94–0.996), and baselines for seven optimizers. Using the benchmark we find that (i) quality signals from small calibration sets cannot rank good configurations (CLIP split-half reliability 0.50, while fidelity to the original model ranks configurations consistently, 0.96 rank agreement across splits); (ii) evolutionary multi-objective optimizers need roughly 336–840 evaluations before beating random search; (iii) training-free removal of U-Net blocks adds nothing once weights are stored in int8; and (iv) apparent gains of prompt-adaptive routing vanish against a mixture-of-fixed-policies baseline. A configuration found and validated with the benchmark halves U-Net bytes, cuts peak VRAM by 0.83 GB and runs 1.58× faster while being non-inferior in prompt alignment on 500 held-out prompts.

## 1. Introduction
- Gap: NAS benchmarks (NAS-Bench-101/201/301, EvoXBench, CitySeg/MOP) cover classifiers and segmenters; DiffBench covers acceleration code generation. No benchmark exists for many-objective configuration of diffusion inference on real hardware.
- Why it is hard to do ad hoc: each evaluation is ~40 s of GPU time; objectives are heterogeneous (noisy quality, deterministic cost); method papers report one knob at a time with incomparable baselines (e.g. Flexiffusion's 5.1× is against 50-step PLMS).

## 2. Benchmark
- Model and protocol: SD1.5, 512², batch 1, RTX 6000 Ada, 48 COCO calibration prompts × 2 seeds, one reference run per job for latency normalization and fidelity anchoring.
- Search space (45 genes): method (DDIM/DPM-Solver++, DeepCache, Flexiffusion segments), steps, guidance, caching parameters, 15 removable U-Net units chosen by a sensitivity probe, fp16/int8.
- Objectives and how each is measured; reference point; scaling.
- Data: [839 → ~2,700] configurations from random search, NSGA-II, and surrogate-assisted LHFiD/NSGA-II runs [C2]; per-image CLIP, fidelity, black-image flags; provenance (GPU, commits).
- Surrogate mode: ExtraTrees per objective; accuracy overall and inside the reference box (CLIP 0.61–0.69 in-box, near the 0.71 reliability ceiling).
- Splits: calibration, hold-out, validation (128), test (500), all image- and caption-disjoint.

## 3. Findings
1. **Quality objectives are the bottleneck.** CLIP reliability 0.50 among good configurations; fidelity ranks consistently (0.96 calibration → validation).
2. **Budget crossover.** At 150 evaluations random search is best; LHFiD, RVEA, MOEA/D and NSGA-III overtake it after 336–840; LHFiD gives the best 6-objective archive HV at 3,360.
3. **Training-free unit removal does not pay.** Single removals look safe, combinations interact (five safe attentions together −3.1 CLIP); on top of int8 removal saves ≤ 8% bytes and always costs fidelity; on 500 prompts one removed unit raises black images 1 → 8 and FID 85.4 → 90.0.
4. **Prompt-adaptive routing: a cautionary baseline.** Category-based routing beats the best single policy at matched mean latency but not the best random mixture of two fixed policies; per-prompt sensitivity is not reproducible across seeds (r = 0.12).
5. **What works.** int8 weights (non-inferior, half the bytes) plus 8–10-step samplers; DeepCache gives the best distribution match (KID) at 2.2×.

## 4. Baselines on the benchmark
- Random, NSGA-II, NSGA-III, MOEA/D, RVEA, LHFiD (two preference rules), surrogate-assisted variants; table of HV at 150 / 336 / 840 / 1,680 / 3,360 evaluations for M = 4 and 6.

## 5. Validation of benchmark-derived configurations
- Untouched validation split (43 policies, one allocation) and 500-prompt test (C1): non-inferiority of DDIM-10 + int8; comparison with BK-SDM at matched sampler and precision.

## 6. Limitations
- One GPU model, one model family (SD1.5), CLIP/Inception-based metrics; extension to SDXL / edge hardware left for future versions.
