# SD 1.5 calibration and hold-out benchmark — 29–30 September 2026

Status: calibration generation, CLIP scoring, Pareto construction, and hold-out validation completed.

## Protocol

- Model: Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), 512 × 512, batch size 1, FP16, guidance scale 7.5.
- Hardware: one NVIDIA RTX 6000 Ada Generation (48 GB) allocation for all three generation runs. This is a workstation benchmark, not an edge-device measurement.
- Evaluation: 48 frozen COCO calibration captions × seeds 42 and 123 = 96 images per policy; 10 full-path warm-ups, then one timed repetition per prompt–seed pair.
- Quality: frozen CLIP-L/14 (`openai/clip-vit-large-patch14`), scored after generation so scoring time is excluded from latency.
- Policies: baseline = 20-step DDIM; DeepCache = 20-step DDIM, interval 3, branch 0, uniform skip; AutoDiffusion adaptation = four DPM-Solver++ timesteps [999, 749, 499, 249].
- Artifact provenance: all three manifests identify the same GPU and project commit `91be533c447cc77df577364d50a0aeea48feacd2`. They also report `project_dirty: true`; preserve the exact checkout and artifacts for publication.

## Measured results

| Policy | Fidelity | p50 latency (ms) | p90 (ms) | Speedup vs baseline | Mean CLIP | Peak allocated (MB) | Safety-black images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Reference | 505.03 | 511.79 | 1.00× | 26.215 | 3,256.42 | 0/96 |
| DeepCache | Paper implementation | 239.29 | 243.93 | 2.11× | 26.157 | 3,380.40 | 1/96 |
| AutoDiffusion | SD 1.5 adaptation | 142.81 | 146.40 | 3.54× | 25.264 | 3,269.03 | 4/96 |

All runs succeeded and produced 96 timed samples and 96 CLIP scores. DeepCache used about 124 MB more peak allocated memory than baseline; AutoDiffusion used about 13 MB more. Energy was not measured. The [calibration Pareto artifact](../artifacts/calibration/pareto_front.json) marks all three policies nondominated under mean CLIP (maximize), p50 latency (minimize), and peak allocated memory (minimize), with no skipped runs.

## Interpretation and limits

- DeepCache cuts median latency by about 53% with a mean CLIP difference of −0.057 versus baseline. An exploratory paired bootstrap over the 48 prompts gives a 95% interval of roughly [−0.80, +0.55] CLIP points; the observed quality gap is not resolved by this calibration set.
- AutoDiffusion cuts median latency by about 72%, with mean CLIP lower by 0.951. Four outputs were blacked out by the safety checker versus zero for baseline. Excluding those four paired samples changes the observed mean gap to about −0.30, but that exclusion is diagnostic only and must not replace the prespecified score.
- The AutoDiffusion comparison changes both solver and step count (four-step DPM-Solver++ versus 20-step DDIM). It does **not** isolate the benefit of its particular timestep schedule or establish an architecture-search gain. A native four-step DPM-Solver++ control is still needed for that attribution.
- The three-point Pareto front is provisional: it does not measure energy, final distributional quality (FID/KID), search performance, or latency/memory on the intended edge device. CLIP measures prompt–image alignment, not overall image fidelity.

## Hold-out validation — 30 September 2026

The 128 image-disjoint COCO hold-out captions used seeds 101 and 202, giving 256 timed images and 256 CLIP scores per policy. All three runs and the scoring/validation job succeeded on an RTX 6000 Ada.

| Policy | p50 latency (ms) | p90 (ms) | Speedup vs hold-out baseline | Mean CLIP | Peak allocated (MB) | Safety-black images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 514.09 | 518.47 | 1.00× | 27.068 | 3,271.34 | 0/256 |
| DeepCache | 237.92 | 240.19 | 2.16× | 26.908 | 3,380.53 | 1/256 |
| AutoDiffusion | 140.88 | 142.43 | 3.65× | 25.894 | 3,269.03 | 6/256 |

The [hold-out validation report](../artifacts/holdout/holdout_validation.json) records **3/3 calibration-front policies retained** on the [hold-out Pareto front](../artifacts/holdout/holdout_pareto_front.json), with no skipped runs. The CLIP ranking is unchanged (Spearman ρ = 1.0). This correlation is computed from only **three policies**; it does not validate ranking across a larger search space.

- DeepCache minus baseline mean CLIP is −0.160. An exploratory paired bootstrap resampling 128 prompts (10,000 draws, keeping both seeds together) gives a 95% interval of about [−0.461, +0.123]. The quality difference remains unresolved, while the roughly 2.16× latency gain persists.
- AutoDiffusion minus baseline mean CLIP is −1.174, with paired prompt-bootstrap 95% interval about [−1.600, −0.741]. Six outputs were safety-black versus zero for baseline. Excluding those six pairs leaves a diagnostic mean gap of about −0.953, so filtering alone does not explain the hold-out quality loss. The prespecified score includes all outputs.
- All three hold-out means exceed their calibration means; the prompt sets differ, so this is not a within-policy improvement. Compare paired methods *within* each split, not raw means *between* splits.

Hold-out manifests record project commit `6f9f1f8ad5b5ddd6cf1a3381843a1d128acfea31` and `project_dirty: true`. No files under `src/`, `scripts/`, or `configs/` changed between the calibration and hold-out commits. Hold-out results were not fed back into calibration selection.

## Source artifacts

- [Baseline summary](../artifacts/calibration/runs/20260929T192729671988Z_baseline_1abddf7f4f/summary.json)
- [DeepCache summary](../artifacts/calibration/runs/20260929T193009025063Z_deepcache_c205271943/summary.json)
- [AutoDiffusion summary](../artifacts/calibration/runs/20260929T192938924723Z_autodiffusion_58af617a7f/summary.json)
- [Hold-out baseline summary](../artifacts/holdout/runs/20260930T182013972434Z_baseline_2f20413cee/summary.json)
- [Hold-out DeepCache summary](../artifacts/holdout/runs/20260930T181842130607Z_deepcache_d81fb487bc/summary.json)
- [Hold-out AutoDiffusion summary](../artifacts/holdout/runs/20260930T181729968814Z_autodiffusion_0f38954d63/summary.json)
- Slurm logs: `slurm-8658.out` (calibration generation), `slurm-8659.out` (CLIP), `slurm-8660.out` (Pareto), and `slurm-8689.out` (hold-out).
