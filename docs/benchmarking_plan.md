# Benchmarking Plan: Training-Free Diffusion Inference Policies

## Goal and scope

Benchmark only **post-hoc, training-free** diffusion acceleration methods. A method is eligible only if it uses public pretrained weights and can be reproduced without fine-tuning, distillation, supernet training, or more than two A100-40GB days of compute.

The benchmark asks: for one fixed pretrained checkpoint and one fixed edge target, what quality–latency–memory–energy Pareto frontier is achievable through inference-time policies?

## Papers and baselines to benchmark

| Tier | Paper / method | What to benchmark | Status in the protocol |
|---|---|---|---|
| Mandatory | Native Stable Diffusion pipeline | DDIM and DPM-Solver/DPM-Solver++ at fixed NFEs | Establishes the no-acceleration frontier |
| Mandatory | **AutoDiffusion (SD1.5 adaptation)** | Timestep-sequence search/application using the paper's Stable Diffusion path as reference | Training-free schedule baseline; label adapted because upstream uses SD-v1.4 and does not expose SD1.5 architecture search |
| Mandatory | **DeepCache** | Feature reuse/caching heuristic | Strong training-free caching baseline for Stable Diffusion |
| Recommended | **ECAD** | Evolutionary caching schedules | Include only if its released code/checkpoint path supports the selected model; do not port it across model families for this first benchmark |
| Mandatory control | Random search | Same inference-policy space and same evaluation budget as evolutionary search | Tests whether search is actually useful |
| Mandatory control | NSGA-II | Same policy space and evaluation budget | Standard multi-objective evolutionary baseline |
| Optional | NSGA-III or MOEA/D | Same policy space and evaluation budget | Use only with four or more reliable objectives |

### Explicit exclusions

- **SnapFusion:** requires distillation/training; cite as context only.
- **LDMOES:** requires supernet/knowledge-distillation training; cite as architecture-MOO context only.
- Any pruning/quantization method needing weight recovery or QAT.
- Methods that require a non-public checkpoint or unsupported hardware/runtime.

This gives a fair benchmark of **inference-time policy optimization**, rather than mixing it with trained compact-model papers.

## Dataset and prompt protocol

### Primary evaluation: MS-COCO captions

Use an MS-COCO caption split so generated images can be evaluated consistently with text-to-image literature.

- **Calibration set:** 48 prompts, balanced across simple object, multi-object/compositional, and detailed-scene prompts; use 2 fixed seeds/prompt (96 generations per candidate).
- **Hold-out set:** 128 prompts with 2 fixed seeds/prompt. Never use these results in optimization or policy selection.
- **Final quality set:** 1,000 captions with one fixed seed each, evaluated only for 3–5 final nondominated policies and the strongest baselines.

Keep exact captions, seed values, negative prompts, resolution, guidance scale, and sampler settings in version control. For every method that owns the timestep/sampler decision, record that policy rather than forcing it to share a sampler.

### Optional stress set: PartiPrompts or equivalent

After the primary benchmark works, add a small 64-prompt diagnostic split covering compositional prompts, attributes, text, and long prompts. Report CLIP/alignment and failure rates only; do not merge it into COCO FID/KID.

## Model and hardware settings

### Initial model

Use **Stable Diffusion 1.5, 512x512, batch size 1**. Lock the current Diffusers mirror, `stable-diffusion-v1-5/stable-diffusion-v1-5`, because the older RunwayML repository used by upstream code is deprecated. For AutoDiffusion, initially compare timestep-sequence search only; do not imply that the paper's separate architecture-search code has been reproduced on SD1.5. If SD1.5 cannot execute on the actual edge device, switch once to a smaller public latent-diffusion model and rerun all baselines; do not mix model families in the main frontier.

### Hardware

Use one physical target device first. Record device SKU, RAM, OS, driver, inference runtime, precision, compiler/graph mode, and thermal/power mode. Benchmark on a fixed performance mode after a warm-up period.

An A100 may be used for development, calibration, and final image metrics, but **the claimed deployment frontier must use the edge target’s measurements**.

## Metrics

### Optimization-time metrics (calibration prompts)

Use metrics cheap enough to evaluate 100–250 candidates:

| Objective | Metric | Direction | Notes |
|---|---|---|---|
| Prompt quality | CLIP text-image similarity | maximize | Normalize only within a locked benchmark run |
| Latency | End-to-end p50 ms/image | minimize | Include text encoding, denoising, VAE decode; exclude one-time model load |
| Tail latency | p90 ms/image | minimize / report | Do not optimize separately at first unless it changes rankings |
| Memory | Peak RAM/VRAM | minimize | Include cache memory |
| Energy | Joules/image | minimize | Include only with a real external/device power measurement |
| Validity | crash/OOM rate | minimize / constraint | Invalid policies are infeasible, not poor Pareto points |

Use `1 - normalized CLIP`, p50 latency, peak memory, and energy as the initial minimization objectives. If energy cannot be measured reliably, run three-objective optimization rather than inventing an energy proxy. Record MACs/FLOPs as diagnostics only.

### Hold-out and final-quality metrics

- **KID:** preferred for the 1,000-image final set because it has an unbiased finite-sample estimator.
- **FID:** report only with the number of generated/reference images and preprocessing stated; use it to compare policies within this exact protocol, not across papers.
- **CLIP similarity:** prompt alignment.
- **LPIPS diversity:** for 32 selected prompts, generate four seeds each and report mean pairwise LPIPS. This catches a policy that buys speed by collapsing diversity.

### Pareto/search metrics

- Hypervolume (HV), after fixed objective normalization and reference point declaration.
- Nondominated-set size and empirical attainment/frontier plots.
- HV versus number of real candidate evaluations.
- Search cost: wall-clock time, A100-hours (if used), edge-device hours, candidate count, and calibration generations.

Do not report IGD unless a defensible reference front exists; for a new black-box diffusion policy space, the true front is unknown.

## Fairness rules

1. Compare policies only under the same checkpoint, resolution, prompt/seed split, device, precision, and end-to-end latency definition.
2. All optimizers receive the same number of **real** policy evaluations and the same calibration prompts.
3. All methods must use public, frozen weights; no post-hoc weight update is permitted.
4. Measure latency after 10 warm-up generations, then use at least 30 timed runs. Report median and p90, not a single best run.
5. Check every final nondominated policy on the hold-out prompts. A calibration-only frontier is provisional.
6. Include cache activation memory. A faster cached policy that OOMs on the target is infeasible.

## Execution steps

### Step 1 — Build the policy evaluator (Days 1–2)

Create a single policy JSON schema:

`{sampler, timesteps/NFE, guidance, block_mask, cache_policy, precision}`

The evaluator takes this JSON and emits images, per-run timing, peak memory, energy (if available), and quality scores. First validate 12 native policies: DDIM and DPM-Solver variants at 4/8/12/20 steps.

**Output:** raw result table and first native quality–latency–memory plot.

### Step 2 — Reproduce post-hoc baselines (Days 3–4)

Run DeepCache and the SD1.5 AutoDiffusion timestep adaptation with the locked protocol. Record implementation fidelity (`paper_implementation`, `adapted`, or `reference`) in every run manifest. Add ECAD only if the official implementation supports the selected base model without a substantial reimplementation.

**Output:** one matched baseline table; record unavailable methods and the exact incompatibility.

### Step 3 — Build the calibration archive (Days 5–6)

Evaluate 80–120 feasible policies sampled across all enabled decisions. Use the 48-prompt calibration set. Re-evaluate 15–20 of them on hold-out prompts and compute Spearman rank correlation for calibration CLIP versus hold-out CLIP/KID.

**Output:** evidence for whether few-shot scores are reliable enough to drive search. Retain direct calibration evaluation if correlation is weak; do not claim a surrogate.

Implementation commands:

```bash
diffusionnas score --runs-root artifacts/calibration/runs
diffusionnas pareto --runs-root artifacts/calibration/runs --output artifacts/calibration/pareto_front.json
diffusionnas holdout --front artifacts/calibration/pareto_front.json --prompts data/prompts/holdout.json --seeds 101,202 --output artifacts/holdout
```

The hold-out command reruns only selected calibration-front policies. It must not modify model weights or feed hold-out scores back into policy selection.

### Step 4 — Benchmark optimizers (Days 7–8)

Run random search and NSGA-II under the same 100–150-candidate budget. Run NSGA-III only if energy is available and the fourth objective is stable. Compute HV curves from calibration results and re-evaluate each method’s final policies on hold-out prompts.

**Output:** matched optimizer comparison and the first validated training-free Pareto frontier.

### Step 5 — Final evaluation (Days 9–10)

Select 3–5 policies from the union of nondominated baseline/search policies plus the native reference. Run the 1,000-prompt final set; measure KID, FID, CLIP, LPIPS diversity, p50/p90 latency, memory, and energy.

**Output:** final Pareto plot and a reproducibility bundle containing policy JSONs, raw measurements, captions, seeds, and environment file.

## Decision gates

- If the edge device cannot fit SD1.5 at batch 1, change the base model before Step 2, then restart the benchmark.
- If caching improves latency but violates memory, keep it as an infeasible/constraint-violating result; do not hide memory.
- If few-shot calibration ranking is weak, do not introduce a learned surrogate. Use direct few-shot evaluation and keep the search budget small.
- If no method dominates the native solver baseline, the useful output is still a negative result: identify which control—schedule, capacity, or cache—fails to translate to the target hardware.
