# DiffusionNAS: instructions for Claude

Read this file before changing code or interpreting results. Work from the repository root. Check `git status --short` first; existing runs and uncommitted work belong to the user.

## Research goal and scope

The project seeks a better empirical quality–latency–memory Pareto frontier for diffusion inference on constrained hardware. The user's current direction is **training-free multi-objective optimization of inference policies**, with a small calibration prompt set for post-hoc policy selection and a separate hold-out set for validation. The present three-objective study is multi-objective; reserve "many-objective" for a genuinely higher-dimensional study with measured, nonredundant objectives. Do not introduce full retraining, distillation, supernet training, or a method needing more than two A100 40 GB days without explicit direction. Keep the first study short enough to execute in days, not weeks.

Use one frozen checkpoint for matched comparisons: `stable-diffusion-v1-5/stable-diffusion-v1-5`, 512×512, batch size 1. Compare methods on identical prompts, seeds, precision, and physical GPU. Record the exact GPU model: the user's recent interactive SLURM allocation exposed an **NVIDIA RTX 6000 Ada Generation**, which is different from an RTX A6000. Neither is an edge device; a frontier measured there is a cluster-GPU result, not an edge deployment claim.

The [Notion project hub](https://app.notion.com/p/3bc73ccc0c7e8008b0dec9839afa47b8) links the [literature review](https://app.notion.com/p/3bd73ccc0c7e80b2ad61d976e402e70d), [research directions](https://app.notion.com/p/3c773ccc0c7e80e7a1d0ec24083d22bd), and [implementation decision log](https://app.notion.com/p/3db73ccc0c7e81838a9cfacc6dd42d5c). The broader Notion proposals include ambitious architecture, surrogate, and adaptive-compute ideas. Apply the user's later training-free and compute-budget constraints when choosing what to implement next. Do not claim novelty from evolutionary diffusion NAS or evolutionary caching alone; AutoDiffusion, LDMOES, Flexiffusion, and ECAD are important prior work.

### Compression terminology and closest prior work

- **Static model compression** means fewer deployed parameters/bytes (and possibly lower resident weight memory), not just fewer operations. [LDMOES](https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Lightweight-Diffusion-Models-Based-on-Multi-Objective/991029465502346) searches lighter U-Nets using multi-objective evolution **with knowledge distillation**; [SnapFusion](https://arxiv.org/abs/2306.00980) redesigns a mobile U-Net/decoder and uses distillation. They are relevant compact-model comparisons, but their training pipelines are out of scope for our reproduction budget. A MAC reduction alone does not establish a smaller checkpoint.
- **Training-free execution reduction** means fewer denoising steps, skipped layers, or partial computation while keeping pretrained weights. [AutoDiffusion](https://arxiv.org/abs/2309.10438) searches timesteps and per-step layer-skipped subnetworks; its architecture-search experiments are on ADM-G, whereas its Stable Diffusion experiments search timesteps. [Flexiffusion](https://arxiv.org/abs/2409.17566) searches full/partial/null step patterns without retraining and explicitly evaluates SD1.5, reporting 5.1× speedup in its own protocol. Flexiffusion is therefore a **closer training-free SD1.5/U-Net precedent than ECAD** for execution-structure search. Do not compare its headline speedup numerically with ours without matching baseline, hardware, sampler, prompts, and quality protocol.
- **Feature reuse** is a different execution-reduction mechanism: [DeepCache](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.html) caches U-Net features; [ECAD](https://proceedings.iclr.cc/paper_files/paper/2026/file/e68274fc4f158dbcbd4dddc672f7ee9c-Paper-Conference.pdf) evolves per-step layer-wise cache schedules, evaluated on PixArt and FLUX diffusion transformers. Neither removes model weights by itself; activation caches may increase peak memory. The fact that ECAD was not evaluated on SD1.5 does **not** prove it cannot work on U-Nets.
- Call our present results **inference acceleration / effective compute reduction**, not deployed-model compression. A compression claim needs measured exported checkpoint size, parameter count, resident RAM/VRAM, and target-device latency. Even a layer-skipping policy may still require all original weights across its different steps unless a common smaller subnetwork is exported.

## What is implemented

| Area | Files | Current behavior / fidelity |
| --- | --- | --- |
| Config and execution | `src/diffusionnas/schema.py`, `runner.py`, `sd15_worker.py`, `cli.py` | Validates the locked SD1.5 protocol; creates per-run configs, manifests, logs, measurements, summaries, and images. CUDA-synchronized end-to-end generation latency and peak allocated/reserved VRAM are recorded. |
| Native reference | `configs/baseline_sd15.json` | DDIM baseline, labeled `reference`. |
| AutoDiffusion | `configs/autodiffusion_sd15.json`, worker, `search_space.py` | **SD1.5 timestep-schedule adaptation**, labeled `adapted`. It uses explicit descending timesteps with DPM-Solver++; it does not implement the paper's per-step layer-skipping/architecture search. The upstream repository is pinned for provenance, not invoked by this path. |
| DeepCache | `configs/deepcache_sd15.json`, worker | Calls the pinned upstream `DeepCacheSDHelper` for SD1.5; labeled `paper_implementation`. The runner adds the submodule to child `PYTHONPATH` without modifying or installing into it. |
| Quality and Pareto | `quality.py`, `pareto.py`, `holdout.py` | Scores saved images with frozen CLIP-L/14 outside timing; builds an empirical nondominated set; reruns selected policies on separate hold-out prompts/seeds. |
| Flexiffusion | `flexiffusion.py`, `configs/flexiffusion_sd15.json`, worker | **Reimplementation** of Flexiffusion's segment schedules (no official code), labeled `adapted`. Each segment = one full step, `interval − 1` partial steps (DeepCache reuse at `branch`, DeepCache `cache_branch_id` numbering) and null steps. Requires DDIM: a null step reuses the previous noise prediction, which for deterministic DDIM equals skipping that timestep (unit-tested). Uniform 6×[b0, i3] reproduces upstream DeepCache images bit-exactly on GPU. Flexiffusion's evolutionary search (Alg. 1–2, rFID ranking) is **not** implemented yet. |
| Controls and strata | `scripts/prepare_controls.py`, `scripts/slurm/controls_benchmark.sbatch`, `scripts/analyze_strata.py` | One-allocation control benchmark: references, native 4-step DPM-Solver++, and 18 uniform Flexiffusion schedules from its SD1.5 space (N-segment {7,9,11} × branch {1,3,6} × interval {2,3}); outputs under git-ignored `artifacts/controls/`. CPU per-stratum analysis writes to git-ignored `artifacts/analysis/`. |
| Search | `search_space.py`, `random_search.py`, `ecad.py`, `hypervolume.py`, `search_cli.py` | Shared 16-bit inference-policy space, equal-budget random control, and an **ECAD-inspired local evolutionary/NSGA-II-style search** with hypervolume traces. This is not the official ECAD paper implementation on SD1.5. |
| Cluster jobs | `scripts/prepare_experiment.py`, `scripts/slurm/`, `scripts/setup_cluster.sh` | Generates matched configs and provides separate SLURM jobs for generation, CLIP, Pareto, hold-out, random search, and evolutionary search. |

Official AutoDiffusion and DeepCache repositories are pinned Git submodules under `third_party/`. Initialize them with `git submodule update --init --recursive`; preserve their pinned commits and avoid edits inside them. See `README.md` and `scripts/README.md` for command details. The parent folder has separate `../benchmarking_plan.md` and `../training_free_diffusion_moo_plan.md`; keep benchmark procedure and overall research plan distinct.

The current search varies method, step count, scheduler, guidance, DeepCache interval/branch, and three predefined timestep patterns. It does **not** search U-Net layer removal, per-step cache decisions, or arbitrary timestep sequences. Thus it is an inference-policy search, not yet neural architecture search or an ECAD reproduction. Do not interpret comparisons between its evolutionary and random variants as a test of official ECAD on U-Nets.

## Completed experiments and interpretation (October 2026)

The matched calibration and hold-out numbers are in `docs/calibration_results.md`. All figures below are on an RTX 6000 Ada, not an edge device. Calibration used 48 COCO prompts × 2 seeds; hold-out used 128 image-disjoint COCO prompts × 2 different seeds. CLIP scoring was outside timed inference.

| Policy | Calibration: p50 ms / CLIP / peak MB | Hold-out: p50 ms / CLIP / peak MB | Interpretation |
| --- | --- | --- | --- |
| 20-step DDIM baseline | 505.03 / 26.215 / 3256.42 | 514.09 / 27.068 / 3271.34 | Reference. |
| 20-step DDIM DeepCache, interval 3, branch 0 | 239.29 / 26.157 / 3380.40 | 237.92 / 26.908 / 3380.53 | ~2.1× faster, but ~110–124 MB more peak allocated memory; hold-out CLIP difference −0.160 vs baseline is unresolved by paired bootstrap. |
| 4-step DPM-Solver++ custom timestep adaptation | 142.81 / 25.264 / 3269.03 | 140.88 / 25.894 / 3269.03 | ~3.6× faster, but hold-out CLIP difference −1.174 and six safety-black outputs out of 256. Solver and step count both change, so no attribution to the custom schedule yet. |

**AutoDiffusion attribution.** The adaptation's list `[999, 749, 499, 249]` is exactly Diffusers' native DPM-Solver++ `trailing` spacing; the SD1.5 default (`linspace`, no `timestep_spacing` in its scheduler config) gives `[999, 749, 500, 250]`. Treat the point as a native 4-step DPM-Solver++ result unless the pending control shows otherwise.

**Per-stratum analysis (October 2026, `scripts/analyze_strata.py`).** On hold-out, DeepCache's CLIP loss is concentrated in *detailed* prompts (−0.455, 95% CI [−0.99, +0.09]) and is ~0 on simple (+0.000) and relational (−0.021); the 4-step DPM-Solver++ loss is also largest on detailed (−1.60). Across 106 unique calibration search policies, per-stratum CLIP ranks correlate 0.83–0.86, and best-under-budget winners differ by stratum, but 16 prompts per stratum cannot resolve those differences. This is an exploratory input-adaptive signal, not a result.

Both 100-candidate searches completed under the same calibration protocol (random 72.8 min, evolutionary 62.5 min wall time on the RTX 6000 Ada, far below D017's 5h17m estimate). Random search found 71 unique effective policies and the ECAD-inspired search 59; duplicates consumed 29 and 41 evaluations respectively. The evolutionary run's raw hypervolume was only 0.124% higher, and its lead disappears when one low-quality candidate is removed or a quality floor is imposed. Repeated policies also showed ~2% run-level latency drift. **Do not claim evolutionary search beat random search.** An attractive evolutionary calibration finalist was a plain 8-step DDIM baseline (guidance 10; 235.55 ms, CLIP 26.565, 2/96 black outputs), not an architecture-compressed or cached policy; its quality advantage over the 20-step baseline was not resolved by the calibration bootstrap. Search finalists have **not** undergone hold-out validation. The existing hold-out covers only the three initial benchmark policies.

Immediate, bounded follow-up: lock a small set of calibration-selected finalists **before evaluation**; use a fresh image-disjoint validation split for a clean final test, since the original hold-out results have already been inspected. The existing hold-out can serve as a secondary comparison but is no longer fully untouched for new hypotheses. Add a **native four-step DPM-Solver++ matched control** to isolate the AutoDiffusion timetable; include Flexiffusion as the closest SD1.5 training-free execution-search baseline in research claims and, if feasible within the time budget, in benchmarking. Deduplicate policies (note that DeepCache interval 1 executes no caching and duplicates a baseline policy under another label) and use quality constraints/uncertainty when rebuilding fronts. Only after these controls should a new joint U-Net architecture/timestep/cache search be claimed as a demonstrated contribution. A target-device study is required before any edge Pareto or compression claim.

## Evaluation contract

- Calibration quality: `openai/clip-vit-large-patch14`; score is `100 * max(cosine_similarity(image, text), 0)`. It is a prompt-alignment proxy, not a complete image-quality measure.
- Default Pareto objectives: mean calibration CLIP **maximize**, p50 end-to-end generation latency **minimize**, peak allocated GPU memory **minimize**. p90 and reserved memory are logged. Energy remains `null`; include it only after measuring actual joules/image for every eligible policy. No imputed objectives.
- Completed split: 48 calibration prompts × 2 seeds (96 images per candidate); 128 image-disjoint hold-out prompts × 2 different seeds. The original hold-out was used for the three initial benchmark policies and its results are now known. Prompt JSON files are supplied separately and are not bundled. `runtime.repetitions=1` means one timed run **per prompt/seed pair**, not one run per policy.
- Hold-out validation reruns only calibration-front policies, recomputes their front, and reports retention plus CLIP rank correlation. Never use hold-out results to tune the search. This selected-only check cannot show whether a calibration-dominated policy would become nondominated on hold-out.
- Final-set KID/FID, diversity/LPIPS, measured energy, and edge-device benchmarking appear in the plans but are **not implemented** here. Treat current fronts as provisional and hardware-specific.

Each run lives under its configured `output_root`, normally `artifacts/runs/` or `artifacts/calibration/runs/`, with `config.json`, `manifest.json`, `stdout.log`, `stderr.log`, `measurements.jsonl`, `summary.json`, and saved images. Scoring adds `quality_measurements.jsonl`. The manifest records fidelity, project/upstream commits, versions, and the CUDA-visible GPU. Failed and dry runs must not enter a measured Pareto front. Existing `artifacts/` files are user experiment data; preserve them. The current `.gitignore` does not ignore all artifacts, so inspect what is staged before any commit.

## Known failures and fixes

1. **Custom timesteps with DDIM failed.** In this Diffusers 0.32 harness, DDIM did not accept the supplied timestep list. AutoDiffusion candidates require DPM-Solver++ and descending timesteps; schema validation enforces this.
2. **NumPy/PyTorch ABI warning in the global environment.** Use the project `.venv` with `numpy<2`; do not alter global Python. Import DeepCache from its submodule path so it stays clean.
3. **SLURM batch path failure.** `sbatch` copies `.sbatch` files to `/var/lib/slurm/.../slurm_script`. Deriving the repository root from `BASH_SOURCE[0]` produced `/var/lib/slurm/scripts/slurm/job_env.sh` and failed before inference. All batch scripts now use `DIFFUSIONNAS_ROOT`, falling back to `SLURM_SUBMIT_DIR`. Submit from the repo root or export an absolute `DIFFUSIONNAS_ROOT`. The checked-in `scripts/slurm/slurm-8498_*.out` logs are from this **pre-fix** failure.
4. **CPU Pareto job would have failed CUDA preflight.** `build_pareto.sbatch` now sets `DIFFUSIONNAS_REQUIRE_GPU=0`; other jobs still require CUDA.
5. **Flexiffusion details absent from the paper.** No official code, no published SD1.5 schedules, guidance scale or null-step sampler semantics; the paper's SD baseline is 50-step PLMS. Our choices (DDIM, guidance 7.5, DeepCache branch numbering, segment length 3 per Appendix E.1) are documented in `flexiffusion.py`. Do not compare its reported 5.1× speedup with ours.
6. **First timed image was slow after latent-only warm-up.** The worker now warms with `output_type="pil"`, matching timed calls, and `provenance.py` records exact GPU name, total memory, compute capability, and CUDA runtime. A local post-fix baseline smoke run succeeded; the Notion decision log reports the first cluster baseline smoke success. Neither is a full matched benchmark.

## How to work here

Start with `git status --short`, the relevant source files, and the newest run's `manifest.json`, `summary.json`, and `stderr.log`. Some dated status prose in `README.md` and `docs/benchmark_plan_review.md` predates CLIP, Pareto, hold-out, and search code; verify current behavior in source and tests. Preserve existing configs, prompt splits, and run artifacts unless the user explicitly requests changes.

Useful local checks:

```bash
python -m unittest discover -s tests -v
bash -n scripts/slurm/*.sbatch scripts/slurm/job_env.sh
diffusionnas preflight
diffusionnas run configs/baseline_sd15.json --dry-run
python -m diffusionnas.search_cli random --config configs/random_search_sd15.json --dry-run
python -m diffusionnas.search_cli ecad --config configs/ecad_sd15.json --dry-run
```

On the cluster, an interactive `srun --gres=gpu:1 --pty bash -i` can check CUDA and run the one-prompt baseline. It does not test `sbatch` spool behavior or array variables. For batch execution, prepare configs from a real calibration JSON file, then submit `scripts/slurm/benchmark_array.sbatch` from the checkout with `sbatch --export=ALL,DIFFUSIONNAS_ROOT="$PWD" ...`. Follow with CLIP scoring, CPU Pareto construction, and hold-out validation; see `scripts/README.md`. Verify `nvidia-smi` and the recorded GPU before comparing latency across jobs.

## Sources

- [Notion project hub](https://app.notion.com/p/3bc73ccc0c7e8008b0dec9839afa47b8)
- [Notion literature review](https://app.notion.com/p/3bd73ccc0c7e80b2ad61d976e402e70d)
- [Notion research directions](https://app.notion.com/p/3c773ccc0c7e80e7a1d0ec24083d22bd)
- [Notion implementation decision log, D001–D018](https://app.notion.com/p/3db73ccc0c7e81838a9cfacc6dd42d5c)
