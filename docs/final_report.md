# Training-free multi-objective neural architecture search for Stable Diffusion 1.5

*DiffusionNAS — BTP final report (draft; results sections are filled from committed analysis scripts).*

## 1. Question

Can a training-free, multi-objective search over **U-Net architecture edits** (unit removal, int8 weight storage) and **inference policies** (sampler, steps, guidance, feature caching, segment-wise step skipping) produce Stable Diffusion 1.5 configurations that are smaller and faster than the frozen model while keeping image quality, and how do they compare with established training-free accelerators and with a distilled compact model?

## 2. Setup

- **Model.** `stable-diffusion-v1-5/stable-diffusion-v1-5`, frozen weights, 512×512, batch 1, fp16, safety checker on. No training, fine-tuning or distillation anywhere.
- **Hardware.** One NVIDIA RTX 6000 Ada Generation (48 GB) per job; every latency comparison is within one GPU model, and within one allocation or normalized by a reference measured in the same allocation. This is a workstation GPU, not an edge device.
- **Data.** Three disjoint MS-COCO 2014 caption splits, stratified by caption complexity (simple / relational / detailed): calibration (48 prompts × seeds 42, 123) for every search and selection decision; the original hold-out (128 prompts), used only for the first benchmark; and an untouched validation split (128 prompts × seeds 303, 404) used once, for the final evaluation.
- **Metrics.**
  - *Prompt alignment:* CLIP score, `100·cos(image, text)` with frozen CLIP-L/14.
  - *Fidelity to the original model:* CLIP image–image similarity to the 20-step DDIM reference image for the same prompt and seed.
  - *Distribution quality:* KID (primary) and FID with the standard Inception-v3 features, against 5,000 real COCO images and against the reference model's own images.
  - *Cost:* CUDA-synchronized end-to-end p50 latency, GFLOPs per image (text encoder + all U-Net calls + VAE decode, counted by `torch.utils.flop_counter`), peak allocated VRAM, U-Net parameters and stored weight bytes.
  - *Failures:* images replaced by the safety checker (all black) are counted and kept in every score.
  - *Uncertainty:* paired prompt-level bootstrap (10,000 draws, seeds kept together) for every quality difference.

## 3. Search space

| Group | Choices | What it changes |
| --- | --- | --- |
| Static unit removal | Any subset of 15 U-Net units selected by a single-unit sensitivity probe (attention blocks and channel-preserving ResNets) | Deletes weights: parameters, bytes, VRAM, FLOPs |
| Weight precision | fp16 or int8 weight-only (per-output-channel scale, dequantized per call) | Halves U-Net bytes; adds latency in this implementation |
| Sampler | DDIM or DPM-Solver++; 4–25 steps; guidance 5–12 | Number of U-Net evaluations |
| Feature reuse | DeepCache (interval 2–5, branch 0–11) or Flexiffusion segment schedules (4–11 segments of length 3, per-segment branch {1, 3, 6} and interval {1, 2, 3}) | Partial and skipped U-Net evaluations |

Removable units are those whose output can be replaced by their input without breaking tensor shapes: the 16 residual attention blocks and the 8 ResNet blocks whose input and output channels match. Up-block ResNets consume concatenated skip connections and are not removable without retraining. Removal deletes the modules, so savings in parameters and bytes are measured, not estimated.

## 4. Method

1. **Sensitivity probe (calibration).** Each of the 24 removable units removed alone, six groups, and int8 weights; a unit enters the search space if its removal does not significantly lower CLIP (upper end of the paired 95% CI ≥ 0) and keeps fidelity ≥ 80.
2. **Search (calibration).** Random search and NSGA-II with identical variation operators, 150 unique candidates each (an execution signature is never evaluated twice), three optimizer seeds, one GPU job per seed with its own reference run. Search objectives: CLIP, fidelity, latency, U-Net bytes; GFLOPs and peak VRAM are measured for every candidate. Hypervolume uses a reference point fixed before the searches: CLIP 25.2 (reference − 1.0), fidelity 75, 600 ms, 1,700 MB.
3. **Finalist lock.** A rule committed before the searches ran: pool all candidates, normalize latency by each job's reference, merge identical signatures, keep CLIP ≥ reference − 1.0, fidelity ≥ 75 and at most two extra black images, take the 4-objective front, choose up to 10 by per-objective best plus farthest-point spread.
4. **Validation (untouched split, one allocation).** The finalists; an *architecture-ablation twin* for each finalist (same sampler and caching, no unit removal, fp16; and int8 without removal for int8 finalists); DDIM and DPM-Solver++ step sweeps; DeepCache; int8 at 20 steps; two Flexiffusion schedules; BK-SDM base/small/tiny; and the 20-step reference at the start, middle and end of the job to measure drift. Run order is shuffled with a fixed seed.
5. **Many-objective optimizer benchmark (surrogate).** Real evaluations allow only 150 candidates per run, too few to compare selection schemes designed for thousands. Surrogates of all six objectives are fitted to the 900 real evaluations, and random search, NSGA-II, NSGA-III and a reimplementation of LHFiD (Saxena, Mittal, Kapoor and Deb, COIN Report 2021020) are compared at 3,360 evaluations on 4- and 6-objective formulations, across five independently fitted surrogates × seven seeds, with the protocol fixed in `configs/nas/surrogate_benchmark.json` before any run. Points from each optimizer are then measured for real.

## 5. Results

*To be filled from `scripts/summarize_probe.py`, `scripts/analyze_search.py`, `scripts/surrogate_benchmark.py`, `scripts/summarize_confirmation.py` and `scripts/analyze_validation.py`.*

## 6. Reproducibility

Every number in this report comes from a committed script and a result file under `artifacts/` (git-ignored because of size; copied from the cluster). The phase tracker and run log are in `docs/RESEARCH_PLAN.md`; protocol rules and known failures are in `CLAUDE.md`; design decisions are in the Notion decision log (D001–D021+).
