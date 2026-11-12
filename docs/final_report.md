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

All numbers are from the RTX 6000 Ada; the 20-step DDIM reference takes about 505 ms, 34.7 TFLOPs, a 1,639 MB U-Net (859.5 M parameters) and 3.27 GB peak VRAM per image.

### 5.1 Sensitivity probe (calibration) — `scripts/summarize_probe.py`

- **int8 weights are free in quality.** U-Net 1,639 → 821 MB, peak VRAM 3,271 → 2,438 MB, CLIP +0.04 [−0.26, +0.32], fidelity 95.9. The cost is +24% latency from dequantizing on every call; a fused int8 kernel would remove most of it, so this penalty is an implementation property, not an int8 property.
- **15 of 24 units are individually removable** (CLIP CI reaches 0, fidelity ≥ 80). Deep units save 59–66 MB each; 64-px attentions save 4.5% FLOPs each but only 5 MB.
- **Harmful units:** `down0.res0` (−11.5 CLIP, 16 black images), `up3.attn2` (−5.1, 15 black), `down0.res1`, `mid.attn0`, `down2.res1`, `up1.attn0/1`, `mid.res0`, `down1.attn0`.
- **Removals interact.** Five individually harmless 32-px attentions removed together cost −3.1 CLIP; all 16-px attentions −7.7.
- **Latency does not follow FLOPs at batch 1.** Removing all 64-px attention cuts 22% of FLOPs but 18% of latency; deep-unit removals move latency by about ±3%.

### 5.2 Search (calibration) — `scripts/analyze_search.py`, `docs/figures/search_*.png`

- 900 unique candidates (3 seeds × random / NSGA-II × 150); 233 pass the quality floors; job-to-job reference drift 5% (485–511 ms), so latency is normalized per job.
- **No consistent optimizer advantage at 150 real evaluations.** NSGA-II has higher hypervolume in 2 of 3 seeds, with the same pattern when latency is replaced by the drift-free GFLOPs, and job order does not explain it.
- **Why:** within the quality-floor region, calibration CLIP has a split-half reliability of only 0.50 (Spearman–Brown), i.e. at most 0.71 correlation with its own noise-free value. With 96 images per candidate, ranking *good* candidates by CLIP is mostly prompt-sampling noise.

### 5.3 Validation (untouched split, one allocation) — `scripts/analyze_validation.py`, `docs/figures/validation_front.png`

Reference latency at the start, middle and end of the job: 507.1 / 508.0 / 507.3 ms (no drift). Quality differences are paired prompt-bootstrap 95% CIs against the reference; "significant" means Holm-adjusted p < 0.05 over all 42 comparisons.

| Finalist | Policy | Latency | Speedup | GFLOPs | Peak VRAM | U-Net | ΔCLIP [95% CI] | Fidelity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| nas_01 | DPM++ 6 steps, g 6.5 | 181 ms | 2.80× | 12,180 | 3,270 MB | 1,639 MB | −0.10 [−0.41, +0.21] | 86.9 |
| **nas_04** | DDIM 10, g 10.5, int8, −1 unit | 322 ms | 1.58× | 18,210 | **2,439 MB** | **812 MB** | +0.10 [−0.26, +0.49] | 87.5 |
| nas_05 | Flexiffusion 30 steps, int8, −1 unit | 476 ms | 1.06× | 28,376 | 2,554 MB | 818 MB | −0.07 [−0.29, +0.15] | 90.6 |
| nas_08 | DPM++ 20, int8, −3 units | 566 ms | 0.90× | 31,630 | 2,398 MB | 776 MB | −0.02 [−0.57, +0.54] | 82.4 |
| nas_09 | DPM++ 25, g 6.5, int8 | 761 ms | 0.67× | 42,705 | 2,438 MB | 821 MB | +0.10 [−0.23, +0.46] | 91.0 |
| nas_02 | DPM++ 6, g 5, int8 | 214 ms | 2.37× | 12,180 | 2,446 MB | 821 MB | −0.57 [−0.99, −0.20] | 86.7 |
| nas_03 | DPM++ 6, g 12, int8, −1 unit | 212 ms | 2.40× | 11,970 | 2,415 MB | 788 MB | −0.88 [−1.37, −0.38]* | 80.2 |
| nas_00 | DPM++ 6, g 5, −4 units | 161 ms | 3.14× | 10,995 | 3,128 MB | 1,504 MB | −1.11 [−1.66, −0.59]* | 79.7 |
| nas_07 | DPM++ 20, g 9, int8, −7 units | 518 ms | 0.98× | 29,193 | 2,319 MB | 696 MB | −1.20 [−1.81, −0.58]* | 77.0 |

\* significant after Holm correction. nas_06 (Flexiffusion 33 steps, fp16) is omitted: 1.03×, −0.06 CLIP.

- **Generalization.** 7 of 10 finalists remain inside the pre-registered reference box on validation. Calibration-to-validation rank agreement across finalists: CLIP 0.87, fidelity 0.96, latency 0.99.
- **Best compressed configuration (nas_04):** half the U-Net bytes, 830 MB less peak VRAM, 1.58× faster, 47% fewer FLOPs, with no measurable loss in prompt alignment.
- **Fastest quality-preserving configuration (nas_01):** 2.8× faster, CLIP −0.10 [−0.41, +0.21]; it is a sampler/guidance choice (6-step DPM-Solver++ at guidance 6.5), not an architecture change.
- **Training-free baselines in the same job:** DeepCache 2.18× (+0.05), DPM++ 8 steps 2.26× (+0.10), DPM++ 4 steps 3.74× (−1.10*), DDIM 6 steps 2.82× (−1.00*), int8 at 20 steps 0.82× with half the bytes (+0.01, fidelity 96.3). Flexiffusion schedules sit on the same frontier as DeepCache (2.09×, −0.05).

**What the architecture genes contribute (ablation twins).** Each finalist was re-run with the same sampler and caching but no unit removal (fp16, and int8 where applicable):

- int8 twins carry almost all of the compression: removing units on top of int8 saves only 2–125 MB more U-Net bytes.
- Unit removal never significantly improved CLIP over its twin and always cost fidelity: −1.0 to −12.1 points (e.g. nas_07 vs its int8 twin: −1.43 CLIP, −12.0 fidelity; nas_04 vs its int8 twin: +0.09 CLIP, −2.2 fidelity, −9 MB).
- On the validation 4-D hypervolume (pre-registered reference, CLIP floor = validation reference − 1.0), the finalists reach 4.26 × 10⁶ against 0.52 × 10⁶ for the plain training-free baselines; but the baselines *plus* twins reach 4.42 × 10⁶, more than the finalists. The finalists' advantage comes from int8 and the sampler choice the search found, not from removing units.

**Comparison with BK-SDM (distilled from SD-v1.4, external).** At 20-step DDIM, BK-SDM base / small / tiny run at 1.51× / 1.58× / 1.74× with U-Nets of 1,105 / 920 / 617 MB and lose 0.44 / 1.55* / 1.71* CLIP. nas_04 is smaller, uses less VRAM and has higher alignment than BK-SDM base and small at equal or better latency, but the comparison is confounded: nas_04 uses 10 steps and int8, both of which apply to BK-SDM too, and BK-SDM has far fewer parameters (482–579 M vs 851 M). Matched post-hoc BK-SDM runs (10-step DDIM, fp16 and int8) are in §5.5.

### 5.4 Many-objective optimizer benchmark (surrogates) — `scripts/surrogate_benchmark.py`, `docs/figures/optimizer_benchmark.png`

Protocol fixed in advance (`configs/nas/surrogate_benchmark.json`); 5 independently fitted surrogates × 7 seeds per algorithm; Wilcoxon signed-rank with Holm correction; a ranking is claimed only if its direction holds under every surrogate.

- **Surrogate accuracy.** Leave-one-search-seed-out Spearman ≥ 0.94 for every objective overall; inside the reference box CLIP 0.61–0.69 (near the 0.71 reliability ceiling of the measured CLIP itself), fidelity 0.83–0.90, latency/GFLOPs/VRAM/bytes ≥ 0.94.
- **At the real budget (150 evaluations) no evolutionary algorithm beats random search**; random is ahead (e.g. 4 objectives: 0.267 vs 0.217–0.227 median archive HV). This matches the real searches.
- **At 3,360 evaluations with 6 objectives: LHFiD > NSGA-III > NSGA-II ≈ random** on archive HV (medians 0.258, 0.232, 0.213, 0.215), and **LHFiD > NSGA-III > NSGA-II** on final-population HV at both 4 and 6 objectives (6 objectives: 0.199, 0.139, 0.105). All claimed pairs are Holm-significant with a consistent sign across surrogates.
- **LHFiD's lead over NSGA-III grows with the number of objectives** (final-population HV: +0.022 at 4 objectives, +0.064 at 6), the regime its hf-dominance targets. LHFiD's two preference rules (p = 1/w and p = 1/M) are indistinguishable here.
- **Crossover.** The many-objective algorithms overtake random between about 336 and 840 evaluations; LHFiD leads from 336. A real search would need roughly 2–6× our budget (about 4–10 GPU-hours per search on this GPU instead of 1.8) to benefit, and a larger calibration set to make CLIP rankable.
- LHFiD's stabilization-timed nadir estimation fired in 134 of 140 runs (median first detection at generation 45–49).
- These conclusions are conditional on the surrogates; §5.5 reports real measurements of points each optimizer selected.

### 5.5 Surrogate-assisted LHFiD at the real budget (preliminary, simulation) — `scripts/sa_simulation.py`

The real-budget failure of every optimizer (§5.2, §5.4) motivates a surrogate-assisted search: spend 50 real evaluations on random configurations, then 5 rounds of (refit an internal surrogate on the noisy observations → run a many-objective optimizer for 1,680 virtual evaluations → evaluate 20 novel candidates, spread across the predicted front). Protocol fixed in `configs/nas/sa_simulation.json` before any run: the "truth" is a different model family (HistGradientBoosting or RandomForest) fitted on all 900 real evaluations; CLIP and fidelity observations carry Gaussian noise calibrated to the measured reliabilities (0.50 and 0.90); hypervolume is computed on the noise-free truth; 31 seeds per arm and truth; Mann–Whitney U with Holm correction.

| Arm (150 evaluations) | HV, HGB truth | HV, RF truth |
| --- | ---: | ---: |
| **SA-LHFiD** | **0.415** | **0.338** |
| SA-LHFiD, noise-aware ties | 0.395 | 0.329 |
| SA-MOEA/D (extension) | 0.374 | 0.320 |
| SA-RVEA (extension) | 0.369 | 0.314 |
| LHFiD | 0.333 | 0.263 |
| SA-NSGA-II | 0.303 | 0.257 |
| Random | 0.292 | 0.268 |
| NSGA-II | 0.260 | 0.208 |

- SA-LHFiD beats random (+26–42%), NSGA-II, LHFiD and SA-NSGA-II under both truths (Holm p ≤ 1.6 × 10⁻⁵), beats SA-RVEA under both truths (p ≤ 0.011), and beats SA-MOEA/D under one truth (p = 0.006; +0.017, n.s., under the other).
- Surrogate assistance only pays off with a many-objective-capable selector: SA-NSGA-II is no better than random, while SA-RVEA, SA-MOEA/D and SA-LHFiD all beat their direct versions; LHFiD is the best inner optimizer.
- The noise-aware tie rule does not help (ablation).
- RVEA and MOEA/D were added after the pre-registered comparisons, as labelled extensions on the same truths and seeds. On the surrogate benchmark (§5.4) they are close to LHFiD: LHFiD keeps the best 6-objective archive HV (+0.008, consistent across surrogates) but ties on final-population HV, so LHFiD's large lead is over Pareto-dominance methods (NSGA-II/III).
- This is a simulation calibrated to the real problem; the pre-registered real confirmation (C2, `configs/nas/c2_protocol.json`, 6 seeds per arm) is running.

### 5.6 Pending (cluster)

- KID/FID vs real COCO and vs the reference for all validation runs (the validation job's final scoring step failed on newer SciPy; fixed, rescoring).
- C1: pre-registered non-inferiority test on a fresh 500-prompt split (`configs/nas/c1_protocol.json`, margin 0.5 CLIP).
- C2: real SA-LHFiD vs SA-NSGA-II, random and NSGA-II, 6 seeds each.
- Real calibration measurements of 25 surrogate-selected points (surrogate error on optimizer-chosen points).
- Post-hoc BK-SDM at 10-step DDIM, fp16 and int8.

## 6. Conclusions so far

1. **Training-free compression of SD1.5 works mainly through int8 weights and sampler choice.** The best searched configuration halves U-Net bytes and peak VRAM by 0.83 GB at 1.58× speed with no measurable alignment loss; a 2.8× faster configuration keeps alignment at full size.
2. **Static unit removal is not worth it without retraining.** Individually safe units interact, and on top of int8 they save at most ~8% more bytes while always reducing fidelity to the original model.
3. **The search's value is in combining orthogonal knobs** (precision × steps × guidance × caching), not in finding a better sub-network.
4. **Optimizer choice only matters with enough evaluations.** At 150 real evaluations nothing beats random search; at thousands, LHFiD is the best of the four selection schemes, increasingly so with more objectives.
5. **The calibration quality signal is the bottleneck.** With 96 images, CLIP cannot rank good candidates (reliability 0.50); larger calibration sets matter more than the optimizer.

Limitations: one workstation GPU, not an edge device; CLIP and Inception-based metrics only (no human study); int8 latency reflects a simple dequantize-per-call implementation; BK-SDM is distilled from SD-v1.4, not SD1.5; surrogate-benchmark conclusions are conditional on surrogate accuracy.

## 6. Reproducibility

Every number in this report comes from a committed script and a result file under `artifacts/` (git-ignored because of size; copied from the cluster). The phase tracker and run log are in `docs/RESEARCH_PLAN.md`; protocol rules and known failures are in `CLAUDE.md`; design decisions are in the Notion decision log (D001–D021+).
