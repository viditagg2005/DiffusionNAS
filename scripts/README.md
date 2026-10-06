# Cluster experiment scripts

These scripts assume one 48 GB NVIDIA GPU, Stable Diffusion 1.5 at 512×512, and a repository-local `.venv`.

## Files

- `setup_cluster.sh`: initialize submodules, create the environment, install CUDA PyTorch and project dependencies, then run CPU tests/preflight.
- `build_coco_prompt_splits.py`: select deterministic, image-disjoint COCO calibration and hold-out captions.
- `prepare_experiment.py`: inject versioned prompt splits and seeds into matched benchmark and search configs.
- `slurm/benchmark_array.sbatch`: baseline, AutoDiffusion-adaptation, and DeepCache sequentially on one GPU allocation. The filename is retained for compatibility; it is no longer an array job.
- `slurm/score_calibration.sbatch`: frozen CLIP-L/14 quality scoring.
- `slurm/build_pareto.sbatch`: CPU-only calibration Pareto construction.
- `slurm/holdout.sbatch`: untouched prompt validation of calibration-front policies.
- `slurm/random_search.sbatch` and `slurm/ecad_search.sbatch`: equal-budget search experiments.

GPU jobs request `--gres=gpu:1`. The benchmark keeps all three methods on the same allocated GPU so latency comparisons cannot mix GPU models. The observed cluster allocation was an RTX 6000 Ada (48 GB), not an RTX A6000. CLIP scoring defaults to batch size 8; set `CLIP_BATCH_SIZE` to override it.

Submit jobs from the repository root. SLURM copies batch scripts into a spool directory, so the jobs resolve the checkout using `SLURM_SUBMIT_DIR`, not the batch file's runtime path. When submitting from another directory, pass the checkout explicitly:

```bash
sbatch --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/benchmark_array.sbatch
```

`prepare_experiment.py` deliberately sets calibration benchmark repetitions to one. With 48 prompts and two seeds, each policy already has 96 timed samples; repetitions apply to every prompt/seed pair.

`sbatch` submits jobs; `squeue` only displays queued/running jobs. Submit from the checkout on the cluster. To queue the calibration pipeline in order, use `--dependency=afterok` so later stages do not run after a failure:

```bash
benchmark_job=$(sbatch --parsable --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/benchmark_array.sbatch)
score_job=$(sbatch --parsable --dependency="afterok:${benchmark_job}" --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/score_calibration.sbatch)
pareto_job=$(sbatch --parsable --dependency="afterok:${score_job}" --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/build_pareto.sbatch)
holdout_job=$(sbatch --parsable --dependency="afterok:${pareto_job}" --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/holdout.sbatch)
echo "benchmark=$benchmark_job score=$score_job pareto=$pareto_job holdout=$holdout_job"
squeue -j "$benchmark_job,$score_job,$pareto_job,$holdout_job"
```

Each job writes `slurm-JOB_ID.out` in the submission directory. After jobs leave `squeue`, inspect their final states with `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList`. A dependency-held job with state `DependencyNeverSatisfied` means an upstream stage failed.

## Search jobs on the 24-hour partition

Both search scripts request **one GPU for 23h50m**, below the partition's 24-hour limit. Each search has 100 candidates, 48 calibration prompts and two seeds. The September calibration run on the same RTX 6000 Ada measured approximately 129 seconds for a 20-step baseline candidate (including initial model load), 30 seconds for the four-step AutoDiffusion adaptation, and 39 seconds for the DeepCache candidate, before CLIP scoring. Even a conservative 130 seconds of generation plus 60 seconds of scoring per candidate projects about **5h17m per 100-candidate search**, plus overhead. This is an estimate, not a guaranteed completion time; unusual candidates, model downloads, or cluster contention may extend it.

The scripts stop starting new candidates after **23 hours** (`SEARCH_MAX_WALLTIME_SECONDS=82800`), leaving 50 minutes before Slurm's hard limit. After every fully scored candidate, `artifacts/search/{random,ecad}/checkpoint.json` is atomically updated with its chromosome, objectives, run path, and elapsed time. If a job stops during a candidate, at most that candidate's generation/scoring must be repeated. Re-submit the **same** script against the **same** output directory to resume. The search configuration, CLIP settings, and GPU identity must remain unchanged. Do not run two jobs for the same search/output directory concurrently.

A completed search exits 0 and writes `search_results.json` and `pareto_front.json`. A soft-time-limit pause exits 75, leaves `checkpoint.json` with `"status": "paused"`, and requires another submission. Check progress with:

```bash
python -c 'import json; from pathlib import Path; [(lambda p, d: print(p, d["status"], len(d["evaluations"]), "/", d["budget"]))(p, json.loads(p.read_text())) for p in (Path("artifacts/search/random/checkpoint.json"), Path("artifacts/search/ecad/checkpoint.json")) if p.exists()]'
```

Previously submitted jobs retain their old 47h30m Slurm request and will not pick up the new `#SBATCH` directives. Once the updated checkout is on the cluster, cancel pending jobs **8695 and 8694 only if they are still yours and pending**, then submit random search afresh. Submit ECAD when random has succeeded; a paused random job exits 75, so an `afterok`-dependent ECAD job will not run automatically.

```bash
squeue -j 8694,8695
scancel 8695 8694
cd /home2/home/dhish_saxena/DiffusionNAS
git pull --ff-only
random_job=$(sbatch --parsable --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/random_search.sbatch)
echo "random_job=$random_job"
squeue -j "$random_job"
```

After random search exits, check `sacct -j "$random_job" --format=JobID,State,ExitCode,Elapsed` and the checkpoint status. If paused, re-submit `random_search.sbatch` using the same `sbatch` command. When it says `succeeded`, submit ECAD (and re-submit it if it pauses):

```bash
ecad_job=$(sbatch --parsable --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/ecad_search.sbatch)
echo "ecad_job=$ecad_job"
squeue -j "$ecad_job"
```
