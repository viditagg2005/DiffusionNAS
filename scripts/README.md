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

