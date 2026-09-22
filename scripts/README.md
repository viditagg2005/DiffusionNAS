# Cluster experiment scripts

These scripts assume one NVIDIA A6000-class GPU, Stable Diffusion 1.5 at 512×512, and a repository-local `.venv`.

## Files

- `setup_cluster.sh`: initialize submodules, create the environment, install CUDA PyTorch and project dependencies, then run CPU tests/preflight.
- `prepare_experiment.py`: inject versioned prompt splits and seeds into matched benchmark and search configs.
- `slurm/benchmark_array.sbatch`: sequential baseline, AutoDiffusion-adaptation, and DeepCache runs.
- `slurm/score_calibration.sbatch`: frozen CLIP-L/14 quality scoring.
- `slurm/build_pareto.sbatch`: CPU-only calibration Pareto construction.
- `slurm/holdout.sbatch`: untouched prompt validation of calibration-front policies.
- `slurm/random_search.sbatch` and `slurm/ecad_search.sbatch`: equal-budget search experiments.

The SLURM files request `--gres=gpu:1`. Select the A6000 partition/GRES in the `sbatch` command because cluster resource names are site-specific. The benchmark array is throttled with `%1`, avoiding simultaneous model downloads and reducing timing interference.

`prepare_experiment.py` deliberately sets calibration benchmark repetitions to one. With 48 prompts and two seeds, each policy already has 96 timed samples; repetitions apply to every prompt/seed pair.

