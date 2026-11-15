# Literature check for novelty claims (2026-11-15)

Searches covered surrogate-assisted LHFiD / hf-dominance, surrogate-assisted many-objective EAs, surrogate-assisted evolutionary NAS for diffusion models, and benchmarks for multi-objective diffusion inference configuration. Search engines and publisher pages only; Google Scholar citation lists of LHFiD were not checked and should be before submission.

## Findings

| Work | What it does | Relation to our papers |
| --- | --- | --- |
| Saxena, Mittal, Kapoor, Deb, "A Localized High-Fidelity-Dominance-Based Many-Objective Evolutionary Algorithm", IEEE TEVC 27(4):923–937, 2023, doi:10.1109/TEVC.2022.3188064 | LHFiD | Cite the journal version (not only COIN 2021020). No surrogate-assisted LHFiD found. |
| Saxena, Mittal, Deb, Goodman, *Machine Learning Assisted Evolutionary Multi- and Many-Objective Optimization*, Springer, 2024 | Book on ML-assisted EMO by the LHFiD authors | **Check its contents before claiming priority**: it may contain surrogate- or ML-assisted variants of LHFiD-style algorithms. Ask the supervisor. |
| Santoshkumar, Deb, "Addressing Heterogeneous Evaluation Times in Constrained Multi-Objective Optimization using a Mixed-Fidelity Evaluation Technique", COIN Report 2025001 | Surrogate-assisted EMO that decides which objectives/constraints to evaluate at high fidelity when evaluation times differ by orders of magnitude | Directly relevant: our cost objectives (bytes, GFLOPs, latency) are cheap and nearly exact, quality is expensive and noisy. Cite as related work; mixed-fidelity evaluation is a natural extension of SA-LHFiD. |
| Chugh et al., K-RVEA, IEEE TEVC 2018; Pan et al., CSEA, IEEE TEVC 2019; Yuan and Banzhaf, dominance prediction, IEEE TEVC 2022; Knowles, ParEGO, IEEE TEVC 2006 | Standard surrogate-assisted (many-)objective EAs | Baselines/related work for paper B. K-RVEA is the closest (reference-vector EA + Kriging); a SA-RVEA arm is already in our simulation extension. |
| Song (advisor Ahn), *Surrogate-Assisted Evolutionary Optimization for Diffusion Models: Perspectives on Personalization and Distillation*, Ph.D. thesis, GIST (DELTA-Diff) | Surrogate-guided evolutionary selection of multi-stage **distillation** paths for Stable Diffusion U-Net compression | Closest prior work combining surrogate-assisted evolution and SD compression. Ours differs: training-free (no distillation), many-objective with measured hardware objectives, pre-registered real-hardware confirmation. Must be cited. |
| ECAD (ICLR 2026, arXiv 2506.15682) | Genetic-algorithm search over per-layer caching schedules, quality–latency Pareto front, 100 calibration prompts, PixArt/FLUX | Closest multi-objective diffusion-inference search; bi-objective, caching only, no architecture or precision genes, not surrogate-assisted. Already cited. |
| AutoDiffusion (ICCV 2023), Flexiffusion (arXiv 2409.17566), LDMOES, DeepCache, BK-SDM | Training-free step/architecture search; segment-wise step schedules; MOEA + distillation; caching; distilled compact SD | Already cited and/or reproduced as baselines. |
| EDNAG (arXiv 2504.17827), EmoDM (arXiv 2401.15931) | Diffusion models used inside NAS / MOO | Adjacent direction (diffusion as optimizer), not our problem. |

No benchmark was found that jointly scores quality, latency, memory and size across diffusion inference configurations on real hardware (DiffBench evaluates acceleration code generation, not configuration optimization).

## Wording to use

- Paper B: "To our knowledge, LHFiD has not previously been used inside a surrogate-assisted loop, nor applied to training-free diffusion model compression." Keep "to our knowledge" until the Springer book and LHFiD's citing papers are checked.
- Paper A: "We are not aware of a benchmark that measures quality, latency, memory and model size of diffusion inference configurations on real hardware."
- Position against DELTA-Diff (distillation-based, surrogate-assisted) and ECAD (bi-objective caching search) explicitly.

## Sources

- https://ieeexplore.ieee.org/document/9814856/ ; https://dblp.uni-trier.de/rec/journals/tec/SaxenaMKD23.html ; https://peeref.com/works/28402156
- https://www.egr.msu.edu/~kdeb/papers/c2025001.pdf
- https://scholar.gist.ac.kr/handle/local/34605
- https://arxiv.org/html/2506.15682v3 ; https://proceedings.iclr.cc/paper_files/paper/2026/file/e68274fc4f158dbcbd4dddc672f7ee9c-Paper-Conference.pdf
- https://arxiv.org/html/2409.17566v2 ; https://openaccess.thecvf.com/content/ICCV2023/papers/Li_AutoDiffusion_Training-Free_Optimization_of_Time_Steps_and_Architectures_for_Automated_ICCV_2023_paper.pdf
- https://arxiv.org/abs/2504.17827 ; https://arxiv.org/html/2401.15931
- https://www.sciencedirect.com/science/article/abs/pii/S0020025519305560 ; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10423721/
