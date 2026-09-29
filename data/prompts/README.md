# Prompt files

The tracked calibration.json and holdout.json contain disjoint captions from MS-COCO 2014 validation images. Do not use hold-out prompts or scores for optimization or policy selection.

Each prompt includes its source image ID, annotation ID, and caption-complexity stratum. There are 48 calibration prompts (16 per stratum) and 128 hold-out prompts (42 simple, 43 relational, 43 detailed). The selection seed and source SHA-256 are stored in both files. Whitespace is normalized; other wording is unchanged. Strata are text heuristics, not COCO object labels.

To regenerate the files from the pinned COCO caption annotation:
https://raw.githubusercontent.com/tylin/coco-caption/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/annotations/captions_val2014.json

~~~bash
python scripts/build_coco_prompt_splits.py --annotations /path/to/captions_val2014.json
python scripts/prepare_experiment.py --calibration-prompts data/prompts/calibration.json --calibration-seeds 42,123
~~~

Use calibration seeds 42 and 123, and hold-out seeds 101 and 202. The generated benchmark and search configs are tracked under configs/generated.

Captions are attributed to the COCO Consortium under CC BY 4.0:
https://cocodataset.org/#termsofuse

The caption annotation file comes from the COCO caption evaluation authors' repository. No COCO images are bundled.
