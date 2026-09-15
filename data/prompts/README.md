# Prompt files

Do not tune policies on the hold-out prompts. Store calibration and hold-out splits as separate JSON files:

```json
{
  "split": "calibration",
  "source": "MS-COCO captions",
  "prompts": [
    "A dog running through a field."
  ]
}
```

The planned protocol uses 48 calibration prompts with two fixed seeds and 128 untouched hold-out prompts with two different fixed seeds. Dataset files are not bundled in this repository.

