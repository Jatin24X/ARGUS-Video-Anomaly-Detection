# Avenue Dataset Audit for Stream A

## Audited source

`C:\Users\jatin\OneDrive\Desktop\SD-MAE\Avenue_Extracted\Avenue Dataset`

Audit date: `2026-04-15`

## What is present

Top-level folders/files:

- `training_videos`
- `testing_videos`
- `train/frames`
- `test/frames`
- `train/gradients2`
- `test/gradients2`
- `training_vol`
- `testing_vol`
- `Raw_Videos.zip.zip`

## Inventory summary

| Path | Files | Size |
|---|---:|---:|
| `training_videos` | 16 `.avi` | `0.143 GB` |
| `testing_videos` | 21 `.avi` | `0.150 GB` |
| `train/frames` | 15,328 `.png` | `4.797 GB` |
| `test/frames` | 15,324 `.png` | `4.845 GB` |
| `train/gradients2` | 15,328 `.png` | `1.638 GB` |
| `test/gradients2` | 15,324 `.png` | `1.707 GB` |
| `training_vol` | 16 `.mat` | `0.234 GB` |
| `testing_vol` | 21 `.mat` | `0.233 GB` |
| `Raw_Videos.zip.zip` | 1 `.zip` | `0.312 GB` |

Frame resolution check:

- `train/frames/01/0000.png` -> `640x360 RGB`
- `test/frames/01/0000.png` -> `640x360 RGB`

Frame counts match the expected Avenue split sizes:

- train videos: `16`
- test videos: `21`

## What each folder actually is

### Canonical raw inputs

- `training_videos`
- `testing_videos`

These are the original Avenue `.avi` videos and are valid demo/runtime assets.

### Canonical pre-extracted frames

- `train/frames`
- `test/frames`

These are already-decoded RGB frames and are the best source for Stream A feature extraction.

Reason:

- avoids a second video decode step
- preserves one folder per video
- matches the exact split counts
- easy to map to Stream A metadata

### SD-MAE-specific derived artifacts

Practical note for Stream A feature extraction:

- the current standalone feature extractor works from videos, not frame folders
- for the least-friction Avenue port, run feature extraction separately on:
  - `training_videos` with prefix `train`
  - `testing_videos` with prefix `test`

- `train/gradients2`
- `test/gradients2`

These are derived image caches and are not needed for `VideoMAE + MULDE` Stream A.

### Not labels: low-resolution video volumes

- `training_vol`
- `testing_vol`

These `.mat` files do **not** contain anomaly labels.

Example inspection:

- `testing_vol/vol01.mat` contains `vol` with shape `(120, 160, 1439)` and dtype `uint8`
- `training_vol/vol01.mat` contains `vol` with shape `(120, 160, 1364)` and dtype `uint8`

So they are low-resolution video volumes, not ground-truth anomaly annotations.

### Backup archive

- `Raw_Videos.zip.zip`

Keep only as backup. It is not needed if the extracted videos/folders remain intact.

## What Stream A actually needs

For Stream A on Avenue, the minimum useful inputs are:

1. visual input
   - preferably `train/frames` and `test/frames`
   - optionally `training_videos` and `testing_videos` for demo/runtime

2. metadata
   - `avenue_splits.json`
   - `avenue_scenes.json`
   - `avenue_frame_labels.json`

3. extracted VideoMAE features
   - `data/features/avenue/videomae/*.npy`

## Important naming issue: train/test ids collide

Avenue train and test both use short ids such as `01`, `02`, and `03`.

If Stream A saves features as plain `01.npy`, then training and test assets
overwrite each other and metadata keys become ambiguous.

So Avenue should use split-prefixed canonical names such as:

- `train_01`
- `train_02`
- `test_01`
- `test_02`

This affects:

- feature filenames
- split metadata
- scene metadata
- frame-label metadata

## What is still missing

The audited folder does **not** contain obvious benchmark label files.

Searches for names like:

- `label`
- `gt`
- `truth`
- `mask`
- `anno`

returned nothing useful under this folder.

That means the current folder is enough to:

- extract Stream A features
- train on normal training videos
- run inference

but it is **not enough to compute real Avenue AUC** yet.

## Important hidden blocker in current Stream A code

The standalone Stream A training loop currently assumes a labeled validation split with anomaly AUC:

- [scripts/train.py](c:/Users/jatin/OneDrive/Desktop/argus/argus%20stream%20A/scripts/train.py)
- [src/training/train_stream.py](c:/Users/jatin/OneDrive/Desktop/argus/argus%20stream%20A/src/training/train_stream.py)

Specifically:

- training loads `split="val"`
- validation is selected by `clip_val_AUC`

This works on UBnormal because UBnormal has a real anomalous validation split.

It does **not** transfer cleanly to Avenue because:

- Avenue has normal-only training videos
- there is no native anomalous validation split

So Avenue readiness is **not** just a metadata problem. We also need one of:

1. an Avenue-specific model-selection strategy on held-out normal data
2. a temporary benchmark-compromise flow that uses test for selection
3. a custom external validation subset with labels

Option 1 is the cleanest.

## Recommended canonical data choice for Stream A

Use:

- `train/frames`
- `test/frames`

Keep as supporting assets:

- `training_videos`
- `testing_videos`

Ignore for Stream A:

- `train/gradients2`
- `test/gradients2`
- `training_vol`
- `testing_vol`
- `Raw_Videos.zip.zip`

## Recommended next steps

1. Standardize Avenue around the frame folders, not the SD-MAE-specific caches.
2. Use split-prefixed canonical names such as `train_01` and `test_01`.
3. Build `avenue_scenes.json` with all videos mapped to scene `1`.
4. Obtain the real Avenue frame-level anomaly labels from the proper annotation source.
5. Add an Avenue-specific Stream A validation-selection path that does not require anomalous `val` videos during training.
6. Only then extract `VideoMAE` features and train the Avenue checkpoint.
