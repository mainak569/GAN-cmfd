# GAN-cmfd

Copy-Move Forgery Detection (CMFD) evaluation pipeline using a UNet-style generator (Pix2Pix-inspired), PatchGAN discriminator definition, CoMoFoD data preparation, and pixel/region-level segmentation metrics.

## Overview

This repository currently focuses on **dataset preparation** and **model evaluation** for copy-move forgery segmentation.

### Model Variations
- **Base 1**: Ablation with basic GAN
- **Base 2**: Ablation with only UNet
- **Final (Our model)**: UNet based patch GAN

Implemented components:
- Leak-free train/val/test split generation from CoMoFoD variants
- `torch.utils.data.Dataset` for paired forged image + binary mask loading
- UNet-style generator architecture (for mask prediction)
- PatchGAN discriminator architecture definition (not used in evaluation script)
- Pixel-level and region-level segmentation metrics
- JSON export of test metrics

## Repository Structure

```text
GAN-cmfd/
├── checkpoints/
│   ├── best_model.pth
│   └── G_epoch_*.pth
├── dataset.py
├── evaluate_comofod.py
├── metrics.py
├── models.py
├── prepare_data.py
├── .gitattributes
├── results/
│   └── comofod_test_results.json
└── README.md
```

## Requirements

Use Python 3.9+ (recommended), then install dependencies:

```bash
pip install torch torchvision pillow numpy scikit-learn scikit-image
```

## Checkpoints in GitHub (Git LFS)

This repository tracks model checkpoints with **Git LFS**.

- Tracked pattern: `checkpoints/*.pth`
- Config file: `.gitattributes`

When cloning, run:

```bash
git lfs install
git clone https://github.com/mainak569/GAN-cmfd.git
cd GAN-cmfd
git lfs pull
```

If Git LFS is not installed, checkpoint files may appear as small pointer text files instead of full binary weights.

## Dataset Assumptions (CoMoFoD)

The code assumes CoMoFoD files follow this naming convention:
- Forged image: `<base>_F.<ext>`
- Corresponding mask: `<base>_M.<ext>`

Supported extensions in current code:
- `.jpg`, `.png`, `.tif`

Only files with an existing matching mask are included.

## Data Preparation

Script: `prepare_data.py`

What it does:
1. Reads CoMoFoD source folder
2. Groups samples by base image ID (text before `_F`)
3. Splits base IDs into train/val/test with ratios:
   - Train: `0.7`
   - Val: `0.15`
   - Test: `0.15`
4. Copies image/mask pairs into:
   - `./data/train/images`, `./data/train/masks`
   - `./data/val/images`, `./data/val/masks`
   - `./data/test/images`, `./data/test/masks`

Important:
- The split is base-ID grouped, reducing data leakage between splits.
- Existing `./data` is deleted and recreated on each run.
- `random.seed(42)` is set for deterministic split order.

Before running, update this line in `prepare_data.py` to your local CoMoFoD path:

```python
COMOFOD_PATH = "/home/23ucc569/DataSources/CoMoFoD_small_v2"
```

Run:

```bash
python prepare_data.py
```

## Dataset Loader

File: `dataset.py` (`CMFDDataset`)

Behavior:
- Loads RGB forged images and grayscale masks
- Resizes both to `size x size` (default `256x256`)
- Converts to tensors
- Binarizes masks with threshold `> 0.5`
- Returns `(image, mask)`

Expected folder pair input:
- `image_dir`: contains forged files (`*_F.*`)
- `mask_dir`: contains matching mask files (`*_M.*`)

## Models

File: `models.py`

### Generator
- UNet-like encoder-decoder
- Output: single-channel logits (`B x 1 x H x W`)
- Uses skip connections and transposed convolutions

### Discriminator
- PatchGAN-style discriminator
- Input: concatenated image (3ch) + mask (1ch) = 4 channels
- Output: patch realism map

### Weight Initialization
- Conv/ConvTranspose: normal mean `0.0`, std `0.02`
- BatchNorm: weight normal mean `1.0`, std `0.02`, bias `0`

## Evaluation

Script: `evaluate_comofod.py`

Evaluation flow:
1. Loads test split from `./data/test/images` and `./data/test/masks`
2. Loads generator checkpoint from:
   - `checkpoints/best_model.pth`
3. Runs inference
4. Applies sigmoid + threshold `0.5` to obtain binary masks
5. Computes dataset-level metrics via `evaluate_segmentation(...)`
6. Prints and saves results to:
   - `results/comofod_test_results.json`

Run:

```bash
python evaluate_comofod.py
```

## Metrics

File: `metrics.py`

### Pixel-level metrics
- Precision
- Recall
- F1
- Accuracy
- IoU (Jaccard)

### Region-level metric
- `Region_mIoU`
  - Connected components are extracted for GT and prediction
  - Each GT region is greedily matched to one predicted region by best IoU
  - Mean IoU is averaged across GT regions per image
  - Dataset value is mean across images

### Dataset-level behavior
`evaluate_segmentation(gt_masks, pred_masks)` computes:
- Pixel metrics globally (micro-style over all flattened pixels)
- Region mIoU as average of per-image `Region_mIoU`

## Current Test Results

From `results/comofod_test_results.json`:

- Pixel Precision: `0.9048`
- Pixel Recall: `0.7792`
- Pixel F1: `0.8373`
- Pixel Accuracy: `0.9821`
- Pixel IoU: `0.7201`
- Region mIoU: `0.6624`

## End-to-End Quick Start

1. Install dependencies
2. Ensure Git LFS files are fetched (checkpoint weights):
   ```bash
   git lfs pull
   ```
3. Set CoMoFoD path in `prepare_data.py`
4. Generate splits:
   ```bash
   python prepare_data.py
   ```
5. Run test evaluation:
   ```bash
   python evaluate_comofod.py
   ```
6. Check output JSON in `results/`

## Known Limitations

- No training script is included in this repository yet.
- `prepare_data.py` currently uses a hardcoded absolute CoMoFoD path.
- Thresholds, resize size, and paths are hardcoded in scripts.
- `.DS_Store` is currently tracked in git and should ideally be ignored.

## Suggested Next Improvements

- Add `train.py` and checkpointing workflow
- Move all hardcoded config values to CLI args or YAML
- Add a `requirements.txt` and/or environment file
- Add a `.gitignore` (`.DS_Store`, `data/`, etc.)
- Add sample visualization script for predictions vs. masks

## Citation / Dataset

If you use this project, please cite the CoMoFoD dataset and any model/training methodology you build on top of this codebase.
