# Satellite Mask R-CNN — iSAID Instance Segmentation

Mask R-CNN (ResNet-50 FPN, torchvision) for **instance segmentation** on the
**iSAID** aerial dataset (15 object categories: planes, ships, vehicles,
storage tanks, harbors, courts, etc.). Annotations are COCO-format polygons.

## Pipeline overview

```
data/annotations/iSAID_train.json   COCO annotations (provided)
data/images/                        REAL satellite images: P0000.png, P0001.png, ...
        │
        │  src/prepare_isaid.py      tile huge images into 800x800 patches
        ▼
data/patches/images/                tile PNGs
data/patches/annotations.json       patched COCO json
        │
        │  src/train.py             train Mask R-CNN
        ▼
outputs/checkpoints/model_best.pth
        │
        ├─ src/evaluate.py          precision / recall / mAP@50 / mask IoU
        └─ src/predict.py           tiled inference on full-size images
```

## Required: the real images

`data/images` must contain the **actual RGB satellite images** named exactly as
the JSON's `file_name` (`P0000.png`, `P0001.png`, ...). These come from the
DOTA/iSAID image download — they are *not* the `*_instance_id_RGB.png` mask
files. `prepare_isaid.py` reports how many source images are missing.

## Steps

1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify GPU (optional):
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```
3. Tile the dataset (one-time preprocessing; writes to `data/patches/`):
   ```bash
   python src/prepare_isaid.py
   # quick smoke test on 20 images first:
   python src/prepare_isaid.py --limit 20
   ```
4. Train:
   ```bash
   python src/train.py
   ```
5. Evaluate a checkpoint:
   ```bash
   python src/evaluate.py outputs/checkpoints/model_best.pth
   ```
6. Predict on a full image (tiled + stitched automatically):
   ```bash
   python src/predict.py --input data/images/P0000.png --save-overlay --save-json
   ```

## Configuration

All paths and hyper-parameters live in [configs/config.yaml](configs/config.yaml):
tile size/overlap, `min_instance_area`, train/val split (split at the
*source-image* level to prevent tile leakage), learning rate, AMP, early
stopping, and inference thresholds. The model uses `num_classes: 16`
(15 foreground + background).

## GPU/CUDA behavior

`train.py`, `evaluate.py`, and `predict.py` auto-select CUDA when available and
fall back to CPU otherwise, printing the chosen device at startup. Mixed
precision (AMP) is enabled via `training.amp` in the config.
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```
If `torch.cuda.is_available()` is `False`, install a CUDA-enabled PyTorch build
for your NVIDIA driver.
