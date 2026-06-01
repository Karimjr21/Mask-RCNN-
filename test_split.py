import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import get_dataloaders


with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not Path(cfg["dataset"]["ann_file"]).is_file():
    print(f"SKIP: patched dataset not found at {cfg['dataset']['ann_file']}.")
    print("      Run:  python src/prepare_isaid.py")
    sys.exit(0)

_, val_loader_1 = get_dataloaders(cfg)
_, val_loader_2 = get_dataloaders(cfg)

indices_1 = list(val_loader_1.dataset.indices)
indices_2 = list(val_loader_2.dataset.indices)

assert indices_1 == indices_2
assert len(indices_1) > 0
assert Path(cfg["dataset"]["split_indices_path"]).is_file()

# No source image should appear in both train and val (no tile leakage).
train_ds = val_loader_1.dataset.dataset  # underlying IsaidCocoDataset
train_idx = set(list(get_dataloaders(cfg)[0].dataset.indices))
val_idx = set(indices_1)
train_sources = {train_ds.source_image(i) for i in train_idx}
val_sources = {train_ds.source_image(i) for i in val_idx}
assert train_sources.isdisjoint(val_sources), "source image leaked across split!"

print(f"Deterministic, leak-free split test passed. Val tiles: {len(indices_1)}")
