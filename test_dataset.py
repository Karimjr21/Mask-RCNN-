import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import IsaidCocoDataset

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dc = cfg["dataset"]
if not Path(dc["ann_file"]).is_file():
    print(f"SKIP: patched dataset not found at {dc['ann_file']}.")
    print("      Run:  python src/prepare_isaid.py")
    sys.exit(0)

ds = IsaidCocoDataset(
    dc["images_dir"], dc["ann_file"],
    min_instance_area=dc.get("min_instance_area", 1),
)
print(f"Total tiles: {len(ds)}")

img, target = ds[0]
print("Image shape:   ", tuple(img.shape))
print("Labels:        ", target["labels"].tolist())
print("Num instances: ", len(target["boxes"]))
print("Mask shape:    ", tuple(target["masks"].shape))
assert img.shape[0] == 3
assert target["masks"].shape[0] == target["boxes"].shape[0] == target["labels"].shape[0]
print("Dataset test passed.")
