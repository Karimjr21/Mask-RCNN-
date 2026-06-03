"""Architecture smoke test: build the model from configs/config.yaml and run a
full forward pass in both train (loss) and eval (inference) modes.

This catches shape/architecture mistakes from the model knobs (arch v1/v2,
anchor aspect-ratio count + RPN-head rebuild, min_size) *locally*, before a
Kaggle GPU session is spent on them. Uses pretrained=False so it needs no
network download.

    python test_model_build.py
"""
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import get_model

with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
mc = cfg["model"]

model = get_model(
    mc["num_classes"],
    pretrained=False,
    trainable_backbone_layers=int(mc.get("trainable_backbone_layers", 3)),
    anchor_sizes=mc.get("anchor_sizes"),
    anchor_aspect_ratios=mc.get("anchor_aspect_ratios"),
    detections_per_img=int(mc.get("detections_per_img", 100)),
    arch=mc.get("arch", "v1"),
    min_size=int(mc.get("min_size", 800)),
    max_size=int(mc.get("max_size", 1333)),
)

anchors_per_loc = model.rpn.anchor_generator.num_anchors_per_location()
print(f"arch              : {mc.get('arch', 'v1')}")
print(f"trainable layers  : {mc.get('trainable_backbone_layers')}")
print(f"anchor sizes      : {mc.get('anchor_sizes')}")
print(f"aspect ratios     : {mc.get('anchor_aspect_ratios')}")
print(f"anchors / location: {anchors_per_loc}")
print(f"min/max size      : {mc.get('min_size')}/{mc.get('max_size')}")

# Shrink the input transform for the forward pass ONLY. min_size/max_size change
# activation sizes, not parameter shapes — so a small input fully validates the
# architecture (incl. the rebuilt RPN head <-> anchor-count wiring) while keeping
# the CPU forward light enough to run on a laptop.
model.transform.min_size = (256,)
model.transform.max_size = 512

H = W = 320
imgs = [torch.rand(3, H, W), torch.rand(3, H, W)]
targets = []
for _ in range(2):
    masks = torch.zeros(2, H, W, dtype=torch.uint8)
    masks[0, 10:60, 10:50] = 1
    masks[1, 30:90, 20:80] = 1
    targets.append({
        "boxes": torch.tensor([[10, 10, 50, 60], [20, 30, 80, 90]], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "masks": masks,
    })

model.train()
losses = model(imgs, targets)
print("\ntrain-mode losses :", {k: round(float(v), 4) for k, v in losses.items()})
assert all(torch.isfinite(v) for v in losses.values()), "non-finite loss!"

model.eval()
with torch.no_grad():
    out = model([torch.rand(3, H, W)])
print("eval-mode output  :", list(out[0].keys()), "| detections:", len(out[0]["scores"]))
assert {"boxes", "labels", "scores", "masks"} <= set(out[0].keys())

print("\nMODEL BUILD TEST PASSED — architecture is valid in train + eval modes.")
