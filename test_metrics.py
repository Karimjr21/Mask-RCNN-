import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from metrics import DetectionMetric


def box_mask(x1, y1, x2, y2, size=16):
    mask = torch.zeros((size, size), dtype=torch.bool)
    mask[y1:y2, x1:x2] = True
    return mask


metric = DetectionMetric(iou_threshold=0.5, mask_threshold=0.5)

output = {
    "scores": torch.tensor([0.95, 0.90, 0.80]),
    "labels": torch.tensor([1, 1, 2]),
    "masks": torch.stack([
        box_mask(1, 1, 8, 8).float(),
        box_mask(10, 10, 14, 14).float(),
        box_mask(1, 1, 8, 8).float(),
    ]).unsqueeze(1),
}
target = {
    "labels": torch.tensor([1, 1]),
    "masks": torch.stack([
        box_mask(1, 1, 8, 8),
        box_mask(8, 8, 14, 14),
    ]),
}

metric.update(output, target)
result = metric.compute()

assert result["tp"] == 1, result
assert result["fp"] == 2, result
assert result["fn"] == 1, result
assert result["per_class"][0]["tp"] == 1, result
assert result["per_class"][0]["fp"] == 1, result
assert result["per_class"][0]["fn"] == 1, result
assert result["per_class"][1]["fp"] == 1, result
assert result["per_class"][1]["tp"] == 0, result

print("Metric matching test passed.")
