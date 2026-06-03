"""Throwaway smoke test: load last.pth and run the (edited) validation path.

Confirms the validation code runs end-to-end with the new loader settings and
reports peak GPU memory so we can see the 4 GB headroom. Does NOT reproduce the
train->val handoff pressure (a real resume would) -- it starts from a clean
CUDA cache, so treat the peak below as "validation alone".
"""
import os
# Opt-in only: set SMOKE_ALLOC to test an allocator config; default is torch's.
if os.environ.get("SMOKE_ALLOC"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["SMOKE_ALLOC"])

import sys
import itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import yaml
import torch

from model import get_model
from dataset import get_dataloaders
from utils import select_device
from train import validate_detection, load_training_state

MAX_BATCHES = int(os.environ.get("SMOKE_MAX_BATCHES", "60"))
COMPUTE_LOSS = os.environ.get("SMOKE_COMPUTE_LOSS", "0") == "1"


def run():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = select_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        print(f"  alloc conf: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(torch default)')}")

    mc = cfg["model"]
    eval_score_threshold = float(cfg["inference"]["score_threshold"])
    box_score_thresh = max(0.0, eval_score_threshold - 1e-4)
    model = get_model(
        mc["num_classes"],
        pretrained=False,
        trainable_backbone_layers=int(mc.get("trainable_backbone_layers", 3)),
        anchor_sizes=mc.get("anchor_sizes"),
        anchor_aspect_ratios=mc.get("anchor_aspect_ratios"),
        detections_per_img=int(mc.get("detections_per_img", 100)),
        box_score_thresh=box_score_thresh,
        arch=mc.get("arch", "v1"),
        min_size=int(mc.get("min_size", 800)),
        max_size=int(mc.get("max_size", 1333)),
    )
    state = load_training_state("outputs/checkpoints/last.pth", device)
    model.load_state_dict(state["model"])
    model.to(device)
    print(f"  Loaded last.pth (epoch {state['epoch']})")

    train_loader, val_loader = get_dataloaders(cfg)
    print(f"  val tiles after val_fraction: {len(val_loader.dataset)} "
          f"(loader batches: {len(val_loader)})")
    print(f"  val loader pin_memory={val_loader.pin_memory}")
    print(f"  Running validation on first {MAX_BATCHES} tiles "
          f"(compute_loss={COMPUTE_LOSS})...\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    capped = itertools.islice(val_loader, MAX_BATCHES)
    val_loss, metrics = validate_detection(
        model, capped, device,
        score_threshold=eval_score_threshold,
        amp_enabled=False,
        metric_iou_threshold=cfg["inference"].get("metric_iou_threshold", 0.5),
        mask_threshold=cfg["inference"].get("mask_threshold", 0.5),
        compute_loss=COMPUTE_LOSS,
        eval_fp16=bool(cfg["inference"].get("val_eval_fp16", False)) and device.type == "cuda",
    )

    print("\n=== RESULT ===")
    print(f"  val_loss      : {val_loss}")
    print(f"  map50         : {metrics.get('map50')}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  peak allocated: {peak:.2f} GB")
        print(f"  peak reserved : {reserved:.2f} GB  of {total:.2f} GB total")
    print("\nSMOKE TEST PASSED: validation ran to completion without crashing.")


if __name__ == "__main__":
    run()
