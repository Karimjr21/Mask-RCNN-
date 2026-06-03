import yaml
import torch
from tqdm import tqdm

from dataset import get_dataloaders
from metrics import DetectionMetric
from model import get_model
from utils import CLASS_NAMES, load_state_dict, select_device


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def quality_label(value: float) -> str:
    if value >= 0.85:
        return "excellent"
    if value >= 0.70:
        return "good"
    if value >= 0.50:
        return "fair"
    if value >= 0.30:
        return "low"
    return "very low"


def quality_percent(value: float) -> str:
    return f"{format_percent(value)} ({quality_label(value)})"


def evaluate(
    model,
    loader,
    device,
    score_threshold=0.3,
    iou_threshold=0.5,
    mask_threshold=0.5,
):
    model.eval()
    non_blocking = device.type == "cuda"
    metric = DetectionMetric(
        iou_threshold=iou_threshold,
        mask_threshold=mask_threshold,
    )

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = [img.to(device, non_blocking=non_blocking) for img in images]
            targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                keep = output["scores"] >= score_threshold
                metric.update(
                    {
                        "scores": output["scores"][keep],
                        "labels": output["labels"][keep],
                        "masks": output["masks"][keep],
                    },
                    target,
                )

    metrics = metric.compute()
    print("\n" + "=" * 60)
    print("  Validation quality")
    print(f"  Detection quality       : {quality_percent(metrics['map50'])}")
    print(f"  Correct detections      : {quality_percent(metrics['precision'])}")
    print(f"  Objects found           : {quality_percent(metrics['recall'])}")
    print(f"  Overall detection score : {quality_percent(metrics['f1'])}")
    print(f"  Mask overlap quality    : {quality_percent(metrics['mask_iou'])}")
    print(
        "  Object counts           : "
        f"{metrics['tp']} correct, {metrics['fp']} false alarms, {metrics['fn']} missed"
    )
    print("-" * 60)
    print("  Per-class quality:")
    for cid in sorted(CLASS_NAMES):
        values = metrics["per_class"].get(cid)
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        if not values or values["gt"] == 0:
            print(f"    [{cid}] {name:<14} no validation examples")
            continue
        print(
            f"    [{cid}] {name:<14} "
            f"quality={format_percent(values['ap50'] or 0.0)} "
            f"correct={format_percent(values['precision'])} "
            f"found={format_percent(values['recall'])} "
            f"score={format_percent(values['f1'])} "
            f"mask_overlap={format_percent(values['mask_iou'])} "
            f"ground_truth={values['gt']}"
        )
    print("=" * 60 + "\n")
    return metrics


if __name__ == "__main__":
    import sys

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = select_device()
    checkpoint = (
        sys.argv[1] if len(sys.argv) > 1
        else "outputs/checkpoints/model_best.pth"
    )

    # Always evaluate on the FULL validation set here, regardless of the
    # training-time `val_fraction` subsample (which exists only to keep
    # per-epoch monitoring fast). The reported number must be honest.
    cfg.setdefault("training", {})["val_fraction"] = 1.0

    _, val_loader = get_dataloaders(cfg)
    mc = cfg["model"]
    model = get_model(
        mc["num_classes"], pretrained=False,
        trainable_backbone_layers=int(mc.get("trainable_backbone_layers", 3)),
        anchor_sizes=mc.get("anchor_sizes"),
        anchor_aspect_ratios=mc.get("anchor_aspect_ratios"),
        detections_per_img=int(mc.get("detections_per_img", 100)),
        arch=mc.get("arch", "v1"),
        min_size=int(mc.get("min_size", 800)),
        max_size=int(mc.get("max_size", 1333)),
    )
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.to(device)

    evaluate(
        model,
        val_loader,
        device,
        score_threshold=cfg["inference"]["score_threshold"],
        iou_threshold=cfg["inference"].get("metric_iou_threshold", 0.5),
        mask_threshold=cfg["inference"].get("mask_threshold", 0.5),
    )
