import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.ops import nms

from model import get_model
from utils import CLASS_NAMES, load_state_dict, overlay_masks, select_device


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_model(cfg, checkpoint_path, device):
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
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model.to(device)
    model.eval()
    return model


def resolve_checkpoint(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_file():
        return checkpoint

    # Fresh runs create model_best.pth, but older runs may only have final/epoch files.
    if checkpoint.name == "model_best.pth":
        checkpoint_dir = checkpoint.parent
        final_checkpoint = checkpoint_dir / "model_final.pth"
        if final_checkpoint.is_file():
            print(f"Best checkpoint not found; using: {final_checkpoint}")
            return final_checkpoint

        epoch_checkpoints = sorted(
            checkpoint_dir.glob("model_epoch_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if epoch_checkpoints:
            print(f"Best checkpoint not found; using latest epoch checkpoint: {epoch_checkpoints[0]}")
            return epoch_checkpoints[0]

    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")


def list_image_paths(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]

    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No supported images found in: {path}")
        return images

    raise FileNotFoundError(f"Input path does not exist: {path}")


def get_default_image_path(images_dir):
    images = list_image_paths(images_dir)
    return images[0]


def tile_slices(height, width, tile_size, overlap):
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return [(0, height, 0, width)]

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("--tile-overlap must be >= 0 and smaller than --tile-size")

    stride = tile_size - overlap

    def starts(length):
        values = list(range(0, max(length - tile_size, 0) + 1, stride))
        last = max(length - tile_size, 0)
        if not values or values[-1] != last:
            values.append(last)
        return values

    slices = []
    for y0 in starts(height):
        for x0 in starts(width):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            slices.append((y0, y1, x0, x1))
    return slices


def run_model_on_array(model, image_array, device):
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image_array))
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .unsqueeze(0)
        .to(device, non_blocking=device.type == "cuda")
    )
    with torch.no_grad():
        return model(tensor)[0]


def apply_class_aware_nms(boxes, scores, labels, nms_threshold):
    if len(scores) == 0:
        return np.array([], dtype=np.int64)

    keep_indices = []
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    labels_np = np.asarray(labels)

    for label in sorted(set(labels_np.tolist())):
        label_indices = np.where(labels_np == label)[0]
        kept = nms(boxes_t[label_indices], scores_t[label_indices], nms_threshold)
        keep_indices.extend(label_indices[kept.cpu().numpy()].tolist())

    keep_indices.sort(key=lambda i: float(scores[i]), reverse=True)
    return np.asarray(keep_indices, dtype=np.int64)


def predict_image(
    model,
    image_path,
    device,
    score_threshold=0.3,
    nms_threshold=0.3,
    tile_size=1024,
    tile_overlap=128,
):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    boxes, scores, labels, masks = [], [], [], []
    for y0, y1, x0, x1 in tile_slices(height, width, tile_size, tile_overlap):
        tile = image_array[y0:y1, x0:x1]
        output = run_model_on_array(model, tile, device)
        keep = output["scores"] >= score_threshold

        tile_boxes = output["boxes"][keep].detach().cpu().numpy()
        tile_scores = output["scores"][keep].detach().cpu().numpy()
        tile_labels = output["labels"][keep].detach().cpu().numpy()
        tile_masks = output["masks"][keep].detach().cpu().squeeze(1).numpy()

        for box, score, label, mask in zip(tile_boxes, tile_scores, tile_labels, tile_masks):
            global_box = box.copy()
            global_box[[0, 2]] += x0
            global_box[[1, 3]] += y0

            global_mask = np.zeros((height, width), dtype=np.float32)
            global_mask[y0:y1, x0:x1] = mask[: y1 - y0, : x1 - x0]

            boxes.append(global_box)
            scores.append(float(score))
            labels.append(int(label))
            masks.append(global_mask)

    if not scores:
        return image_array, np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, height, width))

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.float32)

    keep_indices = apply_class_aware_nms(boxes, scores, labels, nms_threshold)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    masks = masks[keep_indices]

    return image_array, boxes, scores, labels, masks


def build_pixel_outputs(image, boxes, scores, labels, masks, mask_threshold):
    height, width = image.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    predictions = []

    order = np.argsort(scores)[::-1]
    for instance_id, pred_index in enumerate(order, start=1):
        label = int(labels[pred_index])
        binary_mask = masks[pred_index] >= mask_threshold
        write_mask = binary_mask & (instance_mask == 0)
        if not np.any(write_mask):
            continue

        instance_mask[write_mask] = instance_id
        class_mask[write_mask] = label

        semantic_id = label - 1
        x1, y1, x2, y2 = boxes[pred_index].tolist()
        predictions.append({
            "instance_id": instance_id,
            "label": label,
            "semantic_class_id": semantic_id,
            "class_name": CLASS_NAMES.get(semantic_id, f"class_{semantic_id}"),
            "score": float(scores[pred_index]),
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "area_pixels": int(write_mask.sum()),
        })

    overlay = overlay_masks(image, masks[order], labels=labels[order], alpha=0.45)
    return class_mask, instance_mask, overlay, predictions


def save_prediction_outputs(
    image_path,
    output_dir,
    image,
    boxes,
    scores,
    labels,
    masks,
    mask_threshold,
    score_threshold,
    save_overlay,
    save_class_mask,
    save_instance_mask,
    save_json,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    class_mask, instance_mask, overlay, predictions = build_pixel_outputs(
        image, boxes, scores, labels, masks, mask_threshold
    )

    saved = {}
    if save_overlay:
        path = output_dir / f"{stem}_overlay.png"
        Image.fromarray(overlay).save(path)
        saved["overlay"] = str(path)

    if save_class_mask:
        path = output_dir / f"{stem}_class_mask.png"
        Image.fromarray(class_mask).save(path)
        saved["class_mask"] = str(path)

    if save_instance_mask:
        path = output_dir / f"{stem}_instance_mask.png"
        Image.fromarray(instance_mask).save(path)
        saved["instance_mask"] = str(path)

    if save_json:
        path = output_dir / f"{stem}_predictions.json"
        payload = {
            "image": str(image_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "score_threshold": float(score_threshold),
            "mask_threshold": float(mask_threshold),
            "num_predictions": len(predictions),
            "predictions": predictions,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved["json"] = str(path)

    return saved, class_mask.shape, instance_mask.shape


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mask R-CNN instance segmentation on one image or a folder."
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model_best.pth")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="outputs/results")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--tile-size", type=int, default=800)
    parser.add_argument("--tile-overlap", type=int, default=200)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--save-class-mask", action="store_true")
    parser.add_argument("--save-instance-mask", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not any([args.save_overlay, args.save_class_mask, args.save_instance_mask, args.save_json]):
        args.save_overlay = True

    checkpoint = resolve_checkpoint(args.checkpoint)

    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(cfg["inference"]["score_threshold"])
    )
    nms_threshold = float(cfg["inference"].get("nms_threshold", 0.3))
    device = select_device(args.device)
    model = load_model(cfg, checkpoint, device)

    input_path = args.input
    if input_path is None:
        input_path = get_default_image_path(cfg["dataset"]["images_dir"])
        print(f"No --input provided; using first dataset image: {input_path}")

    image_paths = list_image_paths(input_path)
    print(f"Using checkpoint: {checkpoint}")
    print(f"Images to process: {len(image_paths)}")

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Predicting: {image_path}")
        image, boxes, scores, labels, masks = predict_image(
            model,
            image_path,
            device,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
        saved, class_shape, instance_shape = save_prediction_outputs(
            image_path=image_path,
            output_dir=args.output,
            image=image,
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks=masks,
            mask_threshold=args.mask_threshold,
            score_threshold=score_threshold,
            save_overlay=args.save_overlay,
            save_class_mask=args.save_class_mask,
            save_instance_mask=args.save_instance_mask,
            save_json=args.save_json,
        )
        print(
            f"  Kept {len(scores)} predictions; "
            f"class_mask={class_shape}; instance_mask={instance_shape}"
        )
        for kind, path in saved.items():
            print(f"  Saved {kind}: {path}")


if __name__ == "__main__":
    main()
