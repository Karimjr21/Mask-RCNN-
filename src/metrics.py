from collections import defaultdict

import numpy as np
import torch


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a.bool()
    b = mask_b.bool()
    intersection = (a & b).sum().item()
    union = (a | b).sum().item()
    return float(intersection / union) if union else 0.0


# Cap the transient [chunk, pixels] boolean tensor at ~64 MiB. A dense iSAID
# tile can hold thousands of GT instances; broadcasting one prediction against
# *all* of them at full resolution builds an [n_gt, H*W] bool temporary that on
# this 800px tiling reaches gigabytes — enough to OOM a 4 GB card mid-validation
# even though steady-state VRAM is only ~1.5 GB. Chunking the GT axis keeps the
# peak transient bounded so it fits in PyTorch's already-reserved pool.
_IOU_CHUNK_BYTES = 64 * 1024 * 1024


def _iou_matrix_on(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> np.ndarray:
    """Pairwise mask IoU matrix [n_pred, n_gt] on the masks' current device.

    ``pred_masks`` / ``gt_masks`` are boolean tensors of shape [N, H, W]. Only
    the small [n_pred, n_gt] result crosses back to the host. Exact integer
    pixel counts (int64) then float64 division match the old per-pair
    ``mask_iou`` (Python int / int) bit-for-bit, so a borderline IoU can never
    flip across the threshold.
    """
    n_pred = pred_masks.shape[0]
    n_gt = gt_masks.shape[0]
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    pred_flat = pred_masks.reshape(n_pred, -1)
    gt_flat = gt_masks.reshape(n_gt, -1)
    pixels = pred_flat.shape[1]
    pred_area = pred_flat.sum(dim=1, dtype=torch.int64)        # [n_pred]
    gt_area = gt_flat.sum(dim=1, dtype=torch.int64)            # [n_gt]

    # Intersection per (pred, gt). Loop over predictions (typically far fewer
    # than pixels) and broadcast each against a bounded block of GT at a time —
    # vectorised GPU reductions, no per-pair host sync, bounded peak memory.
    gt_chunk = max(1, min(n_gt, _IOU_CHUNK_BYTES // max(pixels, 1)))
    inter = torch.empty((n_pred, n_gt), device=pred_masks.device, dtype=torch.int64)
    for i in range(n_pred):
        row = pred_flat[i:i + 1]
        for j in range(0, n_gt, gt_chunk):
            block = gt_flat[j:j + gt_chunk]
            inter[i, j:j + gt_chunk] = (row & block).sum(dim=1, dtype=torch.int64)

    union = pred_area[:, None] + gt_area[None, :] - inter
    iou = torch.where(
        union > 0,
        inter.double() / union.double(),
        torch.zeros((), dtype=torch.float64, device=inter.device),
    )
    return iou.cpu().numpy()


def _iou_matrix(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> np.ndarray:
    """Pairwise mask IoU, on the GPU when it fits, falling back to CPU on OOM.

    Chunking bounds the transient to ~64 MiB, which clears the common case. If
    the card is so fragmented it cannot even serve that (validation can run with
    near-zero free VRAM), recompute on the CPU rather than crash the run — the
    masks are boolean and the result is identical either way.
    """
    if pred_masks.is_cuda:
        try:
            return _iou_matrix_on(pred_masks, gt_masks)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return _iou_matrix_on(pred_masks.cpu(), gt_masks.cpu())
    return _iou_matrix_on(pred_masks, gt_masks)


def _average_precision(tp_flags, scores, num_gt):
    if num_gt == 0:
        return None
    if not tp_flags:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    tp = np.asarray(tp_flags, dtype=np.float32)[order]
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(float(num_gt), 1.0)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


class DetectionMetric:
    def __init__(self, iou_threshold=0.5, mask_threshold=0.5):
        self.iou_threshold = float(iou_threshold)
        self.mask_threshold = float(mask_threshold)
        self.class_stats = defaultdict(lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gt": 0,
            "scores": [],
            "tp_flags": [],
            "ious": [],
        })

    def update(self, output, target):
        # Keep tensors on their original device (GPU); only small label/score
        # vectors and the per-class IoU matrices are moved to the host.
        scores = output["scores"].detach()
        labels = output["labels"].detach()
        masks = output["masks"].detach()
        if masks.dim() == 4:                      # [N, 1, H, W] -> [N, H, W]
            masks = masks.squeeze(1)
        masks = masks >= self.mask_threshold

        gt_labels = target["labels"].detach()
        gt_masks = target["masks"].detach().bool()

        labels_cpu = labels.cpu()
        gt_labels_cpu = gt_labels.cpu()

        for label in sorted(set(labels_cpu.tolist()) | set(gt_labels_cpu.tolist())):
            label = int(label)
            pred_idx = torch.where(labels == label)[0]
            gt_idx = torch.where(gt_labels == label)[0]

            stats = self.class_stats[label]
            stats["gt"] += int(len(gt_idx))

            if len(pred_idx) == 0:
                stats["fn"] += int(len(gt_idx))
                continue

            # Predictions in descending-score order (same greedy order as before).
            order = torch.argsort(scores[pred_idx], descending=True)
            sorted_pred = pred_idx[order]
            sorted_scores = scores[sorted_pred].cpu().tolist()

            # IoU rows are aligned to sorted_pred; columns to gt_idx.
            if len(gt_idx) > 0:
                iou_mat = _iou_matrix(masks[sorted_pred], gt_masks[gt_idx])
            else:
                iou_mat = None

            matched_gt = set()                    # local GT column indices
            for pi in range(len(sorted_pred)):
                best_iou = 0.0
                best_gi = None
                if iou_mat is not None:
                    row = iou_mat[pi]
                    for gi in range(len(gt_idx)):
                        if gi in matched_gt:
                            continue
                        if row[gi] > best_iou:
                            best_iou = float(row[gi])
                            best_gi = gi

                stats["scores"].append(sorted_scores[pi])
                if best_gi is not None and best_iou >= self.iou_threshold:
                    matched_gt.add(best_gi)
                    stats["tp"] += 1
                    stats["tp_flags"].append(1)
                    stats["ious"].append(best_iou)
                else:
                    stats["fp"] += 1
                    stats["tp_flags"].append(0)

            stats["fn"] += max(0, int(len(gt_idx)) - len(matched_gt))

    def compute(self):
        total_tp = sum(s["tp"] for s in self.class_stats.values())
        total_fp = sum(s["fp"] for s in self.class_stats.values())
        total_fn = sum(s["fn"] for s in self.class_stats.values())

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class = {}
        aps = []
        all_ious = []
        for label, stats in sorted(self.class_stats.items()):
            class_precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
            class_recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
            class_f1 = (
                2 * class_precision * class_recall / max(class_precision + class_recall, 1e-12)
            )
            ap = _average_precision(stats["tp_flags"], stats["scores"], stats["gt"])
            if ap is not None:
                aps.append(ap)
            all_ious.extend(stats["ious"])
            per_class[label - 1] = {
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1,
                "ap50": ap,
                "mask_iou": float(np.mean(stats["ious"])) if stats["ious"] else 0.0,
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "gt": stats["gt"],
            }

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": float(np.mean(aps)) if aps else 0.0,
            "mask_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "per_class": per_class,
        }
