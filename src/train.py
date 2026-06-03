import os
import csv
import sys
import math
import shutil
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from model import get_model
from dataset import get_dataloaders
from metrics import DetectionMetric
from utils import CLASS_NAMES, overlay_masks, select_device
from kaggle_sync import maybe_sync_checkpoints


# CUDA reports an out-of-memory either as a clean OutOfMemoryError (caught at the
# allocation site) or, when it surfaces from an async kernel, as a generic
# AcceleratorError / RuntimeError. Catch whichever this torch build exposes; we
# still filter on the message before swallowing it, so non-OOM errors re-raise.
_OOM_ERRORS = tuple({
    t for t in (
        getattr(torch.cuda, "OutOfMemoryError", None),
        getattr(torch, "AcceleratorError", None),
        RuntimeError,
    ) if isinstance(t, type)
})


# ═══════════════════════════════════════════════════════════════════════════════
#  Console styling — premium, colour-aware epoch reports
# ═══════════════════════════════════════════════════════════════════════════════

try:                               # make box-drawing / gauge glyphs safe on Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _enable_ansi_colors():
    """Best-effort enable of ANSI handling; return whether colour should be used."""
    if os.environ.get("NO_COLOR"):
        return False
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetStdHandle(-11)   # STD_OUTPUT_HANDLE
            ctypes.windll.kernel32.SetConsoleMode(handle, 7)    # +VT processing
        except Exception:
            return False
    return True


_COLOR = _enable_ansi_colors()


def _c(code):
    """An SGR escape for ``code``, or "" when colour is disabled."""
    return f"\033[{code}m" if _COLOR else ""


RESET = _c("0")
BOLD  = _c("1")
DIM   = _c("2")
GOLD  = _c("38;5;220")
CREAM = _c("38;5;230")
SLATE = _c("38;5;245")

# Quality tiers: (minimum value, label, 256-colour SGR code).
_TIERS = (
    (0.85, "Excellent", "38;5;46"),
    (0.70, "Good",      "38;5;82"),
    (0.50, "Fair",      "38;5;178"),
    (0.30, "Low",       "38;5;208"),
    (0.00, "Very low",  "38;5;196"),
)


def _tier(value):
    for threshold, label, code in _TIERS:
        if value >= threshold:
            return label, code
    return _TIERS[-1][1], _TIERS[-1][2]


def gauge(value, width=18):
    """A colour-graded progress bar for a 0-1 quality value."""
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    _, code = _tier(value)
    return (f"{_c(code)}{'█' * filled}{RESET}"
            f"{DIM}{'░' * (width - filled)}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(obj, path):
    """Save a torch object atomically.

    Writes to a temporary file in the same directory, then atomically replaces
    the target with ``os.replace``. A crash or full disk during the write then
    leaves the previous good checkpoint intact instead of a truncated, corrupt
    (0-byte) one. On a full disk ``torch.save`` raises a cryptic
    ``ios_base::badbit set`` / ``unexpected pos`` error — we translate that into
    a clear message naming the target and the remaining free space.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except (RuntimeError, OSError) as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        free_mb = shutil.disk_usage(directory).free / (1024 * 1024)
        raise RuntimeError(
            f"Could not save checkpoint to '{path}': {exc}\n"
            f"  Free space on the target drive is {free_mb:.0f} MB; a Mask R-CNN "
            f"checkpoint needs ~170-350 MB (last.pth with optimizer state more).\n"
            f"  Free disk space, or point training.checkpoint_dir / the logging "
            f"paths in configs/config.yaml at a drive with room."
        ) from exc


def prune_epoch_checkpoints(checkpoint_dir, keep):
    """Keep only the ``keep`` most recent periodic ``model_epoch_*.pth`` files.

    Over a 50-epoch run the per-``save_every`` checkpoints (~170 MB each) would
    otherwise pile up and fill the disk. ``model_best.pth`` and ``last.pth`` are
    not matched by the pattern, so they are always preserved. ``keep <= 0``
    disables pruning (all epoch checkpoints are retained).
    """
    if keep <= 0:
        return
    epoch_ckpts = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("model_epoch_") and name.endswith(".pth"):
            stem = name[len("model_epoch_"):-len(".pth")]
            if stem.isdigit():
                epoch_ckpts.append((int(stem), os.path.join(checkpoint_dir, name)))
    epoch_ckpts.sort()                       # oldest epoch first
    for _, old_path in epoch_ckpts[:-keep]:  # everything but the newest `keep`
        try:
            os.remove(old_path)
            print(f"  -> Pruned old checkpoint: {old_path}")
        except OSError:
            pass


def load_training_state(path, device):
    """Load a local resume checkpoint with PyTorch's restricted unpickler."""
    safe_numpy_types = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_numpy_types):
        return torch.load(path, map_location=device, weights_only=True)


def checkpoint_matches_model(model, ckpt_state) -> bool:
    """True iff a saved model state_dict has the same keys and shapes as ``model``.

    Changing the architecture (e.g. arch v1 -> v2) or the anchor count changes the
    parameter keys and/or shapes, so an old checkpoint cannot be loaded into the
    new model. We check first and, on a mismatch, skip the resume and train fresh —
    rather than letting ``load_state_dict`` abort the run with a wall of
    missing/unexpected-key errors (the usual symptom of a v1 last.pth meeting a v2
    config).
    """
    msd = model.state_dict()
    if set(msd.keys()) != set(ckpt_state.keys()):
        return False
    return all(msd[k].shape == ckpt_state[k].shape for k in msd)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training — one epoch
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    scheduler=None,
    accum_steps=1,
    gradient_clip_norm=0.0,
):
    """
    One training epoch with gradient accumulation.

    On a 4 GB GPU we are stuck at batch_size 1, whose gradients are very noisy.
    Accumulating gradients over ``accum_steps`` mini-batches before each
    optimizer step simulates an effective batch of that size — far more stable
    learning at no extra memory cost. The per-step LR ``scheduler`` (warmup +
    cosine) is advanced once per *optimizer* step.

    Returns:
        avg_loss  (float)  — mean total loss over all batches
        accuracy  (float)  — classification-loss-based progress proxy (0-1)
    """
    model.train()
    non_blocking = device.type == "cuda"
    total_loss    = 0.0
    correct_props = 0.0
    n_batches     = 0
    num_classes   = model.roi_heads.box_predictor.cls_score.out_features
    max_ce        = np.log(num_classes)
    n = len(loader)

    optimizer.zero_grad(set_to_none=True)
    for i, (images, targets) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images  = [img.to(device, non_blocking=non_blocking) for img in images]
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())
        (losses / accum_steps).backward()

        # Step once every accum_steps mini-batches (and flush the tail).
        if (i + 1) % accum_steps == 0 or (i + 1) == n:
            if gradient_clip_norm and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += losses.item()
        n_batches  += 1

        # ── Accuracy proxy ────────────────────────────────────────────
        # acc_proxy = 1 - (loss_classifier / log(num_classes)); rises as the
        # classifier improves (Mask R-CNN hides proposal labels in train mode).
        cls_loss       = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
        correct_props += max(0.0, 1.0 - cls_loss / max_ce)

    avg_loss = total_loss / n_batches if n_batches else 0.0
    accuracy = correct_props / n_batches if n_batches else 0.0
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation — loss + mean IoU
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_iou(mask_pred: torch.Tensor, mask_gt: torch.Tensor,
                 threshold: float = 0.5) -> float:
    pred_bin     = mask_pred > threshold
    intersection = (pred_bin & mask_gt.bool()).sum().float()
    union        = (pred_bin | mask_gt.bool()).sum().float()
    return (intersection / union).item() if union > 0 else 0.0


def validate(model, loader, device, score_threshold: float = 0.5, amp_enabled=False):
    """
    Returns:
        val_loss  (float)  — mean total loss on validation set
        mean_iou  (float)  — mean IoU across all matched instances
        per_class (dict)   — {semantic_class_id: mean_iou}
    """
    model.train()           # keep BN/dropout in train mode for loss computation
    non_blocking = device.type == "cuda"
    total_loss = 0.0

    all_ious       = []
    per_class_ious = {}     # semantic_id → [iou, ...]

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="  Val  ", leave=False):
            images  = [img.to(device, non_blocking=non_blocking) for img in images]
            targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

            # ── Validation loss ───────────────────────────────────────
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss_dict  = model(images, targets)
                losses     = sum(loss_dict.values())
            total_loss += losses.item()

            # ── IoU (need eval mode predictions) ─────────────────────
            model.eval()
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(images)
            model.train()

            for output, target in zip(outputs, targets):
                keep        = output["scores"] >= score_threshold
                pred_masks  = output["masks"][keep].squeeze(1)
                pred_labels = output["labels"][keep]
                gt_masks    = target["masks"]
                gt_labels   = target["labels"]

                for pm, pl in zip(pred_masks, pred_labels):
                    best_iou      = 0.0
                    best_gt_label = None
                    for gm, gl in zip(gt_masks, gt_labels):
                        iou = _compute_iou(pm, gm)
                        if iou > best_iou:
                            best_iou      = iou
                            best_gt_label = gl.item()

                    all_ious.append(best_iou)
                    if best_gt_label is not None:
                        sem_id = best_gt_label - 1
                        per_class_ious.setdefault(sem_id, []).append(best_iou)

    val_loss  = total_loss / len(loader) if len(loader) > 0 else 0.0
    mean_iou  = float(np.mean(all_ious)) if all_ious else 0.0
    per_class = {k: float(np.mean(v)) for k, v in per_class_ious.items()}

    model.eval()   # leave in eval mode; caller will call model.train() next epoch
    return val_loss, mean_iou, per_class


# ═══════════════════════════════════════════════════════════════════════════════
#  Fit diagnosis
# ═══════════════════════════════════════════════════════════════════════════════

def validate_detection(
    model,
    loader,
    device,
    score_threshold: float = 0.5,
    amp_enabled=False,
    metric_iou_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    compute_loss: bool = True,
    eval_fp16: bool = False,
):
    non_blocking = device.type == "cuda"
    total_loss = 0.0
    loss_batches = 0
    metric = DetectionMetric(
        iou_threshold=metric_iou_threshold,
        mask_threshold=mask_threshold,
    )

    # inference_mode is a stricter (and slightly faster) no_grad: it skips
    # version-counter bookkeeping. We never backprop through validation.
    skipped = 0
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="  Val  ", leave=False):
            try:
                images = [img.to(device, non_blocking=non_blocking) for img in images]
                targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

                if compute_loss:
                    # Loss needs a train-mode forward, which MUST stay fp32:
                    # torchvision Mask R-CNN NaNs under fp16 autocast on this GPU.
                    model.train()
                    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                        loss_dict = model(images, targets)
                        losses = sum(loss_dict.values())
                    total_loss += losses.item()
                    loss_batches += 1

                # The eval-mode forward CAN run fp16 (no backprop, boxes stay
                # finite): it roughly halves VRAM, so dense tiles fit in 4 GB
                # instead of spilling to system RAM and thrashing at 20-75 s/tile.
                model.eval()
                with torch.amp.autocast(device_type="cuda",
                                        enabled=(eval_fp16 or amp_enabled),
                                        dtype=torch.float16):
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
            except _OOM_ERRORS as exc:
                if "out of memory" not in str(exc).lower():
                    raise   # a real error, not memory pressure — let it surface
                # One pathologically dense tile blew the 4 GB budget. Drop every
                # reference to this batch's GPU tensors, reclaim the cached pool,
                # and keep validating instead of killing the whole run.
                skipped += 1
                images = targets = outputs = loss_dict = losses = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

    model.eval()
    if skipped:
        print(f"  ⚠  Skipped {skipped} validation tile(s) that exceeded GPU memory "
              f"(metric computed on the rest).")
    val_loss = total_loss / loss_batches if loss_batches else None
    return val_loss, metric.compute()


def diagnose_fit(train_losses, val_losses, val_scores, cfg):
    """Print a concise English summary of the training run."""
    tc = cfg["training"]
    overfit_gap = tc.get("overfit_gap_threshold", 0.15)
    underfit_score = tc.get("underfit_iou_threshold", 0.30)
    good_score = tc.get("good_iou_threshold", 0.45)

    final_train = train_losses[-1]
    measured_val_losses = [v for v in val_losses if v is not None and not math.isnan(v)]
    final_val = measured_val_losses[-1] if measured_val_losses else float("nan")
    # val_scores carries None on epochs where validation was skipped
    # (validation_interval > 1); rank only the epochs that were actually scored.
    measured_scores = [(i, v) for i, v in enumerate(val_scores) if v is not None]
    if not measured_scores:
        print("  No validation scores recorded; skipping fit diagnosis.")
        return "NO VALIDATION"
    final_score = measured_scores[-1][1]
    best_idx, best_score = max(measured_scores, key=lambda iv: iv[1])
    best_epoch = best_idx + 1
    gap = final_val - final_train if not math.isnan(final_val) else float("nan")

    window_values = measured_val_losses[-10:]
    plateaued = (max(window_values) - min(window_values) < 0.01) if len(window_values) >= 2 else False

    print()
    print("=" * 60)
    print("  Training diagnosis")
    print("=" * 60)
    print(f"  Final training loss       : {final_train:.4f}")
    if math.isnan(final_val):
        print("  Final validation loss     : not measured")
    else:
        print(f"  Final validation loss     : {final_val:.4f}   (gap: {gap:+.4f})")
    print(f"  Final detection quality   : {quality_percent(final_score)}")
    print(f"  Best detection quality    : {quality_percent(best_score)}   (epoch {best_epoch})")
    print(f"  Validation loss plateau   : {'yes - last 10 epochs vary < 0.01' if plateaued else 'no'}")
    print("-" * 60)

    if best_score < underfit_score:
        verdict = "UNDERFITTING"
        reasons = [
            "Best detection quality is below the underfitting threshold "
            f"({format_percent(best_score)} < {format_percent(underfit_score)}).",
            "The model has not learned the task well enough.",
        ]
        suggestions = [
            "Train for more epochs.",
            "Lower learning rate decay in config.yaml.",
            "Use stronger augmentation or more training data.",
        ]
    elif not math.isnan(gap) and gap > overfit_gap and final_score < best_score - 0.05:
        verdict = "POSSIBLE OVERFITTING"
        reasons = [
            f"Validation loss is higher than training loss (gap={gap:+.4f} > {overfit_gap}).",
            "Final detection quality is meaningfully worse than the best checkpoint.",
        ]
        suggestions = [
            "Use model_best.pth instead of model_final.pth.",
            "Keep early stopping enabled.",
            "Add more true instance-labeled data if possible.",
        ]
    elif plateaued and best_score < good_score:
        verdict = "UNDERFITTING (PLATEAU)"
        reasons = [
            "Validation loss has plateaued but detection quality is still below the good threshold.",
            f"{format_percent(best_score)} is below {format_percent(good_score)}.",
        ]
        suggestions = [
            "Try a cyclic or cosine learning-rate schedule.",
            "Fine-tune more of the backbone.",
            "Lower score_threshold in config.yaml to capture more predictions.",
        ]
    else:
        verdict = "BEST CHECKPOINT SELECTED"
        reasons = [
            f"Best detection quality is {quality_percent(best_score)}.",
            "The best checkpoint is selected by validation detection quality.",
        ]
        suggestions = [
            "Use model_best.pth for prediction.",
            "Run predict.py on new images to visually inspect results.",
            "Try test-time augmentation for better inference.",
        ]

    print()
    print(f"  Verdict: {verdict}")
    print()
    print("  Why:")
    for reason in reasons:
        print(f"    - {reason}")
    print()
    print("  Suggestions:")
    for suggestion in suggestions:
        print(f"    - {suggestion}")
    print("=" * 60)
    print()

    return verdict

def save_plots(train_losses, val_losses, val_scores, accuracies, plot_path, verdict):
    epochs = list(range(1, len(train_losses) + 1))
    val_plot = [np.nan if v is None else v for v in val_losses]
    # None on epochs where detection validation was skipped -> gap in the line.
    score_plot = [np.nan if v is None else v for v in val_scores]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Training summary  |  Verdict: {verdict}", fontsize=13, weight="bold")

    # ── Loss ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, train_losses, label="Training loss", color="#378ADD", linewidth=2)
    axes[0].plot(epochs, val_plot, label="Validation loss", color="#E24B4A", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy proxy ────────────────────────────────────────────────
    axes[1].plot(epochs, [a * 100 for a in accuracies],
                 label="Training progress", color="#1D9E75", linewidth=2)
    axes[1].set_title("Training progress")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Val IoU ───────────────────────────────────────────────────────
    axes[2].plot(epochs, [score * 100 for score in score_plot], label="Detection quality", color="#534AB7", linewidth=2, marker="o", markersize=3)
    axes[2].axhline(y=45, color="#888780", linestyle=":", linewidth=1, label="Good threshold")
    axes[2].axhline(y=30, color="#E24B4A", linestyle=":", linewidth=1, label="Low threshold")
    axes[2].set_title("Detection quality")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Quality (%)")
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV logger
# ═══════════════════════════════════════════════════════════════════════════════

def init_csv(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_precision", "val_recall",
            "val_f1", "val_map50", "val_mask_iou",
        ])


def append_csv(log_path, epoch, train_loss, train_acc, val_loss, metrics):
    val_loss_text = "" if val_loss is None else f"{val_loss:.4f}"
    # metrics is None on epochs where validation was skipped (validation_interval
    # > 1): the detection columns are left blank rather than carrying stale data.
    if metrics is None:
        metric_cols = ["", "", "", "", ""]
    else:
        metric_cols = [
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['map50']:.4f}",
            f"{metrics['mask_iou']:.4f}",
        ]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_acc:.4f}",
            val_loss_text,
            *metric_cols,
        ])


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-class IoU print helper
# ═══════════════════════════════════════════════════════════════════════════════

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


def print_per_class(per_class: dict):
    if not per_class:
        return
    parts = []
    for cid in sorted(per_class):
        name = CLASS_NAMES.get(cid, f"cls{cid}")
        parts.append(f"{name}: mask overlap {quality_percent(per_class[cid])}")
    print("    Per-class mask overlap: " + "  |  ".join(parts))


# ==============================================================================
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def print_epoch_header(epoch, total, lr):
    """A gold-bordered banner marking the start of an epoch."""
    inner = 62
    left_plain  = f"  EPOCH {epoch} / {total}"
    right_plain = f"learning rate {lr:.6f}  "
    gap = max(1, inner - len(left_plain) - len(right_plain))
    left  = f"  {GOLD}{BOLD}EPOCH {epoch}{RESET}{CREAM} / {total}{RESET}"
    right = f"{DIM}learning rate {RESET}{CREAM}{lr:.6f}{RESET}  "
    print()
    print(f"{GOLD}╔{'═' * inner}╗{RESET}")
    print(f"{GOLD}║{RESET}{left}{' ' * gap}{right}{GOLD}║{RESET}")
    print(f"{GOLD}╚{'═' * inner}╝{RESET}")


def _section(title):
    print(f"\n  {GOLD}◆ {BOLD}{title}{RESET}")


def _metric_row(label, value, width=18):
    """Aligned row:  label  ▕gauge▏  value%  tier."""
    tlabel, code = _tier(value)
    return (f"    {CREAM}{label:<15}{RESET}{gauge(value, width)}  "
            f"{_c(code)}{BOLD}{value * 100:5.1f}%{RESET}  {DIM}{tlabel}{RESET}")


def _plain_row(label, text):
    return f"    {CREAM}{label:<15}{RESET}{BOLD}{text}{RESET}"


def print_epoch_report(train_loss, train_acc, val_loss, metrics, val_loss_measured=True):
    """A premium, colour-graded summary of one finished epoch."""
    _section("TRAINING")
    print(_plain_row("Loss", f"{train_loss:.4f}"))
    print(_metric_row("Progress", train_acc))

    _section("VALIDATION")
    if val_loss_measured and val_loss is not None:
        print(_plain_row("Loss", f"{val_loss:.4f}"))
    else:
        print(_plain_row("Loss", "skipped this epoch"))
    print(_metric_row("Detection mAP", metrics["map50"]))
    print(_metric_row("Precision",     metrics["precision"]))
    print(_metric_row("Recall",        metrics["recall"]))
    print(_metric_row("F1 score",      metrics["f1"]))
    print(_metric_row("Mask overlap",  metrics["mask_iou"]))

    _section("DETECTIONS")
    green = _c("38;5;46"); red = _c("38;5;196"); amber = _c("38;5;208")
    print(f"    {green}✓ {metrics['tp']:,} correct{RESET}      "
          f"{red}✗ {metrics['fp']:,} false alarms{RESET}      "
          f"{amber}⊘ {metrics['fn']:,} missed{RESET}")

    _section("PER-CLASS  ·  ranked by score")
    rows = []
    for cid in CLASS_NAMES:
        name = CLASS_NAMES.get(cid, f"cls{cid}")
        values = metrics["per_class"].get(cid)
        rows.append((values["f1"] if values and values["gt"] else -1.0, name, values))
    rows.sort(key=lambda r: r[0], reverse=True)
    for score, name, values in rows:
        if values is None or values["gt"] == 0:
            print(f"    {SLATE}{name:<20}{RESET}{DIM}—  no validation examples{RESET}")
            continue
        f1 = values["f1"]
        _, code = _tier(f1)
        print(f"    {CREAM}{name:<20}{RESET}{gauge(f1, 12)}  "
              f"{_c(code)}{BOLD}{f1 * 100:5.1f}%{RESET}  "
              f"{DIM}quality {(values['ap50'] or 0.0) * 100:.0f}%  ·  "
              f"found {values['recall'] * 100:.0f}%{RESET}")


def save_validation_previews(model, val_loader, device, cfg, epoch):
    preview_count = int(cfg.get("logging", {}).get("val_preview_count", 0))
    if preview_count <= 0:
        return

    preview_dir = cfg.get("logging", {}).get("val_preview_dir", "outputs/validation_previews")
    epoch_dir = os.path.join(preview_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    score_threshold = cfg["inference"]["score_threshold"]
    model.eval()
    non_blocking = device.type == "cuda"
    saved = 0
    with torch.inference_mode():
        for images, _ in val_loader:
            image_tensor = images[0]
            image_uint8 = (
                image_tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
            ).astype(np.uint8)

            output = model([image_tensor.to(device, non_blocking=non_blocking)])[0]
            keep = output["scores"] >= score_threshold
            masks = output["masks"][keep].detach().cpu().squeeze(1).numpy()
            labels = output["labels"][keep].detach().cpu().numpy()

            overlay = overlay_masks(image_uint8, masks, labels=labels, alpha=0.45)
            Image.fromarray(overlay).save(os.path.join(epoch_dir, f"val_{saved:02d}.png"))
            saved += 1
            if saved >= preview_count:
                break
    print(f"  Saved validation previews: {epoch_dir}")


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = select_device()
    print()
    if device.type == "cpu":
        print("  ⚠  Training on CPU is slow. Consider Google Colab for GPU access.\n")

    if device.type == "cuda":
        # cudnn.benchmark autotunes conv algorithms by trial-running each one,
        # probing ~3 GB of workspace on the first 800px forward. On a 4 GB GPU
        # that collides with dense-tile activations, spills into system RAM
        # (Windows sysmem fallback), and makes the first conv appear to hang for
        # minutes. Heuristic selection (benchmark=False) picks an algo instantly,
        # uses ~1 GB, and the first iter runs in ~1s. Keep it off here.
        torch.backends.cudnn.benchmark = False

    mc = cfg["model"]
    tc = cfg["training"]
    train_loader, val_loader = get_dataloaders(cfg)

    # Skip the mask head on detections the metric will discard anyway. We
    # filter predictions at inference.score_threshold, so there is no point
    # rasterising masks for lower-scoring boxes. Sit a hair below the eval
    # threshold (torchvision drops with strict ``>``, we keep with ``>=``) so a
    # detection scoring exactly at the threshold is never lost — the kept set is
    # identical, validation is just faster. Training ignores this (eval-only).
    eval_score_threshold = float(cfg["inference"]["score_threshold"])
    box_score_thresh = max(0.0, eval_score_threshold - 1e-4)
    model = get_model(
        mc["num_classes"],
        pretrained=mc.get("pretrained", True),
        trainable_backbone_layers=int(mc.get("trainable_backbone_layers", 3)),
        anchor_sizes=mc.get("anchor_sizes"),
        anchor_aspect_ratios=mc.get("anchor_aspect_ratios"),
        detections_per_img=int(mc.get("detections_per_img", 100)),
        box_score_thresh=box_score_thresh,
        arch=mc.get("arch", "v1"),
        min_size=int(mc.get("min_size", 800)),
        max_size=int(mc.get("max_size", 1333)),
    )
    model.to(device)

    accum_steps = max(1, int(tc.get("accumulation_steps", 1)))
    if device.type == "cuda" and bool(tc.get("amp", False)):
        print("  NOTE: AMP requested but ignored — torchvision Mask R-CNN gives "
              "NaN under fp16 autocast on this setup. Training in fp32.")
    print(f"  Effective batch size: {int(tc.get('batch_size', 1)) * accum_steps} "
          f"(batch {int(tc.get('batch_size', 1))} x {accum_steps} accumulation steps)")

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=tc["learning_rate"],
        momentum=tc["momentum"],
        weight_decay=tc["weight_decay"],
    )

    # Per-optimizer-step schedule: linear warmup -> cosine decay to ~0.
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = max(1, int(tc["epochs"]) * steps_per_epoch)
    warmup_steps = min(int(tc.get("warmup_iters", 1000)), max(1, total_steps // 2))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    log_path  = cfg["logging"]["log_file"]
    plot_path = cfg["logging"]["plot_file"]
    os.makedirs(os.path.dirname(log_path),  exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # ── History ───────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []
    val_scores   = []
    accuracies   = []

    total_epochs = cfg["training"]["epochs"]
    best_score = -1.0
    patience = int(cfg["training"].get("early_stopping_patience", 8))
    min_epochs = int(cfg["training"].get("early_stopping_min_epochs", 0))
    min_delta = float(cfg["training"].get("early_stopping_min_delta", 0.0))
    gradient_clip_norm = float(cfg["training"].get("gradient_clip_norm", 0.0))
    keep_checkpoints = int(cfg["training"].get("keep_last_checkpoints", 3))
    val_loss_interval = max(0, int(cfg["training"].get("validation_loss_interval", 1)))
    # Full detection validation (the expensive eval-mode forward over the whole
    # val set) every `validation_interval` epochs. The first and last epochs
    # always run so the run is bookended by a measured score. Best-checkpoint
    # selection and early stopping act only on epochs that were scored.
    validation_interval = max(1, int(cfg["training"].get("validation_interval", 1)))
    epochs_without_improvement = 0
    best_ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_best.pth")

    # ── Resume support ─────────────────────────────────────────────────
    # A long local run will get interrupted (reboot/sleep). last.pth holds the
    # full training state so re-running train.py picks up where it left off.
    last_ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "last.pth")
    start_epoch = 1
    if os.path.isfile(last_ckpt):
        state = load_training_state(last_ckpt, device)
        if not checkpoint_matches_model(model, state["model"]):
            # Most common cause: a v1 last.pth left over while the config now builds
            # a v2 model (different heads/FPN-norm/anchor count). Don't crash —
            # ignore the stale checkpoint and train fresh from epoch 1.
            print(f"  ⚠  {last_ckpt} does not match the current model architecture "
                  f"(arch/anchors changed). Ignoring it and training FRESH from "
                  f"epoch 1. Delete this file to silence the warning.")
        else:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            train_losses = state["train_losses"]
            val_losses   = state["val_losses"]
            val_scores   = state["val_scores"]
            accuracies   = state["accuracies"]
            best_score   = state["best_score"]
            epochs_without_improvement = state["epochs_without_improvement"]
            start_epoch  = state["epoch"] + 1
            print(f"  Resuming from {last_ckpt}: continuing at epoch {start_epoch} "
                  f"(best detection quality so far: {quality_percent(best_score)})")

    if start_epoch == 1:
        init_csv(log_path)

    for epoch in range(start_epoch, total_epochs + 1):
        print_epoch_header(epoch, total_epochs, scheduler.get_last_lr()[0])

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss, train_acc = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            scheduler=scheduler,
            accum_steps=accum_steps,
            gradient_clip_norm=gradient_clip_norm,
        )

        # Run the full detection validation this epoch?
        run_val = (
            validation_interval == 1
            or epoch == 1
            or epoch == total_epochs
            or epoch % validation_interval == 0
        )
        compute_val_loss = run_val and (
            val_loss_interval == 1
            or epoch == 1
            or epoch == total_epochs
            or (val_loss_interval > 1 and epoch % val_loss_interval == 0)
        )

        # (LR scheduler is advanced per optimizer step inside train_one_epoch.)

        if run_val:
            # ── Validate ──────────────────────────────────────────────
            # Release the cached training pool (gradients, SGD momentum,
            # batch-2 activations) before validation. On a 4 GB GPU the
            # eval-mode forward materialises full-resolution masks for every
            # detection and needs that headroom; without this the train→val
            # handoff OOMs (the error often surfaces in the pin-memory thread).
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_loss, metrics = validate_detection(
                model, val_loader, device,
                score_threshold=cfg["inference"]["score_threshold"],
                amp_enabled=False,
                metric_iou_threshold=cfg["inference"].get("metric_iou_threshold", 0.5),
                mask_threshold=cfg["inference"].get("mask_threshold", 0.5),
                compute_loss=compute_val_loss,
                eval_fp16=bool(cfg["inference"].get("val_eval_fp16", False)) and device.type == "cuda",
            )
            model.train()   # restore for next epoch
            if device.type == "cuda":
                torch.cuda.empty_cache()   # free val activations before next epoch

            # ── Record ────────────────────────────────────────────────
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_scores.append(metrics["map50"])
            accuracies.append(train_acc)

            append_csv(log_path, epoch, train_loss, train_acc, val_loss, metrics)

            # ── Print ──────────────────────────────────────────────────
            print_epoch_report(
                train_loss,
                train_acc,
                val_loss,
                metrics,
                val_loss_measured=compute_val_loss,
            )

            if metrics["map50"] > best_score + min_delta:
                best_score = metrics["map50"]
                epochs_without_improvement = 0
                save_checkpoint(model.state_dict(), best_ckpt)
                print(f"\n  {GOLD}★ New best{RESET}  ·  detection quality "
                      f"{GOLD}{BOLD}{format_percent(best_score)}{RESET}  "
                      f"{DIM}· saved to {best_ckpt}{RESET}")
            else:
                epochs_without_improvement += 1
                print(f"\n  {DIM}No improvement ({epochs_without_improvement}/{patience}) "
                      f"· best so far {format_percent(best_score)}{RESET}")
        else:
            # ── Validation skipped this epoch (validation_interval > 1) ──
            train_losses.append(train_loss)
            val_losses.append(None)
            val_scores.append(None)
            accuracies.append(train_acc)
            append_csv(log_path, epoch, train_loss, train_acc, None, None)

            _section("TRAINING")
            print(_plain_row("Loss", f"{train_loss:.4f}"))
            print(_metric_row("Progress", train_acc))
            print(f"\n  {DIM}Validation skipped this epoch · next full validation "
                  f"at epoch {((epoch // validation_interval) + 1) * validation_interval} "
                  f"· best so far {format_percent(max(best_score, 0.0))}{RESET}")

        # ── Full training state (for resume) ──────────────────────────
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_scores": val_scores,
            "accuracies": accuracies,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
        }, last_ckpt)

        # ── Mirror checkpoints to a Kaggle Dataset (no-op unless on Kaggle) ──
        # Gated by KAGGLE_CKPT_DATASET; pushes last.pth + model_best.pth every
        # KAGGLE_CKPT_INTERVAL epochs so a Kaggle session disconnect costs at
        # most a few epochs and the run resumes by re-attaching the dataset.
        maybe_sync_checkpoints(
            epoch,
            cfg["training"]["checkpoint_dir"],
            log_file=cfg["logging"].get("log_file"),
        )

        # ── Checkpoint ────────────────────────────────────────────────
        if epoch % cfg["training"]["save_every"] == 0:
            ckpt = os.path.join(
                cfg["training"]["checkpoint_dir"],
                f"model_epoch_{epoch}.pth",
            )
            save_checkpoint(model.state_dict(), ckpt)
            print(f"  {DIM}· checkpoint saved: {ckpt}{RESET}")
            prune_epoch_checkpoints(cfg["training"]["checkpoint_dir"], keep_checkpoints)

    # ── End-of-training summary ───────────────────────────────────────
        if epoch % cfg["training"]["save_every"] == 0:
            save_validation_previews(model, val_loader, device, cfg, epoch)

        if (
            patience > 0
            and epoch >= min_epochs
            and epochs_without_improvement >= patience
        ):
            print(
                f"  Early stopping: detection quality did not improve for "
                f"{epochs_without_improvement} epochs."
            )
            break

    verdict = diagnose_fit(train_losses, val_losses, val_scores, cfg)
    save_plots(train_losses, val_losses, val_scores, accuracies, plot_path, verdict)
    print(f"\n  Training log saved to : {log_path}")
    print(f"  Plots saved to        : {plot_path}\n")

    # Save final model
    final_ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_final.pth")
    save_checkpoint(model.state_dict(), final_ckpt)
    print(f"  Final model saved to  : {final_ckpt}\n")
    if best_score >= 0:
        print(f"  Best model saved to   : {best_ckpt} (detection quality: {quality_percent(best_score)})\n")

    # Final mirror so the best/last checkpoints land on Kaggle even when the run
    # ended on a non-sync epoch or via early stopping (no-op unless on Kaggle).
    maybe_sync_checkpoints(
        len(train_losses),
        cfg["training"]["checkpoint_dir"],
        log_file=cfg["logging"].get("log_file"),
        force=True,
    )


if __name__ == "__main__":
    main()
