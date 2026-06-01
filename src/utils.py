import numpy as np
import cv2
import torch


# ─────────────────────────────────────────────
#  iSAID instance categories
#
#  iSAID is a COCO-format instance-segmentation dataset with 15 foreground
#  object categories. Category ids in the annotation JSON are 1..15.
#
#  Mask R-CNN convention:
#      model label 0      = background
#      model label 1..15  = iSAID category 1..15  (used as-is)
#
#  Throughout the codebase, "semantic id" = model_label - 1 (range 0..14).
#  CLASS_NAMES / CLASS_COLORS below are keyed by that 0-based semantic id so
#  that the metrics/printing code (which subtracts 1 from the model label)
#  reads naturally.
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0:  "storage_tank",
    1:  "large_vehicle",
    2:  "small_vehicle",
    3:  "plane",
    4:  "ship",
    5:  "swimming_pool",
    6:  "harbor",
    7:  "tennis_court",
    8:  "ground_track_field",
    9:  "soccer_ball_field",
    10: "baseball_diamond",
    11: "bridge",
    12: "basketball_court",
    13: "roundabout",
    14: "helicopter",
}

# A fixed, visually distinct palette (one RGB colour per semantic id) used for
# overlay visualisations during training/prediction.
CLASS_COLORS = {
    0:  (230,  25,  75),
    1:  ( 60, 180,  75),
    2:  (255, 225,  25),
    3:  (  0, 130, 200),
    4:  (245, 130,  48),
    5:  (145,  30, 180),
    6:  ( 70, 240, 240),
    7:  (240,  50, 230),
    8:  (210, 245,  60),
    9:  (250, 190, 212),
    10: (  0, 128, 128),
    11: (220, 190, 255),
    12: (170, 110,  40),
    13: (255, 250, 200),
    14: (128,   0,   0),
}

# num_classes for the model = number of foreground classes + 1 (background).
NUM_CLASSES = len(CLASS_NAMES) + 1   # 16


# ─────────────────────────────────────────────
#  Overlay predicted/ground-truth masks on an image
# ─────────────────────────────────────────────
def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend instance masks onto an RGB image.

    Args:
        image:  (H, W, 3) uint8 RGB image.
        masks:  (N, H, W) float32 soft masks in [0, 1] (or binary).
        labels: (N,) int array of model label ids (1-indexed; semantic id =
                label - 1). If None, colours cycle through the palette.
        alpha:  Blend factor for the mask colour.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    out = image.copy().astype(np.float32)
    color_list = list(CLASS_COLORS.values())   # fallback cycle

    for i, mask in enumerate(masks):
        if labels is not None:
            semantic_class = int(labels[i]) - 1
            color = CLASS_COLORS.get(semantic_class, color_list[i % len(color_list)])
        else:
            color = color_list[i % len(color_list)]

        binary = mask > 0.5
        for c in range(3):
            out[:, :, c] = np.where(
                binary,
                out[:, :, c] * (1 - alpha) + color[c] * alpha,
                out[:, :, c],
            )

    return out.astype(np.uint8)


# ─────────────────────────────────────────────
#  Misc helpers
# ─────────────────────────────────────────────
def resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    padded = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    padded[:nh, :nw] = resized
    return padded


def select_device(preference: str = "auto") -> torch.device:
    """
    Prefer NVIDIA CUDA when available, otherwise fall back to CPU.
    Prints concise runtime diagnostics to make the device choice explicit.
    """
    preference = preference.lower()
    if preference not in {"auto", "cuda", "cpu"}:
        raise ValueError("device preference must be one of: auto, cuda, cpu")

    cuda_available = torch.cuda.is_available()
    if preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")

    use_cuda = cuda_available and preference != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using device: {device}")
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        cuda_build = torch.version.cuda
        if cuda_build is None:
            print("  CUDA runtime not available in this PyTorch build (CPU-only build detected).")
        else:
            print(f"  PyTorch CUDA build: {cuda_build}, but no CUDA device is currently available.")

    return device


def load_state_dict(checkpoint_path: str, device: torch.device):
    """
    Load a model state dict with PyTorch's safer weights-only mode when
    available. Checkpoints should still come from trusted training runs.
    """
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
