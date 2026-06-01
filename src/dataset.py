import os
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2
from PIL import Image
from pycocotools.coco import COCO

Image.MAX_IMAGE_PIXELS = None


class IsaidCocoDataset(Dataset):
    """
    Instance-segmentation dataset for iSAID stored in COCO format.

    Expects the *patched* dataset produced by ``prepare_isaid.py``:
        images_dir/   -> tile PNGs (e.g. P0000_0_0.png)
        ann_file      -> COCO json whose `images` carry a `source_image` field

    Each annotation's `category_id` (1..15) is used directly as the Mask R-CNN
    label (0 is reserved for background), so the model needs num_classes = 16.
    """

    def __init__(self, images_dir: str, ann_file: str = None, transforms=None,
                 min_instance_area: int = 1, coco=None, max_instances: int = None):
        self.images_dir = images_dir
        self.transforms = transforms
        self.min_instance_area = int(min_instance_area)
        # Keep the path so DataLoader workers can rebuild COCO lazily (see
        # __getstate__): the patched index is ~0.5 GB, so pickling a shared copy
        # into every spawned Windows worker is wasteful. Workers instead receive
        # only the lightweight path and load the index once on first access.
        self.ann_file = ann_file
        # Reuse a shared COCO object when given — loading the patched index
        # (hundreds of thousands of annotations) several times exhausts RAM.
        self.coco = coco if coco is not None else COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        # Cap instances per tile (training only). Dense iSAID tiles can hold
        # hundreds of instances; stacking that many full 800x800 masks blows
        # host + GPU memory. None = keep all (used for validation).
        self.max_instances = max_instances
        self._rng = np.random.default_rng()

    def __getstate__(self):
        # When DataLoader spawns workers (Windows = spawn) the dataset is
        # pickled. Drop the heavy COCO index from the payload and let each
        # worker rebuild it once from ``ann_file`` (see ``_get_coco``). This
        # keeps worker startup cheap and avoids serialising ~0.5 GB per worker.
        state = self.__dict__.copy()
        if self.ann_file is not None:
            state["coco"] = None
        return state

    def _get_coco(self):
        if self.coco is None:
            self.coco = COCO(self.ann_file)
        return self.coco

    def __len__(self):
        return len(self.ids)

    def source_image(self, idx: int) -> str:
        info = self._get_coco().imgs[self.ids[idx]]
        return info.get("source_image", info["file_name"])

    def __getitem__(self, idx):
        coco = self._get_coco()
        img_id = self.ids[idx]
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, info["file_name"])
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # Cap very dense tiles before rasterising any masks (memory guard).
        if self.max_instances is not None and len(anns) > self.max_instances:
            keep = self._rng.choice(len(anns), self.max_instances, replace=False)
            anns = [anns[i] for i in keep]

        masks, labels = [], []
        for ann in anns:
            m = coco.annToMask(ann)
            if m.shape != image.shape[:2] or int(m.sum()) < self.min_instance_area:
                continue
            masks.append(m.astype(np.uint8))
            labels.append(int(ann["category_id"]))

        # ── Augment image + instance masks together ───────────────────────
        if self.transforms is not None:
            if masks:
                t = self.transforms(image=image, masks=masks)
                image, masks = t["image"], t["masks"]
            else:
                image = self.transforms(image=image)["image"]

        # ── Rebuild boxes from (possibly transformed) masks; drop empties ──
        boxes, valid_masks, valid_labels = [], [], []
        for m, lab in zip(masks, labels):
            ys, xs = np.where(m > 0)
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1 + 1, y1 + 1])
            valid_masks.append((m > 0).astype(np.uint8))
            valid_labels.append(lab)

        h, w = image.shape[:2]
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(valid_labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(valid_masks), dtype=torch.uint8)
            area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            area = torch.zeros(0, dtype=torch.float32)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": torch.zeros(len(boxes_t), dtype=torch.int64),
        }
        return image_t, target


# ─────────────────────────────────────────────
#  Augmentation pipelines (operate on numpy image + instance masks)
# ─────────────────────────────────────────────
def get_transforms(train: bool = True, image_size: int = None):
    ops = []
    if image_size:
        ops.append(A.Resize(height=image_size, width=image_size))
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
        ]
    return A.Compose(ops) if ops else None


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────
#  Source-image-grouped train/val split
# ─────────────────────────────────────────────
def _grouped_split_indices(dataset, cfg):
    """
    Split tile indices into train/val by *source image* so that tiles cropped
    from one large satellite image never straddle the split (avoids leakage).
    The result is cached to disk for reproducibility.
    """
    dc = cfg.get("dataset", {})
    split_path = dc.get("split_indices_path")
    seed = int(dc.get("seed", 42))
    train_split = float(dc.get("train_split", 0.8))

    groups = defaultdict(list)
    for idx in range(len(dataset)):
        groups[dataset.source_image(idx)].append(idx)

    keys = sorted(groups.keys())

    if split_path and os.path.isfile(split_path):
        with open(split_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("source_keys") == keys and saved.get("seed") == seed:
            return saved["train_indices"], saved["val_indices"]

    rng = np.random.default_rng(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_split))
    train_keys = set(shuffled[:n_train])

    train_indices, val_indices = [], []
    for k in keys:
        (train_indices if k in train_keys else val_indices).extend(groups[k])

    if split_path:
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({
                "seed": seed,
                "train_split": train_split,
                "source_keys": keys,
                "train_indices": train_indices,
                "val_indices": val_indices,
            }, f, indent=2)

    return train_indices, val_indices


def get_dataloaders(cfg):
    dc = cfg.get("dataset", {})
    tc = cfg.get("training", {})
    images_dir = dc["images_dir"]
    ann_file = dc["ann_file"]
    image_size = dc.get("image_size")
    min_area = int(dc.get("min_instance_area", 1))
    max_instances = dc.get("max_instances")
    max_instances = int(max_instances) if max_instances else None

    if not os.path.isfile(ann_file):
        raise FileNotFoundError(
            f"Patched annotation file not found: {ann_file}\n"
            f"Run the preprocessing step first:  python src/prepare_isaid.py"
        )

    # Load the COCO index ONCE and share it across the train/val datasets — the
    # patched index has hundreds of thousands of annotations and loading it
    # several times (or copying it into dataloader workers) exhausts host RAM.
    coco = COCO(ann_file)

    train_ds_full = IsaidCocoDataset(
        images_dir, ann_file=ann_file, coco=coco,
        transforms=get_transforms(train=True, image_size=image_size),
        min_instance_area=min_area, max_instances=max_instances,
    )
    val_ds_full = IsaidCocoDataset(
        images_dir, ann_file=ann_file, coco=coco,
        transforms=get_transforms(train=False, image_size=image_size),
        min_instance_area=min_area, max_instances=None,   # keep all GT for honest metrics
    )
    train_indices, val_indices = _grouped_split_indices(train_ds_full, cfg)

    # Validate on a representative fraction of the val set (config: val_fraction).
    # An evenly-spaced stride keeps the class mix and is deterministic, so the
    # metric stays comparable across epochs. Fewer val tiles also means less time
    # and far less chance of a memory spike from a dense tile.
    val_fraction = float(tc.get("val_fraction", 1.0))
    if 0.0 < val_fraction < 1.0 and val_indices:
        step = max(1, round(1.0 / val_fraction))
        val_indices = val_indices[::step]

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)

    use_cuda = torch.cuda.is_available()
    num_workers = int(tc.get("num_workers", 2))
    pin_memory = bool(tc.get("pin_memory", use_cuda)) and use_cuda
    persistent = bool(tc.get("persistent_workers", num_workers > 0)) and num_workers > 0

    # Extra kwargs only valid when workers run in their own processes. A
    # prefetch_factor > 1 keeps each worker decoding/augmenting the next few
    # tiles while the GPU is busy, so the ~0.1 s/tile data cost overlaps the
    # GPU step instead of stalling it.
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent
        loader_kwargs["prefetch_factor"] = int(tc.get("prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds, batch_size=int(tc.get("batch_size", 2)), shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, **loader_kwargs,
    )
    # The val loader pins nothing and prefetches less. Pinning a batch-of-1 saves
    # almost no time, but the pinned-host buffers held by BOTH loaders' persistent
    # workers were the exact allocation that OOM'd at the train->val handoff, so
    # we remove that path entirely on the validation side.
    val_loader_kwargs = dict(loader_kwargs)
    if num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=False, **val_loader_kwargs,
    )
    return train_loader, val_loader
