"""
Patch the iSAID dataset into Mask R-CNN-friendly tiles.

iSAID images are huge (often several thousand pixels per side), far larger than
a Mask R-CNN can ingest. This script slices every image into fixed-size,
overlapping tiles and rewrites the COCO annotations so each tile becomes a
standalone COCO image with its own polygon/RLE instances.

Inputs  (from configs/config.yaml -> dataset):
    raw_images_dir   directory holding the real RGB satellite images (P####.png)
    raw_ann_file     the original iSAID COCO json (e.g. iSAID_train.json)

Outputs:
    images_dir       tile PNGs            (e.g. data/patches/images)
    ann_file         patched COCO json    (e.g. data/patches/annotations.json)

Each output tile keeps a ``source_image`` field so train/val splitting can be
done at the source-image level (no tiles from one big image leak across the
split).

Usage:
    python src/prepare_isaid.py
    python src/prepare_isaid.py --tile-size 800 --overlap 200 --limit 50
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from tqdm import tqdm

# iSAID images can exceed Pillow's decompression-bomb guard; raise the limit.
Image.MAX_IMAGE_PIXELS = None


def rasterize_polys_to_tile(segm, x0, y0, th, tw):
    """
    Rasterise an instance's polygon(s) directly into a tile-sized binary mask.

    Coordinates are shifted into the tile's local frame; cv2 clips anything that
    falls outside. This only ever allocates a tile-sized array (<= tile*tile),
    so it is far faster and lighter than rasterising the full-resolution image
    mask and cropping it.
    """
    m = np.zeros((th, tw), dtype=np.uint8)
    for poly in segm:
        if len(poly) < 6:        # need >= 3 (x, y) points
            continue
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


def tile_starts(length: int, tile: int, overlap: int):
    """Return the top-left start offsets that cover ``length`` with overlap."""
    if length <= tile:
        return [0]
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must be >= 0 and smaller than tile size")
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def boxes_overlap(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> bool:
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def main():
    parser = argparse.ArgumentParser(description="Tile iSAID into Mask R-CNN patches.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=None,
                        help="Drop instance fragments smaller than this (pixels) in a tile.")
    parser.add_argument("--keep-empty", action="store_true",
                        help="Also write tiles that contain no instances.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N source images (for quick tests).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dc = cfg["dataset"]

    raw_images_dir = dc["raw_images_dir"]
    raw_ann_file = dc["raw_ann_file"]
    out_images_dir = dc["images_dir"]
    out_ann_file = dc["ann_file"]

    tile_size = args.tile_size or int(dc.get("tile_size", 800))
    overlap = args.overlap if args.overlap is not None else int(dc.get("tile_overlap", 200))
    min_area = args.min_area if args.min_area is not None else int(dc.get("min_instance_area", 100))
    keep_empty = args.keep_empty or bool(dc.get("keep_empty_tiles", False))

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_ann_file), exist_ok=True)

    print(f"Loading annotations: {raw_ann_file}")
    coco = COCO(raw_ann_file)
    categories = coco.loadCats(coco.getCatIds())

    img_ids = sorted(coco.imgs.keys())
    if args.limit:
        img_ids = img_ids[: args.limit]

    out_images, out_anns = [], []
    next_image_id = 1
    next_ann_id = 1
    missing = 0

    print(f"Tiling {len(img_ids)} images  (tile={tile_size}, overlap={overlap}, "
          f"min_area={min_area}, keep_empty={keep_empty})")

    for img_id in tqdm(img_ids, desc="Images"):
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(raw_images_dir, info["file_name"])
        if not os.path.isfile(img_path):
            missing += 1
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]
        stem = os.path.splitext(info["file_name"])[0]

        # iSAID image records lack width/height, which annToMask needs to
        # rasterise polygons. Inject the real dimensions (info is the same
        # dict object held in coco.imgs, so annToMask picks them up).
        info["height"], info["width"] = H, W

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # Tile grid as (x0, y0, x1, y1).
        tiles = []
        for y0 in tile_starts(H, tile_size, overlap):
            for x0 in tile_starts(W, tile_size, overlap):
                tiles.append((x0, y0, min(x0 + tile_size, W), min(y0 + tile_size, H)))
        tile_buckets = [[] for _ in tiles]

        # For each instance, rasterise its polygon only into the tiles its
        # bounding box overlaps. No full-resolution mask is ever allocated.
        for ann in anns:
            seg = ann["segmentation"]
            cat = int(ann["category_id"])
            bx, by, bw, bh = ann["bbox"]
            ax0, ay0, ax1, ay1 = bx, by, bx + bw, by + bh
            rle_fallback = isinstance(seg, dict)   # crowd/RLE annotation (rare)
            full = coco.annToMask(ann) if rle_fallback else None

            for ti, (x0, y0, x1, y1) in enumerate(tiles):
                if not boxes_overlap(x0, y0, x1, y1, ax0, ay0, ax1, ay1):
                    continue
                th, tw = y1 - y0, x1 - x0
                if rle_fallback:
                    sub = full[y0:y1, x0:x1].astype(np.uint8)
                else:
                    sub = rasterize_polys_to_tile(seg, x0, y0, th, tw)
                if int(sub.sum()) < min_area:
                    continue
                rle = mask_util.encode(np.asfortranarray(sub))
                bbox = mask_util.toBbox(rle).tolist()           # [x, y, w, h]
                area = float(mask_util.area(rle))
                rle["counts"] = rle["counts"].decode("ascii")   # JSON-serialisable
                tile_buckets[ti].append({
                    "category_id": cat,
                    "segmentation": {"size": [th, tw], "counts": rle["counts"]},
                    "bbox": [float(b) for b in bbox],
                    "area": area,
                    "iscrowd": 0,
                })

        for (x0, y0, x1, y1), tile_anns in zip(tiles, tile_buckets):
            th, tw = y1 - y0, x1 - x0

            if not tile_anns and not keep_empty:
                continue

            tile_name = f"{stem}_{x0}_{y0}.png"
            Image.fromarray(image[y0:y1, x0:x1]).save(os.path.join(out_images_dir, tile_name))

            out_images.append({
                "id": next_image_id,
                "file_name": tile_name,
                "width": int(tw),
                "height": int(th),
                "source_image": info["file_name"],
                "x_offset": int(x0),
                "y_offset": int(y0),
            })
            for ta in tile_anns:
                ta["id"] = next_ann_id
                ta["image_id"] = next_image_id
                out_anns.append(ta)
                next_ann_id += 1
            next_image_id += 1

    coco_out = {"images": out_images, "annotations": out_anns, "categories": categories}
    with open(out_ann_file, "w", encoding="utf-8") as f:
        json.dump(coco_out, f)

    print("\nDone.")
    print(f"  Source images processed : {len(img_ids) - missing}")
    if missing:
        print(f"  Source images MISSING   : {missing}  "
              f"(no matching file in {raw_images_dir} — add the real P####.png images)")
    print(f"  Tiles written           : {len(out_images)}")
    print(f"  Instances written       : {len(out_anns)}")
    print(f"  Tile images dir         : {out_images_dir}")
    print(f"  Patched annotations     : {out_ann_file}")


if __name__ == "__main__":
    main()
