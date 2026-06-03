import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead


def get_model(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    anchor_sizes=None,
    anchor_aspect_ratios=None,
    detections_per_img: int = 100,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    box_score_thresh: float = None,
    arch: str = "v1",
    min_size: int = 800,
    max_size: int = 1333,
):
    """
    Mask R-CNN (ResNet-50 FPN) configured for iSAID-style aerial imagery.

    Tuning knobs that matter for dense, small-object satellite tiles:
      arch                      "v1" = maskrcnn_resnet50_fpn (original recipe);
                                "v2" = maskrcnn_resnet50_fpn_v2 (improved heads +
                                training recipe, several AP points higher on COCO).
      anchor_sizes              one size per FPN level (5 values). iSAID objects
                                (vehicles, etc.) are small, so smaller anchors than
                                the COCO default ((32,64,128,256,512)) help a lot.
      anchor_aspect_ratios      aspect ratios shared across FPN levels. iSAID has
                                very elongated objects (ships, harbors, bridges,
                                large vehicles), so ratios beyond the COCO default
                                (0.5, 1, 2) — e.g. 0.33 and 3.0 — improve coverage.
                                Changing the *count* of ratios changes the number of
                                anchors per location, so the RPN head is rebuilt to
                                match (its pretrained weights are then re-learned —
                                a small head that converges in a few epochs).
      min_size / max_size       network input short / long side. Tiles are 800 px;
                                raising min_size (e.g. 1000-1100) gives small objects
                                more pixels but costs ~(min_size/800)^2 in compute
                                and memory.
      detections_per_img        max objects returned per image. A single tile can
                                hold hundreds of vehicles; the default 100 caps recall.
      trainable_backbone_layers how many ResNet stages to fine-tune (0-5).
      box_score_thresh          eval-only: drop detections below this score *before*
                                the (expensive) mask head runs. torchvision's default
                                of 0.05 makes the mask head rasterise up to
                                ``detections_per_img`` low-confidence boxes that a
                                downstream score filter then discards. Set this to the
                                evaluation threshold to skip that wasted work — the set
                                of kept detections is unchanged. Leave None (= 0.05) for
                                prediction, where a lower threshold may be wanted.
    """
    arch = (arch or "v1").lower()
    if arch == "v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        builder = maskrcnn_resnet50_fpn_v2
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        builder = maskrcnn_resnet50_fpn

    extra = {} if box_score_thresh is None else {"box_score_thresh": float(box_score_thresh)}
    model = builder(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        box_detections_per_img=detections_per_img,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        min_size=min_size,
        max_size=max_size,
        **extra,
    )

    # Replace box classifier head for our class count.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask head for our class count.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Smaller, more elongated anchors for aerial objects. With the default three
    # aspect ratios the RPN head's 3-anchors-per-location matches the pretrained
    # weights; adding ratios changes the anchor count, so the RPN head is rebuilt
    # to the new count (and re-learned — it is small and converges quickly).
    if anchor_sizes is not None or anchor_aspect_ratios is not None:
        sizes_src = anchor_sizes if anchor_sizes is not None else (32, 64, 128, 256, 512)
        sizes = tuple((int(s),) for s in sizes_src)
        ratios_src = anchor_aspect_ratios if anchor_aspect_ratios is not None else (0.5, 1.0, 2.0)
        ratios = tuple(float(r) for r in ratios_src)
        aspect_ratios = (ratios,) * len(sizes)
        model.rpn.anchor_generator = AnchorGenerator(sizes, aspect_ratios)

        num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
        if num_anchors != 3:                 # 3 = pretrained head's anchor count
            out_channels = model.backbone.out_channels
            conv_depth = 2 if arch == "v2" else 1   # v2 RPN head uses 2 conv layers
            try:
                model.rpn.head = RPNHead(out_channels, num_anchors, conv_depth=conv_depth)
            except TypeError:                # older torchvision: no conv_depth kwarg
                model.rpn.head = RPNHead(out_channels, num_anchors)

    return model
