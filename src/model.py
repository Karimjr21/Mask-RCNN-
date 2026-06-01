import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator


def get_model(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    anchor_sizes=None,
    detections_per_img: int = 100,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    box_score_thresh: float = None,
):
    """
    Mask R-CNN (ResNet-50 FPN) configured for iSAID-style aerial imagery.

    Tuning knobs that matter for dense, small-object satellite tiles:
      anchor_sizes              one size per FPN level (5 values). iSAID objects
                                (vehicles, etc.) are small, so smaller anchors than
                                the COCO default ((32,64,128,256,512)) help a lot.
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
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    extra = {} if box_score_thresh is None else {"box_score_thresh": float(box_score_thresh)}
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        box_detections_per_img=detections_per_img,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
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

    # Smaller anchors for small aerial objects. The RPN head has 3 anchors per
    # location (one size x three aspect ratios), unchanged from the pretrained
    # model, so the pretrained RPN weights still load cleanly.
    if anchor_sizes is not None:
        sizes = tuple((int(s),) for s in anchor_sizes)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
        model.rpn.anchor_generator = AnchorGenerator(sizes, aspect_ratios)

    return model
