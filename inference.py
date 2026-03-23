import argparse
import cv2
import os
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

def setup_cfg(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--output", default="output.jpg", help="Path to save output")

    args = parser.parse_args()

    # Setup config
    cfg = setup_cfg(args.weights)

    # Setup metadata (clean names)
    metadata = MetadataCatalog.get("inference_dataset")
    metadata.thing_classes = ["Peritoneum", "Ovary", "TIE", "Uterus"]

    predictor = DefaultPredictor(cfg)

    # Read image
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")

    # Predict
    outputs = predictor(img)

    # Visualize
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save result
    cv2.imwrite(args.output, out.get_image()[:, :, ::-1])

    print(f"✅ Output saved to {args.output}")


if __name__ == "__main__":
    main()