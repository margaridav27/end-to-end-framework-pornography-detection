from src.utils.misc import set_device
from src.utils.data import load_split
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import json
import cv2
import albumentations as A

import torch
from torch.utils.data import DataLoader

from ultralytics import YOLO


def _parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO face detection")
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--weights", type=str, default="yolov8n-face.pt")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")
    parser.add_argument("--input_shape", type=int, default=224)

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        parser.error("Invalid --data_loc argument.")

    os.makedirs(args.save_loc, exist_ok=True)

    return args


def main():
    args = _parse_arguments() 

    device = set_device()

    h_transf = w_transf = args.input_shape
    data_transforms = A.Resize(height=h_transf, width=w_transf)

    df_test = load_split(args.data_loc, args.split, "test")["test"]
    dataset = PornographyFrameDataset(args.data_loc, df_test, data_transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    model = YOLO(args.weights)

    for names, _, _, orig_shapes in dataloader:        
        results = model.predict(
            source=[f"{args.data_loc}/{name}" for name in names],
            conf=args.conf_thres,
            device=device,
        )

        for i, result in enumerate(results):
            result = result.cpu()

            n_detected = result.boxes.data.size()[0]
            print(f"{names[i]}: {n_detected} faces detected")

            # check if any bounding boxes were detected
            if n_detected == 0: continue

            # discard the keypoints
            result.keypoints = None

            h_orig, w_orig, _ = orig_shapes[i]
            h_ratio, w_ratio = h_transf / h_orig, w_transf / w_orig

            # adjust bounding boxes coordinates
            # had to do this because the performance was worse if detecting on the resized images
            adjusted_boxes = torch.clone(result.boxes.data)
            adjusted_boxes[:, 0] = torch.round(adjusted_boxes[:, 0] * w_ratio)
            adjusted_boxes[:, 1] = torch.round(adjusted_boxes[:, 1] * h_ratio)
            adjusted_boxes[:, 2] = torch.round(adjusted_boxes[:, 2] * w_ratio)
            adjusted_boxes[:, 3] = torch.round(adjusted_boxes[:, 3] * h_ratio)

            result.update(boxes=adjusted_boxes)

            with open(f"{args.save_loc}/{os.path.splitext(names[i])[0]}.json", "w") as file:
                json.dump(json.loads(result.tojson()), file)


if __name__ == "__main__":
    main()
