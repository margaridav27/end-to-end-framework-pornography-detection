from src.utils.misc import set_device

import os
import argparse
import json
import cv2
import albumentations as A

import torch

from ultralytics import YOLO


def _parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO face detection")
    parser.add_argument(
        "--data_loc",
        type=str,
        required=True,
        help="Directory path where the input images are stored.",
    )
    parser.add_argument(
        "--save_loc",
        type=str,
        help="""Directory where the detection results (as JSON files) will be saved. 
                This argument is required if the --save flag is set to True.""",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8-weights/yolov8m-face.pt",
        help="Path to the YOLO model weights file.",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.6,
        help="""The confidence threshold for detecting objects (faces). 
                Detections with a confidence score below this threshold will be discarded.""",
    )
    parser.add_argument("--input_shape", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--to_detect",
        type=str,
        nargs="*",
        help="""A list of specific image filenames (within --data_loc) to run face detection on. 
                If not provided, all images in the --data_loc directory will be processed.""",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="If set, the detected results (with bounding boxes) will be displayed on the screen.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="If set, the detection results will be saved as JSON files in the specified --save_loc directory.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        parser.error("Invalid --data_loc argument.")

    if args.save:
        if args.save_loc is None:
            parser.error("You must specify --save_loc argument when --save is True.")
        else:
            os.makedirs(args.save_loc, exist_ok=True)

    return args


def main():
    args = _parse_arguments()

    if not (args.show or args.save):
        return

    device = set_device()

    model = YOLO(args.weights)

    to_detect = (
        args.to_detect
        if args.to_detect is not None and len(args.to_detect) > 0
        else os.listdir(args.data_loc)
    )

    if args.show and args.input_shape is not None:
        data_transforms = A.Resize(height=args.input_shape, width=args.input_shape)

    for filename in to_detect:
        result = model.predict(
            source=f"{args.data_loc}/{filename}",
            conf=args.conf_thres,
            device=device,
        )[0].cpu()

        n_detected = result.boxes.data.size()[0]
        print(f"{filename}: {n_detected} faces detected")

        # Check if any bounding boxes were detected
        if n_detected == 0:
            continue

        # Discard the keypoints
        result.keypoints = None

        # Map bounding box coordinates to given input shape
        if args.input_shape is not None:
            image = cv2.imread(os.path.join(args.data_loc, filename))

            h_orig, w_orig, _ = image.shape
            h_ratio, w_ratio = args.input_shape / h_orig, args.input_shape / w_orig

            # adjust bounding boxes coordinates
            # had to do this because the performance was worse if detecting on the resized images
            adjusted_boxes = torch.clone(result.boxes.data)
            adjusted_boxes[:, 0] = torch.round(adjusted_boxes[:, 0] * w_ratio)
            adjusted_boxes[:, 1] = torch.round(adjusted_boxes[:, 1] * h_ratio)
            adjusted_boxes[:, 2] = torch.round(adjusted_boxes[:, 2] * w_ratio)
            adjusted_boxes[:, 3] = torch.round(adjusted_boxes[:, 3] * h_ratio)

            result.update(boxes=adjusted_boxes)

            # No need to transform the image if not showing the result
            if args.show:
                image = data_transforms(image=image)["image"]
                result.orig_img = image

        if args.save:
            with open(f"{args.save_loc}/{os.path.splitext(filename)[0]}.json", "w") as file:
                json.dump(json.loads(result.tojson()), file)

        if args.show:
            result.show()


if __name__ == "__main__":
    main()
