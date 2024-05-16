from src.utils.explainability import visualize_explanation

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import albumentations as A


def _calculate_attribution_in_box(attr, box_coords):
    x1, y1, x2, y2 = box_coords
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    total_attr = np.sum(attr)
    box_attr = attr[y1:y2, x1:x2]
    total_box_attr = np.sum(box_attr)
    
    return (total_box_attr / total_attr) * 100


def _blur_box(image, box_coords):
    x1, y1, x2, y2 = box_coords
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    roi = image[y1:y2, x1:x2]
    if len(roi) == 0: return
    KSIZE, SIGMA = (9, 9), 15
    roi = cv2.GaussianBlur(roi, KSIZE, SIGMA)
    
    image[y1:y2, x1:x2] = roi


def _add_entry(results, frame, box=None, conf=None, area=None, perc=None, attr=None):
    results.setdefault("frame", []).append(frame)
    results.setdefault("box", []).append(box)
    results.setdefault("conf", []).append(conf)
    results.setdefault("area", []).append(area)
    results.setdefault("perc", []).append(perc)
    results.setdefault("attr", []).append(attr)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--faces_loc", type=str, required=True)
    parser.add_argument("--explanations_loc", type=str)
    parser.add_argument("--save_loc", type=str, required=True, help="Directory to save the results and explanations (if applied).")
    parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")
    parser.add_argument("--input_shape", type=int, default=224)
    parser.add_argument("--side_by_side", action="store_true", default=False)
    parser.add_argument("--show_colorbar", action="store_true", default=False)
    parser.add_argument("--colormap", type=str, default="jet")
    parser.add_argument("--outlier_perc", default=2)
    parser.add_argument("--alpha_overlay", type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        parser.error("Invalid --data_loc argument.")

    if not os.path.exists(args.faces_loc):
        parser.error("Invalid --faces_loc argument.")

    if args.explanations_loc is not None and not os.path.exists(args.explanations_loc):
        parser.error("Invalid --explanations_loc argument.")

    os.makedirs(args.save_loc, exist_ok=True)

    return args


def main():
    args = _parse_arguments()

    results = {}

    H = W = args.input_shape
    img_area = H * W
    data_transforms = A.Resize(height=H, width=W)

    for filename in os.listdir(args.data_loc):        
        if filename.startswith(".") or not filename.endswith((".png", ".jpg", ".jpeg")): 
            continue

        img_name, _ = os.path.splitext(filename)  # filename without extension
        json_file = os.path.join(args.faces_loc, img_name + ".json")
        attr_file = os.path.join(args.explanations_loc, img_name + ".npy")

        # Skip saving explanations for images where no faces were detected (no .json file)
        if not os.path.isfile(json_file):
            _add_entry(results=results, frame=filename)
            continue

        data = None
        with open(json_file, "r") as infile:
            data = json.load(infile)

        boxes = [
            [
                entry["box"]["x1"],
                entry["box"]["y1"],
                entry["box"]["x2"],
                entry["box"]["y2"],
                entry["confidence"],
                entry["class"],
            ]
            for entry in data
        ]

        if not os.path.isfile(attr_file):
            print(f"No available explanation .npy for {img_name} in the provided directory")

            for box in boxes:
                x1, y1, x2, y2, conf, _ = box
                area = (x2 - x1) * (y2 - y1)

                _add_entry(
                    results=results,
                    frame=filename,
                    box=(x1, y1, x2, y2),
                    conf=conf,
                    area=area,
                    perc=(area / img_area) * 100,
                )

            continue

        # Load original image
        img = cv2.imread(os.path.join(args.data_loc, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = data_transforms(image=img)["image"]

        # Load attribution numpy
        attr_np = np.load(attr_file)

        for box in boxes:
            x1, y1, x2, y2, conf, _ = box
            area = (x2 - x1) * (y2 - y1)

            _add_entry(
                results=results,
                frame=filename,
                box=(x1, y1, x2, y2),
                conf=conf,
                area=area,
                perc=(area / img_area) * 100,
                attr=_calculate_attribution_in_box(attr_np, (x1, y1, x2, y2)),
            )

            if area != 0: _blur_box(img, (x1, y1, x2, y2))

        # Save blurred explanation
        fig = visualize_explanation(
            image=img,
            attr=attr_np,
            side_by_side=args.side_by_side,
            show_colorbar=args.show_colorbar,
            colormap=args.colormap,
            outlier_perc=args.outlier_perc,
            alpha_overlay=args.alpha_overlay,
        )
        fig.savefig(f"{args.save_loc}/{filename}")
        plt.close(fig)

    pd.DataFrame(results).to_csv(f"{args.save_loc}/results.csv", index=False)


if __name__ == "__main__":
    main()
