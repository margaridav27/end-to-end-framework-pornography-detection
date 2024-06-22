from src.utils.xai_visualization import visualize_explanation, normalize_attr

import os
import argparse
import json
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from IPython.display import display


CHW = lambda shape: shape[0] in (3, 4)
HWC = lambda input: np.transpose(input, (1, 2, 0))


def _calculate_attribution_in_box(attr, box_coords):
    x1, y1, x2, y2 = box_coords
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    reduction_axis = None
    if attr.ndim == 3:
        reduction_axis = 0 if CHW(attr.shape) else 2

    abs_attr = normalize_attr(attr, sign="absolute_value", reduction_axis=reduction_axis)
    pos_attr = neg_attr = None

    # If there are only positive attribution values, it is not possible to calculate neg_attr
    # Also, there is no need to calculate pos_attr because it will be equal to abs_attr
    if attr.min() < 0.0:
        pos_attr = normalize_attr(attr=attr, sign="positive", reduction_axis=reduction_axis)
        neg_attr = normalize_attr(attr=attr, sign="negative", reduction_axis=reduction_axis)

    total_attr = np.sum(abs_attr)
    total_box_attr = np.sum(abs_attr[y1:y2, x1:x2])
    perc_attr = (total_box_attr / total_attr) * 100

    perc_pos_attr = perc_neg_attr = None
    if pos_attr is not None:
        total_pos_attr = np.sum(pos_attr)
        total_box_pos_attr = np.sum(pos_attr[y1:y2, x1:x2])
        perc_pos_attr = (total_box_pos_attr / total_pos_attr) * 100
    if neg_attr is not None:
        total_neg_attr = np.sum(neg_attr)
        total_box_neg_attr = np.sum(neg_attr[y1:y2, x1:x2])
        perc_neg_attr = (total_box_neg_attr / total_neg_attr) * 100

    return perc_attr, perc_pos_attr, perc_neg_attr


def _blur_box(image, box_coords):
    x1, y1, x2, y2 = box_coords
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    roi = image[y1:y2, x1:x2]
    if len(roi) == 0: return
    KSIZE, SIGMA = (15, 15), 21
    roi = cv2.GaussianBlur(roi, KSIZE, SIGMA)

    image[y1:y2, x1:x2] = roi


def _add_entry(results, frame, box=None, conf=None, area=None, perc=None, abs_attr=None, pos_attr=None, neg_attr=None):
    results.setdefault("frame", []).append(frame)
    results.setdefault("box", []).append(box)
    results.setdefault("conf", []).append(conf)
    results.setdefault("area", []).append(area)
    results.setdefault("perc", []).append(perc)
    results.setdefault("abs_attr", []).append(abs_attr)
    results.setdefault("pos_attr", []).append(pos_attr)
    results.setdefault("neg_attr", []).append(neg_attr)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_loc",
        type=str,
        required=True,
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--faces_loc",
        type=str,
        required=True,
        help="""Directory containing detected faces data in JSON (.json) format. 
                Each JSON file should have the same name as the corresponding image file 
                (e.g., 'image.jpg' should have 'image.json' in this directory), 
                containing information about the detected faces for that image.""",
    )
    parser.add_argument(
        "--explanations_loc",
        type=str,
        required=True,
        help="""Directory containing explanations data in NumPy (.npy) format. 
                Each NumPy file should have the same name as the corresponding image file 
                (e.g., 'image.jpg' should have 'image.npy' in this directory), 
                containing attributions (explanations) for that image.""",
    )
    parser.add_argument(
        "--save_loc",
        type=str,
        required=True,
        help="Directory to save the anonymized explanations (if applied) and results.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save anonymized explanation(s). Note that the final results' dataframe is saved even when set to False.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show anonymized explanation(s). Note that it might not work as expected if not running in a notebook.",
    )
    parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")
    parser.add_argument("--to_anonymize", type=str, nargs="*")
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

    if not os.path.exists(args.explanations_loc):
        parser.error("Invalid --explanations_loc argument.")

    os.makedirs(args.save_loc, exist_ok=True)

    return args


def main():
    args = _parse_arguments()

    H = W = args.input_shape
    img_area = H * W
    data_transforms = A.Resize(height=H, width=W)

    results = {}

    to_anonymize = (
        args.to_anonymize
        if args.to_anonymize is not None and len(args.to_anonymize) > 0
        else os.listdir(args.data_loc)
    )

    for filename in to_anonymize:    
        # Skip non-image files
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
            print(f"No available explanation .npy for {filename} in the provided directory")

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

            attr, pos_attr, neg_attr = _calculate_attribution_in_box(attr_np, (x1, y1, x2, y2))

            _add_entry(
                results=results,
                frame=filename,
                box=(x1, y1, x2, y2),
                conf=conf,
                area=area,
                perc=(area / img_area) * 100,
                abs_attr=attr,
                pos_attr=pos_attr,
                neg_attr=neg_attr
            )

            if (args.save or args.show) and area != 0: 
                _blur_box(img, (x1, y1, x2, y2))

        if args.save: # Save blurred explanation
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
        elif args.show:  # Show blurred explanation
            visualize_explanation(
                image=img,
                attr=attr_np,
                side_by_side=args.side_by_side,
                show_colorbar=args.show_colorbar,
                colormap=args.colormap,
                outlier_perc=args.outlier_perc,
                alpha_overlay=args.alpha_overlay,
            ).show()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.save_loc}/results.csv", index=False) # Results dataframe will be saved either way
    if args.show:
        display(results_df)


if __name__ == "__main__":
    main()
