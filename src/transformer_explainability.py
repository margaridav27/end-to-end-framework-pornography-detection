from src.utils.misc import set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, predict
from src.datasets.pornography_frame_dataset import PornographyFrameDataset
from src.interpretable_transformers.vit_config import vit_base_patch16_224
from src.interpretable_transformers.xai_utils import generate_attribution, generate_attribution_visualization

import os
import argparse
from typing import List
import cv2

import torch
import torch.nn as nn


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training a pytorch model to classify pornographic content"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--state_dict_loc", type=str, required=True)
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True)
    parser.add_argument(
        "--to_explain",
        type=str,
        nargs="*",
        default=[],
        help="Frame names for which an explanation is desired. If no names are given, an explanation for each prediction will be generated.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.state_dict_loc):
        raise ValueError("Invalid --state_dict_loc argument.")

    if not os.path.exists(args.data_loc):
        raise ValueError("Invalid --data_loc argument.")

    return args


def _load_test_dataset(
    data_loc: str,
    split: List[float],
    input_shape: int,
    norm_mean: List[float],
    norm_std: List[float],
) -> PornographyFrameDataset:
    df_test = load_split(data_loc, split, ["test"])["test"]
    data_transforms = get_transforms(False, input_shape, norm_mean, norm_std)["test"]
    return PornographyFrameDataset(data_loc, df_test, data_transforms)


def main():
    args = _parse_arguments()

    device = set_device()

    print(f"Loading transformer {args.model_name} and test data")

    NUM_CLASSES = 2
    model = vit_base_patch16_224(pretrained=True, num_classes=NUM_CLASSES)
    model = nn.DataParallel(model)
    model = model.to(device)

    state_dict = torch.load(args.state_dict_loc, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    _, _, split = parse_model_filename(args.state_dict_loc)
    dataset = _load_test_dataset(
        args.data_loc, split, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    )

    for image_name in args.to_explain:
        _, input, label = dataset[image_name]

        input = input.to(device)

        _, pred = predict(model, input.unsqueeze(0))
        print(f"Prediction for '{image_name}': {pred.item()}")

        original_image, attr = generate_attribution(
            image=input,
            ground_truth_label=label,
            model=model,
            mean_array=[0.5, 0.5, 0.5],
            std_array=[0.5, 0.5, 0.5],
        )

        vis = generate_attribution_visualization(
            image=original_image,
            attr=attr
        )
        cv2.imwrite("transformer_vis.png", vis)


if __name__ == "__main__":
    main()
