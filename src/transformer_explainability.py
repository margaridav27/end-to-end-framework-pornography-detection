from src.utils.misc import set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, predict
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import src.interpretable_transformers.vit_config as ViTs
from src.interpretable_transformers.vit_config import *
from src.interpretable_transformers.xai_utils import (
    generate_attribution,
    generate_attribution_visualization,
)

import os
import argparse
from typing import List
import matplotlib.pyplot as plt
import numpy as np
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


def _explain_and_save(model, cfg, sample, save_loc):
    name, input, label, pred = sample

    original_image, attr = generate_attribution(
        image=input,
        ground_truth_label=label,
        model=model,
        mean_array=cfg["mean"],
        std_array=cfg["std"],
    )

    overlay = generate_attribution_visualization(
        image=original_image, 
        attr=attr
    )

    # Save image and overlay, side by side
    jpgs_save_loc = os.path.join(save_loc, "jpgs")
    os.makedirs(jpgs_save_loc, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axs.flat: ax.axis("off") 
    axs[0].imshow(original_image)
    axs[1].imshow(overlay)
    fig.savefig(f"{jpgs_save_loc}/{name}_pred_{pred}.png")
    plt.close(fig)

    # Save attribution .npy
    npys_save_loc = os.path.join(save_loc, "npys")
    os.makedirs(npys_save_loc, exist_ok=True)

    np.save(f"{npys_save_loc}/{name}_pred_{pred}.npy", attr)


def main():
    args = _parse_arguments()

    device = set_device()

    print(f"Loading transformer {args.model_name} and test data")

    constructor = getattr(ViTs, args.model_name, None)
    assert constructor is not None, "Invalid --model_name"

    NUM_CLASSES = 2
    model = constructor(num_classes=NUM_CLASSES)
    model = nn.DataParallel(model)
    model = model.to(device)

    state_dict = torch.load(args.state_dict_loc, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    model_filename, _, split = parse_model_filename(args.state_dict_loc)
    cfg = model.module.default_cfg
    dataset = _load_test_dataset(
        args.data_loc, 
        split, 
        cfg["input_size"][1], 
        cfg["mean"], 
        cfg["std"]
    )

    save_loc = os.path.join(args.save_loc, model_filename)

    if len(args.to_explain) == 0:  # Generate for entire dataset
        for name, input, label in dataset:
            if "NonPorn" in name:  # Skip non-porn samples for now
                continue

            input = input.to(device)

            _, pred = predict(model, input.unsqueeze(0))
            print(f"Prediction for '{name}': {pred.item()}")

            # temporary
            if os.path.isfile(f"{os.path.join(save_loc, 'jpgs')}/{name}_pred_{pred.item()}.png"): 
                continue

            _explain_and_save(
                model=model,
                cfg=cfg,
                sample=(name, input, label, pred.item()),
                save_loc=save_loc,
            )
    else:
        for image_name in args.to_explain:
            _, input, label = dataset[image_name]

            input = input.to(device)

            _, pred = predict(model, input.unsqueeze(0))
            print(f"Prediction for '{image_name}': {pred.item()}")

            _explain_and_save(
                model=model,
                cfg=cfg,
                sample=(image_name, input, label, pred.item()),
                save_loc=save_loc,
            )


if __name__ == "__main__":
    main()
