from src.utils.misc import set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, predict
from src.utils.xai import save_explanation
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import src.interpretable_transformers.vit_config as ViTs
from src.interpretable_transformers.vit_config import *
from src.interpretable_transformers.xai_utils import generate_attribution

import os
import gc
import argparse

import torch
import torch.nn as nn


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Generating explanations for a transformer's predictions")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--state_dict_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True)
    parser.add_argument("--to_explain", type=str, nargs="*", default=[], help="Frame names for which an explanation is desired. If no names are given, an explanation for each prediction will be generated.")
    parser.add_argument("--side_by_side", action="store_true", default=False)
    parser.add_argument("--show_colorbar", action="store_true", default=False)
    parser.add_argument("--colormap", type=str, default="jet")
    parser.add_argument("--outlier_perc", default=2)
    parser.add_argument("--alpha_overlay", type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        raise ValueError("Invalid --data_loc argument.")

    if not os.path.exists(args.state_dict_loc):
        raise ValueError("Invalid --state_dict_loc argument.")

    return args


def main():
    args = _parse_arguments()

    device = set_device()

    model_filename, _, split = parse_model_filename(args.state_dict_loc)

    print(f"Loading transformer {args.model_name} and test data")

    constructor = getattr(ViTs, args.model_name, None)
    assert constructor is not None, "Invalid --model_name argument."

    NUM_CLASSES = 2
    model = constructor(num_classes=NUM_CLASSES)
    model = nn.DataParallel(model)
    model = model.to(device)

    state_dict = torch.load(args.state_dict_loc, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    cfg = model.module.default_cfg
    data_transforms = get_transforms(
        data_aug=False, 
        input_shape=cfg["input_size"][1], 
        norm_mean=cfg["mean"], 
        norm_std=cfg["std"]
    )["test"]
    dataset = PornographyFrameDataset(
        data_loc=args.data_loc,
        df=load_split(args.data_loc, split, "test")["test"],
        transform=data_transforms,
    )

    if len(args.to_explain) == 0:  # Generate for entire dataset
        for name, input, label, _ in dataset:
            input = input.to(device)

            _, pred = predict(model, input.unsqueeze(0))
            print(f"Prediction for '{name}': {pred.item()}")

            attr = generate_attribution(model=model, input=input, label=label)
            save_explanation(
                save_loc=os.path.join(
                    args.save_loc,
                    model_filename,
                    "correct" if pred.item() == label else "incorrect",
                ),
                image=input,
                image_name=name,
                attr=attr,
                side_by_side=args.side_by_side,
                show_colorbar=args.show_colorbar,
                colormap=args.colormap,
                outlier_perc=args.outlier_perc,
                alpha_overlay=args.alpha_overlay,
                norm_mean=cfg["mean"],
                norm_std=cfg["std"],
            )
    else:
        for image_name in args.to_explain:
            _, input, label, _ = dataset[image_name]

            input = input.to(device)

            _, pred = predict(model, input.unsqueeze(0))
            print(f"Prediction for '{image_name}': {pred.item()}")

            attr = generate_attribution(model=model, input=input, label=label)
            save_explanation(
                save_loc=os.path.join(
                    args.save_loc,
                    model_filename,
                    "correct" if pred.item() == label else "incorrect",
                ),
                image=input,
                image_name=image_name,
                attr=attr,
                side_by_side=args.side_by_side,
                show_colorbar=args.show_colorbar,
                colormap=args.colormap,
                outlier_perc=args.outlier_perc,
                alpha_overlay=args.alpha_overlay,
                norm_mean=cfg["mean"],
                norm_std=cfg["std"],
            )

    # Clear model
    del model

    # Run garbage collector
    gc.collect()

if __name__ == "__main__":
    main()
