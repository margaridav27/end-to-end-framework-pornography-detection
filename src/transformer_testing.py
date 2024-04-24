from src.utils.misc import seed, set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, test_model
from src.utils.evaluation import save_confusion_matrix
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import src.interpretable_transformers.vit_config as ViTs
from src.interpretable_transformers.vit_config import *

import os
import argparse
import pandas as pd
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Testing a trained pytorch model")
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--state_dict_loc", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        parser.error("Invalid --data_loc argument.")

    if not os.path.exists(args.state_dict_loc):
        parser.error("Invalid --state_dict_loc argument.")

    return args


def _get_test_dataloader(
    data_loc: str,
    split: List[float],
    batch_size: int,
    input_shape: int,
    norm_mean: List[float],
    norm_std: List[float],
) -> DataLoader:
    df_test = load_split(data_loc, split, ["test"])["test"]
    data_transforms = get_transforms(False, input_shape, norm_mean, norm_std)["test"]
    dataset = PornographyFrameDataset(data_loc, df_test, data_transforms)
    return DataLoader(dataset, batch_size)


def main():
    seed()

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
    dataloader = _get_test_dataloader(
        args.data_loc,
        split,
        args.batch_size,
        cfg["input_size"][1],
        cfg["mean"],
        cfg["std"]
    )

    print("Testing: started")

    results = test_model(model, dataloader, device)

    print("Saving results")

    os.makedirs(args.save_loc, exist_ok=True)
    results_save_loc = f"{args.save_loc}/{model_filename}"
    pd.DataFrame(results).to_csv(f"{results_save_loc}.csv", index=False)
    save_confusion_matrix(
        f"{results_save_loc}_confusion_matrix.png",
        results["Target"],
        results["Prediction"],
    )

    print("Testing: finished\n")


if __name__ == "__main__":
    main()
