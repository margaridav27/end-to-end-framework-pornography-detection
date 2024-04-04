from src.utils.misc import seed, set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import init_model, test_model
from src.utils.evaluation import save_confusion_matrix
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Testing a trained pytorch model")
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True)
    parser.add_argument("--state_dict_loc", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_shape", type=int, default=224)
    parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
    
    args = parser.parse_args()

    if not os.path.exists(args.state_dict_loc):
        parser.error("Invalid --state_dict_loc argument.")

    return args


def _get_model_filename_and_name(state_dict_loc : str) -> Tuple[str, str]:
    _, model_filename = os.path.split(state_dict_loc) # Includes .pth
    model_filename, _ = os.path.splitext(model_filename) # Does not include .pth

    model_filename_split = model_filename.split("_")[0]
    model_name = model_filename_split[0] if model_filename_split[1] == "freeze" else model_filename_split[:2]
    
    return model_filename, model_name


def _get_split(model_filename : str) -> List[float]:
    split = model_filename.split("_")[-2:]
    return [float(i)/100 for i in split]

    
def _get_test_dataloader(
    data_loc : str, 
    split : List[float], 
    batch_size : int,
    input_shape : int, 
    norm_mean : List[float], 
    norm_std : List[float]
) -> DataLoader:
    df_test = load_split(data_loc, split, ["test"])["test"]
    data_transforms = get_transforms(False, input_shape, norm_mean, norm_std)["test"]
    dataset = PornographyFrameDataset(data_loc, df_test, data_transforms)
    return DataLoader(dataset, batch_size)


def _load_model(model_name : str, state_dict_loc : str, device : str) -> nn.Module:
    print(f"Loading {model_name}...")

    state_dict = torch.load(state_dict_loc)
    model = init_model(model_name)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def main():
    seed()

    args = _parse_arguments()
    
    device = set_device()

    print("Loading model and test data...")
    model_filename, model_name = _get_model_filename_and_name(args.state_dict_loc)
    model = _load_model(model_name, args.state_dict_loc, device)
    split = _get_split(model_filename)
    dataloader = _get_test_dataloader(
        args.data_loc,
        split,
        args.batch_size,
        args.input_shape,
        args.norm_mean,
        args.norm_std
    )
    
    print("Model testing started...\n")
    results = test_model(model, dataloader, device)

    print("Saving results...")
    os.makedirs(args.save_loc, exist_ok=True)
    results_save_loc = f"{args.save_loc}/{model_filename}"
    pd.DataFrame(results).to_csv(f"{results_save_loc}.csv", index=False)
    save_confusion_matrix(f"{results_save_loc}_confusion_matrix.png", results["Target"], results["Prediction"])
    print("Results saved. Testing process has finished.\n\n")


if __name__ == "__main__":
    main()
