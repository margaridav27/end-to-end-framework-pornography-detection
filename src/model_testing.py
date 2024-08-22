from src.utils.misc import seed, set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, load_model, test_model
from src.utils.evaluation import save_confusion_matrix
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import pandas as pd
from typing import List

from torch.utils.data import DataLoader


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Testing a trained pytorch model")
    parser.add_argument(
        "--data_loc",
        type=str,
        required=True,
        help="Directory path where the test dataset is stored.",
    )
    parser.add_argument(
        "--save_loc",
        type=str,
        required=True,
        help="Directory where the test results, including predictions and confusion matrix, will be saved.",
    )
    parser.add_argument(
        "--state_dict_loc",
        type=str,
        required=True,
        help="File path to the saved state dictionary (checkpoint) of the trained model.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_shape", type=int, default=224)
    parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])

    args = parser.parse_args()

    if not os.path.exists(args.state_dict_loc):
        parser.error("Invalid --state_dict_loc argument.")

    return args


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


def main():
    seed()

    args = _parse_arguments()
    
    device = set_device()

    model_filename, model_name, split = parse_model_filename(args.state_dict_loc)
    
    print(f"Loading {model_name} and test data...")
    model = load_model(model_name, args.state_dict_loc, device)
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
