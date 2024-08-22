from src.utils.misc import seed, set_device
from src.utils.data import init_data
from src.utils.model import train_model
from src.utils.evaluation import save_train_val_curves

import src.interpretable_transformers.vit_config as ViTs
from src.interpretable_transformers.vit_config import *

import os
import argparse
import pandas as pd

import torch
import torch.nn as nn

import wandb


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Training a transformer to classify pornographic content")
    parser.add_argument(
        "--project_title",
        type=str,
        help="""Title of the project when logging experiments to Weights & Biases (W&B).
                Required if the --wandb flag is set to True.""",
    )
    parser.add_argument(
        "--data_loc",
        type=str,
        required=True,
        help="Directory path where the dataset is stored",
    )
    parser.add_argument(
        "--model_save_loc",
        type=str,
        required=True,
        help="Directory where the trained model's checkpoint will be saved.",
    )
    parser.add_argument(
        "--metrics_save_loc",
        type=str,
        required=True,
        help="Directory where the training metrics will be saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit_base_patch16_224",
        choices=[
            "vit_base_patch16_224",
            "vit_large_patch16_224",
            "deit_base_patch16_224",
        ],
        help="""The name of the transformer model configuration to use.
                The script will attempt to load this model configuration from the vit_config module.""",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="If set, loads a pretrained version of the specified transformer model.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Number of hidden layers in the model.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument(
        "--split",
        type=float,
        nargs="*",
        default=[0.1, 0.2],
        help="The fractions of the dataset to use for validation and testing, respectively.",
    )
    parser.add_argument(
        "--data_aug",
        action="store_true",
        default=False,
        help="If set, applies data augmentation techniques to the training data.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="""If set, enables logging to Weights & Biases (W&B).
                Requires --project_title to be specified.""",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        parser.error(f"Invalid --data_loc {args.data_loc}")

    if args.wandb and not args.project_title:
        parser.error("Argument --project_title is required because --wandb was set to True")

    return args


def main():
    seed()

    args = _parse_arguments()

    device = set_device()

    print("Connecting to W&B")

    if args.wandb: 
        wandb.login()
        wandb.init(
            project=args.project_title,
            config = {
                "learning_rate": args.learning_rate, 
                "epochs": args.epochs, 
                "batch_size": args.batch_size
            }
        )

    print(f"Loading transformer {args.model_name} and data")

    constructor = getattr(ViTs, args.model_name, None)
    assert constructor is not None, "Invalid --model_name"

    NUM_CLASSES = 2
    model = constructor(pretrained=args.pretrained, num_classes=NUM_CLASSES)
    model = nn.DataParallel(model)
    model = model.to(device)

    cfg = model.module.default_cfg
    dataloaders, dataset_sizes = init_data(
        args.data_loc,
        args.data_aug,
        args.batch_size,
        args.split,
        cfg["input_size"][1],
        cfg["mean"],
        cfg["std"]
    )

    print("Training: started")

    best_model_state_dict, metrics = train_model(
        model,
        dataloaders,
        dataset_sizes,
        args.optimizer,
        args.learning_rate,
        args.epochs,
        device,
        args.wandb
    )

    print("Saving metrics and best checkpoint")

    os.makedirs(args.model_save_loc, exist_ok=True)
    os.makedirs(args.metrics_save_loc, exist_ok=True)

    model_name = f"{args.model_name}_epochs_{args.epochs}_batch_{args.batch_size}_optim_{args.optimizer}_aug_{args.data_aug}_split_{int(args.split[0]*100)}_{int(args.split[1]*100)}"
    torch.save(best_model_state_dict, f"{args.model_save_loc}/{model_name}.pth")
    pd.DataFrame(metrics).to_csv(f"{args.metrics_save_loc}/{model_name}.csv", index=False)
    save_train_val_curves(f"{args.metrics_save_loc}/{model_name}.png", metrics)

    print("Training: finished\n")


if __name__ == "__main__":
    main()
