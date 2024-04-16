from src.utils.misc import seed, set_device
from src.utils.data import init_data
from src.utils.model import init_model, train_model
from src.utils.evaluation import save_train_val_curves

import os
import argparse
import pandas as pd

import torch
import torch.nn as nn

import wandb


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
    parser.add_argument("--project_title", type=str, required=True)
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--model_save_loc", type=str, required=True)
    parser.add_argument("--metrics_save_loc", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--freeze_layers", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")
    parser.add_argument("--data_aug", action="store_true", default=False)
    parser.add_argument("--input_shape", type=int, default=224)
    parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
    parser.add_argument("--wandb", action="store_true", default=False)

    return parser.parse_args()


def main():
    seed()
    
    args = _parse_arguments()

    device = set_device()

    # Configure W&B
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

    print(f"Loading {args.model_name} and data...")

    model = init_model(args.model_name, args.weights, args.freeze_layers)
    model = nn.DataParallel(model)
    model = model.to(device)

    dataloaders, dataset_sizes = init_data(
        args.data_loc, 
        args.data_aug, 
        args.batch_size, 
        args.split,
        args.input_shape, 
        args.norm_mean,
        args.norm_std
    )

    print("Model training started...\n")
    
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

    print("Saving the best model and corresponding metrics...")
    
    os.makedirs(args.model_save_loc, exist_ok=True)
    os.makedirs(args.metrics_save_loc, exist_ok=True)
    
    model_name = f"{args.model_name}_freeze_{args.freeze_layers}_epochs_{args.epochs}_batch_{args.batch_size}_optim_{args.optimizer}_aug_{args.data_aug}_split_{int(args.split[0]*100)}_{int(args.split[1]*100)}"
    torch.save(best_model_state_dict, f"{args.model_save_loc}/{model_name}.pth")
    pd.DataFrame(metrics).to_csv(f"{args.metrics_save_loc}/{model_name}.csv", index=False)
    save_train_val_curves(f"{args.metrics_save_loc}/{model_name}.png", metrics)
    
    print("Model and metrics saved. Training process has finished.\n\n")


if __name__ == "__main__":
    main()
