from data_utilities import init_data
from model_utilities import init_model, train_model
from eval_utilities import save_train_val_curves

import os
import argparse
import random
import pandas as pd
import numpy as np

import torch


parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--model_save_loc", type=str, required=True)
parser.add_argument("--metrics_save_loc", type=str, required=True)
parser.add_argument("--model_name", type=str, default="resnet50")
parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
parser.add_argument("--freeze_layers", action="store_true", default=False)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--optimized", action="store_true", default=False)
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--data_aug", action="store_true", default=False)
parser.add_argument("--split", type=float, nargs="*", default=[0.05, 0.2], help="Validation and test")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataloaders, dataset_sizes = init_data(
    args.data_loc, args.input_shape, args.batch_size, args.split
)

print(f"Loading {args.model_name}...")

model = init_model(args.model_name, args.weights, args.freeze_layers, args.optimized)
model = torch.nn.DataParallel(model)
model = model.to(device)

print("Model training started...\n")

best_model, metrics = train_model(
    model,
    dataloaders,
    dataset_sizes,
    args.optimizer,
    args.epochs,
    device,
)

print("Saving the best model...")

if not os.path.exists(args.model_save_loc):
    os.makedirs(args.model_save_loc)

model_name = f"{args.model_name}_freeze_{args.freeze_layers}_epochs_{args.epochs}_batch_{args.batch_size}_optim_{args.optimizer}_optimized_{args.optimized}"
model_save_loc = f"{args.model_save_loc}/model_{model_name}.pth"
torch.save(best_model.state_dict(), model_save_loc)

print("Model saved. Saving metrics...")

if not os.path.exists(args.metrics_save_loc):
    os.makedirs(args.metrics_save_loc)

pd.DataFrame(metrics).to_csv(f"{args.metrics_save_loc}/{model_name}.csv", index=False)
save_train_val_curves(f"{args.metrics_save_loc}/{model_name}.png", metrics)

print("Metrics saved. Training process has finished.\n\n")
