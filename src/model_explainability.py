from src.utils.data import load_split, get_transforms
from src.utils.model import init_model, train_model
from src.utils.evaluation import save_train_val_curves
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import random
import pandas as pd
import numpy as np

import torch


parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
parser.add_argument("--state_dict_loc", type=str, required=True)
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--save_loc", type=str, required=True)
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
args = parser.parse_args()

_, model_filename = os.path.split(args.state_dict_loc) # Includes .pth
model_filename = model_filename.split(".")[0] # Does not include .pth
model_name = model_filename.split("_")[0]

split = model_filename.split("_")[-2:]
split = [float(i)/100 for i in split]

print(f"Loading dataset...")
if not os.path.exists(args.data_loc):
    raise ValueError("Invalid --data_loc argument.")

df_test = load_split(args.data_loc, split, ["test"])["test"]
data_transforms = get_transforms(False, args.input_shape, args.norm_mean, args.norm_std)["test"]
dataset = PornographyFrameDataset(args.data_loc, df_test, data_transforms)

print(f"Loading {model_name}...")
if not os.path.exists(args.state_dict_loc):
    raise ValueError("Invalid --state_dict_loc argument.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dict = torch.load(args.state_dict_loc)
model = init_model(model_name)
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
