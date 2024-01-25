from data_utilities import load_split, get_transforms
from model_utilities import init_model, test_model
from pornography_frame_dataset import PornographyFrameDataset

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="Testing a trained pytorch model")
parser.add_argument('--data_loc', type=str, required=True)
parser.add_argument('--save_loc', type=str, required=True)
parser.add_argument('--state_dict_loc', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--input_shape', type=int, default=224)

args = parser.parse_args()

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO seeding

print("Assembling test partition...")

df_test = load_split(args.data_loc, ["test"])["test"]
n_classes = len(df_test["label"].unique())
data_transforms = get_transforms(args.input_shape)["test"]
dataset = PornographyFrameDataset(args.data_loc, df_test, data_transforms)
dataloader = DataLoader(dataset, args.batch_size)

print(f"Loading model...")

if not os.path.exists(args.state_dict_loc):
  raise ValueError("Invalid --state_dict_loc argument.")

state_dict = torch.load(args.state_dict_loc)
model = init_model(args.model_name, n_classes)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

print("Model testing started...\n")

results_save_loc = f"{args.save_loc}/{args.state_dict_loc.split('/')[-1].split('.')[0]}.csv"
test_model(model, dataloader, len(dataset), device, results_save_loc)
