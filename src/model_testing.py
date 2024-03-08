from src.utils.data import load_split, get_transforms
from src.utils.model import init_model, test_model
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="Testing a trained pytorch model")
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--save_loc", type=str, required=True)
parser.add_argument("--state_dict_loc", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

_, model_filename = os.path.split(args.state_dict_loc) # Includes .pth
model_filename = model_filename.split(".")[0] # Does not include .pth
model_name = model_filename.split("_")[0]

split = model_filename.split("_")[-2:]
split = [float(i)/100 for i in split]

df_test = load_split(args.data_loc, split, ["test"])["test"]
data_transforms = get_transforms(False, args.input_shape, args.norm_mean, args.norm_std)["test"]
dataset = PornographyFrameDataset(args.data_loc, df_test, data_transforms)
dataloader = DataLoader(dataset, args.batch_size)

print(f"Loading {model_name}...")

if not os.path.exists(args.state_dict_loc):
    raise ValueError("Invalid --state_dict_loc argument.")

state_dict = torch.load(args.state_dict_loc)
model = init_model(model_name)
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)
model = model.to(device)

print("Model testing started...")

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

results_save_loc = f"{args.save_loc}/{model_filename}"

test_model(model, dataloader, device, results_save_loc)

print("Ground-truth and predictions. Testing process has finished.\n\n")
