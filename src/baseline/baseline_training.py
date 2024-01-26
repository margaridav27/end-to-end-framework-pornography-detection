from data_utilities import init_data
from model_utilities import init_model, train_model

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--save_loc", type=str, required=True)
parser.add_argument("--model_name", type=str, default="resnet50")
parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
parser.add_argument("--freeze_layers", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--input_shape", type=int, default=224)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Seeding to ensure reproducibility...")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)

print("Assembling data...")

dataloaders, dataset_sizes, n_classes = init_data(args.data_loc, args.input_shape, args.batch_size)

print(f"Loading the {args.model_name} model...")

model = init_model(args.model_name, n_classes, args.weights, args.freeze_layers)
model = torch.nn.DataParallel(model)
model = model.to(device)

print("Defining criterion, optimizer, and scheduler for model training...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)

print("Model training started...\n")

best_model = train_model(
  model,
  dataloaders,
  dataset_sizes,
  criterion,
  optimizer,
  scheduler,
  args.epochs,
  device
)

print("Saving the best model...")

if not os.path.exists(args.save_loc):
  os.makedirs(args.save_loc)

model_save_loc = (f"{args.save_loc}/{args.model_name}_freeze_layers_{args.freeze_layers}.pth")
torch.save(best_model.state_dict(), model_save_loc)

print("Model saved. Training process has finished.")
