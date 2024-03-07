from src.utils.data import init_data
from src.utils.model import train_model
from src.utils.evaluation import save_train_val_curves

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data_loc = "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/middle-5"
input_shape = 224
batch_size = 32
split = [0.1, 0.2] # validation and test
dataloaders, dataset_sizes = init_data(data_loc, input_shape, batch_size, split)

model = BasicCNN()
model = model.to(device)

optimizer = "sgd"
epochs = 20
best_model, metrics = train_model(
    model,
    dataloaders,
    dataset_sizes,
    optimizer,
    epochs,
    device,
)

save_loc = "."
model_name = "basic_cnn"

torch.save(best_model.state_dict(), f"{save_loc}/{model_name}.pth")
pd.DataFrame(metrics).to_csv(f"{save_loc}/{model_name}.csv", index=False)
save_train_val_curves(f"{save_loc}/{model_name}.png", metrics)
