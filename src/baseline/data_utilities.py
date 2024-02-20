from pornography_frame_dataset import PornographyFrameDataset

import os
from typing import Dict
import pandas as pd 
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms


def split_data(df_frames : pd.DataFrame, split : list) -> Dict[str, pd.DataFrame]:
  df_frames["video"] = [frame_name.split("#")[0] for frame_name in df_frames["frame"]]

  agg = { "video": "first", "label": "first" }

  df_videos = df_frames[["video", "label"]]
  df_videos = df_videos.groupby("video").aggregate(agg).reset_index(drop=True)

  df_frames = df_frames.drop("video", axis=1)

  val_size, test_size = split
  real_val_size = (1 - test_size) * val_size

  train_videos, test_videos = train_test_split(df_videos, test_size=0.2, random_state=42)
  train_videos, val_videos = train_test_split(train_videos, test_size=real_val_size, random_state=42)

  train_frames = df_frames[df_frames["frame"].str.contains("|".join(train_videos["video"]))]
  val_frames = df_frames[df_frames["frame"].str.contains("|".join(val_videos["video"]))]
  test_frames = df_frames[df_frames["frame"].str.contains("|".join(test_videos["video"]))]
  
  split = { "train": train_frames, "val": val_frames, "test": test_frames }
  print(f"Created split.")
  log_split(split)

  return split


def load_split(data_loc : str, partitions : list=[]) -> Dict[str, pd.DataFrame]:
  df = pd.read_csv(f"{data_loc}/split.csv")
  if not partitions: partitions = list(df["partition"].unique())

  split = { p: df[df["partition"] == p] for p in partitions }
  print("Loaded split.")
  log_split(split)

  return split


def save_split(save_loc : str, partitions : list, dfs : Dict[str, pd.DataFrame]):
  for p in partitions: dfs[p]["partition"] = p
  split = pd.concat(dfs.values(), ignore_index=True)
  split.to_csv(f"{save_loc}/split.csv", index=False)


def check_split(data_loc : str) -> bool:
  return os.path.isfile(f"{data_loc}/split.csv")


def log_split(split : Dict[str, pd.DataFrame]):
  for partition, df in split.items():
    print(f"{partition}: total ({len(df)}); porn ({len(df[df['label'] == 1])}); non-porn ({len(df[df['label'] == 0])})")
  

def get_transforms(input_shape : int) -> Dict[str, transforms.Compose]:
  scale = 256
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

  # TODO: add support for data aug
  return {
    "train": transforms.Compose([
      transforms.Resize(scale),
      transforms.CenterCrop(input_shape),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
    ]),
    "val": transforms.Compose([
      transforms.Resize(scale),
      transforms.CenterCrop(input_shape),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
    ]),
    "test": transforms.Compose([
      transforms.Resize(scale),
      transforms.CenterCrop(input_shape),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
    ])
  } 


def init_data(data_loc : str, input_shape : int, batch_size : int, split : list):
  data_transforms = get_transforms(input_shape) 

  df_frames = pd.read_csv(f"{data_loc}/data.csv")

  partitions = ["train", "val", "test"]
  if not check_split(data_loc):
    dfs = split_data(df_frames, split)
    save_split(data_loc, partitions, dfs)
  else:
    dfs = load_split(data_loc)

  datasets = { p: PornographyFrameDataset(data_loc, dfs[p], data_transforms.get(p)) for p in partitions }
  dataloaders = { p: DataLoader(datasets[p], batch_size) for p in partitions }
  dataset_sizes = { p: len(datasets[p]) for p in partitions }

  return dataloaders, dataset_sizes
