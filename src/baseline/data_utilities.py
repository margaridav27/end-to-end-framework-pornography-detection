from pornography_frame_dataset import PornographyFrameDataset

from typing import Dict
import pandas as pd 
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torchvision import transforms


def split_data(df_frames : pd.DataFrame, val_size : int, test_size : int) -> Dict[str, pd.DataFrame]:
  df_frames["video"] = [frame_name.split("#")[0] for frame_name in df_frames["frame"]]

  agg = { "video": "first", "label": "first" }

  df_videos = df_frames[["video", "label"]]
  df_videos = df_videos.groupby("video").aggregate(agg).reset_index(drop=True)

  df_frames = df_frames.drop("video", axis=1)

  real_val_size = (1 - test_size) * val_size

  train_videos, test_videos = train_test_split(df_videos, test_size=0.2, random_state=42)
  train_videos, val_videos = train_test_split(train_videos, test_size=real_val_size, random_state=42)

  train_frames = df_frames[df_frames['frame'].str.contains('|'.join(train_videos['video']))]
  val_frames = df_frames[df_frames['frame'].str.contains('|'.join(val_videos['video']))]
  test_frames = df_frames[df_frames['frame'].str.contains('|'.join(test_videos['video']))]

  return { "train": train_frames, "val": val_frames, "test": test_frames }


def load_split(data_loc : str, partitions : list=[]) -> Dict[str, pd.DataFrame]:
  df = pd.read_csv(f"{data_loc}/split.csv")
  if not partitions: partitions = list(df["partition"].unique())
  return { p: df[df["partition"] == p] for p in partitions }


def save_split(save_loc : str, partitions : list, dfs : Dict[str, pd.DataFrame]):
  for p in partitions: dfs[p]["partition"] = p
  split = pd.concat(dfs.values(), ignore_index=True)
  split.to_csv(f"{save_loc}/split.csv", index=False)


def get_transforms(input_shape : int) -> Dict[str, transforms.Compose]:
  scale = 256
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

  return {
    "train": transforms.Compose([
      transforms.Resize(scale),
      transforms.RandomResizedCrop(input_shape),
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


def init_data(data_loc : str, input_shape : int, batch_size : int):
  data_transforms = get_transforms(input_shape) 

  df_frames = pd.read_csv(f"{data_loc}/data.csv")

  partitions = ["train", "val", "test"]
  dfs = split_data(df_frames, 0.05, 0.2)
  save_split(data_loc, partitions, dfs)

  datasets = { p: PornographyFrameDataset(data_loc, dfs[p], data_transforms.get(p)) for p in partitions }
  dataloaders = { p: DataLoader(datasets[p], batch_size) for p in partitions }
  dataset_sizes = { p: len(datasets[p]) for p in partitions }
  n_classes = len(df_frames["label"].unique())

  return dataloaders, dataset_sizes, n_classes
