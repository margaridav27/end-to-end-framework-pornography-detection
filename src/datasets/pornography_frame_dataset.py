import pandas as pd
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset

class PornographyFrameDataset(Dataset):
  def __init__(self, data_loc : str, df : pd.DataFrame, transform : A.Compose=None):
    self.data_loc = data_loc
    self.frames = df["frame"].tolist()
    self.labels = df["label"].tolist()
    self.frame_label_dict = { frame: label for frame, label in zip(self.frames, self.labels) }
    self.transform = transform    

  def __len__(self):  
    return len(self.frames)
    
  def __getitem__(self, index):    
    if torch.is_tensor(index):
      index = index.tolist()

    frame_name = index if isinstance(index, str) else self.frames[index]
    frame_path = f"{self.data_loc}/{frame_name}"
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_label = self.frame_label_dict[frame_name] if isinstance(index, str) else self.labels[index]

    if self.transform:
      frame = self.transform(image=frame)["image"]

    return frame_name, frame, frame_label
