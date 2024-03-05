import torch
from torch.utils.data import Dataset
from PIL import Image

class PornographyFrameDataset(Dataset):
  def __init__(self, data_loc, df, transform=None):
    self.data_loc = data_loc
    self.frames = df["frame"].tolist()
    self.labels = df["label"].tolist()
    self.transform = transform    

  def __len__(self):  
    return len(self.frames)
  
  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()

    frame_name = self.frames[index]
    frame_path = f"{self.data_loc}/{frame_name}"
    frame = Image.open(frame_path).convert("RGB")
    if self.transform:
      frame = self.transform(frame)

    return frame_name, frame, self.labels[index]
