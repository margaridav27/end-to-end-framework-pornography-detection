import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import torch
from torch.utils.data import Dataset


class PornographyFrameDataset(Dataset):
    def __init__(
        self,
        data_loc: str,
        df: pd.DataFrame,
        transform: A.Compose = None,
        subset: float = 1.0,
    ):
        self.data_loc = data_loc

        assert (subset > 0.0 and subset <= 1.0), "Invalid value for parameter subset. Must be a value between 0 and 1."

        if subset < 1.0:
            df = self.sample_subset(df, subset)

        self.frames = df["frame"].tolist()
        self.labels = df["label"].tolist()
        self.frame_label_dict = {frame: label for frame, label in zip(self.frames, self.labels)}
        self.transform = transform

    def sample_subset(self, df: pd.DataFrame, subset_size: float):
        subset, _ = train_test_split(df, train_size=subset_size, random_state=42)
        print(f"Created subset from dataset with size {len(subset)}.")
        return subset

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        frame_name = index if isinstance(index, str) else self.frames[index]
        frame_path = f"{self.data_loc}/{frame_name}"
        frame_label = (
            self.frame_label_dict[frame_name]
            if isinstance(index, str)
            else self.labels[index]
        )

        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        orig_shape = torch.tensor(frame.shape)

        if self.transform:
            frame = self.transform(image=frame)["image"]

        return frame_name, frame, frame_label, orig_shape
