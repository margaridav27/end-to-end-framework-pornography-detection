# Imports
import os
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset


# Class: PornographyDatabase
class PornographyDatabase(Dataset):
    # Method: __init__
    def __init__(self, data_dir, binary=True, transform=None):
        # Get directories
        self.data_dir = data_dir
        self.database_dir = os.path.join(self.data_dir, "Database")
        self.frames_dir = os.path.join(self.data_dir, "Frames")
        self.segments_dir = os.path.join(self.data_dir, "Segments")

        # Get labels
        # If binary problem, vNonPornEasy will not be distinguished from vNonPornDifficulty
        self.labels_to_int_dict = {
            "vNonPornEasy": 0,
            "vNonPornDifficulty": 0 if binary else 1,
            "vPorn": 1 if binary else 2,
        }

        video_names, video_labels = [], []
        seg_names, seg_labels = [], []
        frame_names, frame_labels = [], []
        
        for label, label_int in self.labels_to_int_dict.items():
            # Get video data
            for v in os.listdir(os.path.join(self.database_dir, label)):
                if v.startswith("."): continue
                video_names.append(v)
                video_labels.append(label_int)

            # Get segments data
            for s in os.listdir(os.path.join(self.segments_dir, label)):
                if s.startswith("."): continue
                seg_names.append(s)
                seg_labels.append(label_int)

            # Get frames data
            for f in os.listdir(os.path.join(self.frames_dir, label)):
                if f.startswith("."): continue
                frame_names.append(f)
                frame_labels.append(label_int)

        self.video_names = video_names
        self.video_labels = video_labels

        self.seg_names = seg_names
        self.seg_labels = seg_labels

        self.frame_names = frame_names
        self.frame_labels = frame_labels

        self.transform = transform

    # Method: __len__
    def __len__(self):
        return len(self.frame_names)

    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_path = os.path.join(self.frames_dir, self.frame_names[idx])
        frame_label = self.frame_labels[idx]

        frame = Image.open(frame_path).convert('RGB')

        if self.transform:
            frame = self.transform(frame)

        return frame, frame_label
