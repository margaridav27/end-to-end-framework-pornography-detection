# Imports
import os
import pandas as pd

# PyTorch Imports
import torch
from torch.utils.data import Dataset


# Class: Pornography2kDatabase
class Pornography2kDatabase(Dataset):
    # Method: __init__
    def __init__(self, data_dir, frame_extractor, transform=None):
        self.frame_extractor = frame_extractor

        # Get directories
        self.data_dir = data_dir
        self.folds_dir = os.path.join(data_dir, "folds")
        self.original_dir = os.path.join(data_dir, "original")

        # Get video filenames and respective labels from that folder
        video_names, video_labels = [], []
        frame_names, frame_labels = [], []
        for v in os.listdir(self.original_dir):
            if v.startswith("."): continue

            label = 0 if "NonPorn" in v else 1

            video_names.append(v)
            video_labels.append(label)
            
            # While working only with frames
            video_path = os.path.join(self.original_dir, v)
            n_frames = self.frame_extractor.get_n_frames_to_extract(video_path)
            frame_names.extend([f"{v}#{i}" for i in range(n_frames)])
            frame_labels.extend([label for _ in range(n_frames)])

        # Read folds
        folds_dict = dict()
        for f in [1, 2, 3, 4, 5]:
            folds_dict[f"s{f}"] = dict()
            train_videos, train_labels = [], []
            test_videos, test_labels = [], []

            # Open train data
            train_positive = pd.read_csv(os.path.join(self.folds_dir, f"s{f}_positive_training.txt")).values
            train_negative = pd.read_csv(os.path.join(self.folds_dir, f"s{f}_negative_training.txt")).values
            for v in train_positive:
                train_videos.append(v)
                train_labels.append(1)
            for v in train_negative:
                train_videos.append(v)
                train_labels.append(0)

            # Open test data
            test_positive = pd.read_csv(os.path.join(self.folds_dir, f"s{f}_positive_test.txt")).values
            test_negative = pd.read_csv(os.path.join(self.folds_dir, f"s{f}_negative_test.txt")).values
            for v in test_positive:
                test_videos.append(v)
                test_labels.append(1)
            for v in test_negative:
                test_videos.append(v)
                test_labels.append(0)

            # Complete fold dictionary
            folds_dict[f"s{f}"]["train_videos"] = train_videos
            folds_dict[f"s{f}"]["train_labels"] = train_labels
            folds_dict[f"s{f}"]["test_videos"] = test_videos
            folds_dict[f"s{f}"]["test_labels"] = test_labels

        self.video_names = video_names
        self.video_labels = video_labels

        self.frame_names = frame_names
        self.frame_labels = frame_labels

        self.folds_dict = folds_dict
        
        self.transform = transform

    # Method: __len__
    def __len__(self):
        return len(self.frame_names)

    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_name = self.frame_names[idx]
        video_fname, frame_i = frame_name.split("#")
        frame = self.frame_extractor.extract_frame(os.path.join(self.original_dir, video_fname), int(frame_i))
        frame_label = self.frame_labels[idx]

        if self.transform:
            frame = self.transform(frame)

        return frame, frame_label
