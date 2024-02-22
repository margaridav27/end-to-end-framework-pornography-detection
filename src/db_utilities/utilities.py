import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
  def __init__(self, subset, transform=None):
    self.subset = subset
    self.transform = transform

  def __len__(self):
    return len(self.subset)

  def __getitem__(self, index):
    x, y = self.subset[index]
    if self.transform: x = self.transform(x)
    return x, y


class FramesExtractor():
  def __init__(self):
    pass

  def _open_video(self, video_path : str) -> cv.VideoCapture:
    video = cv.VideoCapture(video_path)
    assert video.isOpened(), "Failed to capture video. NoneType was detected."   
    return video
  
  def _close_video(self, video : cv.VideoCapture):
    video.release()

  def _get_frame_count(self, v : str or cv.VideoCapture) -> int:
    video = self._open_video(v) if type(v) == str else v
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    if type(v) == str: self._close_video(video)
    return frame_count


class MiddleFramesExtractor(FramesExtractor):
  def __init__(self, n_frames : int):
    super().__init__()

    self.default_n_frames = n_frames
  
  def get_n_frames_to_extract(self, v : str or cv.VideoCapture) -> int:    
    frame_count = self._get_frame_count(v)
    return frame_count if frame_count < self.default_n_frames else self.default_n_frames

  def extract_frame(self, video_path : str, rel_frame_i : int) -> Image.Image:
    video = self._open_video(video_path)

    frame_count = self._get_frame_count(video)

    # Set the video to the position of the frame to be extracted
    n_frames = self.get_n_frames_to_extract(video)
    abs_frame_i = frame_count // 2 - n_frames // 2 + rel_frame_i
    video.set(cv.CAP_PROP_POS_FRAMES, abs_frame_i)
    
    _, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    self._close_video(video)

    return frame
