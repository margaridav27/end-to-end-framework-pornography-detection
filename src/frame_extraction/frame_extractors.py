import os
import cv2 as cv
import pandas as pd

class FrameExtractor():
  def __init__(self):
    pass

  def _open_video(self, video_path : str) -> cv.VideoCapture:
    video = cv.VideoCapture(video_path)
    assert video.isOpened(), "Failed to capture video. NoneType was detected."   
    return video
  
  def _close_video(self, video : cv.VideoCapture):
    video.release()

  def _get_frame_count(self, video : cv.VideoCapture) -> int:
    return int(video.get(cv.CAP_PROP_FRAME_COUNT))
 

class MiddleFrameExtractor(FrameExtractor):
  def __init__(self, data_loc : list, save_loc : str, n_frames : int):
    super().__init__()

    for loc in data_loc:
      assert os.path.exists(loc) and os.path.isdir(loc), "Invalid --data_loc argument."
    assert os.path.exists(save_loc) and os.path.isdir(save_loc), "Invalid --save_loc argument."

    self.data_loc = data_loc
    self.save_loc = save_loc
    self.n_frames = n_frames

  def __call__(self):
    data = { "frame": [], "label": [] }

    for loc in self.data_loc:
      for v in os.listdir(loc):
        if v.startswith("."): continue

        video_path = os.path.join(loc, v)
        video = self._open_video(video_path)
        
        frame_count = self._get_frame_count(video)
        real_n_frames = self.n_frames
        if self.n_frames > frame_count:
          real_n_frames = frame_count
          print(f"Number of frames {self.n_frames} exceeds {frame_count}. "
                f"Extracting {frame_count} frames instead")

        # Set the video to the position of the frame to be extracted
        start_frame_i = frame_count // 2 - real_n_frames // 2
        video.set(cv.CAP_PROP_POS_FRAMES, start_frame_i)

        for i in range(real_n_frames):
          ret, frame = video.read()

          if not ret:
            print(f"Failed to extract frame from {v}")
            continue

          frame_name = f"{v.split('.')[0]}#{i}.jpg"
          frame_label = 0 if "NonPorn" in v else 1

          data["frame"].append(frame_name)
          data["label"].append(frame_label)

          cv.imwrite(f"{self.save_loc}/{frame_name}", frame)

        self._close_video(video)
    
    pd.DataFrame(data).to_csv(f"{self.save_loc}/data.csv", index=False)
