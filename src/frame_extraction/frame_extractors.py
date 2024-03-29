import os
import math
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


class EvenFrameExtractor(FrameExtractor):
  def __init__(self, data_loc : list, save_loc : str, n_frames : int, perc : float):
    super().__init__()

    for loc in data_loc:
      assert os.path.exists(loc) and os.path.isdir(loc), "Invalid --data_loc argument."
    assert os.path.exists(save_loc) and os.path.isdir(save_loc), "Invalid --save_loc argument."

    self.data_loc = data_loc
    self.save_loc = save_loc
    self.n_frames = n_frames
    self.perc = perc

  def __call__(self):
    data = { "frame": [], "label": [] }

    for loc in self.data_loc:
      for v in os.listdir(loc):
        if v.startswith("."): continue

        video_path = os.path.join(loc, v)

        max_attempts = 15
        extracted_frames = {}

        for _ in range(max_attempts):
          video = self._open_video(video_path)
          
          frame_count = self._get_frame_count(video)
          real_n_frames = self.n_frames
          ignore_frames = math.floor(frame_count * self.perc) # Number of frames to be ignored at the beginning and at the end

          if self.n_frames > frame_count:
            real_n_frames = frame_count
            ignore_frames = 0 # In this case, simply extract all the frames
            print(f"Number of frames {self.n_frames} exceeds {frame_count}. "
                  f"Extracting {frame_count} frames instead")
            
          while frame_count - 2 * ignore_frames < real_n_frames and ignore_frames > 0:
            ignore_frames //= 2

          interval = max((frame_count - 2 * ignore_frames) // real_n_frames, 1) # Interval at which frames are to be extracted
      
          frame_i = 0

          while True:
            # If we've extracted all the frames we need, or we're at the end of the video
            if len(extracted_frames) == real_n_frames or frame_i == frame_count - 1:
              break

            ret, frame = video.read()

            # Advance to next frame
            if not ret or frame_i < ignore_frames: 
              frame_i += 1
              continue

            # Even if we reach the portion of the video that was supposed to be ignored,
            # we still continue to extract frames if we haven't extracted them all yet
            # The loop breaks at the first condition if we've already extracted all frames
    
            if frame_i % interval == 0:
              frame_name = f"{v.split('.')[0]}#{len(extracted_frames)}.jpg"
              frame_label = 0 if "NonPorn" in v else 1

              extracted_frames[frame_name] = frame_label
              
              cv.imwrite(f"{self.save_loc}/{frame_name}", frame)
            
            frame_i += 1

          self._close_video(video)

          if len(extracted_frames) == real_n_frames: break

        data["frame"].extend(extracted_frames.keys())
        data["label"].extend(extracted_frames.values())

    pd.DataFrame(data).to_csv(f"{self.save_loc}/data.csv", index=False)
