import os
import argparse

from frame_extractors import MiddleFrameExtractor

parser = argparse.ArgumentParser(description="Extracting frames from dataset")

parser.add_argument("--data_loc", required=True, nargs="*", type=str, help="Path of folder(s) that contain the videos from which frames are to be extracted")
parser.add_argument("--save_loc", required=True, type=str, help="Path of the folder where the extracted frames are to be saved")
parser.add_argument("--n_frames", required=True, type=int, help="Number of frames to extract from each video")

args = parser.parse_args()

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

extractor = MiddleFrameExtractor(args.data_loc, args.save_loc, args.n_frames)

print("-"*20 + " Frame extraction started " + "-"*20)
extractor()
print("-"*19 + " Frame extraction completed " + "-"*19)