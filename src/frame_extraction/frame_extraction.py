import os
import argparse

from src.frame_extraction.frame_extractors import FrameExtractor, MiddleFrameExtractor, EvenFrameExtractor

parser = argparse.ArgumentParser(description="Extracting frames from dataset")

parser.add_argument("--data_loc", required=True, nargs="*", type=str, help="Path of folder(s) that contain the videos from which frames are to be extracted")
parser.add_argument("--save_loc", required=True, type=str, help="Path of the folder where the extracted frames are to be saved")
parser.add_argument("--n_frames", required=True, type=int, help="Number of frames to extract from each video")
parser.add_argument("--strat", type=str, default="middle", help="Strategy to use in the extraction (either middle or even)")
parser.add_argument("--perc", type=float, default=0.2, help="Percentage of video to ignore at the beginning and at the end (only if strat=even)")

args = parser.parse_args()

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

extractor = FrameExtractor()
if args.strat == "middle":
    extractor = MiddleFrameExtractor(args.data_loc, args.save_loc, args.n_frames)
elif args.strat == "even":
    extractor = EvenFrameExtractor(args.data_loc, args.save_loc, args.n_frames, args.perc)
else:
    raise ValueError("Invalid --strat argument")

extractor()
